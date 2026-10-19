//! Shared live frame feed for desktop environments.
//!
//! Each environment has at most one producer thread, however many viewers show
//! it. The thread waits on the compositor's publish futex, copies each new frame
//! out of shared memory, shrinks it for display, and publishes it once to every
//! subscriber. Subscribers only see the newest value, so a slow foreground skips
//! frames instead of queueing them. The thread exits when nobody is subscribed
//! or the environment has ended.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError, Weak};
use std::thread;
use std::time::Duration;

use gpui::RenderImage;
use image::{Frame, ImageBuffer, Rgba};
use juto_desktop::protocol::DesktopEvent;
use juto_desktop::shm::FrameWait;
use juto_desktop::viewer::{EventStream, ViewerClient};
use tokio::sync::watch;

/// Longest shown edges are 854x480 (480p, 16:9). Larger frames are shrunk
/// before upload; the environment keeps its resolution and input is still
/// mapped to environment pixels.
const SHOWN_MAX_WIDTH: u32 = 854;
const SHOWN_MAX_HEIGHT: u32 = 480;
/// Longest a producer blocks before rechecking liveness and subscribers.
const WAIT_SLICE: Duration = Duration::from_millis(100);
/// Pause after a failed wait before the producer tries again.
const RETRY_DELAY: Duration = Duration::from_millis(100);

/// A composed frame ready for display.
pub(crate) struct StreamFrame {
    /// Environment resolution; input coordinates are mapped to this.
    pub(crate) source: (u32, u32),
    /// Upload handle shared by every viewer showing this frame.
    pub(crate) image: Arc<RenderImage>,
    /// Compositor generation the frame was composed under; input carries it back.
    pub(crate) generation: u64,
}

/// The latest-value slot a producer publishes into.
#[derive(Clone)]
enum Feed {
    /// No frame has been composed yet.
    Waiting,
    Frame(Arc<StreamFrame>),
    /// The environment was destroyed; no further frames will arrive.
    Ended,
}

/// One environment's producer state, shared by its subscribers.
struct Stream {
    feed: watch::Sender<Feed>,
    /// True while a producer thread runs. Guards spawning and retirement.
    producing: Mutex<bool>,
}

/// Identity of one live stream: environment socket path and id.
type StreamKey = (PathBuf, String);

/// Live streams keyed by environment socket path and id. Weak, so a stream with
/// no producer and no subscribers is freed.
static REGISTRY: LazyLock<Mutex<HashMap<StreamKey, Weak<Stream>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A viewer's handle on one environment's frames.
pub(crate) struct FrameSubscription {
    feed: watch::Receiver<Feed>,
    /// The first call delivers the value already present; later calls wait for a newer one.
    first: bool,
}

impl FrameSubscription {
    /// Waits for the next composed frame. Returns `None` once the environment has ended.
    pub(crate) async fn next(&mut self) -> Option<Arc<StreamFrame>> {
        loop {
            if !self.first && self.feed.changed().await.is_err() {
                return None;
            }
            self.first = false;
            // Reading marks the value seen, so the next `changed` waits for a newer one.
            let current = self.feed.borrow_and_update().clone();
            match current {
                Feed::Frame(frame) => return Some(frame),
                Feed::Ended => return None,
                Feed::Waiting => {}
            }
        }
    }
}

/// Subscribes to the frames of environment `env_id` served at `socket`,
/// starting its producer thread if none is running.
pub(crate) fn subscribe(socket: &Path, env_id: &str) -> FrameSubscription {
    let stream = stream_for(socket, env_id);
    // Subscribe before taking the lock: a producer deciding whether to retire
    // checks for receivers under the same lock, so it either sees this one or
    // is restarted below.
    let feed = stream.feed.subscribe();
    let mut producing = lock(&stream.producing);
    if !*producing {
        *producing = true;
        let producer_stream = Arc::clone(&stream);
        let producer_socket = socket.to_path_buf();
        let producer_env = env_id.to_string();
        thread::Builder::new()
            .name("juto-desktop-frames".to_string())
            .spawn(move || run_producer(producer_socket, producer_env, producer_stream))
            .expect("failed to spawn desktop frame producer");
    }
    FrameSubscription { feed, first: true }
}

/// Returns the stream for environment `env_id` at `socket`, creating it when
/// none is live.
fn stream_for(socket: &Path, env_id: &str) -> Arc<Stream> {
    let key = (socket.to_path_buf(), env_id.to_string());
    let mut streams = lock(&REGISTRY);
    streams.retain(|_, stream| stream.strong_count() > 0);
    if let Some(stream) = streams.get(&key).and_then(Weak::upgrade) {
        return stream;
    }
    let (feed, _) = watch::channel(Feed::Waiting);
    let stream = Arc::new(Stream {
        feed,
        producing: Mutex::new(false),
    });
    streams.insert(key, Arc::downgrade(&stream));
    stream
}

/// Runs on the producer thread: publishes each newly composed frame until
/// nobody is subscribed or the environment ends. The viewer connection keeps
/// the environment awake while this runs; a lifecycle watcher ends the stream
/// when the environment stops or fails.
fn run_producer(socket: PathBuf, env_id: String, stream: Arc<Stream>) {
    let (viewer, lifecycle) = match (
        ViewerClient::attach(&socket, &env_id),
        EventStream::subscribe(&socket, &env_id),
    ) {
        (Ok(viewer), Ok(lifecycle)) => (viewer, Arc::new(lifecycle)),
        // The environment is gone or unreachable: no frames will arrive.
        _ => {
            stream.feed.send_replace(Feed::Ended);
            *lock(&stream.producing) = false;
            return;
        }
    };
    let stopping = Arc::new(AtomicBool::new(false));
    let watcher_stream = Arc::clone(&stream);
    let watcher_events = Arc::clone(&lifecycle);
    let watcher_stopping = Arc::clone(&stopping);
    thread::Builder::new()
        .name("juto-desktop-lifecycle".to_string())
        .spawn(move || watch_lifecycle(&watcher_events, &watcher_stream, &watcher_stopping))
        .expect("failed to spawn desktop lifecycle watcher");

    // Starting from 0 reports a frame composed before this thread started, so a
    // subscriber joining an idle environment still sees its current frame.
    let mut seen_publish = 0;
    let mut last_frame_id = None;
    loop {
        if matches!(*stream.feed.borrow(), Feed::Ended) {
            stopping.store(true, Ordering::Release);
            let _ = lifecycle.close();
            *lock(&stream.producing) = false;
            return;
        }
        if retire_unobserved(&stream) {
            stopping.store(true, Ordering::Release);
            let _ = lifecycle.close();
            return;
        }
        match viewer.wait_for_frame(seen_publish, WAIT_SLICE) {
            Ok(FrameWait::Published(current)) => {
                seen_publish = current;
                // A subscriber may have left during the wait. The loop head
                // retires the producer, so skip the copy and let it do so.
                if stream.feed.receiver_count() == 0 {
                    continue;
                }
                if let Some(decoded) = read_newest(&viewer, last_frame_id) {
                    last_frame_id = Some(decoded.frame_id);
                    let image =
                        render_image_from_bgra(decoded.pixels, decoded.width, decoded.height);
                    let frame = StreamFrame {
                        source: decoded.source,
                        generation: decoded.generation,
                        image,
                    };
                    stream.feed.send_replace(Feed::Frame(Arc::new(frame)));
                }
            }
            Ok(FrameWait::TimedOut) => {}
            Err(_) => thread::sleep(RETRY_DELAY),
        }
    }
}

/// Ends the stream when the environment reports that it stopped or failed. A
/// lost lifecycle connection also ends it, unless this producer closed it.
fn watch_lifecycle(events: &EventStream, stream: &Stream, stopping: &AtomicBool) {
    loop {
        match events.next_event() {
            Ok(DesktopEvent::Stopped { .. } | DesktopEvent::Failed { .. }) => {
                stream.feed.send_replace(Feed::Ended);
                return;
            }
            Ok(_) => {}
            Err(_) => {
                if !stopping.load(Ordering::Acquire) {
                    stream.feed.send_replace(Feed::Ended);
                }
                return;
            }
        }
    }
}

/// Clears the producer flag when no receiver remains. Returns `false` when a
/// subscriber is still attached, so the producer keeps running.
fn retire_unobserved(stream: &Stream) -> bool {
    let mut producing = lock(&stream.producing);
    if stream.feed.receiver_count() > 0 {
        return false;
    }
    *producing = false;
    true
}

/// A frame copied out of shared memory and shrunk for display.
struct DecodedFrame {
    frame_id: u64,
    source: (u32, u32),
    generation: u64,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Copies the newest composed frame out of the viewer's shared memory and
/// shrinks it for display. Returns `None` when there is no frame, when it is
/// `after` (already published), or when the slot cannot be read.
fn read_newest(viewer: &ViewerClient, after: Option<u64>) -> Option<DecodedFrame> {
    let (header, pixels) = viewer.latest_frame().ok()??;
    if after == Some(header.frame_id) {
        return None;
    }
    let (width, height, pixels) = fit_shown(&pixels, header.width, header.height)?;
    Some(DecodedFrame {
        frame_id: header.frame_id,
        source: (header.width, header.height),
        generation: header.generation,
        width,
        height,
        pixels,
    })
}

/// Locks `mutex`, recovering the guard from a poisoned lock. The guarded state
/// is a flag or a map that stays consistent across a panic.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Fits packed BGRA8 rows (stride `width * 4`) inside the shown bounds by
/// averaging each destination pixel's source footprint (a box filter), so fine
/// detail shrinks smoothly instead of aliasing. Frames already small enough are
/// copied as-is.
///
/// The filter is separable: each destination row first sums its source rows
/// into one line of column sums, then each destination pixel sums its span of
/// that line. The sums are exact integers, so the output equals the direct
/// footprint average. The hot loops index slices directly rather than through
/// `Vec` indexing or iterator chains, because unoptimised builds pay a call for
/// each of those.
fn fit_shown(bgra: &[u8], width: u32, height: u32) -> Option<(u32, u32, Vec<u8>)> {
    if width == 0 || height == 0 {
        return None;
    }
    let src_row = width as usize * 4;
    let src = bgra.get(..src_row * height as usize)?;
    if width <= SHOWN_MAX_WIDTH && height <= SHOWN_MAX_HEIGHT {
        return Some((width, height, src.to_vec()));
    }
    let (out_w, out_h) = shown_size(width, height);
    let dst_row = out_w as usize * 4;
    let mut column_spans = Vec::with_capacity(out_w as usize);
    for dx in 0..out_w as u64 {
        column_spans.push(source_span(dx, width as u64, out_w as u64));
    }
    let mut out = vec![0u8; dst_row * out_h as usize];
    let mut column_sums = vec![0u32; src_row];
    let sums = column_sums.as_mut_slice();
    let spans = column_spans.as_slice();
    for dy in 0..out_h as usize {
        let (y0, y1) = source_span(dy as u64, height as u64, out_h as u64);
        sums.fill(0);
        for y in y0..y1 {
            let row = &src[y * src_row..(y + 1) * src_row];
            let mut i = 0;
            while i < src_row {
                sums[i] += row[i] as u32;
                i += 1;
            }
        }
        let rows = (y1 - y0) as u32;
        let line = &mut out[dy * dst_row..(dy + 1) * dst_row];
        let mut dx = 0;
        while dx < spans.len() {
            let (x0, x1) = spans[dx];
            let mut total = [0u32; 4];
            let mut at = x0 * 4;
            while at < x1 * 4 {
                total[0] += sums[at];
                total[1] += sums[at + 1];
                total[2] += sums[at + 2];
                total[3] += sums[at + 3];
                at += 4;
            }
            let count = rows * (x1 - x0) as u32;
            let half = count / 2;
            let px = dx * 4;
            line[px] = ((total[0] + half) / count) as u8;
            line[px + 1] = ((total[1] + half) / count) as u8;
            line[px + 2] = ((total[2] + half) / count) as u8;
            line[px + 3] = ((total[3] + half) / count) as u8;
            dx += 1;
        }
    }
    Some((out_w, out_h, out))
}

/// Size of a frame that exceeds the shown bounds, scaled down to fit them while
/// keeping its aspect ratio. Each side is at least one pixel.
fn shown_size(width: u32, height: u32) -> (u32, u32) {
    let scale =
        (SHOWN_MAX_WIDTH as f32 / width as f32).min(SHOWN_MAX_HEIGHT as f32 / height as f32);
    let out_w = ((width as f32 * scale).round() as u32).max(1);
    let out_h = ((height as f32 * scale).round() as u32).max(1);
    (out_w, out_h)
}

/// Source pixel span `[start, end)` covered by destination pixel `d` when `src`
/// pixels are mapped onto `dst`. Always covers at least one pixel.
fn source_span(d: u64, src: u64, dst: u64) -> (usize, usize) {
    let start = d * src / dst;
    let end = ((d + 1) * src / dst).max(start + 1);
    (start as usize, end as usize)
}

/// Builds a gpui `RenderImage` from packed BGRA8 pixels with straight alpha.
///
/// gpui's texture atlas is `Bgra8Unorm`, so the bytes are handed over unchanged.
pub(crate) fn render_image_from_bgra(bgra: Vec<u8>, width: u32, height: u32) -> Arc<RenderImage> {
    let buffer: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, bgra).expect("pixel buffer size mismatch");
    Arc::new(RenderImage::new(vec![Frame::new(buffer)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Alternating black and white pixels, one pixel per cell.
    fn checkerboard_bgra(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                pixels.extend_from_slice(&[value, value, value, 255]);
            }
        }
        pixels
    }

    fn stream_frame(width: u32) -> Feed {
        let image = render_image_from_bgra(vec![0, 0, 0, 255], 1, 1);
        Feed::Frame(Arc::new(StreamFrame {
            source: (width, 1),
            generation: 0,
            image,
        }))
    }

    /// Each published frame reaches a subscriber once, in order, and the end
    /// is reported once. An unchanged feed must not repeat the last frame.
    #[tokio::test]
    async fn subscriber_receives_each_frame_once_then_the_end() {
        let (feed, rx) = watch::channel(Feed::Waiting);
        let mut sub = FrameSubscription {
            feed: rx,
            first: true,
        };
        feed.send_replace(stream_frame(1));
        assert_eq!(sub.next().await.map(|f| f.source.0), Some(1));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), sub.next())
                .await
                .is_err(),
            "an unchanged feed must not repeat a frame"
        );
        feed.send_replace(stream_frame(2));
        assert_eq!(sub.next().await.map(|f| f.source.0), Some(2));
        feed.send_replace(Feed::Ended);
        assert!(sub.next().await.is_none());
    }

    #[test]
    fn downscale_averages_fine_detail_instead_of_sampling_it() {
        // 1708x960 maps onto 854x480 at exactly 1/2, so each destination pixel
        // covers a 2x2 block holding two black and two white pixels.
        let pixels = checkerboard_bgra(1708, 960);
        let (width, height, out) = fit_shown(&pixels, 1708, 960).unwrap();
        assert_eq!((width, height), (854, 480));
        assert!(out.chunks_exact(4).all(|px| px == [128, 128, 128, 255]));
    }

    #[test]
    fn frames_within_bounds_pass_through_unchanged() {
        let pixels = checkerboard_bgra(640, 360);
        let (width, height, out) = fit_shown(&pixels, 640, 360).unwrap();
        assert_eq!((width, height), (640, 360));
        assert_eq!(out, pixels);
    }

    /// The footprint average computed one destination pixel at a time. The
    /// separable filter must match it exactly.
    fn fit_shown_direct(bgra: &[u8], width: u32, height: u32) -> (u32, u32, Vec<u8>) {
        let (out_w, out_h) = shown_size(width, height);
        let src_row = width as usize * 4;
        let mut out = Vec::with_capacity(out_w as usize * out_h as usize * 4);
        for dy in 0..out_h as u64 {
            let (y0, y1) = source_span(dy, height as u64, out_h as u64);
            for dx in 0..out_w as u64 {
                let (x0, x1) = source_span(dx, width as u64, out_w as u64);
                let mut sum = [0u32; 4];
                for row in bgra[y0 * src_row..y1 * src_row].chunks_exact(src_row) {
                    for px in row[x0 * 4..x1 * 4].chunks_exact(4) {
                        for (acc, &channel) in sum.iter_mut().zip(px) {
                            *acc += u32::from(channel);
                        }
                    }
                }
                let count = ((y1 - y0) * (x1 - x0)) as u32;
                out.extend(sum.iter().map(|&total| ((total + count / 2) / count) as u8));
            }
        }
        (out_w, out_h, out)
    }

    /// Deterministic noise, so every channel of every footprint differs.
    fn noise_bgra(width: u32, height: u32, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..width as usize * height as usize * 4)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            })
            .collect()
    }

    /// The separable filter matches the direct footprint average exactly, for
    /// frames that shrink on one or both axes, including odd sizes and frames
    /// one pixel wide or tall.
    #[test]
    fn separable_downscale_matches_direct_footprint_average() {
        let sizes = [
            (1280, 720),
            (1920, 1080),
            (1000, 3000),
            (4000, 600),
            (855, 481),
            (3000, 1001),
            (1708, 960),
            (1000, 1),
            (1, 1000),
        ];
        for (width, height) in sizes {
            let seed = u64::from(width) * 7919 + u64::from(height);
            let pixels = noise_bgra(width, height, seed);
            assert_eq!(
                fit_shown(&pixels, width, height),
                Some(fit_shown_direct(&pixels, width, height)),
                "{width}x{height}"
            );
        }
    }

    /// Destination pixel spans tile the source exactly: contiguous, in order,
    /// each non-empty, from 0 to the source length.
    #[test]
    fn source_spans_tile_every_source_pixel_once() {
        let cases = [
            (1280u64, 854u64),
            (1920, 854),
            (720, 480),
            (1000, 1),
            (1, 1),
            (7, 3),
            (3000, 1001),
            (855, 854),
        ];
        for (src, dst) in cases {
            let mut next = 0;
            for d in 0..dst {
                let (start, end) = source_span(d, src, dst);
                assert_eq!(start, next, "{src}->{dst} at {d}");
                assert!(end > start, "{src}->{dst} at {d}");
                next = end;
            }
            assert_eq!(next, src as usize, "{src}->{dst}");
        }
    }
}
