//! Remote presentation: WebRTC video/input for a desktop environment.
//!
//! One [`PresentationWorker`] owns a GStreamer `webrtcbin` pipeline that encodes
//! composed desktop frames and carries pointer/keyboard input back over a
//! reliable ordered data channel. Signaling (offer/answer and trickled ICE) is
//! exchanged by the caller over an authenticated transport; this module never
//! opens an unauthenticated port.
//!
//! See `docs/agent-compositor.md` section 7.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use glib::prelude::*;
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_sdp as gst_sdp;
use gstreamer_video as gst_video;
use gstreamer_webrtc as gst_webrtc;
use serde::{Deserialize, Serialize};

use crate::error::DesktopError;
use crate::protocol::{InputBatch, InputEvent};
use crate::supervisor::DesktopEnvironment;

/// Video codec negotiated for the remote stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VideoCodec {
    Vp8,
    H264,
}

/// Which side creates the SDP offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Negotiation {
    /// The sender offers; the viewer answers (native receivers).
    SenderOffers,
    /// The viewer offers; the sender answers. Required for browser viewers,
    /// which must create the input data channel themselves.
    ViewerOffers,
}

/// A STUN/TURN server used for ICE.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IceServer {
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub credential: Option<String>,
}

/// Configuration for one authenticated remote streaming session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteStreamConfig {
    pub env_id: String,
    pub viewer_id: String,
    pub video_codec: VideoCodec,
    pub max_bitrate_bps: u64,
    pub max_fps: u32,
    #[serde(default = "default_negotiation")]
    pub negotiation: Negotiation,
    #[serde(default)]
    pub ice_servers: Vec<IceServer>,
}

fn default_negotiation() -> Negotiation {
    Negotiation::ViewerOffers
}

impl Default for RemoteStreamConfig {
    fn default() -> Self {
        Self {
            env_id: String::new(),
            viewer_id: "viewer".to_string(),
            video_codec: VideoCodec::Vp8,
            max_bitrate_bps: 4_000_000,
            max_fps: 30,
            negotiation: Negotiation::ViewerOffers,
            ice_servers: Vec::new(),
        }
    }
}

/// Session description for WebRTC signaling (SDP offer/answer).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebRtcSessionDescription {
    pub sdp_type: String,
    pub sdp: String,
}

/// An ICE candidate trickled during WebRTC connection negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebRtcCandidate {
    pub candidate: String,
    #[serde(default)]
    pub sdp_mid: Option<String>,
    #[serde(default)]
    pub sdp_m_line_index: Option<u32>,
}

/// Signaling messages exchanged with a viewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SignalingMessage {
    /// Sender -> viewer: the SDP offer.
    Offer { sdp: String },
    /// Viewer -> sender: the SDP answer.
    Answer { sdp: String },
    /// Either direction: a trickled ICE candidate.
    Candidate { candidate: String },
    /// Viewer -> sender: input events for the virtual seat.
    Input { events: Vec<InputEvent> },
    /// Sender -> viewer: connection state or an error.
    Status {
        state: String,
        detail: Option<String>,
    },
}

#[derive(Default)]
struct WorkerShared {
    offer: Option<String>,
    answer: Option<String>,
    candidates: Vec<String>,
    connected: bool,
    error: Option<String>,
}

/// A running WebRTC presentation for one environment and viewer.
pub struct PresentationWorker {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    webrtc: gst::Element,
    main_loop: glib::MainLoop,
    main_thread: Option<std::thread::JoinHandle<()>>,
    feed_thread: Option<std::thread::JoinHandle<()>>,
    shared: Arc<Mutex<WorkerShared>>,
    stop: Arc<AtomicBool>,
    env: DesktopEnvironment,
    config: RemoteStreamConfig,
}

static GST_INIT: Once = Once::new();

fn ensure_gst() -> Result<(), DesktopError> {
    let mut result = Ok(());
    GST_INIT.call_once(|| {
        if let Err(err) = gst::init() {
            result = Err(DesktopError::TransportUnavailable(format!(
                "GStreamer initialization failed: {err}"
            )));
        }
    });
    result
}

impl PresentationWorker {
    /// Builds the pipeline, starts the GStreamer main loop, and waits for the
    /// initial SDP offer.
    pub fn new(env: DesktopEnvironment, config: RemoteStreamConfig) -> Result<Self, DesktopError> {
        ensure_gst()?;
        let output = env.output();
        let width = output.width as i32;
        let height = output.height as i32;
        let fps = config.max_fps.max(1) as i32;

        let pipeline = gst::Pipeline::new();
        let appsrc = gst::ElementFactory::make("appsrc")
            .name("desktop-src")
            .property_from_str("format", "time")
            .property("is-live", true)
            .property("do-timestamp", false)
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("appsrc: {e}")))?
            .downcast::<gst_app::AppSrc>()
            .map_err(|_| DesktopError::TransportUnavailable("appsrc downcast".to_string()))?;
        appsrc.set_caps(Some(
            &gst::Caps::builder("video/x-raw")
                .field("format", "BGRA")
                .field("width", width)
                .field("height", height)
                .field("framerate", gst::Fraction::new(fps, 1))
                .build(),
        ));

        let convert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("videoconvert: {e}")))?;
        let encoder = match config.video_codec {
            VideoCodec::Vp8 => gst::ElementFactory::make("vp8enc")
                .property("deadline", 1i64)
                .property("target-bitrate", config.max_bitrate_bps as i32)
                .property("keyframe-max-dist", fps)
                .property("cpu-used", 8i32)
                .build()
                .map_err(|e| DesktopError::TransportUnavailable(format!("vp8enc: {e}")))?,
            VideoCodec::H264 => gst::ElementFactory::make("x264enc")
                .property("bitrate", (config.max_bitrate_bps / 1000) as u32)
                .property("tune", 0x00000004u32) // zerolatency
                .property("speed-preset", 1i32) // ultrafast
                .property("key-int-max", fps as u32)
                .build()
                .map_err(|e| DesktopError::TransportUnavailable(format!("x264enc: {e}")))?,
        };
        let payloader = match config.video_codec {
            VideoCodec::Vp8 => gst::ElementFactory::make("rtpvp8pay")
                .property("pt", 96u32)
                .build(),
            VideoCodec::H264 => gst::ElementFactory::make("rtph264pay")
                .property("pt", 96u32)
                .property("config-interval", -1i32)
                .build(),
        }
        .map_err(|e| DesktopError::TransportUnavailable(format!("payloader: {e}")))?;
        let queue = gst::ElementFactory::make("queue")
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("queue: {e}")))?;

        let mut webrtc_builder = gst::ElementFactory::make("webrtcbin")
            .name("sendrecv")
            .property_from_str("bundle-policy", "max-bundle");
        if let Some(server) = config.ice_servers.first() {
            webrtc_builder = webrtc_builder.property_from_str("stun-server", server.url.as_str());
        }
        let webrtc = webrtc_builder
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("webrtcbin: {e}")))?;

        let appsrc_element: gst::Element = appsrc.clone().upcast();
        let elements = [
            appsrc_element.clone(),
            convert.clone(),
            encoder.clone(),
            payloader.clone(),
            queue.clone(),
            webrtc.clone(),
        ];
        pipeline
            .add_many(elements.iter())
            .map_err(|e| DesktopError::TransportUnavailable(format!("pipeline add: {e}")))?;
        gst::Element::link_many(elements[..5].iter())
            .map_err(|e| DesktopError::TransportUnavailable(format!("pipeline link: {e}")))?;

        // webrtcbin's sink pad is a request pad; request it before linking.
        let sink = webrtc
            .request_pad_simple("sink_%u")
            .or_else(|| {
                webrtc
                    .pad_template("sink_%u")
                    .and_then(|template| webrtc.request_pad(&template, None, None))
            })
            .ok_or_else(|| {
                DesktopError::TransportUnavailable(
                    "webrtcbin did not provide a sink pad (libnice plugin missing?)".to_string(),
                )
            })?;
        queue
            .static_pad("src")
            .ok_or_else(|| DesktopError::TransportUnavailable("queue src pad".to_string()))?
            .link(&sink)
            .map_err(|e| DesktopError::TransportUnavailable(format!("webrtc link: {e}")))?;

        let shared = Arc::new(Mutex::new(WorkerShared::default()));
        let main_loop = glib::MainLoop::new(None, false);

        // Sender-offers mode: webrtcbin drives negotiation itself, which is the
        // only path that reliably produces m-lines for a native receiver.
        if config.negotiation == Negotiation::SenderOffers {
            let shared = shared.clone();
            let webrtc_weak = webrtc.downgrade();
            webrtc.connect("on-negotiation-needed", false, move |_args| {
                let webrtc = webrtc_weak.upgrade()?;
                let shared = shared.clone();
                let webrtc_inner = webrtc.clone();
                let promise = gst::Promise::with_change_func(move |reply| {
                    let Ok(Some(reply)) = reply else { return };
                    let Ok(offer) = reply.get::<gst_webrtc::WebRTCSessionDescription>("offer")
                    else {
                        return;
                    };
                    let sdp = offer.sdp().as_text().unwrap_or_default();
                    let promise = gst::Promise::with_change_func(|_| {});
                    webrtc_inner.emit_by_name::<()>("set-local-description", &[&offer, &promise]);
                    shared.lock().unwrap().offer = Some(sdp);
                });
                webrtc.emit_by_name::<()>("create-offer", &[&None::<gst::Structure>, &promise]);
                None
            });
        }

        // Trickled local ICE candidates.
        {
            let shared = shared.clone();
            webrtc.connect("on-ice-candidate", false, move |args| {
                if let Ok(candidate) = args[2].get::<String>() {
                    shared.lock().unwrap().candidates.push(candidate);
                }
                None
            });
        }

        // Connection state.
        {
            let shared = shared.clone();
            webrtc.connect_notify(Some("connection-state"), move |webrtc, _pspec| {
                let state: gst_webrtc::WebRTCPeerConnectionState =
                    webrtc.property("connection-state");
                let mut guard = shared.lock().unwrap();
                guard.connected = matches!(state, gst_webrtc::WebRTCPeerConnectionState::Connected);
            });
        }

        // Input data channel: the viewer offers it, so accept remote channels.
        {
            let env = env.clone();
            webrtc.connect("on-data-channel", false, move |args| {
                if let Ok(channel) = args[1].get::<gst_webrtc::WebRTCDataChannel>() {
                    let env = env.clone();
                    channel.connect("on-message-string", false, move |args| {
                        if let Ok(text) = args[1].get::<String>() {
                            apply_remote_input(&env, &text);
                        }
                        None
                    });
                }
                None
            });
        }

        let worker = Self {
            pipeline: pipeline.clone(),
            appsrc: appsrc.clone(),
            webrtc: webrtc.clone(),
            main_loop: main_loop.clone(),
            main_thread: None,
            feed_thread: None,
            shared: shared.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            env: env.clone(),
            config: config.clone(),
        };

        // GStreamer signals need a running main context.
        let ml = main_loop.clone();
        let pipeline_for_bus = pipeline.clone();
        let shared_for_bus = shared.clone();
        let main_thread = std::thread::Builder::new()
            .name(format!("juto-webrtc-{}", config.viewer_id))
            .spawn(move || {
                let bus = pipeline_for_bus.bus().unwrap();
                let ml_for_bus = ml.clone();
                let _guard = bus.add_watch(move |_, msg| {
                    use gst::MessageView;
                    match msg.view() {
                        MessageView::Error(err) => {
                            let mut guard = shared_for_bus.lock().unwrap();
                            guard.error = Some(format!("{} ({:?})", err.error(), err.debug()));
                            ml_for_bus.quit();
                        }
                        MessageView::Eos(_) => ml_for_bus.quit(),
                        _ => {}
                    }
                    glib::ControlFlow::Continue
                });
                ml.run();
            })
            .map_err(|e| DesktopError::TransportUnavailable(format!("webrtc thread: {e}")))?;

        let mut worker = worker;
        worker.main_thread = Some(main_thread);

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| DesktopError::TransportUnavailable(format!("pipeline start: {e}")))?;

        worker.start_feeding();
        Ok(worker)
    }

    fn start_feeding(&mut self) {
        let pool = self.env.frame_pool();
        let appsrc = self.appsrc.clone();
        let stop = self.stop.clone();
        let fps = self.config.max_fps.max(1);
        let interval = Duration::from_micros(1_000_000 / fps as u64);
        let handle = std::thread::Builder::new()
            .name(format!("juto-webrtc-feed-{}", self.config.viewer_id))
            .spawn(move || {
                let mut pixels: Vec<u8> = Vec::new();
                let mut frame_index: u64 = 0;
                while !stop.load(Ordering::SeqCst) {
                    let started = std::time::Instant::now();
                    let Ok(Some(frame)) = pool.latest_frame() else {
                        std::thread::sleep(interval);
                        continue;
                    };
                    pool.acquire_slot(frame.slot);
                    let read = pool.read_slot(frame.slot, pool.slot_size());
                    pool.release_slot(frame.slot);
                    let Ok((header, slice)) = read else {
                        std::thread::sleep(interval);
                        continue;
                    };
                    pixels.clear();
                    pixels.extend_from_slice(slice);
                    if header.size_bytes == 0 {
                        std::thread::sleep(interval);
                        continue;
                    }
                    let mut buffer = match gst::Buffer::with_size(pixels.len()) {
                        Ok(buffer) => buffer,
                        Err(_) => continue,
                    };
                    {
                        let Some(buffer_ref) = buffer.get_mut() else {
                            continue;
                        };
                        let Ok(mut map) = buffer_ref.map_writable() else {
                            continue;
                        };
                        map.as_mut_slice().copy_from_slice(&pixels);
                    }
                    frame_index = frame_index.wrapping_add(1);
                    let pts = gst::ClockTime::from_mseconds(frame_index * 1000 / fps as u64);
                    buffer.get_mut().unwrap().set_pts(pts);
                    if appsrc.push_buffer(buffer).is_err() {
                        break;
                    }
                    let elapsed = started.elapsed();
                    if elapsed < interval {
                        std::thread::sleep(interval - elapsed);
                    }
                }
            })
            .ok();
        self.feed_thread = handle;
    }

    /// Creates a local SDP offer for a viewer that answers (native receivers).
    ///
    /// Browser viewers offer instead, so they can create the input data channel;
    /// use [`PresentationWorker::answer_to`] for that direction.
    pub fn offer(&self) -> Result<String, DesktopError> {
        if self.config.negotiation != Negotiation::SenderOffers {
            return Err(DesktopError::TransportUnavailable(
                "this worker answers viewer offers; it does not create one".to_string(),
            ));
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(offer) = self.shared.lock().unwrap().offer.clone() {
                return Ok(offer);
            }
            if let Some(error) = self.shared.lock().unwrap().error.clone() {
                return Err(DesktopError::TransportUnavailable(error));
            }
            if std::time::Instant::now() >= deadline {
                return Err(DesktopError::TransportUnavailable(
                    "timed out waiting for the WebRTC offer".to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Applies the viewer's SDP answer (native receivers).
    pub fn set_answer(&self, sdp: &str) -> Result<(), DesktopError> {
        let answer = gst_webrtc::WebRTCSessionDescription::new(
            gst_webrtc::WebRTCSDPType::Answer,
            gst_sdp::SDPMessage::parse_buffer(sdp.as_bytes())
                .map_err(|e| DesktopError::TransportUnavailable(format!("bad SDP answer: {e}")))?,
        );
        let promise = gst::Promise::with_change_func(|_| {});
        self.webrtc
            .emit_by_name::<()>("set-remote-description", &[&answer, &promise]);
        Ok(())
    }

    /// Applies the viewer's SDP offer and returns the local answer.
    ///
    /// The viewer is the offerer so it can create the input data channel; the
    /// sender answers with the video it is already producing.
    pub fn answer_to(&self, offer_sdp: &str) -> Result<String, DesktopError> {
        let offer = gst_webrtc::WebRTCSessionDescription::new(
            gst_webrtc::WebRTCSDPType::Offer,
            gst_sdp::SDPMessage::parse_buffer(offer_sdp.as_bytes())
                .map_err(|e| DesktopError::TransportUnavailable(format!("bad SDP offer: {e}")))?,
        );
        let shared = self.shared.clone();
        let webrtc = self.webrtc.clone();
        glib::MainContext::default().invoke(move || {
            let promise = gst::Promise::with_change_func(|_| {});
            webrtc.emit_by_name::<()>("set-remote-description", &[&offer, &promise]);

            let shared = shared.clone();
            let webrtc_inner = webrtc.clone();
            let promise = gst::Promise::with_change_func(move |reply| {
                let Ok(Some(reply)) = reply else { return };
                let Ok(answer) = reply.get::<gst_webrtc::WebRTCSessionDescription>("answer") else {
                    return;
                };
                let sdp = answer.sdp().as_text().unwrap_or_default();
                let promise = gst::Promise::with_change_func(|_| {});
                webrtc_inner.emit_by_name::<()>("set-local-description", &[&answer, &promise]);
                shared.lock().unwrap().answer = Some(sdp);
            });
            webrtc.emit_by_name::<()>("create-answer", &[&None::<gst::Structure>, &promise]);
        });

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(answer) = self.shared.lock().unwrap().answer.clone() {
                return Ok(answer);
            }
            if let Some(error) = self.shared.lock().unwrap().error.clone() {
                return Err(DesktopError::TransportUnavailable(error));
            }
            if std::time::Instant::now() >= deadline {
                return Err(DesktopError::TransportUnavailable(
                    "timed out waiting for the WebRTC answer".to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Adds a trickled remote ICE candidate.
    pub fn add_ice_candidate(&self, candidate: &str) -> Result<(), DesktopError> {
        self.webrtc
            .emit_by_name::<()>("add-ice-candidate", &[&0u32, &candidate]);
        Ok(())
    }

    /// Drains local ICE candidates collected since the last call.
    pub fn take_local_candidates(&self) -> Vec<String> {
        std::mem::take(&mut self.shared.lock().unwrap().candidates)
    }

    /// Whether the peer connection has reached the connected state.
    pub fn is_connected(&self) -> bool {
        self.shared.lock().unwrap().connected
    }

    /// The last transport error, if any.
    pub fn error(&self) -> Option<String> {
        self.shared.lock().unwrap().error.clone()
    }

    /// The environment this worker presents.
    pub fn environment(&self) -> &DesktopEnvironment {
        &self.env
    }

    /// Stops the pipeline and joins the worker threads.
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.pipeline.set_state(gst::State::Null);
        self.main_loop.quit();
        if let Some(handle) = self.main_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.feed_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PresentationWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.pipeline.set_state(gst::State::Null);
        self.main_loop.quit();
        if let Some(handle) = self.main_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.feed_thread.take() {
            let _ = handle.join();
        }
    }
}

/// A decoded frame received from a remote presentation.
#[derive(Debug, Clone)]
pub struct ReceivedFrame {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA pixels.
    pub pixels: Vec<u8>,
}

#[derive(Default)]
struct ReceiverShared {
    offer: Option<String>,
    answer: Option<String>,
    candidates: Vec<String>,
    frame: Option<ReceivedFrame>,
    error: Option<String>,
}

/// A native WebRTC receiver that decodes a remote desktop stream.
///
/// This is the spec's "native receive/decode" path: a `webrtcbin` recvonly
/// pipeline whose decoded frames are exposed as RGBA images.
pub struct NativeReceiver {
    pipeline: gst::Pipeline,
    webrtc: gst::Element,
    main_loop: glib::MainLoop,
    main_thread: Option<std::thread::JoinHandle<()>>,
    shared: Arc<Mutex<ReceiverShared>>,
}

impl NativeReceiver {
    /// Builds a recvonly pipeline for `codec`.
    pub fn new(codec: VideoCodec) -> Result<Self, DesktopError> {
        ensure_gst()?;
        let pipeline = gst::Pipeline::new();
        let webrtc = gst::ElementFactory::make("webrtcbin")
            .name("recv")
            .property_from_str("bundle-policy", "max-bundle")
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("webrtcbin: {e}")))?;
        let queue = gst::ElementFactory::make("queue")
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("queue: {e}")))?;
        let depayloader = match codec {
            VideoCodec::Vp8 => gst::ElementFactory::make("rtpvp8depay").build(),
            VideoCodec::H264 => gst::ElementFactory::make("rtph264depay").build(),
        }
        .map_err(|e| DesktopError::TransportUnavailable(format!("depayloader: {e}")))?;
        let decoder = match codec {
            VideoCodec::Vp8 => gst::ElementFactory::make("vp8dec").build(),
            VideoCodec::H264 => gst::ElementFactory::make("avdec_h264").build(),
        }
        .map_err(|e| DesktopError::TransportUnavailable(format!("decoder: {e}")))?;
        let convert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("videoconvert: {e}")))?;
        let sink = gst::ElementFactory::make("appsink")
            .name("frames")
            .property("sync", false)
            .property("max-buffers", 2u32)
            .property("drop", true)
            .build()
            .map_err(|e| DesktopError::TransportUnavailable(format!("appsink: {e}")))?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| DesktopError::TransportUnavailable("appsink downcast".to_string()))?;
        sink.set_caps(Some(
            &gst::Caps::builder("video/x-raw")
                .field("format", "RGBA")
                .build(),
        ));

        let elements = [
            webrtc.clone(),
            queue.clone(),
            depayloader.clone(),
            decoder.clone(),
            convert.clone(),
            sink.clone().upcast::<gst::Element>(),
        ];
        pipeline
            .add_many(elements.iter())
            .map_err(|e| DesktopError::TransportUnavailable(format!("receiver add: {e}")))?;
        gst::Element::link_many(elements[1..].iter())
            .map_err(|e| DesktopError::TransportUnavailable(format!("receiver link: {e}")))?;

        // webrtcbin's src pad is a "sometimes" pad created when a transceiver is
        // negotiated, so link it from the pad-added signal.
        {
            let queue_weak = queue.downgrade();
            webrtc.connect("pad-added", false, move |args| {
                let Ok(pad) = args[1].get::<gst::Pad>() else {
                    return None;
                };
                let queue = queue_weak.upgrade()?;
                if pad.direction() == gst::PadDirection::Src
                    && let Some(sink) = queue.static_pad("sink")
                    && !sink.is_linked()
                {
                    let _ = pad.link(&sink);
                }
                None
            });
        }

        let shared = Arc::new(Mutex::new(ReceiverShared::default()));
        let main_loop = glib::MainLoop::new(None, false);

        {
            let shared = shared.clone();
            sink.set_callbacks(
                gst_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| {
                        let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                        let caps = sample.caps().ok_or(gst::FlowError::Error)?;
                        let info = gst_video::VideoInfo::from_caps(caps)
                            .map_err(|_| gst::FlowError::Error)?;
                        let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                        let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                        let width = info.width();
                        let height = info.height();
                        let stride = info.stride()[0] as usize;
                        let row = width as usize * 4;
                        let mut pixels = Vec::with_capacity(row * height as usize);
                        for y in 0..height as usize {
                            let start = y * stride;
                            let end = start + row;
                            if end > map.as_slice().len() {
                                break;
                            }
                            pixels.extend_from_slice(&map.as_slice()[start..end]);
                        }
                        shared.lock().unwrap().frame = Some(ReceivedFrame {
                            width,
                            height,
                            pixels,
                        });
                        Ok(gst::FlowSuccess::Ok)
                    })
                    .build(),
            );
        }

        {
            let shared = shared.clone();
            webrtc.connect("on-ice-candidate", false, move |args| {
                if let Ok(candidate) = args[2].get::<String>() {
                    shared.lock().unwrap().candidates.push(candidate);
                }
                None
            });
        }

        let mut receiver = Self {
            pipeline: pipeline.clone(),
            webrtc: webrtc.clone(),
            main_loop: main_loop.clone(),
            main_thread: None,
            shared: shared.clone(),
        };

        let ml = main_loop.clone();
        let pipeline_for_bus = pipeline.clone();
        let shared_for_bus = shared.clone();
        let main_thread = std::thread::Builder::new()
            .name("juto-webrtc-recv".to_string())
            .spawn(move || {
                let bus = pipeline_for_bus.bus().unwrap();
                let ml_for_bus = ml.clone();
                let _guard = bus.add_watch(move |_, msg| {
                    use gst::MessageView;
                    match msg.view() {
                        MessageView::Error(err) => {
                            shared_for_bus.lock().unwrap().error =
                                Some(format!("{} ({:?})", err.error(), err.debug()));
                            ml_for_bus.quit();
                        }
                        MessageView::Eos(_) => ml_for_bus.quit(),
                        _ => {}
                    }
                    glib::ControlFlow::Continue
                });
                ml.run();
            })
            .map_err(|e| DesktopError::TransportUnavailable(format!("receiver thread: {e}")))?;
        receiver.main_thread = Some(main_thread);

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| DesktopError::TransportUnavailable(format!("receiver start: {e}")))?;
        Ok(receiver)
    }

    /// Creates a recvonly offer for the video codec this receiver decodes.
    pub fn offer(&self, codec: VideoCodec) -> Result<String, DesktopError> {
        let encoding = match codec {
            VideoCodec::Vp8 => "VP8",
            VideoCodec::H264 => "H264",
        };
        let caps = gst::Caps::builder("application/x-rtp")
            .field("media", "video")
            .field("encoding-name", encoding)
            .field("clock-rate", 90000i32)
            .field("payload", 96i32)
            .build();
        let transceiver: gst_webrtc::WebRTCRTPTransceiver = self.webrtc.emit_by_name(
            "add-transceiver",
            &[&gst_webrtc::WebRTCRTPTransceiverDirection::Recvonly, &caps],
        );
        let _ = transceiver;

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(offer) = self.shared.lock().unwrap().offer.clone() {
                return Ok(offer);
            }
            if let Some(error) = self.shared.lock().unwrap().error.clone() {
                return Err(DesktopError::TransportUnavailable(error));
            }
            if std::time::Instant::now() >= deadline {
                return Err(DesktopError::TransportUnavailable(
                    "timed out waiting for the WebRTC offer".to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Applies the sender's offer and returns the local SDP answer.
    pub fn answer_to(&self, offer_sdp: &str) -> Result<String, DesktopError> {
        let offer = gst_webrtc::WebRTCSessionDescription::new(
            gst_webrtc::WebRTCSDPType::Offer,
            gst_sdp::SDPMessage::parse_buffer(offer_sdp.as_bytes())
                .map_err(|e| DesktopError::TransportUnavailable(format!("bad SDP offer: {e}")))?,
        );
        let shared = self.shared.clone();
        let webrtc = self.webrtc.clone();
        glib::MainContext::default().invoke(move || {
            let promise = gst::Promise::with_change_func(|_| {});
            webrtc.emit_by_name::<()>("set-remote-description", &[&offer, &promise]);

            let shared = shared.clone();
            let webrtc_inner = webrtc.clone();
            let promise = gst::Promise::with_change_func(move |reply| {
                let Ok(Some(reply)) = reply else { return };
                let Ok(answer) = reply.get::<gst_webrtc::WebRTCSessionDescription>("answer") else {
                    return;
                };
                let sdp = answer.sdp().as_text().unwrap_or_default();
                let promise = gst::Promise::with_change_func(|_| {});
                webrtc_inner.emit_by_name::<()>("set-local-description", &[&answer, &promise]);
                shared.lock().unwrap().answer = Some(sdp);
            });
            webrtc.emit_by_name::<()>("create-answer", &[&None::<gst::Structure>, &promise]);
        });

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(answer) = self.shared.lock().unwrap().answer.clone() {
                return Ok(answer);
            }
            if let Some(error) = self.shared.lock().unwrap().error.clone() {
                return Err(DesktopError::TransportUnavailable(error));
            }
            if std::time::Instant::now() >= deadline {
                return Err(DesktopError::TransportUnavailable(
                    "timed out waiting for the WebRTC answer".to_string(),
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Applies the sender's SDP answer.
    pub fn set_remote_answer(&self, sdp: &str) -> Result<(), DesktopError> {
        let answer = gst_webrtc::WebRTCSessionDescription::new(
            gst_webrtc::WebRTCSDPType::Answer,
            gst_sdp::SDPMessage::parse_buffer(sdp.as_bytes())
                .map_err(|e| DesktopError::TransportUnavailable(format!("bad SDP answer: {e}")))?,
        );
        let promise = gst::Promise::with_change_func(|_| {});
        self.webrtc
            .emit_by_name::<()>("set-remote-description", &[&answer, &promise]);
        Ok(())
    }

    /// Adds a trickled remote ICE candidate.
    pub fn add_ice_candidate(&self, candidate: &str) {
        self.webrtc
            .emit_by_name::<()>("add-ice-candidate", &[&0u32, &candidate]);
    }

    /// Drains local ICE candidates collected since the last call.
    pub fn take_local_candidates(&self) -> Vec<String> {
        std::mem::take(&mut self.shared.lock().unwrap().candidates)
    }

    /// The most recently decoded frame, if any.
    pub fn latest_frame(&self) -> Option<ReceivedFrame> {
        self.shared.lock().unwrap().frame.clone()
    }

    /// The last transport error, if any.
    pub fn error(&self) -> Option<String> {
        self.shared.lock().unwrap().error.clone()
    }

    /// Stops the pipeline and joins the worker thread.
    pub fn shutdown(mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
        self.main_loop.quit();
        if let Some(handle) = self.main_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for NativeReceiver {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
        self.main_loop.quit();
        if let Some(handle) = self.main_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Applies one JSON input message from the viewer's data channel.
fn apply_remote_input(env: &DesktopEnvironment, text: &str) {
    let events: Vec<InputEvent> = match serde_json::from_str(text) {
        Ok(events) => events,
        Err(_) => match serde_json::from_str::<SignalingMessage>(text) {
            Ok(SignalingMessage::Input { events }) => events,
            _ => return,
        },
    };
    if events.is_empty() {
        return;
    }
    let owner = env.get_status().owner_principal;
    let lease = match env.get_status().active_control_lease {
        Some(lease) if lease.holder == owner => lease,
        _ => match env.set_control(&owner, "acquire", false) {
            Ok(Some(lease)) => lease,
            _ => return,
        },
    };
    let batch = InputBatch {
        seq: 0,
        client_timestamp_us: 0,
        events,
    };
    let _ = env.dispatch_input(&owner, lease.epoch, env.generation(), batch);
}
