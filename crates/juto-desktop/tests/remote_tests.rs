//! End-to-end WebRTC tests: a presentation worker streams a real desktop to a
//! native receiver over DTLS/SRTP.

mod common;

use std::time::{Duration, Instant};

use juto_desktop::remote::{
    IceServer, NativeReceiver, Negotiation, PresentationWorker, RemoteStreamConfig, VideoCodec,
};
use juto_desktop::supervisor::{
    DesktopSupervisor, EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits,
};

/// GStreamer pipelines share process-global state; these tests build several
/// pipelines in one process, so they run one at a time.
static PIPELINE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The worker binary is the daemon; point the supervisor at it.
fn worker_binary() -> &'static str {
    env!("CARGO_BIN_EXE_juto-desktop")
}

fn unique_env_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::now_v7())
}

fn test_config(env_id: &str) -> EnvironmentConfig {
    // SAFETY: every test sets the same value, and it is read before spawning.
    unsafe { std::env::set_var("JUTO_DESKTOP_WORKER", worker_binary()) };
    EnvironmentConfig {
        env_id: env_id.to_string(),
        owner_principal: "agent:test".to_string(),
        output: OutputConfig {
            width: 640,
            height: 480,
            scale: 1,
            fps: 30,
        },
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
        lease_ttl_secs: 6,
        compositor_mode: juto_desktop::CompositorMode::Worker,
        owner_heartbeat_timeout_secs: 60,
    }
}

#[test]
fn remote_bitrate_is_bounded_and_ice_servers_are_applied() {
    let _guard = PIPELINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(
            test_config(&unique_env_id("bitrate")),
            &unique_env_id("req"),
        )
        .expect("create environment");

    let config = RemoteStreamConfig {
        env_id: env.env_id().to_string(),
        viewer_id: "bitrate-viewer".to_string(),
        negotiation: Negotiation::SenderOffers,
        // Above the environment's total-media ceiling.
        max_bitrate_bps: 50_000_000,
        ice_servers: vec![
            IceServer {
                url: "stun:stun.example.invalid:3478".to_string(),
                username: None,
                credential: None,
            },
            IceServer {
                url: "turn:turn.example.invalid:3478".to_string(),
                username: Some("user".to_string()),
                credential: Some("pass".to_string()),
            },
        ],
        ..RemoteStreamConfig::default()
    };

    let worker = PresentationWorker::new(env.clone(), config).expect("presentation worker");
    assert_eq!(
        worker.effective_bitrate_bps(),
        env.limits().remote_bitrate_max_bps,
        "the requested bitrate must be clamped to the environment ceiling"
    );
    // A TURN-configured session still negotiates: the offer is produced even
    // though the relay is unreachable from this host.
    assert!(worker.offer().is_ok(), "the offer must still be produced");

    worker.shutdown();
    env.destroy();
}

#[test]
fn a_configured_audio_source_adds_an_audio_track() {
    let _guard = PIPELINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("audio")), &unique_env_id("req"))
        .expect("create environment");

    let config = RemoteStreamConfig {
        env_id: env.env_id().to_string(),
        viewer_id: "audio-viewer".to_string(),
        negotiation: Negotiation::SenderOffers,
        // A private audio server is a deployment prerequisite; the track is
        // wired from whatever source the environment provides.
        audio_source: Some("audiotestsrc is-live=true".to_string()),
        ..RemoteStreamConfig::default()
    };
    let worker = PresentationWorker::new(env.clone(), config).expect("presentation worker");
    let offer = worker.offer().expect("worker produced an SDP offer");
    assert!(
        offer.contains("m=audio"),
        "the offer must carry an audio track: {offer}"
    );
    assert!(
        offer.contains("m=video"),
        "the offer must still carry the video track: {offer}"
    );
    // Audio and video share the environment's total-media ceiling.
    assert_eq!(worker.effective_bitrate_bps(), 4_000_000);

    worker.shutdown();
    env.destroy();
}

#[test]
fn webrtc_streams_composed_frames_to_a_native_receiver() {
    let _guard = PIPELINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let supervisor = DesktopSupervisor::new();
    let env_id = format!("remote-{}", uuid::Uuid::now_v7());
    let env = supervisor
        .create_environment(test_config(&env_id), &unique_env_id("req"))
        .expect("create environment");

    let socket = env.socket_path().to_string_lossy().to_string();
    let client = std::thread::spawn(move || common::run(&socket, 30));

    // Wait until the client's full two-tone window is composed, not just its first pixel.
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut ready = false;
    while Instant::now() < deadline {
        if let Ok(info) = env.observe(Duration::from_secs(2)) {
            let pool = env.frame_pool();
            pool.acquire_slot(info.slot);
            let read = pool.read_slot(info.slot, pool.slot_size());
            pool.release_slot(info.slot);
            if let Ok((header, pixels)) = read
                && header.width > 0
                && {
                    let at = |x: u32, y: u32| {
                        let index = (y * header.stride + x * 4) as usize;
                        pixels.get(index..index + 4)
                    };
                    at(10, 10) == Some(&common::RED[..]) && at(300, 10) == Some(&common::BLUE[..])
                }
            {
                ready = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(ready, "the test client never rendered its two-tone frame");

    let worker = PresentationWorker::new(
        env.clone(),
        RemoteStreamConfig {
            env_id: env_id.clone(),
            viewer_id: "native-receiver".to_string(),
            video_codec: VideoCodec::Vp8,
            max_bitrate_bps: 4_000_000,
            max_fps: 30,
            negotiation: Negotiation::SenderOffers,
            ice_servers: Vec::new(),
            audio_source: None,
        },
    )
    .expect("start presentation worker");

    let receiver = NativeReceiver::new(VideoCodec::Vp8).expect("start native receiver");
    let offer = worker.offer().expect("worker produced an SDP offer");
    assert!(
        offer.contains("m=video"),
        "offer has a video m-line: {offer}"
    );
    let answer = receiver.answer_to(&offer).expect("receiver answered");
    worker
        .set_answer(&answer)
        .expect("worker applied the answer");

    // Trickle ICE both ways until a frame decodes.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut frame = None;
    while Instant::now() < deadline {
        for candidate in worker.take_local_candidates() {
            receiver.add_ice_candidate(&candidate);
        }
        for candidate in receiver.take_local_candidates() {
            worker.add_ice_candidate(&candidate).ok();
        }
        if let Some(error) = worker.error() {
            panic!("sender error: {error}");
        }
        if let Some(error) = receiver.error() {
            panic!("receiver error: {error}");
        }
        if let Some(decoded) = receiver.latest_frame() {
            frame = Some(decoded);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let frame = frame.expect("no frame decoded over WebRTC");
    assert_eq!(frame.width, 640);
    assert_eq!(frame.height, 480);
    let pixel = |x: u32, y: u32| -> [u8; 4] {
        let index = ((y * frame.width + x) * 4) as usize;
        [
            frame.pixels[index],
            frame.pixels[index + 1],
            frame.pixels[index + 2],
            frame.pixels[index + 3],
        ]
    };
    // The decoded stream is RGBA; the client's window is at the frame origin.
    // VP8 is lossy (4:2:0 chroma + quantization), so assert dominant channels.
    let left = pixel(10, 10);
    assert!(
        left[0] > 200 && left[1] < 60 && left[2] < 60,
        "left half is red: {left:?}"
    );
    let right = pixel(300, 10);
    assert!(
        right[2] > 200 && right[0] < 60 && right[1] < 60,
        "right half is blue: {right:?}"
    );

    receiver.shutdown();
    worker.shutdown();
    let _ = client.join();
    env.destroy();
}
