//! End-to-end WebRTC tests: a presentation worker streams a real desktop to a
//! native receiver over DTLS/SRTP.

mod common;

use std::time::{Duration, Instant};

use juto_desktop::remote::{
    NativeReceiver, Negotiation, PresentationWorker, RemoteStreamConfig, VideoCodec,
};
use juto_desktop::supervisor::{
    DesktopSupervisor, EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits,
};

fn test_config(env_id: &str) -> EnvironmentConfig {
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
    }
}

#[test]
fn webrtc_streams_composed_frames_to_a_native_receiver() {
    let supervisor = DesktopSupervisor::new();
    let env_id = format!("remote-{}", uuid::Uuid::now_v7());
    let env = supervisor
        .create_environment(test_config(&env_id))
        .expect("create environment");

    let socket = env.socket_path().to_string_lossy().to_string();
    let client = std::thread::spawn(move || common::run(&socket, 30));

    // Wait until the client has drawn something.
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut ready = false;
    while Instant::now() < deadline {
        if let Ok(info) = env.observe(Duration::from_secs(2)) {
            let pool = env.frame_pool();
            pool.acquire_slot(info.slot);
            let mut pixels = Vec::new();
            let header = pool.read_pixels(info.slot, &mut pixels);
            pool.release_slot(info.slot);
            if let Ok(header) = header
                && header.width > 0
                && pixels.len() >= 4 * 4
                && pixels[0..4] == common::RED
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
