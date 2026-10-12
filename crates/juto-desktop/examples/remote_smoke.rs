//! Browser-viewer smoke harness.
//!
//! Creates a desktop environment, launches the deterministic Wayland client,
//! starts the signaling server, and prints the viewer URL. It then waits for the
//! browser viewer to report its own session state (video size and a sampled
//! pixel) and exits 0 on success.
//!
//! Usage: `remote_smoke <test-client-binary> [timeout-seconds]`

#[path = "../tests/common/mod.rs"]
mod client;

use std::sync::Arc;
use std::time::{Duration, Instant};

use juto_desktop::protocol::GrantRole;
use juto_desktop::signaling::SignalingServer;
use juto_desktop::supervisor::{
    DesktopSupervisor, EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client_binary = std::env::args()
        .nth(1)
        .ok_or("usage: remote_smoke <test-client-binary> [timeout-seconds]")?;
    let timeout: u64 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);

    let supervisor = Arc::new(DesktopSupervisor::new());
    let env = supervisor.create_environment(EnvironmentConfig {
        env_id: format!("remote-smoke-{}", uuid::Uuid::now_v7()),
        owner_principal: "agent:smoke".to_string(),
        output: OutputConfig {
            width: 640,
            height: 480,
            scale: 1,
            fps: 30,
        },
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
    })?;

    env.launch(
        "test-client",
        &[client_binary, "120".to_string()],
        None,
        &[],
    )?;

    // Wait for the client's two-tone frame before offering the stream.
    let deadline = Instant::now() + Duration::from_secs(20);
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
                && pixels.len() >= 4
                && pixels[0..4] == client::RED
            {
                ready = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    if !ready {
        return Err("the test client never rendered its two-tone frame".into());
    }

    let server = SignalingServer::start(supervisor.clone())?;
    let grant = server.issue_token(
        env.env_id(),
        "browser-viewer",
        vec![GrantRole::View, GrantRole::Control],
    );
    let url = server.viewer_url(&grant.token);
    println!("VIEWER_URL={url}");
    println!("TOKEN={}", grant.token);
    use std::io::Write;
    std::io::stdout().flush().ok();

    let deadline = Instant::now() + Duration::from_secs(timeout);
    let mut report = None;
    while Instant::now() < deadline {
        if let Some(found) = server.report(&grant.token) {
            if found.connected && found.video_width > 0 {
                report = Some(found);
                break;
            }
            if found.error.is_some() {
                report = Some(found);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    let report = report.ok_or("the browser viewer never reported a live session")?;
    println!("REPORT={}", serde_json::to_string(&report)?);
    if let Some(error) = &report.error {
        return Err(format!("viewer reported an error: {error}").into());
    }
    let pixel = report
        .sampled_pixel
        .clone()
        .ok_or("viewer did not sample a pixel")?;
    if pixel.len() < 3 || pixel[0] < 200 || pixel[1] > 60 || pixel[2] > 60 {
        return Err(format!("viewer sampled an unexpected pixel: {pixel:?}").into());
    }
    if !report.input_sent {
        return Err("the viewer never sent an input event over the data channel".into());
    }
    // The data channel input must have reached the environment's seat.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if env.get_status().active_control_lease.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    if env.get_status().active_control_lease.is_none() {
        return Err("data-channel input did not reach the environment".into());
    }
    println!("BROWSER_SMOKE_OK");
    env.destroy();
    Ok(())
}
