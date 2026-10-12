//! Standalone runner for the deterministic Wayland test client.
//!
//! Connects to `WAYLAND_DISPLAY`, maps one toplevel, paints a fixed two-tone
//! pattern, and stays alive for `JUTO_TEST_CLIENT_SECONDS` (default 3).

#[path = "../tests/common/mod.rs"]
mod client;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let socket = std::env::var("WAYLAND_DISPLAY")?;
    let seconds = std::env::var("JUTO_TEST_CLIENT_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let frames = client::run(&socket, seconds)?;
    eprintln!("test-client: presented {frames} frame(s)");
    Ok(())
}
