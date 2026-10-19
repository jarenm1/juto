//! Input-to-published-frame latency of a live desktop environment.
//!
//! Starts an environment with the production worker compositor, maps one
//! client, and runs `samples` rounds. Each round sends one pointer move through
//! the real input path, then times how long until the frame pool publishes a
//! frame stamped after the input was applied. The observer either sleeps
//! between reads (`poll`, the viewer's pre-change 33 ms cadence) or blocks on
//! the publish futex (`wait`). Also reports the composed-frame rate while idle.
//!
//! Build first so the worker binary exists next to the example:
//! `cargo build -p juto-desktop --bin juto-desktop --example frame_latency`
//!
//! Usage: `frame_latency <poll|wait> [fps] [samples]`

#[path = "../tests/common/mod.rs"]
mod client;

use std::error::Error;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use juto_desktop::protocol::{InputBatch, InputEvent};
use juto_desktop::shm::ShmReader;
use juto_desktop::supervisor::{
    CompositorMode, DesktopEnvironment, DesktopSupervisor, EnvironmentConfig, OutputConfig,
    RenderProfile, SystemLimits,
};

/// The viewer's poll cadence before the change; `poll` measures that path.
const POLL_INTERVAL: Duration = Duration::from_millis(33);
const WAIT_SLICE: Duration = Duration::from_millis(100);
const SAMPLE_TIMEOUT: Duration = Duration::from_secs(5);
/// Spacing between samples; not a multiple of the frame interval, so the
/// observer does not phase-lock with the pacer.
const SAMPLE_GAP: Duration = Duration::from_millis(13);
const WARMUP: Duration = Duration::from_millis(1500);
const IDLE_WINDOW: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Observer {
    Poll,
    Wait,
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Blocks until a frame newer than `last_id` and stamped at or after `sent_us`
/// is published. Returns its id and the instant the observer saw it.
fn await_frame(
    pool: &ShmReader,
    observer: Observer,
    last_id: u64,
    sent_us: u64,
) -> Result<(u64, Instant), String> {
    let deadline = Instant::now() + SAMPLE_TIMEOUT;
    loop {
        // Read the sequence before the frame so a publish between the two
        // reads makes `wait_for_frame` return immediately.
        let seq = pool.publish_seq();
        let latest = pool
            .latest_frame()
            .map_err(|e| format!("latest_frame: {e}"))?;
        if let Some(frame) =
            latest.filter(|f| f.header.frame_id > last_id && f.header.timestamp_us >= sent_us)
        {
            return Ok((frame.header.frame_id, Instant::now()));
        }
        if Instant::now() >= deadline {
            return Err("no frame published after the input within the timeout".into());
        }
        match observer {
            Observer::Poll => std::thread::sleep(POLL_INTERVAL),
            Observer::Wait => {
                pool.wait_for_frame(seq, WAIT_SLICE)
                    .map_err(|e| format!("wait_for_frame: {e}"))?;
            }
        }
    }
}

fn latest_id(pool: &ShmReader) -> Result<u64, String> {
    Ok(pool
        .latest_frame()
        .map_err(|e| format!("latest_frame: {e}"))?
        .map(|f| f.header.frame_id)
        .unwrap_or(0))
}

fn acquire_control(env: &DesktopEnvironment) -> Result<(String, u64), String> {
    let owner = env.get_status().owner_principal;
    match env.set_control(&owner, "acquire", false) {
        Ok(Some(lease)) => Ok((owner, lease.epoch)),
        Ok(None) => Err("set_control acquire returned no lease".into()),
        Err(e) => Err(format!("set_control acquire: {e}")),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let observer = match args.next().as_deref() {
        Some("poll") => Observer::Poll,
        Some("wait") => Observer::Wait,
        _ => return Err("usage: frame_latency <poll|wait> [fps] [samples]".into()),
    };
    let fps: u32 = match args.next() {
        Some(v) => v.parse()?,
        None => 30,
    };
    let samples: usize = match args.next() {
        Some(v) => v.parse()?,
        None => 200,
    };
    if samples == 0 {
        return Err("samples must be at least 1".into());
    }

    if std::env::var_os("JUTO_DESKTOP_WORKER").is_none() {
        // target/<profile>/examples/frame_latency -> target/<profile>/juto-desktop
        let exe = std::env::current_exe()?;
        let worker = exe
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("juto-desktop"))
            .ok_or("cannot locate juto-desktop next to the example")?;
        // SAFETY: set before any thread is spawned.
        unsafe { std::env::set_var("JUTO_DESKTOP_WORKER", worker) };
    }

    let supervisor = DesktopSupervisor::new();
    let env_id = format!("latency-{}", std::process::id());
    let config = EnvironmentConfig {
        env_id: env_id.clone(),
        owner_principal: "agent:latency".to_string(),
        output: OutputConfig {
            width: 640,
            height: 480,
            scale: 1,
            fps,
        },
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
        lease_ttl_secs: 60,
        compositor_mode: CompositorMode::Worker,
        owner_heartbeat_timeout_secs: 60,
        idle: juto_desktop::IdlePolicy::default(),
    };
    let env = supervisor
        .create_environment(config, &format!("req-{env_id}"))
        .map_err(|e| format!("create_environment: {e}"))?;

    let socket = env.socket_path().to_string_lossy().to_string();
    // Detached: the process exit ends the client along with the environment.
    std::thread::spawn(move || client::run(&socket, 600));

    let pool = env.frame_pool();
    std::thread::sleep(WARMUP);
    let (owner, epoch) = acquire_control(&env)?;

    let mut last_id = latest_id(&pool)?;
    let mut latencies_ms = Vec::with_capacity(samples);
    for i in 0..samples {
        let dx = if i % 2 == 0 { 1 } else { -1 };
        let batch = InputBatch {
            seq: i as u64 + 1,
            client_timestamp_us: 0,
            events: vec![InputEvent::PointerDelta { dx, dy: 0 }],
        };
        env.dispatch_input(&owner, epoch, env.generation(), batch)
            .map_err(|e| format!("dispatch_input: {e}"))?;
        // Timed from after dispatch returns, so the input is already applied.
        let sent = Instant::now();
        let sent_us = now_us();
        let (id, seen) = await_frame(&pool, observer, last_id, sent_us)?;
        last_id = id;
        latencies_ms.push(seen.duration_since(sent).as_secs_f64() * 1e3);
        std::thread::sleep(SAMPLE_GAP);
    }

    let idle_start_id = latest_id(&pool)?;
    let idle_start = Instant::now();
    std::thread::sleep(IDLE_WINDOW);
    let idle_frames = latest_id(&pool)?.saturating_sub(idle_start_id);
    let idle_fps = idle_frames as f64 / idle_start.elapsed().as_secs_f64();

    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let mean = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    println!(
        "observer={} fps={} samples={} input_to_frame_ms p50={:.2} p95={:.2} max={:.2} mean={:.2} idle_composed_fps={:.1}",
        if observer == Observer::Poll {
            "poll"
        } else {
            "wait"
        },
        fps,
        samples,
        percentile(&latencies_ms, 0.50),
        percentile(&latencies_ms, 0.95),
        latencies_ms[latencies_ms.len() - 1],
        mean,
        idle_fps,
    );

    env.destroy();
    Ok(())
}
