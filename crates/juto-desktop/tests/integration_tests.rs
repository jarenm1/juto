//! Integration tests for the headless compositor and environment supervisor.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use juto_desktop::DesktopError;
use juto_desktop::protocol::*;
use juto_desktop::shm::{ShmFramePool, ShmReader};
use juto_desktop::supervisor::{
    DesktopSupervisor, EnvironmentConfig, HostPolicy, OutputConfig, RenderProfile, SystemLimits,
};

/// The worker binary is the daemon; point the supervisor at it.
fn worker_binary() -> &'static str {
    env!("CARGO_BIN_EXE_juto-desktop")
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

fn unique_env_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::now_v7())
}

#[test]
fn environment_lifecycle_admission_and_generation() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(
            test_config(&unique_env_id("lifecycle")),
            &unique_env_id("req"),
        )
        .expect("create environment");

    assert_eq!(env.generation(), 1);
    let status = env.get_status();
    assert!(status.is_ready);
    assert_eq!(status.output.width, 640);
    assert!(env.socket_path().exists(), "wayland socket must exist");

    // 4K @ 60 FPS exceeds the software profile's admitted pixel rate.
    let excessive = OutputConfig {
        width: 3840,
        height: 2160,
        scale: 1,
        fps: 60,
    };
    assert!(env.configure_output(1, excessive).is_err());

    // A valid resize increments the geometry generation.
    let valid = OutputConfig {
        width: 800,
        height: 600,
        scale: 1,
        fps: 30,
    };
    let new_gen = env
        .configure_output(1, valid.clone())
        .expect("valid resize");
    assert_eq!(new_gen, 2);
    assert_eq!(env.generation(), 2);
    assert_eq!(env.output().width, 800);

    // Stale generation is rejected.
    assert!(env.configure_output(1, valid).is_err());

    env.destroy();
    assert!(!env.runtime_dir().exists(), "runtime dir must be removed");
}

#[test]
fn shm_pool_sealing_and_roundtrip() {
    let pool = ShmFramePool::create("seal-test", 64, 48).expect("create pool");
    let pixels = vec![0x7Fu8; 64 * 48 * 4];
    let frame = pool
        .write_frame(7, 3, 1234, 64, 48, &pixels)
        .expect("write frame");
    assert_eq!(frame.header.frame_id, 7);
    assert_eq!(frame.header.generation, 3);

    // A reader mapping the sealed memfd sees the written payload.
    let fd = unsafe { libc::dup(pool.raw_fd()) };
    assert!(fd >= 0);
    let reader = ShmReader::attach(fd, pool.size()).expect("attach reader");
    let (header, payload) = reader
        .read_slot(frame.slot, pool.slot_size())
        .expect("read slot");
    assert_eq!(header.frame_id, 7);
    assert_eq!(header.width, 64);
    assert_eq!(payload.len(), 64 * 48 * 4);
    assert!(payload.iter().all(|b| *b == 0x7F));

    // The producer mapping remains writable after sealing.
    let second = pool
        .write_frame(8, 3, 1235, 64, 48, &pixels)
        .expect("second write");
    assert_ne!(second.slot, frame.slot);
}

#[test]
fn control_lease_and_human_preemption() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("lease")), &unique_env_id("req"))
        .expect("create environment");

    let agent = env
        .set_control("agent:test", "acquire", false)
        .expect("agent acquires")
        .expect("lease present");
    assert_eq!(agent.epoch, 1);

    // A second agent cannot steal control.
    assert!(env.set_control("agent:other", "acquire", false).is_err());

    // A human preempts the agent and increments the epoch.
    let human = env
        .set_control("human:test", "acquire", true)
        .expect("human preempts")
        .expect("lease present");
    assert_eq!(human.epoch, 2);
    assert!(human.is_human);

    // Releasing returns to no controller.
    assert!(
        env.set_control("human:test", "release", true)
            .unwrap()
            .is_none()
    );
    assert!(env.get_status().active_control_lease.is_none());

    env.destroy();
}

#[test]
fn input_dispatch_validation() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("input")), &unique_env_id("req"))
        .expect("create environment");

    let batch = InputBatch {
        seq: 1,
        client_timestamp_us: 0,
        events: vec![InputEvent::PointerMoveAbs { x: 10, y: 10 }],
    };

    // No lease: unauthorized.
    assert!(
        env.dispatch_input("agent:test", 1, 1, batch.clone())
            .is_err()
    );

    let lease = env
        .set_control("agent:test", "acquire", false)
        .unwrap()
        .unwrap();

    // Stale epoch and stale generation are rejected.
    assert!(
        env.dispatch_input("agent:test", 99, 1, batch.clone())
            .is_err()
    );
    assert!(
        env.dispatch_input("agent:test", lease.epoch, 99, batch.clone())
            .is_err()
    );

    // Matching lease and generation are accepted.
    let watermark = env
        .dispatch_input("agent:test", lease.epoch, 1, batch)
        .expect("input accepted");
    assert_eq!(watermark, 1);

    // Text input is forwarded to the focused text-input instance; with no
    // focused client it is accepted and applied to nothing.
    let text_batch = InputBatch {
        seq: 2,
        client_timestamp_us: 0,
        events: vec![InputEvent::TextCommit {
            text: "hello".to_string(),
            focus_epoch: 1,
        }],
    };
    assert!(
        env.dispatch_input("agent:test", lease.epoch, 1, text_batch)
            .is_ok()
    );

    env.destroy();
}

#[test]
fn compositor_renders_real_client_frame() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("render")), &unique_env_id("req"))
        .expect("create environment");

    let socket = env.socket_path().to_string_lossy().to_string();
    let client = std::thread::spawn(move || common::run(&socket, 10));

    let pool = env.frame_pool();
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut verified = None;
    while Instant::now() < deadline {
        let info = match env.observe(Duration::from_secs(2)) {
            Ok(info) => info,
            Err(err) => panic!("observe failed: {err}"),
        };
        let left = read_pixel(&pool, info.slot, 10, 10, info.width);
        let right = read_pixel(&pool, info.slot, 300, 10, info.width);
        if left == common::RED && right == common::BLUE {
            verified = Some(info);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let info = verified.expect("compositor never produced the client's two-tone frame");
    assert_eq!(info.width, 640);
    assert_eq!(info.height, 480);
    assert_eq!(info.generation, 1);

    let frames = client.join().expect("client thread").expect("client run");
    assert!(frames > 0, "client presented at least one frame");

    env.destroy();
}

#[test]
fn hardware_profile_renders_and_serializes_on_the_shared_device() {
    let supervisor = DesktopSupervisor::new();
    let mut config = test_config(&unique_env_id("hardware"));
    config.render_profile = RenderProfile::Hardware;
    let env = supervisor
        .create_environment(config, &unique_env_id("req"))
        .expect("hardware environment");

    let socket = env.socket_path().to_string_lossy().to_string();
    let client = std::thread::spawn(move || common::run(&socket, 10));

    let pool = env.frame_pool();
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut verified = None;
    while Instant::now() < deadline {
        let info = env.observe(Duration::from_secs(2)).expect("observe");
        let left = read_pixel(&pool, info.slot, 10, 10, info.width);
        let right = read_pixel(&pool, info.slot, 300, 10, info.width);
        if left == common::RED && right == common::BLUE {
            verified = Some(info);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        verified.is_some(),
        "the hardware profile never composed the client's two-tone frame"
    );

    // A second hardware environment cannot share the physical device.
    let mut second = test_config(&unique_env_id("hardware"));
    second.render_profile = RenderProfile::Hardware;
    assert!(matches!(
        supervisor.create_environment(second, &unique_env_id("req")),
        Err(DesktopError::GpuBusy)
    ));

    env.destroy();
    let _ = client.join();

    // Destroying the environment releases the shared device.
    let mut third = test_config(&unique_env_id("hardware"));
    third.render_profile = RenderProfile::Hardware;
    let third = supervisor
        .create_environment(third, &unique_env_id("req"))
        .expect("the shared device is released after destroy");
    third.destroy();
}

#[test]
fn the_environment_has_a_private_audio_server() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("audio")), &unique_env_id("req"))
        .expect("create environment");

    // The environment's own PulseAudio socket must appear.
    let socket = env.runtime_dir().join("pulse/native");
    let deadline = Instant::now() + Duration::from_secs(20);
    while !socket.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        socket.exists(),
        "the environment's private audio socket never appeared"
    );

    // The private server must expose the environment's own sink, which is what
    // the remote encoder captures; the host graph is never involved.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut sink = false;
    while Instant::now() < deadline {
        let nodes = std::process::Command::new("pw-cli")
            .args(["ls", "Node"])
            .env("PIPEWIRE_RUNTIME_DIR", env.runtime_dir())
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        if nodes.contains(juto_desktop::supervisor::AUDIO_SINK_NAME) && nodes.contains("Audio/Sink")
        {
            sink = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(sink, "the environment's private audio sink never appeared");

    // The encoder is pointed at the environment's sink monitor.
    assert_eq!(
        env.audio_source(),
        format!(
            "pulsesrc device={}.monitor",
            juto_desktop::supervisor::AUDIO_SINK_NAME
        )
    );

    env.destroy();
}

#[test]
fn host_aggregate_pixel_rate_is_admitted() {
    // 640x480@30 is 9,216,000 px/s, so two fit and a third does not.
    let supervisor = DesktopSupervisor::with_host_policy(HostPolicy {
        max_aggregate_pixel_rate: 20_000_000,
    });
    let first = supervisor
        .create_environment(test_config(&unique_env_id("agg")), &unique_env_id("req"))
        .expect("first environment");
    let second = supervisor
        .create_environment(test_config(&unique_env_id("agg")), &unique_env_id("req"))
        .expect("second environment");
    assert!(matches!(
        supervisor.create_environment(test_config(&unique_env_id("agg")), &unique_env_id("req")),
        Err(DesktopError::LimitExceeded(_))
    ));

    // Destroying an environment releases its share of the host budget.
    first.destroy();
    let third = supervisor
        .create_environment(test_config(&unique_env_id("agg")), &unique_env_id("req"))
        .expect("the host budget is released after destroy");
    second.destroy();
    third.destroy();
}

#[test]
fn xwayland_maps_an_x11_client_into_the_desktop() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("x11")), &unique_env_id("req"))
        .expect("create environment");

    // Xwayland is spawned per environment; wait for it to report its display.
    let deadline = Instant::now() + Duration::from_secs(20);
    while env.x11_display().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    let display = env.x11_display().expect("Xwayland must become ready");
    assert!(display > 0, "Xwayland must allocate a display number");

    env.launch(
        "xeyes",
        &[
            "xeyes".to_string(),
            "-geometry".to_string(),
            "200x150".to_string(),
        ],
        None,
        &[],
        &unique_env_id("req"),
    )
    .expect("launch xeyes");

    // The X11 client's window must appear in the composed desktop.
    let pool = env.frame_pool();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut mapped = false;
    let mut painted = false;
    while Instant::now() < deadline {
        let info = env.observe(Duration::from_secs(2)).expect("observe");
        if env
            .get_status()
            .mapped_windows
            .iter()
            .any(|w| w.app_id.contains("xeyes") || w.title.contains("xeyes"))
        {
            mapped = true;
        }
        // xeyes paints a light window over the dark compositor background.
        let sample = read_pixel(&pool, info.slot, 40, 40, info.width);
        if mapped && sample != [13, 15, 23, 255] {
            painted = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(mapped, "the X11 client's window was never mapped");
    assert!(
        painted,
        "the composed desktop never showed the X11 client's pixels"
    );

    env.destroy();
}

#[test]
fn create_and_launch_are_idempotent_per_request_id() {
    let supervisor = DesktopSupervisor::new();
    let first_id = unique_env_id("idem");
    let first = supervisor
        .create_environment(test_config(&first_id), "req-1")
        .expect("create environment");

    // A retried Create with the same owner and request id returns the existing
    // environment instead of allocating the one named in the retry.
    let other_id = unique_env_id("idem-other");
    let repeated = supervisor
        .create_environment(test_config(&other_id), "req-1")
        .expect("repeat create");
    assert_eq!(repeated.env_id(), first_id);
    assert!(
        supervisor.get_environment(&other_id).is_err(),
        "a retried Create must not allocate a second environment"
    );

    // A retried Launch with the same request id returns the running handle.
    let handle = first
        .launch(
            "sleeper",
            &["sleep".to_string(), "60".to_string()],
            None,
            &[],
            "launch-1",
        )
        .expect("launch");
    let repeated = first
        .launch(
            "sleeper",
            &["sleep".to_string(), "60".to_string()],
            None,
            &[],
            "launch-1",
        )
        .expect("repeat launch");
    assert_eq!(repeated.pid, handle.pid);
    assert_eq!(
        first.apps().len(),
        1,
        "a retried Launch must not double-launch"
    );

    first.destroy();
}

#[test]
fn destroy_terminates_apps_and_removes_runtime_dir() {
    let supervisor = DesktopSupervisor::new();
    let env_id = unique_env_id("cleanup");
    let env = supervisor
        .create_environment(test_config(&env_id), &unique_env_id("req"))
        .expect("create environment");

    let handle = env
        .launch(
            "sleeper",
            &["sleep".to_string(), "300".to_string()],
            None,
            &[],
            &unique_env_id("req"),
        )
        .expect("launch sleep");
    assert!(handle.pid > 0);
    assert_eq!(env.apps().len(), 1);

    let unit = handle.unit.clone().expect("systemd scope unit");
    assert!(
        wait_for_unit_active(&unit, Duration::from_secs(5)),
        "scope must become active after launch"
    );

    let runtime_dir = env.runtime_dir().to_path_buf();
    assert!(runtime_dir.exists());

    env.destroy();

    assert!(
        !runtime_dir.exists(),
        "runtime dir must be removed on destroy"
    );
    assert!(
        !unit_is_active(&unit),
        "application scope must be stopped on destroy"
    );
    assert!(env.apps().is_empty());
}

#[test]
fn expired_control_lease_is_released_and_rejected() {
    let supervisor = DesktopSupervisor::new();
    let mut config = test_config(&unique_env_id("expiry"));
    config.lease_ttl_secs = 1;
    let env = supervisor
        .create_environment(config, &unique_env_id("req"))
        .expect("create environment");

    let lease = env
        .set_control("agent:test", "acquire", false)
        .unwrap()
        .unwrap();
    assert_eq!(lease.epoch, 1);

    // Renewal by the same holder keeps the epoch.
    let renewed = env
        .set_control("agent:test", "acquire", false)
        .unwrap()
        .unwrap();
    assert_eq!(renewed.epoch, 1);

    // Wait past the TTL: the lease is reported gone and input is rejected.
    std::thread::sleep(Duration::from_millis(2200));
    assert!(env.get_status().active_control_lease.is_none());
    let batch = InputBatch {
        seq: 1,
        client_timestamp_us: 0,
        events: vec![InputEvent::PointerMoveAbs { x: 1, y: 1 }],
    };
    // The reaper clears the lease, so the call is unauthorized; a lease that is
    // still present but past its deadline is rejected as stale.
    assert!(matches!(
        env.dispatch_input("agent:test", lease.epoch, 1, batch),
        Err(juto_desktop::DesktopError::StaleLease { .. })
            | Err(juto_desktop::DesktopError::Unauthorized { .. })
    ));

    env.destroy();
}

#[test]
fn expired_grants_are_not_reported() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("grants")), &unique_env_id("req"))
        .expect("create environment");

    let live = env.issue_grant(
        "human:test".to_string(),
        vec![GrantRole::View],
        "session".to_string(),
        60,
    );
    let expired = env.issue_grant(
        "human:other".to_string(),
        vec![GrantRole::View],
        "session".to_string(),
        0,
    );
    let grants = env.get_status().active_grants;
    assert!(grants.iter().any(|grant| grant.grant_id == live.grant_id));
    assert!(
        !grants
            .iter()
            .any(|grant| grant.grant_id == expired.grant_id)
    );

    env.destroy();
}

#[test]
fn reconcile_stops_orphan_scopes_and_removes_stale_directories() {
    let supervisor = DesktopSupervisor::new();
    let env_id = unique_env_id("orphan");
    let env = supervisor
        .create_environment(test_config(&env_id), &unique_env_id("req"))
        .expect("create environment");

    // A live environment is left alone.
    let report = supervisor.reconcile();
    assert!(env.runtime_dir().exists());
    assert!(
        !report
            .stopped_units
            .iter()
            .any(|unit| unit.contains(&env_id.replace('-', "_")) || unit.contains(&env_id))
    );

    // An environment whose owning supervisor is gone is reclaimed: its scope is
    // stopped and its runtime directory removed.
    let orphan_env = format!("orphan-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let orphan = format!("juto-desktop-{orphan_env}-app.scope");
    let mut orphan_child = std::process::Command::new("systemd-run")
        .args([
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            &format!("--unit={orphan}"),
            "--",
            "sleep",
            "60",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn the orphan scope");
    assert!(
        wait_for_unit_active(&orphan, Duration::from_secs(5)),
        "orphan scope never became active"
    );

    let stale = env.runtime_dir().parent().unwrap().join(&orphan_env);
    std::fs::create_dir_all(&stale).unwrap();
    std::fs::write(stale.join("owner"), "999999999\n").unwrap();

    let report = supervisor.reconcile();
    assert!(
        report.stopped_units.iter().any(|unit| unit == &orphan),
        "orphan scope was not stopped: {:?}",
        report.stopped_units
    );
    assert!(!stale.exists(), "stale directory was not removed");
    assert!(env.runtime_dir().exists(), "live environment was removed");

    let _ = orphan_child.kill();
    let _ = orphan_child.wait();
    env.destroy();
}

#[test]
fn text_input_v3_receives_committed_text() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(test_config(&unique_env_id("text")), &unique_env_id("req"))
        .expect("create environment");

    let text_file = std::env::temp_dir().join(format!("juto-text-{}.txt", uuid::Uuid::now_v7()));
    let socket = env.socket_path().to_string_lossy().to_string();
    let text_path = text_file.to_string_lossy().to_string();
    let client = std::thread::spawn(move || {
        // The client writes committed text to this file.
        unsafe { std::env::set_var("JUTO_TEST_CLIENT_TEXT_FILE", &text_path) };
        common::run(&socket, 8)
    });

    // Wait until the client's window is mapped and focused.
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut focused = false;
    while Instant::now() < deadline {
        if let Ok(info) = env.observe(Duration::from_secs(2)) {
            let pool = env.frame_pool();
            pool.acquire_slot(info.slot);
            let read = pool.read_slot(info.slot, pool.slot_size());
            pool.release_slot(info.slot);
            if let Ok((header, pixels)) = read
                && header.width > 0
                && pixels.len() >= 4
                && pixels[0..4] == common::RED
            {
                focused = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(focused, "the client never rendered");

    let lease = env
        .set_control("agent:test", "acquire", false)
        .unwrap()
        .unwrap();
    let batch = InputBatch {
        seq: 1,
        client_timestamp_us: 0,
        events: vec![InputEvent::TextCommit {
            text: "hello".to_string(),
            focus_epoch: 1,
        }],
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut received = String::new();
    while Instant::now() < deadline {
        let _ = env.dispatch_input("agent:test", lease.epoch, 1, batch.clone());
        if let Ok(text) = std::fs::read_to_string(&text_file) {
            received = text;
            if received.contains("hello") {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(
        received.contains("hello"),
        "client did not receive the committed text: {received:?}"
    );

    let _ = client.join();
    let _ = std::fs::remove_file(&text_file);
    env.destroy();
}

#[test]
fn virtual_clock_paces_without_catch_up_bursts() {
    for fps in [15u32, 30, 60] {
        let supervisor = DesktopSupervisor::new();
        let mut config = test_config(&unique_env_id(&format!("pace{fps}")));
        config.output.fps = fps;
        let env = supervisor
            .create_environment(config, &unique_env_id("req"))
            .expect("create environment");

        // Sample the virtual clock every 100 ms for a second, measuring the
        // real elapsed time so the rate is comparable to the cap.
        let mut samples = Vec::new();
        let started = Instant::now();
        for _ in 0..11 {
            samples.push(env.observe(Duration::from_secs(2)).expect("observe").ticks);
            std::thread::sleep(Duration::from_millis(100));
        }
        let elapsed = started.elapsed().as_secs_f64();
        let deltas: Vec<u64> = samples.windows(2).map(|w| w[1] - w[0]).collect();
        let total: u64 = deltas.iter().sum();
        let max = *deltas.iter().max().unwrap();
        let rate = total as f64 / elapsed;

        // The clock must not run fast, and must not burst to catch up.
        assert!(
            rate <= fps as f64 * 1.25,
            "fps {fps}: measured {rate:.1} ticks/s over {elapsed:.2}s"
        );
        assert!(
            rate >= fps as f64 * 0.5,
            "fps {fps}: measured only {rate:.1} ticks/s over {elapsed:.2}s"
        );
        assert!(
            max <= (fps as u64 / 10) + 2,
            "fps {fps}: burst of {max} ticks in 100ms"
        );

        env.destroy();
    }
}

#[test]
fn environment_slice_aggregates_limits_and_contains_apps() {
    let supervisor = DesktopSupervisor::new();
    let env_id = unique_env_id("slice");
    let env = supervisor
        .create_environment(test_config(&env_id), &unique_env_id("req"))
        .expect("create environment");

    let slice = env.slice().expect("environment slice").to_string();
    assert!(slice.ends_with(".slice"), "{slice}");
    // systemd treats `-` as a path separator, so the slice name must not contain
    // the environment id's dashes.
    let suffix = slice
        .trim_start_matches("juto-desktop-")
        .trim_end_matches(".slice");
    assert!(
        !suffix.contains('-'),
        "slice name must escape dashes: {slice}"
    );
    assert!(unit_is_active(&slice), "slice must be active");

    let properties = systemctl_show(&slice, &["MemoryMax", "TasksMax", "CPUQuotaPerSecUSec"]);
    assert!(
        properties.contains("MemoryMax=8589934592"),
        "slice limits: {properties}"
    );
    assert!(
        properties.contains("TasksMax=512"),
        "slice limits: {properties}"
    );
    assert!(
        !properties.contains("CPUQuotaPerSecUSec=infinity"),
        "slice limits: {properties}"
    );

    // Applications run inside the slice, not beside it.
    let handle = env
        .launch(
            "sleeper",
            &["sleep".to_string(), "60".to_string()],
            None,
            &[],
            &unique_env_id("req"),
        )
        .expect("launch");
    let unit = handle.unit.clone().expect("scope unit");
    // The scope's cgroup path is populated asynchronously.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut scope = String::new();
    while Instant::now() < deadline {
        scope = systemctl_show(&unit, &["ControlGroup"]);
        if scope.contains(&slice) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        scope.contains(&slice),
        "scope {unit} is not inside {slice}: {scope}"
    );

    env.destroy();
    assert!(!unit_is_active(&slice), "slice must be stopped on destroy");
}

#[test]
fn memory_pressure_emits_limit_pressure() {
    let supervisor = DesktopSupervisor::new();
    let env_id = unique_env_id("pressure");
    let mut config = test_config(&env_id);
    config.limits.memory_max = "256M".to_string();
    let env = supervisor
        .create_environment(config, &unique_env_id("req"))
        .expect("create environment");
    let mut events = env.subscribe();

    // The environment slice already holds the compositor worker and Xwayland,
    // so size the hog from the slice's current usage: above the 90% threshold
    // but below the limit, so the process is not OOM-killed before the monitor
    // samples.
    let limit: u64 = 256 * 1024 * 1024;
    let slice = format!("juto-desktop-{}.slice", env_id.replace('-', "_"));
    let baseline = systemctl_show(&slice, &["MemoryCurrent"])
        .lines()
        .find_map(|line| line.strip_prefix("MemoryCurrent="))
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let hog_bytes = ((limit as f64 * 0.95) as u64)
        .saturating_sub(baseline)
        .max(32 * 1024 * 1024);

    env.launch(
        "hog",
        &[
            "perl".to_string(),
            "-e".to_string(),
            format!("my $x = 'a' x ({hog_bytes}); sleep 20;"),
        ],
        None,
        &[],
        &unique_env_id("req"),
    )
    .expect("launch memory hog");

    let deadline = Instant::now() + Duration::from_secs(20);
    let mut observed = false;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(DesktopEvent::LimitPressure { metric, .. }) if metric == "memory" => {
                observed = true;
                break;
            }
            Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }
    assert!(observed, "no memory LimitPressure event was emitted");

    env.destroy();
}

#[test]
fn input_requires_a_geometry_acknowledgement_after_resize() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(
            test_config(&unique_env_id("geometry")),
            &unique_env_id("req"),
        )
        .expect("create environment");
    let lease = env
        .set_control("agent:test", "acquire", false)
        .unwrap()
        .unwrap();
    let batch = InputBatch {
        seq: 1,
        client_timestamp_us: 0,
        events: vec![InputEvent::PointerMoveAbs { x: 1, y: 1 }],
    };

    // The initial generation accepts input without an acknowledgement.
    assert!(
        env.dispatch_input("agent:test", lease.epoch, 1, batch.clone())
            .is_ok()
    );

    // A resize invalidates the acknowledgement.
    let new_gen = env
        .configure_output(
            1,
            OutputConfig {
                width: 800,
                height: 600,
                scale: 1,
                fps: 30,
            },
        )
        .expect("resize");
    assert_eq!(new_gen, 2);
    assert!(env.geometry_ready_viewers().is_empty());
    assert!(matches!(
        env.dispatch_input("agent:test", lease.epoch, 2, batch.clone()),
        Err(juto_desktop::DesktopError::StaleGeneration { .. })
    ));

    // A stale acknowledgement is rejected; the current one resumes input.
    assert!(env.acknowledge_geometry("viewer-1", 1).is_err());
    env.acknowledge_geometry("viewer-1", 2)
        .expect("acknowledge the current generation");
    assert_eq!(env.geometry_ready_viewers(), vec!["viewer-1".to_string()]);
    assert!(
        env.dispatch_input("agent:test", lease.epoch, 2, batch.clone())
            .is_ok()
    );

    // A further resize gates input again.
    env.configure_output(
        2,
        OutputConfig {
            width: 640,
            height: 480,
            scale: 1,
            fps: 30,
        },
    )
    .expect("resize");
    assert!(matches!(
        env.dispatch_input("agent:test", lease.epoch, 3, batch),
        Err(juto_desktop::DesktopError::StaleGeneration { .. })
    ));

    env.destroy();
}

#[test]
fn compositor_worker_runs_inside_the_environment_slice() {
    let supervisor = DesktopSupervisor::new();
    let env_id = unique_env_id("worker");
    let env = supervisor
        .create_environment(test_config(&env_id), &unique_env_id("req"))
        .expect("create environment");
    let slice = env.slice().expect("environment slice").to_string();

    // The compositor runs in its own scope inside the slice, so its memory is
    // charged to the environment.
    let unit = format!("juto-desktop-{env_id}-compositor.scope");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut scope = String::new();
    while Instant::now() < deadline {
        scope = systemctl_show(&unit, &["ControlGroup"]);
        if scope.contains(&slice) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        scope.contains(&slice),
        "compositor scope {unit} is not inside {slice}: {scope}"
    );

    // The compositor is a separate process, not a supervisor thread.
    let cgroup = scope
        .trim()
        .strip_prefix("ControlGroup=")
        .expect("ControlGroup")
        .to_string();
    let procs = std::fs::read_to_string(format!("/sys/fs/cgroup{cgroup}/cgroup.procs"))
        .expect("read the compositor cgroup");
    let pids: Vec<i32> = procs
        .lines()
        .filter_map(|line| line.trim().parse().ok())
        .collect();
    assert!(!pids.is_empty(), "the compositor cgroup has no processes");
    assert!(
        !pids.contains(&(std::process::id() as i32)),
        "the compositor must not be the test process: {pids:?}"
    );

    // Frames still flow through the worker's pool.
    let info = env.observe(Duration::from_secs(5)).expect("observe");
    assert_eq!(info.width, 640);
    assert!(info.ticks > 0);

    env.destroy();
    assert!(
        !unit_is_active(&unit),
        "compositor scope must stop on destroy"
    );
}

#[test]
fn owner_heartbeat_expiry_destroys_an_abandoned_environment() {
    let supervisor = DesktopSupervisor::new();
    let mut config = test_config(&unique_env_id("abandon"));
    config.owner_heartbeat_timeout_secs = 2;
    let env = supervisor
        .create_environment(config, &unique_env_id("req"))
        .expect("create environment");
    let mut events = env.subscribe();
    let runtime_dir = env.runtime_dir().to_path_buf();

    // No heartbeat arrives, so the reaper retires the environment.
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut failed = false;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(DesktopEvent::Failed { reason, .. }) => {
                assert!(reason.contains("heartbeat"), "{reason}");
                failed = true;
                break;
            }
            Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }
    assert!(failed, "no heartbeat-expiry failure was emitted");
    assert!(env.is_destroyed());
    // Teardown follows the failure event.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && runtime_dir.exists() {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!runtime_dir.exists(), "runtime directory must be removed");
}

#[test]
fn heartbeat_renews_and_retention_keeps_an_environment_alive() {
    let supervisor = DesktopSupervisor::new();
    let mut config = test_config(&unique_env_id("retain"));
    config.owner_heartbeat_timeout_secs = 2;
    let env = supervisor
        .create_environment(config, &unique_env_id("req"))
        .expect("create environment");

    // The owner heartbeat renews the deadline.
    let deadline = env.heartbeat("owner", None).expect("owner heartbeat");
    assert!(deadline > 0);

    // The lease heartbeat must match the current epoch.
    let lease = env
        .set_control("agent:test", "acquire", false)
        .unwrap()
        .unwrap();
    assert!(env.heartbeat("lease", Some(lease.epoch)).is_ok());
    assert!(matches!(
        env.heartbeat("lease", Some(lease.epoch + 7)),
        Err(juto_desktop::DesktopError::StaleLease { .. })
    ));

    // Retention outlives the heartbeat timeout.
    let retention = env.retain("human:test", 60).expect("retain");
    assert_eq!(retention.human_owner, "human:test");
    std::thread::sleep(Duration::from_millis(3500));
    assert!(
        !env.is_destroyed(),
        "retention must keep the environment alive"
    );
    assert!(env.retention().is_some());

    // Retention is host-bounded.
    let bounded = env.retain("human:test", 999_999).expect("retain");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(bounded.expires_at <= now + 1801, "{}", bounded.expires_at);

    env.destroy();
}

#[test]
fn launch_rejects_missing_executable() {
    let supervisor = DesktopSupervisor::new();
    let env = supervisor
        .create_environment(
            test_config(&unique_env_id("missing")),
            &unique_env_id("req"),
        )
        .expect("create environment");

    let err = env
        .launch(
            "nope",
            &["definitely-not-a-real-binary-xyz".to_string()],
            None,
            &[],
            &unique_env_id("req"),
        )
        .expect_err("missing executable must fail");
    assert!(matches!(
        err,
        juto_desktop::DesktopError::ExecutableUnavailable { .. }
    ));

    env.destroy();
}

fn read_pixel(pool: &ShmReader, slot: usize, x: u32, y: u32, width: u32) -> [u8; 4] {
    let (_header, pixels) = pool.read_slot(slot, pool.slot_size()).expect("read slot");
    let idx = ((y * width + x) * 4) as usize;
    [
        pixels[idx],
        pixels[idx + 1],
        pixels[idx + 2],
        pixels[idx + 3],
    ]
}

fn systemctl_show(unit: &str, properties: &[&str]) -> String {
    let mut args = vec!["--user", "show", unit, "--no-pager"];
    let flags: Vec<String> = properties
        .iter()
        .map(|p| format!("--property={p}"))
        .collect();
    args.extend(flags.iter().map(|f| f.as_str()));
    std::process::Command::new("systemctl")
        .args(&args)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

fn unit_is_active(unit: &str) -> bool {
    std::process::Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", unit])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn wait_for_unit_active(unit: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if unit_is_active(unit) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[allow(dead_code)]
fn workspace_dir() -> PathBuf {
    std::env::temp_dir()
}
