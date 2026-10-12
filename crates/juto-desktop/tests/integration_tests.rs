use juto_desktop::protocol::*;
use juto_desktop::supervisor::{
    DesktopSupervisor, EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits,
};

#[test]
fn test_environment_lifecycle_admission_and_generation() {
    let supervisor = DesktopSupervisor::new();
    let config = EnvironmentConfig {
        env_id: "test-env-1".to_string(),
        owner_principal: "agent-1".to_string(),
        output: OutputConfig {
            width: 1280,
            height: 720,
            scale: 1,
            fps: 30,
        },
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
    };

    let env = supervisor
        .create_environment(config)
        .expect("create environment");
    assert_eq!(env.env_id(), "test-env-1");
    assert_eq!(env.generation(), 1);

    let status = env.get_status();
    assert!(status.is_ready);
    assert_eq!(status.output.width, 1280);

    // Admission check: 4K @ 60 FPS in Software profile exceeds admitted 62.2M px/s
    let excessive_config = OutputConfig {
        width: 3840,
        height: 2160,
        scale: 1,
        fps: 60,
    };
    let reconfig_err = env.configure_output(1, excessive_config);
    assert!(reconfig_err.is_err(), "should reject excessive pixel rate");

    // Valid resize increments generation
    let valid_reconfig = OutputConfig {
        width: 1920,
        height: 1080,
        scale: 1,
        fps: 30,
    };
    let new_gen = env
        .configure_output(1, valid_reconfig)
        .expect("valid resize");
    assert_eq!(new_gen, 2);
    assert_eq!(env.generation(), 2);
    assert_eq!(env.output().width, 1920);
}

#[test]
fn test_shm_frame_producer_and_consumer_sealing() {
    let supervisor = DesktopSupervisor::new();
    let config = EnvironmentConfig {
        env_id: "test-env-shm".to_string(),
        owner_principal: "agent-2".to_string(),
        output: OutputConfig {
            width: 320,
            height: 240,
            scale: 1,
            fps: 30,
        },
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
    };

    let env = supervisor
        .create_environment(config)
        .expect("create environment");

    // Simulate game frame generation
    let fake_pixels = vec![0xABu8; 320 * 240 * 4];
    let obs = env.produce_frame(&fake_pixels).expect("produce frame");

    if let DesktopResponse::Observation {
        frame_id,
        generation,
        shm_slot,
        ..
    } = obs
    {
        assert_eq!(frame_id, 1);
        assert_eq!(generation, 1);
        assert_eq!(shm_slot, 1); // first advance from slot 0
    } else {
        panic!("unexpected response variant");
    }
}

#[test]
fn test_control_lease_and_human_preemption() {
    let supervisor = DesktopSupervisor::new();
    let config = EnvironmentConfig {
        env_id: "test-env-lease".to_string(),
        owner_principal: "agent-builder".to_string(),
        output: OutputConfig::default(),
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
    };

    let env = supervisor
        .create_environment(config)
        .expect("create environment");

    // Agent acquires initial control
    let agent_lease = env
        .set_control("agent-builder", "acquire", false)
        .expect("agent acquire")
        .expect("some lease");
    assert_eq!(agent_lease.holder, "agent-builder");
    assert_eq!(agent_lease.epoch, 1);
    assert!(!agent_lease.is_human);

    // Another agent cannot steal control
    let busy_err = env.set_control("agent-rogue", "acquire", false);
    assert!(busy_err.is_err());

    // A human CAN preempt the agent
    let human_lease = env
        .set_control("human-user", "acquire", true)
        .expect("human acquire")
        .expect("some lease");
    assert_eq!(human_lease.holder, "human-user");
    assert_eq!(human_lease.epoch, 2);
    assert!(human_lease.is_human);

    // Now an agent cannot preempt the human
    let agent_blocked = env.set_control("agent-builder", "acquire", false);
    assert!(agent_blocked.is_err());

    // Human releases control
    let released = env
        .set_control("human-user", "release", true)
        .expect("release");
    assert!(released.is_none());
    assert!(env.get_status().active_control_lease.is_none());
}

#[test]
fn test_input_dispatch_validation() {
    let supervisor = DesktopSupervisor::new();
    let config = EnvironmentConfig {
        env_id: "test-env-input".to_string(),
        owner_principal: "agent-1".to_string(),
        output: OutputConfig::default(),
        render_profile: RenderProfile::Software,
        limits: SystemLimits::default(),
        workspace_dir: std::env::temp_dir(),
    };

    let env = supervisor
        .create_environment(config)
        .expect("create environment");
    let lease = env
        .set_control("controller-1", "acquire", true)
        .expect("acquire")
        .unwrap();

    let batch = InputBatch {
        seq: 42,
        client_timestamp_us: 1000,
        events: vec![
            InputEvent::PointerMoveAbs { x: 100, y: 200 },
            InputEvent::PointerButton {
                button: 1,
                pressed: true,
            },
        ],
    };

    // Stale generation is rejected
    let bad_gen = env.dispatch_input("controller-1", lease.epoch, 999, batch.clone());
    assert!(bad_gen.is_err());

    // Stale lease epoch is rejected
    let bad_epoch = env.dispatch_input("controller-1", 999, env.generation(), batch.clone());
    assert!(bad_epoch.is_err());

    // Valid dispatch succeeds
    let ack = env
        .dispatch_input("controller-1", lease.epoch, env.generation(), batch)
        .expect("dispatch input");
    assert_eq!(ack, 42);
}
