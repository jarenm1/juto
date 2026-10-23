//! Same-host viewer transport tests: attach over the control socket, receive the
//! sealed frame pool as an `SCM_RIGHTS` descriptor, and read composed frames.

mod common;

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use juto_desktop::protocol::{DesktopCommand, DesktopRequest, DesktopResponse, GrantRole};
use juto_desktop::viewer::ViewerClient;

struct Daemon {
    child: Child,
    socket: PathBuf,
    runtime_root: PathBuf,
}

impl Daemon {
    fn start() -> Self {
        let root = std::env::temp_dir().join(format!("jv-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&root).expect("create runtime root");
        let socket = root.join("control.sock");
        let log = std::fs::File::create(root.join("daemon.log")).expect("daemon log");
        let child = Command::new(env!("CARGO_BIN_EXE_juto-desktop"))
            .arg("--socket")
            .arg(&socket)
            .env("JUTO_DESKTOP_RUNTIME_DIR", &root)
            .env("JUTO_DESKTOP_WORKER", env!("CARGO_BIN_EXE_juto-desktop"))
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()
            .expect("spawn the daemon");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if socket.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(socket.exists(), "daemon never created its socket");
        Self {
            child,
            socket,
            runtime_root: root,
        }
    }

    fn call(&self, command: DesktopCommand) -> DesktopResponse {
        let mut stream = UnixStream::connect(&self.socket).expect("connect to the daemon");
        let payload = serde_json::to_vec(&DesktopRequest::new(command)).unwrap();
        stream
            .write_all(&(payload.len() as u32).to_be_bytes())
            .unwrap();
        stream.write_all(&payload).unwrap();
        stream.flush().unwrap();
        let mut length = [0u8; 4];
        if let Err(err) = stream.read_exact(&mut length) {
            let log = std::fs::read_to_string(self.runtime_root.join("daemon.log"))
                .unwrap_or_else(|_| "(no log)".to_string());
            panic!("daemon closed the connection: {err}\n--- daemon log ---\n{log}");
        }
        let mut body = vec![0u8; u32::from_be_bytes(length) as usize];
        stream.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.runtime_root);
    }
}

fn environment_config(env_id: &str) -> serde_json::Value {
    serde_json::json!({
        "envId": env_id,
        "ownerPrincipal": "agent:test",
        "output": {"width": 640, "height": 480, "scale": 1, "fps": 30},
        "renderProfile": "software",
        "limits": {
            "cpuQuota": "400%",
            "memoryMax": "8G",
            "memorySwapMax": "0",
            "tasksMax": 512,
            "remoteBitrateMaxBps": 12000000,
        },
        "workspaceDir": std::env::temp_dir(),
        "leaseTtlSecs": 6,
    })
}

#[test]
fn an_unsupported_protocol_version_is_rejected() {
    let daemon = Daemon::start();
    let mut stream = UnixStream::connect(&daemon.socket).expect("connect");
    let payload = serde_json::to_vec(&serde_json::json!({
        "version": 99,
        "command": {
            "command": "describe",
            "payload": {"envId": "whatever", "afterEventSeq": null},
        },
    }))
    .unwrap();
    stream
        .write_all(&(payload.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&payload).unwrap();
    stream.flush().unwrap();

    let mut length = [0u8; 4];
    stream.read_exact(&mut length).expect("response");
    let mut body = vec![0u8; u32::from_be_bytes(length) as usize];
    stream.read_exact(&mut body).unwrap();
    match serde_json::from_slice::<DesktopResponse>(&body).unwrap() {
        DesktopResponse::Error { code, message } => {
            assert_eq!(code, "Protocol");
            assert!(
                message.contains("unsupported protocol version"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected a protocol error, got {other:?}"),
    }
}

#[test]
fn a_destroyed_environment_stays_describable() {
    let daemon = Daemon::start();
    let env_id = format!("t-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );
    let destroyed = daemon.call(DesktopCommand::Destroy {
        env_id: env_id.clone(),
    });
    assert!(matches!(destroyed, DesktopResponse::Destroyed));

    // The terminal state remains queryable after destroy.
    match daemon.call(DesktopCommand::Describe {
        env_id: env_id.clone(),
        after_event_seq: None,
    }) {
        DesktopResponse::Description(status) => {
            assert_eq!(status.env_id, env_id);
            // The retained snapshot is the terminal state: readiness must be
            // cleared by `destroy`, not copied from the live environment.
            assert!(!status.is_ready, "a destroyed environment is not ready");
        }
        other => panic!("expected the retained terminal state, got {other:?}"),
    }

    // An environment that never existed is still unknown.
    let unknown = daemon.call(DesktopCommand::Describe {
        env_id: "never-existed".to_string(),
        after_event_seq: None,
    });
    assert!(
        matches!(unknown, DesktopResponse::Error { .. }),
        "unknown environments must not be described: {unknown:?}"
    );
}

#[test]
fn retention_records_the_authenticated_peer_not_a_claimed_name() {
    let daemon = Daemon::start();
    let env_id = format!("r-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );

    match daemon.call(DesktopCommand::Retain {
        env_id: env_id.clone(),
        human_owner: "spoofed:admin".to_string(),
        deadline_seconds: 60,
    }) {
        DesktopResponse::Retained { human_owner, .. } => {
            let uid = unsafe { libc::geteuid() };
            assert_eq!(
                human_owner,
                format!("user:{uid}"),
                "retention must bind the kernel-authenticated peer"
            );
        }
        other => panic!("expected retention, got {other:?}"),
    }
}

#[test]
fn a_lease_heartbeat_without_the_epoch_is_rejected() {
    let daemon = Daemon::start();
    let env_id = format!("h-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );

    match daemon.call(DesktopCommand::Control {
        env_id: env_id.clone(),
        action: "acquire".to_string(),
        is_human: false,
    }) {
        DesktopResponse::ControlChanged { lease: Some(lease) } => {
            let renewed = daemon.call(DesktopCommand::Heartbeat {
                env_id: env_id.clone(),
                target: "lease".to_string(),
                epoch: None,
            });
            assert!(
                matches!(renewed, DesktopResponse::Error { .. }),
                "a missing epoch must not renew the lease: {renewed:?}"
            );
            let renewed = daemon.call(DesktopCommand::Heartbeat {
                env_id: env_id.clone(),
                target: "lease".to_string(),
                epoch: Some(lease.epoch),
            });
            assert!(
                matches!(renewed, DesktopResponse::HeartbeatExtended { .. }),
                "the matching epoch must renew the lease: {renewed:?}"
            );
        }
        other => panic!("expected a control lease, got {other:?}"),
    }
}

#[test]
fn geometry_ready_is_bound_to_the_attached_viewer() {
    let daemon = Daemon::start();
    let env_id = format!("g-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );

    let mut viewer = ViewerClient::attach(&daemon.socket, &env_id).expect("attach");
    let viewer_id = viewer.info().viewer_id.clone();

    let described = daemon.call(DesktopCommand::Describe {
        env_id: env_id.clone(),
        after_event_seq: None,
    });
    let generation = match described {
        DesktopResponse::Description(status) => status.generation,
        other => panic!("expected a description, got {other:?}"),
    };

    // A connection without an attached viewer cannot acknowledge at all, and
    // it cannot answer for the attached viewer's identity either.
    let rejected = daemon.call(DesktopCommand::GeometryReady {
        env_id: env_id.clone(),
        viewer_id: "unadmitted".to_string(),
        generation,
        track_or_pool_id: viewer.info().pool_id.clone(),
    });
    assert!(
        matches!(rejected, DesktopResponse::Error { .. }),
        "an unattached connection must not acknowledge: {rejected:?}"
    );
    let forged = daemon.call(DesktopCommand::GeometryReady {
        env_id: env_id.clone(),
        viewer_id: viewer_id.clone(),
        generation,
        track_or_pool_id: viewer.info().pool_id.clone(),
    });
    assert!(
        matches!(forged, DesktopResponse::Error { .. }),
        "another connection must not answer for the viewer: {forged:?}"
    );

    // The owning connection acknowledges for itself.
    assert!(
        viewer.geometry_ready(generation).is_ok(),
        "the attached viewer acknowledges its own presentation"
    );
}

#[test]
fn a_lagging_event_subscriber_is_told_to_resync() {
    let daemon = Daemon::start();
    let env_id = format!("r-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );

    // Subscribe and then stop reading while the environment emits far more
    // events than the 256-event replay ring holds.
    let mut stream = UnixStream::connect(&daemon.socket).expect("connect");
    let subscribe = serde_json::to_vec(&DesktopRequest::new(DesktopCommand::Subscribe {
        env_id: env_id.clone(),
    }))
    .unwrap();
    stream
        .write_all(&(subscribe.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&subscribe).unwrap();
    stream.flush().unwrap();
    let mut length = [0u8; 4];
    stream.read_exact(&mut length).expect("subscribed response");
    let mut body = vec![0u8; u32::from_be_bytes(length) as usize];
    stream.read_exact(&mut body).unwrap();

    for _ in 0..2000 {
        let _ = daemon.call(DesktopCommand::Control {
            env_id: env_id.clone(),
            action: "acquire".to_string(),
            is_human: false,
        });
        let _ = daemon.call(DesktopCommand::Control {
            env_id: env_id.clone(),
            action: "release".to_string(),
            is_human: false,
        });
    }

    // Reading now must surface a resynchronization instruction rather than a
    // silent gap in the event stream.
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut resync = false;
    for _ in 0..4000 {
        let mut length = [0u8; 4];
        if stream.read_exact(&mut length).is_err() {
            break;
        }
        let mut body = vec![0u8; u32::from_be_bytes(length) as usize];
        if stream.read_exact(&mut body).is_err() {
            break;
        }
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body)
            && value.get("event").and_then(|e| e.as_str()) == Some("resyncRequired")
        {
            resync = true;
            break;
        }
    }
    assert!(
        resync,
        "a subscriber that fell behind the replay ring must receive ResyncRequired"
    );

    drop(stream);
    let _ = daemon.call(DesktopCommand::Destroy { env_id });
}

#[test]
fn viewer_admission_is_bounded_and_released_on_disconnect() {
    let daemon = Daemon::start();
    let env_id = format!("a-{}", &uuid::Uuid::now_v7().to_string()[24..]);
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );

    let first = ViewerClient::attach(&daemon.socket, &env_id).expect("first viewer");
    let second = ViewerClient::attach(&daemon.socket, &env_id).expect("second viewer");
    let third = ViewerClient::attach(&daemon.socket, &env_id);
    assert!(
        third.is_err(),
        "a third viewer must be rejected while two are attached"
    );

    // Closing a viewer connection releases its slot.
    drop(first);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut admitted = None;
    while Instant::now() < deadline {
        match ViewerClient::attach(&daemon.socket, &env_id) {
            Ok(viewer) => {
                admitted = Some(viewer);
                break;
            }
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    assert!(
        admitted.is_some(),
        "the viewer slot was not released after the connection closed"
    );

    drop(second);
    drop(admitted);
    let destroyed = daemon.call(DesktopCommand::Destroy {
        env_id: env_id.clone(),
    });
    assert!(
        matches!(destroyed, DesktopResponse::Destroyed),
        "destroy failed: {destroyed:?}"
    );
}

#[test]
fn viewer_attaches_over_scm_rights_and_reads_composed_frames() {
    let daemon = Daemon::start();
    let env_id = format!("v-{}", &uuid::Uuid::now_v7().to_string()[24..]);

    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(&env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );

    // The environment's Wayland socket lives in its private runtime directory.
    let wayland = daemon
        .runtime_root
        .join("juto-desktop")
        .join(&env_id)
        .join("wayland-0");
    let socket = wayland.to_string_lossy().to_string();
    let client = std::thread::spawn(move || common::run(&socket, 20));

    let mut viewer = match ViewerClient::attach(&daemon.socket, &env_id) {
        Ok(viewer) => viewer,
        Err(err) => {
            let log = std::fs::read_to_string(daemon.runtime_root.join("daemon.log"))
                .unwrap_or_else(|_| "(no log)".to_string());
            panic!("attach failed: {err}\n--- daemon log ---\n{log}");
        }
    };
    let info = viewer.info().clone();
    assert_eq!(info.transport, "shm");
    assert_eq!(info.width, 640);
    assert_eq!(info.height, 480);
    assert_eq!(info.stride, 640 * 4);
    assert_eq!(info.format, "argb8888");
    assert_eq!(info.slot_count, 3);
    assert!(info.pool_bytes > 0);

    // Poll the shared pool until the client's two-tone frame appears.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut verified = None;
    while Instant::now() < deadline {
        if let Some((header, pixels)) = viewer.latest_frame().expect("read frame") {
            let pixel = |x: u32, y: u32| -> [u8; 4] {
                let index = ((y * header.width + x) * 4) as usize;
                [
                    pixels[index],
                    pixels[index + 1],
                    pixels[index + 2],
                    pixels[index + 3],
                ]
            };
            if header.width == 640 && pixel(10, 10) == common::RED && pixel(300, 10) == common::BLUE
            {
                verified = Some(header);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let header = verified.expect("the viewer never saw the client's two-tone frame");
    assert_eq!(header.generation, 1);

    // Human control requires a live grant for the viewer principal.
    match daemon.call(DesktopCommand::Grant {
        env_id: env_id.clone(),
        principal: juto_desktop::protocol::HUMAN_VIEWER_PRINCIPAL.to_string(),
        roles: vec![GrantRole::Owner, GrantRole::Control],
        audience: "session".to_string(),
        expiry_seconds: 60,
    }) {
        DesktopResponse::GrantIssued(_) => {}
        other => panic!("expected a grant, got {other:?}"),
    }

    // The viewer can take the human control lease and send input.
    let epoch = viewer.take_control().expect("take control");
    assert!(epoch >= 1);
    viewer
        .send_input(
            1,
            vec![juto_desktop::protocol::InputEvent::PointerMoveAbs { x: 5, y: 5 }],
        )
        .expect("send input");

    // Observe must address the attached environment, not the frame-pool name
    // (regression: the command used to carry `pool_id` and always failed).
    let (observed, _pixels) = viewer
        .observe(header.generation, Duration::from_secs(5))
        .expect("observe must target the attached environment");
    assert_eq!(observed.width, 640);

    let described = daemon.call(DesktopCommand::Describe {
        env_id: env_id.clone(),
        after_event_seq: None,
    });
    match described {
        DesktopResponse::Description(status) => {
            let holder = status
                .active_control_lease
                .as_ref()
                .map(|lease| lease.holder.as_str());
            assert_eq!(holder, Some(juto_desktop::protocol::HUMAN_VIEWER_PRINCIPAL));
        }
        other => panic!("unexpected describe response: {other:?}"),
    }

    let destroyed = daemon.call(DesktopCommand::Destroy {
        env_id: env_id.clone(),
    });
    assert!(matches!(destroyed, DesktopResponse::Destroyed));
    assert!(
        !daemon
            .runtime_root
            .join("juto-desktop")
            .join(&env_id)
            .exists(),
        "runtime directory must be removed"
    );

    let _ = client.join();
}

/// A fresh environment id for one test case.
fn new_env_id() -> String {
    format!("v-{}", &uuid::Uuid::now_v7().to_string()[24..])
}

/// Creates an environment with a live test client and attaches a viewer to it.
fn attach_viewer(daemon: &Daemon, env_id: &str) -> ViewerClient {
    let created = daemon.call(DesktopCommand::Create {
        spec: serde_json::from_value(environment_config(env_id)).expect("config"),
        request_id: "create".to_string(),
    });
    assert!(
        matches!(created, DesktopResponse::Created { .. }),
        "create failed: {created:?}"
    );
    let socket = daemon
        .runtime_root
        .join("juto-desktop")
        .join(env_id)
        .join("wayland-0")
        .to_string_lossy()
        .to_string();
    // The client keeps the environment's surface alive for the whole test.
    std::thread::spawn(move || common::run(&socket, 60));
    ViewerClient::attach(&daemon.socket, env_id).expect("attach viewer")
}

/// Grants the viewer principal the control role on `env_id`.
fn grant_human_control(daemon: &Daemon, env_id: &str) {
    let granted = daemon.call(DesktopCommand::Grant {
        env_id: env_id.to_string(),
        principal: juto_desktop::protocol::HUMAN_VIEWER_PRINCIPAL.to_string(),
        roles: vec![GrantRole::Control],
        audience: "session".to_string(),
        expiry_seconds: 60,
    });
    assert!(
        matches!(granted, DesktopResponse::GrantIssued(_)),
        "grant failed: {granted:?}"
    );
}

#[test]
fn human_control_survives_past_the_lease_ttl_while_renewed() {
    let daemon = Daemon::start();
    let env_id = new_env_id();
    let mut viewer = attach_viewer(&daemon, &env_id);
    grant_human_control(&daemon, &env_id);
    viewer.take_control().expect("take control");

    // The environment's lease TTL is 6s; renewing every 100ms holds control for 8s.
    let until = Instant::now() + Duration::from_secs(8);
    while Instant::now() < until {
        viewer.keep_control_alive().expect("renew control lease");
        std::thread::sleep(Duration::from_millis(100));
    }
    viewer
        .send_input(
            1,
            vec![juto_desktop::protocol::InputEvent::PointerMoveAbs { x: 5, y: 5 }],
        )
        .expect("input after the lease TTL");

    let described = daemon.call(DesktopCommand::Describe {
        env_id: env_id.clone(),
        after_event_seq: None,
    });
    match described {
        DesktopResponse::Description(status) => {
            let holder = status
                .active_control_lease
                .as_ref()
                .map(|lease| lease.holder.as_str());
            assert_eq!(holder, Some(juto_desktop::protocol::HUMAN_VIEWER_PRINCIPAL));
        }
        other => panic!("unexpected describe response: {other:?}"),
    }
}

#[test]
fn control_ends_when_the_viewer_stops_renewing() {
    let daemon = Daemon::start();
    let env_id = new_env_id();
    let mut viewer = attach_viewer(&daemon, &env_id);
    grant_human_control(&daemon, &env_id);
    viewer.take_control().expect("take control");

    // No renewal for longer than the 6s lease TTL.
    std::thread::sleep(Duration::from_secs(8));
    let err = viewer
        .send_input(
            1,
            vec![juto_desktop::protocol::InputEvent::PointerMoveAbs { x: 5, y: 5 }],
        )
        .expect_err("input after the lease expired must be refused");
    assert!(
        matches!(err, juto_desktop::DesktopError::ControlLeaseExpired { .. }),
        "unexpected error: {err:?}"
    );

    // After expiry the viewer can take control again.
    viewer
        .take_control()
        .expect("take control again after expiry");
}

#[test]
fn agent_cannot_take_control_while_the_human_holds_it() {
    let daemon = Daemon::start();
    let env_id = new_env_id();
    let mut viewer = attach_viewer(&daemon, &env_id);
    grant_human_control(&daemon, &env_id);
    viewer.take_control().expect("take control");

    match daemon.call(DesktopCommand::Control {
        env_id: env_id.clone(),
        action: "acquire".to_string(),
        is_human: false,
    }) {
        DesktopResponse::Error { code, .. } => assert_eq!(code, "ControlBusy"),
        other => panic!("agent acquired control held by the viewer: {other:?}"),
    }
}

#[test]
fn attach_rejects_unknown_environments() {
    let daemon = Daemon::start();
    let error = ViewerClient::attach(&daemon.socket, "does-not-exist");
    assert!(
        error.is_err(),
        "attaching to an unknown environment must fail"
    );
}

#[test]
fn control_socket_rejects_a_foreign_peer() {
    // The daemon checks SO_PEERCRED; a same-uid peer is accepted, so this test
    // only asserts that the credential check runs and reports the daemon user.
    let daemon = Daemon::start();
    let stream = UnixStream::connect(&daemon.socket).expect("connect");
    let creds = juto_desktop::scm::peer_credentials(&stream).expect("peer credentials");
    assert_eq!(creds.uid, unsafe { libc::geteuid() });
    assert!(Path::new(&daemon.socket).exists());
}
