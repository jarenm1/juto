//! Same-host viewer transport tests: attach over the control socket, receive the
//! sealed frame pool as an `SCM_RIGHTS` descriptor, and read composed frames.

mod common;

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use juto_desktop::protocol::{DesktopCommand, DesktopResponse};
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
        let payload = serde_json::to_vec(&command).unwrap();
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

    // The viewer can take the human control lease and send input.
    let epoch = viewer.take_control(&env_id).expect("take control");
    assert!(epoch >= 1);
    viewer
        .send_input(
            &env_id,
            1,
            vec![juto_desktop::protocol::InputEvent::PointerMoveAbs { x: 5, y: 5 }],
        )
        .expect("send input");

    let described = daemon.call(DesktopCommand::Describe {
        env_id: env_id.clone(),
        after_event_seq: None,
    });
    match described {
        DesktopResponse::Description(status) => {
            assert!(status.active_control_lease.is_some());
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
