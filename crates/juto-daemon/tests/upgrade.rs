//! Exercise an actual exec handover and client reconnect, in an isolated process
//! because daemon discovery uses environment variables.

use std::os::unix::fs::PermissionsExt;
use std::process::{Child, Command};
use std::time::Duration;

use juto_daemon::client::Client;
use juto_daemon::protocol::{
    ClientFrame, PROTOCOL_VERSION, Reply, Request, ServerFrame, binary_id, paths,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

struct Daemon(Child);
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn a_different_running_build_upgrades_itself_and_preserves_sessions() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    let runtime = root.path().join("run");
    let install = root.path().join("bin");
    for dir in [&workspace, &runtime, &install] {
        std::fs::create_dir(dir).unwrap();
    }
    let data = root.path().join("data");
    std::fs::create_dir(&data).unwrap();
    std::fs::write(
        data.join("config.yml"),
        r#"{
        "model": "upgrade/test",
        "customModels": [{
            "id": "test", "name": "Upgrade test", "provider": "upgrade",
            "api": "openai-completions", "baseUrl": "http://127.0.0.1:9"
        }]
    }"#,
    )
    .unwrap();
    let workspace = workspace.canonicalize().unwrap();
    let executable = std::path::Path::new(env!("CARGO_BIN_EXE_juto-daemon"));
    let replacement = install.join("juto-daemon");
    std::fs::copy(executable, &replacement).unwrap();

    // Give the running image a distinct GNU build ID without building a second
    // dependency tree. Only the ELF note changes; both versions speak our protocol.
    let previous = install.join("previous");
    let mut image = std::fs::read(executable).unwrap();
    let build_id = binary_id(executable).unwrap();
    let bytes: Vec<u8> = (0..build_id.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&build_id[offset..offset + 2], 16).unwrap())
        .collect();
    let offset = image
        .windows(bytes.len())
        .position(|window| window == bytes)
        .unwrap();
    image[offset] ^= 1;
    std::fs::write(&previous, image).unwrap();
    std::fs::set_permissions(&previous, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_ne!(binary_id(&previous).unwrap(), build_id);
    let path = std::env::var_os("PATH").unwrap();
    let mut paths_env = vec![install.clone()];
    paths_env.extend(std::env::split_paths(&path));
    // SAFETY: one test in this binary; configure discovery before starting tasks.
    unsafe {
        std::env::set_var("XDG_RUNTIME_DIR", &runtime);
        std::env::set_var("JUTO_DATA_DIR", root.path().join("data"));
        std::env::set_var("PATH", std::env::join_paths(paths_env).unwrap());
    }
    let mut daemon = Daemon(
        Command::new(&previous)
            .arg("--workspace")
            .arg(&workspace)
            .spawn()
            .unwrap(),
    );
    let socket = paths::daemon_socket(&workspace).unwrap();
    let stream = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            assert!(
                daemon.0.try_wait().unwrap().is_none(),
                "old daemon exited before handover"
            );
            if let Ok(stream) = UnixStream::connect(&socket).await {
                break stream;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let (read, mut write) = stream.into_split();
    let mut lines = BufReader::new(read).lines();
    let mut session_id = None;
    for (id, request) in [
        (
            1,
            Request::Hello {
                client: "upgrade-test".into(),
                protocol_version: PROTOCOL_VERSION,
            },
        ),
        (2, Request::CreateSession),
    ] {
        write
            .write_all(
                format!(
                    "{}\n",
                    serde_json::to_string(&ClientFrame { id, request }).unwrap()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        loop {
            let frame: ServerFrame =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            if let ServerFrame::Response {
                id: response_id,
                result,
            } = frame
            {
                assert_eq!(response_id, id);
                if let Reply::Session(view) = result.unwrap() {
                    session_id = Some(view.info.id);
                }
                break;
            }
        }
    }
    let session_id = session_id.unwrap();
    let client = Client::connect(&workspace).await.unwrap();
    let mut events = client.subscribe();
    tokio::time::timeout(Duration::from_secs(15), async {
        while client
            .hello()
            .is_none_or(|hello| hello.binary_id != build_id)
        {
            let _ = events.recv().await;
        }
    })
    .await
    .expect("client must reconnect to the installed build automatically");
    let hello = client.hello().unwrap();
    assert_eq!(
        hello.pid,
        daemon.0.id(),
        "exec must retain systemd's main PID"
    );
    let Reply::Session(view) = client
        .request(Request::OpenSession {
            session_id: session_id.clone(),
        })
        .await
        .unwrap()
    else {
        panic!("session must survive the upgrade");
    };
    assert_eq!(view.info.id, session_id);
    assert!(view.info.run.is_none());
    assert!(matches!(
        client.request(Request::CreateSession).await.unwrap(),
        Reply::Session(_)
    ));
    assert!(daemon.0.try_wait().unwrap().is_none());
}
