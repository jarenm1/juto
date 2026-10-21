//! One test per process: Client::connect resolves the launcher from PATH, and
//! daemon socket/data paths also use process-wide environment variables.

use std::os::unix::fs::PermissionsExt;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use juto_daemon::client::Client;
use juto_daemon::protocol::{Reply, Request, binary_id, paths};

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn a_running_daemon_accepts_a_wrapped_copy_of_its_build() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    let runtime = root.path().join("run");
    let install = root.path().join("bin");
    for dir in [&workspace, &runtime, &install] {
        std::fs::create_dir(dir).unwrap();
    }
    let workspace = workspace.canonicalize().unwrap();
    let executable = std::path::Path::new(env!("CARGO_BIN_EXE_juto-daemon"));
    let copied = install.join(".juto-daemon-wrapped");
    std::fs::copy(executable, &copied).unwrap();
    let launcher = install.join("juto-daemon");
    std::fs::write(
        &launcher,
        format!("#!/bin/sh\nexec '{}' \"$@\"\n", copied.display()),
    )
    .unwrap();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).unwrap();

    // SAFETY: this integration test is the only test in its process. Set these
    // before starting the daemon or the client connection task.
    unsafe {
        std::env::set_var("XDG_RUNTIME_DIR", &runtime);
        std::env::set_var("JUTO_DATA_DIR", root.path().join("data"));
        std::env::set_var("PATH", &install);
    }
    // Start the original ELF, not the installed copy. Both the wrapper and the
    // copied inode must be irrelevant to the client's build comparison.
    let mut daemon = Daemon(
        Command::new(executable)
            .arg("--workspace")
            .arg(&workspace)
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let socket = paths::daemon_socket(&workspace).unwrap();
    tokio::time::timeout(Duration::from_secs(15), async {
        while !socket.exists() {
            assert!(daemon.0.try_wait().unwrap().is_none(), "daemon exited");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("daemon must open its socket");

    let client = Client::connect(&workspace)
        .await
        .expect("a wrapper launching an identical copied build must be accepted");
    assert_eq!(client.hello().unwrap().pid, daemon.0.id());
    assert!(matches!(
        client.request(Request::ListSessions).await.unwrap(),
        Reply::Sessions(sessions) if sessions.is_empty()
    ));
    assert_eq!(binary_id(executable).unwrap(), binary_id(&copied).unwrap());
    assert_ne!(
        binary_id(executable).unwrap(),
        binary_id(&std::env::current_exe().unwrap()).unwrap(),
        "different linked executables must not share a build identity"
    );
}
