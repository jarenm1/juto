//! Live tests for the workspace sandbox and resource governance.
//!
//! Each test runs a real command through `systemd-run` and observes what the
//! kernel allowed it to do: which files it could read or write, and which
//! environment variables it saw. Tests that need `systemd-run` or `$HOME` skip
//! with a message rather than fail. Workspaces live under `$HOME` because the
//! sandbox gives the command a private `/tmp`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use juto_runtime::workspace::{
    ResourceLimits, Sandbox, ScopedStdio, SubagentWorkspace, WorkspaceError,
    is_systemd_run_available,
};

/// Serializes tests that mutate process-global environment variables.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Sets a process environment variable until dropped, then restores it.
///
/// Hold [`env_lock`] for as long as the guard lives.
struct EnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVar {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVar {
    fn drop(&mut self) {
        // SAFETY: serialized by ENV_LOCK, which the test holds.
        match &self.previous {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

/// A file under `$HOME` that is removed when the test ends, even on failure.
struct HomeFile(PathBuf);

impl HomeFile {
    /// A path under `home` that does not exist yet.
    fn new(home: &Path, name: &str) -> Self {
        let path = home.join(format!(".{name}-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        Self(path)
    }
}

impl Drop for HomeFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn workspace_at(path: &Path) -> SubagentWorkspace {
    SubagentWorkspace {
        path: path.to_path_buf(),
        node_id: "test-node".to_string(),
        thread: "test-ws".to_string(),
        repo_root: path.to_path_buf(),
        subagent_id: "test-agent".to_string(),
    }
}

/// `$HOME`, where live tests place their files. `None` skips the test: it needs
/// `systemd-run` to sandbox anything, the network proxy helper (`juto-cli`, or
/// `JUTO_PROXY_HELPER`) that every sandboxed command runs under, and a home
/// directory to place files in.
fn live_home(test: &str) -> Option<PathBuf> {
    if !is_systemd_run_available() {
        eprintln!("skipping {test}: systemd-run unavailable");
        return None;
    }
    if let Err(err) = juto_runtime::network::helper() {
        eprintln!("skipping {test}: {err}");
        return None;
    }
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from);
    if home.is_none() {
        eprintln!("skipping {test}: HOME unset");
    }
    home
}

fn tempdir_in(home: &Path, prefix: &str) -> Option<tempfile::TempDir> {
    match tempfile::Builder::new().prefix(prefix).tempdir_in(home) {
        Ok(dir) => Some(dir),
        Err(err) => {
            eprintln!("skipping: cannot create {prefix}* under HOME: {err}");
            None
        }
    }
}

/// Runs `script` under `sh -c` in the sandbox described by `limits`, with the
/// workspace as its working directory, and waits for it to exit.
fn run_sandboxed(ws: &SubagentWorkspace, limits: &ResourceLimits, script: &str) -> ExitStatus {
    let mut original = std::process::Command::new("sh");
    original.arg("-c").arg(script).current_dir(&ws.path);
    run_command_sandboxed(ws, limits, &original)
}

/// Runs `original` in the sandbox described by `limits` and waits for it to exit.
fn run_command_sandboxed(
    ws: &SubagentWorkspace,
    limits: &ResourceLimits,
    original: &std::process::Command,
) -> ExitStatus {
    ws.build_scoped_std_command(original, limits)
        .expect("build sandboxed command")
        .status()
        .expect("run sandboxed command")
}

#[test]
fn sandbox_allows_workspace_writes_and_denies_home_writes() {
    let test = "sandbox_allows_workspace_writes_and_denies_home_writes";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-ws-") else {
        return;
    };
    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    let inside = run_sandboxed(&ws, &limits, "echo ok > inside.txt");
    assert!(
        inside.success(),
        "a write inside the workspace must succeed: {inside:?}"
    );
    assert!(ws_dir.path().join("inside.txt").exists());

    let outside = HomeFile::new(&home, "juto-sandbox-outside");
    let status = run_sandboxed(
        &ws,
        &limits,
        &format!("echo canary > {}", outside.0.display()),
    );
    assert!(
        !status.success(),
        "a write under $HOME, outside the workspace, must fail: {status:?}"
    );
    assert!(!outside.0.exists(), "the sandbox must not create the file");
}

#[test]
fn sandbox_hides_credentials_under_home() {
    let test = "sandbox_hides_credentials_under_home";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-cred-") else {
        return;
    };
    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    let secret = HomeFile::new(&home, "juto-sandbox-secret");
    std::fs::write(&secret.0, "credential-canary").unwrap();

    let status = run_sandboxed(
        &ws,
        &limits,
        &format!("cat {} > leaked.txt", secret.0.display()),
    );
    assert!(
        !status.success(),
        "reading a file under $HOME must fail inside the sandbox: {status:?}"
    );
    let leaked = std::fs::read_to_string(ws_dir.path().join("leaked.txt")).unwrap_or_default();
    assert!(
        !leaked.contains("credential-canary"),
        "the credential must not reach the workspace"
    );
}

#[test]
fn sandbox_grants_expose_only_the_listed_paths() {
    let test = "sandbox_grants_expose_only_the_listed_paths";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-grants-") else {
        return;
    };
    let Some(read_dir) = tempdir_in(&home, "juto-sandbox-ro-") else {
        return;
    };
    let Some(write_dir) = tempdir_in(&home, "juto-sandbox-rw-") else {
        return;
    };
    let Some(hidden_dir) = tempdir_in(&home, "juto-sandbox-hidden-") else {
        return;
    };
    std::fs::write(read_dir.path().join("note.txt"), "granted").unwrap();
    std::fs::write(hidden_dir.path().join("note.txt"), "hidden").unwrap();

    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox {
        read_paths: vec![read_dir.path().to_path_buf()],
        write_paths: vec![write_dir.path().to_path_buf()],
        ..Sandbox::default()
    });

    let read = run_sandboxed(
        &ws,
        &limits,
        &format!("cat {}/note.txt > read.txt", read_dir.path().display()),
    );
    assert!(read.success(), "a read grant must be readable: {read:?}");
    assert_eq!(
        std::fs::read_to_string(ws_dir.path().join("read.txt")).unwrap(),
        "granted"
    );

    let read_only_write = run_sandboxed(
        &ws,
        &limits,
        &format!("echo x > {}/new.txt", read_dir.path().display()),
    );
    assert!(
        !read_only_write.success(),
        "a read grant must not be writable: {read_only_write:?}"
    );
    assert!(!read_dir.path().join("new.txt").exists());

    let write = run_sandboxed(
        &ws,
        &limits,
        &format!("echo ok > {}/new.txt", write_dir.path().display()),
    );
    assert!(write.success(), "a write grant must be writable: {write:?}");
    assert_eq!(
        std::fs::read_to_string(write_dir.path().join("new.txt"))
            .unwrap()
            .trim(),
        "ok"
    );

    let hidden = run_sandboxed(
        &ws,
        &limits,
        &format!("cat {}/note.txt > hidden.txt", hidden_dir.path().display()),
    );
    assert!(
        !hidden.success(),
        "an ungranted directory under $HOME must be hidden: {hidden:?}"
    );
}

/// The sandbox sees the caller's allowlisted environment, not the whole of it.
#[test]
fn sandboxed_command_sees_only_the_allowlisted_environment() {
    let test = "sandboxed_command_sees_only_the_allowlisted_environment";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-env-") else {
        return;
    };
    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    let _lock = env_lock();
    let _agent = EnvVar::set("SSH_AUTH_SOCK", "/nonexistent/juto-probe.sock");

    let status = run_sandboxed(&ws, &limits, "env > env.txt");
    assert!(status.success(), "the env probe must run: {status:?}");
    let env = std::fs::read_to_string(ws_dir.path().join("env.txt")).unwrap();
    assert!(
        !env.lines().any(|line| line.starts_with("SSH_AUTH_SOCK=")),
        "the agent socket must not reach a sandboxed command"
    );
    assert!(
        env.lines().any(|line| line.starts_with("PATH=")),
        "a sandboxed command must still get PATH"
    );
}

/// Explicit `Command` environment entries reach the sandbox. An override wins over
/// the inherited value, an explicit removal unsets it, and an explicit entry outside
/// the allowlist is still forwarded.
#[test]
fn explicit_environment_entries_reach_the_sandbox() {
    let test = "explicit_environment_entries_reach_the_sandbox";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-explicit-env-") else {
        return;
    };
    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    let _lock = env_lock();
    let _overridden = EnvVar::set("CARGO_JUTO_PROBE_OVERRIDDEN", "inherited");
    let _removed = EnvVar::set("CARGO_JUTO_PROBE_REMOVED", "present");

    let mut original = std::process::Command::new("sh");
    original
        .arg("-c")
        .arg("env > env.txt")
        .current_dir(ws_dir.path())
        .env("CARGO_JUTO_PROBE_OVERRIDDEN", "overridden")
        .env("JUTO_PROBE_EXPLICIT", "explicit")
        .env_remove("CARGO_JUTO_PROBE_REMOVED");
    let status = run_command_sandboxed(&ws, &limits, &original);
    assert!(status.success(), "the env probe must run: {status:?}");

    let env = std::fs::read_to_string(ws_dir.path().join("env.txt")).unwrap();
    let lines: Vec<&str> = env.lines().collect();
    assert!(
        lines.contains(&"CARGO_JUTO_PROBE_OVERRIDDEN=overridden"),
        "the override must win over the inherited value:\n{env}"
    );
    assert!(
        lines.contains(&"JUTO_PROBE_EXPLICIT=explicit"),
        "an explicit entry must be forwarded:\n{env}"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.starts_with("CARGO_JUTO_PROBE_REMOVED=")),
        "an explicitly removed variable must not reach the sandbox:\n{env}"
    );
}

/// The agent sandbox makes the cargo home writable, so `cargo` can fetch and
/// build inside the sandbox.
#[test]
fn agent_workspace_sandbox_writes_the_cargo_home() {
    let test = "agent_workspace_sandbox_writes_the_cargo_home";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-cargo-ws-") else {
        return;
    };
    let Some(cargo_home) = tempdir_in(&home, "juto-sandbox-cargo-home-") else {
        return;
    };
    let ws = workspace_at(ws_dir.path());
    let _lock = env_lock();
    let _cargo = EnvVar::set("CARGO_HOME", cargo_home.path());
    // `for_agent_workspace` reads CARGO_HOME when it is built, so it must be built
    // after the variable is set.
    let limits = ResourceLimits::default().with_sandbox(Sandbox::for_agent_workspace());

    let status = run_sandboxed(&ws, &limits, "echo ok > \"$CARGO_HOME/probe.txt\"");
    assert!(
        status.success(),
        "the cargo home must be writable: {status:?}"
    );
    assert!(cargo_home.path().join("probe.txt").exists());
}

/// Without `systemd-run` a governed command is refused, never run bare.
#[test]
fn missing_systemd_run_refuses_the_command() {
    let dir = tempfile::tempdir().unwrap();
    let empty_path = tempfile::tempdir().unwrap();
    let ws = workspace_at(dir.path());
    let mut original = std::process::Command::new("sh");
    original
        .arg("-c")
        .arg("echo ran > marker.txt")
        .current_dir(dir.path());

    let _lock = env_lock();
    let _path = EnvVar::set("PATH", empty_path.path());

    let capped = ws.build_scoped_std_command(&original, &ResourceLimits::default());
    assert!(
        matches!(capped, Err(WorkspaceError::SystemdUnavailable)),
        "a capped command must be refused without systemd-run"
    );
    let sandboxed = ws.build_scoped_std_command(
        &original,
        &ResourceLimits::default().with_sandbox(Sandbox::default()),
    );
    assert!(
        matches!(sandboxed, Err(WorkspaceError::SystemdUnavailable)),
        "a sandboxed command must be refused without systemd-run"
    );
}

/// Whether the transient unit `unit` is currently active.
fn unit_is_active(unit: &str) -> bool {
    std::process::Command::new("systemctl")
        .args(["--user", "is-active", unit])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim() == "active")
        .unwrap_or(false)
}

/// Live proof that dropping a sandboxed child stops its transient unit.
///
/// A transient service outlives its `systemd-run` client, so without the guard a
/// cancelled command would keep working.
#[tokio::test]
async fn dropping_a_sandboxed_child_stops_its_unit() {
    let test = "dropping_a_sandboxed_child_stops_its_unit";
    let Some(home) = live_home(test) else {
        return;
    };
    let Some(ws_dir) = tempdir_in(&home, "juto-sandbox-stop-") else {
        return;
    };
    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    let mut command = tokio::process::Command::new("sleep");
    command.arg("60");
    let child = ws
        .spawn_scoped(&mut command, limits, ScopedStdio::default())
        .await
        .expect("spawn sandboxed child");
    let unit = child
        .unit()
        .expect("a sandboxed child must name its unit")
        .to_string();
    // `systemd-run` starts the unit asynchronously, so it becomes active
    // shortly after the spawn rather than immediately.
    let mut active = false;
    for _ in 0..50 {
        if unit_is_active(&unit) {
            active = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(active, "the unit must become active while running");

    drop(child);

    let mut stopped = false;
    for _ in 0..50 {
        if !unit_is_active(&unit) {
            stopped = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(stopped, "dropping the child must stop unit {unit}");
}
