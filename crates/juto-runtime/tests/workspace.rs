//! Behavioral tests for the workspace resource/sandbox command construction.
//!
//! The argument-construction tests build a scoped command exactly like the
//! inline `test_scoped_command_construction` unit test does, and assert on the
//! resulting `systemd-run` argv. The live test actually spawns a sandboxed
//! command and observes the filesystem boundary.

use std::path::{Path, PathBuf};

use juto_runtime::workspace::{
    ProtectHome, ResourceLimits, Sandbox, ScopedStdio, SubagentWorkspace, is_systemd_run_available,
};

/// Serializes tests that mutate process-global environment variables.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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

fn args_of(cmd: &std::process::Command) -> Vec<String> {
    cmd.get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

fn index_of(args: &[String], needle: &str) -> usize {
    args.iter()
        .position(|arg| arg == needle)
        .unwrap_or_else(|| panic!("missing arg {needle:?} in {args:?}"))
}

/// The workspace path as the sandbox helper resolves it (canonicalized).
fn resolved_workspace(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[test]
fn sandbox_properties_emitted_on_non_gpu_path() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let ws = workspace_at(dir.path());

    let mut orig = std::process::Command::new("cargo");
    orig.arg("build").arg("--release");

    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());
    let scoped = ws
        .build_scoped_std_command(&orig, &limits)
        .expect("build scoped");

    assert_eq!(scoped.get_program(), "systemd-run");
    let args = args_of(&scoped);
    let dashdash = index_of(&args, "--");

    for prop in [
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "PrivateTmp=yes",
    ] {
        let idx = index_of(&args, prop);
        assert!(idx < dashdash, "{prop} must precede `--`: {args:?}");
    }

    let read_write = format!(
        "ReadWritePaths={}",
        resolved_workspace(dir.path()).display()
    );
    let idx = index_of(&args, &read_write);
    assert!(idx < dashdash, "{read_write} must precede `--`: {args:?}");

    // The scoped program still follows `--`.
    assert_eq!(args[dashdash + 1], "cargo");
    assert_eq!(args[dashdash + 2], "build");
    assert_eq!(args[dashdash + 3], "--release");

    // Namespace properties are unsupported by transient scopes, so a sandboxed
    // command runs as a transient service instead.
    assert!(args.iter().any(|arg| arg == "--pipe"), "{args:?}");
    assert!(!args.iter().any(|arg| arg == "--scope"), "{args:?}");
}

#[test]
fn sandbox_properties_emitted_on_gpu_path() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let ws = workspace_at(dir.path());

    let mut orig = std::process::Command::new("cargo");
    orig.arg("run");

    let limits = ResourceLimits::default()
        .with_gpu(true)
        .with_sandbox(Sandbox::default());
    let scoped = ws
        .build_scoped_std_command(&orig, &limits)
        .expect("build scoped with gpu");

    assert_eq!(scoped.get_program(), "flock");
    let args = args_of(&scoped);

    // `flock <lock>` prefix, then the systemd-run invocation.
    assert!(args[0].contains("gpu.lock"), "{args:?}");
    let systemd = index_of(&args, "systemd-run");
    assert!(systemd > 0, "{args:?}");
    let dashdash = index_of(&args, "--");
    assert!(dashdash > systemd, "{args:?}");

    for prop in [
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "PrivateTmp=yes",
    ] {
        let idx = index_of(&args, prop);
        assert!(
            idx > systemd && idx < dashdash,
            "{prop} must appear after `flock <lock>` and before `--`: {args:?}"
        );
    }

    let read_write = format!(
        "ReadWritePaths={}",
        resolved_workspace(dir.path()).display()
    );
    let idx = index_of(&args, &read_write);
    assert!(idx > systemd && idx < dashdash, "{args:?}");
}

#[test]
fn no_sandbox_properties_when_sandbox_is_none() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }

    // `ResourceLimits::default()` reads AGENT_SANDBOX; serialize against the
    // env test and clear it so the default is deterministic.
    let _guard = env_lock();
    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::remove_var("AGENT_SANDBOX") };

    let dir = tempfile::tempdir().expect("tempdir");
    let ws = workspace_at(dir.path());

    let mut orig = std::process::Command::new("cargo");
    orig.arg("build");

    let limits = ResourceLimits::default();
    assert!(limits.sandbox.is_none(), "sandbox must be opt-in");

    let scoped = ws
        .build_scoped_std_command(&orig, &limits)
        .expect("build scoped");
    let args = args_of(&scoped);

    for prop in [
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ProtectHome=yes",
        "PrivateTmp=yes",
    ] {
        assert!(
            !args.iter().any(|arg| arg == prop),
            "unexpected {prop} without a sandbox: {args:?}"
        );
    }
    assert!(
        !args.iter().any(|arg| arg.starts_with("ReadWritePaths=")),
        "unexpected ReadWritePaths without a sandbox: {args:?}"
    );
    // Without a sandbox the invocation keeps the transient scope.
    assert!(args.iter().any(|arg| arg == "--scope"), "{args:?}");
}

#[test]
fn sandbox_write_paths_and_protect_home_variants() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let ws = workspace_at(dir.path());
    let extra = dir.path().join("extra");
    std::fs::create_dir_all(&extra).expect("create extra dir");

    let sandbox = Sandbox {
        protect_system: false,
        protect_home: ProtectHome::Off,
        private_tmp: false,
        write_paths: vec![extra.clone(), PathBuf::from("relative-dir")],
    };
    let limits = ResourceLimits::default().with_sandbox(sandbox);

    let orig = std::process::Command::new("true");
    let scoped = ws
        .build_scoped_std_command(&orig, &limits)
        .expect("build scoped");
    let args = args_of(&scoped);

    assert!(!args.iter().any(|arg| arg == "ProtectSystem=strict"));
    assert!(!args.iter().any(|arg| arg.starts_with("ProtectHome=")));
    assert!(!args.iter().any(|arg| arg == "PrivateTmp=yes"));

    let workspace = resolved_workspace(dir.path());
    assert!(
        args.contains(&format!("ReadWritePaths={}", workspace.display())),
        "workspace must always be writable: {args:?}"
    );
    assert!(
        args.contains(&format!(
            "ReadWritePaths={}",
            extra.canonicalize().expect("canonicalize extra").display()
        )),
        "absolute write path must be emitted: {args:?}"
    );
    assert!(
        args.contains(&format!(
            "ReadWritePaths={}",
            workspace.join("relative-dir").display()
        )),
        "relative write path must resolve against the workspace: {args:?}"
    );
}

#[test]
fn sandbox_protect_home_yes_emits_property() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let ws = workspace_at(dir.path());

    let sandbox = Sandbox {
        protect_home: ProtectHome::Yes,
        ..Sandbox::default()
    };
    let limits = ResourceLimits::default().with_sandbox(sandbox);

    let orig = std::process::Command::new("true");
    let scoped = ws
        .build_scoped_std_command(&orig, &limits)
        .expect("build scoped");
    let args = args_of(&scoped);

    assert!(args.iter().any(|arg| arg == "ProtectHome=yes"), "{args:?}");
    assert!(!args.iter().any(|arg| arg == "ProtectHome=read-only"));
}

#[test]
fn agent_sandbox_env_enables_the_toolchain_sandbox() {
    let _guard = env_lock();

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::remove_var("AGENT_SANDBOX") };
    assert!(
        ResourceLimits::default().sandbox.is_none(),
        "sandbox must be off by default"
    );

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::set_var("AGENT_SANDBOX", "1") };
    assert_eq!(
        ResourceLimits::default().sandbox,
        Some(Sandbox::for_agent_workspace()),
        "AGENT_SANDBOX=1 must enable the toolchain sandbox"
    );

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::set_var("AGENT_SANDBOX", "true") };
    assert_eq!(
        ResourceLimits::default().sandbox,
        Some(Sandbox::for_agent_workspace()),
        "AGENT_SANDBOX=true must enable the toolchain sandbox"
    );

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::set_var("AGENT_SANDBOX", "0") };
    assert!(
        ResourceLimits::default().sandbox.is_none(),
        "AGENT_SANDBOX=0 must not enable the sandbox"
    );

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::remove_var("AGENT_SANDBOX") };
}

#[test]
fn toolchain_sandbox_grants_the_build_directories() {
    let _guard = env_lock();
    let previous = std::env::var_os("CARGO_HOME");

    // SAFETY: serialized by ENV_LOCK.
    unsafe { std::env::set_var("CARGO_HOME", "/tmp/juto-cargo-home") };
    let sandbox = Sandbox::for_agent_workspace();
    assert!(
        sandbox
            .write_paths
            .contains(&PathBuf::from("/tmp/juto-cargo-home")),
        "an explicit CARGO_HOME must be granted: {:?}",
        sandbox.write_paths
    );
    // The base defaults are unchanged.
    assert!(sandbox.protect_system);
    assert_eq!(sandbox.protect_home, ProtectHome::ReadOnly);
    assert!(sandbox.private_tmp);

    // SAFETY: serialized by ENV_LOCK.
    match previous {
        Some(value) => unsafe { std::env::set_var("CARGO_HOME", value) },
        None => unsafe { std::env::remove_var("CARGO_HOME") },
    }
}

/// Live proof that the sandbox is enforced by the kernel, not just by argv.
///
/// The workspace must live outside `/tmp`: `PrivateTmp=yes` gives the command a
/// private `/tmp`, so a workspace under `/tmp` would not be visible inside the
/// sandbox. A workspace under `$HOME` is visible (read-only) and granted write
/// access through `ReadWritePaths`.
#[tokio::test]
async fn sandbox_allows_workspace_writes_and_denies_home_writes() {
    if !is_systemd_run_available() {
        eprintln!("skipping sandbox live test: systemd-run unavailable");
        return;
    }

    let Some(home) = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!("skipping sandbox live test: HOME unset");
        return;
    };

    let ws_dir = match tempfile::Builder::new()
        .prefix("juto-sandbox-ws-")
        .tempdir_in(&home)
    {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("skipping sandbox live test: cannot create workspace under HOME: {err}");
            return;
        }
    };

    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    // (a) A write inside the workspace succeeds: ReadWritePaths grants it.
    let mut inside = tokio::process::Command::new("sh");
    inside.arg("-c").arg("echo ok > inside.txt");
    let status = ws
        .run_scoped(&mut inside, limits.clone())
        .await
        .expect("run scoped inside");
    assert!(
        status.success(),
        "write inside the workspace must succeed: {status:?}"
    );
    assert!(
        ws_dir.path().join("inside.txt").exists(),
        "inside.txt must exist in the workspace"
    );

    // (b) A write outside the workspace (under $HOME) fails: ProtectHome=read-only.
    let canary = home.join(format!(".juto-sandbox-canary-{}", std::process::id()));
    let _ = std::fs::remove_file(&canary);
    let mut outside = tokio::process::Command::new("sh");
    outside
        .arg("-c")
        .arg(format!("echo canary > {}", canary.display()));
    let status = ws
        .run_scoped(&mut outside, limits)
        .await
        .expect("run scoped outside");
    assert!(
        !status.success(),
        "write outside the workspace must fail: {status:?}"
    );
    assert!(!canary.exists(), "canary must not be created");
    let _ = std::fs::remove_file(&canary);
}

/// Live proof that a sandboxed command keeps the caller's environment.
///
/// A transient *service* runs in the user manager's environment, so without
/// explicit forwarding a sandboxed build loses `PATH` and fails with
/// ``linker `clang` not found``. A scope inherits the environment; a service
/// does not.
#[tokio::test]
async fn sandboxed_service_inherits_the_caller_environment() {
    if !is_systemd_run_available() {
        eprintln!("skipping sandbox live test: systemd-run unavailable");
        return;
    }

    let Some(home) = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!("skipping sandbox live test: HOME unset");
        return;
    };

    let ws_dir = match tempfile::Builder::new()
        .prefix("juto-sandbox-env-")
        .tempdir_in(&home)
    {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("skipping sandbox live test: cannot create workspace under HOME: {err}");
            return;
        }
    };

    let ws = workspace_at(ws_dir.path());
    let limits = ResourceLimits::default().with_sandbox(Sandbox::default());

    let mut command = tokio::process::Command::new("sh");
    command
        .arg("-c")
        .arg("test -n \"$JUTO_SANDBOX_PROBE\" && printf '%s' \"$JUTO_SANDBOX_PROBE\" > probe.txt");
    command.env("JUTO_SANDBOX_PROBE", "inherited");
    let status = ws
        .run_scoped(&mut command, limits)
        .await
        .expect("run scoped env probe");
    assert!(
        status.success(),
        "the probe must see its variable: {status:?}"
    );
    assert_eq!(
        std::fs::read_to_string(ws_dir.path().join("probe.txt")).unwrap(),
        "inherited",
        "a sandboxed service must inherit the caller's environment"
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
    if !is_systemd_run_available() {
        eprintln!("skipping sandbox live test: systemd-run unavailable");
        return;
    }

    let Some(home) = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!("skipping sandbox live test: HOME unset");
        return;
    };

    let ws_dir = match tempfile::Builder::new()
        .prefix("juto-sandbox-stop-")
        .tempdir_in(&home)
    {
        Ok(dir) => dir,
        Err(err) => {
            eprintln!("skipping sandbox live test: cannot create workspace under HOME: {err}");
            return;
        }
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
