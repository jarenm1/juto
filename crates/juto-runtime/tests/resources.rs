//! Consumer-visible behavior of per-command resource usage recording.

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

use juto_runtime::{
    resources::{Outcome, UsageStore, command_class},
    workspace::{ResourceLimits, Sandbox, SubagentWorkspace, is_systemd_run_available},
};
use tokio::process::Command;

/// Burns CPU for a moment, then exits with a distinctive status.
const BUSY_SCRIPT: &str = "i=0; while [ $i -lt 2000000 ]; do i=$((i + 1)); done; exit 3";

fn workspace_at(path: &Path) -> SubagentWorkspace {
    SubagentWorkspace {
        path: path.to_path_buf(),
        node_id: "test-node".to_string(),
        thread: "test-thread".to_string(),
        repo_root: path.to_path_buf(),
        subagent_id: "test-subagent".to_string(),
    }
}

#[tokio::test]
async fn scoped_command_records_its_completed_usage() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let store = UsageStore::new(dir.path().join("resources"));
    let workspace = workspace_at(dir.path());
    let args = ["-c", BUSY_SCRIPT];
    let class_args: Vec<OsString> = args.iter().map(OsString::from).collect();
    let mut command = Command::new("sh");
    command.args(args);

    let status = workspace
        .run_scoped(
            &mut command,
            ResourceLimits::default().with_usage_store(store.clone()),
        )
        .await
        .expect("scoped run");
    assert_eq!(status.code(), Some(3));

    let records = store.read().expect("read usage store");
    assert_eq!(records.len(), 1, "{records:?}");
    let record = &records[0];
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(record.exit_code, Some(3));
    assert_eq!(record.repo, dir.path().display().to_string());
    assert_eq!(record.class, command_class(OsStr::new("sh"), &class_args));
    assert!(record.cpu_usec.is_some_and(|usec| usec > 0), "{record:?}");
    assert!(
        record.memory_peak_bytes.is_some_and(|bytes| bytes > 0),
        "{record:?}"
    );

    // The pending report is consumed once the command is recorded.
    let leftovers = std::fs::read_dir(store.dir().join("pending"))
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(leftovers, 0);
}

#[tokio::test]
async fn scoped_command_without_usage_store_records_nothing() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let workspace = workspace_at(dir.path());
    let mut command = Command::new("sh");
    command.args(["-c", "exit 0"]);

    let status = workspace
        .run_scoped(&mut command, ResourceLimits::default())
        .await
        .expect("scoped run");
    assert!(status.success());
    assert!(!dir.path().join("resources").exists());
}

#[tokio::test]
async fn sandboxed_run_records_a_completed_usage() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }
    if let Err(err) = juto_runtime::network::helper() {
        eprintln!("skipping: {err}");
        return;
    }
    let Some(home) = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(std::path::PathBuf::from)
    else {
        eprintln!("skipping: HOME unset");
        return;
    };
    // The sandbox hides $HOME, so the workspace and the data dir live under it,
    // and the data dir is denied exactly as the runtime denies it.
    let data = tempfile::Builder::new()
        .prefix("juto-usage-data-")
        .tempdir_in(&home)
        .expect("data dir under HOME");
    let root = tempfile::Builder::new()
        .prefix("juto-usage-ws-")
        .tempdir_in(&home)
        .expect("workspace under HOME");
    let store = UsageStore::new(data.path().join("resources"));
    let workspace = workspace_at(root.path());
    let mut command = Command::new("sh");
    command.args(["-c", BUSY_SCRIPT]);

    let mut sandbox = Sandbox::default();
    sandbox.denied_paths.push(data.path().to_path_buf());
    let limits = ResourceLimits::default()
        .with_sandbox(sandbox)
        .with_usage_store(store.clone());
    let status = workspace
        .run_scoped(&mut command, limits)
        .await
        .expect("sandboxed scoped run");
    assert_eq!(status.code(), Some(3));

    let records = store.read().expect("read usage store");
    assert_eq!(records.len(), 1, "{records:?}");
    let record = &records[0];
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(record.exit_code, Some(3));
    assert!(record.cpu_usec.is_some_and(|usec| usec > 0), "{record:?}");
    let leftovers = std::fs::read_dir(store.dir().join("pending"))
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(leftovers, 0);
}

/// Agent slice limits are applied to the live slice and never written into the
/// user's persistent systemd configuration.
#[tokio::test]
async fn agent_slice_limits_are_not_persisted_to_user_config() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .expect("HOME or XDG_CONFIG_HOME");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let slice = format!("juto-test-{}-{nanos}.slice", std::process::id());
    let persisted = config_home
        .join("systemd/user.control")
        .join(format!("{slice}.d"));

    let dir = tempfile::tempdir().expect("tempdir");
    let workspace = workspace_at(dir.path());
    let limits = ResourceLimits {
        slice: Some(slice.clone()),
        ..ResourceLimits::default()
    };
    let mut command = Command::new("sh");
    command.args(["-c", "exit 0"]);
    let status = workspace
        .run_scoped(&mut command, limits)
        .await
        .expect("scoped run");

    let written = persisted.exists();
    // Undo this test's slice state whether or not the bug leaked it.
    let _ = std::process::Command::new("systemctl")
        .args(["--user", "revert", &slice])
        .output();
    let _ = std::fs::remove_dir_all(&persisted);
    assert!(status.success());
    assert!(
        !written,
        "slice limits persisted to {}",
        persisted.display()
    );
}
