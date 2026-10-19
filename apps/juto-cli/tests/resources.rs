//! `juto-cli resources` as a user sees it: the binary, its flags, and its output.

use std::{path::Path, process::Command};

use juto_runtime::{
    resources::{Outcome, UsageRecord, UsageStore},
    workspace::is_systemd_run_available,
};

const MIB: u64 = 1024 * 1024;

fn record(class: &str, repo: &str, peak: u64, outcome: Outcome) -> UsageRecord {
    UsageRecord {
        recorded_at_ms: 0,
        repo: repo.to_string(),
        class: class.to_string(),
        outcome,
        exit_code: Some(0),
        wall_ms: 2_000,
        memory_peak_bytes: Some(peak),
        cpu_usec: Some(1_500_000),
        oom_kills: Some(0),
        io_read_bytes: None,
        io_write_bytes: None,
    }
}

fn juto_cli(data_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_juto-cli"));
    command.arg("--data-dir").arg(data_dir);
    command
}

fn seed(data_dir: &Path) {
    let store = UsageStore::new(data_dir.join("resources"));
    for record in [
        record(
            "cargo test -p juto-runtime",
            "/repo/a",
            1024 * MIB,
            Outcome::Completed,
        ),
        record(
            "cargo test -p juto-runtime",
            "/repo/a",
            2048 * MIB,
            Outcome::Completed,
        ),
        record(
            "cargo test -p juto-runtime",
            "/repo/a",
            512 * MIB,
            Outcome::Aborted,
        ),
        record(
            "cargo build -p juto",
            "/repo/b",
            4096 * MIB,
            Outcome::Completed,
        ),
    ] {
        store.append(&record).expect("append usage record");
    }
}

fn json_listing(output: std::process::Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("listing is JSON")
}

#[test]
fn json_listing_reports_per_class_counts_and_percentiles() {
    let dir = tempfile::tempdir().expect("tempdir");
    seed(dir.path());

    let listing = json_listing(
        juto_cli(dir.path())
            .args(["--json", "resources"])
            .output()
            .expect("run juto-cli"),
    );

    let classes = listing.as_array().expect("array of classes");
    assert_eq!(classes.len(), 2, "{listing}");
    let tests = classes
        .iter()
        .find(|class| class["class"] == "cargo test -p juto-runtime")
        .expect("test class listed");
    assert_eq!(tests["count"], 3);
    assert_eq!(tests["aborted"], 1);
    // Nearest rank over the two completed peaks (1 GiB, 2 GiB).
    assert_eq!(tests["memory_peak_bytes"]["p50"], 1024 * MIB);
    assert_eq!(tests["memory_peak_bytes"]["p95"], 2048 * MIB);
    assert_eq!(tests["cpu_usec"]["p50"], 1_500_000);
}

#[test]
fn repo_filter_limits_the_listing_to_one_repository() {
    let dir = tempfile::tempdir().expect("tempdir");
    seed(dir.path());

    let listing = json_listing(
        juto_cli(dir.path())
            .args(["--json", "resources", "--repo", "/repo/b"])
            .output()
            .expect("run juto-cli"),
    );

    let classes = listing.as_array().expect("array of classes");
    assert_eq!(classes.len(), 1, "{listing}");
    assert_eq!(classes[0]["class"], "cargo build -p juto");
    assert_eq!(classes[0]["count"], 1);
}

#[test]
fn table_listing_prints_a_row_per_class() {
    let dir = tempfile::tempdir().expect("tempdir");
    seed(dir.path());

    let output = juto_cli(dir.path())
        .arg("resources")
        .output()
        .expect("run juto-cli");
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).expect("utf-8 table");
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0].starts_with("CLASS"), "{text}");
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("cargo test -p juto-runtime") && line.contains("1024 MiB")),
        "{text}"
    );
}

#[test]
fn empty_store_says_so_in_text_and_lists_nothing_in_json() {
    let dir = tempfile::tempdir().expect("tempdir");

    let text = juto_cli(dir.path())
        .arg("resources")
        .output()
        .expect("run juto-cli");
    assert!(text.status.success());
    assert!(
        String::from_utf8_lossy(&text.stdout).contains("no resource usage recorded"),
        "{}",
        String::from_utf8_lossy(&text.stdout)
    );

    let listing = json_listing(
        juto_cli(dir.path())
            .args(["--json", "resources"])
            .output()
            .expect("run juto-cli"),
    );
    assert_eq!(listing, serde_json::json!([]));
}

#[test]
fn run_passes_the_exit_code_through_and_records_the_command() {
    if !is_systemd_run_available() {
        eprintln!("skipping: systemd-run unavailable");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let repo = tempfile::tempdir().expect("repo tempdir");

    let status = juto_cli(dir.path())
        .arg("--cwd")
        .arg(repo.path())
        .args(["resources", "run", "--", "sh", "-c", "exit 4"])
        .status()
        .expect("run juto-cli");
    assert_eq!(status.code(), Some(4));

    let listing = json_listing(
        juto_cli(dir.path())
            .args(["--json", "resources"])
            .output()
            .expect("run juto-cli"),
    );
    let classes = listing.as_array().expect("array of classes");
    assert_eq!(classes.len(), 1, "{listing}");
    assert_eq!(classes[0]["class"], "exit 4");
    assert_eq!(classes[0]["count"], 1);
}

#[test]
fn grant_shell_prints_assignments_for_scripts_to_eval() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = juto_cli(dir.path())
        .args([
            "resources",
            "grant",
            "--shell",
            "--memory-max",
            "2G",
            "--no-slice",
            "--",
            "cargo",
            "build",
        ])
        .output()
        .expect("run juto-cli");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("GRANT_SLICE=''\n"), "{stdout}");
    assert!(stdout.contains("GRANT_CARGO_JOBS='"), "{stdout}");
    assert!(stdout.contains("'CPUWeight=100'"), "{stdout}");
    assert!(stdout.contains("'MemoryMax=2G'"), "{stdout}");
    assert!(stdout.contains(")\n"), "{stdout}");
    assert!(!stdout.contains(" )"), "{stdout}");
}
