//! Per-command resource usage: one record per resource-scoped command.
//!
//! A scoped command's cgroup accounting (`memory.peak`, `cpu.stat`, `io.stat`,
//! `memory.events`) is only readable while its unit exists. systemd drops it on
//! exit, so the payload is wrapped in a small POSIX shell that reads the
//! accounting from inside the cgroup just before the payload exits and writes
//! one report line to a pending file. The runtime reads that file after the
//! child is reaped and appends a [`UsageRecord`] to the store.
//!
//! Layout under the store directory:
//!
//! - `usage.jsonl`: one JSON record per line, append-only.
//! - `pending/<uuid>.usage`: transient report written by the wrapper.
//!
//! Records are keyed by repository root and by a *command class*: the program
//! and its plain option/subcommand tokens. Path-like tokens are dropped so that
//! the same command in two checkouts lands in one class.

use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitStatus,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use juto_capacity::MachineCapacity;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const USAGE_LOG: &str = "usage.jsonl";
const PENDING_DIR: &str = "pending";
const MAX_CLASS_TOKENS: usize = 8;
const MAX_CLASS_TOKEN_LEN: usize = 64;

/// POSIX shell wrapper placed around a scoped payload.
///
/// Arguments: `$1` is the pending report file, the rest is the payload argv.
/// The payload runs unchanged and its exit status is preserved. Accounting is
/// read from the cgroup the wrapper itself is in, which is the unit's cgroup;
/// a value that cannot be read is reported as `-`.
pub(crate) const USAGE_WRAPPER: &str = r#"usage=$1; shift
"$@"
rc=$?
cg=$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null)
d=/nonexistent
case "$cg" in /?*) d=/sys/fs/cgroup$cg ;; esac
peak=$(cat "$d/memory.peak" 2>/dev/null)
cpu=$(awk '$1 == "usage_usec" { print $2 }' "$d/cpu.stat" 2>/dev/null)
oom=$(awk '$1 == "oom_kill" { print $2 }' "$d/memory.events" 2>/dev/null)
io=-
if [ -r "$d/io.stat" ]; then
  io=$(awk '{ for (i = 2; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "rbytes") r += kv[2]; if (kv[1] == "wbytes") w += kv[2] } } END { print r + 0, w + 0 }' "$d/io.stat" 2>/dev/null)
fi
set -- $io
printf '%s %s %s %s %s %s\n' "$rc" "${peak:--}" "${cpu:--}" "${oom:--}" "${1:--}" "${2:--}" >> "$usage"
exit "$rc"
"#;

/// How a scoped command ended, as far as usage accounting can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The wrapper reported the payload's exit status and accounting.
    Completed,
    /// The command was stopped before the wrapper could report (timeout,
    /// cancellation, a kill of the whole unit). Only wall time is known.
    Aborted,
}

/// Resource usage of one scoped command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Unix epoch milliseconds when the command was reaped.
    pub recorded_at_ms: u64,
    /// Repository root the command ran in.
    pub repo: String,
    /// Command class fingerprint; see [`command_class`].
    pub class: String,
    pub outcome: Outcome,
    /// Payload exit status; `128 + signal` when it was killed by a signal.
    pub exit_code: Option<i32>,
    pub wall_ms: u64,
    /// Peak cgroup memory in bytes (`memory.peak`).
    pub memory_peak_bytes: Option<u64>,
    /// Total CPU time in microseconds (`cpu.stat` `usage_usec`).
    pub cpu_usec: Option<u64>,
    /// Number of OOM kills inside the cgroup (`memory.events` `oom_kill`).
    pub oom_kills: Option<u64>,
    /// Bytes read from block devices (`io.stat` `rbytes`, summed).
    pub io_read_bytes: Option<u64>,
    /// Bytes written to block devices (`io.stat` `wbytes`, summed).
    pub io_write_bytes: Option<u64>,
}

/// The line the wrapper writes to its pending file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WrapperReport {
    pub exit_code: i32,
    pub memory_peak_bytes: Option<u64>,
    pub cpu_usec: Option<u64>,
    pub oom_kills: Option<u64>,
    pub io_read_bytes: Option<u64>,
    pub io_write_bytes: Option<u64>,
}

impl WrapperReport {
    /// Parse `rc peak cpu oom rbytes wbytes`, with `-` for unavailable fields.
    pub(crate) fn parse(line: &str) -> Option<Self> {
        let mut fields = line.split_whitespace();
        let exit_code = fields.next()?.parse().ok()?;
        let mut value = || -> Option<Option<u64>> {
            let field = fields.next()?;
            Some(if field == "-" {
                None
            } else {
                Some(field.parse().ok()?)
            })
        };
        let memory_peak_bytes = value()?;
        let cpu_usec = value()?;
        let oom_kills = value()?;
        let io_read_bytes = value()?;
        let io_write_bytes = value()?;
        if fields.next().is_some() {
            return None;
        }
        Some(Self {
            exit_code,
            memory_peak_bytes,
            cpu_usec,
            oom_kills,
            io_read_bytes,
            io_write_bytes,
        })
    }
}

/// Append-only usage store rooted at a directory such as `<data_dir>/resources`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageStore {
    dir: PathBuf,
}

impl UsageStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Path of the JSON-lines record log.
    pub fn log_path(&self) -> PathBuf {
        self.dir.join(USAGE_LOG)
    }

    /// Allocate a fresh pending report path for one command.
    pub(crate) fn new_pending_file(&self) -> io::Result<PathBuf> {
        let dir = self.dir.join(PENDING_DIR);
        fs::create_dir_all(&dir)?;
        Ok(dir.join(format!("{}.usage", Uuid::now_v7())))
    }

    /// Append one record as a single JSON line.
    pub fn append(&self, record: &UsageRecord) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let mut line = serde_json::to_string(record).map_err(io::Error::other)?;
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path())?;
        file.write_all(line.as_bytes())
    }

    /// Read every record. A missing log is empty; malformed lines are skipped
    /// so one torn write cannot hide the rest of the history.
    pub fn read(&self) -> io::Result<Vec<UsageRecord>> {
        let text = match fs::read_to_string(self.log_path()) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        Ok(text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect())
    }
}

/// Build the record for a command whose wrapper may or may not have reported.
pub(crate) fn record_for(
    repo: &str,
    class: &str,
    wall_ms: u64,
    status_code: Option<i32>,
    report: Option<WrapperReport>,
) -> UsageRecord {
    let recorded_at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0);
    let (outcome, exit_code) = match report {
        Some(report) => (Outcome::Completed, Some(report.exit_code)),
        None => (Outcome::Aborted, status_code),
    };
    UsageRecord {
        recorded_at_ms,
        repo: repo.to_string(),
        class: class.to_string(),
        outcome,
        exit_code,
        wall_ms,
        memory_peak_bytes: report.and_then(|r| r.memory_peak_bytes),
        cpu_usec: report.and_then(|r| r.cpu_usec),
        oom_kills: report.and_then(|r| r.oom_kills),
        io_read_bytes: report.and_then(|r| r.io_read_bytes),
        io_write_bytes: report.and_then(|r| r.io_write_bytes),
    }
}

/// Fingerprint a command as `program` plus its plain tokens, in order.
///
/// Shell command strings (`sh -c '…'`, `bash -c '…'`) are classified by their
/// first simple command, skipping leading `cd`, `export`, and `NAME=value`
/// assignments. Tokens containing `/`, `$`, quotes, or other characters outside
/// `[A-Za-z0-9._:=+@-]` are dropped, so checkout paths and variables do not
/// split one command into many classes.
pub fn command_class(program: &OsStr, args: &[OsString]) -> String {
    let program = basename(program);
    let shell = matches!(program.as_str(), "sh" | "bash" | "zsh" | "dash");
    if shell && args.first().map(|a| a.as_os_str()) == Some(OsStr::new("-c")) {
        if let Some(words) = args
            .get(1)
            .and_then(|s| s.to_str())
            .and_then(first_simple_command)
        {
            return class_from_words(words);
        }
        return program;
    }
    let mut words = vec![program];
    words.extend(args.iter().map(|a| a.to_string_lossy().into_owned()));
    class_from_words(words)
}

fn basename(program: &OsStr) -> String {
    Path::new(program)
        .file_name()
        .unwrap_or(program)
        .to_string_lossy()
        .into_owned()
}

fn class_from_words(words: Vec<String>) -> String {
    words
        .into_iter()
        .filter(|word| is_plain_token(word))
        .take(MAX_CLASS_TOKENS)
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_plain_token(word: &str) -> bool {
    !word.is_empty()
        && word.len() <= MAX_CLASS_TOKEN_LEN
        && word.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'=' | b'+' | b'@' | b'-')
        })
}

/// Words of the first command in `script` that is not `cd`, `export`, `set`,
/// or `true`, after leading `NAME=value` assignments. Splits on the shell
/// separators `;`, `&`, `|`, and newline.
fn first_simple_command(script: &str) -> Option<Vec<String>> {
    for segment in script.split([';', '&', '|', '\n']) {
        let words: Vec<&str> = segment
            .split_whitespace()
            .skip_while(|word| is_assignment(word))
            .collect();
        match words.first() {
            None | Some(&("cd" | "export" | "set" | "true")) => continue,
            Some(_) => return Some(words.into_iter().map(str::to_string).collect()),
        }
    }
    None
}

fn is_assignment(word: &str) -> bool {
    word.contains('=') && !word.starts_with('-')
}

/// Nearest-rank percentile of `sorted` (ascending). `None` when empty.
pub fn nearest_rank(sorted: &[u64], percent: u64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len() as u64;
    let rank = (percent * n).div_ceil(100).clamp(1, n);
    Some(sorted[(rank - 1) as usize])
}

/// p50 and p95 of one metric over a class's completed samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Percentiles {
    pub p50: Option<u64>,
    pub p95: Option<u64>,
}

/// Aggregate usage for one command class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClassSummary {
    pub class: String,
    pub count: usize,
    pub aborted: usize,
    pub oom_kills: u64,
    pub memory_peak_bytes: Percentiles,
    pub cpu_usec: Percentiles,
    pub wall_ms: Percentiles,
}

/// Group records by class and compute nearest-rank percentiles per metric.
///
/// Memory and CPU percentiles use only records where the wrapper reported
/// them; wall time uses every record.
pub fn summarize(records: &[UsageRecord]) -> Vec<ClassSummary> {
    let mut groups: BTreeMap<&str, Vec<&UsageRecord>> = BTreeMap::new();
    for record in records {
        groups
            .entry(record.class.as_str())
            .or_default()
            .push(record);
    }
    groups
        .into_iter()
        .map(|(class, group)| {
            let percentiles = |values: Vec<u64>| {
                let mut values = values;
                values.sort_unstable();
                Percentiles {
                    p50: nearest_rank(&values, 50),
                    p95: nearest_rank(&values, 95),
                }
            };
            ClassSummary {
                class: class.to_string(),
                count: group.len(),
                aborted: group
                    .iter()
                    .filter(|r| r.outcome == Outcome::Aborted)
                    .count(),
                oom_kills: group.iter().filter_map(|r| r.oom_kills).sum(),
                memory_peak_bytes: percentiles(
                    group.iter().filter_map(|r| r.memory_peak_bytes).collect(),
                ),
                cpu_usec: percentiles(group.iter().filter_map(|r| r.cpu_usec).collect()),
                wall_ms: percentiles(group.iter().map(|r| r.wall_ms).collect()),
            }
        })
        .collect()
}

/// One scoped command's pending report and the context needed to record it.
///
/// The report path is part of the payload argv, so the ticket exists before
/// spawn. [`UsageTicket::finish`] records the command after it is reaped. A
/// ticket dropped unfinished (spawn failure, cancelled wait) removes its
/// pending file; a command whose unit was stopped before the wrapper reported
/// leaves no record.
#[derive(Debug)]
pub(crate) struct UsageTicket {
    store: UsageStore,
    pending: PathBuf,
    repo: String,
    class: String,
    started: Instant,
}

impl UsageTicket {
    pub(crate) fn start(store: &UsageStore, repo: String, class: String) -> io::Result<Self> {
        Ok(Self {
            store: store.clone(),
            pending: store.new_pending_file()?,
            repo,
            class,
            started: Instant::now(),
        })
    }

    /// The report file the wrapper appends to.
    pub(crate) fn pending_file(&self) -> &Path {
        &self.pending
    }

    /// The directory that holds the pending report, for a sandbox grant.
    pub(crate) fn pending_dir(&self) -> &Path {
        self.pending.parent().unwrap_or(&self.pending)
    }

    /// Read the wrapper's report, then append the record. Recording is
    /// best-effort: a failed append is reported on stderr and does not change
    /// the command's result.
    pub(crate) fn finish(self, status: ExitStatus) {
        let report = fs::read_to_string(&self.pending)
            .ok()
            .and_then(|text| text.lines().next().and_then(WrapperReport::parse));
        let wall_ms = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let record = record_for(&self.repo, &self.class, wall_ms, exit_code(status), report);
        if let Err(error) = self.store.append(&record) {
            eprintln!(
                "juto: could not record resource usage in {}: {error}",
                self.store.dir().display()
            );
        }
    }
}

impl Drop for UsageTicket {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.pending);
    }
}

#[cfg(unix)]
fn exit_code(status: ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|signal| 128 + signal))
}

#[cfg(not(unix))]
fn exit_code(status: ExitStatus) -> Option<i32> {
    status.code()
}

/// Memory and CPU grant for one command, derived from measured capacity and the
/// usage history of its repository and class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandGrant {
    /// Kill line for the scope (`MemoryMax`).
    pub memory_max_bytes: u64,
    /// Throttle line for the scope (`MemoryHigh`): nine tenths of `memory_max_bytes`.
    pub memory_high_bytes: u64,
    /// `TasksMax` for the scope.
    pub tasks_max: u32,
    /// Parallel jobs `cargo` may use under this grant.
    pub cargo_jobs: usize,
    /// False when no sample exists and the first-run rule applied.
    pub from_history: bool,
}

/// Percentile of observed peaks that a learned estimate is based on.
const ESTIMATE_PERCENTILE: u64 = 95;

/// A command with no history may use this fraction of the agent budget.
const FIRST_RUN_BUDGET_DIVISOR: u64 = 2;

/// Derive the grant for a command in `repo` of `class`.
///
/// The learned estimate is the p95 of recorded peaks plus a quarter. A sample
/// killed by the OOM killer reports a lower bound, so its peak is doubled. The
/// result never exceeds the agent budget.
pub fn command_grant(
    capacity: &MachineCapacity,
    store: Option<&UsageStore>,
    repo: &str,
    class: &str,
) -> io::Result<CommandGrant> {
    let records = match store {
        Some(store) => store.read()?,
        None => Vec::new(),
    };
    let samples: Vec<&UsageRecord> = records
        .iter()
        .filter(|record| record.repo == repo && record.class == class)
        .collect();
    let mut peaks: Vec<u64> = samples.iter().filter_map(|r| r.memory_peak_bytes).collect();
    peaks.sort_unstable();
    let learned = nearest_rank(&peaks, ESTIMATE_PERCENTILE);
    let mut bytes = match learned {
        Some(peak) => peak.saturating_add(peak / 4),
        None => capacity.budget_bytes / FIRST_RUN_BUDGET_DIVISOR,
    };
    for sample in &samples {
        if sample.oom_kills.unwrap_or(0) > 0 {
            bytes = bytes.max(sample.memory_peak_bytes.unwrap_or(bytes).saturating_mul(2));
        }
    }
    let memory = bytes.clamp(1, capacity.budget_bytes);
    Ok(CommandGrant {
        memory_max_bytes: memory,
        memory_high_bytes: memory / 10 * 9,
        tasks_max: capacity.tasks_max,
        cargo_jobs: capacity.cargo_jobs(memory),
        from_history: learned.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(words: &[&str]) -> Vec<OsString> {
        words.iter().map(OsString::from).collect()
    }

    #[test]
    fn class_drops_paths_and_keeps_subcommand_and_flags() {
        let class = command_class(
            OsStr::new("/usr/bin/cargo"),
            &os(&[
                "build",
                "-p",
                "juto-runtime",
                "--manifest-path",
                "/home/x/y/Cargo.toml",
            ]),
        );
        assert_eq!(class, "cargo build -p juto-runtime --manifest-path");
    }

    #[test]
    fn shell_command_is_classified_by_first_simple_command() {
        let class = command_class(
            OsStr::new("sh"),
            &os(&[
                "-c",
                "cd /tmp/checkout && CARGO_TARGET_DIR=$X cargo test -p juto-runtime | tail",
            ]),
        );
        assert_eq!(class, "cargo test -p juto-runtime");
    }

    #[test]
    fn nearest_rank_matches_definition() {
        let values = [10, 20, 30, 40];
        assert_eq!(nearest_rank(&values, 50), Some(20));
        assert_eq!(nearest_rank(&values, 95), Some(40));
        assert_eq!(nearest_rank(&values, 1), Some(10));
        assert_eq!(nearest_rank(&[], 50), None);
    }

    #[test]
    fn wrapper_report_parses_unknown_fields_as_none() {
        let report = WrapperReport::parse("0 4096 1500 - 10 20").expect("parse");
        assert_eq!(report.exit_code, 0);
        assert_eq!(report.memory_peak_bytes, Some(4096));
        assert_eq!(report.oom_kills, None);
        assert_eq!(report.io_write_bytes, Some(20));
        assert!(WrapperReport::parse("0 1 2 3 4").is_none());
    }

    #[test]
    fn store_round_trips_and_skips_malformed_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = UsageStore::new(dir.path().join("resources"));
        assert!(store.read().expect("read empty").is_empty());
        let record = record_for(
            "/repo",
            "cargo build",
            1200,
            Some(0),
            WrapperReport::parse("0 1 2 0 - -"),
        );
        store.append(&record).expect("append");
        fs::OpenOptions::new()
            .append(true)
            .open(store.log_path())
            .and_then(|mut f| f.write_all(b"{not json\n"))
            .expect("torn line");
        assert_eq!(store.read().expect("read"), vec![record]);
    }

    #[test]
    fn summary_reports_per_class_percentiles_and_aborts() {
        let report = |peak: u64| WrapperReport {
            exit_code: 0,
            memory_peak_bytes: Some(peak),
            cpu_usec: Some(peak / 2),
            oom_kills: Some(0),
            io_read_bytes: None,
            io_write_bytes: None,
        };
        let mut records: Vec<UsageRecord> = (1..=20)
            .map(|i| {
                record_for(
                    "/repo",
                    "cargo build",
                    i * 10,
                    Some(0),
                    Some(report(i * 100)),
                )
            })
            .collect();
        records.push(record_for("/repo", "cargo build", 5, Some(137), None));
        records.push(record_for("/repo", "cargo test", 7, Some(0), None));

        let summaries = summarize(&records);
        assert_eq!(summaries.len(), 2);
        let build = &summaries[0];
        assert_eq!(build.class, "cargo build");
        assert_eq!(build.count, 21);
        assert_eq!(build.aborted, 1);
        assert_eq!(build.memory_peak_bytes.p50, Some(1000));
        assert_eq!(build.memory_peak_bytes.p95, Some(1900));
        assert_eq!(build.wall_ms.p50, Some(100));
    }

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    fn capacity() -> MachineCapacity {
        MachineCapacity::from_measurements(64 * GIB, 16, 2 * GIB).expect("capacity")
    }

    fn peak_report(peak: u64, oom_kills: u64) -> Option<WrapperReport> {
        Some(WrapperReport {
            exit_code: 0,
            memory_peak_bytes: Some(peak),
            cpu_usec: None,
            oom_kills: Some(oom_kills),
            io_read_bytes: None,
            io_write_bytes: None,
        })
    }

    fn store_with(records: &[UsageRecord]) -> (tempfile::TempDir, UsageStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = UsageStore::new(dir.path().join("resources"));
        for record in records {
            store.append(record).expect("append");
        }
        (dir, store)
    }

    #[test]
    fn first_run_grants_half_the_budget_without_history() {
        let capacity = capacity();
        let grant = command_grant(&capacity, None, "/repo", "cargo build").expect("grant");
        assert!(!grant.from_history);
        assert_eq!(grant.memory_max_bytes, capacity.budget_bytes / 2);
        assert_eq!(grant.memory_high_bytes, grant.memory_max_bytes / 10 * 9);
        assert_eq!(grant.tasks_max, capacity.tasks_max);
        assert_eq!(
            grant.cargo_jobs,
            capacity.cargo_jobs(grant.memory_max_bytes)
        );
    }

    #[test]
    fn learned_grant_is_p95_of_peaks_plus_a_quarter() {
        // Twenty runs peaking at 100, 200, …, 2000 MiB: nearest-rank p95 is 1900 MiB.
        let records: Vec<UsageRecord> = (1..=20)
            .map(|i| {
                record_for(
                    "/repo",
                    "cargo build",
                    10,
                    Some(0),
                    peak_report(i * 100 * MIB, 0),
                )
            })
            .collect();
        let (_dir, store) = store_with(&records);
        let grant =
            command_grant(&capacity(), Some(&store), "/repo", "cargo build").expect("grant");
        assert!(grant.from_history);
        assert_eq!(grant.memory_max_bytes, 1900 * MIB + 1900 * MIB / 4);
    }

    #[test]
    fn oom_kill_doubles_the_observed_peak() {
        let records = [record_for(
            "/repo",
            "cargo build",
            10,
            Some(137),
            peak_report(GIB, 1),
        )];
        let (_dir, store) = store_with(&records);
        let grant =
            command_grant(&capacity(), Some(&store), "/repo", "cargo build").expect("grant");
        assert_eq!(grant.memory_max_bytes, 2 * GIB);
    }

    #[test]
    fn grant_never_exceeds_the_agent_budget() {
        let capacity = capacity();
        let records = [record_for(
            "/repo",
            "cargo build",
            10,
            Some(0),
            peak_report(capacity.budget_bytes * 2, 0),
        )];
        let (_dir, store) = store_with(&records);
        let grant = command_grant(&capacity, Some(&store), "/repo", "cargo build").expect("grant");
        assert_eq!(grant.memory_max_bytes, capacity.budget_bytes);
    }

    #[test]
    fn history_of_other_repos_and_classes_is_ignored() {
        let capacity = capacity();
        let records = [
            record_for(
                "/other",
                "cargo build",
                10,
                Some(0),
                peak_report(10 * GIB, 0),
            ),
            record_for("/repo", "cargo test", 10, Some(0), peak_report(10 * GIB, 0)),
        ];
        let (_dir, store) = store_with(&records);
        let grant = command_grant(&capacity, Some(&store), "/repo", "cargo build").expect("grant");
        assert!(!grant.from_history);
        assert_eq!(grant.memory_max_bytes, capacity.budget_bytes / 2);
    }
}
