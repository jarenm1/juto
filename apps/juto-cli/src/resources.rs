//! `juto-cli resources`: the measured machine capacity, the limits a command
//! would receive (`grant`), measured resource use per command class, and `run`,
//! which executes a command under the agent resource governance so that its
//! usage is recorded.

use std::{ffi::OsString, io::Write as _, path::Path, process::ExitStatus};

use anyhow::{Context as _, Result};
use clap::Subcommand;
use juto_runtime::{
    ResolvedLimits, ResourceLimits, ScopedStdio, SubagentWorkspace,
    broker::{Broker, Priority},
    resources::{ClassSummary, UsageRecord, UsageStore, summarize},
    workspace::QueueObserver,
};
use tokio::process::Command;

#[derive(Subcommand)]
pub enum ResourcesCommand {
    /// Run a command under agent resource governance, recording its usage.
    ///
    /// Everything after `run` is passed through unchanged, including flags.
    Run {
        /// Admission class when the resource budget is short: `attended` (default),
        /// `verification`, `background`, or `scheduled`.
        #[arg(long)]
        priority: Option<Priority>,
        #[arg(
            required = true,
            num_args = 1..,
            trailing_var_arg = true,
            allow_hyphen_values = true
        )]
        argv: Vec<OsString>,
    },
    /// Show the machine capacity every agent budget is derived from.
    Capacity,
    /// Show the limits a command would receive, without running it.
    ///
    /// `--shell` prints assignments for `eval`, which `scripts/agent-scope.sh`
    /// uses to build its `systemd-run` scope.
    Grant {
        /// Emit shell assignments instead of a table.
        #[arg(long, conflicts_with = "json")]
        shell: bool,
        /// Override the scope's `CPUQuota` (default: none, or `AGENT_CPU_QUOTA`).
        #[arg(long)]
        cpu_quota: Option<String>,
        /// Override the scope's `MemoryMax` (default: derived, or `AGENT_MEMORY_MAX`).
        #[arg(long)]
        memory_max: Option<String>,
        /// Join this slice instead of the default.
        #[arg(long, conflicts_with = "no_slice")]
        slice: Option<String>,
        /// Run without joining any shared slice.
        #[arg(long)]
        no_slice: bool,
        /// Apply the slice's transient limits (`systemctl set-property --runtime`).
        #[arg(long)]
        apply_slice: bool,
        #[arg(
            required = true,
            num_args = 1..,
            trailing_var_arg = true,
            allow_hyphen_values = true
        )]
        argv: Vec<OsString>,
    },
}

/// Execute `resources` and return the process exit code.
///
/// `cwd` is the repository root the usage is attributed to; `data_dir` holds
/// the usage store under `resources/`. A listing returns zero; `run` returns
/// the child's exit code.
pub async fn run(
    cwd: &Path,
    data_dir: &Path,
    repo: Option<&Path>,
    command: Option<&ResourcesCommand>,
    json: bool,
) -> Result<i32> {
    let store = UsageStore::new(data_dir.join("resources"));
    match command {
        None => {
            list(&store, repo, json)?;
            Ok(0)
        }
        Some(ResourcesCommand::Capacity) => {
            print_capacity(&limits_for(&store), json)?;
            Ok(0)
        }
        Some(ResourcesCommand::Grant {
            shell,
            cpu_quota,
            memory_max,
            slice,
            no_slice,
            apply_slice,
            argv,
        }) => {
            let overrides = GrantOverrides {
                cpu_quota: cpu_quota.as_deref(),
                memory_max: memory_max.as_deref(),
                slice: slice.as_deref(),
                no_slice: *no_slice,
            };
            grant(cwd, &store, &overrides, *apply_slice, *shell, json, argv)?;
            Ok(0)
        }
        Some(ResourcesCommand::Run { priority, argv }) => {
            run_command(cwd, &store, priority.unwrap_or(Priority::Attended), argv).await
        }
    }
}

async fn run_command(
    cwd: &Path,
    store: &UsageStore,
    priority: Priority,
    argv: &[OsString],
) -> Result<i32> {
    let (program, args) = argv.split_first().context("no command given")?;
    let workspace = SubagentWorkspace {
        path: cwd.to_path_buf(),
        node_id: String::new(),
        thread: String::new(),
        repo_root: cwd.to_path_buf(),
        subagent_id: String::new(),
    };
    // Default limits enforce the scope: if systemd cannot apply it, the command
    // fails instead of running unbounded. Admission goes through the same
    // ledger the agents use, so a short budget queues this command too.
    let limits = ResourceLimits::default()
        .with_usage_store(store.clone())
        .with_admission(Broker::new(store.dir().join("broker")))
        .with_priority(priority)
        .with_queue_observer(QueueObserver::new(|position| {
            eprintln!("queued at position {position}: waiting for the resource budget");
        }));
    let mut command = Command::new(program);
    command.args(args);
    let mut child = workspace
        .spawn_scoped(&mut command, limits, ScopedStdio::default())
        .await?;
    let status = child.wait().await.context("waiting for the command")?;
    Ok(exit_code(status))
}

/// Limits as configured by the environment; shared by `capacity` and `grant`.
fn limits_for(store: &UsageStore) -> ResourceLimits {
    ResourceLimits::default().with_usage_store(store.clone())
}

struct GrantOverrides<'a> {
    cpu_quota: Option<&'a str>,
    memory_max: Option<&'a str>,
    slice: Option<&'a str>,
    no_slice: bool,
}

fn grant(
    cwd: &Path,
    store: &UsageStore,
    overrides: &GrantOverrides<'_>,
    apply_slice: bool,
    shell: bool,
    json: bool,
    argv: &[OsString],
) -> Result<()> {
    let (program, args) = argv.split_first().context("no command given")?;
    let mut limits = limits_for(store);
    if let Some(quota) = overrides.cpu_quota {
        limits = limits.with_cpu_quota(quota);
    }
    if let Some(max) = overrides.memory_max {
        limits = limits.with_memory_max(max);
    }
    if overrides.no_slice {
        limits.slice = None;
    } else if let Some(slice) = overrides.slice {
        limits.slice = Some(slice.to_string());
    }
    let repo = cwd.to_string_lossy();
    let resolved = limits.resolve(program, args, &repo)?;
    if apply_slice {
        resolved.apply_slice()?;
    }
    let mut out = std::io::stdout().lock();
    if shell {
        write!(out, "{}", shell_assignments(&resolved))?;
    } else if json {
        serde_json::to_writer_pretty(&mut out, &resolved)?;
        writeln!(out)?;
    } else {
        write!(out, "{}", grant_table(&resolved))?;
    }
    Ok(())
}

/// Shell assignments consumed by `scripts/agent-scope.sh`. Every value is
/// single-quoted so no value can inject shell syntax.
fn shell_assignments(resolved: &ResolvedLimits) -> String {
    let quoted = |value: &str| format!("'{}'", value.replace('\'', "'\\''"));
    let mut text = format!(
        "GRANT_CARGO_JOBS={}\n",
        quoted(&resolved.cargo_jobs.to_string())
    );
    text.push_str(&format!(
        "GRANT_SLICE={}\n",
        quoted(resolved.slice.as_deref().unwrap_or(""))
    ));
    let properties: Vec<String> = resolved
        .scope_properties()
        .iter()
        .map(|property| quoted(property))
        .collect();
    text.push_str(&format!("GRANT_PROPERTIES=({})\n", properties.join(" ")));
    text
}

fn grant_table(resolved: &ResolvedLimits) -> String {
    let mut text = format!("cargo jobs:        {}\n", resolved.cargo_jobs);
    text.push_str(&format!(
        "slice:             {}\n",
        resolved.slice.as_deref().unwrap_or("(none)")
    ));
    text.push_str(&format!("memory from history: {}\n", resolved.from_history));
    for property in resolved.scope_properties() {
        text.push_str(&format!("  {property}\n"));
    }
    text
}

fn print_capacity(limits: &ResourceLimits, json: bool) -> Result<()> {
    let capacity = limits.capacity()?;
    let mut out = std::io::stdout().lock();
    if json {
        serde_json::to_writer_pretty(&mut out, &capacity)?;
        writeln!(out)?;
    } else {
        writeln!(out, "cores:            {}", capacity.cores)?;
        writeln!(
            out,
            "memory total:     {}",
            mebibytes(Some(capacity.mem_total_bytes))
        )?;
        writeln!(
            out,
            "session reserve:  {}",
            mebibytes(Some(capacity.reserved_bytes))
        )?;
        writeln!(
            out,
            "agent budget:     {}",
            mebibytes(Some(capacity.budget_bytes))
        )?;
        writeln!(out, "tasks max:        {}", capacity.tasks_max)?;
    }
    Ok(())
}

/// The shell convention: the code, or `128 + signal` for a signalled child.
fn exit_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        status.signal().map_or(1, |signal| 128 + signal)
    }
    #[cfg(not(unix))]
    {
        1
    }
}

fn list(store: &UsageStore, repo: Option<&Path>, json: bool) -> Result<()> {
    let repo_key = repo.map(|path| {
        path.canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .display()
            .to_string()
    });
    let records: Vec<UsageRecord> = store
        .read()
        .with_context(|| format!("reading {}", store.log_path().display()))?
        .into_iter()
        .filter(|record| repo_key.as_deref().is_none_or(|key| record.repo == key))
        .collect();
    let summaries = summarize(&records);

    let mut out = std::io::stdout().lock();
    if json {
        serde_json::to_writer_pretty(&mut out, &summaries)?;
        writeln!(out)?;
    } else if summaries.is_empty() {
        writeln!(
            out,
            "no resource usage recorded in {}",
            store.dir().display()
        )?;
    } else {
        write_table(&mut out, &summaries)?;
    }
    Ok(())
}

fn write_table(out: &mut impl std::io::Write, summaries: &[ClassSummary]) -> std::io::Result<()> {
    let mut rows = vec![
        [
            "CLASS", "RUNS", "ABORTED", "OOM", "MEM P50", "MEM P95", "CPU P50", "CPU P95",
            "WALL P50", "WALL P95",
        ]
        .map(str::to_string)
        .to_vec(),
    ];
    for summary in summaries {
        rows.push(vec![
            summary.class.clone(),
            summary.count.to_string(),
            summary.aborted.to_string(),
            summary.oom_kills.to_string(),
            mebibytes(summary.memory_peak_bytes.p50),
            mebibytes(summary.memory_peak_bytes.p95),
            cpu_seconds(summary.cpu_usec.p50),
            cpu_seconds(summary.cpu_usec.p95),
            wall_seconds(summary.wall_ms.p50),
            wall_seconds(summary.wall_ms.p95),
        ]);
    }
    let widths: Vec<usize> = (0..rows[0].len())
        .map(|column| rows.iter().map(|row| row[column].len()).max().unwrap_or(0))
        .collect();
    for row in &rows {
        let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(column, cell)| format!("{cell:<width$}", width = widths[column]))
            .collect();
        writeln!(out, "{}", cells.join("  ").trim_end())?;
    }
    Ok(())
}

fn mebibytes(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "-".to_string(),
        |bytes| format!("{:.0} MiB", bytes as f64 / 1_048_576.0),
    )
}

fn cpu_seconds(usec: Option<u64>) -> String {
    usec.map_or_else(
        || "-".to_string(),
        |usec| format!("{:.2} s", usec as f64 / 1_000_000.0),
    )
}

fn wall_seconds(ms: Option<u64>) -> String {
    ms.map_or_else(
        || "-".to_string(),
        |ms| format!("{:.2} s", ms as f64 / 1_000.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_missing_values_as_dashes() {
        assert_eq!(mebibytes(None), "-");
        assert_eq!(cpu_seconds(None), "-");
        assert_eq!(wall_seconds(None), "-");
        assert_eq!(mebibytes(Some(3 * 1_048_576)), "3 MiB");
        assert_eq!(cpu_seconds(Some(1_500_000)), "1.50 s");
        assert_eq!(wall_seconds(Some(2_250)), "2.25 s");
    }

    #[test]
    fn exit_code_follows_shell_convention_for_signals() {
        use std::os::unix::process::ExitStatusExt as _;
        assert_eq!(exit_code(ExitStatus::from_raw(3 << 8)), 3);
        assert_eq!(exit_code(ExitStatus::from_raw(9)), 128 + 9);
    }
}
