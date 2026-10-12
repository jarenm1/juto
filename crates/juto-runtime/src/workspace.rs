//! Workspace isolation and resource governance for Juto subagents.
//!
//! Provides an isolated native Juto workspace per subagent task (an independent
//! directory forked from an immutable live change-graph point, sharing only the
//! Git object store) and enforces resource caps (CPU, memory, swap) via systemd
//! user scopes (`agents.slice`) and serialized GPU access via advisory file
//! locks (`flock`).
//!
//! Architectural reference: `docs/orchestration-plan.md` §4 and `scripts/agent-scope.sh`.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::Arc,
};

use fs2::FileExt;
use serde::Deserialize;
use thiserror::Error;
pub use tokio::process::Command;

use crate::change_graph::ChangeGraph;
use crate::git_transfer::GitRepository;

/// Name of the native fork link marker written at the root of every fork workspace.
const WORKSPACE_MARKER: &str = ".juto-workspace.json";

/// The subset of the native link marker needed to validate workspace ownership.
#[derive(Debug, Deserialize)]
struct WorkspaceMarker {
    git_dir: PathBuf,
    thread: String,
}

/// Wrap a native VCS failure with its full context chain.
fn vcs_error(error: anyhow::Error) -> WorkspaceError {
    WorkspaceError::Vcs(format!("{error:#}"))
}

/// Read an environment variable, treating unset, empty, and whitespace-only
/// values as absent.
///
/// `scripts/agent-scope.sh` uses `${VAR:-default}`, so an exported-but-empty
/// variable means "unset". Reading it verbatim would forward an empty value to
/// `systemd-run` (e.g. `-p CPUQuota=` applies `infinity`, silently uncapping a
/// scope that claims `enforce_scope`).
fn env_var_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// Default CPU quota assigned to a single subagent scope.
pub const DEFAULT_CPU_QUOTA: &str = "400%";

/// Default maximum memory assigned to a single subagent scope.
pub const DEFAULT_MEMORY_MAX: &str = "8G";

/// Default memory swap max assigned to a single subagent scope (no swap).
pub const DEFAULT_MEMORY_SWAP_MAX: &str = "0";

/// Default shared systemd slice under which all agent scopes run.
pub const DEFAULT_SLICE: &str = "agents.slice";

/// Default aggregate CPU quota for the shared slice.
pub const DEFAULT_SLICE_CPU_QUOTA: &str = "600%";

/// Default aggregate memory max for the shared slice.
pub const DEFAULT_SLICE_MEMORY_MAX: &str = "16G";

/// Errors that can occur during workspace lifecycle and resource-scoped execution.
#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("native VCS error: {0}")]
    Vcs(String),

    #[error("directory is not inside a native Juto repository: {0}")]
    NotInRepo(PathBuf),

    #[error("command working directory `{cwd}` is outside assigned workspace `{workspace}`")]
    CwdEscapedWorkspace { cwd: PathBuf, workspace: PathBuf },

    #[error("systemd-run is unavailable; refusing uncapped execution")]
    SystemdUnavailable,

    #[error("failed to spawn scoped command `{command}`: {source}")]
    CommandSpawnFailed {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("scoped command failed: {0}")]
    ExecutionFailed(String),

    #[error("GPU lock failed on `{path}`: {source}")]
    GpuLockFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("workspace I/O error on `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("workspace error: {0}")]
    Other(String),
}

/// Resource constraints and governance parameters for subagent execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Per-scope CPU quota (e.g. "400%", "200%").
    pub cpu_quota: String,
    /// Per-scope memory limit (e.g. "8G", "4G").
    pub memory_max: String,
    /// Per-scope swap limit (e.g. "0").
    pub memory_swap_max: String,
    /// Shared systemd slice under which scopes run (e.g. "agents.slice").
    pub slice: Option<String>,
    /// Aggregate CPU quota for the shared slice (e.g. "600%").
    pub slice_cpu_quota: Option<String>,
    /// Aggregate memory limit for the shared slice (e.g. "16G").
    pub slice_memory_max: Option<String>,
    /// Whether to serialize execution behind a shared GPU flock.
    pub gpu: bool,
    /// Path to the shared GPU file lock.
    pub gpu_lock: Option<PathBuf>,
    /// When true, fails if `systemd-run` is not available rather than running uncapped.
    pub enforce_scope: bool,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let cpu_quota =
            env_var_nonempty("AGENT_CPU_QUOTA").unwrap_or_else(|| DEFAULT_CPU_QUOTA.to_string());
        let memory_max =
            env_var_nonempty("AGENT_MEMORY_MAX").unwrap_or_else(|| DEFAULT_MEMORY_MAX.to_string());
        let slice = env_var_nonempty("AGENT_SLICE").or_else(|| Some(DEFAULT_SLICE.to_string()));
        let slice_cpu_quota = env_var_nonempty("AGENT_SLICE_CPU_QUOTA")
            .or_else(|| Some(DEFAULT_SLICE_CPU_QUOTA.to_string()));
        let slice_memory_max = env_var_nonempty("AGENT_SLICE_MEMORY_MAX")
            .or_else(|| Some(DEFAULT_SLICE_MEMORY_MAX.to_string()));
        let gpu_lock = env_var_nonempty("AGENT_GPU_LOCK")
            .map(PathBuf::from)
            .or_else(|| Some(default_gpu_lock_path()));

        Self {
            cpu_quota,
            memory_max,
            memory_swap_max: DEFAULT_MEMORY_SWAP_MAX.to_string(),
            slice,
            slice_cpu_quota,
            slice_memory_max,
            gpu: false,
            gpu_lock,
            enforce_scope: true,
        }
    }
}

impl ResourceLimits {
    /// Enable GPU locking on this limit set.
    pub fn with_gpu(mut self, gpu: bool) -> Self {
        self.gpu = gpu;
        self
    }

    /// Set explicit per-scope CPU quota.
    pub fn with_cpu_quota(mut self, quota: impl Into<String>) -> Self {
        self.cpu_quota = quota.into();
        self
    }

    /// Set explicit per-scope memory ceiling.
    pub fn with_memory_max(mut self, mem: impl Into<String>) -> Self {
        self.memory_max = mem.into();
        self
    }

    /// Configure whether systemd scope enforcement is mandatory.
    pub fn with_enforce_scope(mut self, enforce: bool) -> Self {
        self.enforce_scope = enforce;
        self
    }
}

/// RAII guard holding an advisory exclusive flock on the GPU lock file.
pub struct GpuLockGuard {
    _file: Arc<std::fs::File>,
    path: PathBuf,
}

impl GpuLockGuard {
    /// Acquire an exclusive advisory flock on the target path, blocking until available.
    pub fn acquire(path: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|source| WorkspaceError::GpuLockFailed {
                path: path.clone(),
                source,
            })?;

        file.lock_exclusive()
            .map_err(|source| WorkspaceError::GpuLockFailed {
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            _file: Arc::new(file),
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for GpuLockGuard {
    fn drop(&mut self) {
        let _ = self._file.unlock();
    }
}

/// Stdio configuration applied to a resource-scoped command.
///
/// `std::process::Command` does not expose its configured stdio on stable, so a
/// scoped command cannot inherit the original command's streams; callers pass
/// what they need explicitly. [`Default`] inherits the parent's streams, which
/// matches `std::process::Command`'s own default.
#[derive(Debug)]
pub struct ScopedStdio {
    pub stdin: Stdio,
    pub stdout: Stdio,
    pub stderr: Stdio,
}

impl Default for ScopedStdio {
    fn default() -> Self {
        Self {
            stdin: Stdio::inherit(),
            stdout: Stdio::inherit(),
            stderr: Stdio::inherit(),
        }
    }
}

/// An isolated native Juto workspace dedicated to a single subagent task.
///
/// The workspace is an independent directory forked from an immutable live
/// change-graph point. It shares only the parent repository's Git object store
/// through the native link marker, so its edits never touch the parent working
/// copy and its history stays in the shared DAG.
#[derive(Debug, Clone)]
pub struct SubagentWorkspace {
    /// Filesystem path to the subagent's dedicated working copy directory.
    pub path: PathBuf,
    /// Native change-graph node ID this workspace was forked from.
    pub node_id: String,
    /// Native thread name assigned to this workspace (the exact subagent id).
    pub thread: String,
    /// Root path of the parent native repository.
    pub repo_root: PathBuf,
    /// Subagent identifier.
    pub subagent_id: String,
}

impl SubagentWorkspace {
    /// Provision an isolated native workspace from `base_change` for `subagent_id`.
    ///
    /// Locates the enclosing native repository from the current directory (or
    /// `JUTO_REPO_ROOT` if set) and creates `~/workspaces/juto-<subagent-id>` (or
    /// under `JUTO_WORKSPACES_DIR` if set).
    pub async fn provision(base_change: &str, subagent_id: &str) -> Result<Self, WorkspaceError> {
        let repo_root = if let Some(root) = env_var_nonempty("JUTO_REPO_ROOT") {
            PathBuf::from(root)
        } else {
            let cwd = std::env::current_dir().map_err(|source| WorkspaceError::Io {
                path: PathBuf::from("."),
                source,
            })?;
            find_repo_root(&cwd).ok_or(WorkspaceError::NotInRepo(cwd))?
        };

        let workspaces_dir = default_workspaces_dir();
        let sanitized_id = sanitize_workspace_slug(subagent_id);
        let workspace_path = workspaces_dir.join(format!("juto-{}", sanitized_id));

        Self::provision_in(&repo_root, &workspace_path, base_change, subagent_id).await
    }

    /// Provision an isolated native workspace at an explicit filesystem path.
    ///
    /// `base_change` is a native change-graph node ID or checkpoint name. The
    /// fork materializes that immutable tree into a brand-new directory and
    /// links it back to the shared object store; no Git branch, worktree, or
    /// commit is created. All Git work runs on a blocking thread.
    pub async fn provision_in(
        repo_root: &Path,
        workspace_path: &Path,
        base_change: &str,
        subagent_id: &str,
    ) -> Result<Self, WorkspaceError> {
        let repo_root = repo_root.to_path_buf();
        let workspace_path = workspace_path.to_path_buf();
        let base_change = base_change.to_owned();
        let subagent_id = subagent_id.to_owned();
        tokio::task::spawn_blocking(move || -> Result<Self, WorkspaceError> {
            let graph = ChangeGraph::open(&repo_root).map_err(vcs_error)?;
            let state = graph
                .fork(&subagent_id, &base_change, &workspace_path)
                .map_err(vcs_error)?;
            let node_id = state
                .heads
                .get(&subagent_id)
                .cloned()
                .unwrap_or_else(|| base_change.clone());
            Ok(Self {
                path: workspace_path,
                node_id,
                thread: subagent_id.clone(),
                repo_root,
                subagent_id,
            })
        })
        .await
        .map_err(|source| {
            WorkspaceError::Other(format!("workspace provisioning task failed: {source}"))
        })?
    }

    /// Remove this workspace's owned directory after verifying its native link
    /// marker still names this thread and repository.
    ///
    /// The shared DAG keeps every node this workspace produced: cleanup only
    /// deletes the fork directory, never history. A missing or foreign marker
    /// (different thread or repository) is refused so a directory this
    /// workspace does not own is never removed.
    pub async fn cleanup(self) -> Result<(), WorkspaceError> {
        let path = self.path.clone();
        let thread = self.thread.clone();
        let repo_root = self.repo_root.clone();
        tokio::task::spawn_blocking(move || -> Result<(), WorkspaceError> {
            let marker_path = path.join(WORKSPACE_MARKER);
            let raw = std::fs::read(&marker_path).map_err(|source| WorkspaceError::Io {
                path: marker_path.clone(),
                source,
            })?;
            let marker: WorkspaceMarker = serde_json::from_slice(&raw).map_err(|error| {
                WorkspaceError::Other(format!(
                    "invalid native workspace marker {}: {error}",
                    marker_path.display()
                ))
            })?;
            if marker.thread != thread {
                return Err(WorkspaceError::Other(format!(
                    "workspace marker thread {:?} does not match {:?}",
                    marker.thread, thread
                )));
            }
            let expected = GitRepository::open(&repo_root)
                .map_err(vcs_error)?
                .git_dir()
                .to_path_buf();
            let actual = marker
                .git_dir
                .canonicalize()
                .map_err(|source| WorkspaceError::Io {
                    path: marker.git_dir.clone(),
                    source,
                })?;
            if actual != expected {
                return Err(WorkspaceError::Other(format!(
                    "workspace marker repository {} does not match {}",
                    actual.display(),
                    expected.display()
                )));
            }
            if path.exists() {
                std::fs::remove_dir_all(&path).map_err(|source| WorkspaceError::Io {
                    path: path.clone(),
                    source,
                })?;
            }
            Ok(())
        })
        .await
        .map_err(|source| {
            WorkspaceError::Other(format!("workspace cleanup task failed: {source}"))
        })?
    }

    /// Run a command inside this workspace under resource governance.
    ///
    /// Wraps execution in `systemd-run --user --scope` with CPU, memory, and swap caps,
    /// and optionally serializes behind `flock` when `limits.gpu` is enabled.
    ///
    /// Convenience wrapper over [`Self::spawn_scoped`] with inherited stdio.
    /// Callers that need to capture output must use `spawn_scoped` directly:
    /// `std::process::Command` does not expose its configured stdio on stable,
    /// so the scoped command cannot inherit the original command's streams.
    pub async fn run_scoped(
        &self,
        cmd: &mut Command,
        limits: ResourceLimits,
    ) -> Result<ExitStatus, WorkspaceError> {
        let mut child = self
            .spawn_scoped(cmd, limits, ScopedStdio::default())
            .await?;

        child
            .wait()
            .await
            .map_err(|source| WorkspaceError::ExecutionFailed(source.to_string()))
    }

    /// Spawn a command inside this workspace under resource governance.
    ///
    /// Returns the running child so the caller can read piped output and control
    /// its lifetime. The child is spawned with `kill_on_drop`, so dropping it
    /// (or cancelling the awaiting future) terminates the process instead of
    /// leaking it.
    pub async fn spawn_scoped(
        &self,
        cmd: &mut Command,
        limits: ResourceLimits,
        stdio: ScopedStdio,
    ) -> Result<tokio::process::Child, WorkspaceError> {
        let mut scoped = self.build_scoped_command(cmd.as_std(), &limits)?;
        scoped
            .stdin(stdio.stdin)
            .stdout(stdio.stdout)
            .stderr(stdio.stderr)
            .kill_on_drop(true);

        scoped
            .spawn()
            .map_err(|source| WorkspaceError::CommandSpawnFailed {
                command: format!("{:?}", cmd.as_std().get_program()),
                source,
            })
    }

    /// Run a synchronous `std::process::Command` inside this workspace under resource governance.
    ///
    /// Convenience wrapper over [`Self::spawn_scoped_blocking`] with inherited
    /// stdio.
    pub fn run_scoped_blocking(
        &self,
        cmd: &mut std::process::Command,
        limits: ResourceLimits,
    ) -> Result<ExitStatus, WorkspaceError> {
        let mut child = self.spawn_scoped_blocking(cmd, limits, ScopedStdio::default())?;

        child
            .wait()
            .map_err(|source| WorkspaceError::ExecutionFailed(source.to_string()))
    }

    /// Spawn a synchronous `std::process::Command` inside this workspace under
    /// resource governance, returning the running child.
    pub fn spawn_scoped_blocking(
        &self,
        cmd: &mut std::process::Command,
        limits: ResourceLimits,
        stdio: ScopedStdio,
    ) -> Result<std::process::Child, WorkspaceError> {
        let mut scoped = self.build_scoped_std_command(cmd, &limits)?;
        scoped
            .stdin(stdio.stdin)
            .stdout(stdio.stdout)
            .stderr(stdio.stderr);

        scoped
            .spawn()
            .map_err(|source| WorkspaceError::CommandSpawnFailed {
                command: format!("{:?}", cmd.get_program()),
                source,
            })
    }

    /// Build a resource-scoped `tokio::process::Command` from an existing `std::process::Command`.
    pub fn build_scoped_command(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
    ) -> Result<Command, WorkspaceError> {
        let std_cmd = self.build_scoped_std_command(original, limits)?;
        Ok(Command::from(std_cmd))
    }

    /// Build a resource-scoped `std::process::Command` from an existing `std::process::Command`.
    pub fn build_scoped_std_command(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
    ) -> Result<std::process::Command, WorkspaceError> {
        let program = original.get_program().to_os_string();
        let orig_args: Vec<OsString> = original.get_args().map(|s| s.to_os_string()).collect();
        let work_dir = validate_scoped_cwd(&self.path, original.get_current_dir())?;

        let systemd_available = is_systemd_run_available();
        if !systemd_available && limits.enforce_scope {
            return Err(WorkspaceError::SystemdUnavailable);
        }

        // Configure shared slice properties if needed
        if systemd_available && let Some(slice) = &limits.slice {
            let cpu_quota = limits
                .slice_cpu_quota
                .as_deref()
                .unwrap_or(DEFAULT_SLICE_CPU_QUOTA);
            let mem_max = limits
                .slice_memory_max
                .as_deref()
                .unwrap_or(DEFAULT_SLICE_MEMORY_MAX);
            configure_slice_properties(slice, cpu_quota, mem_max);
        }

        // Derive CARGO_BUILD_JOBS from cpu_quota if not already present
        let jobs = calculate_cargo_build_jobs(&limits.cpu_quota);

        let mut cmd: std::process::Command;

        if limits.gpu {
            let gpu_lock = limits
                .gpu_lock
                .clone()
                .unwrap_or_else(default_gpu_lock_path);
            // The wrapper's cwd is the workspace, so a relative lock path would
            // resolve inside each agent's own workspace and two agents would
            // lock different files. Resolve it against the caller's process cwd,
            // matching `GpuLockGuard::acquire`.
            let gpu_lock = if gpu_lock.is_absolute() {
                gpu_lock
            } else {
                let caller_cwd = std::env::current_dir().map_err(|source| WorkspaceError::Io {
                    path: PathBuf::from("."),
                    source,
                })?;
                caller_cwd.join(&gpu_lock)
            };
            if let Some(parent) = gpu_lock.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            cmd = std::process::Command::new("flock");
            cmd.arg(&gpu_lock);

            if systemd_available {
                cmd.arg("systemd-run");
                cmd.arg("--user");
                cmd.arg("--scope");
                cmd.arg("--quiet");
                cmd.arg("-p").arg(format!("CPUQuota={}", limits.cpu_quota));
                cmd.arg("-p")
                    .arg(format!("MemoryMax={}", limits.memory_max));
                cmd.arg("-p")
                    .arg(format!("MemorySwapMax={}", limits.memory_swap_max));

                if let Some(slice) = &limits.slice {
                    cmd.arg(format!("--slice={}", slice));
                }

                cmd.arg("--");
                cmd.arg(&program);
                cmd.args(&orig_args);
            } else {
                cmd.arg(&program);
                cmd.args(&orig_args);
            }
        } else if systemd_available {
            cmd = std::process::Command::new("systemd-run");
            cmd.arg("--user");
            cmd.arg("--scope");
            cmd.arg("--quiet");
            cmd.arg("-p").arg(format!("CPUQuota={}", limits.cpu_quota));
            cmd.arg("-p")
                .arg(format!("MemoryMax={}", limits.memory_max));
            cmd.arg("-p")
                .arg(format!("MemorySwapMax={}", limits.memory_swap_max));

            if let Some(slice) = &limits.slice {
                cmd.arg(format!("--slice={}", slice));
            }

            cmd.arg("--");
            cmd.arg(&program);
            cmd.args(&orig_args);
        } else {
            cmd = std::process::Command::new(&program);
            cmd.args(&orig_args);
        }

        // Set working directory
        cmd.current_dir(work_dir);

        // Forward and inherit all original environment variables
        for (key, val) in original.get_envs() {
            if let Some(val) = val {
                cmd.env(key, val);
            } else {
                cmd.env_remove(key);
            }
        }

        // Set CARGO_BUILD_JOBS if not explicitly defined
        let has_cargo_jobs = original
            .get_envs()
            .any(|(k, v)| k == "CARGO_BUILD_JOBS" && v.is_some())
            || std::env::var_os("CARGO_BUILD_JOBS").is_some();

        if !has_cargo_jobs {
            cmd.env("CARGO_BUILD_JOBS", jobs.to_string());
        }

        Ok(cmd)
    }
}

/// Calculate the recommended `CARGO_BUILD_JOBS` from a CPU quota string (e.g. "400%" -> 4).
pub fn calculate_cargo_build_jobs(cpu_quota: &str) -> usize {
    let digits = cpu_quota.trim().trim_end_matches('%');
    if let Ok(pct) = digits.parse::<usize>() {
        let jobs = pct / 100;
        jobs.max(1)
    } else {
        1
    }
}

/// Check if `systemd-run` is available in PATH.
pub fn is_systemd_run_available() -> bool {
    std::process::Command::new("systemd-run")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Locate the nearest ancestor that is a native Juto workspace root.
///
/// A root is either an ordinary Git worktree (`.git` present) or an independent
/// native fork directory (`.juto-workspace.json` marker present).
pub fn find_repo_root(start_dir: &Path) -> Option<PathBuf> {
    let mut curr = start_dir
        .canonicalize()
        .ok()
        .or_else(|| Some(start_dir.to_path_buf()))?;
    loop {
        if curr.join(".git").exists() || curr.join(WORKSPACE_MARKER).exists() {
            return Some(curr);
        }
        if !curr.pop() {
            break;
        }
    }
    None
}

/// Returns the default directory where subagent workspaces are placed (`~/workspaces`).
pub fn default_workspaces_dir() -> PathBuf {
    if let Some(dir) = env_var_nonempty("JUTO_WORKSPACES_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = env_var_nonempty("HOME") {
        return PathBuf::from(home).join("workspaces");
    }
    PathBuf::from("/tmp/workspaces")
}

/// Returns the default GPU lock file path (`/tmp/jtech-agent-gpu.lock` or XDG_RUNTIME_DIR).
pub fn default_gpu_lock_path() -> PathBuf {
    if let Some(path) = env_var_nonempty("AGENT_GPU_LOCK") {
        return PathBuf::from(path);
    }
    if let Some(xdg) = env_var_nonempty("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("jtech-agent-gpu.lock");
    }
    PathBuf::from("/tmp/jtech-agent-gpu.lock")
}

/// Sanitize an arbitrary subagent identifier into a safe workspace directory slug.
fn sanitize_workspace_slug(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Validate that a requested working directory does not escape the assigned workspace boundary.
///
/// The requested directory is resolved physically: symlinks and `..` are
/// followed by the kernel at `chdir`, so a lexical check alone is not enough.
/// `out/missing/..` collapses lexically to `out`, which may be a symlink
/// pointing outside the workspace; the canonicalized result is therefore
/// re-checked against the canonical workspace root. A directory that cannot be
/// canonicalized (for example one that does not exist) is rejected rather than
/// silently accepted as a lexical path, because the kernel would fail the
/// `chdir` anyway and a lexical path can still hide an escaping symlink.
pub fn validate_scoped_cwd(
    workspace: &Path,
    requested_cwd: Option<&Path>,
) -> Result<PathBuf, WorkspaceError> {
    let ws_canon = workspace
        .canonicalize()
        .unwrap_or_else(|_| normalize_path(workspace));

    let target = match requested_cwd {
        None => ws_canon,
        Some(dir) => {
            let resolved = if dir.is_absolute() {
                dir.to_path_buf()
            } else {
                ws_canon.join(dir)
            };
            match resolved.canonicalize() {
                // The workspace itself may be reached through symlinked
                // parents, so containment is checked against the canonical
                // result rather than the lexical path.
                Ok(canon) => {
                    if !canon.starts_with(&ws_canon) {
                        return Err(WorkspaceError::CwdEscapedWorkspace {
                            cwd: dir.to_path_buf(),
                            workspace: workspace.to_path_buf(),
                        });
                    }
                    canon
                }
                Err(source) => {
                    // Unresolvable: keep the dedicated error for obvious
                    // lexical traversal, and reject everything else instead of
                    // falling back to a lexical path.
                    if !normalize_path(&resolved).starts_with(&ws_canon) {
                        return Err(WorkspaceError::CwdEscapedWorkspace {
                            cwd: dir.to_path_buf(),
                            workspace: workspace.to_path_buf(),
                        });
                    }
                    return Err(WorkspaceError::Io {
                        path: resolved,
                        source,
                    });
                }
            }
        }
    };
    Ok(target)
}

/// Lexically normalize a path resolving `.` and `..` components.
pub fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut normalized = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            other => normalized.push(other),
        }
    }
    normalized
}

/// Best-effort systemd slice property initialization matching `scripts/agent-scope.sh`.
fn configure_slice_properties(slice: &str, cpu_quota: &str, mem_max: &str) {
    let mut props: Vec<String> = Vec::new();

    let cpu_now = std::process::Command::new("systemctl")
        .arg("--user")
        .arg("show")
        .arg(slice)
        .arg("-p")
        .arg("CPUQuotaPerSecUSec")
        .arg("--value")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    if cpu_now.is_empty() || cpu_now == "infinity" {
        props.push(format!("CPUQuota={}", cpu_quota));
    }

    let mem_now = std::process::Command::new("systemctl")
        .arg("--user")
        .arg("show")
        .arg(slice)
        .arg("-p")
        .arg("MemoryMax")
        .arg("--value")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    if mem_now.is_empty() || mem_now == "infinity" {
        props.push(format!("MemoryMax={}", mem_max));
    }

    if !props.is_empty() {
        let mut cmd = std::process::Command::new("systemctl");
        cmd.arg("--user").arg("set-property").arg(slice);
        for prop in props {
            cmd.arg(prop);
        }
        let _ = cmd.output();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_graph::{ChangeGraph, EditProvenance};
    use tempfile::tempdir;

    /// Serializes tests that mutate process-global environment variables.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Clear the named environment variables for the guard's lifetime.
    ///
    /// Callers must hold [`ENV_LOCK`] for as long as the guard lives.
    struct EnvGuard(Vec<&'static str>);

    impl EnvGuard {
        fn clear(vars: &[&'static str]) -> Self {
            for var in vars {
                // SAFETY: callers hold ENV_LOCK for the guard's lifetime.
                unsafe { std::env::remove_var(var) };
            }
            Self(vars.to_vec())
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for var in &self.0 {
                // SAFETY: callers hold ENV_LOCK for the guard's lifetime.
                unsafe { std::env::remove_var(var) };
            }
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

    #[test]
    fn test_resource_limits_defaults_and_builder() {
        let _guard = env_lock();
        let _env = EnvGuard::clear(&[
            "AGENT_CPU_QUOTA",
            "AGENT_MEMORY_MAX",
            "AGENT_SLICE",
            "AGENT_SLICE_CPU_QUOTA",
            "AGENT_SLICE_MEMORY_MAX",
            "AGENT_GPU_LOCK",
        ]);
        let limits = ResourceLimits::default();
        assert_eq!(limits.cpu_quota, DEFAULT_CPU_QUOTA);
        assert_eq!(limits.memory_max, DEFAULT_MEMORY_MAX);
        assert_eq!(limits.memory_swap_max, DEFAULT_MEMORY_SWAP_MAX);
        assert_eq!(limits.slice.as_deref(), Some(DEFAULT_SLICE));
        assert!(!limits.gpu);
        assert!(limits.enforce_scope);

        let customized = limits
            .with_cpu_quota("200%")
            .with_memory_max("4G")
            .with_gpu(true)
            .with_enforce_scope(false);

        assert_eq!(customized.cpu_quota, "200%");
        assert_eq!(customized.memory_max, "4G");
        assert!(customized.gpu);
        assert!(!customized.enforce_scope);
    }

    #[test]
    fn test_cargo_build_jobs_calculation() {
        assert_eq!(calculate_cargo_build_jobs("400%"), 4);
        assert_eq!(calculate_cargo_build_jobs("200%"), 2);
        assert_eq!(calculate_cargo_build_jobs("100%"), 1);
        assert_eq!(calculate_cargo_build_jobs("50%"), 1);
        assert_eq!(calculate_cargo_build_jobs("0%"), 1);
        assert_eq!(calculate_cargo_build_jobs("invalid"), 1);
    }

    #[test]
    fn test_workspace_slug_sanitization() {
        assert_eq!(sanitize_workspace_slug("builder-1"), "builder-1");
        assert_eq!(
            sanitize_workspace_slug("subagent/task:42"),
            "subagent-task-42"
        );
        assert_eq!(
            sanitize_workspace_slug("worker_alpha.run"),
            "worker_alpha-run"
        );
    }

    #[test]
    fn test_find_repo_root_detects_git_and_native_markers() {
        let dir = tempdir().expect("tempdir");
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(repo.join("nested/deep")).expect("create dirs");
        std::fs::create_dir_all(repo.join(".git")).expect("create .git");
        assert_eq!(
            find_repo_root(&repo.join("nested/deep")),
            Some(repo.canonicalize().unwrap())
        );

        let fork = dir.path().join("fork");
        std::fs::create_dir_all(&fork).expect("create fork");
        std::fs::write(fork.join(".juto-workspace.json"), "{}").expect("write marker");
        assert_eq!(find_repo_root(&fork), Some(fork.canonicalize().unwrap()));
    }

    #[test]
    fn test_gpu_flock_lock_mutual_exclusion() {
        let dir = tempdir().expect("tempdir");
        let lock_file = dir.path().join("gpu.lock");

        let guard1 = GpuLockGuard::acquire(&lock_file).expect("acquire guard 1");
        assert_eq!(guard1.path(), lock_file);

        // Attempting a non-blocking lock on the same file should fail while guard1 is held
        let file2 = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_file)
            .expect("open second handle");

        let try_res = file2.try_lock_exclusive();
        assert!(try_res.is_err(), "exclusive lock should be held by guard1");

        // Once guard1 is dropped, lock can be acquired
        drop(guard1);
        let acquired2 = file2.try_lock_exclusive();
        assert!(acquired2.is_ok(), "lock should be free after drop");
        let _ = file2.unlock();
    }

    #[test]
    fn test_scoped_command_construction() {
        let ws = SubagentWorkspace {
            path: PathBuf::from("/tmp/test-ws"),
            node_id: "test-node".to_string(),
            thread: "test-ws".to_string(),
            repo_root: PathBuf::from("/tmp/test-repo"),
            subagent_id: "test-agent".to_string(),
        };

        let mut orig = std::process::Command::new("cargo");
        orig.arg("build").arg("--release");

        let limits = ResourceLimits::default()
            .with_cpu_quota("200%")
            .with_memory_max("4G");

        let scoped = ws
            .build_scoped_std_command(&orig, &limits)
            .expect("build scoped");

        if is_systemd_run_available() {
            assert_eq!(scoped.get_program(), "systemd-run");
            let args: Vec<String> = scoped
                .get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();

            assert!(
                args.contains(&"CPUQuota=200%".to_string())
                    || args.iter().any(|a| a.contains("CPUQuota=200%"))
            );
            assert!(
                args.contains(&"MemoryMax=4G".to_string())
                    || args.iter().any(|a| a.contains("MemoryMax=4G"))
            );
            assert!(args.contains(&"--slice=agents.slice".to_string()));
            assert!(args.contains(&"--".to_string()));
            assert!(args.contains(&"cargo".to_string()));
            assert!(args.contains(&"build".to_string()));
            assert!(args.contains(&"--release".to_string()));
        }
    }

    #[test]
    fn test_scoped_command_with_gpu_construction() {
        let ws = SubagentWorkspace {
            path: PathBuf::from("/tmp/test-ws"),
            node_id: "test-node".to_string(),
            thread: "test-ws".to_string(),
            repo_root: PathBuf::from("/tmp/test-repo"),
            subagent_id: "test-agent".to_string(),
        };

        let mut orig = std::process::Command::new("cargo");
        orig.arg("run");

        let limits = ResourceLimits::default()
            .with_gpu(true)
            .with_cpu_quota("400%");

        let scoped = ws
            .build_scoped_std_command(&orig, &limits)
            .expect("build scoped with gpu");
        assert_eq!(scoped.get_program(), "flock");

        let args: Vec<String> = scoped
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(!args.is_empty());
        // First arg is the lock path
        assert!(args[0].contains("gpu.lock"));
    }

    #[tokio::test]
    async fn test_run_scoped_execution() {
        let dir = tempdir().expect("tempdir");
        let ws = SubagentWorkspace {
            path: dir.path().to_path_buf(),
            node_id: "test-node".to_string(),
            thread: "test-ws".to_string(),
            repo_root: dir.path().to_path_buf(),
            subagent_id: "test-agent".to_string(),
        };

        let limits = ResourceLimits::default()
            .with_cpu_quota("200%")
            .with_memory_max("1G");

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("echo 'scoped hello'");

        let status = ws.run_scoped(&mut cmd, limits).await.expect("run_scoped");
        assert!(status.success());
    }

    /// Initialize a plain Git repository with a deterministic identity and one
    /// base commit.
    fn native_repo(dir: &Path) {
        git(dir, &["init", "-q"]);
        git(dir, &["config", "user.name", "Workspace Test"]);
        git(dir, &["config", "user.email", "workspace@example.test"]);
        std::fs::write(dir.join("README.md"), "# Initial\n").expect("write base file");
        git(dir, &["add", "-A"]);
        git(dir, &["commit", "-q", "-m", "initial"]);
    }

    fn git(dir: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[tokio::test]
    async fn test_native_workspace_lifecycle_end_to_end() {
        let test_root = tempdir().expect("tempdir");
        let repo_dir = test_root.path().join("repo");
        std::fs::create_dir_all(&repo_dir).expect("create repo dir");
        native_repo(&repo_dir);

        // Capture an immutable live point on the main thread.
        let graph = ChangeGraph::open(&repo_dir).expect("open graph");
        let source = graph
            .capture(
                "main",
                EditProvenance {
                    session_id: "workspace-test".to_string(),
                    tool: "fork".to_string(),
                    selection_mode: None,
                    path: None,
                },
                None,
            )
            .expect("capture source point");

        let ws_dir = test_root.path().join("subagent-ws");
        let ws = SubagentWorkspace::provision_in(&repo_dir, &ws_dir, &source.id, "test-subagent")
            .await
            .expect("provision workspace");

        // The fork materialized the source tree and linked back to the store.
        assert!(ws.path.exists());
        assert_eq!(ws.path, ws_dir);
        assert_eq!(ws.node_id, source.id);
        assert_eq!(ws.thread, "test-subagent");
        assert_eq!(ws.subagent_id, "test-subagent");
        assert_eq!(
            std::fs::read_to_string(ws_dir.join("README.md")).unwrap(),
            "# Initial\n"
        );
        assert!(ws_dir.join(".juto-workspace.json").exists());

        // The fork is independent: editing it never touches the parent.
        std::fs::write(ws.path.join("README.md"), "# Modified by subagent\n").expect("modify");
        assert_eq!(
            std::fs::read_to_string(repo_dir.join("README.md")).unwrap(),
            "# Initial\n"
        );

        // Run a scoped command inside the workspace.
        let limits = ResourceLimits::default().with_cpu_quota("200%");
        let mut test_cmd = Command::new("sh");
        test_cmd.arg("-c").arg("test -f README.md");
        assert!(
            ws.run_scoped(&mut test_cmd, limits)
                .await
                .expect("scoped command")
                .success()
        );

        // Cleanup removes the owned directory; the shared DAG keeps its history.
        ws.cleanup().await.expect("cleanup");
        assert!(!ws_dir.exists(), "workspace directory should be removed");
        let state = ChangeGraph::open(&repo_dir)
            .expect("reopen graph")
            .state()
            .expect("state");
        assert!(state.heads.contains_key("test-subagent"));
        assert!(state.nodes.iter().any(|node| node.id == source.id));
    }

    #[tokio::test]
    async fn test_systemd_run_non_zero_exit_code() {
        let dir = tempdir().expect("tempdir");
        let ws = SubagentWorkspace {
            path: dir.path().to_path_buf(),
            node_id: "test-node".to_string(),
            thread: "test-ws".to_string(),
            repo_root: dir.path().to_path_buf(),
            subagent_id: "test-agent".to_string(),
        };

        let limits = ResourceLimits::default()
            .with_cpu_quota("100%")
            .with_memory_max("512M");

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("exit 7");

        let status = ws.run_scoped(&mut cmd, limits).await.expect("run_scoped");
        assert_eq!(status.code(), Some(7));
    }

    #[tokio::test]
    async fn test_systemd_run_env_propagation() {
        let dir = tempdir().expect("tempdir");
        let ws = SubagentWorkspace {
            path: dir.path().to_path_buf(),
            node_id: "test-node".to_string(),
            thread: "test-ws".to_string(),
            repo_root: dir.path().to_path_buf(),
            subagent_id: "test-agent".to_string(),
        };

        let limits = ResourceLimits::default();

        let mut cmd = Command::new("sh");
        cmd.env("CUSTOM_SUBAGENT_VAR", "injected_value_42");
        cmd.arg("-c")
            .arg("test \"$CUSTOM_SUBAGENT_VAR\" = \"injected_value_42\"");

        let status = ws.run_scoped(&mut cmd, limits).await.expect("run_scoped");
        assert!(status.success());
    }

    #[tokio::test]
    async fn test_workspace_not_in_repo_error() {
        let non_repo = tempdir().expect("tempdir");
        let ws_dir = non_repo.path().join("ws");

        let result = SubagentWorkspace::provision_in(
            non_repo.path(),
            &ws_dir,
            "0".repeat(40).as_str(),
            "agent-fail",
        )
        .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            WorkspaceError::Vcs(message) => {
                assert!(!message.is_empty(), "native VCS failure must be honest");
            }
            other => panic!("expected Vcs error, got {other:?}"),
        }
        assert!(!ws_dir.exists(), "a failed fork must not leave a directory");
    }

    #[test]
    fn test_scoped_cwd_escape_rejected() {
        let dir = tempdir().expect("tempdir");
        let ws_path = dir.path().join("subagent-ws");
        std::fs::create_dir_all(ws_path.join("subdir")).expect("create subdir");

        let ws = SubagentWorkspace {
            path: ws_path.clone(),
            node_id: "test-node".to_string(),
            thread: "subagent-ws".to_string(),
            repo_root: dir.path().to_path_buf(),
            subagent_id: "test-agent".to_string(),
        };

        let limits = ResourceLimits::default();

        // 1. Subdirectory within workspace is accepted
        let mut valid_cmd = std::process::Command::new("ls");
        valid_cmd.current_dir(ws_path.join("subdir"));
        let scoped_valid = ws.build_scoped_std_command(&valid_cmd, &limits);
        assert!(scoped_valid.is_ok());

        // 2. Relative traversal outside workspace is rejected
        let mut escape_rel = std::process::Command::new("ls");
        escape_rel.current_dir(ws_path.join("../outside"));
        let scoped_rel = ws.build_scoped_std_command(&escape_rel, &limits);
        assert!(scoped_rel.is_err());
        match scoped_rel.unwrap_err() {
            WorkspaceError::CwdEscapedWorkspace { .. } => {}
            other => panic!("expected CwdEscapedWorkspace, got {:?}", other),
        }

        // 3. Absolute path outside workspace is rejected
        let mut escape_abs = std::process::Command::new("ls");
        escape_abs.current_dir(dir.path());
        let scoped_abs = ws.build_scoped_std_command(&escape_abs, &limits);
        assert!(scoped_abs.is_err());
        match scoped_abs.unwrap_err() {
            WorkspaceError::CwdEscapedWorkspace { .. } => {}
            other => panic!("expected CwdEscapedWorkspace, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cleanup_rejects_foreign_marker_and_preserves_directory() {
        let dir = tempdir().expect("tempdir");
        let ws_path = dir.path().join("untracked-ws");
        std::fs::create_dir_all(&ws_path).expect("create dir");
        std::fs::write(ws_path.join("critical_work.rs"), "fn review() {}").expect("write artifact");
        // A marker naming a different thread than the workspace claims.
        std::fs::write(
            ws_path.join(".juto-workspace.json"),
            r#"{"git_dir":"/nonexistent","thread":"other"}"#,
        )
        .expect("write marker");

        let ws = SubagentWorkspace {
            path: ws_path.clone(),
            node_id: "nonexistent".to_string(),
            thread: "agent-1".to_string(),
            repo_root: dir.path().join("nonexistent-repo"),
            subagent_id: "agent-1".to_string(),
        };

        let cleanup_res = ws.cleanup().await;
        assert!(
            cleanup_res.is_err(),
            "cleanup must refuse a directory it does not own"
        );
        assert!(
            ws_path.exists(),
            "workspace directory must be preserved when ownership cannot be proven"
        );
        assert!(
            ws_path.join("critical_work.rs").exists(),
            "review artifacts must remain intact"
        );
    }

    #[tokio::test]
    async fn test_cleanup_removes_owned_directory() {
        let test_root = tempdir().expect("tempdir");
        let repo_dir = test_root.path().join("repo");
        std::fs::create_dir_all(&repo_dir).expect("create repo dir");
        native_repo(&repo_dir);

        let graph = ChangeGraph::open(&repo_dir).expect("open graph");
        let source = graph
            .capture(
                "main",
                EditProvenance {
                    session_id: "workspace-test".to_string(),
                    tool: "fork".to_string(),
                    selection_mode: None,
                    path: None,
                },
                None,
            )
            .expect("capture source point");

        let ws_dir = test_root.path().join("owned-ws");
        let ws = SubagentWorkspace::provision_in(&repo_dir, &ws_dir, &source.id, "owned-agent")
            .await
            .expect("provision workspace");
        assert!(ws_dir.exists());
        ws.cleanup().await.expect("cleanup");
        assert!(!ws_dir.exists(), "owned workspace directory is removed");
    }

    #[tokio::test]
    async fn test_cgroup_membership_and_limits_smoke() {
        if !is_systemd_run_available() {
            return;
        }

        let dir = tempdir().expect("tempdir");
        let ws = SubagentWorkspace {
            path: dir.path().to_path_buf(),
            node_id: "test-node".to_string(),
            thread: "test-ws".to_string(),
            repo_root: dir.path().to_path_buf(),
            subagent_id: "test-agent".to_string(),
        };

        let limits = ResourceLimits::default()
            .with_cpu_quota("200%")
            .with_memory_max("2G");

        let mut cmd = Command::new("sh");
        // Inside the scope, check cgroup path
        cmd.arg("-c").arg("cat /proc/self/cgroup");

        let status = ws.run_scoped(&mut cmd, limits).await.expect("run_scoped");
        assert!(status.success());
    }

    #[test]
    fn test_slice_preexisting_limits_preserved() {
        // If agents.slice already exists and has non-infinity properties,
        // configure_slice_properties should not fail or clobber them.
        configure_slice_properties("agents.slice", "200%", "2G");
    }

    #[test]
    fn test_scoped_cwd_rejects_symlink_escape() {
        let dir = tempdir().expect("tempdir");
        let ws_path = dir.path().join("subagent-ws");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&ws_path).expect("create workspace");
        std::fs::create_dir_all(&outside).expect("create outside");
        std::os::unix::fs::symlink(&outside, ws_path.join("out")).expect("create symlink");

        let ws = workspace_at(&ws_path);
        let limits = ResourceLimits::default();

        // A symlink inside the workspace that points outside is rejected.
        let mut direct = std::process::Command::new("ls");
        direct.current_dir(ws_path.join("out"));
        assert!(matches!(
            ws.build_scoped_std_command(&direct, &limits),
            Err(WorkspaceError::CwdEscapedWorkspace { .. })
        ));

        // `out/missing/..` collapses lexically to `out`, which is a symlink
        // pointing outside the workspace. Canonicalization fails because
        // `missing` does not exist, so the request must be rejected rather than
        // accepted as the escaping symlink.
        let mut traversal = std::process::Command::new("ls");
        traversal.current_dir(ws_path.join("out/missing/.."));
        assert!(
            ws.build_scoped_std_command(&traversal, &limits).is_err(),
            "a cwd collapsing to a symlink outside the workspace must be rejected"
        );

        // A real subdirectory still resolves inside the workspace.
        std::fs::create_dir_all(ws_path.join("subdir")).expect("create subdir");
        let mut valid = std::process::Command::new("ls");
        valid.current_dir(ws_path.join("subdir"));
        let scoped = ws
            .build_scoped_std_command(&valid, &limits)
            .expect("real subdirectory is accepted");
        assert_eq!(
            scoped.get_current_dir(),
            Some(ws_path.canonicalize().unwrap().join("subdir").as_path())
        );

        // A workspace reached through a symlinked parent still accepts an
        // absolute cwd expressed through that symlink.
        let real_root = dir.path().join("real");
        std::fs::create_dir_all(real_root.join("ws/subdir")).expect("create real workspace");
        std::os::unix::fs::symlink(&real_root, dir.path().join("link")).expect("parent symlink");
        let linked_ws = dir.path().join("link/ws");
        let linked = workspace_at(&linked_ws);
        let mut through_link = std::process::Command::new("ls");
        through_link.current_dir(linked_ws.join("subdir"));
        let scoped = linked
            .build_scoped_std_command(&through_link, &limits)
            .expect("a cwd through a symlinked parent that resolves inside is accepted");
        assert_eq!(
            scoped.get_current_dir(),
            Some(
                real_root
                    .canonicalize()
                    .unwrap()
                    .join("ws/subdir")
                    .as_path()
            )
        );
    }

    #[test]
    fn test_resource_limits_treat_blank_env_as_unset() {
        let _guard = env_lock();
        let vars = [
            "AGENT_CPU_QUOTA",
            "AGENT_MEMORY_MAX",
            "AGENT_SLICE",
            "AGENT_SLICE_CPU_QUOTA",
            "AGENT_SLICE_MEMORY_MAX",
            "AGENT_GPU_LOCK",
        ];
        let _env = EnvGuard::clear(&vars);
        for var in vars {
            // SAFETY: serialized by ENV_LOCK.
            unsafe { std::env::set_var(var, "") };
        }
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("AGENT_MEMORY_MAX", "   ") };

        let limits = ResourceLimits::default();
        assert_eq!(limits.cpu_quota, DEFAULT_CPU_QUOTA);
        assert_eq!(limits.memory_max, DEFAULT_MEMORY_MAX);
        assert_eq!(limits.slice.as_deref(), Some(DEFAULT_SLICE));
        assert_eq!(
            limits.slice_cpu_quota.as_deref(),
            Some(DEFAULT_SLICE_CPU_QUOTA)
        );
        assert_eq!(
            limits.slice_memory_max.as_deref(),
            Some(DEFAULT_SLICE_MEMORY_MAX)
        );
        assert_eq!(
            limits.gpu_lock.as_deref(),
            Some(default_gpu_lock_path().as_path())
        );
    }

    #[test]
    fn test_default_workspaces_dir_treats_blank_env_as_unset() {
        let _guard = env_lock();
        let _env = EnvGuard::clear(&["JUTO_WORKSPACES_DIR"]);
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("JUTO_WORKSPACES_DIR", "") };

        let expected = match std::env::var("HOME") {
            Ok(home) if !home.trim().is_empty() => PathBuf::from(home).join("workspaces"),
            _ => PathBuf::from("/tmp/workspaces"),
        };
        assert_eq!(default_workspaces_dir(), expected);
    }

    #[test]
    fn test_env_var_nonempty_treats_blank_as_unset() {
        let _guard = env_lock();
        let _env = EnvGuard::clear(&["JUTO_TEST_ENV_VAR"]);

        assert_eq!(env_var_nonempty("JUTO_TEST_ENV_VAR"), None);
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("JUTO_TEST_ENV_VAR", "") };
        assert_eq!(env_var_nonempty("JUTO_TEST_ENV_VAR"), None);
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("JUTO_TEST_ENV_VAR", "  \t ") };
        assert_eq!(env_var_nonempty("JUTO_TEST_ENV_VAR"), None);
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("JUTO_TEST_ENV_VAR", "value") };
        assert_eq!(
            env_var_nonempty("JUTO_TEST_ENV_VAR").as_deref(),
            Some("value")
        );
    }

    #[test]
    fn test_gpu_lock_relative_path_resolved_against_caller_cwd() {
        let dir = tempdir().expect("tempdir");
        let ws_path = dir.path().join("subagent-ws");
        std::fs::create_dir_all(&ws_path).expect("create workspace");
        let ws = workspace_at(&ws_path);

        let mut limits = ResourceLimits::default().with_gpu(true);
        limits.gpu_lock = Some(PathBuf::from("relative-gpu.lock"));

        let orig = std::process::Command::new("true");
        let scoped = ws
            .build_scoped_std_command(&orig, &limits)
            .expect("build scoped with gpu");
        assert_eq!(scoped.get_program(), "flock");

        let lock_arg = scoped.get_args().next().expect("lock path argument");
        let lock_path = PathBuf::from(lock_arg);
        assert!(
            lock_path.is_absolute(),
            "gpu lock must be absolute so every agent locks the same file, got {lock_path:?}"
        );
        assert!(lock_path.ends_with("relative-gpu.lock"));
        assert_eq!(
            lock_path.parent(),
            Some(std::env::current_dir().expect("process cwd").as_path())
        );
    }

    #[tokio::test]
    async fn test_spawn_scoped_forwards_stdio_and_captures_output() {
        use tokio::io::AsyncReadExt as _;

        let dir = tempdir().expect("tempdir");
        let ws = workspace_at(dir.path());
        let limits = ResourceLimits::default().with_cpu_quota("200%");

        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("echo captured-line; echo captured-err >&2");

        let mut child = ws
            .spawn_scoped(
                &mut cmd,
                limits,
                ScopedStdio {
                    stdin: Stdio::null(),
                    stdout: Stdio::piped(),
                    stderr: Stdio::piped(),
                },
            )
            .await
            .expect("spawn scoped");

        let mut stdout = String::new();
        child
            .stdout
            .take()
            .expect("stdout must be piped")
            .read_to_string(&mut stdout)
            .await
            .expect("read stdout");
        let mut stderr = String::new();
        child
            .stderr
            .take()
            .expect("stderr must be piped")
            .read_to_string(&mut stderr)
            .await
            .expect("read stderr");
        let status = child.wait().await.expect("wait");

        assert!(status.success());
        assert!(stdout.contains("captured-line"), "stdout: {stdout:?}");
        assert!(stderr.contains("captured-err"), "stderr: {stderr:?}");
    }

    #[tokio::test]
    async fn test_spawn_scoped_kills_child_on_drop() {
        let dir = tempdir().expect("tempdir");
        let ws = workspace_at(dir.path());
        let limits = ResourceLimits::default().with_cpu_quota("200%");
        let marker = dir.path().join("survived");

        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(format!("sleep 1; touch {}", marker.display()));

        let child = ws
            .spawn_scoped(&mut cmd, limits, ScopedStdio::default())
            .await
            .expect("spawn scoped");
        drop(child);

        tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
        assert!(
            !marker.exists(),
            "dropping a scoped child must terminate the process instead of leaking it"
        );
    }
}
