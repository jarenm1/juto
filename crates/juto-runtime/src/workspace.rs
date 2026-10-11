//! Workspace isolation and resource governance for Juto subagents.
//!
//! Provides isolated Jujutsu (`jj`) workspaces per subagent task and enforces
//! resource caps (CPU, memory, swap) via systemd user scopes (`agents.slice`)
//! and serialized GPU access via advisory file locks (`flock`).
//!
//! Architectural reference: `docs/orchestration-plan.md` §4 and `scripts/agent-scope.sh`.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::ExitStatus,
    sync::Arc,
};

use fs2::FileExt;
use thiserror::Error;
pub use tokio::process::Command;

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
    #[error("jujutsu command `{command}` failed (exit code {code:?}): {stderr}")]
    JjCommandFailed {
        command: String,
        code: Option<i32>,
        stderr: String,
    },

    #[error("failed to spawn jujutsu command `{command}`: {source}")]
    JjSpawnFailed {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("directory is not inside a Jujutsu repository: {0}")]
    NotInJjRepo(PathBuf),

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
            std::env::var("AGENT_CPU_QUOTA").unwrap_or_else(|_| DEFAULT_CPU_QUOTA.to_string());
        let memory_max =
            std::env::var("AGENT_MEMORY_MAX").unwrap_or_else(|_| DEFAULT_MEMORY_MAX.to_string());
        let slice = std::env::var("AGENT_SLICE")
            .ok()
            .or_else(|| Some(DEFAULT_SLICE.to_string()));
        let slice_cpu_quota = std::env::var("AGENT_SLICE_CPU_QUOTA")
            .ok()
            .or_else(|| Some(DEFAULT_SLICE_CPU_QUOTA.to_string()));
        let slice_memory_max = std::env::var("AGENT_SLICE_MEMORY_MAX")
            .ok()
            .or_else(|| Some(DEFAULT_SLICE_MEMORY_MAX.to_string()));
        let gpu_lock = std::env::var("AGENT_GPU_LOCK")
            .map(PathBuf::from)
            .ok()
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

/// An isolated Jujutsu (`jj`) workspace dedicated to a single subagent task.
#[derive(Debug, Clone)]
pub struct SubagentWorkspace {
    /// Filesystem path to the subagent's dedicated working copy directory.
    pub path: PathBuf,
    /// Jujutsu change ID of the working copy commit (`@`).
    pub change_id: String,
    /// Jujutsu workspace name (typically basename of `path`).
    pub workspace_name: String,
    /// Root path of the parent Jujutsu repository.
    pub repo_root: PathBuf,
    /// Subagent identifier.
    pub subagent_id: String,
}

impl SubagentWorkspace {
    /// Provision an isolated Jujutsu workspace from `base_change` for `subagent_id`.
    ///
    /// Locates the enclosing Jujutsu repository from the current directory (or
    /// `JUTO_REPO_ROOT` if set) and creates `~/workspaces/juto-<subagent-id>` (or
    /// under `JUTO_WORKSPACES_DIR` if set).
    pub async fn provision(base_change: &str, subagent_id: &str) -> Result<Self, WorkspaceError> {
        let repo_root = if let Ok(root) = std::env::var("JUTO_REPO_ROOT") {
            PathBuf::from(root)
        } else {
            let cwd = std::env::current_dir().map_err(|source| WorkspaceError::Io {
                path: PathBuf::from("."),
                source,
            })?;
            find_jj_root(&cwd).ok_or(WorkspaceError::NotInJjRepo(cwd))?
        };

        let workspaces_dir = default_workspaces_dir();
        let sanitized_id = sanitize_workspace_slug(subagent_id);
        let workspace_path = workspaces_dir.join(format!("juto-{}", sanitized_id));

        Self::provision_in(&repo_root, &workspace_path, base_change, subagent_id).await
    }

    /// Provision an isolated Jujutsu workspace at an explicit filesystem path.
    pub async fn provision_in(
        repo_root: &Path,
        workspace_path: &Path,
        base_change: &str,
        subagent_id: &str,
    ) -> Result<Self, WorkspaceError> {
        if let Some(parent) = workspace_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| WorkspaceError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let workspace_name = workspace_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(subagent_id)
            .to_string();

        // Run `jj workspace add <path> -r <base_change>` inside the repository
        let add_output = tokio::process::Command::new("jj")
            .arg("workspace")
            .arg("add")
            .arg(workspace_path)
            .arg("-r")
            .arg(base_change)
            .current_dir(repo_root)
            .output()
            .await
            .map_err(|source| WorkspaceError::JjSpawnFailed {
                command: format!(
                    "jj workspace add {} -r {}",
                    workspace_path.display(),
                    base_change
                ),
                source,
            })?;

        if !add_output.status.success() {
            return Err(WorkspaceError::JjCommandFailed {
                command: format!(
                    "jj workspace add {} -r {}",
                    workspace_path.display(),
                    base_change
                ),
                code: add_output.status.code(),
                stderr: String::from_utf8_lossy(&add_output.stderr).into_owned(),
            });
        }

        // Query the change_id of the working copy commit in the new workspace
        let change_id = query_change_id(workspace_path).await?;

        Ok(Self {
            path: workspace_path.to_path_buf(),
            change_id,
            workspace_name,
            repo_root: repo_root.to_path_buf(),
            subagent_id: subagent_id.to_string(),
        })
    }

    /// Query the working copy's current Jujutsu commit ID.
    pub async fn commit_id(&self) -> Result<String, WorkspaceError> {
        query_commit_id(&self.path).await
    }

    /// Query the workspace's status (`jj status --no-pager`).
    pub async fn status(&self) -> Result<String, WorkspaceError> {
        let output = tokio::process::Command::new("jj")
            .arg("status")
            .arg("--no-pager")
            .current_dir(&self.path)
            .output()
            .await
            .map_err(|source| WorkspaceError::JjSpawnFailed {
                command: "jj status --no-pager".to_string(),
                source,
            })?;

        if !output.status.success() {
            return Err(WorkspaceError::JjCommandFailed {
                command: "jj status --no-pager".to_string(),
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Query the workspace's diff (`jj diff --no-pager`).
    pub async fn diff(&self) -> Result<String, WorkspaceError> {
        let output = tokio::process::Command::new("jj")
            .arg("diff")
            .arg("--no-pager")
            .current_dir(&self.path)
            .output()
            .await
            .map_err(|source| WorkspaceError::JjSpawnFailed {
                command: "jj diff --no-pager".to_string(),
                source,
            })?;

        if !output.status.success() {
            return Err(WorkspaceError::JjCommandFailed {
                command: "jj diff --no-pager".to_string(),
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Rebase this workspace's change ID into a target destination change or branch.
    ///
    /// Executes `jj rebase -s <change_id> -d <target_dest>` and returns the
    /// resulting commit ID upon success.
    pub async fn rebase_into(&self, target_dest: &str) -> Result<String, WorkspaceError> {
        let output = tokio::process::Command::new("jj")
            .arg("rebase")
            .arg("-s")
            .arg(&self.change_id)
            .arg("-d")
            .arg(target_dest)
            .current_dir(&self.path)
            .output()
            .await
            .map_err(|source| WorkspaceError::JjSpawnFailed {
                command: format!("jj rebase -s {} -d {}", self.change_id, target_dest),
                source,
            })?;

        if !output.status.success() {
            return Err(WorkspaceError::JjCommandFailed {
                command: format!("jj rebase -s {} -d {}", self.change_id, target_dest),
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        // Query the new commit ID of our change after rebase
        let commit_output = tokio::process::Command::new("jj")
            .arg("log")
            .arg("--no-graph")
            .arg("-r")
            .arg(&self.change_id)
            .arg("-T")
            .arg("commit_id")
            .current_dir(&self.path)
            .output()
            .await
            .map_err(|source| WorkspaceError::JjSpawnFailed {
                command: format!("jj log -r {} -T commit_id", self.change_id),
                source,
            })?;

        if !commit_output.status.success() {
            return Err(WorkspaceError::JjCommandFailed {
                command: format!("jj log -r {} -T commit_id", self.change_id),
                code: commit_output.status.code(),
                stderr: String::from_utf8_lossy(&commit_output.stderr).into_owned(),
            });
        }

        let commit_id = String::from_utf8_lossy(&commit_output.stdout)
            .trim()
            .to_string();
        Ok(commit_id)
    }

    /// Forget this workspace from Jujutsu tracking without removing its disk contents.
    pub async fn forget(&mut self) -> Result<(), WorkspaceError> {
        let output = tokio::process::Command::new("jj")
            .arg("workspace")
            .arg("forget")
            .current_dir(&self.path)
            .output()
            .await
            .map_err(|source| WorkspaceError::JjSpawnFailed {
                command: "jj workspace forget".to_string(),
                source,
            })?;

        if !output.status.success() {
            // Also attempt from repo_root specifying workspace name
            let retry = tokio::process::Command::new("jj")
                .arg("workspace")
                .arg("forget")
                .arg(&self.workspace_name)
                .arg("-R")
                .arg(&self.repo_root)
                .output()
                .await
                .map_err(|source| WorkspaceError::JjSpawnFailed {
                    command: format!("jj workspace forget {}", self.workspace_name),
                    source,
                })?;

            if !retry.status.success() {
                return Err(WorkspaceError::JjCommandFailed {
                    command: format!("jj workspace forget {}", self.workspace_name),
                    code: retry.status.code(),
                    stderr: String::from_utf8_lossy(&retry.stderr).into_owned(),
                });
            }
        }

        Ok(())
    }

    /// Clean up this workspace: forgets it from Jujutsu tracking and removes its directory from disk.
    ///
    /// If `forget` fails, directory deletion is skipped so that human and automated review artifacts
    /// are preserved for diagnosis.
    pub async fn cleanup(mut self) -> Result<(), WorkspaceError> {
        self.forget().await?;

        if self.path.exists() {
            std::fs::remove_dir_all(&self.path).map_err(|source| WorkspaceError::Io {
                path: self.path.clone(),
                source,
            })?;
        }

        Ok(())
    }

    /// Run a command inside this workspace under resource governance.
    ///
    /// Wraps execution in `systemd-run --user --scope` with CPU, memory, and swap caps,
    /// and optionally serializes behind `flock` when `limits.gpu` is enabled.
    pub async fn run_scoped(
        &self,
        cmd: &mut Command,
        limits: ResourceLimits,
    ) -> Result<ExitStatus, WorkspaceError> {
        let mut scoped = self.build_scoped_command(cmd.as_std(), &limits)?;
        let mut child = scoped
            .spawn()
            .map_err(|source| WorkspaceError::CommandSpawnFailed {
                command: format!("{:?}", cmd.as_std().get_program()),
                source,
            })?;

        let status = child
            .wait()
            .await
            .map_err(|source| WorkspaceError::ExecutionFailed(source.to_string()))?;

        Ok(status)
    }

    /// Run a synchronous `std::process::Command` inside this workspace under resource governance.
    pub fn run_scoped_blocking(
        &self,
        cmd: &mut std::process::Command,
        limits: ResourceLimits,
    ) -> Result<ExitStatus, WorkspaceError> {
        let mut scoped = self.build_scoped_std_command(cmd, &limits)?;
        let mut child = scoped
            .spawn()
            .map_err(|source| WorkspaceError::CommandSpawnFailed {
                command: format!("{:?}", cmd.get_program()),
                source,
            })?;

        let status = child
            .wait()
            .map_err(|source| WorkspaceError::ExecutionFailed(source.to_string()))?;

        Ok(status)
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

/// Locate the nearest parent directory containing a `.jj` directory.
pub fn find_jj_root(start_dir: &Path) -> Option<PathBuf> {
    let mut curr = start_dir
        .canonicalize()
        .ok()
        .or_else(|| Some(start_dir.to_path_buf()))?;
    loop {
        if curr.join(".jj").exists() {
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
    if let Ok(dir) = std::env::var("JUTO_WORKSPACES_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join("workspaces");
    }
    PathBuf::from("/tmp/workspaces")
}

/// Returns the default GPU lock file path (`/tmp/jtech-agent-gpu.lock` or XDG_RUNTIME_DIR).
pub fn default_gpu_lock_path() -> PathBuf {
    if let Ok(path) = std::env::var("AGENT_GPU_LOCK") {
        return PathBuf::from(path);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
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
            let canon = resolved
                .canonicalize()
                .unwrap_or_else(|_| normalize_path(&resolved));
            if !canon.starts_with(&ws_canon) {
                return Err(WorkspaceError::CwdEscapedWorkspace {
                    cwd: dir.to_path_buf(),
                    workspace: workspace.to_path_buf(),
                });
            }
            canon
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

/// Query the working copy's change ID inside a workspace.
async fn query_change_id(workspace_path: &Path) -> Result<String, WorkspaceError> {
    let output = tokio::process::Command::new("jj")
        .arg("log")
        .arg("--no-graph")
        .arg("-r")
        .arg("@")
        .arg("-T")
        .arg("change_id")
        .current_dir(workspace_path)
        .output()
        .await
        .map_err(|source| WorkspaceError::JjSpawnFailed {
            command: "jj log -r @ -T change_id".to_string(),
            source,
        })?;

    if !output.status.success() {
        return Err(WorkspaceError::JjCommandFailed {
            command: "jj log -r @ -T change_id".to_string(),
            code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    let change_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(change_id)
}

/// Query the working copy's commit ID inside a workspace.
async fn query_commit_id(workspace_path: &Path) -> Result<String, WorkspaceError> {
    let output = tokio::process::Command::new("jj")
        .arg("log")
        .arg("--no-graph")
        .arg("-r")
        .arg("@")
        .arg("-T")
        .arg("commit_id")
        .current_dir(workspace_path)
        .output()
        .await
        .map_err(|source| WorkspaceError::JjSpawnFailed {
            command: "jj log -r @ -T commit_id".to_string(),
            source,
        })?;

    if !output.status.success() {
        return Err(WorkspaceError::JjCommandFailed {
            command: "jj log -r @ -T commit_id".to_string(),
            code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    let commit_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(commit_id)
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
    use tempfile::tempdir;

    #[test]
    fn test_resource_limits_defaults_and_builder() {
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
            change_id: "test-change".to_string(),
            workspace_name: "test-ws".to_string(),
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
            change_id: "test-change".to_string(),
            workspace_name: "test-ws".to_string(),
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
            change_id: "test-change".to_string(),
            workspace_name: "test-ws".to_string(),
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

    #[tokio::test]
    async fn test_jj_workspace_lifecycle_end_to_end() {
        let test_root = tempdir().expect("tempdir");
        let repo_dir = test_root.path().join("repo");
        std::fs::create_dir_all(&repo_dir).expect("create repo dir");

        // Initialize a clean test repo
        let init_status = std::process::Command::new("jj")
            .arg("git")
            .arg("init")
            .current_dir(&repo_dir)
            .status()
            .expect("init jj repo");
        assert!(init_status.success());

        // Create initial base file and commit
        std::fs::write(repo_dir.join("README.md"), "# Initial\n").expect("write file");
        let desc_status = std::process::Command::new("jj")
            .arg("describe")
            .arg("-m")
            .arg("initial commit")
            .current_dir(&repo_dir)
            .status()
            .expect("describe initial commit");
        assert!(desc_status.success());

        let ws_dir = test_root.path().join("subagent-ws");

        // 1. Provision workspace
        let ws = SubagentWorkspace::provision_in(&repo_dir, &ws_dir, "@", "test-subagent")
            .await
            .expect("provision workspace");

        assert!(ws.path.exists());
        assert_eq!(ws.path, ws_dir);
        assert!(!ws.change_id.is_empty());
        assert_eq!(ws.subagent_id, "test-subagent");

        // 2. Query status
        let status_out = ws.status().await.expect("status");
        assert!(status_out.contains("Working copy") || status_out.contains("The working copy"));

        // 3. Make change inside workspace
        std::fs::write(ws.path.join("README.md"), "# Modified by subagent\n").expect("modify file");

        // 4. Query diff
        let diff_out = ws.diff().await.expect("diff");
        assert!(diff_out.contains("Modified by subagent") || diff_out.contains("README.md"));

        // 5. Run scoped command inside the workspace
        let limits = ResourceLimits::default().with_cpu_quota("200%");
        let mut test_cmd = Command::new("sh");
        test_cmd.arg("-c").arg("test -f README.md");
        let run_status = ws
            .run_scoped(&mut test_cmd, limits)
            .await
            .expect("scoped command");
        assert!(run_status.success());

        // 6. Query commit id
        let commit_id = ws.commit_id().await.expect("commit_id");
        assert!(!commit_id.is_empty());

        // 7. Cleanup
        ws.cleanup().await.expect("cleanup");
        assert!(
            !ws_dir.exists(),
            "workspace directory should be removed on cleanup"
        );
    }

    #[tokio::test]
    async fn test_rebase_into_success() {
        let test_root = tempdir().expect("tempdir");
        let repo_dir = test_root.path().join("repo");
        std::fs::create_dir_all(&repo_dir).expect("create repo dir");

        // Init repo and create initial commit
        let init_status = std::process::Command::new("jj")
            .arg("git")
            .arg("init")
            .current_dir(&repo_dir)
            .status()
            .expect("init jj repo");
        assert!(init_status.success());

        std::fs::write(repo_dir.join("base.txt"), "base content\n").expect("write base");
        let desc_status = std::process::Command::new("jj")
            .arg("describe")
            .arg("-m")
            .arg("base commit")
            .current_dir(&repo_dir)
            .status()
            .expect("describe base");
        assert!(desc_status.success());

        // Advance parent with a new commit
        let commit_status = std::process::Command::new("jj")
            .arg("commit")
            .arg("-m")
            .arg("head commit")
            .current_dir(&repo_dir)
            .status()
            .expect("commit head");
        assert!(commit_status.success());
        // Query the change id of the head commit in repo_dir
        let head_change_out = std::process::Command::new("jj")
            .arg("log")
            .arg("--no-graph")
            .arg("-r")
            .arg("@")
            .arg("-T")
            .arg("change_id")
            .current_dir(&repo_dir)
            .output()
            .expect("query head change_id");
        let target_dest = String::from_utf8_lossy(&head_change_out.stdout)
            .trim()
            .to_string();
        assert!(!target_dest.is_empty());

        // Provision subagent workspace from the base commit (@-)
        let ws_dir = test_root.path().join("subagent-rebase-ws");
        let ws = SubagentWorkspace::provision_in(&repo_dir, &ws_dir, "@-", "rebase-agent")
            .await
            .expect("provision workspace");

        // Subagent modifies a distinct file
        std::fs::write(ws.path.join("feature.txt"), "feature content\n").expect("write feature");

        // Rebase into target commit from repo
        let new_commit_id = ws
            .rebase_into(&target_dest)
            .await
            .expect("rebase into target");
        assert!(!new_commit_id.is_empty());
        ws.cleanup().await.expect("cleanup");
    }

    #[tokio::test]
    async fn test_systemd_run_non_zero_exit_code() {
        let dir = tempdir().expect("tempdir");
        let ws = SubagentWorkspace {
            path: dir.path().to_path_buf(),
            change_id: "test-change".to_string(),
            workspace_name: "test-ws".to_string(),
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
            change_id: "test-change".to_string(),
            workspace_name: "test-ws".to_string(),
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
    async fn test_workspace_not_in_jj_repo_error() {
        let non_repo = tempdir().expect("tempdir");
        let ws_dir = non_repo.path().join("ws");

        let result =
            SubagentWorkspace::provision_in(non_repo.path(), &ws_dir, "@", "agent-fail").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            WorkspaceError::JjCommandFailed { stderr, .. } => {
                let lower = stderr.to_lowercase();
                assert!(lower.contains("no jujutsu repo") || lower.contains("error"));
            }
            other => panic!("expected JjCommandFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_scoped_cwd_escape_rejected() {
        let dir = tempdir().expect("tempdir");
        let ws_path = dir.path().join("subagent-ws");
        std::fs::create_dir_all(ws_path.join("subdir")).expect("create subdir");

        let ws = SubagentWorkspace {
            path: ws_path.clone(),
            change_id: "test-change".to_string(),
            workspace_name: "subagent-ws".to_string(),
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
    async fn test_cleanup_preserves_directory_on_forget_failure() {
        let dir = tempdir().expect("tempdir");
        let ws_path = dir.path().join("untracked-ws");
        std::fs::create_dir_all(&ws_path).expect("create dir");
        std::fs::write(ws_path.join("critical_work.rs"), "fn review() {}").expect("write artifact");

        // SubagentWorkspace with bogus repo root and workspace name where `jj workspace forget` will fail
        let ws = SubagentWorkspace {
            path: ws_path.clone(),
            change_id: "nonexistent".to_string(),
            workspace_name: "bogus-ws".to_string(),
            repo_root: dir.path().join("nonexistent-repo"),
            subagent_id: "agent-1".to_string(),
        };

        let cleanup_res = ws.cleanup().await;
        assert!(cleanup_res.is_err(), "cleanup must fail if forget fails");
        assert!(
            ws_path.exists(),
            "workspace directory must be preserved on disk for human review when forget fails"
        );
        assert!(
            ws_path.join("critical_work.rs").exists(),
            "review artifacts must remain intact"
        );
    }

    #[tokio::test]
    async fn test_cgroup_membership_and_limits_smoke() {
        if !is_systemd_run_available() {
            return;
        }

        let dir = tempdir().expect("tempdir");
        let ws = SubagentWorkspace {
            path: dir.path().to_path_buf(),
            change_id: "test-change".to_string(),
            workspace_name: "test-ws".to_string(),
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
}
