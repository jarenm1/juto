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
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::Arc,
};

use fs2::FileExt;
use juto_capacity::MachineCapacity;
use serde::Deserialize;
use thiserror::Error;
pub use tokio::process::Command;
use uuid::Uuid;

use crate::broker::{Broker, BrokerError, Lease, Priority, Request, Submission};
use crate::change_graph::ChangeGraph;
use crate::git_transfer::GitRepository;
use crate::resources::{USAGE_WRAPPER, UsageStore, UsageTicket, command_class, command_grant};

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
/// `systemd-run` (e.g. `-p CPUQuota=` applies `infinity`, silently uncapping
/// the scope).
fn env_var_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// Default memory swap max assigned to a single subagent scope (no swap).
pub const DEFAULT_MEMORY_SWAP_MAX: &str = "0";

/// Default shared systemd slice under which all agent scopes run.
pub const DEFAULT_SLICE: &str = juto_capacity::AGENTS_SLICE;

/// CPU weight of every scope and the shared slice. Equal shares; priority
/// weighting arrives with the resource broker.
pub const DEFAULT_CPU_WEIGHT: u32 = 100;

/// I/O weight of every scope and the shared slice.
pub const DEFAULT_IO_WEIGHT: u32 = 100;

/// Errors that can occur during workspace lifecycle and resource-scoped execution.
#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("native VCS error: {0}")]
    Vcs(String),

    #[error("directory is not inside a native Juto repository: {0}")]
    NotInRepo(PathBuf),

    #[error("command working directory `{cwd}` is outside assigned workspace `{workspace}`")]
    CwdEscapedWorkspace { cwd: PathBuf, workspace: PathBuf },

    #[error(
        "systemd-run is unavailable; refusing to run the command without its resource governance"
    )]
    SystemdUnavailable,

    #[error("failed to spawn scoped command `{command}`: {source}")]
    CommandSpawnFailed {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("scoped command failed: {0}")]
    ExecutionFailed(String),

    #[error("resource capacity unavailable: {0}")]
    Capacity(String),

    #[error("shared slice limits not applied: {0}")]
    Slice(String),

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

    #[error(transparent)]
    Admission(#[from] BrokerError),
}

/// Filesystem sandbox applied to a governed command.
///
/// The command runs in a transient systemd *service* whose filesystem view is
/// restricted:
///
/// * `protect_system` mounts the system hierarchy (`/usr`, `/etc`, ...)
///   read-only (`ProtectSystem=strict`).
/// * `hide_home` replaces `$HOME` with an empty tmpfs (`ProtectHome=tmpfs`).
///   Credentials under it (`~/.ssh`, `~/.config/gh`, juto's data directory) are
///   absent; only the grants below reappear.
/// * `private_tmp` gives the command a private `/tmp` and `/var/tmp`.
/// * the workspace is bind-mounted writable. [`Sandbox::write_paths`] adds more
///   writable directories and [`Sandbox::read_paths`] read-only ones. Grants that
///   do not exist on the host are skipped.
///
/// The command's environment is an allowlist, applied with `env -i`: the user
/// manager's environment is not inherited.
///
/// Grants are bind mounts (`BindPaths`, `BindReadOnlyPaths`), not
/// `ReadWritePaths`: a `ReadWritePaths` entry fails with status 226 when its
/// mount point lies under the hidden home or the private `/tmp`.
///
/// # Running `cargo` inside a sandbox
///
/// `ProtectSystem=strict` makes the system read-only and `hide_home` hides
/// `$HOME`, so a build tool that keeps state there needs its directories granted.
/// [`Sandbox::for_agent_workspace`] grants `CARGO_HOME`, `RUSTUP_HOME`, and
/// `XDG_CACHE_HOME`, plus the flake checkout named by `JUTO_FLAKE` so that
/// `nix develop` can read it.
///
/// # Not a boundary against hostile code
///
/// The sandbox limits ordinary tool use: accidental writes and credential
/// exposure through files and the environment. It does not stop deliberately
/// malicious code. The nix daemon socket stays reachable, so builds and
/// substitutes the daemon runs are outside this sandbox. See `docs/sandbox.md`.
///
/// # Implementation note
///
/// Namespace properties are not supported by transient *scopes*:
/// `systemd-run --user --scope -p ProtectSystem=strict` fails with `Unknown
/// assignment`. A sandboxed command therefore runs as a transient *service* (see
/// [`SubagentWorkspace::build_scoped_std_command`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sandbox {
    /// Mount the system hierarchy read-only (`ProtectSystem=strict`).
    pub protect_system: bool,
    /// Replace `$HOME` with an empty tmpfs (`ProtectHome=tmpfs`).
    pub hide_home: bool,
    /// Give the command a private `/tmp` and `/var/tmp` (`PrivateTmp=yes`).
    pub private_tmp: bool,
    /// Extra directories granted write access (`BindPaths`).
    ///
    /// Relative entries are resolved against the workspace root. The workspace
    /// itself is always writable and does not need to be listed here.
    pub write_paths: Vec<PathBuf>,
    /// Extra directories granted read-only access (`BindReadOnlyPaths`).
    ///
    /// Relative entries are resolved against the workspace root.
    pub read_paths: Vec<PathBuf>,
    /// Credential locations hidden even when an enclosing directory is granted.
    pub denied_paths: Vec<PathBuf>,
    /// Exact-host HTTP/HTTPS policy; all sandboxed commands have PrivateNetwork.
    pub network: crate::network::NetworkPolicy,
}

impl Default for Sandbox {
    fn default() -> Self {
        Self {
            protect_system: true,
            hide_home: true,
            private_tmp: true,
            write_paths: Vec::new(),
            read_paths: Vec::new(),
            denied_paths: credential_paths(&[]),
            network: crate::network::NetworkPolicy::default(),
        }
    }
}

impl Sandbox {
    /// The default sandbox plus the toolchain and flake grants an agent build needs.
    ///
    /// `CARGO_HOME`, `RUSTUP_HOME`, and `XDG_CACHE_HOME` become write grants, taken
    /// from the environment or defaulting to `$HOME/.cargo`, `$HOME/.rustup`, and
    /// `$HOME/.cache`. The checkout named by `JUTO_FLAKE` becomes a read grant.
    /// Without these, cargo and `nix develop` fail inside the sandbox.
    pub fn for_agent_workspace() -> Self {
        let mut sandbox = Self::default();
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let toolchain = [
            ("CARGO_HOME", home.as_ref().map(|home| home.join(".cargo"))),
            (
                "RUSTUP_HOME",
                home.as_ref().map(|home| home.join(".rustup")),
            ),
            (
                "XDG_CACHE_HOME",
                home.as_ref().map(|home| home.join(".cache")),
            ),
        ];
        for (key, fallback) in toolchain {
            let path = std::env::var_os(key).map(PathBuf::from).or(fallback);
            if let Some(path) = path
                && path.is_absolute()
                && !sandbox.write_paths.contains(&path)
            {
                sandbox.write_paths.push(path);
            }
        }
        if let Some(flake) = flake_source_path() {
            sandbox.read_paths.push(flake);
        }
        sandbox
    }
}

/// The local checkout that `JUTO_FLAKE` (`git+file://<path>?rev=...`) names.
///
/// `nix develop "$JUTO_FLAKE"` inside a sandbox must read this checkout.
fn flake_source_path() -> Option<PathBuf> {
    let flake = std::env::var("JUTO_FLAKE").ok()?;
    let path = PathBuf::from(flake.strip_prefix("git+file://")?.split('?').next()?);
    path.is_absolute().then_some(path)
}

/// Known credential locations, including custom runtime-owned paths. Keep lexical
/// names as well as symlink targets: hiding only one would leave the other visible.
pub(crate) fn credential_paths(extra: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = extra.to_vec();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        for name in [
            ".ssh",
            ".gnupg",
            ".config/gh",
            ".netrc",
            ".aws",
            ".docker",
            ".local/share/juto",
            ".config/juto",
            ".git-credentials",
            ".npmrc",
            ".pypirc",
        ] {
            paths.push(home.join(name));
        }
        let cargo = std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".cargo"));
        paths.push(cargo.join("credentials"));
        paths.push(cargo.join("credentials.toml"));
    }
    for variable in [
        "JUTO_DATA_DIR",
        "GNUPGHOME",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "DOCKER_CONFIG",
    ] {
        if let Some(path) = std::env::var_os(variable) {
            paths.push(PathBuf::from(path));
        }
    }
    if let Some(config) = std::env::var_os("XDG_CONFIG_HOME") {
        paths.push(PathBuf::from(config).join("gh"));
    }
    if let Some(data) = std::env::var_os("XDG_DATA_HOME") {
        paths.push(PathBuf::from(data).join("juto"));
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut resolved = Vec::new();
    for path in paths {
        let path = if path.is_absolute() {
            path
        } else {
            cwd.join(path)
        };
        if let Ok(canonical) = path.canonicalize()
            && !resolved.contains(&canonical)
        {
            resolved.push(canonical);
        }
        if !resolved.contains(&path) {
            resolved.push(path);
        }
    }
    resolved
}

/// Receives the queue position when a scoped command waits for admission.
///
/// Equality is identity: two observers are equal only when they share one
/// callback, which keeps [`ResourceLimits`] comparable.
#[derive(Clone)]
pub struct QueueObserver(Arc<dyn Fn(usize) + Send + Sync>);

impl QueueObserver {
    /// Wrap `callback` as an observer.
    pub fn new(callback: impl Fn(usize) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }

    fn notify(&self, position: usize) {
        (self.0)(position);
    }
}

impl std::fmt::Debug for QueueObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QueueObserver")
    }
}

impl PartialEq for QueueObserver {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for QueueObserver {}

/// Parse a systemd byte size (`4G`, `512M`, `1048576`) into bytes.
///
/// Suffixes are powers of 1024, as systemd reads them. An unbounded or
/// unparsable limit cannot be reserved, so it is an error.
fn memory_reservation_bytes(size: &str) -> Result<u64, WorkspaceError> {
    let trimmed = size.trim();
    let (digits, multiplier) = match trimmed.chars().last() {
        Some(suffix) if suffix.is_ascii_alphabetic() => {
            let multiplier: u64 = match suffix.to_ascii_uppercase() {
                'K' => 1 << 10,
                'M' => 1 << 20,
                'G' => 1 << 30,
                'T' => 1 << 40,
                _ => return Err(unreservable(size)),
            };
            (&trimmed[..trimmed.len() - 1], multiplier)
        }
        _ => (trimmed, 1),
    };
    digits
        .parse::<u64>()
        .ok()
        .and_then(|value| value.checked_mul(multiplier))
        .ok_or_else(|| unreservable(size))
}

fn unreservable(size: &str) -> WorkspaceError {
    WorkspaceError::Capacity(format!(
        "memory limit `{size}` cannot be reserved; set a finite size"
    ))
}

/// Resource caps, and an optional sandbox, for one governed command.
///
/// Governance always runs the command under `systemd-run`. When `systemd-run` is
/// missing the command fails with [`WorkspaceError::SystemdUnavailable`]: there
/// is no uncapped fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Explicit per-scope CPU cap (e.g. "200%"); unset leaves CPU shared by weight.
    pub cpu_quota: Option<String>,
    /// Explicit per-scope memory cap (e.g. "4G"); unset derives it from capacity and history.
    pub memory_max: Option<String>,
    /// Per-scope swap limit (e.g. "0").
    pub memory_swap_max: String,
    /// Shared systemd slice under which scopes run (e.g. "agents.slice").
    pub slice: Option<String>,
    /// Explicit aggregate CPU cap for the shared slice (e.g. "600%"); unset leaves it uncapped.
    pub slice_cpu_quota: Option<String>,
    /// Explicit aggregate memory cap for the shared slice; unset derives it from measured capacity.
    pub slice_memory_max: Option<String>,
    /// Whether to serialize execution behind a shared GPU flock.
    pub gpu: bool,
    /// Path to the shared GPU file lock.
    pub gpu_lock: Option<PathBuf>,
    /// Filesystem sandbox for the command. `None` runs it under caps only; see [`Sandbox`].
    pub sandbox: Option<Sandbox>,
    /// Store that records each scoped command's resource usage; `None` records nothing.
    pub usage_store: Option<UsageStore>,
    /// Admission ledger; `None` starts commands without waiting for capacity.
    pub broker: Option<Broker>,
    /// Admission class; attended commands are served before background ones.
    pub priority: Priority,
    /// Notified with the queue position when this command has to wait.
    pub queue_observer: Option<QueueObserver>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let cpu_quota = env_var_nonempty("AGENT_CPU_QUOTA");
        let memory_max = env_var_nonempty("AGENT_MEMORY_MAX");
        let slice = env_var_nonempty("AGENT_SLICE").or_else(|| Some(DEFAULT_SLICE.to_string()));
        let slice_cpu_quota = env_var_nonempty("AGENT_SLICE_CPU_QUOTA");
        let slice_memory_max = env_var_nonempty("AGENT_SLICE_MEMORY_MAX");
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
            sandbox: None,
            usage_store: None,
            broker: None,
            priority: Priority::default(),
            queue_observer: None,
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
        self.cpu_quota = Some(quota.into());
        self
    }

    /// Set explicit per-scope memory ceiling.
    pub fn with_memory_max(mut self, mem: impl Into<String>) -> Self {
        self.memory_max = Some(mem.into());
        self
    }

    /// Enable filesystem sandboxing for this limit set.
    pub fn with_sandbox(mut self, sandbox: Sandbox) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Record each scoped command's resource usage in `store`.
    ///
    /// Recording applies to the spawn and run paths when systemd measures the
    /// command. The default records nothing.
    pub fn with_usage_store(mut self, store: UsageStore) -> Self {
        self.usage_store = Some(store);
        self
    }

    /// Admit scoped commands through `broker` before they start, so the memory
    /// reserved by all running commands fits the agent budget.
    ///
    /// Admission applies where systemd measures the command. Without a broker,
    /// commands start without waiting for capacity.
    pub fn with_admission(mut self, broker: Broker) -> Self {
        self.broker = Some(broker);
        self
    }

    /// Set this command's class in the admission queue.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Call `observer` with the queue position when this command has to wait.
    pub fn with_queue_observer(mut self, observer: QueueObserver) -> Self {
        self.queue_observer = Some(observer);
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

/// A command's admission before it joins the broker's queue.
pub struct Admission {
    broker: Broker,
    request: Request,
    budget: u64,
}

impl Admission {
    /// Join the broker's queue; see [`Broker::submit`].
    pub fn submit(self) -> Result<Submission, BrokerError> {
        self.broker.submit(self.request, self.budget)
    }

    /// Memory the command reserves while it is queued or running, in bytes.
    pub fn bytes(&self) -> u64 {
        self.request.bytes
    }
}

/// A scoped child whose transient systemd unit is stopped when it is dropped.
///
/// A sandboxed command runs as a transient *service*, which outlives its
/// `systemd-run` client: killing the client alone leaves the payload running, so
/// a cancelled command would keep working. Dropping this guard stops the unit.
/// An unsandboxed command runs as a *scope*, which the client owns, so there is
/// no unit to stop.
pub struct ScopedChild {
    /// `None` only after [`ScopedChild::wait_with_output`] has taken it.
    child: Option<tokio::process::Child>,
    unit: Option<String>,
    /// Usage ticket finished when the child is reaped; `None` when not recording.
    usage: Option<UsageTicket>,
    /// Admission reservation, released when the child is dropped.
    _lease: Option<Lease>,
}

impl ScopedChild {
    /// Wrap a child that has no transient unit to stop.
    ///
    /// An unsandboxed command runs as a scope, which its client owns, so there
    /// is nothing to stop on drop. This lets a caller use one type for both
    /// paths.
    pub fn unscoped(child: tokio::process::Child) -> Self {
        Self {
            child: Some(child),
            unit: None,
            usage: None,
            _lease: None,
        }
    }

    /// The underlying child, for reading piped output or signalling it.
    pub fn child_mut(&mut self) -> &mut tokio::process::Child {
        self.child.as_mut().expect("child is present")
    }

    /// The child's process id, when it is still running.
    pub fn id(&self) -> Option<u32> {
        self.child.as_ref().and_then(tokio::process::Child::id)
    }

    /// The transient unit name, when this child is sandboxed.
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Terminate the service payload before awaiting its output pipes.
    ///
    /// Killing `systemd-run` only kills the client: the service may retain the
    /// pipe descriptors. Waiting for those descriptors before dropping this
    /// guard would postpone cleanup until the payload exits on its own.
    pub async fn kill_unit(&mut self) -> std::io::Result<()> {
        let Some(unit) = &self.unit else {
            return Ok(());
        };
        // A collected unit may already be absent; that is a successful cleanup.
        // Keep the unit on the guard so cancellation during this await still
        // schedules the normal stop operation.
        // Kill the whole unit, then stop it. A kill alone can land before systemd has
        // forked the service's process, and then the process starts afterwards;
        // `stop` cancels a pending start job and waits until the unit is inactive.
        for args in [
            &["kill", "--signal=SIGKILL", "--kill-whom=all"][..],
            &["stop"][..],
        ] {
            let _ = tokio::process::Command::new("systemctl")
                .arg("--user")
                .args(args)
                .arg(unit)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await?;
        }
        Ok(())
    }

    /// Wait for the child to exit, recording its usage when recording is on.
    pub async fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.child_mut().wait().await?;
        if let Some(usage) = self.usage.take() {
            usage.finish(status);
        }
        Ok(status)
    }

    /// Wait for the child to exit, collecting its piped output.
    ///
    /// The guard stays alive until this returns, so a sandboxed command's unit
    /// is stopped even though the child itself is consumed.
    pub async fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        let child = self.child.take().expect("child is present");
        let output = child.wait_with_output().await?;
        if let Some(usage) = self.usage.take() {
            usage.finish(output.status);
        }
        Ok(output)
    }
}

impl Drop for ScopedChild {
    fn drop(&mut self) {
        let Some(unit) = &self.unit else {
            return;
        };
        // Fire-and-forget: `Drop` cannot await, and the stop is best-effort
        // because the unit may already be gone.
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "stop", unit])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
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
    ) -> Result<ScopedChild, WorkspaceError> {
        // Admission comes before the command is built, so a queued command holds
        // no process and no transient unit while it waits.
        let lease = match self.admission(cmd, &limits)? {
            Some(admission) => match admission.submit()? {
                Submission::Admitted(lease) => Some(lease),
                Submission::Queued(ticket) => {
                    if let Some(observer) = &limits.queue_observer {
                        observer.notify(ticket.position());
                    }
                    Some(ticket.admitted().await?)
                }
            },
            None => None,
        };
        self.spawn_admitted(cmd, &limits, stdio, lease)
    }

    /// The broker admission `cmd` needs, or `None` when no broker is configured.
    /// A host without `systemd-run` fails closed: the broker cannot measure or
    /// enforce the command's memory, so the command is refused, not run bare.
    pub fn admission(
        &self,
        cmd: &Command,
        limits: &ResourceLimits,
    ) -> Result<Option<Admission>, WorkspaceError> {
        let Some(broker) = &limits.broker else {
            return Ok(None);
        };
        if !is_systemd_run_available() {
            return Err(WorkspaceError::SystemdUnavailable);
        }
        let program = cmd.as_std().get_program().to_os_string();
        let args: Vec<OsString> = cmd.as_std().get_args().map(OsStr::to_os_string).collect();
        let repo = self.repo_root.display().to_string();
        let resolved = limits.resolve(&program, &args, &repo)?;
        let bytes = memory_reservation_bytes(&resolved.memory_max)?;
        Ok(Some(Admission {
            broker: broker.clone(),
            request: Request {
                bytes,
                gpu: limits.gpu,
                priority: limits.priority,
                label: command_class(&program, &args),
            },
            budget: limits.capacity()?.budget_bytes,
        }))
    }

    /// Spawn a command whose admission is already decided (see [`Self::admission`]).
    /// `lease` lives in the returned child, so the reservation ends with it.
    pub fn spawn_admitted(
        &self,
        cmd: &mut Command,
        limits: &ResourceLimits,
        stdio: ScopedStdio,
        lease: Option<Lease>,
    ) -> Result<ScopedChild, WorkspaceError> {
        // A sandboxed command runs as a transient *service*, which outlives its
        // `systemd-run` client. Naming the unit is what lets the guard stop it.
        let unit = limits
            .sandbox
            .is_some()
            .then(|| format!("juto-{}", Uuid::now_v7()));
        let (built, usage) =
            self.build_scoped_with_usage(cmd.as_std(), limits, unit.as_deref(), true)?;
        let mut scoped = Command::from(built);
        scoped
            .stdin(stdio.stdin)
            .stdout(stdio.stdout)
            .stderr(stdio.stderr)
            .kill_on_drop(true);
        // Its own process group, so a caller can signal the whole tree without
        // touching its own group.
        #[cfg(unix)]
        scoped.process_group(0);

        let child = scoped
            .spawn()
            .map_err(|source| WorkspaceError::CommandSpawnFailed {
                command: format!("{:?}", cmd.as_std().get_program()),
                source,
            })?;
        Ok(ScopedChild {
            child: Some(child),
            unit,
            usage,
            _lease: lease,
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
        let (mut child, usage) =
            self.spawn_blocking_with_usage(cmd, limits, ScopedStdio::default(), true)?;

        let status = child
            .wait()
            .map_err(|source| WorkspaceError::ExecutionFailed(source.to_string()))?;
        if let Some(usage) = usage {
            usage.finish(status);
        }
        Ok(status)
    }

    /// Spawn a synchronous `std::process::Command` inside this workspace under
    /// resource governance, returning the running child.
    pub fn spawn_scoped_blocking(
        &self,
        cmd: &mut std::process::Command,
        limits: ResourceLimits,
        stdio: ScopedStdio,
    ) -> Result<std::process::Child, WorkspaceError> {
        self.spawn_blocking_with_usage(cmd, limits, stdio, false)
            .map(|(child, _)| child)
    }

    /// Spawn a blocking scoped child, attaching a usage ticket when `record` is set.
    fn spawn_blocking_with_usage(
        &self,
        cmd: &mut std::process::Command,
        limits: ResourceLimits,
        stdio: ScopedStdio,
        record: bool,
    ) -> Result<(std::process::Child, Option<UsageTicket>), WorkspaceError> {
        let (mut scoped, usage) = self.build_scoped_with_usage(cmd, &limits, None, record)?;
        scoped
            .stdin(stdio.stdin)
            .stdout(stdio.stdout)
            .stderr(stdio.stderr);

        let child = scoped
            .spawn()
            .map_err(|source| WorkspaceError::CommandSpawnFailed {
                command: format!("{:?}", cmd.get_program()),
                source,
            })?;
        Ok((child, usage))
    }

    /// Build a resource-scoped `tokio::process::Command` from an existing `std::process::Command`.
    pub fn build_scoped_command(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
    ) -> Result<Command, WorkspaceError> {
        self.build_scoped_command_named(original, limits, None)
    }

    /// Build a resource-scoped `tokio::process::Command`, naming the unit.
    pub fn build_scoped_command_named(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
        unit: Option<&str>,
    ) -> Result<Command, WorkspaceError> {
        let std_cmd = self.build_scoped_std_command_named(original, limits, unit)?;
        Ok(Command::from(std_cmd))
    }

    /// Build a resource-scoped `std::process::Command` from an existing `std::process::Command`.
    pub fn build_scoped_std_command(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
    ) -> Result<std::process::Command, WorkspaceError> {
        self.build_scoped_std_command_named(original, limits, None)
    }

    /// Build a scoped command, naming the transient unit when `unit` is set.
    ///
    /// A sandboxed command runs as a transient *service*, which outlives its
    /// `systemd-run` client: killing the client leaves the payload running.
    /// Naming the unit is what lets a caller stop it.
    pub fn build_scoped_std_command_named(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
        unit: Option<&str>,
    ) -> Result<std::process::Command, WorkspaceError> {
        self.build_scoped_with_usage(original, limits, unit, false)
            .map(|(cmd, _)| cmd)
    }

    /// Build a scoped command, attaching a usage ticket when `record` is set
    /// and the limits name a usage store.
    fn build_scoped_with_usage(
        &self,
        original: &std::process::Command,
        limits: &ResourceLimits,
        unit: Option<&str>,
        record: bool,
    ) -> Result<(std::process::Command, Option<UsageTicket>), WorkspaceError> {
        let program = original.get_program().to_os_string();
        let orig_args: Vec<OsString> = original.get_args().map(|s| s.to_os_string()).collect();
        let work_dir = validate_scoped_cwd(&self.path, original.get_current_dir())?;

        // Governance always runs under systemd: a missing `systemd-run` is an
        // error, never a silent run without the caps or the sandbox.
        if !is_systemd_run_available() {
            return Err(WorkspaceError::SystemdUnavailable);
        }
        // Usage is measured from inside the unit's cgroup, so it needs systemd,
        // which the check above guarantees.
        let usage = match (record, &limits.usage_store) {
            (true, Some(store)) => {
                let repo = self.path.display().to_string();
                let class = command_class(&program, &orig_args);
                match UsageTicket::start(store, repo, class) {
                    Ok(ticket) => Some(ticket),
                    Err(error) => {
                        eprintln!("juto: resource usage not recorded: {error}");
                        None
                    }
                }
            }
            _ => None,
        };

        // Resolve every limit from measured capacity and this command's history.
        let repo = self.repo_root.display().to_string();
        let resolved = limits.resolve(&program, &orig_args, &repo)?;
        resolved.apply_slice()?;

        let mut cmd = if limits.gpu {
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
            let mut flock = std::process::Command::new("flock");
            flock.arg(&gpu_lock).arg("systemd-run");
            flock
        } else {
            std::process::Command::new("systemd-run")
        };
        cmd.arg("--user");
        append_systemd_unit_mode(&mut cmd, limits.sandbox.is_some());
        if let Some(unit) = unit {
            cmd.arg(format!("--unit={unit}"));
        }
        cmd.arg("--quiet");
        push_scope_properties(&mut cmd, &resolved);
        if let Some(slice) = &limits.slice {
            cmd.arg(format!("--slice={}", slice));
        }

        match &limits.sandbox {
            Some(sandbox) => {
                emit_sandbox_properties(
                    &mut cmd,
                    sandbox,
                    &self.path,
                    usage.as_ref().map(UsageTicket::pending_dir),
                );
                push_usage_grant(&mut cmd, usage.as_ref());
                let helper = crate::network::helper()
                    .map_err(|error| WorkspaceError::Other(error.to_string()))?;
                let socket = sandbox
                    .network
                    .socket()
                    .map_err(|error| WorkspaceError::Other(error.to_string()))?;
                // PrivateTmp/ProtectHome hide both locations unless explicitly
                // mounted. Only this policy's socket crosses the network boundary.
                for path in [&helper, &socket] {
                    cmd.arg("-p")
                        .arg(format!("BindReadOnlyPaths={}", path.display()));
                }
                let env = sandbox_environment(original, resolved.cargo_jobs);
                let mut forwarded = vec![
                    OsString::from(crate::network::FORWARDER_ARGUMENT),
                    socket.into_os_string(),
                ];
                forwarded.extend(payload_argv(&program, &orig_args, usage.as_ref()));
                append_sandbox_command(&mut cmd, &env, helper.as_os_str(), &forwarded);
            }
            None => {
                cmd.arg("--");
                cmd.args(payload_argv(&program, &orig_args, usage.as_ref()));
            }
        }

        // Set working directory
        cmd.current_dir(work_dir);

        // Forward and inherit all original environment variables. A sandboxed
        // command ignores these: its environment travels as `env -i` arguments.
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
            cmd.env("CARGO_BUILD_JOBS", resolved.cargo_jobs.to_string());
        }

        Ok((cmd, usage))
    }
}

/// Every limit a scoped command receives: explicit settings applied over values
/// derived from measured capacity and the command's recorded history.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ResolvedLimits {
    /// `CPUQuota`; present only when explicitly configured.
    pub cpu_quota: Option<String>,
    /// `CPUWeight`.
    pub cpu_weight: u32,
    /// `IOWeight`.
    pub io_weight: u32,
    /// `MemoryMax` in bytes, or the explicit setting.
    pub memory_max: String,
    /// `MemoryHigh` in bytes; absent when `memory_max` is explicit.
    pub memory_high: Option<String>,
    /// `MemorySwapMax`.
    pub memory_swap_max: String,
    /// `TasksMax`.
    pub tasks_max: u32,
    /// `CARGO_BUILD_JOBS` for the command.
    pub cargo_jobs: usize,
    /// Shared slice the scope joins.
    pub slice: Option<String>,
    /// Explicit aggregate CPU cap for the slice, if any.
    pub slice_cpu_quota: Option<String>,
    /// `MemoryMax` for the slice in bytes, or the explicit setting.
    pub slice_memory_max: String,
    /// `MemoryHigh` for the slice; absent when `slice_memory_max` is explicit.
    pub slice_memory_high: Option<String>,
    /// Whether the memory grant came from recorded history rather than the first-run rule.
    pub from_history: bool,
}

impl ResolvedLimits {
    /// `KEY=VALUE` systemd properties for the command's scope.
    pub fn scope_properties(&self) -> Vec<String> {
        let mut properties = vec![
            format!("CPUWeight={}", self.cpu_weight),
            format!("IOWeight={}", self.io_weight),
            format!("MemoryMax={}", self.memory_max),
            format!("MemorySwapMax={}", self.memory_swap_max),
            format!("TasksMax={}", self.tasks_max),
        ];
        if let Some(quota) = &self.cpu_quota {
            properties.push(format!("CPUQuota={quota}"));
        }
        if let Some(high) = &self.memory_high {
            properties.push(format!("MemoryHigh={high}"));
        }
        properties
    }
}

impl ResolvedLimits {
    /// Apply the shared slice's transient limits. A no-op without a slice.
    pub fn apply_slice(&self) -> Result<(), WorkspaceError> {
        match &self.slice {
            Some(slice) => configure_slice_properties(
                slice,
                self.slice_cpu_quota.as_deref(),
                &self.slice_memory_max,
                self.slice_memory_high.as_deref(),
            ),
            None => Ok(()),
        }
    }
}

impl ResourceLimits {
    /// Measure the host's capacity, the basis of every derived limit.
    pub fn capacity(&self) -> Result<MachineCapacity, WorkspaceError> {
        let slice = self.slice.as_deref().unwrap_or(DEFAULT_SLICE);
        MachineCapacity::measure(slice).map_err(capacity_error)
    }

    /// Resolve every limit for `program args` run in `repo`. Measures the host and
    /// reads the usage store, so the result reflects current capacity and history.
    pub fn resolve(
        &self,
        program: &OsStr,
        args: &[OsString],
        repo: &str,
    ) -> Result<ResolvedLimits, WorkspaceError> {
        let capacity = self.capacity()?;
        let class = command_class(program, args);
        let grant = command_grant(&capacity, self.usage_store.as_ref(), repo, &class)
            .map_err(capacity_error)?;
        let (memory_max, memory_high) = match &self.memory_max {
            Some(explicit) => (explicit.clone(), None),
            None => (
                grant.memory_max_bytes.to_string(),
                Some(grant.memory_high_bytes.to_string()),
            ),
        };
        let (slice_memory_max, slice_memory_high) = match &self.slice_memory_max {
            Some(explicit) => (explicit.clone(), None),
            None => (
                capacity.budget_bytes.to_string(),
                Some(capacity.slice_memory_high_bytes.to_string()),
            ),
        };
        Ok(ResolvedLimits {
            cpu_quota: self.cpu_quota.clone(),
            cpu_weight: DEFAULT_CPU_WEIGHT,
            io_weight: DEFAULT_IO_WEIGHT,
            memory_max,
            memory_high,
            memory_swap_max: self.memory_swap_max.clone(),
            tasks_max: grant.tasks_max,
            cargo_jobs: grant.cargo_jobs,
            slice: self.slice.clone(),
            slice_cpu_quota: self.slice_cpu_quota.clone(),
            slice_memory_max,
            slice_memory_high,
            from_history: grant.from_history,
        })
    }
}

fn capacity_error(error: std::io::Error) -> WorkspaceError {
    WorkspaceError::Capacity(error.to_string())
}

/// Append a scope's systemd properties as `-p KEY=VALUE` arguments.
fn push_scope_properties(cmd: &mut std::process::Command, resolved: &ResolvedLimits) {
    for property in resolved.scope_properties() {
        cmd.arg("-p").arg(property);
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

/// Select the transient unit type for a `systemd-run` invocation.
///
/// Resource caps work on both scopes and services, but namespace properties
/// (`ProtectSystem`, `ProtectHome`, `PrivateTmp`, `BindPaths`) are only
/// supported by services: `systemd-run --user --scope -p ProtectSystem=strict`
/// is rejected with `Unknown assignment`. A sandboxed command therefore runs as
/// a transient service, which additionally needs `--pipe` (forward stdio and
/// wait for the exit status), `--collect` (unload the unit afterwards), and
/// `-d` (a service does not inherit the caller's working directory).
fn append_systemd_unit_mode(cmd: &mut std::process::Command, sandboxed: bool) {
    if sandboxed {
        cmd.arg("--pipe");
        cmd.arg("--collect");
        cmd.arg("-d");
    } else {
        cmd.arg("--scope");
    }
}

/// Program that applies a sandboxed command's environment before it runs.
const SANDBOX_ENV_PROGRAM: &str = "/usr/bin/env";

/// Variables a sandboxed command inherits by exact name.
///
/// A sandboxed command starts from the user manager's environment, which can
/// carry the desktop session's credentials (`SSH_AUTH_SOCK`,
/// `DBUS_SESSION_BUS_ADDRESS`, `GPG_TTY`, API tokens). Only these names, and the
/// prefixes in [`SANDBOX_ENV_PREFIXES`], are forwarded from this process.
const SANDBOX_ENV_NAMES: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LANGUAGE",
    "TERM",
    "TZ",
    "JUTO_FLAKE",
    "IN_NIX_SHELL",
    "NIX_PATH",
    "NIX_PROFILES",
    "NIX_USER_PROFILE_DIR",
    "NIX_SSL_CERT_FILE",
    "SSL_CERT_FILE",
    "LOCALE_ARCHIVE",
    "PKG_CONFIG_PATH",
    "LD_LIBRARY_PATH",
    "LIBCLANG_PATH",
    "CC",
    "CXX",
    "AR",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "XDG_CACHE_HOME",
];

/// Name prefixes a sandboxed command inherits from this process.
const SANDBOX_ENV_PREFIXES: &[&str] = &["LC_", "CARGO_", "RUST_", "RUSTUP_", "CMAKE_"];

/// Words that mark a variable as a credential, even under an allowed prefix
/// (`CARGO_REGISTRIES_<name>_TOKEN`, for example).
const SANDBOX_ENV_SECRET_WORDS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
    "API_KEY",
];

/// Whether the variable `name` is forwarded into a sandboxed command.
fn sandbox_env_allowed(name: &str) -> bool {
    if SANDBOX_ENV_SECRET_WORDS
        .iter()
        .any(|word| name.contains(word))
    {
        return false;
    }
    SANDBOX_ENV_NAMES.contains(&name)
        || SANDBOX_ENV_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

/// The environment a sandboxed command runs with.
///
/// This process's allowlisted variables, with `original`'s explicit entries
/// applied (an explicit `None` unsets), and `CARGO_BUILD_JOBS` derived from the
/// cpu quota when no value is set.
fn sandbox_environment(
    original: &std::process::Command,
    jobs: usize,
) -> BTreeMap<OsString, OsString> {
    let mut env: BTreeMap<OsString, OsString> = std::env::vars_os()
        .filter(|(key, _)| key.to_str().is_some_and(sandbox_env_allowed))
        .collect();
    for (key, value) in original.get_envs() {
        match value {
            Some(value) => {
                env.insert(key.to_os_string(), value.to_os_string());
            }
            None => {
                env.remove(key);
            }
        }
    }
    env.entry(OsString::from("CARGO_BUILD_JOBS"))
        .or_insert_with(|| OsString::from(jobs.to_string()));
    env
}

/// Append the sandboxed program: `-- /usr/bin/env -i KEY=VALUE... program args`.
///
/// `env -i` clears the service's inherited environment, so the program sees
/// exactly `env`. The assignments travel as arguments, which is what makes an
/// unset possible: `--setenv` can only set variables, never remove them.
fn append_sandbox_command(
    cmd: &mut std::process::Command,
    env: &BTreeMap<OsString, OsString>,
    program: &OsStr,
    args: &[OsString],
) {
    cmd.arg("--").arg(SANDBOX_ENV_PROGRAM).arg("-i");
    for (key, value) in env {
        let mut assignment = key.clone();
        assignment.push("=");
        assignment.push(value);
        cmd.arg(assignment);
    }
    cmd.arg(program).args(args);
}

/// The argv a payload runs: the program and its arguments, wrapped by the
/// usage reporter when a ticket is attached.
fn payload_argv(program: &OsStr, args: &[OsString], usage: Option<&UsageTicket>) -> Vec<OsString> {
    let mut argv = Vec::new();
    if let Some(ticket) = usage {
        argv.push(OsString::from("sh"));
        argv.push(OsString::from("-c"));
        argv.push(OsString::from(USAGE_WRAPPER));
        argv.push(OsString::from("juto-usage"));
        argv.push(ticket.pending_file().as_os_str().to_os_string());
    }
    argv.push(program.to_os_string());
    argv.extend(args.iter().cloned());
    argv
}

/// Grant a sandboxed payload write access to its usage report directory, the
/// only path the reporter adds beyond the sandbox's own grants.
fn push_usage_grant(cmd: &mut std::process::Command, usage: Option<&UsageTicket>) {
    if let Some(ticket) = usage {
        let dir = ticket.pending_dir();
        let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        cmd.arg("-p").arg(format!("BindPaths={}", dir.display()));
    }
}

/// The paths to mask so that `denied` is inaccessible except along the way to `keep`.
///
/// Masking the whole directory would hide `keep` as well, and systemd cannot bind
/// a path beneath a masked directory, so the masks are the directory's entries
/// other than the chain leading to `keep`.
fn masked_paths(denied: &Path, keep: Option<&Path>) -> Vec<PathBuf> {
    let Some(keep) = keep.filter(|keep| keep.starts_with(denied) && *keep != denied) else {
        return vec![denied.to_path_buf()];
    };
    let mut masked = Vec::new();
    let Ok(entries) = std::fs::read_dir(denied) else {
        return masked;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep {
            continue;
        }
        if keep.starts_with(&path) {
            masked.extend(masked_paths(&path, Some(keep)));
        } else {
            masked.push(path);
        }
    }
    masked
}

/// Emit the `systemd-run` properties that implement `sandbox`.
///
/// Grants are bind mounts (`BindPaths`, `BindReadOnlyPaths`), not
/// `ReadWritePaths`: a bind creates its mount point inside the hidden home or
/// the private `/tmp`, where `ReadWritePaths` fails with status 226. Grants that
/// do not exist on the host are skipped. The workspace is always writable.
fn emit_sandbox_properties(
    cmd: &mut std::process::Command,
    sandbox: &Sandbox,
    workspace: &Path,
    keep: Option<&Path>,
) {
    cmd.arg("-p").arg("PrivateNetwork=yes");
    if sandbox.protect_system {
        cmd.arg("-p").arg("ProtectSystem=strict");
    }
    if sandbox.hide_home {
        cmd.arg("-p").arg("ProtectHome=tmpfs");
    }
    if sandbox.private_tmp {
        cmd.arg("-p").arg("PrivateTmp=yes");
    }
    for path in &sandbox.denied_paths {
        // A missing credential path is harmless. Existing paths remain
        // inaccessible even when a cache or explicit grant contains them.
        for masked in masked_paths(path, keep) {
            cmd.arg("-p")
                .arg(format!("InaccessiblePaths=-{}", masked.display()));
        }
    }

    let workspace = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let mut writable = vec![workspace.clone()];
    for path in &sandbox.write_paths {
        if let Some(resolved) = resolve_grant(path, &workspace)
            && !writable.contains(&resolved)
        {
            writable.push(resolved);
        }
    }
    for path in &writable {
        cmd.arg("-p").arg(format!("BindPaths={}", path.display()));
    }

    for path in &sandbox.read_paths {
        let Some(resolved) = resolve_grant(path, &workspace) else {
            continue;
        };
        // A read-only bind over a writable directory would hide its writes, so a
        // grant that contains a writable path is skipped: it is already writable.
        if writable.iter().any(|w| w.starts_with(&resolved)) {
            continue;
        }
        cmd.arg("-p")
            .arg(format!("BindReadOnlyPaths={}", resolved.display()));
    }
}

/// Resolve a grant against the workspace and canonicalize it; `None` when the
/// path does not exist on the host.
fn resolve_grant(path: &Path, workspace: &Path) -> Option<PathBuf> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };
    resolved.canonicalize().ok()
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

/// Applies the shared slice's limits at runtime only. A failed `systemctl` call
/// is returned so that unattended runs fail closed instead of running unbounded.
///
/// A property already set to a finite value is left alone, since a person may
/// have set it; only an unset (`infinity`) property is replaced. `--runtime`
/// keeps the value out of the persistent user drop-in directory.
fn configure_slice_properties(
    slice: &str,
    cpu_quota: Option<&str>,
    mem_max: &str,
    mem_high: Option<&str>,
) -> Result<(), WorkspaceError> {
    let mut props: Vec<String> = Vec::new();
    if let Some(quota) = cpu_quota
        && slice_property_unset(slice, "CPUQuotaPerSecUSec")
    {
        props.push(format!("CPUQuota={quota}"));
    }
    if slice_property_unset(slice, "MemoryMax") {
        props.push(format!("MemoryMax={mem_max}"));
    }
    if let Some(high) = mem_high
        && slice_property_unset(slice, "MemoryHigh")
    {
        props.push(format!("MemoryHigh={high}"));
    }
    if props.is_empty() {
        return Ok(());
    }
    let mut cmd = std::process::Command::new("systemctl");
    cmd.arg("--user")
        .arg("--runtime")
        .arg("set-property")
        .arg(slice)
        .args(props);
    let output = cmd
        .output()
        .map_err(|error| WorkspaceError::Slice(error.to_string()))?;
    if !output.status.success() {
        return Err(WorkspaceError::Slice(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(())
}

fn slice_property_unset(slice: &str, property: &str) -> bool {
    let value = std::process::Command::new("systemctl")
        .arg("--user")
        .arg("show")
        .arg(slice)
        .arg("-p")
        .arg(property)
        .arg("--value")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    value.is_empty() || value == "infinity"
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
        assert_eq!(limits.cpu_quota, None);
        assert_eq!(limits.memory_max, None);
        assert_eq!(limits.memory_swap_max, DEFAULT_MEMORY_SWAP_MAX);
        assert_eq!(limits.slice.as_deref(), Some(DEFAULT_SLICE));
        assert!(!limits.gpu);
        assert!(limits.sandbox.is_none());

        let customized = limits
            .with_cpu_quota("200%")
            .with_memory_max("4G")
            .with_gpu(true)
            .with_sandbox(Sandbox::default());

        assert_eq!(customized.cpu_quota.as_deref(), Some("200%"));
        assert_eq!(customized.memory_max.as_deref(), Some("4G"));
        assert!(customized.gpu);
        assert_eq!(customized.sandbox, Some(Sandbox::default()));
    }

    #[test]
    fn test_scope_properties_derive_weights_and_omit_unset_quota() {
        let resolved = ResolvedLimits {
            cpu_quota: None,
            cpu_weight: DEFAULT_CPU_WEIGHT,
            io_weight: DEFAULT_IO_WEIGHT,
            memory_max: "2147483648".to_string(),
            memory_high: Some("1932735283".to_string()),
            memory_swap_max: DEFAULT_MEMORY_SWAP_MAX.to_string(),
            tasks_max: 512,
            cargo_jobs: 2,
            slice: Some(DEFAULT_SLICE.to_string()),
            slice_cpu_quota: None,
            slice_memory_max: "21474836480".to_string(),
            slice_memory_high: Some("19327352832".to_string()),
            from_history: false,
        };
        let properties = resolved.scope_properties();
        assert!(properties.contains(&"CPUWeight=100".to_string()));
        assert!(properties.contains(&"MemoryMax=2147483648".to_string()));
        assert!(properties.contains(&"MemoryHigh=1932735283".to_string()));
        assert!(!properties.iter().any(|p| p.starts_with("CPUQuota=")));

        let capped = ResolvedLimits {
            cpu_quota: Some("200%".to_string()),
            memory_high: None,
            ..resolved
        };
        let properties = capped.scope_properties();
        assert!(properties.contains(&"CPUQuota=200%".to_string()));
        assert!(!properties.iter().any(|p| p.starts_with("MemoryHigh=")));
    }

    #[test]
    fn test_resolve_explicit_limits_override_derived_values() {
        let limits = ResourceLimits::default()
            .with_cpu_quota("200%")
            .with_memory_max("4G");
        let resolved = limits
            .resolve(std::ffi::OsStr::new("cargo"), &[], "/repo")
            .expect("resolve");
        assert_eq!(resolved.cpu_quota.as_deref(), Some("200%"));
        assert_eq!(resolved.memory_max, "4G");
        assert_eq!(resolved.memory_high, None);
        assert_eq!(resolved.cpu_weight, DEFAULT_CPU_WEIGHT);
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
    fn unbounded_memory_limit_is_refused_before_admission() {
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
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("true");
        let mut limits = ResourceLimits::default().with_memory_max("infinity");
        limits.broker = Some(Broker::new(dir.path().join("broker")));

        let error = ws
            .admission(&cmd, &limits)
            .err()
            .expect("an unbounded limit must not be admitted");
        assert!(
            error.to_string().contains("cannot be reserved"),
            "refusal must name the unreservable limit: {error}"
        );
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
        configure_slice_properties("agents.slice", Some("200%"), "2G", Some("1G"))
            .expect("existing slice limits are left alone");
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
        assert_eq!(limits.cpu_quota, None);
        assert_eq!(limits.memory_max, None);
        assert_eq!(limits.slice.as_deref(), Some(DEFAULT_SLICE));
        assert_eq!(limits.slice_cpu_quota, None);
        assert_eq!(limits.slice_memory_max, None);
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
            .child_mut()
            .stdout
            .take()
            .expect("stdout must be piped")
            .read_to_string(&mut stdout)
            .await
            .expect("read stdout");
        let mut stderr = String::new();
        child
            .child_mut()
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
