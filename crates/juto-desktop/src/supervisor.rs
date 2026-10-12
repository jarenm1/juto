//! Desktop environment supervisor: admission, ownership, lifecycle, application
//! launch, observation, and cleanup.
//!
//! One [`DesktopEnvironment`] owns a private runtime directory, a headless
//! Smithay compositor, and the applications launched into it. Applications run
//! in a per-environment systemd scope (or an explicit, loudly logged unscoped
//! development mode) and are terminated as a process tree on destroy.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

use crate::compositor::{Compositor, CompositorCommand, CompositorConfig, SnapshotInfo};
use crate::error::DesktopError;
use crate::protocol::*;
use crate::shm::{ShmFramePool, ShmReader};
use crate::worker::{WorkerCompositor, WorkerRequest};

/// Environment variables inherited by launched applications.
const ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_NUMERIC",
    "TERM",
    "TZ",
];

/// Default grace period before forced termination of an application tree.
const TERMINATE_GRACE: Duration = Duration::from_secs(5);

/// Output resolution and framerate settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OutputConfig {
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub fps: u32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            scale: 1,
            fps: 30,
        }
    }
}

/// Where the compositor runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CompositorMode {
    /// A separate process inside the environment slice (production).
    Worker,
    /// A thread in the supervisor process (tests and hosts without the worker
    /// binary).
    InProcess,
}

fn default_compositor_mode() -> CompositorMode {
    CompositorMode::Worker
}

/// Rendering profile: software LLVMpipe or hardware render node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RenderProfile {
    Software,
    Hardware,
}

impl std::str::FromStr for RenderProfile {
    type Err = DesktopError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "software" => Ok(RenderProfile::Software),
            "hardware" => Ok(RenderProfile::Hardware),
            other => Err(DesktopError::Protocol(format!(
                "unknown render profile: {other}"
            ))),
        }
    }
}

/// Profile configurations admitted by the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputProfile {
    pub max_width: u32,
    pub max_height: u32,
    pub max_fps: u32,
    pub max_pixel_rate: u64,
}

impl RenderProfile {
    /// Stable wire/CLI spelling for the profile.
    pub fn as_str(&self) -> &'static str {
        match self {
            RenderProfile::Software => "software",
            RenderProfile::Hardware => "hardware",
        }
    }

    pub fn admitted_profile(&self) -> OutputProfile {
        match self {
            RenderProfile::Software => OutputProfile {
                max_width: 3840,
                max_height: 2160,
                max_fps: 60,
                max_pixel_rate: 62_208_000, // 1080p30 or 720p60
            },
            RenderProfile::Hardware => OutputProfile {
                max_width: 3840,
                max_height: 2160,
                max_fps: 120,
                max_pixel_rate: 497_664_000, // 4K60
            },
        }
    }
}

/// Resource constraints enforced on the environment process tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemLimits {
    pub cpu_quota: String,
    pub memory_max: String,
    pub memory_swap_max: String,
    pub tasks_max: u32,
    pub remote_bitrate_max_bps: u64,
}

impl Default for SystemLimits {
    fn default() -> Self {
        Self {
            cpu_quota: "400%".to_string(),
            memory_max: "8G".to_string(),
            memory_swap_max: "0".to_string(),
            tasks_max: 512,
            remote_bitrate_max_bps: 12_000_000,
        }
    }
}

/// Configuration specified when creating an agent desktop environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentConfig {
    pub env_id: String,
    pub owner_principal: String,
    pub output: OutputConfig,
    pub render_profile: RenderProfile,
    pub limits: SystemLimits,
    pub workspace_dir: PathBuf,
    /// Control-lease lifetime in seconds (spec section 8: heartbeat 2s, expiry 6s).
    #[serde(default = "default_lease_ttl")]
    pub lease_ttl_secs: u64,
    /// Where the compositor runs.
    #[serde(default = "default_compositor_mode")]
    pub compositor_mode: CompositorMode,
    /// Owner heartbeat timeout in seconds; 0 disables expiry (spec section 3).
    #[serde(default = "default_owner_timeout")]
    pub owner_heartbeat_timeout_secs: u64,
}

fn default_owner_timeout() -> u64 {
    60
}

/// Bounded wait for the shared GPU lock before reporting the device busy.
const GPU_LOCK_WAIT_SECS: u64 = 5;

/// Path of the repository's shared advisory GPU lock, shared with `jtech`.
fn gpu_lock_path() -> PathBuf {
    if let Some(path) = std::env::var_os("AGENT_GPU_LOCK") {
        return PathBuf::from(path);
    }
    let base = std::env::var_os("XDG_RUNTIME_DIR").unwrap_or_else(|| PathBuf::from("/tmp").into());
    PathBuf::from(base).join("jtech-agent-gpu.lock")
}

/// Acquires the shared advisory GPU lock, waiting a bounded time for a peer.
fn acquire_gpu_lock() -> Result<std::fs::File, DesktopError> {
    let path = gpu_lock_path();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|source| DesktopError::Io {
            path: path.clone(),
            source,
        })?;
    let deadline = Instant::now() + Duration::from_secs(GPU_LOCK_WAIT_SECS);
    loop {
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return Ok(file);
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(DesktopError::Io { path, source: err });
        }
        if Instant::now() >= deadline {
            return Err(DesktopError::GpuBusy);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// A bounded human acceptance of an environment's lifetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Retention {
    pub human_owner: String,
    pub expires_at: u64,
}

/// Maximum retention a host grants without further approval (spec section 4).
const MAX_RETENTION_SECS: u64 = 1800;

fn default_lease_ttl() -> u64 {
    6
}

/// Live status reported by the supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentStatus {
    pub env_id: String,
    pub owner_principal: String,
    pub output: OutputConfig,
    pub generation: GeometryGeneration,
    pub is_ready: bool,
    pub active_control_lease: Option<ControlLease>,
    pub mapped_windows: Vec<WindowInfo>,
    pub active_grants: Vec<Grant>,
}

/// A launched application tracked by the environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppHandle {
    pub app_id: String,
    pub pid: u32,
    pub unit: Option<String>,
    pub argv: Vec<String>,
}

struct InnerEnvironment {
    config: EnvironmentConfig,
    runtime_dir: PathBuf,
    socket_path: PathBuf,
    generation: AtomicU64,
    output: Mutex<OutputConfig>,
    is_ready: Mutex<bool>,
    lease: Mutex<Option<ControlLease>>,
    grants: Mutex<HashMap<String, Grant>>,
    windows: Mutex<HashMap<u32, WindowInfo>>,
    pool: Arc<ShmReader>,
    compositor: Mutex<Option<CompositorHandle>>,
    apps: Mutex<HashMap<String, AppHandle>>,
    events: broadcast::Sender<DesktopEvent>,
    last_automation_barrier: Mutex<Option<AutomationBarrier>>,
    /// Viewers that have presented a frame of the current geometry generation.
    geometry_ready: Mutex<HashMap<String, GeometryGeneration>>,
    /// Last owner heartbeat, in seconds since the epoch.
    owner_heartbeat: Mutex<u64>,
    /// Explicit human retention of the environment's lifetime.
    retention: Mutex<Option<Retention>>,
    destroyed: AtomicBool,
    use_scope: bool,
    slice: Option<String>,
    /// Shared advisory GPU lock held for a hardware environment's lifetime.
    gpu_lock: Mutex<Option<std::fs::File>>,
}

/// An active, isolated agent desktop environment.
#[derive(Clone)]
pub struct DesktopEnvironment {
    inner: Arc<InnerEnvironment>,
}

impl DesktopEnvironment {
    pub fn new(config: EnvironmentConfig) -> Result<Self, DesktopError> {
        let admitted = config.render_profile.admitted_profile();
        let pixel_rate = (config.output.width as u64)
            * (config.output.height as u64)
            * (config.output.fps as u64);
        if pixel_rate > admitted.max_pixel_rate {
            return Err(DesktopError::InvalidOutputMode {
                width: config.output.width,
                height: config.output.height,
                reason: format!(
                    "requested pixel rate {} px/s exceeds admitted limit {} px/s for {:?}",
                    pixel_rate, admitted.max_pixel_rate, config.render_profile
                ),
            });
        }
        if config.output.width == 0 || config.output.height == 0 || config.output.fps == 0 {
            return Err(DesktopError::InvalidOutputMode {
                width: config.output.width,
                height: config.output.height,
                reason: "width, height, and fps must be non-zero".to_string(),
            });
        }

        // A hardware environment serializes on the repository's shared advisory
        // GPU lock for its whole lifetime, including its encoder (spec section 6).
        let gpu_lock = if config.render_profile == RenderProfile::Hardware {
            Some(acquire_gpu_lock()?)
        } else {
            None
        };

        let use_scope = scope_enforced()?;
        let slice = if use_scope {
            Some(create_environment_slice(&config.env_id, &config.limits)?)
        } else {
            None
        };

        let runtime_dir = environment_runtime_dir(&config.env_id);
        prepare_runtime_dir(&runtime_dir)?;
        let socket_path = runtime_dir.join("wayland-0");
        // Unix socket paths are limited to SUN_LEN (108 bytes including the
        // terminator); fail with a clear error rather than a bind failure.
        if socket_path.as_os_str().len() >= 108 {
            let _ = std::fs::remove_dir_all(&runtime_dir);
            return Err(DesktopError::Protocol(format!(
                "environment `{}` runtime path is too long for a Unix socket ({} bytes, limit 107): {}",
                config.env_id,
                socket_path.as_os_str().len(),
                socket_path.display()
            )));
        }

        let (events, _) = broadcast::channel(256);
        let (comp_tx, comp_rx) = std_mpsc::channel::<DesktopEvent>();

        // The compositor runs in its own process inside the environment slice, so
        // its memory is charged to the environment and its failure cannot take
        // the supervisor down.
        let compositor_config = CompositorConfig {
            env_id: config.env_id.clone(),
            width: config.output.width,
            height: config.output.height,
            scale: config.output.scale,
            fps: config.output.fps,
            render_profile: config.render_profile,
            socket_path: socket_path.clone(),
        };
        let (compositor, pool) = match config.compositor_mode {
            CompositorMode::Worker => {
                let compositor_unit =
                    format!("juto-desktop-{}-compositor.scope", sanitize(&config.env_id));
                let scope = slice
                    .as_deref()
                    .map(|slice| (slice, compositor_unit.as_str()));
                match WorkerCompositor::spawn(compositor_config, comp_tx, scope) {
                    Ok(compositor) => {
                        let reader = compositor.reader();
                        (CompositorHandle::Worker(Arc::new(compositor)), reader)
                    }
                    Err(err) => {
                        if let Some(slice) = slice.as_ref() {
                            let _ = Command::new("systemctl")
                                .args(["--user", "stop", slice])
                                .stdout(Stdio::null())
                                .stderr(Stdio::null())
                                .status();
                        }
                        let _ = std::fs::remove_dir_all(&runtime_dir);
                        return Err(err);
                    }
                }
            }
            CompositorMode::InProcess => {
                let pool = match ShmFramePool::create(
                    format!("{}-primary", config.env_id),
                    config.output.width,
                    config.output.height,
                ) {
                    Ok(pool) => Arc::new(pool),
                    Err(err) => {
                        let _ = std::fs::remove_dir_all(&runtime_dir);
                        return Err(err);
                    }
                };
                let compositor = match Compositor::spawn(compositor_config, pool.clone(), comp_tx) {
                    Ok(compositor) => compositor,
                    Err(err) => {
                        let _ = std::fs::remove_dir_all(&runtime_dir);
                        return Err(err);
                    }
                };
                let fd = unsafe { libc::dup(pool.raw_fd()) };
                if fd < 0 {
                    let _ = std::fs::remove_dir_all(&runtime_dir);
                    return Err(DesktopError::Shm(
                        "cannot duplicate the pool fd".to_string(),
                    ));
                }
                let reader = Arc::new(ShmReader::attach_named(
                    fd,
                    pool.size(),
                    pool.pool_id().to_string(),
                    None,
                )?);
                (CompositorHandle::InProcess(Arc::new(compositor)), reader)
            }
        };

        let inner = Arc::new(InnerEnvironment {
            config: config.clone(),
            runtime_dir,
            socket_path,
            generation: AtomicU64::new(1),
            output: Mutex::new(config.output.clone()),
            is_ready: Mutex::new(true),
            lease: Mutex::new(None),
            grants: Mutex::new(HashMap::new()),
            windows: Mutex::new(HashMap::new()),
            pool,
            compositor: Mutex::new(Some(compositor)),
            apps: Mutex::new(HashMap::new()),
            events: events.clone(),
            last_automation_barrier: Mutex::new(None),
            geometry_ready: Mutex::new(HashMap::new()),
            owner_heartbeat: Mutex::new(now_secs()),
            retention: Mutex::new(None),
            destroyed: AtomicBool::new(false),
            use_scope,
            slice,
            gpu_lock: Mutex::new(gpu_lock),
        });

        spawn_event_forwarder(inner.clone(), comp_rx);
        spawn_lease_reaper(Arc::downgrade(&inner));
        spawn_pressure_monitor(Arc::downgrade(&inner));

        let _ = events.send(DesktopEvent::Ready {
            env_id: config.env_id.clone(),
        });

        Ok(Self { inner })
    }

    pub fn env_id(&self) -> &str {
        &self.inner.config.env_id
    }

    /// Absolute path of the environment's private Wayland socket.
    pub fn socket_path(&self) -> &Path {
        &self.inner.socket_path
    }

    /// Private runtime directory (`XDG_RUNTIME_DIR`) for the environment.
    pub fn runtime_dir(&self) -> &Path {
        &self.inner.runtime_dir
    }

    /// The transient slice that aggregates the environment's limits.
    pub fn slice(&self) -> Option<&str> {
        self.inner.slice.as_deref()
    }

    /// The environment's primary composed-frame pool, read-only.
    ///
    /// The compositor worker owns the writable mapping; every reader in this
    /// process maps it read-only and signals slot reservations back to the
    /// worker.
    pub fn frame_pool(&self) -> Arc<ShmReader> {
        self.inner.pool.clone()
    }

    pub fn generation(&self) -> GeometryGeneration {
        self.inner.generation.load(Ordering::SeqCst)
    }

    pub fn output(&self) -> OutputConfig {
        self.inner.output.lock().clone()
    }

    /// Effective resource limits for the environment.
    pub fn limits(&self) -> SystemLimits {
        self.inner.config.limits.clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DesktopEvent> {
        self.inner.events.subscribe()
    }

    pub fn is_destroyed(&self) -> bool {
        self.inner.destroyed.load(Ordering::SeqCst)
    }

    /// Issues an authenticated grant for a principal.
    pub fn issue_grant(
        &self,
        principal: String,
        roles: Vec<GrantRole>,
        audience: String,
        expiry_seconds: u64,
    ) -> Grant {
        let now_s = now_secs();
        let grant_id = format!("grant-{}", uuid::Uuid::now_v7());
        let grant = Grant {
            grant_id: grant_id.clone(),
            principal,
            roles,
            audience,
            expires_at: now_s + expiry_seconds,
        };
        self.inner.grants.lock().insert(grant_id, grant.clone());
        grant
    }

    /// Looks up a grant by id, including one that has expired.
    pub fn grant(&self, grant_id: &str) -> Option<Grant> {
        self.inner.grants.lock().get(grant_id).cloned()
    }

    /// Revokes an existing grant and releases control held through it.
    pub fn revoke_grant(&self, grant_id: &str) -> bool {
        let mut grants = self.inner.grants.lock();
        if let Some(grant) = grants.remove(grant_id) {
            let mut lease = self.inner.lease.lock();
            if let Some(l) = lease.as_ref()
                && l.holder == grant.principal
            {
                *lease = None;
                let _ = self.inner.events.send(DesktopEvent::ControlChanged {
                    env_id: self.env_id().to_string(),
                    lease: None,
                });
            }
            true
        } else {
            false
        }
    }

    /// Acquires or releases exclusive interactive control.
    pub fn set_control(
        &self,
        caller: &str,
        action: &str,
        is_human: bool,
    ) -> Result<Option<ControlLease>, DesktopError> {
        let mut lease = self.inner.lease.lock();
        match action {
            "acquire" => {
                if let Some(current) = lease.as_ref() {
                    if current.holder == caller && current.is_human == is_human {
                        // Renewal: same holder keeps its epoch and extends the
                        // deadline, so a heartbeat cannot invalidate its own lease.
                        let renewed = ControlLease {
                            expires_at: now_secs() + self.inner.config.lease_ttl_secs.max(1),
                            ..current.clone()
                        };
                        *lease = Some(renewed.clone());
                        return Ok(Some(renewed));
                    }
                    if (!is_human && current.is_human)
                        || (!is_human && !current.is_human && current.holder != caller)
                    {
                        return Err(DesktopError::ControlBusy {
                            holder: current.holder.clone(),
                            epoch: current.epoch,
                        });
                    }
                }
                let new_epoch = lease.as_ref().map(|l| l.epoch + 1).unwrap_or(1);
                let new_lease = ControlLease {
                    holder: caller.to_string(),
                    epoch: new_epoch,
                    expires_at: now_secs() + self.inner.config.lease_ttl_secs.max(1),
                    is_human,
                };
                *lease = Some(new_lease.clone());
                let _ = self.inner.events.send(DesktopEvent::ControlChanged {
                    env_id: self.env_id().to_string(),
                    lease: Some(new_lease.clone()),
                });
                Ok(Some(new_lease))
            }
            "release" => {
                *lease = None;
                let _ = self.inner.events.send(DesktopEvent::ControlChanged {
                    env_id: self.env_id().to_string(),
                    lease: None,
                });
                Ok(None)
            }
            _ => Err(DesktopError::Protocol(format!(
                "unknown control action `{action}`"
            ))),
        }
    }

    /// Confirms that an in-flight automation batch has quiesced before handoff.
    pub fn acknowledge_automation_quiescence(&self, old_epoch: u64, barrier_id: String) {
        let barrier = AutomationBarrier {
            barrier_id,
            old_epoch,
            drained_at: now_secs(),
        };
        *self.inner.last_automation_barrier.lock() = Some(barrier);
    }

    /// Launches an application into the environment.
    pub fn launch(
        &self,
        app_id: &str,
        argv: &[String],
        cwd: Option<&Path>,
        env_overrides: &[(String, String)],
    ) -> Result<AppHandle, DesktopError> {
        if self.inner.destroyed.load(Ordering::SeqCst) {
            return Err(DesktopError::EnvironmentFailed(
                self.env_id().to_string(),
                "environment is destroyed".to_string(),
            ));
        }
        if argv.is_empty() {
            return Err(DesktopError::ExecutableUnavailable {
                path: PathBuf::new(),
                reason: "empty argv".to_string(),
            });
        }
        {
            let apps = self.inner.apps.lock();
            if apps.contains_key(app_id) {
                return Err(DesktopError::Protocol(format!(
                    "application `{app_id}` is already running"
                )));
            }
        }

        let program = resolve_executable(&argv[0])?;
        let work_dir = match cwd {
            Some(dir) => {
                let dir = dir.to_path_buf();
                if !dir.starts_with(&self.inner.config.workspace_dir) {
                    return Err(DesktopError::Protocol(format!(
                        "cwd `{}` escapes the workspace `{}`",
                        dir.display(),
                        self.inner.config.workspace_dir.display()
                    )));
                }
                dir
            }
            None => self.inner.config.workspace_dir.clone(),
        };

        let unit = self
            .inner
            .use_scope
            .then(|| scope_unit_name(&self.inner.config.env_id, app_id));

        // The application environment is built explicitly: an allowlist of
        // benign host variables plus environment-private paths and explicit
        // overrides. No host display, session bus, SSH agent, or provider
        // secrets are inherited.
        let home = self.inner.runtime_dir.join("home");
        let mut app_env: Vec<(String, String)> = Vec::new();
        for key in ENV_ALLOWLIST {
            if let Ok(value) = std::env::var(key) {
                app_env.push((key.to_string(), value));
            }
        }
        app_env.push((
            "XDG_RUNTIME_DIR".to_string(),
            self.inner.runtime_dir.display().to_string(),
        ));
        app_env.push((
            "WAYLAND_DISPLAY".to_string(),
            self.inner.socket_path.display().to_string(),
        ));
        app_env.push(("HOME".to_string(), home.display().to_string()));
        app_env.push((
            "XDG_CONFIG_HOME".to_string(),
            home.join(".config").display().to_string(),
        ));
        app_env.push((
            "XDG_CACHE_HOME".to_string(),
            home.join(".cache").display().to_string(),
        ));
        app_env.push((
            "XDG_DATA_HOME".to_string(),
            home.join(".local/share").display().to_string(),
        ));
        app_env.push((
            "XDG_STATE_HOME".to_string(),
            home.join(".local/state").display().to_string(),
        ));
        app_env.push((
            "TMPDIR".to_string(),
            self.inner.runtime_dir.join("tmp").display().to_string(),
        ));
        app_env.push(("PWD".to_string(), work_dir.display().to_string()));
        for (key, value) in env_overrides {
            app_env.retain(|(k, _)| k != key);
            app_env.push((key.clone(), value.clone()));
        }

        let mut command = if let Some(unit) = unit.as_ref() {
            let limits = &self.inner.config.limits;
            let mut c = Command::new("systemd-run");
            c.args(["--user", "--scope", "--quiet", "--collect"]);
            c.arg(format!("--unit={unit}"));
            if let Some(slice) = self.inner.slice.as_ref() {
                c.arg(format!("--slice={slice}"));
            }
            c.arg("-p").arg(format!("CPUQuota={}", limits.cpu_quota));
            c.arg("-p").arg(format!("MemoryMax={}", limits.memory_max));
            c.arg("-p")
                .arg(format!("MemorySwapMax={}", limits.memory_swap_max));
            c.arg("-p").arg(format!("TasksMax={}", limits.tasks_max));
            c.arg("--");
            // `env -i` gives the application exactly the approved environment
            // while systemd-run itself keeps the session variables it needs to
            // reach the user manager.
            c.arg("env").arg("-i");
            for (key, value) in &app_env {
                c.arg(format!("{key}={value}"));
            }
            c.arg(&program);
            c.args(&argv[1..]);
            c
        } else {
            let mut c = Command::new(&program);
            c.args(&argv[1..]);
            c.env_clear();
            for (key, value) in &app_env {
                c.env(key, value);
            }
            c
        };

        command
            .current_dir(&work_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let child = command
            .spawn()
            .map_err(|e| DesktopError::ExecutableUnavailable {
                path: program.clone(),
                reason: e.to_string(),
            })?;
        let pid = child.id();
        let handle = AppHandle {
            app_id: app_id.to_string(),
            pid,
            unit: unit.clone(),
            argv: argv.to_vec(),
        };

        self.inner
            .apps
            .lock()
            .insert(app_id.to_string(), handle.clone());

        if let Some(unit) = unit.as_ref()
            && let Err(err) = verify_scope(unit, &self.inner.config.limits)
        {
            let _ = Command::new("systemctl")
                .args(["--user", "stop", unit])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            self.inner.apps.lock().remove(app_id);
            return Err(err);
        }

        spawn_app_waiter(Arc::downgrade(&self.inner), app_id.to_string(), unit, child);
        Ok(handle)
    }

    /// Terminates a launched application and its descendants.
    pub fn terminate(&self, app_id: &str) -> Result<(), DesktopError> {
        let mut apps = self.inner.apps.lock();
        let handle = apps
            .remove(app_id)
            .ok_or_else(|| DesktopError::Protocol(format!("unknown application `{app_id}`")))?;
        drop(apps);
        terminate_app(&handle);
        Ok(())
    }

    /// Returns the list of running applications.
    pub fn apps(&self) -> Vec<AppHandle> {
        self.inner.apps.lock().values().cloned().collect()
    }

    /// Dispatches input events to the virtual seat.
    pub fn dispatch_input(
        &self,
        caller: &str,
        epoch: u64,
        generation: GeometryGeneration,
        batch: InputBatch,
    ) -> Result<u64, DesktopError> {
        let lease = self.inner.lease.lock();
        let current_lease = lease.as_ref().ok_or_else(|| DesktopError::Unauthorized {
            caller: caller.to_string(),
            role: "Control".to_string(),
            environment: self.env_id().to_string(),
        })?;
        if current_lease.epoch != epoch || current_lease.holder != caller {
            return Err(DesktopError::StaleLease {
                expected: current_lease.epoch,
                received: epoch,
            });
        }
        if current_lease.expires_at <= now_secs() {
            return Err(DesktopError::StaleLease {
                expected: current_lease.epoch,
                received: epoch,
            });
        }
        let current_gen = self.generation();
        if current_gen != generation {
            return Err(DesktopError::StaleGeneration {
                expected: current_gen,
                received: generation,
            });
        }
        // After a resize, pointer input resumes only once a viewer has presented
        // a frame of the new generation (spec section 6). There is exactly one
        // active control lease, so the acknowledging viewer is the one whose
        // input is gated; the acknowledgement is tracked per environment.
        if current_gen > 1 && self.inner.geometry_ready.lock().is_empty() {
            return Err(DesktopError::StaleGeneration {
                expected: current_gen,
                received: generation,
            });
        }
        drop(lease);

        let compositor = self.compositor()?;
        for event in batch.events {
            compositor.send(WorkerRequest::Input(event))?;
        }
        Ok(batch.seq)
    }

    /// Renews the owner lifetime or the control lease.
    ///
    /// Returns the renewed deadline. A lease heartbeat must match the current
    /// holder and epoch; input traffic never renews either deadline implicitly.
    pub fn heartbeat(&self, target: &str, epoch: Option<u64>) -> Result<u64, DesktopError> {
        match target {
            "owner" => {
                let now = now_secs();
                *self.inner.owner_heartbeat.lock() = now;
                Ok(now + self.inner.config.owner_heartbeat_timeout_secs)
            }
            "lease" => {
                let mut lease = self.inner.lease.lock();
                let current = lease.as_ref().ok_or_else(|| DesktopError::Unauthorized {
                    caller: "heartbeat".to_string(),
                    role: "Control".to_string(),
                    environment: self.env_id().to_string(),
                })?;
                if let Some(epoch) = epoch
                    && epoch != current.epoch
                {
                    return Err(DesktopError::StaleLease {
                        expected: current.epoch,
                        received: epoch,
                    });
                }
                let renewed = ControlLease {
                    expires_at: now_secs() + self.inner.config.lease_ttl_secs.max(1),
                    ..current.clone()
                };
                *lease = Some(renewed.clone());
                Ok(renewed.expires_at)
            }
            other => Err(DesktopError::Protocol(format!(
                "unknown heartbeat target `{other}`"
            ))),
        }
    }

    /// Transfers lifetime responsibility to an authenticated human session.
    ///
    /// The deadline is host-bounded; an owner cannot nominate an unbounded
    /// retention.
    pub fn retain(
        &self,
        human_owner: &str,
        deadline_seconds: u64,
    ) -> Result<Retention, DesktopError> {
        if human_owner.trim().is_empty() {
            return Err(DesktopError::Protocol(
                "retention requires a human owner".to_string(),
            ));
        }
        let bounded = deadline_seconds.clamp(1, MAX_RETENTION_SECS);
        let retention = Retention {
            human_owner: human_owner.to_string(),
            expires_at: now_secs() + bounded,
        };
        *self.inner.retention.lock() = Some(retention.clone());
        Ok(retention)
    }

    /// The active retention, if any.
    pub fn retention(&self) -> Option<Retention> {
        self.inner
            .retention
            .lock()
            .clone()
            .filter(|retention| retention.expires_at > now_secs())
    }

    /// Records that a viewer has presented a frame of `generation`.
    ///
    /// Input from that viewer resumes only after this acknowledgement, so a
    /// click cannot land on a stale frame at the same encoded size.
    pub fn acknowledge_geometry(
        &self,
        viewer: &str,
        generation: GeometryGeneration,
    ) -> Result<(), DesktopError> {
        let current = self.generation();
        if generation != current {
            return Err(DesktopError::StaleGeneration {
                expected: current,
                received: generation,
            });
        }
        self.inner
            .geometry_ready
            .lock()
            .insert(viewer.to_string(), generation);
        Ok(())
    }

    /// Forces a fresh composition and returns the resulting frame metadata.
    pub fn observe(&self, timeout: Duration) -> Result<SnapshotInfo, DesktopError> {
        self.compositor()?.snapshot(timeout)
    }

    /// Observes a fresh frame and encodes it as a lossless PNG.
    ///
    /// The returned image is the composed desktop, not a streamed frame: it is
    /// suitable as durable evidence for a caller.
    pub fn snapshot_png(&self, timeout: Duration) -> Result<(SnapshotInfo, Vec<u8>), DesktopError> {
        let info = self.observe(timeout)?;
        let pool = self.frame_pool();
        pool.acquire_slot(info.slot);
        let read = pool.read_slot(info.slot, pool.slot_size());
        pool.release_slot(info.slot);
        let (header, pixels) = read?;
        let png = crate::png::encode_bgra(header.width, header.height, pixels)?;
        Ok((info, png))
    }

    /// Reconfigures output mode atomically with a new geometry generation.
    pub fn configure_output(
        &self,
        expected_generation: GeometryGeneration,
        new_mode: OutputConfig,
    ) -> Result<GeometryGeneration, DesktopError> {
        let current_gen = self.generation();
        if expected_generation != current_gen {
            return Err(DesktopError::StaleGeneration {
                expected: current_gen,
                received: expected_generation,
            });
        }
        let admitted = self.inner.config.render_profile.admitted_profile();
        let pixel_rate = (new_mode.width as u64) * (new_mode.height as u64) * (new_mode.fps as u64);
        if pixel_rate > admitted.max_pixel_rate {
            return Err(DesktopError::InvalidOutputMode {
                width: new_mode.width,
                height: new_mode.height,
                reason: format!("exceeds admitted rate {}", admitted.max_pixel_rate),
            });
        }

        let new_gen = current_gen + 1;
        let compositor = self.compositor()?;
        compositor.send(WorkerRequest::Configure {
            width: new_mode.width,
            height: new_mode.height,
            scale: new_mode.scale,
            fps: new_mode.fps,
            generation: new_gen,
        })?;
        *self.inner.output.lock() = new_mode;
        self.inner.generation.store(new_gen, Ordering::SeqCst);
        // Every viewer must present a new-generation frame before input resumes.
        self.inner.geometry_ready.lock().clear();
        Ok(new_gen)
    }

    pub fn get_status(&self) -> EnvironmentStatus {
        let now = now_secs();
        let lease = self
            .inner
            .lease
            .lock()
            .clone()
            .filter(|lease| lease.expires_at > now);
        let grants = self
            .inner
            .grants
            .lock()
            .values()
            .filter(|grant| grant.expires_at > now)
            .cloned()
            .collect();
        EnvironmentStatus {
            env_id: self.env_id().to_string(),
            owner_principal: self.inner.config.owner_principal.clone(),
            output: self.output(),
            generation: self.generation(),
            is_ready: *self.inner.is_ready.lock(),
            active_control_lease: lease,
            mapped_windows: self.inner.windows.lock().values().cloned().collect(),
            active_grants: grants,
        }
    }

    /// Viewers that have acknowledged the current geometry generation.
    pub fn geometry_ready_viewers(&self) -> Vec<String> {
        let current = self.generation();
        self.inner
            .geometry_ready
            .lock()
            .iter()
            .filter(|(_, generation)| **generation == current)
            .map(|(viewer, _)| viewer.clone())
            .collect()
    }

    /// Destroys the environment when its owner heartbeat and retention have both
    /// lapsed, so an abandoned desktop does not outlive its task.
    fn expire_owner_if_abandoned(&self) {
        let timeout = self.inner.config.owner_heartbeat_timeout_secs;
        if timeout == 0 {
            return;
        }
        if self.retention().is_some() {
            return;
        }
        let last = *self.inner.owner_heartbeat.lock();
        if now_secs().saturating_sub(last) <= timeout {
            return;
        }
        let _ = self.inner.events.send(DesktopEvent::Failed {
            env_id: self.env_id().to_string(),
            reason: format!("owner heartbeat expired after {timeout}s"),
        });
        self.destroy();
    }

    /// Clears an expired control lease and reports the change.
    fn expire_lease_if_stale(&self) {
        let mut lease = self.inner.lease.lock();
        let expired = lease
            .as_ref()
            .map(|lease| lease.expires_at <= now_secs())
            .unwrap_or(false);
        if expired {
            *lease = None;
            drop(lease);
            let _ = self.inner.events.send(DesktopEvent::ControlChanged {
                env_id: self.env_id().to_string(),
                lease: None,
            });
        }
    }

    /// Destroys the environment: terminates applications, stops the compositor,
    /// and removes the private runtime directory. Idempotent.
    pub fn destroy(&self) {
        if self.inner.destroyed.swap(true, Ordering::SeqCst) {
            return;
        }

        let apps: Vec<AppHandle> = self.inner.apps.lock().drain().map(|(_, a)| a).collect();
        for app in apps {
            terminate_app(&app);
        }

        if let Some(compositor) = self.inner.compositor.lock().take() {
            compositor.shutdown();
        }

        // Stopping the slice retires every scope inside it, including any
        // process that outlived its launcher.
        if let Some(slice) = self.inner.slice.as_ref() {
            let _ = Command::new("systemctl")
                .args(["--user", "stop", slice])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }

        // Dropping the lock file releases the shared GPU lock.
        drop(self.inner.gpu_lock.lock().take());

        let _ = std::fs::remove_dir_all(&self.inner.runtime_dir);
        *self.inner.is_ready.lock() = false;
        let _ = self.inner.events.send(DesktopEvent::Stopped {
            env_id: self.env_id().to_string(),
        });
    }

    fn compositor(&self) -> Result<CompositorHandle, DesktopError> {
        self.inner
            .compositor
            .lock()
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                DesktopError::EnvironmentFailed(
                    self.env_id().to_string(),
                    "compositor is not running".to_string(),
                )
            })
    }
}

impl Drop for InnerEnvironment {
    fn drop(&mut self) {
        if !self.destroyed.swap(true, Ordering::SeqCst) {
            if let Some(compositor) = self.compositor.lock().take() {
                compositor.shutdown();
            }
            if let Some(slice) = self.slice.as_ref() {
                let _ = Command::new("systemctl")
                    .args(["--user", "stop", slice])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            let _ = std::fs::remove_dir_all(&self.runtime_dir);
        }
    }
}

fn spawn_event_forwarder(inner: Arc<InnerEnvironment>, rx: std_mpsc::Receiver<DesktopEvent>) {
    let weak = Arc::downgrade(&inner);
    let name = format!("juto-desktop-events-{}", inner.config.env_id);
    drop(inner);
    let _ = std::thread::Builder::new().name(name).spawn(move || {
        while let Ok(event) = rx.recv() {
            // The environment may already be gone; the compositor's sender
            // drops with it, which ends this loop on the next receive.
            let Some(inner) = weak.upgrade() else {
                continue;
            };
            match &event {
                DesktopEvent::WindowMapped { window, .. }
                | DesktopEvent::WindowChanged { window, .. } => {
                    inner
                        .windows
                        .lock()
                        .insert(window.window_id, window.clone());
                }
                DesktopEvent::WindowUnmapped { window_id, .. } => {
                    inner.windows.lock().remove(window_id);
                }
                _ => {}
            }
            let _ = inner.events.send(event);
        }
    });
}

/// Fraction of a limit at which a `LimitPressure` event is emitted.
const PRESSURE_THRESHOLD: f64 = 0.9;

/// Samples the environment slice's cgroup and reports pressure.
///
/// The slice aggregates the whole environment, so this observes the effective
/// limits rather than a single process.
fn spawn_pressure_monitor(inner: std::sync::Weak<InnerEnvironment>) {
    let _ = std::thread::Builder::new()
        .name("juto-desktop-pressure".to_string())
        .spawn(move || {
            let mut cgroup: Option<PathBuf> = None;
            let mut last_throttled = 0u64;
            loop {
                std::thread::sleep(Duration::from_secs(2));
                let Some(inner) = inner.upgrade() else {
                    return;
                };
                if inner.destroyed.load(Ordering::SeqCst) {
                    return;
                }
                let Some(slice) = inner.slice.as_ref() else {
                    continue;
                };
                if cgroup.is_none() {
                    cgroup = unit_cgroup_path(slice);
                }
                let Some(cgroup) = cgroup.as_ref() else {
                    continue;
                };

                if let (Some(current), Some(max)) = (
                    read_cgroup_bytes(&cgroup.join("memory.current")),
                    read_cgroup_bytes(&cgroup.join("memory.max")),
                ) && max > 0
                {
                    let ratio = current as f64 / max as f64;
                    if ratio >= PRESSURE_THRESHOLD {
                        let _ = inner.events.send(DesktopEvent::LimitPressure {
                            env_id: inner.config.env_id.clone(),
                            metric: "memory".to_string(),
                            current_value: current,
                            threshold: max,
                        });
                    }
                }

                if let Ok(stat) = std::fs::read_to_string(cgroup.join("cpu.stat")) {
                    let throttled = stat
                        .lines()
                        .find_map(|line| line.strip_prefix("nr_throttled "))
                        .and_then(|value| value.trim().parse::<u64>().ok())
                        .unwrap_or(0);
                    if throttled > last_throttled {
                        last_throttled = throttled;
                        let _ = inner.events.send(DesktopEvent::LimitPressure {
                            env_id: inner.config.env_id.clone(),
                            metric: "cpu_throttled".to_string(),
                            current_value: throttled,
                            threshold: 0,
                        });
                    }
                }
            }
        });
}

/// Periodically clears expired control leases so a stale holder cannot keep
/// control after its heartbeat stops.
fn spawn_lease_reaper(inner: std::sync::Weak<InnerEnvironment>) {
    let _ = std::thread::Builder::new()
        .name("juto-desktop-lease-reaper".to_string())
        .spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let Some(inner) = inner.upgrade() else {
                    return;
                };
                if inner.destroyed.load(Ordering::SeqCst) {
                    return;
                }
                let env = DesktopEnvironment { inner };
                env.expire_lease_if_stale();
                env.expire_owner_if_abandoned();
            }
        });
}

fn spawn_app_waiter(
    inner: std::sync::Weak<InnerEnvironment>,
    app_id: String,
    unit: Option<String>,
    mut child: Child,
) {
    let _ = std::thread::Builder::new()
        .name(format!("juto-desktop-app-{app_id}"))
        .spawn(move || {
            let status = child.wait();
            let exit_code = status.ok().and_then(|s| s.code());
            if let Some(unit) = unit {
                let _ = Command::new("systemctl")
                    .args(["--user", "reset-failed", &unit])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            let Some(inner) = inner.upgrade() else {
                return;
            };
            inner.apps.lock().remove(&app_id);
            let _ = inner.events.send(DesktopEvent::ApplicationExited {
                env_id: inner.config.env_id.clone(),
                app_id,
                exit_code,
            });
        });
}

fn terminate_app(app: &AppHandle) {
    if let Some(unit) = app.unit.as_ref() {
        let _ = Command::new("systemctl")
            .args(["--user", "stop", unit])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let deadline = std::time::Instant::now() + TERMINATE_GRACE;
        while std::time::Instant::now() < deadline {
            let active = Command::new("systemctl")
                .args(["--user", "is-active", "--quiet", unit])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !active {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = Command::new("systemctl")
            .args(["--user", "kill", "--signal=SIGKILL", unit])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    } else {
        #[cfg(unix)]
        unsafe {
            libc::killpg(app.pid as libc::pid_t, libc::SIGTERM);
        }
        std::thread::sleep(Duration::from_millis(200));
        #[cfg(unix)]
        unsafe {
            libc::killpg(app.pid as libc::pid_t, libc::SIGKILL);
        }
    }
}

/// Verifies that a launched scope actually enforces the configured limits.
///
/// Returns `IsolationUnavailable` when the effective properties do not match,
/// so a silently uncapped application is never reported as isolated.
fn verify_scope(unit: &str, limits: &SystemLimits) -> Result<(), DesktopError> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut last = String::new();
    loop {
        let output = Command::new("systemctl")
            .args([
                "--user",
                "show",
                unit,
                "--property=CPUQuotaPerSecUSec",
                "--property=MemoryMax",
                "--property=TasksMax",
                "--no-pager",
            ])
            .output();
        if let Ok(output) = output {
            let text = String::from_utf8_lossy(&output.stdout).to_string();
            if let Some(problem) = scope_mismatch(&text, limits) {
                last = problem;
            } else {
                return Ok(());
            }
        } else if let Err(err) = output {
            last = format!("systemctl show failed: {err}");
        }
        if std::time::Instant::now() >= deadline {
            return Err(DesktopError::IsolationUnavailable(format!(
                "scope `{unit}` did not enforce the requested limits: {last}"
            )));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn scope_mismatch(properties: &str, limits: &SystemLimits) -> Option<String> {
    let mut cpu_quota = None;
    let mut memory_max = None;
    let mut tasks_max = None;
    for line in properties.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "CPUQuotaPerSecUSec" => cpu_quota = Some(value.trim().to_string()),
            "MemoryMax" => memory_max = Some(value.trim().to_string()),
            "TasksMax" => tasks_max = Some(value.trim().to_string()),
            _ => {}
        }
    }

    let quota = cpu_quota.unwrap_or_default();
    if quota.is_empty() || quota == "infinity" || quota.starts_with('0') {
        return Some(format!("CPUQuotaPerSecUSec={quota}"));
    }
    let expected_memory = parse_size(&limits.memory_max);
    let actual_memory = memory_max.as_deref().and_then(parse_size);
    if let (Some(expected), Some(actual)) = (expected_memory, actual_memory)
        && actual != expected
    {
        return Some(format!("MemoryMax={actual} (expected {expected})"));
    }
    if actual_memory.is_none() {
        return Some(format!(
            "MemoryMax={}",
            memory_max.unwrap_or_else(|| "unset".to_string())
        ));
    }
    let expected_tasks = u64::from(limits.tasks_max);
    let actual_tasks = tasks_max
        .as_deref()
        .and_then(|value| value.parse::<u64>().ok());
    if actual_tasks != Some(expected_tasks) {
        return Some(format!(
            "TasksMax={} (expected {expected_tasks})",
            tasks_max.unwrap_or_else(|| "unset".to_string())
        ));
    }
    None
}

/// Parses a systemd size string such as `8G` or `8589934592`.
fn parse_size(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() || value == "infinity" {
        return None;
    }
    let (digits, multiplier) = match value.chars().last()? {
        'K' => (&value[..value.len() - 1], 1024u64),
        'M' => (&value[..value.len() - 1], 1024 * 1024),
        'G' => (&value[..value.len() - 1], 1024 * 1024 * 1024),
        'T' => (&value[..value.len() - 1], 1024 * 1024 * 1024 * 1024),
        _ => (value, 1),
    };
    digits.trim().parse::<u64>().ok().map(|n| n * multiplier)
}

/// What a reconciliation pass cleaned up.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReconcileReport {
    pub stopped_units: Vec<String>,
    pub removed_directories: Vec<PathBuf>,
}

/// A compositor running either in its own process or in a supervisor thread.
#[derive(Clone)]
enum CompositorHandle {
    Worker(Arc<WorkerCompositor>),
    InProcess(Arc<Compositor>),
}

impl CompositorHandle {
    fn send(&self, request: WorkerRequest) -> Result<(), DesktopError> {
        match self {
            CompositorHandle::Worker(worker) => worker.send(request),
            CompositorHandle::InProcess(compositor) => compositor.send(in_process_command(request)),
        }
    }

    fn snapshot(&self, timeout: Duration) -> Result<SnapshotInfo, DesktopError> {
        match self {
            CompositorHandle::Worker(worker) => worker.snapshot(timeout),
            CompositorHandle::InProcess(compositor) => {
                let (reply, rx) = std_mpsc::sync_channel(1);
                compositor.send(CompositorCommand::Snapshot { reply })?;
                match rx.recv_timeout(timeout) {
                    Ok(Ok(info)) => Ok(info),
                    Ok(Err(message)) => Err(DesktopError::Compositor(message)),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {
                        Err(DesktopError::ObservationTimeout {
                            timeout_ms: timeout.as_millis() as u64,
                        })
                    }
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                        Err(DesktopError::Compositor("compositor stopped".to_string()))
                    }
                }
            }
        }
    }

    fn shutdown(&self) {
        match self {
            CompositorHandle::Worker(worker) => worker.shutdown(),
            CompositorHandle::InProcess(_) => {}
        }
    }
}

/// Maps a worker request onto the in-process command set.
fn in_process_command(request: WorkerRequest) -> CompositorCommand {
    match request {
        WorkerRequest::Input(event) => CompositorCommand::Input(event),
        WorkerRequest::Configure {
            width,
            height,
            scale,
            fps,
            generation,
        } => {
            let (reply, _rx) = std_mpsc::sync_channel(1);
            CompositorCommand::Configure {
                width,
                height,
                scale,
                fps,
                generation,
                reply,
            }
        }
        WorkerRequest::AcquireSlot { slot } => CompositorCommand::AcquireSlot {
            slot: slot as usize,
        },
        WorkerRequest::ReleaseSlot { slot } => CompositorCommand::ReleaseSlot {
            slot: slot as usize,
        },
        WorkerRequest::Snapshot | WorkerRequest::Shutdown => CompositorCommand::Shutdown,
    }
}

/// Central supervisor managing multiple isolated desktop environments.
pub struct DesktopSupervisor {
    environments: Mutex<HashMap<String, DesktopEnvironment>>,
}

impl Default for DesktopSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopSupervisor {
    pub fn new() -> Self {
        Self {
            environments: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_environment(
        &self,
        config: EnvironmentConfig,
    ) -> Result<DesktopEnvironment, DesktopError> {
        let env_id = config.env_id.clone();
        let env = DesktopEnvironment::new(config)?;
        self.environments.lock().insert(env_id, env.clone());
        Ok(env)
    }

    pub fn get_environment(&self, env_id: &str) -> Result<DesktopEnvironment, DesktopError> {
        self.environments
            .lock()
            .get(env_id)
            .cloned()
            .ok_or_else(|| DesktopError::EnvironmentNotFound(env_id.to_string()))
    }

    pub fn destroy_environment(&self, env_id: &str) -> Result<(), DesktopError> {
        let env = self.environments.lock().remove(env_id);
        match env {
            Some(env) => {
                env.destroy();
                Ok(())
            }
            None => Err(DesktopError::EnvironmentNotFound(env_id.to_string())),
        }
    }

    /// Kills orphan environment scopes and removes stale runtime directories
    /// left behind by a previous supervisor process.
    ///
    /// Live environments owned by this supervisor are left untouched, and so are
    /// environments whose owning supervisor process is still alive.
    pub fn reconcile(&self) -> ReconcileReport {
        let mut report = ReconcileReport::default();
        let live: Vec<String> = self
            .environments
            .lock()
            .keys()
            .map(|id| sanitize(id))
            .collect();

        let root = desktop_runtime_root();
        let Ok(entries) = std::fs::read_dir(&root) else {
            return report;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if live.contains(&name) {
                continue;
            }
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            // Only reclaim an environment whose owning supervisor is gone;
            // another live supervisor's environments must be left alone.
            let owner = std::fs::read_to_string(path.join("owner"))
                .ok()
                .and_then(|text| text.trim().parse::<i32>().ok());
            if owner.map(process_alive).unwrap_or(false) {
                continue;
            }
            let mut units = scopes_for_environment(&name);
            units.push(environment_slice_name(&name));
            for unit in units {
                let _ = Command::new("systemctl")
                    .args(["--user", "stop", &unit])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                report.stopped_units.push(unit);
            }
            if std::fs::remove_dir_all(&path).is_ok() {
                report.removed_directories.push(path);
            }
        }
        report
    }

    /// Destroys every environment. Used on supervisor shutdown.
    pub fn destroy_all(&self) {
        let envs: Vec<DesktopEnvironment> =
            self.environments.lock().drain().map(|(_, e)| e).collect();
        for env in envs {
            env.destroy();
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Whether per-environment systemd scopes are enforced.
///
/// Returns an error when `systemd-run` is unavailable unless the operator has
/// explicitly opted into unscoped development mode with
/// `JUTO_DESKTOP_ALLOW_UNSCOPED=1`.
fn scope_enforced() -> Result<bool, DesktopError> {
    let available = Command::new("systemd-run")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if available {
        return Ok(true);
    }
    if std::env::var("JUTO_DESKTOP_ALLOW_UNSCOPED").as_deref() == Ok("1") {
        tracing::warn!(
            "systemd-run unavailable; running desktop applications unscoped because \
             JUTO_DESKTOP_ALLOW_UNSCOPED=1"
        );
        return Ok(false);
    }
    Err(DesktopError::IsolationUnavailable(
        "systemd-run is unavailable; refusing to run desktop applications without \
         resource enforcement (set JUTO_DESKTOP_ALLOW_UNSCOPED=1 to override)"
            .to_string(),
    ))
}

fn desktop_runtime_root() -> PathBuf {
    let base = std::env::var_os("JUTO_DESKTOP_RUNTIME_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("juto-desktop")
}

fn environment_runtime_dir(env_id: &str) -> PathBuf {
    desktop_runtime_root().join(sanitize(env_id))
}

fn prepare_runtime_dir(dir: &Path) -> Result<(), DesktopError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).map_err(|e| DesktopError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| {
        DesktopError::Io {
            path: dir.to_path_buf(),
            source: e,
        }
    })?;
    for sub in ["home", "tmp"] {
        let path = dir.join(sub);
        std::fs::create_dir_all(&path).map_err(|e| DesktopError::Io {
            path: path.clone(),
            source: e,
        })?;
    }
    // Record the owning supervisor so reconciliation can tell a live
    // environment from one left behind by a dead process.
    let owner = dir.join("owner");
    std::fs::write(&owner, format!("{}\n", std::process::id())).map_err(|e| DesktopError::Io {
        path: owner.clone(),
        source: e,
    })?;
    Ok(())
}

/// Whether a process is still alive.
fn process_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    let result = unsafe { libc::kill(pid, 0) };
    if result == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Scope units belonging to one environment id.
fn scopes_for_environment(sanitized_env_id: &str) -> Vec<String> {
    let prefix = format!("juto-desktop-{sanitized_env_id}-");
    let Ok(output) = Command::new("systemctl")
        .args([
            "--user",
            "list-units",
            "--all",
            "--no-legend",
            "--plain",
            "juto-desktop-*.scope",
        ])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|unit| unit.starts_with(&prefix))
        .map(|unit| unit.to_string())
        .collect()
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The transient slice that aggregates an environment's cgroup limits.
///
/// systemd treats `-` in a unit name as a path separator, so a literal dash in
/// the environment id would create a chain of parent slices. Dashes are escaped
/// to `_` for the slice name only.
fn environment_slice_name(env_id: &str) -> String {
    let escaped: String = sanitize(env_id)
        .chars()
        .map(|c| if c == '-' { '_' } else { c })
        .collect();
    format!("juto-desktop-{escaped}.slice")
}

/// Creates the environment slice and applies the aggregate limits.
///
/// Applications and (later) the compositor worker run as scopes inside this
/// slice, so the limits bound the whole environment rather than each process.
fn create_environment_slice(env_id: &str, limits: &SystemLimits) -> Result<String, DesktopError> {
    let slice = environment_slice_name(env_id);
    let started = Command::new("systemctl")
        .args(["--user", "start", &slice])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !started {
        return Err(DesktopError::IsolationUnavailable(format!(
            "cannot create the environment slice `{slice}`"
        )));
    }
    let applied = Command::new("systemctl")
        .args([
            "--user",
            "set-property",
            &slice,
            &format!("CPUQuota={}", limits.cpu_quota),
            &format!("MemoryMax={}", limits.memory_max),
            &format!("MemorySwapMax={}", limits.memory_swap_max),
            &format!("TasksMax={}", limits.tasks_max),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !applied {
        let _ = Command::new("systemctl")
            .args(["--user", "stop", &slice])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        return Err(DesktopError::IsolationUnavailable(format!(
            "cannot apply resource limits to the environment slice `{slice}`"
        )));
    }
    Ok(slice)
}

/// The cgroup path backing a unit, if the unit exists.
fn unit_cgroup_path(unit: &str) -> Option<PathBuf> {
    let output = Command::new("systemctl")
        .args([
            "--user",
            "show",
            unit,
            "--property=ControlGroup",
            "--no-pager",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let path = text
        .lines()
        .find_map(|line| line.strip_prefix("ControlGroup="))?
        .trim();
    if path.is_empty() {
        return None;
    }
    Some(PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/')))
}

/// Reads a cgroup file as a byte count, treating `max` as unlimited.
fn read_cgroup_bytes(path: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text == "max" {
        return None;
    }
    text.parse::<u64>().ok()
}

fn scope_unit_name(env_id: &str, app_id: &str) -> String {
    format!(
        "juto-desktop-{}-{}.scope",
        sanitize(env_id),
        sanitize(app_id)
    )
}

fn resolve_executable(program: &str) -> Result<PathBuf, DesktopError> {
    let path = Path::new(program);
    if path.is_absolute() {
        return if path.is_file() {
            Ok(path.to_path_buf())
        } else {
            Err(DesktopError::ExecutableUnavailable {
                path: path.to_path_buf(),
                reason: "not a regular file".to_string(),
            })
        };
    }
    if program.contains('/') {
        let candidate = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path);
        return if candidate.is_file() {
            Ok(candidate)
        } else {
            Err(DesktopError::ExecutableUnavailable {
                path: candidate,
                reason: "not a regular file".to_string(),
            })
        };
    }
    let search = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&search) {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(DesktopError::ExecutableUnavailable {
        path: PathBuf::from(program),
        reason: "not found in PATH".to_string(),
    })
}
