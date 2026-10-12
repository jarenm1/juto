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
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

use crate::compositor::{Compositor, CompositorCommand, CompositorConfig, SnapshotInfo};
use crate::error::DesktopError;
use crate::protocol::*;
use crate::shm::ShmFramePool;

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

/// Rendering profile: software LLVMpipe or hardware render node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RenderProfile {
    Software,
    Hardware,
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
    pool: Arc<ShmFramePool>,
    compositor: Mutex<Option<Compositor>>,
    apps: Mutex<HashMap<String, AppHandle>>,
    events: broadcast::Sender<DesktopEvent>,
    last_automation_barrier: Mutex<Option<AutomationBarrier>>,
    destroyed: AtomicBool,
    use_scope: bool,
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

        let use_scope = scope_enforced()?;

        let runtime_dir = environment_runtime_dir(&config.env_id);
        prepare_runtime_dir(&runtime_dir)?;
        let socket_path = runtime_dir.join("wayland-0");

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

        let (events, _) = broadcast::channel(256);
        let (comp_tx, comp_rx) = std_mpsc::channel::<DesktopEvent>();

        let compositor = match Compositor::spawn(
            CompositorConfig {
                env_id: config.env_id.clone(),
                width: config.output.width,
                height: config.output.height,
                scale: config.output.scale,
                fps: config.output.fps,
                socket_path: socket_path.clone(),
            },
            pool.clone(),
            comp_tx,
        ) {
            Ok(compositor) => compositor,
            Err(err) => {
                let _ = std::fs::remove_dir_all(&runtime_dir);
                return Err(err);
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
            destroyed: AtomicBool::new(false),
            use_scope,
        });

        spawn_event_forwarder(inner.clone(), comp_rx);

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

    /// The environment's primary composed-frame pool.
    pub fn frame_pool(&self) -> Arc<ShmFramePool> {
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
                if let Some(current) = lease.as_ref()
                    && ((!is_human && current.is_human)
                        || (!is_human && !current.is_human && current.holder != caller))
                {
                    return Err(DesktopError::ControlBusy {
                        holder: current.holder.clone(),
                        epoch: current.epoch,
                    });
                }
                let new_epoch = lease.as_ref().map(|l| l.epoch + 1).unwrap_or(1);
                let new_lease = ControlLease {
                    holder: caller.to_string(),
                    epoch: new_epoch,
                    expires_at: now_secs() + 6,
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
        let current_gen = self.generation();
        if current_gen != generation {
            return Err(DesktopError::StaleGeneration {
                expected: current_gen,
                received: generation,
            });
        }
        drop(lease);

        if batch
            .events
            .iter()
            .any(|event| matches!(event, InputEvent::TextCommit { .. }))
        {
            return Err(DesktopError::TextUnavailable(
                "text-input-v3 is not advertised by this compositor".to_string(),
            ));
        }

        let compositor = self.compositor()?;
        for event in batch.events {
            compositor.send(CompositorCommand::Input(event))?;
        }
        Ok(batch.seq)
    }

    /// Forces a fresh composition and returns the resulting frame metadata.
    pub fn observe(&self, timeout: Duration) -> Result<SnapshotInfo, DesktopError> {
        let compositor = self.compositor()?;
        let (reply, rx) = std_mpsc::sync_channel(1);
        compositor.send(CompositorCommand::Snapshot { reply })?;
        match rx.recv_timeout(timeout) {
            Ok(Ok(info)) => Ok(info),
            Ok(Err(msg)) => Err(DesktopError::Compositor(msg)),
            Err(std_mpsc::RecvTimeoutError::Timeout) => Err(DesktopError::ObservationTimeout {
                timeout_ms: timeout.as_millis() as u64,
            }),
            Err(std_mpsc::RecvTimeoutError::Disconnected) => Err(DesktopError::Compositor(
                "compositor stopped before producing a frame".to_string(),
            )),
        }
    }

    /// Observes a fresh frame and encodes it as a lossless PNG.
    ///
    /// The returned image is the composed desktop, not a streamed frame: it is
    /// suitable as durable evidence for a caller.
    pub fn snapshot_png(&self, timeout: Duration) -> Result<(SnapshotInfo, Vec<u8>), DesktopError> {
        let info = self.observe(timeout)?;
        let pool = self.frame_pool();
        pool.acquire_slot(info.slot);
        let mut pixels = Vec::new();
        let header = pool.read_pixels(info.slot, &mut pixels);
        pool.release_slot(info.slot);
        let header = header?;
        let png = crate::png::encode_bgra(header.width, header.height, &pixels)?;
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
        let (reply, rx) = std_mpsc::sync_channel(1);
        compositor.send(CompositorCommand::Configure {
            width: new_mode.width,
            height: new_mode.height,
            scale: new_mode.scale,
            fps: new_mode.fps,
            generation: new_gen,
            reply,
        })?;
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => return Err(DesktopError::Compositor(msg)),
            Err(std_mpsc::RecvTimeoutError::Timeout) => {
                return Err(DesktopError::Compositor(
                    "output reconfiguration timed out".to_string(),
                ));
            }
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                return Err(DesktopError::Compositor(
                    "compositor stopped during reconfiguration".to_string(),
                ));
            }
        }
        *self.inner.output.lock() = new_mode;
        self.inner.generation.store(new_gen, Ordering::SeqCst);
        Ok(new_gen)
    }

    pub fn get_status(&self) -> EnvironmentStatus {
        EnvironmentStatus {
            env_id: self.env_id().to_string(),
            owner_principal: self.inner.config.owner_principal.clone(),
            output: self.output(),
            generation: self.generation(),
            is_ready: *self.inner.is_ready.lock(),
            active_control_lease: self.inner.lease.lock().clone(),
            mapped_windows: self.inner.windows.lock().values().cloned().collect(),
            active_grants: self.inner.grants.lock().values().cloned().collect(),
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

        let _ = std::fs::remove_dir_all(&self.inner.runtime_dir);
        *self.inner.is_ready.lock() = false;
        let _ = self.inner.events.send(DesktopEvent::Stopped {
            env_id: self.env_id().to_string(),
        });
    }

    fn compositor(&self) -> Result<crate::compositor::CompositorSender, DesktopError> {
        self.inner
            .compositor
            .lock()
            .as_ref()
            .map(|c| c.sender())
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

fn environment_runtime_dir(env_id: &str) -> PathBuf {
    let base = std::env::var_os("JUTO_DESKTOP_RUNTIME_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("juto-desktop").join(sanitize(env_id))
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
    Ok(())
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
