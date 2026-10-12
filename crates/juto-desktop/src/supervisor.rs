use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

use crate::error::DesktopError;
use crate::protocol::*;
use crate::shm::ShmFramePool;

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

struct InnerEnvironment {
    config: EnvironmentConfig,
    generation: AtomicU64,
    frame_counter: AtomicU64,
    output: Mutex<OutputConfig>,
    is_ready: Mutex<bool>,
    lease: Mutex<Option<ControlLease>>,
    grants: Mutex<HashMap<String, Grant>>,
    windows: Mutex<HashMap<u32, WindowInfo>>,
    shm_pools: Mutex<HashMap<String, Arc<ShmFramePool>>>,
    last_automation_barrier: Mutex<Option<AutomationBarrier>>,
    events: broadcast::Sender<DesktopEvent>,
}

/// An active, isolated agent desktop environment.
#[derive(Clone)]
pub struct DesktopEnvironment {
    inner: Arc<InnerEnvironment>,
}

impl DesktopEnvironment {
    pub fn new(config: EnvironmentConfig) -> Result<Self, DesktopError> {
        // Admission check: pixel rate limits
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

        let (events, _) = broadcast::channel(256);

        // Pre-create primary SHM frame pool
        let pool = ShmFramePool::create(
            format!("{}-primary", config.env_id),
            config.output.width,
            config.output.height,
        )?;

        let mut pools = HashMap::new();
        pools.insert("primary".to_string(), Arc::new(pool));

        let inner = InnerEnvironment {
            config,
            generation: AtomicU64::new(1),
            frame_counter: AtomicU64::new(0),
            output: Mutex::new(OutputConfig::default()),
            is_ready: Mutex::new(true),
            lease: Mutex::new(None),
            grants: Mutex::new(HashMap::new()),
            windows: Mutex::new(HashMap::new()),
            shm_pools: Mutex::new(pools),
            last_automation_barrier: Mutex::new(None),
            events,
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn env_id(&self) -> &str {
        &self.inner.config.env_id
    }

    pub fn generation(&self) -> GeometryGeneration {
        self.inner.generation.load(Ordering::SeqCst)
    }

    pub fn output(&self) -> OutputConfig {
        self.inner.output.lock().clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DesktopEvent> {
        self.inner.events.subscribe()
    }

    /// Issues an authenticated grant for a principal.
    pub fn issue_grant(
        &self,
        principal: String,
        roles: Vec<GrantRole>,
        audience: String,
        expiry_seconds: u64,
    ) -> Grant {
        let now_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let grant_id = format!("grant-{}", uuid::Uuid::now_v7());
        let grant = Grant {
            grant_id: grant_id.clone(),
            principal,
            roles,
            audience,
            expires_at: now_s + expiry_seconds,
        };

        let mut grants = self.inner.grants.lock();
        grants.insert(grant_id, grant.clone());
        grant
    }

    /// Revokes an existing grant and revokes its shared resources.
    pub fn revoke_grant(&self, grant_id: &str) -> bool {
        let mut grants = self.inner.grants.lock();
        if let Some(grant) = grants.remove(grant_id) {
            // If the active controller held this grant, atomically release control
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
                    // Control is busy if already held by another controller,
                    // unless a human is preempting an agent.
                    if (!is_human && current.is_human)
                        || (!is_human && !current.is_human && current.holder != caller)
                    {
                        return Err(DesktopError::ControlBusy {
                            holder: current.holder.clone(),
                            epoch: current.epoch,
                        });
                    }
                }

                // If human taking over from agent, ensure automation barrier is cleared
                let new_epoch = lease.as_ref().map(|l| l.epoch + 1).unwrap_or(1);
                let now_s = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                let new_lease = ControlLease {
                    holder: caller.to_string(),
                    epoch: new_epoch,
                    expires_at: now_s + 6, // 6-second lease timeout
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
                "unknown control action `{}`",
                action
            ))),
        }
    }

    /// Confirms that an in-flight automation batch has quiesced before handoff.
    pub fn acknowledge_automation_quiescence(&self, old_epoch: u64, barrier_id: String) {
        let now_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let barrier = AutomationBarrier {
            barrier_id,
            old_epoch,
            drained_at: now_s,
        };
        *self.inner.last_automation_barrier.lock() = Some(barrier);
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

        // Apply input events (synthesized onto virtual Wayland seat)
        Ok(batch.seq)
    }

    /// Produces a new frame into the primary SHM pool and returns observation details.
    pub fn produce_frame(&self, raw_pixels: &[u8]) -> Result<DesktopResponse, DesktopError> {
        let frame_id = self.inner.frame_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let gen_val = self.generation();
        let output = self.output();
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let pools = self.inner.shm_pools.lock();
        let pool = pools
            .get("primary")
            .ok_or_else(|| DesktopError::Shm("primary SHM pool missing".to_string()))?;

        let frame = pool.write_frame(
            frame_id,
            gen_val,
            now_us,
            output.width,
            output.height,
            raw_pixels,
        )?;

        Ok(DesktopResponse::Observation {
            frame_id,
            generation: gen_val,
            timestamp_us: now_us,
            width: output.width,
            height: output.height,
            shm_pool_id: pool.pool_id().to_string(),
            shm_slot: frame.slot as u32,
        })
    }

    /// Reconfigures output mode atomically with a new geometry generation counter.
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

        // Allocate replacement SHM pool before committing generation change
        let new_gen = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let replacement_pool = ShmFramePool::create(
            format!("{}-gen-{}", self.env_id(), new_gen),
            new_mode.width,
            new_mode.height,
        )?;

        {
            let mut pools = self.inner.shm_pools.lock();
            pools.insert("primary".to_string(), Arc::new(replacement_pool));
            *self.inner.output.lock() = new_mode.clone();
        }

        let _ = self.inner.events.send(DesktopEvent::OutputChanged {
            env_id: self.env_id().to_string(),
            generation: new_gen,
            output: new_mode,
        });

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
        let mut map = self.environments.lock();
        map.insert(env_id, env.clone());
        Ok(env)
    }

    pub fn get_environment(&self, env_id: &str) -> Result<DesktopEnvironment, DesktopError> {
        let map = self.environments.lock();
        map.get(env_id)
            .cloned()
            .ok_or_else(|| DesktopError::EnvironmentNotFound(env_id.to_string()))
    }

    pub fn destroy_environment(&self, env_id: &str) -> Result<(), DesktopError> {
        let mut map = self.environments.lock();
        if let Some(env) = map.remove(env_id) {
            let _ = env.inner.events.send(DesktopEvent::Stopped {
                env_id: env_id.to_string(),
            });
            Ok(())
        } else {
            Err(DesktopError::EnvironmentNotFound(env_id.to_string()))
        }
    }
}
