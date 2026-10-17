//! Keeps agent-created desktops alive after the agent run that created them.
//!
//! The supervisor destroys an environment once its owner heartbeat lapses. The
//! desktop tool heartbeats only while its run is acting, so without renewal a
//! desktop would close as soon as its builder finished, before anyone could
//! open it from the main chat. The keepalive renews the owner heartbeat of
//! every desktop it adopted until that desktop is destroyed, so desktops live
//! for the runtime process unless destroyed explicitly.

use std::sync::Arc;
use std::time::Duration;

use juto_desktop::DesktopEnvironment;
use parking_lot::Mutex;

/// Renewal period for adopted desktops. Must stay well under the owner
/// heartbeat timeout (60 s for agent desktops).
const RENEW_INTERVAL: Duration = Duration::from_secs(15);

/// Shared registry of adopted desktops and their renewal task.
#[derive(Clone)]
pub struct DesktopKeepalive {
    interval: Duration,
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    envs: Vec<DesktopEnvironment>,
    /// Whether a renewal task is running. Cleared under the lock when the
    /// task exits, so a later `adopt` starts a new one.
    renewing: bool,
}

impl Default for DesktopKeepalive {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopKeepalive {
    pub fn new() -> Self {
        Self::with_interval(RENEW_INTERVAL)
    }

    fn with_interval(interval: Duration) -> Self {
        Self {
            interval,
            state: Arc::new(Mutex::new(State::default())),
        }
    }

    /// Renews `env`'s owner lifetime until it is destroyed. Requires a Tokio
    /// runtime context.
    pub fn adopt(&self, env: DesktopEnvironment) {
        let mut state = self.state.lock();
        state.envs.push(env);
        if !state.renewing {
            state.renewing = true;
            tokio::spawn(renew(self.state.clone(), self.interval));
        }
    }
}

async fn renew(state: Arc<Mutex<State>>, interval: Duration) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        let mut guard = state.lock();
        guard.envs.retain(|env| !env.is_destroyed());
        if guard.envs.is_empty() {
            guard.renewing = false;
            return;
        }
        for env in &guard.envs {
            // Only unknown lease targets fail; renewing the owner cannot.
            let _ = env.heartbeat("owner", None);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use juto_desktop::DesktopSupervisor;
    use juto_desktop::supervisor::{
        CompositorMode, EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits,
    };

    use super::*;

    /// Owner lifetime short enough to test: the supervisor reaps an
    /// environment once this many seconds pass without a heartbeat.
    const OWNER_TIMEOUT_SECS: u64 = 1;

    fn create(supervisor: &DesktopSupervisor, workspace: &Path) -> DesktopEnvironment {
        supervisor
            .create_environment(
                EnvironmentConfig {
                    env_id: format!("desk-test-{}", uuid::Uuid::now_v7()),
                    owner_principal: "agent:test".to_string(),
                    output: OutputConfig {
                        width: 640,
                        height: 480,
                        scale: 1,
                        fps: 30,
                    },
                    render_profile: RenderProfile::Software,
                    limits: SystemLimits::default(),
                    workspace_dir: workspace.to_path_buf(),
                    lease_ttl_secs: 6,
                    compositor_mode: CompositorMode::InProcess,
                    owner_heartbeat_timeout_secs: OWNER_TIMEOUT_SECS,
                },
                &uuid::Uuid::now_v7().to_string(),
            )
            .expect("create environment")
    }

    async fn wait_destroyed(env: &DesktopEnvironment) -> bool {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !env.is_destroyed() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .is_ok()
    }

    #[tokio::test]
    async fn adopted_desktop_outlives_its_owner_timeout() {
        let workspace = tempfile::tempdir().unwrap();
        let supervisor = DesktopSupervisor::new();
        let env = create(&supervisor, workspace.path());
        let keepalive = DesktopKeepalive::with_interval(Duration::from_millis(100));
        keepalive.adopt(env.clone());

        tokio::time::sleep(Duration::from_secs(3 * OWNER_TIMEOUT_SECS + 1)).await;
        assert!(!env.is_destroyed(), "adopted desktop was reaped");
        env.destroy();
    }

    #[tokio::test]
    async fn unadopted_desktop_is_reaped_after_its_owner_timeout() {
        let workspace = tempfile::tempdir().unwrap();
        let supervisor = DesktopSupervisor::new();
        let env = create(&supervisor, workspace.path());

        assert!(
            wait_destroyed(&env).await,
            "unadopted desktop was never reaped"
        );
    }
}
