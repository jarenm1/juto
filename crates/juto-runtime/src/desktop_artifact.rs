//! Desktop artifact resolution.
//!
//! The chat artifact seam from `docs/agent-compositor.md` section 7: given an
//! authenticated principal and a persisted descriptor, validate the host,
//! environment, and grant, and return a viewer route — or an explicit
//! `Unsupported`, `Unauthorized`, `Expired`, `Stopped`, or `Unavailable`.
//!
//! Artifact creation is not environment creation, and a resolved route is not
//! proof that a stream is live: only received frames establish that.

use std::sync::{Arc, Weak};
use std::time::Duration;

use juto_desktop::DesktopSupervisor;
use juto_desktop::protocol::GrantRole;
use juto_desktop::supervisor::MAX_GRANT_SECS;
use serde::{Deserialize, Serialize};

use crate::SignalingHandle;
use crate::tools::DesktopSessionArtifact;

/// The authenticated principal of the interactive harness session: the local
/// human who opens desktop artifacts. Agent-created desktops grant it access.
pub const SESSION_VIEWER_PRINCIPAL: &str = "human:viewer";

/// How often the session grant keepalive runs.
const GRANT_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(60);
/// Session grants closer than this to expiry are renewed.
const GRANT_RENEW_MARGIN_SECS: u64 = 600;

/// Renews this process's session viewer grants while their environments live.
///
/// A desktop can wait hours for a human to verify it; its chat artifact must
/// still open then. Each renewal is host-bounded and keeps the grant id; an
/// expired or revoked grant is never revived, and a destroyed environment's
/// grants lapse. The thread ends with the supervisor.
pub(crate) fn spawn_session_grant_keepalive(supervisor: Weak<DesktopSupervisor>) {
    let _ = std::thread::Builder::new()
        .name("juto-desktop-grant-keepalive".to_string())
        .spawn(move || {
            loop {
                std::thread::sleep(GRANT_KEEPALIVE_INTERVAL);
                let Some(supervisor) = supervisor.upgrade() else {
                    return;
                };
                renew_session_grants(&supervisor);
            }
        });
}

fn renew_session_grants(supervisor: &DesktopSupervisor) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    for env in supervisor.environments() {
        if env.is_destroyed() {
            continue;
        }
        for grant in env.get_status().active_grants {
            if grant.principal == SESSION_VIEWER_PRINCIPAL
                && grant.expires_at <= now + GRANT_RENEW_MARGIN_SECS
            {
                let _ = env.renew_grant(&grant.grant_id, MAX_GRANT_SECS);
            }
        }
    }
}

/// How a human should view an agent's desktop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewerRoute {
    /// Same-host embedded viewer for an environment owned by this process.
    Local {
        env_id: String,
        roles: Vec<GrantRole>,
        expires_at: u64,
    },
    /// Remote browser viewer reached through the signaling gateway.
    Remote {
        url: String,
        token: String,
        expires_at: u64,
        roles: Vec<GrantRole>,
    },
}

/// Why an artifact could not be opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The descriptor names a host this process does not serve.
    Unsupported(String),
    /// The principal does not hold the referenced grant.
    Unauthorized(String),
    /// The grant has expired.
    Expired(String),
    /// The environment exists but is no longer ready.
    Stopped(String),
    /// The environment is unknown here.
    Unavailable(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::Unsupported(detail) => write!(f, "unsupported: {detail}"),
            ResolveError::Unauthorized(detail) => write!(f, "unauthorized: {detail}"),
            ResolveError::Expired(detail) => write!(f, "expired: {detail}"),
            ResolveError::Stopped(detail) => write!(f, "stopped: {detail}"),
            ResolveError::Unavailable(detail) => write!(f, "unavailable: {detail}"),
        }
    }
}

/// Resolves desktop artifacts against this process's supervisor.
pub struct DesktopArtifactResolver {
    supervisor: Arc<DesktopSupervisor>,
    signaling: Arc<SignalingHandle>,
    host_id: String,
}

impl DesktopArtifactResolver {
    pub fn new(
        supervisor: Arc<DesktopSupervisor>,
        signaling: Arc<SignalingHandle>,
        host_id: String,
    ) -> Self {
        Self {
            supervisor,
            signaling,
            host_id,
        }
    }

    /// The host identity this resolver serves.
    pub fn host_id(&self) -> &str {
        &self.host_id
    }

    /// Validates a descriptor and returns the same-host viewer route.
    pub fn resolve(
        &self,
        principal: &str,
        artifact: &DesktopSessionArtifact,
    ) -> Result<ViewerRoute, ResolveError> {
        let (env_id, roles, expires_at) = self.validate(principal, artifact)?;
        Ok(ViewerRoute::Local {
            env_id,
            roles,
            expires_at,
        })
    }

    /// Validates a descriptor and issues a one-time remote attach token.
    pub fn resolve_remote(
        &self,
        principal: &str,
        artifact: &DesktopSessionArtifact,
    ) -> Result<ViewerRoute, ResolveError> {
        let (env_id, roles, _) = self.validate(principal, artifact)?;
        let server = self
            .signaling
            .get()
            .map_err(|err| ResolveError::Unavailable(err.to_string()))?;
        let grant = server.issue_token(&env_id, principal, roles.clone());
        Ok(ViewerRoute::Remote {
            url: server.viewer_url(&grant.token),
            token: grant.token,
            expires_at: grant.expires_at,
            roles,
        })
    }

    fn validate(
        &self,
        principal: &str,
        artifact: &DesktopSessionArtifact,
    ) -> Result<(String, Vec<GrantRole>, u64), ResolveError> {
        if artifact.host_id != self.host_id {
            return Err(ResolveError::Unsupported(format!(
                "host `{}` is not this host (`{}`)",
                artifact.host_id, self.host_id
            )));
        }
        let env = self
            .supervisor
            .get_environment(&artifact.environment_id)
            .map_err(|_| {
                ResolveError::Unavailable(format!(
                    "environment `{}` is not known here",
                    artifact.environment_id
                ))
            })?;

        let grant = env.grant(&artifact.grant_id).ok_or_else(|| {
            ResolveError::Unauthorized(format!("grant `{}` is not issued", artifact.grant_id))
        })?;
        if grant.principal != principal {
            return Err(ResolveError::Unauthorized(format!(
                "grant `{}` belongs to `{}`, not `{principal}`",
                artifact.grant_id, grant.principal
            )));
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if grant.expires_at <= now {
            return Err(ResolveError::Expired(format!(
                "grant `{}` expired at {}",
                artifact.grant_id, grant.expires_at
            )));
        }
        if !env.get_status().is_ready {
            return Err(ResolveError::Stopped(format!(
                "environment `{}` is stopped",
                artifact.environment_id
            )));
        }
        Ok((env.env_id().to_string(), grant.roles, grant.expires_at))
    }
}
