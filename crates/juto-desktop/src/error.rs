use std::path::PathBuf;
use thiserror::Error;

/// Errors emitted by the desktop compositor, environment supervisor, and presentation transports.
#[derive(Debug, Error)]
pub enum DesktopError {
    #[error("environment `{0}` not found")]
    EnvironmentNotFound(String),

    #[error("unauthorized: caller `{caller}` lacks role `{role}` on environment `{environment}`")]
    Unauthorized {
        caller: String,
        role: String,
        environment: String,
    },

    #[error("invalid output mode: {width}x{height} exceeds admitted limits or pixel rate")]
    InvalidOutputMode {
        width: u32,
        height: u32,
        reason: String,
    },

    #[error("limit exceeded: {0}")]
    LimitExceeded(String),

    #[error("executable `{path}` unavailable or inaccessible: {reason}")]
    ExecutableUnavailable { path: PathBuf, reason: String },

    #[error("control lease busy: held by `{holder}` (epoch {epoch})")]
    ControlBusy { holder: String, epoch: u64 },

    #[error("automation busy: pending mutations during epoch {epoch}")]
    AutomationBusy { epoch: u64 },

    #[error("stale geometry generation: expected {expected}, received {received}")]
    StaleGeneration { expected: u64, received: u64 },

    #[error("stale control lease: expected epoch {expected}, received {received}")]
    StaleLease { expected: u64, received: u64 },

    #[error("observation timeout: no new frame produced within {timeout_ms}ms")]
    ObservationTimeout { timeout_ms: u64 },

    #[error("transport unavailable: {0}")]
    TransportUnavailable(String),

    #[error("environment `{0}` failed: {1}")]
    EnvironmentFailed(String, String),

    #[error("I/O error on `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("IPC protocol error: {0}")]
    Protocol(String),

    #[error("SHM buffer error: {0}")]
    Shm(String),

    #[error("compositor error: {0}")]
    Compositor(String),

    #[error("isolation unavailable: {0}")]
    IsolationUnavailable(String),

    #[error("text input unavailable: {0}")]
    TextUnavailable(String),
}
