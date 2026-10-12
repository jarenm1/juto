//! `juto-desktop`: Agent-first headless desktop compositor, environment supervisor,
//! and split presentation transport daemon.
//!
//! Architectural specification: `docs/agent-compositor.md`.

pub mod compositor;
pub mod error;
pub mod png;
pub mod protocol;
pub mod remote;
pub mod shm;
pub mod supervisor;

pub use compositor::{
    Compositor, CompositorCommand, CompositorConfig, CompositorSender, SnapshotInfo,
};
pub use error::DesktopError;
pub use protocol::{
    AutomationBarrier, ControlLease, DesktopCommand, DesktopEvent, DesktopResponse,
    GeometryGeneration, Grant, GrantRole, InputBatch, InputEvent, WindowInfo,
};
pub use remote::{
    RemoteStreamConfig, RemoteStreamSession, WebRtcCandidate, WebRtcSessionDescription,
};
pub use shm::{ShmFrame, ShmFramePool, ShmReader, ShmWriter};
pub use supervisor::{
    AppHandle, DesktopEnvironment, DesktopSupervisor, EnvironmentConfig, EnvironmentStatus,
    OutputConfig, OutputProfile, RenderProfile, SystemLimits,
};
