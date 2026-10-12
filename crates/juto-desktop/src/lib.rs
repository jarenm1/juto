//! `juto-desktop`: Agent-first headless desktop compositor, environment supervisor,
//! and split presentation transport daemon.
//!
//! Architectural specification: `docs/agent-compositor.md`.

pub mod compositor;
pub mod error;
pub mod png;
pub mod protocol;
pub mod remote;
pub mod scm;
pub mod shm;
pub mod signaling;
pub mod supervisor;
pub mod viewer;

pub use compositor::{
    Compositor, CompositorCommand, CompositorConfig, CompositorSender, SnapshotInfo,
};
pub use error::DesktopError;
pub use protocol::{
    AutomationBarrier, ControlLease, DesktopCommand, DesktopEvent, DesktopResponse,
    GeometryGeneration, Grant, GrantRole, InputBatch, InputEvent, WindowInfo,
};
pub use remote::{
    IceServer, NativeReceiver, PresentationWorker, ReceivedFrame, RemoteStreamConfig,
    SignalingMessage, VideoCodec, WebRtcCandidate, WebRtcSessionDescription,
};
pub use shm::{ShmFrame, ShmFramePool, ShmReader, ShmWriter};
pub use signaling::{AttachGrant, SignalingServer, ViewerReport};
pub use supervisor::{
    AppHandle, DesktopEnvironment, DesktopSupervisor, EnvironmentConfig, EnvironmentStatus,
    OutputConfig, OutputProfile, RenderProfile, SystemLimits,
};
pub use viewer::{AttachInfo, ViewerClient};
