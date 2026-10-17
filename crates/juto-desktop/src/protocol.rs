use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Monotonic geometry generation counter. Incremented on every desktop resize/scale change.
pub type GeometryGeneration = u64;

/// Roles granted to a caller principal on an environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GrantRole {
    Owner,
    View,
    Control,
}

/// An authenticated capability grant issued by the environment owner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Grant {
    pub grant_id: String,
    pub principal: String,
    pub roles: Vec<GrantRole>,
    pub audience: String,
    pub expires_at: u64,
}

/// Exclusive interactive control lease state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlLease {
    pub holder: String,
    pub epoch: u64,
    pub expires_at: u64,
    pub is_human: bool,
}

/// A barrier acknowledging quiescence of alternate automation (application RPCs)
/// before human interactive takeover.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutomationBarrier {
    pub barrier_id: String,
    pub old_epoch: u64,
    pub drained_at: u64,
}

/// Control protocol version. A request carrying any other version is rejected
/// before it is dispatched (spec section 4).
pub const PROTOCOL_VERSION: u32 = 2;

/// Versioned control request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopRequest {
    pub version: u32,
    pub command: DesktopCommand,
}

impl DesktopRequest {
    /// Wraps a command in the current protocol version.
    pub fn new(command: DesktopCommand) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            command,
        }
    }
}

/// Top-level command dispatched to the desktop supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "command",
    content = "payload",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum DesktopCommand {
    Create {
        spec: crate::supervisor::EnvironmentConfig,
        request_id: String,
    },
    Launch {
        env_id: String,
        app_id: String,
        argv: Vec<String>,
        cwd: PathBuf,
        #[serde(default)]
        env: std::collections::HashMap<String, String>,
        grants: Vec<String>,
        request_id: String,
    },
    Terminate {
        env_id: String,
        app_id: String,
    },
    Describe {
        env_id: String,
        after_event_seq: Option<u64>,
    },
    ConfigureOutput {
        env_id: String,
        expected_generation: GeometryGeneration,
        mode: crate::supervisor::OutputConfig,
    },
    Observe {
        env_id: String,
        generation: GeometryGeneration,
        after_frame: Option<u64>,
        timeout_ms: u64,
    },
    Attach {
        env_id: String,
        transport: String,
        capabilities: Vec<String>,
    },
    Grant {
        env_id: String,
        principal: String,
        roles: Vec<GrantRole>,
        audience: String,
        expiry_seconds: u64,
    },
    Revoke {
        env_id: String,
        grant_id: String,
    },
    Heartbeat {
        env_id: String,
        target: String,
        epoch: Option<u64>,
    },
    Retain {
        env_id: String,
        human_owner: String,
        deadline_seconds: u64,
    },
    Control {
        env_id: String,
        action: String, // "acquire" or "release"
        is_human: bool,
    },
    Input {
        env_id: String,
        epoch: u64,
        generation: GeometryGeneration,
        events: InputBatch,
    },
    GeometryReady {
        env_id: String,
        viewer_id: String,
        generation: GeometryGeneration,
        track_or_pool_id: String,
    },
    AutomationQuiesced {
        env_id: String,
        old_epoch: u64,
        barrier_id: String,
    },
    Destroy {
        env_id: String,
    },
    /// Switch this connection to a stream of length-prefixed `DesktopEvent` frames.
    Subscribe {
        env_id: String,
    },
}

/// Responses to synchronous desktop commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "status",
    content = "data",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum DesktopResponse {
    Created {
        env_id: String,
        effective_output: crate::supervisor::OutputConfig,
        effective_limits: crate::supervisor::SystemLimits,
    },
    Launched {
        app_id: String,
        pid: u32,
    },
    Terminated {
        app_id: String,
    },
    Description(Box<crate::supervisor::EnvironmentStatus>),
    OutputConfigured {
        generation: GeometryGeneration,
        effective_output: crate::supervisor::OutputConfig,
    },
    Observation {
        frame_id: u64,
        generation: GeometryGeneration,
        timestamp_us: u64,
        width: u32,
        height: u32,
        shm_pool_id: String,
        shm_slot: u32,
        /// Virtual-clock ticks since the compositor started.
        ticks: u64,
        frames_composed: u64,
        frames_dropped: u64,
        compose_micros: u64,
    },
    Attached {
        viewer_id: String,
        negotiated_transport: String,
        /// Shared-memory pool metadata; the descriptor itself travels as
        /// `SCM_RIGHTS` ancillary data on the control socket.
        pool_id: String,
        width: u32,
        height: u32,
        stride: u32,
        format: String,
        slot_count: u32,
        slot_size: u32,
        pool_bytes: u64,
    },
    GrantIssued(Grant),
    Revoked,
    HeartbeatExtended {
        expires_at: u64,
    },
    Retained {
        human_owner: String,
        expires_at: u64,
    },
    ControlChanged {
        lease: Option<ControlLease>,
    },
    InputAccepted {
        applied_watermark: u64,
    },
    GeometryAcknowledged {
        viewer_id: String,
        generation: GeometryGeneration,
    },
    Destroyed,
    Subscribed {
        env_id: String,
    },
    Error {
        code: String,
        message: String,
    },
}

/// Batch of user or agent input events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputBatch {
    pub seq: u64,
    pub client_timestamp_us: u64,
    pub events: Vec<InputEvent>,
}

/// Discrete input event targeting the virtual seat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum InputEvent {
    PointerMoveAbs {
        x: u32,
        y: u32,
    },
    PointerDelta {
        dx: i32,
        dy: i32,
    },
    PointerButton {
        button: u32,
        pressed: bool,
    },
    PointerScroll {
        axis: u32,
        value: i32,
    },
    Key {
        keycode: u32,
        pressed: bool,
    },
    TextCommit {
        text: String,
        focus_epoch: u64,
    },
    /// Pre-edit (composing) text for the focused text-input instance.
    TextPreedit {
        text: String,
        cursor_begin: i32,
        cursor_end: i32,
        focus_epoch: u64,
    },
}

/// Metadata about a window mapped inside the compositor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowInfo {
    pub window_id: u32,
    pub title: String,
    pub app_id: String,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub is_focused: bool,
}

/// Real-time asynchronous lifecycle event broadcast by the desktop supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "event",
    content = "payload",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum DesktopEvent {
    Ready {
        env_id: String,
    },
    /// A subscriber lagged past the bounded replay ring and must call `Describe`.
    ResyncRequired {
        env_id: String,
    },
    /// Xwayland is ready; applications may use `DISPLAY=:<display_number>`.
    XwaylandReady {
        env_id: String,
        display_number: u32,
    },
    ApplicationExited {
        env_id: String,
        app_id: String,
        exit_code: Option<i32>,
    },
    WindowMapped {
        env_id: String,
        window: WindowInfo,
    },
    WindowChanged {
        env_id: String,
        window: WindowInfo,
    },
    WindowUnmapped {
        env_id: String,
        window_id: u32,
    },
    OutputChanged {
        env_id: String,
        generation: GeometryGeneration,
        output: crate::supervisor::OutputConfig,
    },
    ControlChanged {
        env_id: String,
        lease: Option<ControlLease>,
    },
    /// Pointer shape the application requested: a CSS cursor name, or `none` when hidden.
    CursorShapeChanged {
        env_id: String,
        shape: String,
    },
    /// Client cursor bitmap as straight-alpha BGRA. `width`/`height` are buffer
    /// pixels; `display_*` and the hotspot are environment pixels. `width: 0`
    /// means no bitmap, so the viewer shows the named arrow.
    CursorImageChanged {
        env_id: String,
        width: u32,
        height: u32,
        display_width: u32,
        display_height: u32,
        hotspot_x: i32,
        hotspot_y: i32,
        #[serde(with = "base64_pixels")]
        pixels: Vec<u8>,
    },
    ViewerStateChanged {
        env_id: String,
        viewer_id: String,
        state: String,
    },
    LimitPressure {
        env_id: String,
        metric: String,
        current_value: u64,
        threshold: u64,
    },
    Failed {
        env_id: String,
        reason: String,
    },
    Stopped {
        env_id: String,
    },
}

/// Base64 transport for pixel bytes, keeping remote JSON compact.
mod base64_pixels {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(pixels: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(pixels))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        STANDARD.decode(encoded).map_err(serde::de::Error::custom)
    }
}
