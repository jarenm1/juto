//! Wire contract between `juto-daemon` and its clients.
//!
//! Transport: a Unix stream socket, one JSON object per line (`\n`-terminated).
//! A client sends [`ClientFrame`]s; the daemon answers each with exactly one
//! [`ServerFrame::Response`] carrying the same `id`, and interleaves
//! [`ServerFrame::Event`]s at any time after the handshake.
//!
//! Scope: one daemon serves one workspace (`cwd`), because `Runtime` is bound to
//! a working directory. Clients locate it with [`paths::daemon_socket`].
//!
//! Versioning: the first request on every connection MUST be [`Request::Hello`]
//! with the client's [`PROTOCOL_VERSION`]. A mismatch fails with
//! [`ErrorCode::ProtocolMismatch`] and the daemon closes the connection.
//!
//! Reattach: [`Request::OpenSession`] returns a [`SessionView`] snapshot. The
//! snapshot is taken by the connection's own writer, so every event written
//! before the response is reflected in it, and events written after it are not.
//! Clients replace their session state with the snapshot, then skip any
//! [`SessionEvent::MessageCommitted`] whose `index` is below
//! `SessionView::committed`. That rule removes duplicates from the window where a
//! message is journaled but its event has not been written yet.

use std::path::PathBuf;

use juto_agent::{AgentEvent, ToolTier};
use juto_ai::{Message, StopReason, ToolCall};
use juto_catalog::Model;
use juto_runtime::subagents::MainMessage;
use juto_runtime::{
    RuntimeConfig, SubagentEvent, SubagentSummary, ThreadEvent, TimelineItem, ViewerRoute,
};
use serde::{Deserialize, Serialize};

/// Bumped on any incompatible change to this module.
pub const PROTOCOL_VERSION: u32 = 2;

/// One request from a client. `id` is chosen by the client and echoed back.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClientFrame {
    pub id: u64,
    pub request: Request,
}

/// One frame from the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerFrame {
    /// Answer to the request with the same `id`.
    Response {
        id: u64,
        result: Result<Reply, RpcError>,
    },
    /// Pushed event, delivered to every connected client.
    Event { event: ServerEvent },
}

/// Error returned for a failed request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{code:?}: {message}")]
pub struct RpcError {
    pub code: ErrorCode,
    pub message: String,
}

impl RpcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Client and daemon disagree on [`PROTOCOL_VERSION`], or no `Hello` came first.
    ProtocolMismatch,
    /// Unknown session, subagent, approval, login, or environment id.
    NotFound,
    /// The request conflicts with the current state, for example a second run
    /// while one is active.
    Busy,
    /// The request is malformed or refers to a state that cannot apply.
    Invalid,
    /// Underlying runtime, provider, credential, or desktop failure.
    Failed,
}

/// Client-to-daemon request. Externally tagged: `{"prompt":{...}}`, or the bare
/// string `"list_sessions"` for unit variants.
///
/// Deliberately not `Debug`: `SetApiKey` carries a secret.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// Must be the first request. The daemon answers with [`Reply::Hello`].
    Hello {
        client: String,
        protocol_version: u32,
    },

    // ---- sessions ----
    ListSessions,
    CreateSession,
    /// Opens a session by id and returns its committed transcript plus live state.
    OpenSession {
        session_id: String,
    },
    /// Returns the routed history of a main session, oldest first.
    Timeline {
        session_id: String,
    },
    /// Opens a session journal by path (for `--session <path>`).
    OpenSessionPath {
        path: PathBuf,
    },
    /// Starts a foreground run. Fails with [`ErrorCode::Busy`] while another run
    /// is active on this daemon (one foreground run at a time).
    Prompt {
        session_id: String,
        text: String,
    },
    /// Queues a steering message for the active run of `session_id`. Fails with
    /// [`ErrorCode::Invalid`] if that session has no active run.
    Steer {
        session_id: String,
        text: String,
    },
    /// Cancels the active run of `session_id` (and its children).
    Cancel {
        session_id: String,
    },
    /// Sets the model selector for `session_id` for its future runs.
    SetSessionModel {
        session_id: String,
        selector: String,
    },
    /// Answers a pending approval. `allow: false` denies the tool call.
    ApprovalRespond {
        approval_id: u64,
        allow: bool,
    },

    // ---- configuration and models ----
    GetConfig,
    ListModels,
    /// Persists a model role assignment and returns the updated config.
    SetModelRole {
        role: String,
        selector: String,
    },

    // ---- subagents ----
    ListSubagents,
    SubagentTranscript {
        id: String,
    },
    SubagentCommand {
        id: String,
        command: SubagentCommand,
    },

    // ---- credentials ----
    /// Lists stored credentials and which of `providers` are usable.
    CredentialStatus {
        providers: Vec<String>,
    },
    /// Stores an API key. The key is never echoed back in any reply or event.
    SetApiKey {
        provider: String,
        key: String,
    },
    RemoveCredential {
        provider: String,
    },
    /// Starts an OAuth login. Progress arrives as [`LoginEvent`]s.
    LoginStart {
        provider: String,
    },
    LoginCancel {
        login_id: u64,
    },

    // ---- desktop ----
    /// Creates a desktop environment and launches `argv` in it. Frames and input
    /// go over the desktop socket named in [`HelloInfo::desktop_socket`].
    DesktopOpen {
        argv: Vec<String>,
    },
    DesktopDestroy {
        env_id: String,
    },
    /// Resolves a desktop artifact (from a tool result's details) to a viewer route.
    ResolveDesktopArtifact {
        artifact: serde_json::Value,
    },

    // ---- inbox ----
    /// Lists the notices waiting in the main inbox for the user.
    InboxMessages,
    /// Dismisses one notice from the user's inbox. The agent keeps its copy.
    InboxDismiss {
        message: MainMessage,
    },
}

/// Actions on one subagent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentCommand {
    Pause,
    Resume,
    Cancel,
    /// Answers the child's blocking question; `resolution` is the user's reply.
    Unblock {
        resolution: String,
    },
    Release,
    /// Sends a message to the child as a steer.
    Send {
        text: String,
    },
    /// Sends a user follow-up to the child.
    FollowUp {
        text: String,
    },
}

/// Successful answer to a [`Request`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reply {
    Hello(HelloInfo),
    Sessions(Vec<SessionInfo>),
    Session(SessionView),
    /// `run_id` identifies the started run in events and run-state records.
    Started {
        run_id: String,
    },
    /// `was_running` is false if no run was active (cancel is then a no-op).
    Cancelled {
        was_running: bool,
    },
    Config(RuntimeConfig),
    Models(Vec<Model>),
    Subagents(Vec<SubagentSummary>),
    Transcript(Vec<Message>),
    /// Answers [`Request::Timeline`].
    Timeline {
        items: Vec<TimelineItem>,
    },
    Credentials(CredentialStatus),
    LoginStarted {
        login_id: u64,
    },
    DesktopEnvironment(DesktopEnvironmentInfo),
    InboxMessages {
        messages: Vec<MainMessage>,
    },
    /// `None` when the artifact is not a desktop artifact this daemon can route.
    Route(Option<ViewerRoute>),
    /// Used by requests that succeed without data.
    Ok,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloInfo {
    pub protocol_version: u32,
    pub daemon_version: String,
    pub pid: u32,
    /// Canonical workspace directory this daemon serves.
    pub cwd: PathBuf,
    pub data_dir: PathBuf,
    /// Desktop control socket for this workspace's environments.
    pub desktop_socket: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub path: PathBuf,
    pub cwd: PathBuf,
    pub created_ms: u64,
    pub model: Option<String>,
    /// Set while a run is active for this session.
    pub run: Option<RunInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub started_ms: u64,
}

/// Full state of one session, returned by [`Request::OpenSession`].
///
/// `timeline` is the committed history in time order: routing records and every
/// thread's messages. `committed` is the number of its message items; message
/// indexes below it are already reflected. Partial stream deltas of an in-progress
/// message are not replayed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionView {
    pub info: SessionInfo,
    pub timeline: Vec<TimelineItem>,
    pub committed: u64,
    /// Approvals waiting for an answer on this session's active run.
    pub pending_approvals: Vec<PendingApproval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    pub approval_id: u64,
    pub call: ToolCall,
    pub tier: ToolTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialStatus {
    pub summaries: Vec<juto_ai::CredentialSummary>,
    /// Subset of the requested providers that are usable (stored or from env).
    pub available: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopEnvironmentInfo {
    pub env_id: String,
    pub width: u32,
    pub height: u32,
    /// Grant for the viewer principal on this environment; a desktop artifact
    /// for it carries this id.
    pub grant_id: String,
    /// Host the environment runs on, matching the artifact's `host_id`.
    pub host_id: String,
}

/// Pushed events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerEvent {
    Session {
        session_id: String,
        event: SessionEvent,
    },
    /// A thread of session `session_id` was routed, created, relabeled, or went dormant.
    Thread {
        session_id: String,
        event: ThreadEvent,
    },
    Subagent {
        event: SubagentEvent,
    },
    Login {
        event: LoginEvent,
    },
    /// Config changed (for example a model role was reassigned).
    Config {
        config: RuntimeConfig,
    },
    /// The user's inbox changed; fetch it with `Request::InboxMessages`.
    Inbox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEvent {
    RunStarted {
        run_id: String,
    },
    /// A runtime event from the run, except `MessageEnd` (see
    /// [`SessionEvent::MessageCommitted`]).
    Agent {
        event: AgentEvent,
    },
    /// A message committed to the journal. `index` counts committed message
    /// entries from the start of the journal; see [`SessionView::committed`].
    MessageCommitted {
        index: u64,
        message: Message,
    },
    /// Terminal for `run_id`. `error` is set for failed or interrupted runs.
    RunFinished {
        run_id: String,
        stop_reason: Option<StopReason>,
        error: Option<String>,
    },
    ApprovalRequested {
        approval: PendingApproval,
    },
    ApprovalResolved {
        approval_id: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoginEvent {
    /// The client opens `url` locally (the browser must run on the user's machine).
    Url { login_id: u64, url: String },
    Finished {
        login_id: u64,
        error: Option<String>,
    },
}

/// Local socket paths for one workspace. Shared by daemon and client.
///
/// Layout: `$XDG_RUNTIME_DIR/juto/<workspace key>/{daemon,desktop}.sock`, where
/// the key is the first 16 hex digits of SHA-256 over the canonical workspace path.
pub mod paths {
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use sha2::{Digest, Sha256};

    pub fn workspace_key(canonical_cwd: &Path) -> String {
        let digest = Sha256::digest(canonical_cwd.as_os_str().as_encoded_bytes());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    /// `$XDG_RUNTIME_DIR/juto/<key>`. Fails if `XDG_RUNTIME_DIR` is unset.
    pub fn workspace_runtime_dir(canonical_cwd: &Path) -> Result<PathBuf> {
        let base = std::env::var_os("XDG_RUNTIME_DIR")
            .filter(|v| !v.is_empty())
            .context(
                "XDG_RUNTIME_DIR is not set; the juto daemon needs a per-user runtime directory",
            )?;
        Ok(PathBuf::from(base)
            .join("juto")
            .join(workspace_key(canonical_cwd)))
    }

    pub fn daemon_socket(canonical_cwd: &Path) -> Result<PathBuf> {
        Ok(workspace_runtime_dir(canonical_cwd)?.join("daemon.sock"))
    }

    pub fn desktop_socket(canonical_cwd: &Path) -> Result<PathBuf> {
        Ok(workspace_runtime_dir(canonical_cwd)?.join("desktop.sock"))
    }

    /// systemd `--user` unit name for the daemon serving `canonical_cwd`.
    pub fn unit_name(canonical_cwd: &Path) -> String {
        format!("juto-daemon-{}", workspace_key(canonical_cwd))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use juto_runtime::{Route, RouteKind, ThreadInfo, ThreadStatus};
    use serde::de::DeserializeOwned;
    use std::path::PathBuf;

    fn route() -> Route {
        Route {
            thread: "thread-1".into(),
            label: "Protocol surface".into(),
            kind: RouteKind::Continue,
            confidence: Some(0.9),
            rationale: Some("same topic".into()),
            query: None,
            note: None,
        }
    }

    fn thread() -> ThreadInfo {
        ThreadInfo {
            id: "thread-1".into(),
            label: "Protocol surface".into(),
            status: ThreadStatus::Active,
            session_path: PathBuf::from("/tmp/thread-1.jsonl"),
            created: 1,
            last_routed: 2,
        }
    }

    /// Serializes `value`, parses it back, and checks the JSON survives unchanged.
    fn assert_round_trips<T: Serialize + DeserializeOwned>(value: &T) {
        let json = serde_json::to_value(value).unwrap();
        let parsed: T = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), json);
    }

    #[test]
    fn route_thread_event_and_timeline_item_round_trip() {
        assert_round_trips(&route());
        assert_round_trips(&ThreadEvent::ThreadRouted { route: route() });
        assert_round_trips(&ThreadEvent::ThreadUpdated { thread: thread() });
        assert_round_trips(&ThreadEvent::ThreadLabelFailed {
            thread: "thread-1".into(),
            error: "tiny model unavailable".into(),
        });
        assert_round_trips(&TimelineItem::Routed { route: route() });
        assert_round_trips(&TimelineItem::Message {
            thread: Some("thread-1".into()),
            message: Message::user("hello"),
        });
    }

    #[test]
    fn thread_event_crosses_the_wire_as_a_server_event() {
        let event = ServerEvent::Thread {
            session_id: "session-1".into(),
            event: ThreadEvent::ThreadRouted { route: route() },
        };
        assert_round_trips(&event);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["thread"]["session_id"], "session-1");
        assert_eq!(json["thread"]["event"]["type"], "thread_routed");
    }
}
