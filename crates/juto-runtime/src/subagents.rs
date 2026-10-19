//! Bounded subagent orchestration.
//!
//! [`SubagentManager`] queues child runs behind a shared semaphore, propagates
//! the parent cancellation token to every child [`RunControl`], and keeps
//! terminal results so any number of `wait`/`output` callers observe them.
//! The runtime owns the production factory: child agents inherit its model,
//! instructions and approval policy and write independent durable sessions.
//!
//! Behavioral reference: oh-my-pi `packages/coding-agent/src/task` at
//! 579da1d6; see licenses/OMP-MIT.txt.

use std::collections::HashMap;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::session::Session;
use async_trait::async_trait;
use futures::FutureExt;
use juto_agent::{AgentEvent, RunControl};
use juto_ai::{Message, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex as AsyncMutex, Notify, Semaphore, broadcast, mpsc, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Model-visible description of one child run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpec {
    /// Optional human-facing label; the child id is always generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Role that sets the child's tools and instructions; `None` runs `task`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// The task instructions given to the child agent.
    pub task: String,
    /// Registry selector (`provider/model`) overriding the factory's base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional JSON Schema used to validate the child's structured result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Validation policy for `output_schema`; defaults to strict enforcement.
    #[serde(default)]
    pub schema_mode: SchemaMode,
    /// Restrict the child to read, glob, and grep.
    #[serde(default)]
    pub read_only: bool,
    /// Dedicated Jujutsu workspace requested.
    #[serde(default)]
    pub isolated_workspace: bool,
    /// Open-endedness / problem-solving guidance for the child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solution_space: Option<String>,
    /// A focus thread owns one orchestration-started task. It is always
    /// isolated and may only be spawned by the main thread.
    #[serde(default, skip_serializing_if = "is_false")]
    pub focus: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Policy for enforcing a task's declared structured-output schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaMode {
    /// Fail the child when its structured result is absent or invalid.
    #[default]
    Strict,
    /// Preserve the child result even when structured data is absent or invalid.
    Permissive,
}

impl TaskSpec {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            name: None,
            agent: None,
            task: task.into(),
            model: None,
            output_schema: None,
            schema_mode: SchemaMode::default(),
            read_only: false,
            isolated_workspace: false,
            solution_space: None,
            focus: false,
        }
    }
}

/// Taxonomy of blocker reasons reported by a suspended subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockerReason {
    UserInput,
    Credential,
    Environment,
    Approval,
    #[serde(untagged)]
    Custom(String),
}

impl fmt::Display for BlockerReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserInput => write!(f, "user_input"),
            Self::Credential => write!(f, "credential"),
            Self::Environment => write!(f, "environment"),
            Self::Approval => write!(f, "approval"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Terminal result of a finished child run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentResult {
    pub id: String,
    /// Final assistant text (empty when the run produced none).
    pub output: String,
    /// Structured output when the spec carried an `output_schema`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub stop_reason: StopReason,
    /// Path of the durable child session journal.
    pub session_path: PathBuf,
}

/// Lifecycle status of a spawned child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    Queued,
    Running,
    Blocked {
        reason: BlockerReason,
        prompt: String,
    },
    Verifying,
    Reviewing,
    Completed,
    Failed,
    Cancelled,
}

impl SubagentStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Blocked { .. })
    }
}

/// Status snapshot returned by [`SubagentManager::list`]. Never carries
/// credential material.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub task: String,
    pub status: SubagentStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// `true` for a focus thread that owns one orchestration-started task.
    pub focus: bool,
    /// The child's dedicated workspace path, when it runs isolated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// `true` once the main agent released this focus thread's workspace.
    pub released: bool,
}

/// Errors surfaced by [`SubagentManager`]. Distinct variants preserve the
/// explicit terminal statuses instead of collapsing them into strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentError {
    /// No child with this id was ever spawned.
    Unknown(String),
    /// `send`/`wait`/`block` targeted a child that already reached a terminal state.
    Terminal(SubagentStatus),
    /// The child run failed; the payload is the child's error.
    Failed(String),
    /// The child was cancelled before or during its run.
    Cancelled,
    /// The task spec itself is invalid (e.g. malformed `output_schema`).
    InvalidSpec(String),
    /// `spawn` was called outside a Tokio runtime.
    NoRuntime,
    /// Child was not in the blocked state when unblock was attempted.
    NotBlocked(String),
    /// Child was already blocked when block was attempted.
    AlreadyBlocked(String),
    /// Invalid state transition attempted.
    InvalidTransition {
        from: SubagentStatus,
        to: SubagentStatus,
    },
    /// Operation timed out.
    Timeout(String),
    /// The child is not a focus thread and has not completed, so it has no transcript.
    NotFocus(String),
    /// The child has no persisted transcript: it has not completed a turn, or it ended as failed/cancelled.
    NoTranscript(String),
    /// Reading the child's session file failed.
    Transcript(String),
    /// `release` targeted a child that has not reached a terminal state.
    NotTerminal(String),
    /// The child's workspace was released; follow-ups and transcripts are refused.
    WorkspaceReleased(String),
    /// Removing the child's workspace failed; the workspace stays recorded.
    WorkspaceRelease(String),
}

impl fmt::Display for SubagentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "unknown subagent id: {id}"),
            Self::Terminal(status) => write!(f, "subagent already terminal: {status:?}"),
            Self::Failed(err) => write!(f, "subagent failed: {err}"),
            Self::Cancelled => write!(f, "subagent cancelled"),
            Self::InvalidSpec(err) => write!(f, "invalid task spec: {err}"),
            Self::NoRuntime => write!(f, "no Tokio runtime available for spawn"),
            Self::NotBlocked(id) => write!(f, "subagent {id} is not blocked"),
            Self::AlreadyBlocked(id) => write!(f, "subagent {id} is already blocked"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from:?} to {to:?}")
            }
            Self::Timeout(msg) => write!(f, "subagent operation timed out: {msg}"),
            Self::NotFocus(id) => write!(f, "subagent {id} is not a focus thread"),
            Self::NoTranscript(id) => write!(f, "subagent {id} has no completed transcript"),
            Self::Transcript(err) => write!(f, "failed to read subagent transcript: {err}"),
            Self::NotTerminal(id) => {
                write!(
                    f,
                    "subagent {id} is not terminal; release it after it finishes"
                )
            }
            Self::WorkspaceReleased(id) => write!(f, "subagent {id} workspace was released"),
            Self::WorkspaceRelease(err) => {
                write!(f, "failed to release subagent workspace: {err}")
            }
        }
    }
}

impl std::error::Error for SubagentError {}

/// Executes one child run. `control` is created by the manager before queueing,
/// so `send`/`cancel` apply to queued children too. `events` is the child's
/// agent event stream; implementations forward it and own any persistence.
#[async_trait]
pub trait SubagentFactory: Send + Sync {
    async fn run(
        &self,
        id: &str,
        spec: TaskSpec,
        control: RunControl,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<SubagentResult, String>;

    /// The child's dedicated workspace path, when it runs isolated. Default
    /// for factories that never provision workspaces.
    fn workspace_path(&self, _id: &str, _spec: &TaskSpec) -> Option<PathBuf> {
        None
    }

    /// Run one more turn on a completed child: the same session journal at
    /// `session_path` continues, so the child keeps its transcript, model, and
    /// workspace. The default rejects every call; factories that keep no
    /// resumable child state leave it as-is.
    async fn follow_up(
        &self,
        _id: &str,
        _spec: &TaskSpec,
        _session_path: &Path,
        _prompt: String,
        _control: RunControl,
        _events: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<SubagentResult, String> {
        Err("follow-up not supported by this factory".into())
    }

    /// Remove the child's isolated workspace directory. Called by
    /// [`SubagentManager::release`] for a terminal child only. The default
    /// refuses, since the factory did not provision the workspace.
    async fn release_workspace(&self, _id: &str, _path: &Path) -> Result<(), String> {
        Err("workspace release not supported by this factory".into())
    }
}

/// Snapshot a child exposes to waiters; the watch channel guarantees every
/// waiter sees the latest state even if it subscribed late.
#[derive(Debug, Clone)]
pub struct ChildState {
    pub status: SubagentStatus,
    /// `Some` once the factory returned, regardless of success.
    pub outcome: Option<Result<SubagentResult, String>>,
}

/// Lifecycle notification for every child a manager owns. Observers subscribe
/// with [`SubagentManager::subscribe_events`]. A receiver that reports `Lagged`
/// should resynchronize from [`SubagentManager::list`].
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SubagentEvent {
    /// A child was queued. `focus` children always run isolated.
    Spawned {
        id: String,
        name: Option<String>,
        status: SubagentStatus,
        focus: bool,
    },
    StatusChanged {
        id: String,
        status: SubagentStatus,
    },
    /// Terminal, exactly once per turn. `result` on success, `error` otherwise.
    Finished {
        id: String,
        status: SubagentStatus,
        result: Option<SubagentResult>,
        error: Option<String>,
    },
}

/// Events buffered per manager before a slow receiver lags.
const EVENT_CAPACITY: usize = 256;

/// Kind of message delivered to the main agent's inbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MainMessageKind {
    /// A child sent a message to `agent://main`.
    Message,
    /// A child reached the `Completed` terminal state.
    Completed,
    /// A child reached the `Failed` terminal state.
    Failed,
    /// A child reached the `Cancelled` terminal state.
    Cancelled,
}

impl MainMessageKind {
    fn from_terminal(status: &SubagentStatus) -> Option<Self> {
        match status {
            SubagentStatus::Completed => Some(Self::Completed),
            SubagentStatus::Failed => Some(Self::Failed),
            SubagentStatus::Cancelled => Some(Self::Cancelled),
            _ => None,
        }
    }
}

/// One entry in the main agent's inbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MainMessage {
    /// Child id, or the sender placeholder used when a peer has no sender id.
    pub from: String,
    pub kind: MainMessageKind,
    pub body: String,
}

/// In-memory mailbox the main agent drains with `agent_inbox`. One inbox is
/// shared by every manager in a run tree, so nested children write into the
/// same main thread. Not persisted: messages are lost when the process exits.
#[derive(Clone, Default)]
pub struct MainInbox {
    messages: Arc<Mutex<Vec<MainMessage>>>,
    arrived: Arc<Notify>,
}

impl MainInbox {
    pub fn push(&self, message: MainMessage) {
        self.messages
            .lock()
            .expect("main inbox poisoned")
            .push(message);
        // Stores a permit when no waiter is parked, so a push racing a drain
        // is not missed by the next `arrived` wait.
        self.arrived.notify_one();
    }

    /// Resolves once a message is pushed. Pending messages are read with `drain`.
    pub async fn arrived(&self) {
        self.arrived.notified().await;
    }

    /// Put undelivered messages back at the front, preserving arrival order.
    pub fn requeue_front(&self, messages: Vec<MainMessage>) {
        if messages.is_empty() {
            return;
        }
        self.messages
            .lock()
            .expect("main inbox poisoned")
            .splice(0..0, messages);
        self.arrived.notify_one();
    }

    /// Remove and return every pending message in arrival order.
    pub fn drain(&self) -> Vec<MainMessage> {
        std::mem::take(&mut *self.messages.lock().expect("main inbox poisoned"))
    }
}

/// Where a child's isolated workspace stands. `Releasing` covers the cleanup
/// window so no follow-up or transcript read can race the directory removal.
enum WorkspaceState {
    /// The child never had an isolated workspace.
    Unprovisioned,
    Live(PathBuf),
    Releasing,
    Released,
}

struct Child {
    id: String,
    spec: TaskSpec,
    focus: bool,
    workspace: Mutex<WorkspaceState>,
    /// Replaced per turn: a fresh turn gets a fresh control so a cancelled or
    /// paused first turn cannot poison a follow-up. Always lock it before
    /// touching it so a turn boundary cannot swap mid-call.
    control: Mutex<RunControl>,
    state: watch::Sender<ChildState>,
    status_tx: watch::Sender<SubagentStatus>,
    /// Manager-wide event stream this child reports into.
    events: broadcast::Sender<SubagentEvent>,
    resolution_tx: Mutex<Option<tokio::sync::oneshot::Sender<String>>>,
    resolution_rx: Mutex<Option<tokio::sync::oneshot::Receiver<String>>>,
    /// Serializes `release` calls so a retry waits for the attempt in flight.
    release_gate: AsyncMutex<()>,
}

impl Child {
    fn workspace_path(&self) -> Option<PathBuf> {
        let guard = self.workspace.lock().expect("mutex poisoned");
        match &*guard {
            WorkspaceState::Live(path) => Some(path.clone()),
            _ => None,
        }
    }

    /// True once a release has started or finished. Such a child refuses
    /// follow-ups and transcript reads.
    fn workspace_released(&self) -> bool {
        let guard = self.workspace.lock().expect("mutex poisoned");
        matches!(
            &*guard,
            WorkspaceState::Releasing | WorkspaceState::Released
        )
    }

    /// Publish a status and announce it on the manager's event stream. A repeat
    /// of the current status is not announced again.
    fn set_status(&self, status: SubagentStatus) {
        if *self.status_tx.borrow() == status {
            return;
        }
        self.status_tx
            .send_modify(|current| *current = status.clone());
        let _ = self.events.send(SubagentEvent::StatusChanged {
            id: self.id.clone(),
            status,
        });
    }
}

struct Inner {
    factory: Arc<dyn SubagentFactory>,
    semaphore: Arc<Semaphore>,
    main_inbox: MainInbox,
    events: broadcast::Sender<SubagentEvent>,
    children: Mutex<HashMap<String, Arc<Child>>>,
}

/// Bounded manager for child subagent runs.
///
/// `spawn` never blocks on a semaphore permit; acquisition happens inside the
/// child task so permits are held only by running children.
pub struct SubagentManager {
    inner: Arc<Inner>,
    /// Owns the children this handle spawns: cancelling it stops them.
    owner: CancellationToken,
}

impl SubagentManager {
    pub fn new(
        factory: Arc<dyn SubagentFactory>,
        max_concurrency: usize,
        owner: CancellationToken,
        main_inbox: MainInbox,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                factory,
                semaphore: Arc::new(Semaphore::new(max_concurrency.max(1))),
                main_inbox,
                children: Mutex::new(HashMap::new()),
                events: broadcast::channel(EVENT_CAPACITY).0,
            }),
            owner,
        }
    }

    /// The inbox every child of this manager pushes into. Shared with nested
    /// managers so the whole run tree reports to one main thread.
    pub fn main_inbox(&self) -> MainInbox {
        self.inner.main_inbox.clone()
    }

    /// Token that owns the children this handle spawns; cancelling it stops
    /// them. The registry, semaphore, and inbox are shared by every handle.
    pub(crate) fn owner(&self) -> CancellationToken {
        self.owner.clone()
    }

    /// A handle over the same registry, semaphore, and inbox whose children
    /// are owned by `owner`.
    pub(crate) fn scoped(&self, owner: CancellationToken) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            owner,
        }
    }

    /// Subscribe to lifecycle events for every child this manager owns. A
    /// receiver that lags should resynchronize from [`SubagentManager::list`].
    pub fn subscribe_events(&self) -> broadcast::Receiver<SubagentEvent> {
        self.inner.events.subscribe()
    }

    /// Queue a child run. Returns the generated child id immediately; the
    /// permit wait, run, and terminal-state publication are asynchronous.
    pub fn spawn(&self, mut spec: TaskSpec) -> Result<String, SubagentError> {
        if spec.focus {
            // A focus thread tracks one task through its own VCS view, so it
            // always needs a dedicated workspace regardless of what the
            // caller requested.
            spec.isolated_workspace = true;
        }
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(SubagentError::NoRuntime);
        }
        // Reject malformed output schemas at spawn time, before any work queues.
        let validator = match &spec.output_schema {
            Some(schema) => Some(
                jsonschema::validator_for(schema)
                    .map_err(|err| SubagentError::InvalidSpec(err.to_string()))?,
            ),
            None => None,
        };

        let id = Uuid::now_v7().to_string();
        let control = RunControl::new();
        let (state, _rx) = watch::channel(ChildState {
            status: SubagentStatus::Queued,
            outcome: None,
        });
        let (status_tx, _) = watch::channel(SubagentStatus::Queued);
        let focus = spec.focus;
        let workspace = self.inner.factory.workspace_path(&id, &spec);
        let child = Arc::new(Child {
            id: id.clone(),
            spec,
            focus,
            workspace: Mutex::new(match workspace {
                Some(path) => WorkspaceState::Live(path),
                None => WorkspaceState::Unprovisioned,
            }),
            release_gate: AsyncMutex::new(()),
            control: Mutex::new(control.clone()),
            state,
            status_tx,
            resolution_tx: Mutex::new(None),
            resolution_rx: Mutex::new(None),
            events: self.inner.events.clone(),
        });
        self.inner
            .children
            .lock()
            .expect("subagent registry poisoned")
            .insert(id.clone(), Arc::clone(&child));
        let _ = self.inner.events.send(SubagentEvent::Spawned {
            id: id.clone(),
            name: child.spec.name.clone(),
            status: SubagentStatus::Queued,
            focus,
        });

        let inner = Arc::clone(&self.inner);
        let task_id = id.clone();
        tokio::spawn(run_child(
            inner,
            child,
            task_id,
            control,
            self.owner.clone(),
            validator,
        ));
        Ok(id)
    }

    /// Wait for the child's terminal state and return its result.
    /// Concurrent and post-completion waiters all observe the same outcome.
    pub async fn wait(&self, id: &str) -> Result<SubagentResult, SubagentError> {
        let mut rx = self.state_receiver(id)?;
        loop {
            let snapshot = rx.borrow().clone();
            match snapshot.status {
                SubagentStatus::Queued
                | SubagentStatus::Running
                | SubagentStatus::Verifying
                | SubagentStatus::Reviewing
                | SubagentStatus::Blocked { .. } => {}
                SubagentStatus::Completed => match snapshot.outcome {
                    Some(Ok(result)) => return Ok(result),
                    Some(Err(err)) => return Err(SubagentError::Failed(err)),
                    None => {
                        return Err(SubagentError::Failed(
                            "child completed without a result".into(),
                        ));
                    }
                },
                SubagentStatus::Failed => match snapshot.outcome {
                    Some(Err(err)) => return Err(SubagentError::Failed(err)),
                    _ => {
                        return Err(SubagentError::Failed(
                            "child failed without an error message".into(),
                        ));
                    }
                },
                SubagentStatus::Cancelled => return Err(SubagentError::Cancelled),
            }
            rx.changed().await.map_err(|_| {
                SubagentError::Failed("child terminated without publishing a result".into())
            })?;
        }
    }

    /// Cancel a queued or running child. Idempotent for terminal children.
    pub fn cancel(&self, id: &str) -> Result<(), SubagentError> {
        self.child(id)?
            .control
            .lock()
            .expect("mutex poisoned")
            .cancel();
        Ok(())
    }

    /// Message a child by id. A queued or running child takes the message as a
    /// steer at its next turn boundary. A completed child starts a new turn with
    /// the message as its prompt, as [`SubagentManager::follow_up`] does. Failed
    /// and cancelled children reject it with [`SubagentError::Terminal`].
    pub fn send(&self, id: &str, message: String) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let status = child.state.borrow().status.clone();
        if status == SubagentStatus::Completed {
            return self.follow_up(id, message);
        }
        if status.is_terminal() {
            return Err(SubagentError::Terminal(status));
        }
        child
            .control
            .lock()
            .expect("mutex poisoned")
            .steer(Message::user(message));
        Ok(())
    }

    /// Start a follow-up turn on a completed child, as main requested through
    /// `agent_send`. Its terminal state also pushes a completion notice to the
    /// main inbox.
    pub fn follow_up(&self, id: &str, prompt: String) -> Result<(), SubagentError> {
        self.start_follow_up(id, prompt, TurnOrigin::Main)
    }

    /// Start a follow-up turn the user typed into a focus thread. The turn gets
    /// `USER_TURN_PREFACE` first and reports only to the focus view.
    pub fn follow_up_from_user(&self, id: &str, prompt: String) -> Result<(), SubagentError> {
        let prompt = format!("{USER_TURN_PREFACE}\n\n{prompt}");
        self.start_follow_up(id, prompt, TurnOrigin::User)
    }

    /// Start a follow-up turn on a completed child. The child moves back to
    /// [`SubagentStatus::Running`], runs exactly one more turn in its original
    /// session and workspace under a fresh [`RunControl`], then publishes its
    /// own terminal state; `wait`/`output` then observe the new turn's result.
    ///
    /// Only `Completed` children accept a follow-up: `Failed` and `Cancelled`
    /// children never ran to a usable result, so they stay rejected with
    /// [`SubagentError::Terminal`], and any non-terminal state rejects with
    /// [`SubagentError::InvalidTransition`]. Calling again while a follow-up
    /// turn is in flight also rejects with `InvalidTransition`.
    fn start_follow_up(
        &self,
        id: &str,
        prompt: String,
        origin: TurnOrigin,
    ) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        // Read the session path before the transition clears the outcome.
        let session_path = child
            .state
            .borrow()
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.as_ref().ok())
            .map(|result| result.session_path.clone());

        // The gate and the Running transition share the control mutex so a
        // racing second call can never install its own control for the turn.
        let mut control_guard = child.control.lock().expect("mutex poisoned");
        if child.workspace_released() {
            return Err(SubagentError::WorkspaceReleased(id.to_string()));
        }
        let current = child.state.borrow().status.clone();
        if current.is_terminal() && current != SubagentStatus::Completed {
            return Err(SubagentError::Terminal(current));
        }
        let transitioned = child.state.send_if_modified(|state| {
            if state.status == SubagentStatus::Completed {
                state.status = SubagentStatus::Running;
                state.outcome = None;
                true
            } else {
                false
            }
        });
        if !transitioned {
            return Err(SubagentError::InvalidTransition {
                from: current,
                to: SubagentStatus::Running,
            });
        }
        *control_guard = RunControl::new();
        let control = control_guard.clone();
        drop(control_guard);

        child.set_status(SubagentStatus::Running);

        tokio::spawn(run_follow_up(
            Arc::clone(&self.inner),
            child,
            self.owner.clone(),
            session_path,
            prompt,
            control,
            origin,
        ));
        Ok(())
    }

    /// Snapshot of every known child, newest last.
    pub fn list(&self) -> Vec<SubagentSummary> {
        let mut summaries: Vec<SubagentSummary> = self
            .inner
            .children
            .lock()
            .expect("subagent registry poisoned")
            .iter()
            .map(|(id, child)| {
                let state = child.state.borrow();
                let (stop_reason, usage, error) = match &state.outcome {
                    Some(Ok(result)) => {
                        (Some(result.stop_reason), Some(result.usage.clone()), None)
                    }
                    Some(Err(err)) => (None, None, Some(err.clone())),
                    None => (None, None, None),
                };
                SubagentSummary {
                    id: id.clone(),
                    name: child.spec.name.clone(),
                    task: child.spec.task.clone(),
                    status: state.status.clone(),
                    stop_reason,
                    usage,
                    focus: child.focus,
                    workspace: child.workspace_path(),
                    released: child.workspace_released(),
                    error,
                }
            })
            .collect();
        summaries.sort_by(|a, b| a.id.cmp(&b.id));
        summaries
    }

    /// Successful result for a completed child; pending, failed, and cancelled children have none.
    pub fn output(&self, id: &str) -> Result<Option<SubagentResult>, SubagentError> {
        let child = self.child(id)?;
        let state = child.state.borrow();
        Ok((state.status == SubagentStatus::Completed)
            .then(|| {
                state
                    .outcome
                    .as_ref()
                    .and_then(|outcome| outcome.as_ref().ok())
                    .cloned()
            })
            .flatten())
    }

    /// Persisted transcript of a completed child, read from its session file.
    /// Focus and non-focus children both qualify once they complete. Pending,
    /// failed, and cancelled children have none: a non-focus child reports
    /// `NotFocus`, a focus child reports `NoTranscript`.
    pub fn transcript(&self, id: &str) -> Result<Vec<Message>, SubagentError> {
        let child = self.child(id)?;
        if child.workspace_released() {
            return Err(SubagentError::WorkspaceReleased(id.to_string()));
        }
        let path = {
            let state = child.state.borrow();
            match (&state.status, &state.outcome) {
                (SubagentStatus::Completed, Some(Ok(result))) => result.session_path.clone(),
                _ if !child.focus => return Err(SubagentError::NotFocus(id.to_string())),
                _ => return Err(SubagentError::NoTranscript(id.to_string())),
            }
        };
        let session =
            Session::open(&path).map_err(|err| SubagentError::Transcript(err.to_string()))?;
        session
            .messages()
            .map_err(|err| SubagentError::Transcript(err.to_string()))
    }

    /// Remove a terminal child's isolated workspace. Main calls this once it no
    /// longer needs the child's VCS state. Cleanup runs through the factory's
    /// async path, so no blocking work lands on the tokio worker. A child refuses
    /// follow-ups and transcript reads after a successful release. A failed
    /// cleanup restores the recorded workspace, so a retry is possible.
    pub async fn release(&self, id: &str) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        // Serialize releases: a second caller waits for the first and then sees
        // the settled state.
        let _gate = child.release_gate.lock().await;
        let path = {
            // Lock order is control, then workspace, matching follow-up.
            let _control = child.control.lock().expect("mutex poisoned");
            let status = child.state.borrow().status.clone();
            if !status.is_terminal() {
                return Err(SubagentError::NotTerminal(id.to_string()));
            }
            let mut workspace = child.workspace.lock().expect("mutex poisoned");
            match std::mem::replace(&mut *workspace, WorkspaceState::Releasing) {
                WorkspaceState::Live(path) => path,
                settled => {
                    *workspace = settled;
                    return Ok(());
                }
            }
        };
        match self.inner.factory.release_workspace(id, &path).await {
            Ok(()) => {
                *child.workspace.lock().expect("mutex poisoned") = WorkspaceState::Released;
                Ok(())
            }
            Err(err) => {
                *child.workspace.lock().expect("mutex poisoned") = WorkspaceState::Live(path);
                Err(SubagentError::WorkspaceRelease(err))
            }
        }
    }

    /// Status of one child, `None` for unknown ids.
    pub fn status(&self, id: &str) -> Option<SubagentStatus> {
        self.child(id)
            .ok()
            .map(|child| child.state.borrow().status.clone())
    }
    /// Pause execution of a running child agent.
    pub fn pause(&self, id: &str) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let current = child.state.borrow().status.clone();
        if current.is_terminal() {
            return Err(SubagentError::Terminal(current));
        }
        child.control.lock().expect("mutex poisoned").pause();
        Ok(())
    }

    /// Resume execution of a paused child agent without resolving a blocker.
    pub fn resume(&self, id: &str) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let current = child.state.borrow().status.clone();
        if current.is_terminal() {
            return Err(SubagentError::Terminal(current));
        }
        child.control.lock().expect("mutex poisoned").resume();
        Ok(())
    }

    /// Transition a child into the Blocked state.
    ///
    /// Suspends child progress via [`RunControl::pause`], sets status to
    /// [`SubagentStatus::Blocked`], and prepares an unblock signaling channel.
    pub fn block(
        &self,
        id: &str,
        reason: BlockerReason,
        prompt: impl Into<String>,
    ) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let prompt = prompt.into();
        let current = child.state.borrow().status.clone();
        if current.is_terminal() {
            return Err(SubagentError::Terminal(current));
        }
        if current == SubagentStatus::Queued {
            return Err(SubagentError::InvalidTransition {
                from: current,
                to: SubagentStatus::Blocked { reason, prompt },
            });
        }
        if current.is_blocked() {
            return Err(SubagentError::AlreadyBlocked(id.to_string()));
        }

        let (tx, rx) = tokio::sync::oneshot::channel();
        *child.resolution_tx.lock().expect("mutex poisoned") = Some(tx);
        *child.resolution_rx.lock().expect("mutex poisoned") = Some(rx);

        child.control.lock().expect("mutex poisoned").pause();
        let new_status = SubagentStatus::Blocked { reason, prompt };
        child.state.send_modify(|state| {
            state.status = new_status.clone();
        });
        child.set_status(new_status);
        Ok(())
    }

    /// Asynchronously wait for resolution to a previously reported blocker.
    ///
    /// Respects child cancellation and parent cancellation so waiting never hangs.
    pub async fn wait_for_resolution(&self, id: &str) -> Result<String, SubagentError> {
        let child = self.child(id)?;
        let rx = child
            .resolution_rx
            .lock()
            .expect("mutex poisoned")
            .take()
            .ok_or_else(|| SubagentError::NotBlocked(id.to_string()))?;
        let token = child.control.lock().expect("mutex poisoned").token();
        let owner = self.owner.clone();

        tokio::select! {
            biased;
            _ = token.cancelled() => Err(SubagentError::Cancelled),
            _ = owner.cancelled() => Err(SubagentError::Cancelled),
            res = rx => match res {
                Ok(resolution) => Ok(resolution),
                Err(_) => {
                    if token.is_cancelled() || owner.is_cancelled() {
                        Err(SubagentError::Cancelled)
                    } else {
                        Err(SubagentError::Failed("resolution channel dropped".into()))
                    }
                }
            }
        }
    }

    /// Wait for resolution with an explicit timeout.
    pub async fn wait_for_resolution_timeout(
        &self,
        id: &str,
        duration: std::time::Duration,
    ) -> Result<String, SubagentError> {
        match tokio::time::timeout(duration, self.wait_for_resolution(id)).await {
            Ok(result) => result,
            Err(_) => Err(SubagentError::Timeout(format!(
                "timed out waiting for resolution on child {id}"
            ))),
        }
    }

    /// Block and wait for resolution in one step.
    pub async fn block_and_wait(
        &self,
        id: &str,
        reason: BlockerReason,
        prompt: impl Into<String>,
    ) -> Result<String, SubagentError> {
        self.block(id, reason, prompt)?;
        self.wait_for_resolution(id).await
    }

    /// Block and wait for resolution with a timeout.
    pub async fn block_and_wait_timeout(
        &self,
        id: &str,
        reason: BlockerReason,
        prompt: impl Into<String>,
        duration: std::time::Duration,
    ) -> Result<String, SubagentError> {
        self.block(id, reason, prompt)?;
        self.wait_for_resolution_timeout(id, duration).await
    }

    /// Provide a resolution to a blocked child, transitioning it back to Running.
    ///
    /// Resumes the child's [`RunControl`], steers the child with the resolution
    /// message, and delivers the resolution string to the waiter.
    pub fn unblock(&self, id: &str, resolution: impl Into<String>) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let resolution = resolution.into();
        let current = child.state.borrow().status.clone();
        if !current.is_blocked() {
            return Err(SubagentError::NotBlocked(id.to_string()));
        }

        let tx = child
            .resolution_tx
            .lock()
            .expect("mutex poisoned")
            .take()
            .ok_or_else(|| SubagentError::NotBlocked(id.to_string()))?;
        let is_credential = matches!(
            current,
            SubagentStatus::Blocked {
                reason: BlockerReason::Credential,
                ..
            }
        );

        child.state.send_modify(|state| {
            state.status = SubagentStatus::Running;
        });
        child.set_status(SubagentStatus::Running);

        let control = child.control.lock().expect("mutex poisoned");
        control.resume();
        if !is_credential {
            control.steer(Message::user(resolution.clone()));
        }

        let _ = tx.send(resolution);
        Ok(())
    }

    /// Explicitly transition a running child into a new non-terminal status.
    pub fn transition(&self, id: &str, to: SubagentStatus) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let current = child.state.borrow().status.clone();
        if current.is_terminal() {
            return Err(SubagentError::Terminal(current));
        }
        if to.is_terminal() || to.is_blocked() || to == SubagentStatus::Queued {
            return Err(SubagentError::InvalidTransition { from: current, to });
        }
        if current == SubagentStatus::Queued || current.is_blocked() {
            return Err(SubagentError::InvalidTransition { from: current, to });
        }

        child.state.send_modify(|state| {
            state.status = to.clone();
        });
        child.set_status(to);
        Ok(())
    }

    /// Mark child as verifying.
    pub fn set_verifying(&self, id: &str) -> Result<(), SubagentError> {
        self.transition(id, SubagentStatus::Verifying)
    }

    /// Mark child as reviewing.
    pub fn set_reviewing(&self, id: &str) -> Result<(), SubagentError> {
        self.transition(id, SubagentStatus::Reviewing)
    }

    /// Mark child as running.
    pub fn set_running(&self, id: &str) -> Result<(), SubagentError> {
        self.transition(id, SubagentStatus::Running)
    }

    /// Subscribe to full child state updates (status + terminal outcome).
    pub fn subscribe(&self, id: &str) -> Result<watch::Receiver<ChildState>, SubagentError> {
        Ok(self.child(id)?.state.subscribe())
    }

    /// Subscribe to status-only updates.
    pub fn subscribe_status(
        &self,
        id: &str,
    ) -> Result<watch::Receiver<SubagentStatus>, SubagentError> {
        Ok(self.child(id)?.status_tx.subscribe())
    }

    /// Wait for terminal result with an explicit timeout.
    pub async fn wait_timeout(
        &self,
        id: &str,
        duration: std::time::Duration,
    ) -> Result<SubagentResult, SubagentError> {
        match tokio::time::timeout(duration, self.wait(id)).await {
            Ok(result) => result,
            Err(_) => Err(SubagentError::Timeout(format!(
                "timed out waiting for child {id}"
            ))),
        }
    }

    fn child(&self, id: &str) -> Result<Arc<Child>, SubagentError> {
        self.inner
            .children
            .lock()
            .expect("subagent registry poisoned")
            .get(id)
            .cloned()
            .ok_or_else(|| SubagentError::Unknown(id.to_string()))
    }

    fn state_receiver(&self, id: &str) -> Result<watch::Receiver<ChildState>, SubagentError> {
        Ok(self.child(id)?.state.subscribe())
    }
}

/// Cancellation-safe permit acquisition shared by first turns and follow-up
/// turns; publishes the terminal state itself on the failure paths.
async fn acquire_permit(
    inner: &Inner,
    child: &Child,
    control: &RunControl,
    owner: &CancellationToken,
    origin: TurnOrigin,
) -> Option<tokio::sync::OwnedSemaphorePermit> {
    let cancel = control.token();
    let permit = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            finish(inner, child, SubagentStatus::Cancelled, Err("cancelled while queued".into()), origin);
            return None;
        }
        _ = owner.cancelled() => {
            finish(inner, child, SubagentStatus::Cancelled, Err("cancelled with parent while queued".into()), origin);
            return None;
        }
        permit = inner.semaphore.clone().acquire_owned() => match permit {
            Ok(permit) => permit,
            Err(_) => {
                finish(inner, child, SubagentStatus::Failed, Err("subagent semaphore closed".into()), origin);
                return None;
            }
        },
    };
    Some(permit)
}

/// Lifecycle for one spawned child: cancellation-safe permit wait, then one
/// turn through the shared settle logic.
async fn run_child(
    inner: Arc<Inner>,
    child: Arc<Child>,
    id: String,
    control: RunControl,
    owner: CancellationToken,
    validator: Option<jsonschema::Validator>,
) {
    let Some(_permit) = acquire_permit(&inner, &child, &control, &owner, TurnOrigin::Main).await
    else {
        return;
    };

    child.state.send_modify(|state| {
        state.status = SubagentStatus::Running;
    });
    child.set_status(SubagentStatus::Running);

    let factory = inner.factory.clone();
    let spec = child.spec.clone();
    let task_id = id.clone();
    let task_control = control.clone();
    run_turn(
        &inner,
        &child,
        &control,
        &owner,
        validator,
        TurnOrigin::Main,
        move |events| async move { factory.run(&task_id, spec, task_control, events).await },
    )
    .await;
}

/// One additional turn on a completed child: same session journal, same
/// workspace, fresh [`RunControl`]. Like the first turn it waits on the
/// shared semaphore and publishes exactly one terminal state; `follow_up`
/// already moved the child back to `Running`.
async fn run_follow_up(
    inner: Arc<Inner>,
    child: Arc<Child>,
    owner: CancellationToken,
    session_path: Option<PathBuf>,
    prompt: String,
    control: RunControl,
    origin: TurnOrigin,
) {
    let Some(_permit) = acquire_permit(&inner, &child, &control, &owner, origin).await else {
        return;
    };

    let validator = child
        .spec
        .output_schema
        .as_ref()
        .and_then(|schema| jsonschema::validator_for(schema).ok());
    let id = child.id.clone();
    let factory = inner.factory.clone();
    let spec = child.spec.clone();
    let task_control = control.clone();
    run_turn(
        &inner,
        &child,
        &control,
        &owner,
        validator,
        origin,
        move |events| async move {
            match session_path {
                Some(path) => {
                    factory
                        .follow_up(&id, &spec, &path, prompt, task_control, events)
                        .await
                }
                None => Err("completed child has no recorded session path".into()),
            }
        },
    )
    .await;
}

/// Run one factory turn and publish exactly one terminal state: parent
/// cancellation propagation, the awaited turn (panic-safe), output-schema
/// enforcement, and the cancelled/failed/completed mapping.
async fn run_turn<F, Fut>(
    inner: &Inner,
    child: &Child,
    control: &RunControl,
    owner: &CancellationToken,
    validator: Option<jsonschema::Validator>,
    origin: TurnOrigin,
    turn: F,
) where
    F: FnOnce(mpsc::UnboundedSender<AgentEvent>) -> Fut,
    Fut: Future<Output = Result<SubagentResult, String>>,
{
    // Propagate parent cancellation into the child's RunControl for the whole
    // turn, then release the propagation task when the turn finishes.
    let propagation_done = CancellationToken::new();
    tokio::spawn({
        let owner = owner.clone();
        let control = control.clone();
        let done = propagation_done.clone();
        async move {
            tokio::select! {
                _ = owner.cancelled() => control.cancel(),
                _ = done.cancelled() => {}
            }
        }
    });

    let (events_tx, events_rx) = mpsc::unbounded_channel();
    // The manager owns no consumer for child events; keep the receiver alive
    // so `UnboundedSender::send` does not fail during the run.
    let _events_rx = events_rx;

    let run = turn(events_tx);
    // A panicking factory must still publish a terminal state or every waiter
    // would hang.
    let outcome = AssertUnwindSafe(run).catch_unwind().await;

    propagation_done.cancel();

    let outcome = match outcome {
        Ok(Ok(mut result)) => {
            if result.data.is_none() {
                result.data = extract_json(&result.output);
            }
            Ok(result)
        }
        Ok(Err(err)) => Err(err),
        Err(payload) => Err(format!(
            "subagent panicked: {}",
            panic_message(payload.as_ref())
        )),
    };

    // Enforce declared output schemas against whatever the factory produced;
    // a completed child still fails when its data is absent or invalid.
    let outcome = match (validator, child.spec.schema_mode, outcome) {
        (Some(validator), SchemaMode::Strict, Ok(result)) => match &result.data {
            Some(data) => match validator.validate(data) {
                Ok(()) => Ok(result),
                Err(err) => Err(format!("output schema violation: {err}")),
            },
            None => Err("child produced no structured output for output_schema".into()),
        },
        (_, SchemaMode::Permissive, outcome) | (None, SchemaMode::Strict, outcome) => outcome,
        (Some(_), SchemaMode::Strict, Err(err)) => Err(err),
    };

    let cancelled = control.token().is_cancelled() || owner.is_cancelled();
    let status = match &outcome {
        Ok(result) if result.stop_reason == StopReason::Aborted || cancelled => {
            SubagentStatus::Cancelled
        }
        Ok(_) => SubagentStatus::Completed,
        Err(_) if cancelled => SubagentStatus::Cancelled,
        Err(_) => SubagentStatus::Failed,
    };
    finish(inner, child, status, outcome, origin);
}

/// Prepended to follow-ups the user types in a focus thread. The contract main
/// gave at spawn stays in the session history, so the thread keeps it.
const USER_TURN_PREFACE: &str = "The user is chatting with you directly in your focus thread. Answer them. Your contract from main still applies, including what to report to main and when.";

/// Who started a turn. Main-started turns (the first run and main's follow-ups)
/// push a completion notice to the main inbox. User-started focus turns do not:
/// the user reads their result in the focus view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOrigin {
    Main,
    User,
}

/// Publish the terminal state for a turn — exactly once per turn. The main
/// inbox notice is pushed before the state is published, so a waiter woken by
/// the state already observes the notice.
fn finish(
    inner: &Inner,
    child: &Child,
    status: SubagentStatus,
    outcome: Result<SubagentResult, String>,
    origin: TurnOrigin,
) {
    debug_assert!(status.is_terminal());
    let kind = MainMessageKind::from_terminal(&status).filter(|_| origin == TurnOrigin::Main);
    if let Some(kind) = kind {
        let body = match &outcome {
            Ok(result) => result.output.clone(),
            Err(err) => err.clone(),
        };
        inner.main_inbox.push(MainMessage {
            from: child.id.clone(),
            kind,
            body,
        });
    }
    let (result, error) = match &outcome {
        Ok(result) => (Some(result.clone()), None),
        Err(err) => (None, Some(err.clone())),
    };
    child.state.send_modify(|state| {
        state.status = status.clone();
        state.outcome = Some(outcome);
    });
    child.set_status(status.clone());
    let _ = inner.events.send(SubagentEvent::Finished {
        id: child.id.clone(),
        status,
        result,
        error,
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|msg| (*msg).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// Best-effort JSON extraction from assistant text: whole response first,
/// then the last ```json fence, then the outermost trailing `{...}`/`[...]`.
pub fn extract_json(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    // Last fenced block preferring ```json.
    let mut fenced: Option<Value> = None;
    let mut rest = trimmed;
    while let Some(open) = rest.find("```") {
        let after_marker = &rest[open + 3..];
        let body_start = after_marker
            .find('\n')
            .map(|newline| newline + 1)
            .unwrap_or(0);
        let body = &after_marker[body_start..];
        match body.find("```") {
            Some(close) => {
                let candidate = body[..close].trim();
                if let Ok(value) = serde_json::from_str::<Value>(candidate) {
                    fenced = Some(value);
                }
                rest = &body[close + 3..];
            }
            None => break,
        }
    }
    if fenced.is_some() {
        return fenced;
    }
    // Fallback: outermost trailing object/array slice.
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(start), Some(end)) = (trimmed.find(open), trimmed.rfind(close))
            && start < end
            && let Ok(value) = serde_json::from_str::<Value>(&trimmed[start..=end])
        {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::time::{Duration as TokioDuration, timeout};

    const WAIT_TIMEOUT: TokioDuration = TokioDuration::from_secs(5);

    fn spec(task: &str) -> TaskSpec {
        TaskSpec {
            name: None,
            agent: None,
            task: task.to_string(),
            model: None,
            output_schema: None,
            schema_mode: SchemaMode::default(),
            read_only: false,
            isolated_workspace: false,
            solution_space: None,
            focus: false,
        }
    }

    /// Gate for test children. `watch` stores state instead of edges, so a
    /// child that starts waiting after the gate opened can never miss it.
    fn gate() -> (watch::Sender<bool>, watch::Receiver<bool>) {
        watch::channel(false)
    }

    /// Factory whose runs block on the gate, recording live/max concurrency.
    struct GateFactory {
        gate: watch::Receiver<bool>,
        running: Arc<AtomicUsize>,
        max_running: Arc<AtomicUsize>,
        fail: bool,
        data: Option<Value>,
    }

    #[async_trait]
    impl SubagentFactory for GateFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_running.fetch_max(running, Ordering::SeqCst);
            let mut gate = self.gate.clone();
            let cancel = control.token();
            let result = tokio::select! {
                _ = cancel.cancelled() => Err("cancelled".to_string()),
                opened = gate.wait_for(|open| *open) => match opened {
                    Ok(_) if self.fail => Err("boom".to_string()),
                    Ok(_) => Ok(SubagentResult {
                        id: id.to_string(),
                        output: format!("done:{}", spec.task),
                        data: self.data.clone(),
                        usage: Usage::default(),
                        stop_reason: StopReason::Stop,
                        session_path: PathBuf::from("/tmp/child.jsonl"),
                    }),
                    Err(_) => Err("gate closed".to_string()),
                },
            };
            self.running.fetch_sub(1, Ordering::SeqCst);
            result
        }
    }

    fn gate_factory(
        gate: watch::Receiver<bool>,
        running: Arc<AtomicUsize>,
        max_running: Arc<AtomicUsize>,
        fail: bool,
        data: Option<Value>,
    ) -> Arc<GateFactory> {
        Arc::new(GateFactory {
            gate,
            running,
            max_running,
            fail,
            data,
        })
    }

    fn manager(
        factory: Arc<dyn SubagentFactory>,
        max_concurrency: usize,
        owner: CancellationToken,
    ) -> SubagentManager {
        SubagentManager::new(factory, max_concurrency, owner, MainInbox::default())
    }

    /// Poll until `running` reaches `n`. Sleeps between checks so blocking
    /// filesystem work inside the child can finish; yields alone run out.
    async fn until_running(running: &AtomicUsize, n: usize) {
        let reached = async {
            while running.load(Ordering::SeqCst) != n {
                tokio::time::sleep(TokioDuration::from_millis(1)).await;
            }
        };
        if timeout(WAIT_TIMEOUT, reached).await.is_err() {
            panic!(
                "expected {n} running children, got {}",
                running.load(Ordering::SeqCst)
            );
        }
    }

    #[tokio::test]
    async fn spawn_returns_before_run_and_wait_yields_result() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );

        // Must not block: the child's gate stays closed past spawn return.
        let id = timeout(WAIT_TIMEOUT, async {
            manager.spawn(spec("alpha")).unwrap()
        })
        .await
        .expect("spawn blocked");
        until_running(&running, 1).await;
        assert_eq!(manager.status(&id), Some(SubagentStatus::Running));

        gate.send(true).unwrap();
        let result = timeout(WAIT_TIMEOUT, manager.wait(&id))
            .await
            .expect("wait timed out")
            .expect("wait failed");
        assert_eq!(result.id, id);
        assert_eq!(result.output, "done:alpha");
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn semaphore_bounds_concurrent_children() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let max_running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(gate_rx, running.clone(), max_running.clone(), false, None),
            2,
            CancellationToken::new(),
        );
        let mut ids = Vec::new();
        for n in 0..5 {
            ids.push(manager.spawn(spec(&format!("task-{n}"))).unwrap());
        }
        // Closed gate: only the permit count may be running.
        until_running(&running, 2).await;
        tokio::time::sleep(TokioDuration::from_millis(20)).await;
        assert_eq!(running.load(Ordering::SeqCst), 2);

        gate.send(true).unwrap();
        for id in &ids {
            timeout(WAIT_TIMEOUT, manager.wait(id))
                .await
                .expect("wait timed out")
                .unwrap_or_else(|err| panic!("child {id} failed: {err}"));
        }
        assert_eq!(max_running.load(Ordering::SeqCst), 2);
        assert_eq!(running.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn queued_child_waits_for_permit() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let a = manager.spawn(spec("a")).unwrap();
        let b = manager.spawn(spec("b")).unwrap();
        until_running(&running, 1).await;
        assert_eq!(manager.status(&b), Some(SubagentStatus::Queued));
        gate.send(true).unwrap();
        assert!(manager.wait(&a).await.is_ok());
        assert!(manager.wait(&b).await.is_ok());
    }

    #[tokio::test]
    async fn concurrent_and_late_waiters_observe_same_result() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = Arc::new(manager(
            gate_factory(gate_rx, running, Arc::new(AtomicUsize::new(0)), false, None),
            1,
            CancellationToken::new(),
        ));
        let id = manager.spawn(spec("shared")).unwrap();

        let mut waiters = Vec::new();
        for _ in 0..4 {
            let manager = Arc::clone(&manager);
            let id = id.clone();
            waiters.push(tokio::spawn(async move { manager.wait(&id).await }));
        }
        tokio::task::yield_now().await;
        gate.send(true).unwrap();

        for waiter in waiters {
            let result = timeout(WAIT_TIMEOUT, waiter)
                .await
                .expect("waiter timed out")
                .expect("waiter task panicked")
                .expect("waiter failed");
            assert_eq!(result.output, "done:shared");
        }
        // Late waiter after completion gets the stored result immediately.
        let result = manager.wait(&id).await.expect("late wait failed");
        assert_eq!(result.id, id);
        let output = manager.output(&id).unwrap().expect("missing output");
        assert_eq!(output.output, "done:shared");
    }

    #[tokio::test]
    async fn cancel_queued_child_never_occupies_permit() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let first = manager.spawn(spec("first")).unwrap();
        until_running(&running, 1).await;
        let second = manager.spawn(spec("second")).unwrap();
        manager.cancel(&second).unwrap();
        assert!(matches!(
            manager.wait(&second).await,
            Err(SubagentError::Cancelled)
        ));
        assert_eq!(running.load(Ordering::SeqCst), 1, "queued child ran");

        gate.send(true).unwrap();
        assert!(manager.wait(&first).await.is_ok());
    }

    #[tokio::test]
    async fn cancel_running_child_reports_cancelled() {
        let (_gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("running")).unwrap();
        until_running(&running, 1).await;
        manager.cancel(&id).unwrap();
        assert!(matches!(
            timeout(WAIT_TIMEOUT, manager.wait(&id))
                .await
                .expect("wait timed out"),
            Err(SubagentError::Cancelled)
        ));
        assert!(manager.output(&id).unwrap().is_none());
    }

    #[tokio::test]
    async fn send_steers_live_children_and_rejects_cancelled() {
        // The observable contract: send() succeeds for queued and running
        // children and is a typed error once the child is cancelled.
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let held = manager.spawn(spec("hold")).unwrap();
        until_running(&running, 1).await;
        let queued = manager.spawn(spec("queued")).unwrap();
        manager
            .send(&queued, "steer me".into())
            .expect("send to queued child failed");
        manager
            .send(&held, "steer running".into())
            .expect("send to running child failed");

        manager.cancel(&queued).unwrap();
        assert!(matches!(
            manager.wait(&queued).await,
            Err(SubagentError::Cancelled)
        ));
        assert!(matches!(
            manager.send(&queued, "too late".into()),
            Err(SubagentError::Terminal(SubagentStatus::Cancelled))
        ));
        gate.send(true).unwrap();
        assert!(manager.wait(&held).await.is_ok());
    }

    #[tokio::test]
    async fn send_to_completed_child_starts_a_turn() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let manager = manager(
            follow_up_factory(first_rx, second_rx, Arc::new(AtomicUsize::new(0))),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("thread")).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");

        manager
            .send(&id, "more".to_string())
            .expect("completed child accepts a message as a new turn");
        assert_eq!(manager.status(&id), Some(SubagentStatus::Running));
        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("follow-up turn");
        assert_eq!(result.output, "again:more");
    }

    #[tokio::test]
    async fn output_schema_enforced_on_result() {
        let (gate, gate_rx) = gate();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        });
        let manager1 = manager(
            gate_factory(
                gate_rx.clone(),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                Some(serde_json::json!({"count": 3})),
            ),
            1,
            CancellationToken::new(),
        );
        let mut valid = spec("structured");
        valid.output_schema = Some(schema.clone());
        let id = manager1.spawn(valid).unwrap();
        gate.send(true).unwrap();
        let result = manager1.wait(&id).await.expect("valid data rejected");
        assert_eq!(result.data, Some(serde_json::json!({"count": 3})));

        // Invalid data fails the child even though the factory returned Ok.
        let manager2 = manager(
            gate_factory(
                gate_rx.clone(),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                Some(serde_json::json!({"count": "three"})),
            ),
            1,
            CancellationToken::new(),
        );
        let mut invalid = spec("bad data");
        invalid.output_schema = Some(schema.clone());
        let id2 = manager2.spawn(invalid).unwrap();
        assert!(matches!(
            manager2.wait(&id2).await,
            Err(SubagentError::Failed(err)) if err.contains("schema")
        ));

        // Missing data also fails.
        let manager3 = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let mut missing = spec("no data");
        missing.output_schema = Some(schema);
        let id3 = manager3.spawn(missing).unwrap();
        assert!(matches!(
            manager3.wait(&id3).await,
            Err(SubagentError::Failed(_))
        ));

        // Malformed schema rejected at spawn.
        let mut bad_spec = spec("bad schema");
        bad_spec.output_schema = Some(serde_json::json!({"type": "nonsense"}));
        assert!(matches!(
            manager3.spawn(bad_spec),
            Err(SubagentError::InvalidSpec(_))
        ));
    }

    #[tokio::test]
    async fn permissive_output_schema_preserves_invalid_structured_data() {
        let (gate, gate_rx) = gate();
        let manager = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                Some(serde_json::json!({"count": "three"})),
            ),
            1,
            CancellationToken::new(),
        );
        let mut task = spec("permissive data");
        task.output_schema = Some(serde_json::json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        }));
        task.schema_mode = SchemaMode::Permissive;
        let id = manager.spawn(task).expect("spawn child");
        gate.send(true).expect("release child");

        let result = manager.wait(&id).await.expect("permissive result rejected");
        assert_eq!(result.data, Some(serde_json::json!({"count": "three"})));
        assert_eq!(manager.status(&id), Some(SubagentStatus::Completed));
    }

    #[tokio::test]
    async fn permissive_output_schema_allows_missing_structured_data() {
        let (gate, gate_rx) = gate();
        let manager = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let mut task = spec("permissive missing data");
        task.output_schema = Some(serde_json::json!({"type": "object"}));
        task.schema_mode = SchemaMode::Permissive;
        let id = manager.spawn(task).expect("spawn child");
        gate.send(true).expect("release child");

        let result = manager.wait(&id).await.expect("missing data rejected");
        assert!(result.data.is_none());
        assert_eq!(manager.status(&id), Some(SubagentStatus::Completed));
    }

    #[tokio::test]
    async fn factory_failure_is_terminal_and_waitable() {
        let (gate, gate_rx) = gate();
        let manager = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                true,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("doomed")).unwrap();
        gate.send(true).unwrap();
        match manager.wait(&id).await {
            Err(SubagentError::Failed(err)) => assert!(err.contains("boom")),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(manager.status(&id), Some(SubagentStatus::Failed));
        assert!(manager.output(&id).unwrap().is_none());
        let summary = manager.list().into_iter().find(|s| s.id == id).unwrap();
        assert_eq!(summary.status, SubagentStatus::Failed);
        assert_eq!(summary.error.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn owner_cancel_cancels_queued_and_running() {
        let (_gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let owner = CancellationToken::new();
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            owner.clone(),
        );
        let first = manager.spawn(spec("running")).unwrap();
        let second = manager.spawn(spec("queued")).unwrap();
        until_running(&running, 1).await;
        owner.cancel();
        assert!(matches!(
            timeout(WAIT_TIMEOUT, manager.wait(&first))
                .await
                .expect("wait timed out"),
            Err(SubagentError::Cancelled)
        ));
        assert!(matches!(
            timeout(WAIT_TIMEOUT, manager.wait(&second))
                .await
                .expect("wait timed out"),
            Err(SubagentError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn owner_cancel_marks_every_open_child() {
        let (_gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let owner = CancellationToken::new();
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            owner.clone(),
        );
        let ids: Vec<String> = (0..3)
            .map(|n| manager.spawn(spec(&format!("c{n}"))).unwrap())
            .collect();
        until_running(&running, 1).await;
        owner.cancel();
        for id in ids {
            assert!(matches!(
                manager.wait(&id).await,
                Err(SubagentError::Cancelled)
            ));
        }
    }

    #[tokio::test]
    async fn unknown_ids_error() {
        let (_gate, gate_rx) = gate();
        let manager = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        assert!(matches!(
            manager.wait("nope").await,
            Err(SubagentError::Unknown(_))
        ));
        assert!(matches!(
            manager.cancel("nope"),
            Err(SubagentError::Unknown(_))
        ));
        assert!(matches!(
            manager.send("nope", "x".into()),
            Err(SubagentError::Unknown(_))
        ));
        assert!(matches!(
            manager.output("nope"),
            Err(SubagentError::Unknown(_))
        ));
        assert_eq!(manager.status("nope"), None);
    }

    #[test]
    fn task_spec_round_trips_and_defaults() {
        let parsed: TaskSpec = serde_json::from_str(r#"{"task":"do it"}"#).unwrap();
        assert_eq!(parsed.task, "do it");
        assert_eq!(parsed.schema_mode, SchemaMode::Strict);
        assert!(parsed.name.is_none() && parsed.model.is_none());
        assert!(parsed.output_schema.is_none() && !parsed.read_only);
        assert!(!parsed.isolated_workspace && parsed.solution_space.is_none());

        let mut full = spec("x");
        full.name = Some("named".into());
        full.read_only = true;
        full.isolated_workspace = true;
        full.solution_space = Some("exact fix".into());
        full.output_schema = Some(serde_json::json!({"type": "string"}));
        full.schema_mode = SchemaMode::Permissive;
        let round: TaskSpec = serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
        assert!(round.read_only);
        assert!(round.isolated_workspace);
        assert_eq!(round.solution_space.as_deref(), Some("exact fix"));
        assert_eq!(round.name.as_deref(), Some("named"));
        assert!(round.output_schema.is_some());
        assert_eq!(round.schema_mode, SchemaMode::Permissive);
    }

    #[test]
    fn subagent_status_predicates_and_serde() {
        let queued = SubagentStatus::Queued;
        assert!(!queued.is_terminal());
        assert!(!queued.is_blocked());
        assert_eq!(serde_json::to_string(&queued).unwrap(), "\"queued\"");
        assert_eq!(
            serde_json::from_str::<SubagentStatus>("\"queued\"").unwrap(),
            SubagentStatus::Queued
        );

        let running = SubagentStatus::Running;
        assert!(!running.is_terminal());
        assert!(!running.is_blocked());
        assert_eq!(serde_json::to_string(&running).unwrap(), "\"running\"");

        let verifying = SubagentStatus::Verifying;
        assert!(!verifying.is_terminal());
        assert!(!verifying.is_blocked());
        assert_eq!(serde_json::to_string(&verifying).unwrap(), "\"verifying\"");
        assert_eq!(
            serde_json::from_str::<SubagentStatus>("\"verifying\"").unwrap(),
            SubagentStatus::Verifying
        );

        let reviewing = SubagentStatus::Reviewing;
        assert!(!reviewing.is_terminal());
        assert!(!reviewing.is_blocked());
        assert_eq!(serde_json::to_string(&reviewing).unwrap(), "\"reviewing\"");
        assert_eq!(
            serde_json::from_str::<SubagentStatus>("\"reviewing\"").unwrap(),
            SubagentStatus::Reviewing
        );

        let blocked = SubagentStatus::Blocked {
            reason: BlockerReason::UserInput,
            prompt: "choose A or B".into(),
        };
        assert!(!blocked.is_terminal());
        assert!(blocked.is_blocked());
        let json = serde_json::to_string(&blocked).unwrap();
        assert_eq!(
            serde_json::from_str::<SubagentStatus>(&json).unwrap(),
            blocked
        );

        let custom_blocked = SubagentStatus::Blocked {
            reason: BlockerReason::Custom("custom_reason".into()),
            prompt: "escalate".into(),
        };
        let custom_json = serde_json::to_string(&custom_blocked).unwrap();
        assert_eq!(
            serde_json::from_str::<SubagentStatus>(&custom_json).unwrap(),
            custom_blocked
        );

        assert!(SubagentStatus::Completed.is_terminal());
        assert!(SubagentStatus::Failed.is_terminal());
        assert!(SubagentStatus::Cancelled.is_terminal());
    }

    struct BlockingFactory {
        manager_holder: Arc<Mutex<Option<Arc<SubagentManager>>>>,
        reason: BlockerReason,
        prompt: String,
    }

    #[async_trait]
    impl SubagentFactory for BlockingFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let mgr = self
                .manager_holder
                .lock()
                .unwrap()
                .as_ref()
                .cloned()
                .expect("manager not registered");
            let resolution = mgr
                .block_and_wait(id, self.reason.clone(), &self.prompt)
                .await
                .map_err(|e| e.to_string())?;
            Ok(SubagentResult {
                id: id.to_string(),
                output: format!("{}:{resolution}", spec.task),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: PathBuf::from("/tmp/child.jsonl"),
            })
        }
    }

    #[tokio::test]
    async fn blocker_escalation_and_unblock_resumes_child() {
        let holder = Arc::new(Mutex::new(None));
        let factory = Arc::new(BlockingFactory {
            manager_holder: holder.clone(),
            reason: BlockerReason::UserInput,
            prompt: "Which database?".to_string(),
        });
        let manager = Arc::new(manager(factory, 1, CancellationToken::new()));
        *holder.lock().unwrap() = Some(manager.clone());

        let id = manager.spawn(spec("setup-db")).unwrap();
        let mut status_rx = manager.subscribe_status(&id).unwrap();

        status_rx
            .wait_for(|s| matches!(s, SubagentStatus::Blocked { .. }))
            .await
            .unwrap();

        assert_eq!(
            manager.status(&id),
            Some(SubagentStatus::Blocked {
                reason: BlockerReason::UserInput,
                prompt: "Which database?".to_string(),
            })
        );

        let list = manager.list();
        assert_eq!(list.len(), 1);
        assert_eq!(
            list[0].status,
            SubagentStatus::Blocked {
                reason: BlockerReason::UserInput,
                prompt: "Which database?".to_string(),
            }
        );

        manager.unblock(&id, "postgres").unwrap();

        let result = timeout(WAIT_TIMEOUT, manager.wait(&id))
            .await
            .expect("wait timed out")
            .expect("wait failed");
        assert_eq!(result.output, "setup-db:postgres");
        assert_eq!(manager.status(&id), Some(SubagentStatus::Completed));
    }

    #[tokio::test]
    async fn unblock_on_non_blocked_fails() {
        let (gate, _gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate.subscribe(),
                running,
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("task")).unwrap();
        let err = manager.unblock(&id, "res").unwrap_err();
        assert!(matches!(err, SubagentError::NotBlocked(_)));
    }

    #[tokio::test]
    async fn block_on_already_blocked_fails() {
        let holder = Arc::new(Mutex::new(None));
        let factory = Arc::new(BlockingFactory {
            manager_holder: holder.clone(),
            reason: BlockerReason::Credential,
            prompt: "API key needed".to_string(),
        });
        let manager = Arc::new(manager(factory, 1, CancellationToken::new()));
        *holder.lock().unwrap() = Some(manager.clone());

        let id = manager.spawn(spec("auth")).unwrap();
        let mut status_rx = manager.subscribe_status(&id).unwrap();
        status_rx
            .wait_for(|s| matches!(s, SubagentStatus::Blocked { .. }))
            .await
            .unwrap();

        let second_block = manager.block(&id, BlockerReason::UserInput, "another prompt");
        assert!(matches!(
            second_block,
            Err(SubagentError::AlreadyBlocked(_))
        ));

        manager.unblock(&id, "token123").unwrap();
        let _ = manager.wait(&id).await;
    }

    #[tokio::test]
    async fn block_on_queued_or_terminal_fails() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let first = manager.spawn(spec("first")).unwrap();
        let queued = manager.spawn(spec("queued")).unwrap();

        until_running(&running, 1).await;
        let err = manager
            .block(&queued, BlockerReason::Approval, "approve")
            .unwrap_err();
        assert!(matches!(err, SubagentError::InvalidTransition { .. }));

        gate.send(true).unwrap();
        let _ = manager.wait(&first).await.unwrap();

        let err_terminal = manager
            .block(&first, BlockerReason::Approval, "approve")
            .unwrap_err();
        assert!(matches!(err_terminal, SubagentError::Terminal(_)));
    }

    #[tokio::test]
    async fn cancellation_unblocks_waiting_child() {
        let holder = Arc::new(Mutex::new(None));
        let factory = Arc::new(BlockingFactory {
            manager_holder: holder.clone(),
            reason: BlockerReason::UserInput,
            prompt: "Blocked forever".to_string(),
        });
        let manager = Arc::new(manager(factory, 1, CancellationToken::new()));
        *holder.lock().unwrap() = Some(manager.clone());

        let id = manager.spawn(spec("cancelled-child")).unwrap();
        let mut status_rx = manager.subscribe_status(&id).unwrap();
        status_rx
            .wait_for(|s| matches!(s, SubagentStatus::Blocked { .. }))
            .await
            .unwrap();

        manager.cancel(&id).unwrap();
        let result = manager.wait(&id).await;
        assert!(matches!(result, Err(SubagentError::Cancelled)));
        assert_eq!(manager.status(&id), Some(SubagentStatus::Cancelled));
    }

    #[tokio::test]
    async fn timeout_on_resolution_returns_timeout_error() {
        struct TimeoutFactory {
            manager_holder: Arc<Mutex<Option<Arc<SubagentManager>>>>,
        }

        #[async_trait]
        impl SubagentFactory for TimeoutFactory {
            async fn run(
                &self,
                id: &str,
                _spec: TaskSpec,
                _control: RunControl,
                _events: mpsc::UnboundedSender<AgentEvent>,
            ) -> Result<SubagentResult, String> {
                let mgr = self
                    .manager_holder
                    .lock()
                    .unwrap()
                    .as_ref()
                    .cloned()
                    .unwrap();
                match mgr
                    .block_and_wait_timeout(
                        id,
                        BlockerReason::Environment,
                        "Need env var",
                        TokioDuration::from_millis(30),
                    )
                    .await
                {
                    Ok(_) => Ok(SubagentResult {
                        id: id.to_string(),
                        output: "got-env".to_string(),
                        data: None,
                        usage: Usage::default(),
                        stop_reason: StopReason::Stop,
                        session_path: PathBuf::from("/tmp/child.jsonl"),
                    }),
                    Err(SubagentError::Timeout(msg)) => Err(format!("timeout: {msg}")),
                    Err(e) => Err(e.to_string()),
                }
            }
        }

        let holder = Arc::new(Mutex::new(None));
        let manager = Arc::new(manager(
            Arc::new(TimeoutFactory {
                manager_holder: holder.clone(),
            }),
            1,
            CancellationToken::new(),
        ));
        *holder.lock().unwrap() = Some(manager.clone());

        let id = manager.spawn(spec("timeout-test")).unwrap();
        let res = manager.wait(&id).await;
        assert!(matches!(res, Err(SubagentError::Failed(msg)) if msg.contains("timeout")));
    }

    #[tokio::test]
    async fn state_transitions_verifying_and_reviewing() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("transitions")).unwrap();
        until_running(&running, 1).await;

        let mut status_rx = manager.subscribe_status(&id).unwrap();

        manager.set_verifying(&id).unwrap();
        status_rx
            .wait_for(|s| *s == SubagentStatus::Verifying)
            .await
            .unwrap();
        assert_eq!(manager.status(&id), Some(SubagentStatus::Verifying));

        manager.set_reviewing(&id).unwrap();
        status_rx
            .wait_for(|s| *s == SubagentStatus::Reviewing)
            .await
            .unwrap();
        assert_eq!(manager.status(&id), Some(SubagentStatus::Reviewing));

        manager.set_running(&id).unwrap();
        status_rx
            .wait_for(|s| *s == SubagentStatus::Running)
            .await
            .unwrap();
        assert_eq!(manager.status(&id), Some(SubagentStatus::Running));

        gate.send(true).unwrap();
        let res = manager.wait(&id).await.unwrap();
        assert_eq!(res.output, "done:transitions");
    }

    #[tokio::test]
    async fn wait_survives_non_terminal_transitions_until_completion() {
        let holder = Arc::new(Mutex::new(None));
        let factory = Arc::new(BlockingFactory {
            manager_holder: holder.clone(),
            reason: BlockerReason::Approval,
            prompt: "Need file delete approval".to_string(),
        });
        let manager = Arc::new(manager(factory, 1, CancellationToken::new()));
        *holder.lock().unwrap() = Some(manager.clone());

        let id = manager.spawn(spec("delete-files")).unwrap();

        let mgr = manager.clone();
        let child_id = id.clone();
        let waiter = tokio::spawn(async move { mgr.wait(&child_id).await });

        let mut status_rx = manager.subscribe_status(&id).unwrap();
        status_rx
            .wait_for(|s| matches!(s, SubagentStatus::Blocked { .. }))
            .await
            .unwrap();

        manager.unblock(&id, "approved").unwrap();

        let result = timeout(WAIT_TIMEOUT, waiter)
            .await
            .expect("waiter timed out")
            .expect("waiter task panicked")
            .expect("waiter failed");
        assert_eq!(result.output, "delete-files:approved");
    }
    #[tokio::test]
    async fn transition_rejects_blocked_and_queued() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("transition-guard")).unwrap();
        until_running(&running, 1).await;

        let err_blocked = manager.transition(
            &id,
            SubagentStatus::Blocked {
                reason: BlockerReason::UserInput,
                prompt: "bypass".into(),
            },
        );
        assert!(matches!(
            err_blocked,
            Err(SubagentError::InvalidTransition { .. })
        ));

        let err_queued = manager.transition(&id, SubagentStatus::Queued);
        assert!(matches!(
            err_queued,
            Err(SubagentError::InvalidTransition { .. })
        ));

        gate.send(true).unwrap();
        let _ = manager.wait(&id).await.unwrap();
    }

    #[tokio::test]
    async fn credential_unblock_skips_steering_plaintext() {
        struct SecretSteeringInspector {
            manager_holder: Arc<Mutex<Option<Arc<SubagentManager>>>>,
        }

        #[async_trait]
        impl SubagentFactory for SecretSteeringInspector {
            async fn run(
                &self,
                id: &str,
                _spec: TaskSpec,
                _control: RunControl,
                _events: mpsc::UnboundedSender<AgentEvent>,
            ) -> Result<SubagentResult, String> {
                let mgr = self
                    .manager_holder
                    .lock()
                    .unwrap()
                    .as_ref()
                    .cloned()
                    .unwrap();
                let secret = mgr
                    .block_and_wait(id, BlockerReason::Credential, "API token needed")
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(SubagentResult {
                    id: id.to_string(),
                    output: format!("authed-with-len:{}", secret.len()),
                    data: None,
                    usage: Usage::default(),
                    session_path: PathBuf::from("/tmp/test"),
                    stop_reason: StopReason::Stop,
                })
            }
        }

        let holder = Arc::new(Mutex::new(None));
        let factory = Arc::new(SecretSteeringInspector {
            manager_holder: holder.clone(),
        });
        let manager = Arc::new(manager(factory, 1, CancellationToken::new()));
        *holder.lock().unwrap() = Some(manager.clone());

        let id = manager.spawn(spec("secret-subagent")).unwrap();
        let mut status_rx = manager.subscribe_status(&id).unwrap();
        status_rx
            .wait_for(|s| matches!(s, SubagentStatus::Blocked { .. }))
            .await
            .unwrap();

        manager.unblock(&id, "sk-secret-token-12345").unwrap();
        let res = manager.wait(&id).await.unwrap();
        assert_eq!(res.output, "authed-with-len:21");
    }

    /// Factory that records whether the spawned spec requested isolation and
    /// reports a per-child workspace path, like the runtime's ChildFactory.
    struct WorkspaceFactory {
        base: PathBuf,
        isolated: Arc<parking_lot::Mutex<Vec<bool>>>,
    }

    #[async_trait]
    impl SubagentFactory for WorkspaceFactory {
        async fn run(
            &self,
            id: &str,
            _spec: TaskSpec,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            Ok(SubagentResult {
                id: id.to_string(),
                output: "done".to_string(),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: PathBuf::from("/tmp/child.jsonl"),
            })
        }

        fn workspace_path(&self, id: &str, spec: &TaskSpec) -> Option<PathBuf> {
            self.isolated.lock().push(spec.isolated_workspace);
            spec.isolated_workspace.then(|| self.base.join(id))
        }
    }

    #[tokio::test]
    async fn focus_spawn_forces_isolation_and_reports_workspace() {
        let base = PathBuf::from("/data");
        let isolated = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let factory = Arc::new(WorkspaceFactory {
            base: base.clone(),
            isolated: Arc::clone(&isolated),
        });
        let manager = manager(factory, 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
        assert!(summary.focus);
        assert_eq!(summary.workspace, Some(base.join(&id)));

        // The factory must observe the forced isolation.
        manager.wait(&id).await.unwrap();
        assert_eq!(*isolated.lock(), vec![true]);

        // A non-focus spec keeps its declared flags and reports no workspace.
        let id = manager.spawn(spec("plain task")).unwrap();
        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
        assert!(!summary.focus);
        assert_eq!(summary.workspace, None);
        manager.wait(&id).await.unwrap();
        assert_eq!(*isolated.lock(), vec![true, false]);
    }

    #[tokio::test]
    async fn non_focus_spawn_uses_default_workspace_path() {
        // A factory that leaves `workspace_path` at its default yields no
        // workspace in the summary even for isolated specs.
        let (gate, gate_rx) = gate();
        let _ = gate.send(true);
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(gate_rx, running, Arc::new(AtomicUsize::new(0)), false, None),
            1,
            CancellationToken::new(),
        );

        let mut isolated_spec = spec("isolated task");
        isolated_spec.isolated_workspace = true;
        let id = manager.spawn(isolated_spec).unwrap();
        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
        assert!(!summary.focus);
        assert_eq!(summary.workspace, None);
        manager.wait(&id).await.unwrap();
    }

    /// Factory that also answers follow-up turns: `run` blocks on
    /// `first_gate`, `follow_up` on `second_gate`. Call counts make session
    /// and workspace reuse observable.
    struct FollowUpFactory {
        first_gate: watch::Receiver<bool>,
        second_gate: watch::Receiver<bool>,
        workspace_calls: Arc<AtomicUsize>,
        workspace: PathBuf,
    }

    fn turn_result(id: &str, output: String) -> SubagentResult {
        SubagentResult {
            id: id.to_string(),
            output,
            data: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            session_path: PathBuf::from("/tmp/child.jsonl"),
        }
    }

    #[async_trait]
    impl SubagentFactory for FollowUpFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let mut gate = self.first_gate.clone();
            let cancel = control.token();
            tokio::select! {
                _ = cancel.cancelled() => Err("cancelled".to_string()),
                opened = gate.wait_for(|open| *open) => match opened {
                    Ok(_) => Ok(turn_result(id, format!("done:{}", spec.task))),
                    Err(_) => Err("gate closed".to_string()),
                },
            }
        }

        async fn follow_up(
            &self,
            id: &str,
            _spec: &TaskSpec,
            session_path: &Path,
            prompt: String,
            control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let mut gate = self.second_gate.clone();
            let cancel = control.token();
            tokio::select! {
                _ = cancel.cancelled() => Err("cancelled".to_string()),
                opened = gate.wait_for(|open| *open) => match opened {
                    Ok(_) => {
                        assert_eq!(session_path, Path::new("/tmp/child.jsonl"));
                        Ok(turn_result(id, format!("again:{prompt}")))
                    }
                    Err(_) => Err("gate closed".to_string()),
                },
            }
        }

        fn workspace_path(&self, id: &str, spec: &TaskSpec) -> Option<PathBuf> {
            self.workspace_calls.fetch_add(1, Ordering::SeqCst);
            spec.isolated_workspace.then(|| self.workspace.join(id))
        }
    }

    fn follow_up_factory(
        first_gate: watch::Receiver<bool>,
        second_gate: watch::Receiver<bool>,
        workspace_calls: Arc<AtomicUsize>,
    ) -> Arc<FollowUpFactory> {
        Arc::new(FollowUpFactory {
            first_gate,
            second_gate,
            workspace_calls,
            workspace: PathBuf::from("/data"),
        })
    }

    #[tokio::test]
    async fn follow_up_completed_child_runs_a_new_turn() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let workspace_calls = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            follow_up_factory(first_rx, second_rx, workspace_calls.clone()),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("thread")).unwrap();
        first.send(true).unwrap();
        let result = manager.wait(&id).await.expect("first turn");
        assert_eq!(result.output, "done:thread");

        manager
            .follow_up(&id, "more".to_string())
            .expect("accepted");
        assert_eq!(manager.status(&id), Some(SubagentStatus::Running));
        // While the new turn runs, completed-only readers see nothing new.
        assert!(manager.output(&id).unwrap().is_none());
        // A second follow-up cannot start while one is in flight.
        assert!(matches!(
            manager.follow_up(&id, "again".into()),
            Err(SubagentError::InvalidTransition { .. })
        ));

        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("follow-up turn");
        assert_eq!(result.output, "again:more");
        assert_eq!(manager.status(&id), Some(SubagentStatus::Completed));
        assert_eq!(manager.output(&id).unwrap().unwrap().output, "again:more");
        // The workspace lookup ran once at spawn and never again.
        assert_eq!(workspace_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn user_follow_up_skips_main_notice_and_main_follow_up_pushes_one() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let manager = manager(
            follow_up_factory(first_rx, second_rx, Arc::new(AtomicUsize::new(0))),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("thread")).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: id.clone(),
                kind: MainMessageKind::Completed,
                body: "done:thread".into(),
            }]
        );

        // The user's follow-up runs the turn but leaves main's inbox empty.
        manager
            .follow_up_from_user(&id, "more".to_string())
            .expect("accepted");
        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("user turn");
        assert!(result.output.ends_with("\n\nmore"));
        assert!(manager.main_inbox().drain().is_empty());

        // Main's own follow-up reports to main again.
        manager
            .follow_up(&id, "report".to_string())
            .expect("accepted");
        manager.wait(&id).await.expect("main turn");
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: id,
                kind: MainMessageKind::Completed,
                body: "again:report".into(),
            }]
        );
    }

    #[test]
    fn requeue_front_keeps_arrival_order() {
        let inbox = MainInbox::default();
        let note = |body: &str| MainMessage {
            from: "child".into(),
            kind: MainMessageKind::Message,
            body: body.into(),
        };
        inbox.push(note("third"));
        inbox.requeue_front(vec![note("first"), note("second")]);
        let bodies: Vec<String> = inbox.drain().into_iter().map(|m| m.body).collect();
        assert_eq!(bodies, ["first", "second", "third"]);
    }

    #[tokio::test]
    async fn events_report_spawn_status_and_finish() {
        let (first, first_rx) = gate();
        let manager = manager(
            gate_factory(
                first_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let mut events = manager.subscribe_events();
        let id = manager.spawn(spec("thread")).unwrap();
        match timeout(WAIT_TIMEOUT, events.recv())
            .await
            .expect("spawn event")
            .expect("channel open")
        {
            SubagentEvent::Spawned { id: got, .. } => assert_eq!(got, id),
            other => panic!("expected spawn first, got {other:?}"),
        }

        first.send(true).unwrap();
        manager.wait(&id).await.expect("turn");
        // Status changes may interleave; Finished is this child's terminal event.
        loop {
            match timeout(WAIT_TIMEOUT, events.recv())
                .await
                .expect("finish event")
                .expect("channel open")
            {
                SubagentEvent::Finished {
                    id: got, status, ..
                } => {
                    assert_eq!(got, id);
                    assert_eq!(status, SubagentStatus::Completed);
                    break;
                }
                SubagentEvent::StatusChanged { .. } => {}
                other => panic!("unexpected event {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn follow_up_rejects_non_completed_states() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let make_manager = manager;
        let manager = manager(
            follow_up_factory(first_rx, second_rx, Arc::new(AtomicUsize::new(0))),
            1,
            CancellationToken::new(),
        );

        let queued_a = manager.spawn(spec("a")).unwrap();
        let queued_b = manager.spawn(spec("b")).unwrap();
        // A queued child cannot take a follow-up.
        assert!(matches!(
            manager.follow_up(&queued_b, "x".into()),
            Err(SubagentError::InvalidTransition { .. })
        ));

        // Cancel while still queued so it never takes the only slot.
        manager.cancel(&queued_b).unwrap();
        first.send(true).unwrap();
        manager.wait(&queued_a).await.expect("first turn");

        // A failed child reports its terminal state explicitly.
        let (fail_gate, fail_rx) = gate();
        let failing = make_manager(
            gate_factory(
                fail_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                true,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let failed_id = failing.spawn(spec("doomed")).unwrap();
        fail_gate.send(true).unwrap();
        assert!(matches!(
            failing.wait(&failed_id).await,
            Err(SubagentError::Failed(_))
        ));
        assert!(matches!(
            failing.follow_up(&failed_id, "x".into()),
            Err(SubagentError::Terminal(SubagentStatus::Failed))
        ));

        // A cancelled queued child reports its terminal state explicitly.
        assert!(matches!(
            manager.wait(&queued_b).await,
            Err(SubagentError::Cancelled)
        ));
        assert!(matches!(
            manager.follow_up(&queued_b, "x".into()),
            Err(SubagentError::Terminal(SubagentStatus::Cancelled))
        ));

        // Unknown ids keep the existing not-found error.
        assert!(matches!(
            manager.follow_up("nope", "x".into()),
            Err(SubagentError::Unknown(_))
        ));
        second.send(true).unwrap();
    }

    #[tokio::test]
    async fn follow_up_rejected_while_child_is_running() {
        let (_first, first_rx) = gate();
        let (_second, second_rx) = gate();
        let manager = manager(
            follow_up_factory(first_rx, second_rx, Arc::new(AtomicUsize::new(0))),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("busy")).unwrap();
        let mut status_rx = manager.subscribe_status(&id).unwrap();
        status_rx
            .wait_for(|status| *status == SubagentStatus::Running)
            .await
            .unwrap();
        assert!(matches!(
            manager.follow_up(&id, "too early".into()),
            Err(SubagentError::InvalidTransition { .. })
        ));
        manager.cancel(&id).unwrap();
        let _ = manager.wait(&id).await;
    }

    #[tokio::test]
    async fn follow_up_runs_again_on_a_focus_child() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let workspace_calls = Arc::new(AtomicUsize::new(0));
        let factory = follow_up_factory(first_rx, second_rx, workspace_calls.clone());
        let manager = manager(factory, 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");

        manager
            .follow_up(&id, "next".to_string())
            .expect("accepted");
        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("follow-up turn");
        assert_eq!(result.output, "again:next");

        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
        assert!(summary.focus);
        assert_eq!(summary.workspace, Some(PathBuf::from("/data").join(&id)));
        assert_eq!(workspace_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn follow_up_failure_marks_the_turn_failed() {
        struct FailFollowUp {
            first_gate: watch::Receiver<bool>,
        }

        #[async_trait]
        impl SubagentFactory for FailFollowUp {
            async fn run(
                &self,
                id: &str,
                spec: TaskSpec,
                _control: RunControl,
                _events: mpsc::UnboundedSender<AgentEvent>,
            ) -> Result<SubagentResult, String> {
                let mut gate = self.first_gate.clone();
                let _ = gate.wait_for(|open| *open).await;
                Ok(turn_result(id, format!("done:{}", spec.task)))
            }

            async fn follow_up(
                &self,
                _id: &str,
                _spec: &TaskSpec,
                _session_path: &Path,
                _prompt: String,
                _control: RunControl,
                _events: mpsc::UnboundedSender<AgentEvent>,
            ) -> Result<SubagentResult, String> {
                Err("follow-up exploded".into())
            }
        }

        let (first, first_rx) = gate();
        let manager = manager(
            Arc::new(FailFollowUp {
                first_gate: first_rx,
            }),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("thread")).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");

        manager.follow_up(&id, "go".to_string()).expect("accepted");
        match manager.wait(&id).await {
            Err(SubagentError::Failed(err)) => assert!(err.contains("follow-up exploded")),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(manager.status(&id), Some(SubagentStatus::Failed));
        // The failed turn is terminal again: no further follow-up.
        assert!(matches!(
            manager.follow_up(&id, "again".into()),
            Err(SubagentError::Terminal(SubagentStatus::Failed))
        ));
    }

    #[tokio::test]
    async fn follow_up_uses_a_fresh_run_control() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let manager = manager(
            follow_up_factory(first_rx, second_rx, Arc::new(AtomicUsize::new(0))),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("thread")).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");

        // Cancelling after completion cancels the old control; the next turn
        // must still run under a fresh one.
        manager.cancel(&id).unwrap();
        manager
            .follow_up(&id, "more".to_string())
            .expect("accepted");
        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("follow-up turn");
        assert_eq!(result.output, "again:more");
        assert_eq!(manager.status(&id), Some(SubagentStatus::Completed));
    }

    #[tokio::test]
    async fn follow_up_cancellation_marks_turn_cancelled() {
        let (first, first_rx) = gate();
        let (_second, second_rx) = gate();
        let manager = manager(
            follow_up_factory(first_rx, second_rx, Arc::new(AtomicUsize::new(0))),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("thread")).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");

        manager
            .follow_up(&id, "more".to_string())
            .expect("accepted");
        manager.cancel(&id).unwrap();
        assert!(matches!(
            manager.wait(&id).await,
            Err(SubagentError::Cancelled)
        ));
        assert_eq!(manager.status(&id), Some(SubagentStatus::Cancelled));
    }

    #[test]
    fn extract_json_parses_direct_fenced_and_trailing() {
        assert_eq!(
            extract_json("{\"a\": 1}").unwrap(),
            serde_json::json!({"a": 1})
        );
        assert_eq!(
            extract_json("Here you go:\n```json\n{\"a\": 2}\n```\nDone.").unwrap(),
            serde_json::json!({"a": 2})
        );
        assert_eq!(
            extract_json("Result: {\"a\": 3}").unwrap(),
            serde_json::json!({"a": 3})
        );
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("").is_none());
    }

    #[tokio::test]
    async fn completed_child_pushes_one_completion_notice() {
        let (gate, gate_rx) = gate();
        let manager = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("alpha")).unwrap();
        gate.send(true).unwrap();
        manager.wait(&id).await.expect("wait failed");
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: id,
                kind: MainMessageKind::Completed,
                body: "done:alpha".into(),
            }]
        );
    }

    #[tokio::test]
    async fn failed_child_pushes_one_failure_notice() {
        let (gate, gate_rx) = gate();
        let manager = manager(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                true,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("alpha")).unwrap();
        gate.send(true).unwrap();
        assert!(manager.wait(&id).await.is_err());
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: id,
                kind: MainMessageKind::Failed,
                body: "boom".into(),
            }]
        );
    }

    #[tokio::test]
    async fn cancelled_child_pushes_one_cancellation_notice() {
        let (_gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("alpha")).unwrap();
        until_running(&running, 1).await;
        manager.cancel(&id).unwrap();
        assert!(matches!(
            manager.wait(&id).await,
            Err(SubagentError::Cancelled)
        ));
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: id,
                kind: MainMessageKind::Cancelled,
                body: "cancelled".into(),
            }]
        );
    }

    #[tokio::test]
    async fn running_child_pushes_no_notice_until_terminal() {
        let (gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("alpha")).unwrap();
        until_running(&running, 1).await;
        assert!(manager.main_inbox().drain().is_empty());
        gate.send(true).unwrap();
        manager.wait(&id).await.expect("wait failed");
        assert_eq!(manager.main_inbox().drain().len(), 1);
    }

    #[tokio::test]
    async fn nested_manager_notices_reach_shared_inbox() {
        let inbox = MainInbox::default();
        let (gate, gate_rx) = gate();
        let nested = SubagentManager::new(
            gate_factory(
                gate_rx,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            CancellationToken::new(),
            inbox.clone(),
        );
        let id = nested.spawn(spec("alpha")).unwrap();
        gate.send(true).unwrap();
        nested.wait(&id).await.expect("wait failed");
        assert_eq!(
            inbox.drain(),
            vec![MainMessage {
                from: id,
                kind: MainMessageKind::Completed,
                body: "done:alpha".into(),
            }]
        );
    }

    /// Factory whose completed turns persist a real session journal, so the
    /// transcript accessor has persisted messages to read back.
    struct TranscriptFactory {
        dir: PathBuf,
    }

    #[async_trait]
    impl SubagentFactory for TranscriptFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let mut session =
                Session::create(&self.dir, &self.dir).map_err(|err| err.to_string())?;
            session
                .append(crate::session::EntryKind::Message {
                    message: Message::user(spec.task.clone()),
                })
                .map_err(|err| err.to_string())?;
            session
                .append(crate::session::EntryKind::Message {
                    message: Message::user(format!("reply:{}", spec.task)),
                })
                .map_err(|err| err.to_string())?;
            Ok(SubagentResult {
                id: id.to_string(),
                output: spec.task,
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: session.path().to_path_buf(),
            })
        }
    }

    #[tokio::test]
    async fn completed_focus_child_returns_persisted_transcript() {
        let dir = std::env::temp_dir().join(format!("juto-transcript-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let manager = manager(
            Arc::new(TranscriptFactory { dir: dir.clone() }),
            1,
            CancellationToken::new(),
        );

        let mut focus = spec("focus task");
        focus.focus = true;
        let focus_id = manager.spawn(focus).unwrap();
        manager.wait(&focus_id).await.unwrap();
        let texts: Vec<String> = manager
            .transcript(&focus_id)
            .unwrap()
            .iter()
            .map(Message::text)
            .collect();
        assert_eq!(texts, vec!["focus task", "reply:focus task"]);

        let plain_id = manager.spawn(spec("plain task")).unwrap();
        manager.wait(&plain_id).await.unwrap();
        let plain_texts: Vec<String> = manager
            .transcript(&plain_id)
            .unwrap()
            .iter()
            .map(Message::text)
            .collect();
        assert_eq!(plain_texts, vec!["plain task", "reply:plain task"]);
        assert_eq!(
            manager.transcript("missing").unwrap_err(),
            SubagentError::Unknown("missing".into())
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn non_completed_children_have_no_transcript() {
        let (open, gate) = gate();
        let manager = manager(
            gate_factory(
                gate,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                true,
                None,
            ),
            1,
            CancellationToken::new(),
        );
        open.send(true).unwrap();

        let plain_id = manager.spawn(spec("plain task")).unwrap();
        assert!(manager.wait(&plain_id).await.is_err());
        assert_eq!(
            manager.transcript(&plain_id).unwrap_err(),
            SubagentError::NotFocus(plain_id.clone())
        );

        let mut focus = spec("focus task");
        focus.focus = true;
        let focus_id = manager.spawn(focus).unwrap();
        assert!(manager.wait(&focus_id).await.is_err());
        assert_eq!(
            manager.transcript(&focus_id).unwrap_err(),
            SubagentError::NoTranscript(focus_id.clone())
        );
    }

    fn summary_of(manager: &SubagentManager, id: &str) -> SubagentSummary {
        manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("child is listed")
    }

    /// Focus children run in a real directory under `base` and their
    /// `release_workspace` deletes it. Runs wait on `gate` when given, and
    /// `started` counts runs that have begun.
    struct ReleaseFactory {
        base: PathBuf,
        gate: Option<watch::Receiver<bool>>,
        started: Arc<AtomicUsize>,
        releases: AtomicUsize,
        fail_release: AtomicBool,
    }

    fn release_factory(base: &Path, gate: Option<watch::Receiver<bool>>) -> Arc<ReleaseFactory> {
        Arc::new(ReleaseFactory {
            base: base.to_path_buf(),
            gate,
            started: Arc::new(AtomicUsize::new(0)),
            releases: AtomicUsize::new(0),
            fail_release: AtomicBool::new(false),
        })
    }

    #[async_trait]
    impl SubagentFactory for ReleaseFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            tokio::fs::create_dir_all(self.base.join(id))
                .await
                .map_err(|err| err.to_string())?;
            self.started.fetch_add(1, Ordering::SeqCst);
            if let Some(gate) = &self.gate {
                let mut gate = gate.clone();
                let cancel = control.token();
                tokio::select! {
                    _ = cancel.cancelled() => return Err("cancelled".to_string()),
                    opened = gate.wait_for(|open| *open) => {
                        if opened.is_err() {
                            return Err("gate closed".to_string());
                        }
                    }
                }
            }
            Ok(SubagentResult {
                id: id.to_string(),
                output: format!("done:{}", spec.task),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: PathBuf::from("/tmp/child.jsonl"),
            })
        }

        fn workspace_path(&self, id: &str, spec: &TaskSpec) -> Option<PathBuf> {
            spec.isolated_workspace.then(|| self.base.join(id))
        }

        async fn release_workspace(&self, _id: &str, path: &Path) -> Result<(), String> {
            self.releases.fetch_add(1, Ordering::SeqCst);
            if self.fail_release.load(Ordering::SeqCst) {
                return Err("disk busy".to_string());
            }
            tokio::fs::remove_dir_all(path)
                .await
                .map_err(|err| err.to_string())
        }
    }

    #[tokio::test]
    async fn release_removes_focus_workspace_and_clears_summary() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        manager.wait(&id).await.unwrap();
        let dir = root.path().join(&id);
        assert_eq!(summary_of(&manager, &id).workspace, Some(dir.clone()));

        manager.release(&id).await.unwrap();
        assert!(!dir.exists(), "release removes the focus workspace");
        let summary = summary_of(&manager, &id);
        assert_eq!(summary.workspace, None);
        assert!(summary.released);
        assert_eq!(factory.releases.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn release_rejects_running_child_and_keeps_workspace() {
        let root = tempfile::tempdir().unwrap();
        let (open, gate) = gate();
        let factory = release_factory(root.path(), Some(gate));
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        until_running(&factory.started, 1).await;
        let dir = root.path().join(&id);

        assert_eq!(
            manager.release(&id).await.unwrap_err(),
            SubagentError::NotTerminal(id.clone())
        );
        assert!(dir.exists(), "a running child keeps its workspace");
        assert_eq!(factory.releases.load(Ordering::SeqCst), 0);
        assert_eq!(summary_of(&manager, &id).workspace, Some(dir.clone()));

        open.send(true).unwrap();
        manager.wait(&id).await.unwrap();
        manager.release(&id).await.unwrap();
        assert!(!dir.exists());
    }

    #[tokio::test]
    async fn release_twice_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        manager.wait(&id).await.unwrap();

        manager.release(&id).await.unwrap();
        manager.release(&id).await.unwrap();
        assert_eq!(factory.releases.load(Ordering::SeqCst), 1);
        assert!(summary_of(&manager, &id).released);
    }

    #[tokio::test]
    async fn released_focus_refuses_follow_up_and_transcript() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        manager.wait(&id).await.unwrap();
        manager.release(&id).await.unwrap();

        assert_eq!(
            manager.follow_up(&id, "more".to_string()).unwrap_err(),
            SubagentError::WorkspaceReleased(id.clone())
        );
        assert_eq!(
            manager.transcript(&id).unwrap_err(),
            SubagentError::WorkspaceReleased(id.clone())
        );
    }

    #[tokio::test]
    async fn release_unknown_id_is_unknown() {
        let root = tempfile::tempdir().unwrap();
        let manager = manager(
            release_factory(root.path(), None),
            1,
            CancellationToken::new(),
        );
        assert_eq!(
            manager.release("missing").await.unwrap_err(),
            SubagentError::Unknown("missing".into())
        );
    }

    #[tokio::test]
    async fn failed_release_keeps_workspace_live_and_can_retry() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        factory.fail_release.store(true, Ordering::SeqCst);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.focus = true;
        let id = manager.spawn(focus).unwrap();
        manager.wait(&id).await.unwrap();
        let dir = root.path().join(&id);

        assert!(matches!(
            manager.release(&id).await.unwrap_err(),
            SubagentError::WorkspaceRelease(_)
        ));
        let summary = summary_of(&manager, &id);
        assert_eq!(summary.workspace, Some(dir.clone()));
        assert!(!summary.released);

        factory.fail_release.store(false, Ordering::SeqCst);
        manager.release(&id).await.unwrap();
        assert!(!dir.exists());
        assert!(summary_of(&manager, &id).released);
    }

    #[tokio::test]
    async fn release_without_isolated_workspace_is_a_no_op() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let id = manager.spawn(spec("plain task")).unwrap();
        manager.wait(&id).await.unwrap();
        manager.release(&id).await.unwrap();

        assert_eq!(factory.releases.load(Ordering::SeqCst), 0);
        let summary = summary_of(&manager, &id);
        assert_eq!(summary.workspace, None);
        assert!(!summary.released);
    }
}
