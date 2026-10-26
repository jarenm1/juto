//! Bounded subagent orchestration.
//!
//! [`SubagentManager`] runs child turns: each turn holds a memory desk, and isolated children also hold a review slot.
//! propagates the parent cancellation token to every child [`RunControl`], and keeps
//! terminal results so any number of `wait`/`output` callers observe them.
//! The runtime owns the production factory: child agents inherit its model,
//! instructions and approval policy and write independent durable sessions.
//!
//! Behavioral reference: oh-my-pi `packages/coding-agent/src/task` at
//! 579da1d6; see licenses/OMP-MIT.txt.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::main_inbox_store;
use crate::main_report::{self, DEFAULT_MAIN_REPORT_BYTES};
use crate::pools::{Admission, TurnDesk};
use crate::session::Session;
use crate::tier::ExecutionTier;
use crate::tools::DesktopSessionArtifact;
use async_trait::async_trait;
use futures::FutureExt;
use juto_agent::{AgentEvent, RunControl};
use juto_ai::{Message, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex as AsyncMutex, Notify, broadcast, mpsc, watch};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
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
    /// Session id of the top-level run whose tree spawned this child. Child
    /// reports to `agent://main` return to that session; never model-supplied.
    #[serde(skip)]
    pub origin: Option<String>,
    /// The main thread's session file, which `recall` reads. The runtime sets
    /// it when a `task` call spawns children; the model never does.
    #[serde(skip)]
    pub main_session: Option<PathBuf>,
    /// Immutable main view captured once for all siblings in a task call.
    #[serde(skip)]
    pub main_snapshot: Option<Arc<str>>,
    /// Execution tier for the child. The `task` tool always resolves it against
    /// the parent's tier; `None` (specs built in code) means the fail-closed
    /// default, `workspace`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<ExecutionTier>,
    /// Lets this turn ask the user a question and wait for the answer. The
    /// manager sets it on the spec it hands the factory for each turn; it is
    /// never stored with the child and never model-supplied.
    #[serde(skip)]
    pub ask: Option<UserAsk>,
}

/// A running child's line to the user. [`UserAsk::ask`] marks the child
/// [`SubagentStatus::Blocked`] on [`BlockerReason::UserInput`] with the
/// question as its prompt, and resolves with the user's answer once
/// [`SubagentManager::unblock`] (or a user message) delivers it.
#[derive(Clone)]
pub struct UserAsk {
    inner: Weak<Inner>,
    owner: CancellationToken,
    id: String,
}

impl fmt::Debug for UserAsk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UserAsk").field("id", &self.id).finish()
    }
}

impl UserAsk {
    fn new(inner: &Arc<Inner>, owner: &CancellationToken, id: &str) -> Self {
        Self {
            inner: Arc::downgrade(inner),
            owner: owner.clone(),
            id: id.to_owned(),
        }
    }

    /// Block on the user's answer to `question`. Cancelling the child or its
    /// owner ends the wait with [`SubagentError::Cancelled`].
    pub async fn ask(&self, question: String) -> Result<String, SubagentError> {
        let inner = self
            .inner
            .upgrade()
            .ok_or_else(|| SubagentError::Unknown(self.id.clone()))?;
        let manager = SubagentManager {
            inner,
            owner: self.owner.clone(),
        };
        manager
            .block_and_wait(&self.id, BlockerReason::UserInput, question)
            .await
    }
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
            origin: None,
            main_session: None,
            main_snapshot: None,
            tier: None,
            ask: None,
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
    /// Change-graph node that keeps this run's isolated changes, when it made
    /// any. Joined into the main workspace with `subagent_join`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change_id: Option<String>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// `true` for a focus thread: a subagent the user has messaged (steered,
    /// answered, or followed up). Focus threads get a summary after each turn.
    pub focus: bool,
    /// One or two lines on where a focus thread's work stands, written after
    /// its latest turn. `None` until the first summary, and for non-focus children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// The child's dedicated workspace path, when it runs isolated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// `true` once the main agent released this child's workspace.
    pub released: bool,
    /// `true` once the user archived this finished child: it leaves the task
    /// list, but its workspace and transcript are kept.
    pub archived: bool,
}

/// A delegated child's settled outcome, reported by the run that started it.
#[derive(Debug, Clone)]
pub struct ChildOutcome {
    pub id: String,
    pub name: Option<String>,
    pub status: SubagentStatus,
    /// The child's final text, when it completed.
    pub output: Option<String>,
    /// Why the child failed or was cancelled.
    pub error: Option<String>,
    /// Usage the child reported. Zero for a child that never finished a run.
    pub usage: Usage,
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
    /// The child has no persisted transcript: it has not completed a turn, or it ended as failed/cancelled.
    NoTranscript(String),
    /// Reading the child's session file failed.
    Transcript(String),
    /// `release` or `set_archived` targeted a child that has not reached a terminal state.
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
            Self::NoTranscript(id) => write!(f, "subagent {id} has no completed transcript"),
            Self::Transcript(err) => write!(f, "failed to read subagent transcript: {err}"),
            Self::NotTerminal(id) => {
                write!(f, "subagent {id} is still working; wait until it finishes")
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

    /// Byte budget for one report entering the main thread. Reports over it
    /// are digested or clipped; see `main_report`.
    fn main_report_budget(&self) -> usize {
        DEFAULT_MAIN_REPORT_BYTES
    }

    /// Condense an over-budget report to at most `limit` bytes of text. The
    /// default has no model, so the report is clipped instead.
    async fn digest_report(
        &self,
        _body: &str,
        _limit: usize,
        _cancel: CancellationToken,
    ) -> Result<String, String> {
        Err("digest not supported by this factory".into())
    }

    /// One or two lines on where the child's work stands, read from the
    /// session journal at `session_path` after a completed turn of a focus
    /// thread. The default has no model, so focus threads keep no summary.
    async fn summarize(
        &self,
        _session_path: &Path,
        _cancel: CancellationToken,
    ) -> Result<String, String> {
        Err("summary not supported by this factory".into())
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SubagentEvent {
    /// A bounded report is now readable in the shared main inbox.
    MainInboxReady,
    /// A child was queued.
    Spawned {
        id: String,
        name: Option<String>,
        status: SubagentStatus,
    },
    StatusChanged {
        id: String,
        status: SubagentStatus,
    },
    /// Terminal, exactly once per turn. `result` on success, `error` otherwise.
    Finished {
        id: String,
        status: SubagentStatus,
        result: Option<Box<SubagentResult>>,
        error: Option<String>,
    },
    /// A child's turn posted a desktop artifact that opened successfully.
    /// Observers mount it live; main also gets a `Desktop` inbox notice.
    DesktopReported {
        id: String,
        artifact: DesktopSessionArtifact,
    },
    /// One streaming event of a child turn, in order, as the child produced it.
    /// A receiver that lags misses deltas; it should reload the transcript when
    /// the turn finishes.
    Agent {
        id: String,
        event: AgentEvent,
    },
    /// A finished child was archived or restored. Task lists drop or re-add it.
    ArchiveChanged {
        id: String,
        archived: bool,
    },
    /// A child became a focus thread because the user messaged it.
    Promoted {
        id: String,
    },
    /// A focus thread's summary was rewritten after a turn.
    Summarized {
        id: String,
        summary: String,
    },
}

/// Events buffered per manager before a slow receiver lags.
const EVENT_CAPACITY: usize = 256;

/// Kind of message delivered to the main agent's inbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// A child opened a desktop artifact that main can observe.
    Desktop,
    /// A turn the user started with a child ended. Quiet: it never wakes an
    /// idle main thread, and reaches main with the next report or request.
    UserTurn,
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

    /// Whether this kind may wait for main's next run instead of starting one.
    pub fn is_quiet(self) -> bool {
        matches!(self, Self::UserTurn)
    }
}

/// One entry in the main agent's inbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MainMessage {
    /// Child id, or the sender placeholder used when a peer has no sender id.
    pub from: String,
    /// The child's configured name, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub kind: MainMessageKind,
    /// The body as the main thread receives it: verbatim, or a bounded digest
    /// ending with a pointer to the full text.
    pub body: String,
    /// Byte length of the original report when `body` is a digest or clip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest_of: Option<usize>,
}

/// Main agent's inbox, shared by every manager in a run tree, so nested
/// children write into the same main thread. Each message remembers the
/// session id of the top-level run that spawned its sender (`origin`), so a
/// main session with several internal threads delivers a report to the thread
/// that asked for it.
///
/// Once [`MainInbox::bind`] attaches a file, every change to the queue or to
/// the digest archive is written to it. Reports still waiting in a sender's
/// lane for their digest are not persisted: the digest needs the live model,
/// so a restart before it finishes drops them.
#[derive(Clone, Default)]
pub struct MainInbox {
    messages: Arc<Mutex<Vec<Queued>>>,
    arrived: Arc<Notify>,
    /// Reports from one sender still waiting for their digest, oldest first.
    /// A sender has a lane while any of its reports is pending, so its later
    /// reports queue behind the digests rather than overtake them.
    lanes: Arc<Mutex<HashMap<String, VecDeque<PendingReport>>>>,
    /// Full text of every digested or clipped report, by pointer id.
    archive: Arc<Mutex<HashMap<String, ArchivedReport>>>,
    event_senders: Arc<Mutex<Vec<broadcast::WeakSender<SubagentEvent>>>>,
    /// Backing file, set once by `bind`.
    store: Arc<OnceLock<PathBuf>>,
    /// Tracks execution across every nested manager sharing this run tree.
    tasks: TaskTracker,
    /// Every child spawned anywhere in this run tree, by id, so peers can
    /// address one another and not only their own children. The inbox is the
    /// handle every manager in a tree already shares, so the directory rides on it.
    directory: Arc<Mutex<HashMap<String, PeerEntry>>>,
}

/// Where a child spawned in this tree lives: its manager's registry, the token
/// that owns its turns, and the top-level session its tree reports to.
#[derive(Clone)]
struct PeerEntry {
    inner: Weak<Inner>,
    owner: CancellationToken,
    origin: Option<String>,
}

/// One queued message and the top-level session it is addressed to.
#[derive(Serialize, Deserialize)]
pub(crate) struct Queued {
    pub(crate) origin: Option<String>,
    pub(crate) message: MainMessage,
}

impl MainInbox {
    /// Attach the inbox to the file at `path` and restore what it holds.
    /// Messages already queued in this process stay behind the restored ones.
    /// Binding the same path again is a no-op; binding a different path fails,
    /// because one inbox serves one main session. A file that cannot be read
    /// fails the bind and leaves the inbox unattached.
    pub fn bind(&self, path: PathBuf) -> anyhow::Result<()> {
        let mut messages = self.messages.lock().expect("main inbox poisoned");
        if let Some(bound) = self.store.get() {
            anyhow::ensure!(
                *bound == path,
                "main inbox is already bound to {}",
                bound.display()
            );
            return Ok(());
        }
        let loaded = main_inbox_store::load(&path)?;
        let restored = !loaded.queued.is_empty();
        let mut archive = self.archive.lock().expect("main inbox poisoned");
        archive.extend(loaded.archive);
        messages.splice(0..0, loaded.queued);
        // Checked above under the `messages` lock, so no other bind can have set it.
        let _ = self.store.set(path);
        self.persist_with(&messages, &archive);
        drop(archive);
        drop(messages);
        if restored {
            self.announce();
        }
        Ok(())
    }

    /// Queue a message with no known origin; any top-level run may take it.
    pub fn push(&self, message: MainMessage) {
        self.push_from(None, message);
    }

    /// Queue a message addressed to the top-level session `origin`.
    pub fn push_from(&self, origin: Option<String>, message: MainMessage) {
        {
            let mut messages = self.messages.lock().expect("main inbox poisoned");
            messages.push(Queued { origin, message });
            let archive = self.archive.lock().expect("main inbox poisoned");
            self.persist_with(&messages, &archive);
        }
        self.announce();
    }

    /// Wake waiters and event subscribers: a message is ready to drain.
    fn announce(&self) {
        self.event_senders
            .lock()
            .expect("main inbox poisoned")
            .retain(|weak| {
                if let Some(sender) = weak.upgrade() {
                    let _ = sender.send(SubagentEvent::MainInboxReady);
                    true
                } else {
                    false
                }
            });
        // Stores a permit when no waiter is parked, so a push racing a drain
        // is not missed by the next `arrived` wait.
        self.arrived.notify_one();
    }

    /// Write the queue and the archive to the bound file, if any. Lock order
    /// is `messages`, then `archive`. A failed write is reported on stderr and
    /// the state stays in memory; the next successful write stores all of it
    /// again.
    fn persist(&self, messages: &[Queued]) {
        let archive = self.archive.lock().expect("main inbox poisoned");
        self.persist_with(messages, &archive);
    }

    /// As [`Self::persist`], for callers that already hold `messages` and `archive`.
    fn persist_with(&self, messages: &[Queued], archive: &HashMap<String, ArchivedReport>) {
        let Some(path) = self.store.get() else {
            return;
        };
        if let Err(error) = main_inbox_store::save(path, messages, archive) {
            eprintln!("juto: {error:#}");
        }
    }

    /// Resolves once a message is pushed. Pending messages are read with `drain`.
    pub async fn arrived(&self) {
        self.arrived.notified().await;
    }

    /// Put undelivered messages for `origin` back at the front, preserving
    /// arrival order.
    pub fn requeue_front(&self, origin: Option<&str>, messages: Vec<MainMessage>) {
        if messages.is_empty() {
            return;
        }
        let queued = messages.into_iter().map(|message| Queued {
            origin: origin.map(str::to_owned),
            message,
        });
        let mut guard = self.messages.lock().expect("main inbox poisoned");
        guard.splice(0..0, queued);
        self.persist(&guard);
        drop(guard);
        self.arrived.notify_one();
    }

    /// Remove and return every pending message in arrival order.
    pub fn drain(&self) -> Vec<MainMessage> {
        let mut guard = self.messages.lock().expect("main inbox poisoned");
        if guard.is_empty() {
            return Vec::new();
        }
        let taken = std::mem::take(&mut *guard);
        self.persist(&guard);
        taken.into_iter().map(|queued| queued.message).collect()
    }

    /// Remove and return the messages a run of session `session_id` should
    /// receive: those addressed to it and those with no origin. Messages for
    /// other sessions stay queued in order.
    pub fn drain_for(&self, session_id: &str) -> Vec<MainMessage> {
        let mut guard = self.messages.lock().expect("main inbox poisoned");
        let (taken, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *guard)
            .into_iter()
            .partition(|queued| queued.origin.as_deref().is_none_or(|o| o == session_id));
        *guard = kept;
        if !taken.is_empty() {
            self.persist(&guard);
        }
        taken.into_iter().map(|queued| queued.message).collect()
    }

    /// Remove the oldest origin group that `deliverable` accepts: every pending
    /// message sharing its origin, in arrival order. A group is taken only for a
    /// message that may wake main; quiet messages ride along with it. Groups it
    /// rejects, and the groups behind the taken one, stay queued.
    pub fn take_next_group(
        &self,
        deliverable: impl Fn(Option<&str>) -> bool,
    ) -> Option<(Option<String>, Vec<MainMessage>)> {
        let mut guard = self.messages.lock().expect("main inbox poisoned");
        let origin = guard
            .iter()
            .find(|queued| {
                !queued.message.kind.is_quiet() && deliverable(queued.origin.as_deref())
            })?
            .origin
            .clone();
        let (taken, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *guard)
            .into_iter()
            .partition(|queued| queued.origin == origin);
        *guard = kept;
        self.persist(&guard);
        Some((
            origin,
            taken.into_iter().map(|queued| queued.message).collect(),
        ))
    }

    /// Queue a child report for the main thread. A report within its budget
    /// enters at once unless its sender still has reports pending. Otherwise
    /// it waits in the sender's lane and is bounded in arrival order, so a
    /// sender's reports never overtake one another.
    pub fn submit(
        &self,
        report: MainReport,
        factory: Arc<dyn SubagentFactory>,
        cancel: CancellationToken,
    ) {
        let budget = factory.main_report_budget();
        let mut lanes = self.lanes.lock().expect("main inbox poisoned");
        if !lanes.contains_key(&report.from) && report.body.len() <= budget && report.data.is_none()
        {
            self.push_from(
                report.origin,
                MainMessage {
                    from: report.from,
                    kind: report.kind,
                    name: report.name,
                    digest_of: None,
                    body: report.body,
                },
            );
            return;
        }
        let from = report.from.clone();
        let pending = PendingReport {
            report,
            factory,
            cancel,
            budget,
        };
        match lanes.entry(from.clone()) {
            Entry::Vacant(lane) => {
                lane.insert(VecDeque::from([pending]));
                tokio::spawn(drain_lane(self.clone(), from));
            }
            Entry::Occupied(mut lane) => lane.get_mut().push_back(pending),
        }
    }

    /// Reports whose asynchronous digest must finish before the host can exit.
    pub fn has_pending_reports(&self) -> bool {
        !self.lanes.lock().expect("main inbox poisoned").is_empty()
    }

    /// Full text of a digested or clipped report, as `wait` returns it.
    pub fn archived(&self, id: &str) -> Option<Value> {
        self.archive
            .lock()
            .expect("main inbox poisoned")
            .get(id)
            .map(|report| serde_json::to_value(report).expect("archived report serializes"))
    }
}

/// A child report on its way to the main thread, before it is bounded.
pub struct MainReport {
    /// Topic that spawned the report; retained while asynchronous digestion runs.
    pub origin: Option<String>,
    pub from: String,
    pub name: Option<String>,
    pub kind: MainMessageKind,
    pub body: String,
    /// Structured output of the child, shown in the bounded body when small.
    pub data: Option<Value>,
}

/// A report waiting in its sender's lane for its digest.
struct PendingReport {
    report: MainReport,
    factory: Arc<dyn SubagentFactory>,
    cancel: CancellationToken,
    budget: usize,
}

/// Full text of a digested or clipped report, returned by `wait` under its
/// pointer id.
#[derive(Serialize, Deserialize)]
pub(crate) struct ArchivedReport {
    pub(crate) from: String,
    pub(crate) name: Option<String>,
    pub(crate) kind: MainMessageKind,
    pub(crate) body: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) data: Option<Value>,
}

/// Bounds and delivers one sender's queued reports in arrival order, then
/// retires the lane. The lane lock is never held across an await.
async fn drain_lane(inbox: MainInbox, from: String) {
    loop {
        let popped = {
            let mut lanes = inbox.lanes.lock().expect("main inbox poisoned");
            let popped = lanes.get_mut(&from).and_then(VecDeque::pop_front);
            if popped.is_none() {
                lanes.remove(&from);
            }
            popped
        };
        let Some(next) = popped else { return };
        let origin = next.report.origin.clone();
        let message = bound_pending(&inbox, next).await;
        inbox.push_from(origin, message);
    }
}

async fn bound_pending(inbox: &MainInbox, pending: PendingReport) -> MainMessage {
    let PendingReport {
        report:
            MainReport {
                origin: _,
                from,
                name,
                kind,
                body,
                data,
            },
        factory,
        cancel,
        budget,
    } = pending;
    let key = Uuid::now_v7().to_string();
    let view = main_report::Report {
        body: &body,
        data: data.as_ref(),
        key: &key,
    };
    let text = body.as_str();
    let bounded = main_report::bound(view, budget, main_report::DIGEST_TIMEOUT, move |limit| {
        let factory = factory.clone();
        let cancel = cancel.clone();
        async move {
            tokio::select! {
                _ = cancel.cancelled() => Err("cancelled".into()),
                result = factory.digest_report(text, limit, cancel.clone()) => result,
            }
        }
    })
    .await;
    match bounded {
        None => MainMessage {
            from,
            kind,
            name,
            digest_of: None,
            body,
        },
        Some(bounded) => {
            let digest_of = Some(body.len());
            let archived = ArchivedReport {
                from: from.clone(),
                name: name.clone(),
                kind,
                body,
                data,
            };
            {
                let messages = inbox.messages.lock().expect("main inbox poisoned");
                let mut archive = inbox.archive.lock().expect("main inbox poisoned");
                archive.insert(key, archived);
                inbox.persist_with(&messages, &archive);
            }
            MainMessage {
                from,
                kind,
                name,
                digest_of,
                body: bounded,
            }
        }
    }
}

/// The text a main thread receives for one inbox message. Steering into an active
/// run and waking an idle run both use this framing, so the model reads them alike.
pub fn main_message_text(message: &MainMessage) -> String {
    let name = message
        .name
        .as_deref()
        .map(|name| format!(" name={name:?}"))
        .unwrap_or_default();
    let digest = message
        .digest_of
        .map(|bytes| format!(" (digest of {bytes} bytes)"))
        .unwrap_or_default();
    format!(
        "[subagent {}{name} {:?}{digest}]\n{}",
        message.from, message.kind, message.body
    )
}

/// The prompt that wakes an idle main thread with messages that arrived while no
/// run was active.
pub fn completion_prompt(messages: &[MainMessage]) -> String {
    messages
        .iter()
        .map(main_message_text)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The header a framed main message opens with, as `main_message_text` writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MainMessageHeader {
    pub from: String,
    /// The child's configured name, when `main_message_text` wrote one.
    pub name: Option<String>,
    pub kind: MainMessageKind,
}

/// Parses the first line `main_message_text` writes for a framed main message.
/// `None` for any other line. The digest is display-only, so it is not returned.
pub fn parse_main_message_header(line: &str) -> Option<MainMessageHeader> {
    let inner = line.strip_prefix("[subagent ")?.strip_suffix(']')?;
    let (from, rest) = inner.split_once(' ')?;
    let (name, rest) = match rest.strip_prefix("name=") {
        Some(named) => {
            let (name, after) = parse_debug_string(named)?;
            (Some(name), after.strip_prefix(' ')?)
        }
        None => (None, rest),
    };
    let kind = match rest.split(' ').next()? {
        "Message" => MainMessageKind::Message,
        "Completed" => MainMessageKind::Completed,
        "Failed" => MainMessageKind::Failed,
        "Cancelled" => MainMessageKind::Cancelled,
        "Desktop" => MainMessageKind::Desktop,
        "UserTurn" => MainMessageKind::UserTurn,
        _ => return None,
    };
    Some(MainMessageHeader {
        from: from.to_owned(),
        name,
        kind,
    })
}

/// Decodes the Debug-quoted string that opens `text`. Returns the decoded value
/// and the text after its closing quote.
fn parse_debug_string(text: &str) -> Option<(String, &str)> {
    let body = text.strip_prefix('"')?;
    let mut chars = body.char_indices();
    let mut out = String::new();
    while let Some((at, c)) = chars.next() {
        match c {
            '"' => return Some((out, &body[at + 1..])),
            '\\' => out.push(unescape(&mut chars)?),
            _ => out.push(c),
        }
    }
    None
}

/// Decodes the character after a backslash in a Debug-escaped string.
fn unescape(chars: &mut std::str::CharIndices<'_>) -> Option<char> {
    let (_, escape) = chars.next()?;
    Some(match escape {
        '"' | '\\' | '\'' => escape,
        'n' => '\n',
        'r' => '\r',
        't' => '\t',
        '0' => '\0',
        'u' => {
            let (_, open) = chars.next()?;
            if open != '{' {
                return None;
            }
            let mut hex = String::new();
            loop {
                let (_, digit) = chars.next()?;
                if digit == '}' {
                    break;
                }
                hex.push(digit);
            }
            char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?
        }
        _ => return None,
    })
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
    /// Set once the user messages the child; never cleared.
    focus: AtomicBool,
    /// Latest turn summary; written only while `focus` is set.
    summary: Mutex<Option<String>>,
    /// What the user typed for the follow-up turn in flight, quoted in the
    /// quiet notice main gets when that turn ends.
    user_message: Mutex<Option<String>>,
    /// Hidden from the task list by the user; the child itself is unchanged.
    archived: AtomicBool,
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
    admission: Admission,
    main_inbox: MainInbox,
    /// Top-level session every child of this manager reports to. `None` for
    /// the root manager, whose callers stamp each spec with its own session.
    origin: Option<String>,
    events: broadcast::Sender<SubagentEvent>,
    children: Mutex<HashMap<String, Arc<Child>>>,
}

/// Manager for child subagent runs.
///
/// `spawn` never blocks. Each turn waits for a desk inside the child task, and
/// model requests wait in the provider pool. Focus threads also wait for a review slot.
pub struct SubagentManager {
    inner: Arc<Inner>,
    /// Owns the children this handle spawns: cancelling it stops them.
    owner: CancellationToken,
}

impl SubagentManager {
    pub fn new(
        factory: Arc<dyn SubagentFactory>,
        admission: Admission,
        owner: CancellationToken,
        main_inbox: MainInbox,
    ) -> Self {
        Self::with_origin(factory, admission, owner, main_inbox, None)
    }

    /// A manager whose children all report to the top-level session `origin`,
    /// whatever the spawning tool records. Nested managers use this so a
    /// grandchild's report returns to the thread that started the tree.
    pub fn with_origin(
        factory: Arc<dyn SubagentFactory>,
        admission: Admission,
        owner: CancellationToken,
        main_inbox: MainInbox,
        origin: Option<String>,
    ) -> Self {
        let events = broadcast::channel(EVENT_CAPACITY).0;
        {
            let mut senders = main_inbox
                .event_senders
                .lock()
                .expect("main inbox poisoned");
            senders.retain(|sender| sender.upgrade().is_some());
            senders.push(events.downgrade());
        }
        Self {
            inner: Arc::new(Inner {
                factory,
                admission,
                main_inbox,
                origin,
                children: Mutex::new(HashMap::new()),
                events,
            }),
            owner,
        }
    }

    /// Top-level session this manager's children report to, when fixed.
    pub fn origin(&self) -> Option<&str> {
        self.inner.origin.as_deref()
    }

    /// The inbox every child of this manager pushes into. Shared with nested
    /// managers so the whole run tree reports to one main thread.
    pub fn main_inbox(&self) -> MainInbox {
        self.inner.main_inbox.clone()
    }

    /// Token that owns the children this handle spawns; cancelling it stops
    /// them. The registry, admission, and inbox are shared by every handle.
    pub(crate) fn owner(&self) -> CancellationToken {
        self.owner.clone()
    }

    /// A handle over the same registry, admission, and inbox whose children
    /// are owned by `owner`.
    pub(crate) fn scoped(&self, owner: CancellationToken) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            owner,
        }
    }

    /// Route peer messages through the same ordered, bounded report lane as completion.
    pub fn report_to_main(&self, from: String, body: String) {
        let child = self.child(&from).ok();
        let name = child.as_ref().and_then(|child| child.spec.name.clone());
        let origin = child
            .and_then(|child| child.spec.origin.clone())
            .or_else(|| self.inner.origin.clone());
        self.inner.main_inbox.submit(
            MainReport {
                origin,
                from,
                name,
                kind: MainMessageKind::Message,
                body,
                data: None,
            },
            self.inner.factory.clone(),
            self.owner.clone(),
        );
    }

    /// Subscribe to lifecycle events for every child this manager owns. A
    /// receiver that lags should resynchronize from [`SubagentManager::list`].
    pub fn subscribe_events(&self) -> broadcast::Receiver<SubagentEvent> {
        self.inner.events.subscribe()
    }

    /// Queue a child run. Returns the generated child id immediately; the
    /// desk wait, run, and terminal-state publication are asynchronous.
    pub fn spawn(&self, mut spec: TaskSpec) -> Result<String, SubagentError> {
        if let Some(origin) = &self.inner.origin {
            spec.origin = Some(origin.clone());
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
        let workspace = self.inner.factory.workspace_path(&id, &spec);
        let child = Arc::new(Child {
            id: id.clone(),
            spec,
            focus: AtomicBool::new(false),
            summary: Mutex::new(None),
            user_message: Mutex::new(None),
            archived: AtomicBool::new(false),
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
        {
            let mut directory = self
                .inner
                .main_inbox
                .directory
                .lock()
                .expect("peer directory poisoned");
            directory.retain(|_, entry| entry.inner.strong_count() > 0);
            directory.insert(
                id.clone(),
                PeerEntry {
                    inner: Arc::downgrade(&self.inner),
                    owner: self.owner.clone(),
                    origin: child.spec.origin.clone(),
                },
            );
        }
        let _ = self.inner.events.send(SubagentEvent::Spawned {
            id: id.clone(),
            name: child.spec.name.clone(),
            status: SubagentStatus::Queued,
        });

        let inner = Arc::clone(&self.inner);
        let task_id = id.clone();
        self.inner.main_inbox.tasks.spawn(run_child(
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

    /// The handle that owns `id` among this manager's peers: this manager for
    /// its own children, otherwise any child spawned in the same tree. A nested
    /// manager reaches only children of its own top-level session (`origin`),
    /// so separate trees never address each other.
    fn peer(&self, id: &str) -> Result<SubagentManager, SubagentError> {
        if self.child(id).is_ok() {
            return Ok(self.scoped(self.owner.clone()));
        }
        let entry = self
            .inner
            .main_inbox
            .directory
            .lock()
            .expect("peer directory poisoned")
            .get(id)
            .cloned()
            .filter(|entry| self.inner.origin.is_none() || entry.origin == self.inner.origin)
            .ok_or_else(|| SubagentError::Unknown(id.to_string()))?;
        let inner = entry
            .inner
            .upgrade()
            .ok_or_else(|| SubagentError::Unknown(id.to_string()))?;
        Ok(SubagentManager {
            inner,
            owner: entry.owner,
        })
    }

    /// Ids of every child this manager can address: its own children and the
    /// rest of its tree, as [`SubagentManager::peer`] scopes them. With
    /// `working_only`, finished children are left out.
    pub fn peer_ids(&self, working_only: bool) -> Vec<String> {
        let mut ids: Vec<String> = self.list().into_iter().map(|summary| summary.id).collect();
        let entries: Vec<(String, PeerEntry)> = self
            .inner
            .main_inbox
            .directory
            .lock()
            .expect("peer directory poisoned")
            .iter()
            .map(|(id, entry)| (id.clone(), entry.clone()))
            .collect();
        for (id, entry) in entries {
            let in_scope = self.inner.origin.is_none() || entry.origin == self.inner.origin;
            if in_scope && entry.inner.strong_count() > 0 && !ids.contains(&id) {
                ids.push(id);
            }
        }
        if working_only {
            ids.retain(|id| {
                self.peer_summary(id)
                    .is_ok_and(|summary| !summary.status.is_terminal())
            });
        }
        ids
    }

    /// Message any child this manager can address; see [`SubagentManager::send`].
    pub fn send_peer(&self, id: &str, message: String) -> Result<(), SubagentError> {
        self.peer(id)?.send(id, message)
    }

    /// Status snapshot of any child this manager can address.
    pub fn peer_summary(&self, id: &str) -> Result<SubagentSummary, SubagentError> {
        self.peer(id)?
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .ok_or_else(|| SubagentError::Unknown(id.to_string()))
    }

    /// Completed result of any child this manager can address.
    pub fn peer_output(&self, id: &str) -> Result<Option<SubagentResult>, SubagentError> {
        self.peer(id)?.output(id)
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
            .steer(Message::user(main_turn_text(&message)));
        Ok(())
    }

    /// Message a child from its focus thread. A queued or running child takes it
    /// as a user steer, a blocked child takes it as the answer to its question,
    /// and a completed child starts a user follow-up. Each makes the child a
    /// focus thread.
    pub fn send_from_user(&self, id: &str, message: String) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let status = child.state.borrow().status.clone();
        if status == SubagentStatus::Completed {
            return self.follow_up_from_user(id, message);
        }
        if status.is_terminal() {
            return Err(SubagentError::Terminal(status));
        }
        if status.is_blocked() {
            self.unblock(id, message)?;
        } else {
            child
                .control
                .lock()
                .expect("mutex poisoned")
                .steer(Message::user(format!("{USER_TURN_PREFACE}\n\n{message}")));
        }
        if !child.focus.swap(true, Ordering::AcqRel) {
            let _ = self
                .inner
                .events
                .send(SubagentEvent::Promoted { id: id.to_string() });
        }
        Ok(())
    }

    /// Start a follow-up turn on a completed child, as main requested through
    /// `agent_send`. Its terminal state also pushes a completion notice to the
    /// main inbox.
    pub fn follow_up(&self, id: &str, prompt: String) -> Result<(), SubagentError> {
        self.start_follow_up(id, main_turn_text(&prompt), TurnOrigin::Main)
    }

    /// Start a follow-up turn the user typed into any subagent. The turn gets
    /// `USER_TURN_PREFACE` first and makes the child a focus thread. When it
    /// ends, main gets a quiet notice quoting the user and the outcome.
    pub fn follow_up_from_user(&self, id: &str, prompt: String) -> Result<(), SubagentError> {
        self.start_follow_up(
            id,
            format!("{USER_TURN_PREFACE}\n\n{prompt}"),
            TurnOrigin::User,
        )
    }

    /// Hides a finished child from the user's task list, or shows it again.
    /// Only the listing changes: the workspace, transcript, and follow-up state
    /// stay. A child still at work cannot be hidden.
    pub fn set_archived(&self, id: &str, archived: bool) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        if !child.state.borrow().status.is_terminal() {
            return Err(SubagentError::NotTerminal(id.to_string()));
        }
        child.archived.store(archived, Ordering::Release);
        let _ = self.inner.events.send(SubagentEvent::ArchiveChanged {
            id: id.to_string(),
            archived,
        });
        Ok(())
    }

    /// Start a follow-up turn on a completed child. The child moves back to
    /// [`SubagentStatus::Running`], runs exactly one more turn in its original
    /// session and workspace under a fresh [`RunControl`], then publishes its
    /// own terminal state; `wait`/`output` then observe the new turn's result.
    /// An archived child is unarchived, so the user's task list never hides a
    /// child that is working again.
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
        if matches!(origin, TurnOrigin::User) {
            *child.user_message.lock().expect("mutex poisoned") = Some(
                prompt
                    .strip_prefix(USER_TURN_PREFACE)
                    .and_then(|text| text.strip_prefix("\n\n"))
                    .unwrap_or(&prompt)
                    .to_owned(),
            );
        }
        *control_guard = RunControl::new();
        let control = control_guard.clone();
        drop(control_guard);

        if matches!(origin, TurnOrigin::User) {
            // A user follow-up makes any subagent a focus thread. Store before
            // the Running status publishes: that event refreshes the sidebar
            // from `list`, which must already see the flag.
            child.focus.store(true, Ordering::Release);
        }
        // Like the focus flag, cleared before Running publishes so the sidebar's
        // re-sync already lists the child.
        if child.archived.swap(false, Ordering::AcqRel) {
            let _ = self.inner.events.send(SubagentEvent::ArchiveChanged {
                id: id.to_string(),
                archived: false,
            });
        }
        child.set_status(SubagentStatus::Running);

        self.inner.main_inbox.tasks.spawn(run_follow_up(
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

    /// Includes all nested children, operator waits, and post-turn focus summaries.
    pub fn has_active_children(&self) -> bool {
        !self.inner.main_inbox.tasks.is_empty()
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
                    focus: child.focus.load(Ordering::Acquire),
                    summary: child.summary.lock().expect("mutex poisoned").clone(),
                    workspace: child.workspace_path(),
                    released: child.workspace_released(),
                    archived: child.archived.load(Ordering::Acquire),
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

    /// Waits until every child this manager started has reached a terminal
    /// state, then reports each one. A child can start more children or be
    /// woken by a message, so the pending set is re-read until it is empty.
    pub async fn settle_all(&self) -> Vec<ChildOutcome> {
        loop {
            let pending: Vec<String> = self
                .list()
                .into_iter()
                .filter(|child| !child.status.is_terminal())
                .map(|child| child.id)
                .collect();
            if pending.is_empty() {
                break;
            }
            for id in pending {
                // A child that fails or is cancelled still settles; its outcome is reported below.
                let _ = self.wait(&id).await;
            }
        }
        self.list()
            .into_iter()
            .map(|child| ChildOutcome {
                output: self
                    .output(&child.id)
                    .ok()
                    .flatten()
                    .map(|result| result.output),
                id: child.id,
                name: child.name,
                status: child.status,
                error: child.error,
                usage: child.usage.unwrap_or_default(),
            })
            .collect()
    }

    /// Persisted transcript of a completed child, read from its session file.
    /// Any child qualifies once it completes. Pending, failed, and cancelled
    /// children have none and report `NoTranscript`.
    pub fn transcript(&self, id: &str) -> Result<Vec<Message>, SubagentError> {
        let child = self.child(id)?;
        if child.workspace_released() {
            return Err(SubagentError::WorkspaceReleased(id.to_string()));
        }
        let path = {
            let state = child.state.borrow();
            match (&state.status, &state.outcome) {
                (SubagentStatus::Completed, Some(Ok(result))) => result.session_path.clone(),
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
    /// The resolution reaches the child only through
    /// [`SubagentManager::wait_for_resolution`], so pair every block with one
    /// (or use [`SubagentManager::block_and_wait`]).
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
    /// Resumes the child's [`RunControl`] and delivers the resolution string to
    /// the waiter, which hands it to the child (for `ask_user`, as the tool
    /// result). It is never steered into the transcript as well, so the child
    /// reads the answer once and a credential never enters its history.
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

        child.state.send_modify(|state| {
            state.status = SubagentStatus::Running;
        });
        child.set_status(SubagentStatus::Running);
        child.control.lock().expect("mutex poisoned").resume();

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

/// Reservations one turn holds until its task ends.
struct Seats {
    desk: Arc<TurnDesk>,
    _review: Option<tokio::sync::OwnedSemaphorePermit>,
}

/// Admission for one turn, shared by first and follow-up turns; publishes the
/// terminal state itself when a wait is cancelled or a desk cannot be reserved.
/// Every turn holds a desk for its whole run: its memory, admitted by the broker
/// (queued when the budget is full). Isolated children also take a review slot:
/// each one produces a change set to review, and the slot caps how many are open.
/// The outer `None` means the turn must not run.
async fn acquire_seats(
    inner: &Inner,
    child: &Child,
    control: &RunControl,
    owner: &CancellationToken,
    origin: TurnOrigin,
) -> Option<Seats> {
    let cancel = control.token();
    let desk = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            finish(inner, child, SubagentStatus::Cancelled, Err("cancelled while queued".into()), origin, owner);
            return None;
        }
        _ = owner.cancelled() => {
            finish(inner, child, SubagentStatus::Cancelled, Err("cancelled with parent while queued".into()), origin, owner);
            return None;
        }
        desk = inner.admission.desks.seat(&child.id) => match desk {
            Ok(desk) => desk,
            Err(error) => {
                finish(inner, child, SubagentStatus::Failed, Err(format!("cannot reserve a desk: {error}")), origin, owner);
                return None;
            }
        },
    };
    let review = if child.spec.isolated_workspace {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                finish(inner, child, SubagentStatus::Cancelled, Err("cancelled while queued".into()), origin, owner);
                return None;
            }
            _ = owner.cancelled() => {
                finish(inner, child, SubagentStatus::Cancelled, Err("cancelled with parent while queued".into()), origin, owner);
                return None;
            }
            permit = inner.admission.review_wip.clone().acquire_owned() => match permit {
                Ok(permit) => Some(permit),
                Err(_) => {
                    finish(inner, child, SubagentStatus::Failed, Err("review limit closed".into()), origin, owner);
                    return None;
                }
            },
        }
    } else {
        None
    };
    Some(Seats {
        desk: TurnDesk::new(inner.admission.desks.clone(), child.id.clone(), desk),
        _review: review,
    })
}

/// Lifecycle for one spawned child: cancellation-safe desk wait, then one
/// turn through the shared settle logic.
async fn run_child(
    inner: Arc<Inner>,
    child: Arc<Child>,
    id: String,
    control: RunControl,
    owner: CancellationToken,
    validator: Option<jsonschema::Validator>,
) {
    let Some(seats) = acquire_seats(&inner, &child, &control, &owner, TurnOrigin::Main).await
    else {
        return;
    };

    child.state.send_modify(|state| {
        state.status = SubagentStatus::Running;
    });
    child.set_status(SubagentStatus::Running);

    let factory = inner.factory.clone();
    let spec = TaskSpec {
        ask: Some(UserAsk::new(&inner, &owner, &id)),
        ..child.spec.clone()
    };
    let task_id = id.clone();
    let task_control = control.clone();
    seats
        .desk
        .hold(run_turn(
            &inner,
            &child,
            &control,
            &owner,
            validator,
            TurnOrigin::Main,
            move |events| async move { factory.run(&task_id, spec, task_control, events).await },
        ))
        .await;
    spawn_summary(inner, child, owner);
}

/// One additional turn on a completed child: same session journal, same
/// workspace, fresh [`RunControl`]. Like the first turn it passes the same
/// admission and publishes exactly one terminal state; `follow_up`
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
    let Some(seats) = acquire_seats(&inner, &child, &control, &owner, origin).await else {
        return;
    };

    let validator = child
        .spec
        .output_schema
        .as_ref()
        .and_then(|schema| jsonschema::validator_for(schema).ok());
    let id = child.id.clone();
    let factory = inner.factory.clone();
    let spec = TaskSpec {
        ask: Some(UserAsk::new(&inner, &owner, &id)),
        ..child.spec.clone()
    };
    let task_control = control.clone();
    seats
        .desk
        .hold(run_turn(
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
        ))
        .await;
    spawn_summary(inner, child, owner);
}

/// Rewrites a focus thread's summary from its latest completed turn, off the
/// turn's admission: the terminal state is already published and the desk and
/// review slot go back first. A failed summary keeps the previous one.
fn spawn_summary(inner: Arc<Inner>, child: Arc<Child>, owner: CancellationToken) {
    if !child.focus.load(Ordering::Acquire) {
        return;
    }
    let session_path = match &child.state.borrow().outcome {
        Some(Ok(result)) => result.session_path.clone(),
        _ => return,
    };
    let tasks = inner.main_inbox.tasks.clone();
    tasks.spawn(async move {
        match inner.factory.summarize(&session_path, owner).await {
            Ok(summary) => {
                *child.summary.lock().expect("mutex poisoned") = Some(summary.clone());
                let _ = inner.events.send(SubagentEvent::Summarized {
                    id: child.id.clone(),
                    summary,
                });
            }
            Err(error) => eprintln!("juto: summary for subagent {} failed: {error}", child.id),
        }
    });
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
    // Waits inside the turn give its desk back through the runtime's hook.
    control.set_blocking_wait(Arc::new(crate::pools::DeskWaits));
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
    let reporter = tokio::spawn(relay_turn_events(
        child.id.clone(),
        events_rx,
        inner.events.clone(),
        inner.main_inbox.clone(),
    ));

    let run = turn(events_tx);
    // A panicking factory must still publish a terminal state or every waiter
    // would hang.
    let outcome = AssertUnwindSafe(run).catch_unwind().await;
    // Every report is delivered before the terminal state is published.
    let _ = reporter.await;

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
    finish(inner, child, status, outcome, origin, owner);
}

/// Prepended to follow-ups the user types in a focus thread. The contract main
/// gave at spawn stays in the session history, so the thread keeps it.
pub const USER_TURN_PREFACE: &str = "The user is chatting with you directly in your focus thread. Answer them. Your contract from main still applies, including what to report to main and when.";

/// Prepended to what main sends a child: a steer, a follow-up, or the first
/// prompt. A focus thread labels such turns "Main" and omits the header.
pub const MAIN_TURN_HEADER: &str = "[from main]";

/// `text` framed as a turn main started: the header, a blank line, then `text`.
pub fn main_turn_text(text: &str) -> String {
    format!("{MAIN_TURN_HEADER}\n\n{text}")
}

/// The message inside a turn main started, or `None` when `text` has no header.
pub fn strip_main_turn_header(text: &str) -> Option<&str> {
    text.strip_prefix(MAIN_TURN_HEADER)?.strip_prefix("\n\n")
}

/// Who started a turn. Main-started turns (the first run and main's follow-ups)
/// push a completion notice to the main inbox. User-started focus turns push a
/// quiet notice instead: main reads it on its next run but is not woken for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOrigin {
    Main,
    User,
}

/// Characters of the user's message quoted in a quiet user-turn notice.
const USER_QUOTE_CHARS: usize = 500;

/// Publish the terminal state for a turn — exactly once per turn. The main
/// inbox notice is submitted before the state is published. An over-budget
/// report may still be digesting when the waiter wakes: its digest does not
/// delay the state, and the notice enters the inbox once the digest completes.
fn finish(
    inner: &Inner,
    child: &Child,
    status: SubagentStatus,
    outcome: Result<SubagentResult, String>,
    origin: TurnOrigin,
    owner: &CancellationToken,
) {
    debug_assert!(status.is_terminal());
    let report = match origin {
        TurnOrigin::Main => MainMessageKind::from_terminal(&status).map(|kind| {
            let body = match &outcome {
                Ok(result) => result.output.clone(),
                Err(err) => err.clone(),
            };
            let data = outcome.as_ref().ok().and_then(|result| result.data.clone());
            (kind, body, data)
        }),
        TurnOrigin::User => {
            let asked = child
                .user_message
                .lock()
                .expect("mutex poisoned")
                .take()
                .unwrap_or_default();
            let asked: String = asked.chars().take(USER_QUOTE_CHARS).collect();
            let outcome_text = match &outcome {
                Ok(result) => result.output.clone(),
                Err(err) => err.clone(),
            };
            let body = format!(
                "The user worked with this subagent directly.\nUser: {asked}\nOutcome ({status:?}): {outcome_text}"
            );
            Some((MainMessageKind::UserTurn, body, None))
        }
    };
    if let Some((kind, body, data)) = report {
        inner.main_inbox.submit(
            MainReport {
                origin: child.spec.origin.clone(),
                from: child.id.clone(),
                name: child.spec.name.clone(),
                kind,
                body,
                data,
            },
            inner.factory.clone(),
            owner.clone(),
        );
    }
    let (result, error) = match &outcome {
        Ok(result) => (Some(Box::new(result.clone())), None),
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

/// Relays one child turn's events to observers as they happen, and reports
/// successful desktop artifact posts. Each one becomes a `Desktop` main inbox
/// notice and a `DesktopReported` event; the notice is pushed first, so an
/// observer woken by it already sees the notice. Ends when the turn drops its
/// event sender.
async fn relay_turn_events(
    child_id: String,
    mut events: mpsc::UnboundedReceiver<AgentEvent>,
    observers: broadcast::Sender<SubagentEvent>,
    main_inbox: MainInbox,
) {
    while let Some(event) = events.recv().await {
        let _ = observers.send(SubagentEvent::Agent {
            id: child_id.clone(),
            event: event.clone(),
        });
        let AgentEvent::ToolExecutionEnd { result, .. } = event else {
            continue;
        };
        let Some(artifact) = reported_desktop(&result) else {
            continue;
        };
        main_inbox.push(MainMessage {
            from: child_id.clone(),
            name: None,
            digest_of: None,
            kind: MainMessageKind::Desktop,
            body: format!(
                "Desktop '{}' is open: environment {}, artifact {}.",
                artifact.title, artifact.environment_id, artifact.artifact_id
            ),
        });
        let _ = observers.send(SubagentEvent::DesktopReported {
            id: child_id.clone(),
            artifact,
        });
    }
}

/// The artifact a successful `desktop_artifact` tool result carries, if any.
fn reported_desktop(message: &Message) -> Option<DesktopSessionArtifact> {
    let Message::ToolResult {
        tool_name,
        is_error: false,
        details,
        ..
    } = message
    else {
        return None;
    };
    if tool_name != "desktop_artifact"
        || details.get("type")?.as_str()? != "desktop_session_artifact"
    {
        return None;
    }
    serde_json::from_value(details.get("artifact")?.clone()).ok()
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
    use tokio::sync::Semaphore;
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
            origin: None,
            main_session: None,
            main_snapshot: None,
            tier: None,
            ask: None,
        }
    }

    /// An isolated child: it holds a review slot for each turn.
    fn isolated_spec(task: &str) -> TaskSpec {
        TaskSpec {
            isolated_workspace: true,
            ..spec(task)
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
                        change_id: None,
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
        SubagentManager::new(
            factory,
            crate::pools::Admission::unmetered(max_concurrency),
            owner,
            MainInbox::default(),
        )
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
    async fn ordinary_children_are_not_capped_by_the_review_limit() {
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
        // Closed gate: ordinary children take desks, not review slots, so all five run.
        until_running(&running, 5).await;
        gate.send(true).unwrap();
        for id in &ids {
            timeout(WAIT_TIMEOUT, manager.wait(id))
                .await
                .expect("wait timed out")
                .unwrap_or_else(|err| panic!("child {id} failed: {err}"));
        }
        assert_eq!(max_running.load(Ordering::SeqCst), 5);
        assert_eq!(running.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn isolated_children_are_bounded_by_the_review_limit() {
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
            ids.push(manager.spawn(isolated_spec(&format!("task-{n}"))).unwrap());
        }
        // Closed gate: only the review limit may be running.
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
    async fn isolated_child_waits_for_review_slot() {
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
        let a = manager.spawn(isolated_spec("a")).unwrap();
        let b = manager.spawn(isolated_spec("b")).unwrap();
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
    async fn cancel_queued_isolated_child_never_takes_review_slot() {
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
        let first = manager.spawn(isolated_spec("first")).unwrap();
        until_running(&running, 1).await;
        let second = manager.spawn(isolated_spec("second")).unwrap();
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
        assert_eq!(result.output, format!("again:{}", main_turn_text("more")));
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
        let first = manager.spawn(isolated_spec("running")).unwrap();
        let second = manager.spawn(isolated_spec("queued")).unwrap();
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
            .map(|n| manager.spawn(isolated_spec(&format!("c{n}"))).unwrap())
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
                change_id: None,
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

    /// Asks through the `UserAsk` handle the manager attaches to the spec, and
    /// finishes with the answer it got back.
    struct AskingFactory;

    #[async_trait]
    impl SubagentFactory for AskingFactory {
        async fn run(
            &self,
            id: &str,
            spec: TaskSpec,
            control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let ask = spec.ask.expect("the manager attaches a user line");
            let answer = ask
                .ask("Merge now or verify first?".into())
                .await
                .map_err(|e| e.to_string())?;
            // The answer releases the pause the block engaged.
            if control.is_paused() {
                return Err("the run stayed paused after the answer".into());
            }
            Ok(SubagentResult {
                id: id.to_string(),
                output: format!("answer:{answer}"),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: PathBuf::from("/tmp/test"),
                change_id: None,
            })
        }
    }

    #[tokio::test]
    async fn a_user_message_answers_a_child_asking_the_user() {
        let manager = manager(Arc::new(AskingFactory), 1, CancellationToken::new());
        let id = manager.spawn(spec("review the PR")).unwrap();
        let mut status_rx = manager.subscribe_status(&id).unwrap();
        status_rx
            .wait_for(|s| s.is_blocked())
            .await
            .expect("child blocks on its question");
        assert_eq!(
            summary_of(&manager, &id).status,
            SubagentStatus::Blocked {
                reason: BlockerReason::UserInput,
                prompt: "Merge now or verify first?".into(),
            }
        );
        assert!(!summary_of(&manager, &id).focus);

        manager
            .send_from_user(&id, "verify first".into())
            .expect("the answer is accepted");
        assert!(
            summary_of(&manager, &id).focus,
            "answering promotes the child"
        );

        let result = timeout(WAIT_TIMEOUT, manager.wait(&id))
            .await
            .expect("wait timed out")
            .expect("child completes");
        assert_eq!(result.output, "answer:verify first");
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
        let first = manager.spawn(isolated_spec("first")).unwrap();
        let queued = manager.spawn(isolated_spec("queued")).unwrap();

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
                        change_id: None,
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
    async fn steering_a_running_worker_from_the_user_promotes_it() {
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
        let mut events = manager.subscribe_events();
        let worker = manager.spawn(spec("hold")).unwrap();
        until_running(&running, 1).await;
        assert!(!summary_of(&manager, &worker).focus);

        manager
            .send_from_user(&worker, "what changed?".into())
            .expect("user steer");
        assert!(summary_of(&manager, &worker).focus);

        gate.send(true).unwrap();
        manager.wait(&worker).await.expect("run");
        assert!(summary_of(&manager, &worker).focus);

        let promoted: Vec<String> = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                SubagentEvent::Promoted { id } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(promoted, vec![worker]);
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
                    change_id: None,
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
                change_id: None,
            })
        }

        fn workspace_path(&self, id: &str, spec: &TaskSpec) -> Option<PathBuf> {
            self.isolated.lock().push(spec.isolated_workspace);
            spec.isolated_workspace.then(|| self.base.join(id))
        }
    }

    #[tokio::test]
    async fn isolated_spawn_reports_its_workspace() {
        let base = PathBuf::from("/data");
        let isolated = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let factory = Arc::new(WorkspaceFactory {
            base: base.clone(),
            isolated: Arc::clone(&isolated),
        });
        let manager = manager(factory, 1, CancellationToken::new());

        let mut child_spec = spec("isolated task");
        child_spec.isolated_workspace = true;
        let id = manager.spawn(child_spec).unwrap();
        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
        assert_eq!(summary.workspace, Some(base.join(&id)));

        // The factory must observe the requested isolation.
        manager.wait(&id).await.unwrap();
        assert_eq!(*isolated.lock(), vec![true]);

        // A plain spec reports no workspace.
        let id = manager.spawn(spec("plain task")).unwrap();
        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
        assert_eq!(summary.workspace, None);
        manager.wait(&id).await.unwrap();
        assert_eq!(*isolated.lock(), vec![true, false]);
    }

    #[tokio::test]
    async fn default_workspace_path_reports_no_workspace() {
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
            change_id: None,
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
        assert_eq!(result.output, format!("again:{}", main_turn_text("more")));
        assert_eq!(manager.status(&id), Some(SubagentStatus::Completed));
        assert_eq!(
            manager.output(&id).unwrap().unwrap().output,
            format!("again:{}", main_turn_text("more"))
        );
        // The workspace lookup ran once at spawn and never again.
        assert_eq!(workspace_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn user_follow_up_leaves_a_quiet_notice_that_never_wakes_main() {
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
                name: None,
                digest_of: None,
                body: "done:thread".into(),
            }]
        );

        // The user's follow-up leaves main a quiet notice quoting them. A
        // waker skips it, so main is not started for the user's own turn.
        manager
            .follow_up_from_user(&id, "more".to_string())
            .expect("accepted");
        assert!(matches!(
            manager.follow_up_from_user(&id, "rejected duplicate".into()),
            Err(SubagentError::InvalidTransition { .. })
        ));
        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("user turn");
        assert!(result.output.ends_with("\n\nmore"));
        assert!(manager.main_inbox().take_next_group(|_| true).is_none());
        let quiet = manager.main_inbox().drain();
        assert_eq!(quiet.len(), 1);
        assert_eq!(quiet[0].kind, MainMessageKind::UserTurn);
        assert!(quiet[0].body.contains("User: more"), "{}", quiet[0].body);
        assert!(quiet[0].body.contains("again:"), "{}", quiet[0].body);

        // A loud report takes the quiet one along when it wakes main.
        manager
            .follow_up_from_user(&id, "and this".to_string())
            .expect("accepted");
        manager.wait(&id).await.expect("user turn");

        // Main's own follow-up reports to main again, and waking main for it
        // delivers the quiet notice that was waiting in the same group.
        manager
            .follow_up(&id, "report".to_string())
            .expect("accepted");
        manager.wait(&id).await.expect("main turn");
        let (_, group) = manager
            .main_inbox()
            .take_next_group(|_| true)
            .expect("a loud report wakes main");
        let kinds: Vec<MainMessageKind> = group.iter().map(|message| message.kind).collect();
        assert_eq!(
            kinds,
            [MainMessageKind::UserTurn, MainMessageKind::Completed]
        );
        assert_eq!(group[1].body, format!("again:{}", main_turn_text("report")));
    }

    #[test]
    fn completion_prompt_frames_each_arrival_for_the_model() {
        let arrivals = vec![
            MainMessage {
                from: "w1".into(),
                kind: MainMessageKind::Completed,
                name: None,
                digest_of: None,
                body: "tests pass".into(),
            },
            MainMessage {
                from: "w2".into(),
                kind: MainMessageKind::Failed,
                name: None,
                digest_of: None,
                body: "build error".into(),
            },
        ];

        assert_eq!(
            completion_prompt(&arrivals),
            "[subagent w1 Completed]\ntests pass\n\n[subagent w2 Failed]\nbuild error"
        );
    }

    #[test]
    fn main_message_header_parses_what_main_message_text_writes() {
        let kinds = [
            MainMessageKind::Message,
            MainMessageKind::Completed,
            MainMessageKind::Failed,
            MainMessageKind::Cancelled,
            MainMessageKind::Desktop,
            MainMessageKind::UserTurn,
        ];
        for kind in kinds {
            for (name, digest_of) in [
                (None, None),
                (Some("builder \"two\" ] Failed"), Some(1812)),
                (Some("tab\tback\\slash\nnewline"), None),
            ] {
                let message = MainMessage {
                    from: "w1".into(),
                    kind,
                    name: name.map(Into::into),
                    digest_of,
                    body: "line one\nline two".into(),
                };
                let text = main_message_text(&message);
                let header = parse_main_message_header(text.lines().next().unwrap());
                assert_eq!(
                    header,
                    Some(MainMessageHeader {
                        from: "w1".into(),
                        name: name.map(str::to_owned),
                        kind,
                    }),
                    "{text}"
                );
            }
        }
    }

    #[test]
    fn main_message_header_rejects_other_lines() {
        assert_eq!(
            parse_main_message_header("hello [subagent w1 Completed]"),
            None
        );
        assert_eq!(
            parse_main_message_header("[subagent w1 Completed] tail"),
            None
        );
        assert_eq!(parse_main_message_header("[subagent w1 Bogus]"), None);
        assert_eq!(parse_main_message_header("[subagent w1]"), None);
    }

    #[test]
    fn main_turn_header_frames_and_strips_back() {
        let framed = main_turn_text("look at the parser");
        assert_eq!(strip_main_turn_header(&framed), Some("look at the parser"));
        assert_eq!(strip_main_turn_header("look at the parser"), None);
        assert_eq!(
            strip_main_turn_header(&format!("{USER_TURN_PREFACE}\n\nhi")),
            None
        );
    }

    #[test]
    fn requeue_front_keeps_arrival_order() {
        let inbox = MainInbox::default();
        let note = |body: &str| MainMessage {
            from: "child".into(),
            kind: MainMessageKind::Message,
            name: None,
            digest_of: None,
            body: body.into(),
        };
        inbox.push(note("third"));
        inbox.requeue_front(None, vec![note("first"), note("second")]);
        let bodies: Vec<String> = inbox.drain().into_iter().map(|m| m.body).collect();
        assert_eq!(bodies, ["first", "second", "third"]);
    }

    /// Factory whose one run posts a failed and then a successful desktop artifact.
    struct DesktopFactory;

    #[async_trait]
    impl SubagentFactory for DesktopFactory {
        async fn run(
            &self,
            id: &str,
            _spec: TaskSpec,
            _control: RunControl,
            events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let post = |is_error: bool, artifact_id: &str| AgentEvent::ToolExecutionEnd {
                call_id: artifact_id.to_string(),
                result: Message::ToolResult {
                    tool_call_id: artifact_id.to_string(),
                    tool_name: "desktop_artifact".into(),
                    content: Vec::new(),
                    is_error,
                    details: serde_json::json!({
                        "type": "desktop_session_artifact",
                        "artifact": {
                            "version": 1,
                            "artifactId": artifact_id,
                            "title": "Game",
                            "hostId": "local",
                            "environmentId": "env-1",
                            "grantId": "grant-1",
                        },
                    }),
                    timestamp: 0,
                },
            };
            let _ = events.send(post(true, "failed"));
            let _ = events.send(post(false, "live"));
            Ok(SubagentResult {
                id: id.to_string(),
                output: "done".into(),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: PathBuf::new(),
                change_id: None,
            })
        }
    }

    #[tokio::test]
    async fn successful_desktop_posts_reach_observers_and_main_before_finish() {
        let manager = manager(Arc::new(DesktopFactory), 1, CancellationToken::new());
        let mut events = manager.subscribe_events();
        let id = manager.spawn(spec("build")).unwrap();
        manager.wait(&id).await.expect("run");

        let mut reported = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let SubagentEvent::DesktopReported { id: from, artifact } = event {
                reported.push((from, artifact.artifact_id));
            }
        }
        assert_eq!(reported, vec![(id.clone(), "live".to_string())]);
        assert_eq!(
            manager.main_inbox().drain(),
            vec![
                MainMessage {
                    from: id.clone(),
                    name: None,
                    digest_of: None,
                    kind: MainMessageKind::Desktop,
                    body: "Desktop 'Game' is open: environment env-1, artifact live.".into(),
                },
                MainMessage {
                    from: id,
                    name: None,
                    digest_of: None,
                    kind: MainMessageKind::Completed,
                    body: "done".into(),
                },
            ]
        );
    }

    /// Factory whose one run streams two text deltas before it finishes.
    struct StreamingFactory;

    #[async_trait]
    impl SubagentFactory for StreamingFactory {
        async fn run(
            &self,
            id: &str,
            _spec: TaskSpec,
            _control: RunControl,
            events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            for delta in ["Hel", "lo"] {
                let _ = events.send(AgentEvent::MessageUpdate {
                    event: juto_ai::ProviderEvent::TextDelta {
                        index: 0,
                        delta: delta.into(),
                    },
                });
            }
            Ok(SubagentResult {
                id: id.to_string(),
                output: "Hello".into(),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: PathBuf::new(),
                change_id: None,
            })
        }
    }

    #[tokio::test]
    async fn child_turn_events_stream_to_observers_in_order() {
        let manager = manager(Arc::new(StreamingFactory), 1, CancellationToken::new());
        let mut events = manager.subscribe_events();
        let id = manager.spawn(spec("chat")).unwrap();
        manager.wait(&id).await.expect("run");

        let mut deltas = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let SubagentEvent::Agent {
                id: from,
                event:
                    AgentEvent::MessageUpdate {
                        event: juto_ai::ProviderEvent::TextDelta { delta, .. },
                    },
            } = event
            {
                assert_eq!(from, id);
                deltas.push(delta);
            }
        }
        assert_eq!(deltas, ["Hel", "lo"]);
    }

    #[tokio::test]
    async fn archiving_hides_finished_children_and_refuses_working_ones() {
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
            2,
            CancellationToken::new(),
        );
        let mut events = manager.subscribe_events();
        let worker = manager.spawn(spec("scan")).unwrap();
        until_running(&running, 1).await;

        assert_eq!(
            manager.set_archived(&worker, true),
            Err(SubagentError::NotTerminal(worker.clone()))
        );
        assert!(!summary_of(&manager, &worker).archived);

        gate.send(true).unwrap();
        manager.wait(&worker).await.expect("worker run");
        manager.set_archived(&worker, true).unwrap();
        assert!(summary_of(&manager, &worker).archived);
        manager.set_archived(&worker, false).unwrap();
        assert!(!summary_of(&manager, &worker).archived);

        let archive_events: Vec<(String, bool)> = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                SubagentEvent::ArchiveChanged { id, archived } => Some((id, archived)),
                _ => None,
            })
            .collect();
        assert_eq!(
            archive_events,
            vec![(worker.clone(), true), (worker, false)]
        );
    }

    #[test]
    fn inbox_delivers_reports_to_the_session_that_spawned_them() {
        let inbox = MainInbox::default();
        let note = |body: &str| MainMessage {
            from: "child".into(),
            name: None,
            digest_of: None,
            kind: MainMessageKind::Completed,
            body: body.into(),
        };
        inbox.push_from(Some("thread-a".into()), note("a1"));
        inbox.push_from(Some("thread-b".into()), note("b1"));
        inbox.push(note("anyone"));
        inbox.push_from(Some("thread-a".into()), note("a2"));

        // A run of thread b takes its own report and the unaddressed one,
        // never thread a's.
        let bodies: Vec<String> = inbox
            .drain_for("thread-b")
            .into_iter()
            .map(|m| m.body)
            .collect();
        assert_eq!(bodies, ["b1", "anyone"]);

        // Waking idle threads takes one origin group at a time, oldest first.
        let (origin, group) = inbox.take_next_group(|_| true).expect("thread a group");
        assert_eq!(origin.as_deref(), Some("thread-a"));
        let bodies: Vec<String> = group.into_iter().map(|m| m.body).collect();
        assert_eq!(bodies, ["a1", "a2"]);
        assert!(inbox.take_next_group(|_| true).is_none());
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
                SubagentEvent::StatusChanged { .. } | SubagentEvent::MainInboxReady => {}
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
    async fn follow_up_runs_again_in_an_isolated_child_workspace() {
        let (first, first_rx) = gate();
        let (second, second_rx) = gate();
        let workspace_calls = Arc::new(AtomicUsize::new(0));
        let factory = follow_up_factory(first_rx, second_rx, workspace_calls.clone());
        let manager = manager(factory, 1, CancellationToken::new());

        let mut isolated = spec("isolated task");
        isolated.isolated_workspace = true;
        let id = manager.spawn(isolated).unwrap();
        first.send(true).unwrap();
        manager.wait(&id).await.expect("first turn");

        manager
            .follow_up(&id, "next".to_string())
            .expect("accepted");
        second.send(true).unwrap();
        let result = manager.wait(&id).await.expect("follow-up turn");
        assert_eq!(result.output, format!("again:{}", main_turn_text("next")));

        let summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .expect("spawned child listed");
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
        assert_eq!(result.output, format!("again:{}", main_turn_text("more")));
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
                name: None,
                digest_of: None,
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
                name: None,
                digest_of: None,
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
                name: None,
                digest_of: None,
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
            crate::pools::Admission::unmetered(1),
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
                name: None,
                digest_of: None,
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
                change_id: None,
            })
        }

        async fn follow_up(
            &self,
            id: &str,
            _spec: &TaskSpec,
            session_path: &Path,
            prompt: String,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            let mut session = Session::open(session_path).map_err(|err| err.to_string())?;
            session
                .append(crate::session::EntryKind::Message {
                    message: Message::user(prompt.clone()),
                })
                .map_err(|err| err.to_string())?;
            Ok(SubagentResult {
                id: id.to_string(),
                output: prompt,
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: session_path.to_path_buf(),
                change_id: None,
            })
        }

        /// Summarizes as the journal's last message, so a test can tell which
        /// turn a summary came from.
        async fn summarize(
            &self,
            session_path: &Path,
            _cancel: CancellationToken,
        ) -> Result<String, String> {
            let session = Session::open(session_path).map_err(|err| err.to_string())?;
            let messages = session.messages().map_err(|err| err.to_string())?;
            let last = messages.last().map(Message::text).unwrap_or_default();
            Ok(format!("summary:{last}"))
        }
    }

    /// Waits for the next `Summarized` event for `id`, or fails after the timeout.
    async fn next_summary(events: &mut broadcast::Receiver<SubagentEvent>, id: &str) -> String {
        timeout(WAIT_TIMEOUT, async {
            loop {
                if let Ok(SubagentEvent::Summarized { id: from, summary }) = events.recv().await
                    && from == id
                {
                    return summary;
                }
            }
        })
        .await
        .expect("a summary within the timeout")
    }

    #[tokio::test]
    async fn only_focus_threads_get_a_summary_after_each_turn() {
        let dir = std::env::temp_dir().join(format!("juto-summary-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let manager = manager(
            Arc::new(TranscriptFactory { dir: dir.clone() }),
            1,
            CancellationToken::new(),
        );
        let mut events = manager.subscribe_events();

        let id = manager.spawn(spec("plain task")).unwrap();
        manager.wait(&id).await.unwrap();
        // A main follow-up keeps the child a task: still no summary.
        manager.follow_up(&id, "from main".into()).unwrap();
        manager.wait(&id).await.unwrap();
        assert_eq!(summary_of(&manager, &id).summary, None);

        manager
            .follow_up_from_user(&id, "first ask".into())
            .unwrap();
        assert!(next_summary(&mut events, &id).await.ends_with("first ask"));
        manager.follow_up(&id, "from main again".into()).unwrap();
        assert!(
            next_summary(&mut events, &id)
                .await
                .ends_with("from main again"),
            "a focus thread is summarized after main's turns too"
        );
        assert!(
            summary_of(&manager, &id)
                .summary
                .is_some_and(|summary| summary.ends_with("from main again"))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn completed_child_returns_persisted_transcript() {
        let dir = std::env::temp_dir().join(format!("juto-transcript-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let manager = manager(
            Arc::new(TranscriptFactory { dir: dir.clone() }),
            1,
            CancellationToken::new(),
        );

        let mut focus = spec("focus task");
        focus.isolated_workspace = true;
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
            SubagentError::NoTranscript(plain_id.clone())
        );

        let mut focus = spec("focus task");
        focus.isolated_workspace = true;
        let focus_id = manager.spawn(focus).unwrap();
        assert!(manager.wait(&focus_id).await.is_err());
        assert_eq!(
            manager.transcript(&focus_id).unwrap_err(),
            SubagentError::NoTranscript(focus_id.clone())
        );
    }

    #[tokio::test]
    async fn user_follow_up_makes_plain_child_a_focus_thread() {
        let dir = std::env::temp_dir().join(format!("juto-promote-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let manager = manager(
            Arc::new(TranscriptFactory { dir: dir.clone() }),
            1,
            CancellationToken::new(),
        );

        let id = manager.spawn(spec("plain task")).unwrap();
        manager.wait(&id).await.unwrap();
        assert!(!summary_of(&manager, &id).focus);

        // A main follow-up keeps the child a worker.
        manager.follow_up(&id, "from main".into()).unwrap();
        manager.wait(&id).await.unwrap();
        assert!(!summary_of(&manager, &id).focus);

        manager
            .follow_up_from_user(&id, "from user".into())
            .unwrap();
        manager.wait(&id).await.unwrap();
        assert!(summary_of(&manager, &id).focus);
        let texts: Vec<String> = manager
            .transcript(&id)
            .unwrap()
            .iter()
            .map(Message::text)
            .collect();
        assert!(texts.iter().any(|text| text.contains("from user")));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn messaging_an_archived_child_unarchives_it() {
        let dir = std::env::temp_dir().join(format!("juto-unarchive-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let manager = manager(
            Arc::new(TranscriptFactory { dir: dir.clone() }),
            1,
            CancellationToken::new(),
        );
        let id = manager.spawn(spec("plain task")).unwrap();
        manager.wait(&id).await.unwrap();
        manager.set_archived(&id, true).unwrap();
        let mut events = manager.subscribe_events();

        manager
            .send_from_user(&id, "one more thing".into())
            .unwrap();
        assert!(
            !summary_of(&manager, &id).archived,
            "the child is listed again as soon as its turn starts"
        );
        let unarchived = std::iter::from_fn(|| events.try_recv().ok()).any(|event| {
            matches!(event, SubagentEvent::ArchiveChanged { id: ref changed, archived: false } if *changed == id)
        });
        assert!(unarchived, "observers hear that the child is back");
        manager.wait(&id).await.unwrap();
        assert!(!summary_of(&manager, &id).archived);

        std::fs::remove_dir_all(&dir).unwrap();
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
                change_id: None,
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
    async fn release_removes_the_workspace_and_clears_summary() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.isolated_workspace = true;
        let id = manager.spawn(focus).unwrap();
        manager.wait(&id).await.unwrap();
        let dir = root.path().join(&id);
        assert_eq!(summary_of(&manager, &id).workspace, Some(dir.clone()));

        manager.release(&id).await.unwrap();
        assert!(!dir.exists(), "release removes the workspace");
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
        focus.isolated_workspace = true;
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
        focus.isolated_workspace = true;
        let id = manager.spawn(focus).unwrap();
        manager.wait(&id).await.unwrap();

        manager.release(&id).await.unwrap();
        manager.release(&id).await.unwrap();
        assert_eq!(factory.releases.load(Ordering::SeqCst), 1);
        assert!(summary_of(&manager, &id).released);
    }

    #[tokio::test]
    async fn released_child_refuses_follow_up_and_transcript() {
        let root = tempfile::tempdir().unwrap();
        let factory = release_factory(root.path(), None);
        let manager = manager(factory.clone(), 1, CancellationToken::new());

        let mut focus = spec("focus task");
        focus.isolated_workspace = true;
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
        focus.isolated_workspace = true;
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

    #[tokio::test]
    async fn dropped_managers_release_event_channels_and_prune_registrations() {
        let inbox = MainInbox::default();
        let factory = Arc::new(DesktopFactory);
        let root = SubagentManager::new(
            factory.clone(),
            Admission::unmetered(1),
            CancellationToken::new(),
            inbox.clone(),
        );
        let mut root_events = root.subscribe_events();
        for _ in 0..64 {
            let nested = SubagentManager::with_origin(
                factory.clone(),
                Admission::unmetered(1),
                CancellationToken::new(),
                inbox.clone(),
                Some("topic".into()),
            );
            let mut nested_events = nested.subscribe_events();
            drop(nested);
            assert!(matches!(
                nested_events.try_recv(),
                Err(broadcast::error::TryRecvError::Closed)
            ));
        }
        root.report_to_main("sender".into(), "ready".into());
        assert!(matches!(
            root_events.recv().await.unwrap(),
            SubagentEvent::MainInboxReady
        ));
        assert_eq!(inbox.event_senders.lock().unwrap().len(), 1);
    }

    struct OwnerDigestFactory {
        entered: Semaphore,
        release: Semaphore,
    }

    #[async_trait]
    impl SubagentFactory for OwnerDigestFactory {
        async fn run(
            &self,
            id: &str,
            _spec: TaskSpec,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            Ok(turn_result(id, "first report ".repeat(2000)))
        }

        async fn follow_up(
            &self,
            id: &str,
            _spec: &TaskSpec,
            _path: &Path,
            _prompt: String,
            _control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            Ok(turn_result(id, "follow-up report ".repeat(2000)))
        }

        async fn digest_report(
            &self,
            _body: &str,
            _limit: usize,
            _cancel: CancellationToken,
        ) -> Result<String, String> {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
            Ok("Follow-up verified; no blockers.".into())
        }
    }

    #[tokio::test]
    async fn digest_cancellation_uses_each_turn_owner_not_the_previous_owner() {
        let factory = Arc::new(OwnerDigestFactory {
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
        });
        let owner = CancellationToken::new();
        let manager = manager(factory.clone(), 1, owner.clone());
        let mut task = spec("work");
        task.origin = Some("topic-a".into());
        let id = manager.spawn(task).unwrap();
        manager.wait(&id).await.unwrap();
        timeout(WAIT_TIMEOUT, factory.entered.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        owner.cancel();
        let inbox = manager.main_inbox();
        timeout(WAIT_TIMEOUT, inbox.arrived()).await.unwrap();
        assert!(inbox.drain_for("topic-b").is_empty());
        let first = inbox.drain_for("topic-a");
        assert_eq!(first.len(), 1);
        assert!(first[0].body.contains("digest unavailable: cancelled"));
        assert!(first[0].body.len() <= main_report::DEFAULT_MAIN_REPORT_BYTES);

        let fresh_owner = CancellationToken::new();
        let follow_up = manager.scoped(fresh_owner.clone());
        follow_up.send(&id, "continue".into()).unwrap();
        follow_up.wait(&id).await.unwrap();
        timeout(WAIT_TIMEOUT, factory.entered.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        assert!(
            inbox.drain().is_empty(),
            "a fresh owner must not inherit cancellation"
        );
        factory.release.add_permits(1);
        timeout(WAIT_TIMEOUT, inbox.arrived()).await.unwrap();
        assert!(inbox.drain_for("topic-b").is_empty());
        let second = inbox.drain_for("topic-a");
        assert_eq!(second.len(), 1);
        assert!(
            second[0]
                .body
                .starts_with("Follow-up verified; no blockers.")
        );
        assert!(!fresh_owner.is_cancelled());
    }

    /// A digested report's pointer still resolves after the inbox restarts:
    /// the archive is persisted alongside the queue that points into it.
    #[tokio::test]
    async fn digested_report_pointer_resolves_after_restart() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("inbox.json");
        let factory = Arc::new(OwnerDigestFactory {
            entered: Semaphore::new(0),
            release: Semaphore::new(1),
        });
        let manager = manager(factory, 1, CancellationToken::new());
        let inbox = manager.main_inbox();
        inbox.bind(path.clone()).unwrap();
        let id = manager.spawn(spec("work")).unwrap();
        manager.wait(&id).await.unwrap();
        timeout(WAIT_TIMEOUT, inbox.arrived()).await.unwrap();
        let message = inbox.drain().pop().unwrap();
        let pointer = message
            .body
            .split("\"ids\":[\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .to_owned();

        let restarted = MainInbox::default();
        restarted.bind(path).unwrap();
        let archived = restarted
            .archived(&pointer)
            .expect("pointer resolves after restart");
        assert!(
            archived["body"]
                .as_str()
                .unwrap()
                .starts_with("first report ")
        );
    }
}
