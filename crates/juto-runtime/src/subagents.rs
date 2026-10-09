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
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::FutureExt;
use juto_agent::{AgentEvent, RunControl};
use juto_ai::{Message, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Semaphore, mpsc, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Model-visible description of one child run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpec {
    /// Optional human-facing label; the child id is always generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The task instructions given to the child agent.
    pub task: String,
    /// Registry selector (`provider/model`) overriding the factory's base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// When set, the child must answer with JSON validating against this schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Deny Write/Exec-tier tools to the child.
    #[serde(default)]
    pub read_only: bool,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl SubagentStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
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
}

/// Errors surfaced by [`SubagentManager`]. Distinct variants preserve the
/// explicit terminal statuses instead of collapsing them into strings.
#[derive(Debug)]
pub enum SubagentError {
    /// No child with this id was ever spawned.
    Unknown(String),
    /// `send`/`wait` targeted a child that already reached a terminal state.
    Terminal(SubagentStatus),
    /// The child run failed; the payload is the child's error.
    Failed(String),
    /// The child was cancelled before or during its run.
    Cancelled,
    /// The task spec itself is invalid (e.g. malformed `output_schema`).
    InvalidSpec(String),
    /// `spawn` was called outside a Tokio runtime.
    NoRuntime,
}

impl fmt::Display for SubagentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "unknown subagent id: {id}"),
            Self::Terminal(status) => write!(f, "subagent already {status:?}"),
            Self::Failed(err) => write!(f, "subagent failed: {err}"),
            Self::Cancelled => write!(f, "subagent cancelled"),
            Self::InvalidSpec(err) => write!(f, "invalid task spec: {err}"),
            Self::NoRuntime => write!(f, "no Tokio runtime available for spawn"),
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
}

/// Snapshot a child exposes to waiters; the watch channel guarantees every
/// waiter sees the latest terminal state even if it subscribed late.
#[derive(Debug, Clone)]
struct ChildState {
    status: SubagentStatus,
    /// `Some` once the factory returned, regardless of success.
    outcome: Option<Result<SubagentResult, String>>,
}

struct Child {
    spec: TaskSpec,
    control: RunControl,
    state: watch::Sender<ChildState>,
}

struct Inner {
    factory: Arc<dyn SubagentFactory>,
    semaphore: Arc<Semaphore>,
    parent_cancel: CancellationToken,
    children: Mutex<HashMap<String, Arc<Child>>>,
}

/// Bounded manager for child subagent runs.
///
/// `spawn` never blocks on a semaphore permit; acquisition happens inside the
/// child task so permits are held only by running children.
pub struct SubagentManager {
    inner: Arc<Inner>,
}

impl SubagentManager {
    pub fn new(
        factory: Arc<dyn SubagentFactory>,
        max_concurrency: usize,
        parent_cancel: CancellationToken,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                factory,
                semaphore: Arc::new(Semaphore::new(max_concurrency.max(1))),
                parent_cancel,
                children: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Queue a child run. Returns the generated child id immediately; the
    /// permit wait, run, and terminal-state publication are asynchronous.
    pub fn spawn(&self, spec: TaskSpec) -> Result<String, SubagentError> {
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
        let child = Arc::new(Child {
            spec,
            control: control.clone(),
            state,
        });
        self.inner
            .children
            .lock()
            .expect("subagent registry poisoned")
            .insert(id.clone(), Arc::clone(&child));

        let inner = Arc::clone(&self.inner);
        let task_id = id.clone();
        tokio::spawn(run_child(inner, child, task_id, control, validator));
        Ok(id)
    }

    /// Wait for the child's terminal state and return its result.
    /// Concurrent and post-completion waiters all observe the same outcome.
    pub async fn wait(&self, id: &str) -> Result<SubagentResult, SubagentError> {
        let mut rx = self.state_receiver(id)?;
        loop {
            let snapshot = rx.borrow().clone();
            match snapshot.status {
                SubagentStatus::Queued | SubagentStatus::Running => {}
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
        self.child(id)?.control.cancel();
        Ok(())
    }

    /// Steer a queued or running child with a user message. The child's
    /// [`RunControl`] exists from spawn, so steering a queued child takes
    /// effect on its first turn boundary.
    pub fn send(&self, id: &str, message: String) -> Result<(), SubagentError> {
        let child = self.child(id)?;
        let status = child.state.borrow().status;
        if status.is_terminal() {
            return Err(SubagentError::Terminal(status));
        }
        child.control.steer(Message::user(message));
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
                    status: state.status,
                    stop_reason,
                    usage,
                    error,
                }
            })
            .collect();
        summaries.sort_by(|a, b| a.id.cmp(&b.id));
        summaries
    }

    /// Completed child result, `None` while pending/failed/cancelled.
    pub fn output(&self, id: &str) -> Result<Option<SubagentResult>, SubagentError> {
        let child = self.child(id)?;
        Ok(child
            .state
            .borrow()
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.as_ref().ok())
            .cloned())
    }

    /// Cancel every child that has not reached a terminal state.
    pub fn cancel_all(&self) {
        for child in self
            .inner
            .children
            .lock()
            .expect("subagent registry poisoned")
            .values()
        {
            if !child.state.borrow().status.is_terminal() {
                child.control.cancel();
            }
        }
    }

    /// Status of one child, `None` for unknown ids.
    pub fn status(&self, id: &str) -> Option<SubagentStatus> {
        self.child(id).ok().map(|child| child.state.borrow().status)
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

/// Lifecycle for one spawned child: permit wait (cancellation-safe), parent
/// cancellation propagation, factory run, terminal publication.
async fn run_child(
    inner: Arc<Inner>,
    child: Arc<Child>,
    id: String,
    control: RunControl,
    validator: Option<jsonschema::Validator>,
) {
    // Permit acquisition is cancellation-safe: a cancelled queued child
    // neither steals a permit nor blocks waiting for one.
    let cancel = control.token();
    let permit = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            finish(&child, SubagentStatus::Cancelled, Err("cancelled while queued".into()));
            return;
        }
        _ = inner.parent_cancel.cancelled() => {
            finish(&child, SubagentStatus::Cancelled, Err("cancelled with parent while queued".into()));
            return;
        }
        permit = inner.semaphore.clone().acquire_owned() => match permit {
            Ok(permit) => permit,
            Err(_) => {
                finish(&child, SubagentStatus::Failed, Err("subagent semaphore closed".into()));
                return;
            }
        },
    };
    let _permit = permit; // held for the duration of the run

    child.state.send_modify(|state| {
        state.status = SubagentStatus::Running;
    });

    // Propagate parent cancellation into the child's RunControl for the whole
    // run, then release the propagation task when the child finishes.
    let propagation_done = CancellationToken::new();
    tokio::spawn({
        let parent = inner.parent_cancel.clone();
        let control = control.clone();
        let done = propagation_done.clone();
        async move {
            tokio::select! {
                _ = parent.cancelled() => control.cancel(),
                _ = done.cancelled() => {}
            }
        }
    });

    let (events_tx, events_rx) = mpsc::unbounded_channel();
    // The manager owns no consumer for child events; keep the receiver alive
    // so `UnboundedSender::send` does not fail during the run.
    let _events_rx = events_rx;

    let run = inner
        .factory
        .run(&id, child.spec.clone(), control.clone(), events_tx);
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
    let outcome = match (validator, outcome) {
        (Some(validator), Ok(result)) => match &result.data {
            Some(data) => match validator.validate(data) {
                Ok(()) => Ok(result),
                Err(err) => Err(format!("output schema violation: {err}")),
            },
            None => Err("child produced no structured output for output_schema".into()),
        },
        (_, outcome) => outcome,
    };

    let cancelled = control.token().is_cancelled() || inner.parent_cancel.is_cancelled();
    let status = match &outcome {
        Ok(result) if result.stop_reason == StopReason::Aborted || cancelled => {
            SubagentStatus::Cancelled
        }
        Ok(_) => SubagentStatus::Completed,
        Err(_) if cancelled => SubagentStatus::Cancelled,
        Err(_) => SubagentStatus::Failed,
    };
    finish(&child, status, outcome);
}

/// Publish the terminal state exactly once.
fn finish(child: &Child, status: SubagentStatus, outcome: Result<SubagentResult, String>) {
    debug_assert!(status.is_terminal());
    child.state.send_modify(|state| {
        state.status = status;
        state.outcome = Some(outcome);
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
        if let (Some(start), Some(end)) = (trimmed.find(open), trimmed.rfind(close)) {
            if start < end {
                if let Ok(value) = serde_json::from_str::<Value>(&trimmed[start..=end]) {
                    return Some(value);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{Duration as TokioDuration, timeout};

    const WAIT_TIMEOUT: TokioDuration = TokioDuration::from_secs(5);

    fn spec(task: &str) -> TaskSpec {
        TaskSpec {
            name: None,
            task: task.to_string(),
            model: None,
            output_schema: None,
            read_only: false,
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
        parent: CancellationToken,
    ) -> SubagentManager {
        SubagentManager::new(factory, max_concurrency, parent)
    }

    /// Poll until `running` reaches `n`.
    async fn until_running(running: &AtomicUsize, n: usize) {
        for _ in 0..200 {
            if running.load(Ordering::SeqCst) == n {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "expected {n} running children, got {}",
            running.load(Ordering::SeqCst)
        );
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
    async fn send_steers_queued_and_rejects_terminal() {
        // RunControl queues are internal to juto-agent; the observable
        // contract is that send() succeeds for queued/running children and is
        // a typed error once the child is terminal.
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
        assert!(matches!(
            manager.send(&held, "done".into()),
            Err(SubagentError::Terminal(SubagentStatus::Completed))
        ));
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
    async fn parent_cancel_cancels_queued_and_running() {
        let (_gate, gate_rx) = gate();
        let running = Arc::new(AtomicUsize::new(0));
        let parent = CancellationToken::new();
        let manager = manager(
            gate_factory(
                gate_rx,
                running.clone(),
                Arc::new(AtomicUsize::new(0)),
                false,
                None,
            ),
            1,
            parent.clone(),
        );
        let first = manager.spawn(spec("running")).unwrap();
        let second = manager.spawn(spec("queued")).unwrap();
        until_running(&running, 1).await;
        parent.cancel();
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
    async fn cancel_all_marks_every_open_child() {
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
        let ids: Vec<String> = (0..3)
            .map(|n| manager.spawn(spec(&format!("c{n}"))).unwrap())
            .collect();
        until_running(&running, 1).await;
        manager.cancel_all();
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
        assert!(parsed.name.is_none() && parsed.model.is_none());
        assert!(parsed.output_schema.is_none() && !parsed.read_only);

        let mut full = spec("x");
        full.name = Some("named".into());
        full.read_only = true;
        full.output_schema = Some(serde_json::json!({"type": "string"}));
        let round: TaskSpec = serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
        assert!(round.read_only);
        assert_eq!(round.name.as_deref(), Some("named"));
        assert!(round.output_schema.is_some());
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
}
