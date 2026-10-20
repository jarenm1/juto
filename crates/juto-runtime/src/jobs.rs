//! Durable background jobs: an unattended run in its own isolated workspace,
//! under a wall-clock budget, with a completion contract, checkpoint retry, and
//! an inbox.
//!
//! A job's run is the same unattended run a session gets, but its session lives
//! in a fresh fork of the repository, so a job never writes into the user's
//! working copy. A job succeeds only when its [`JobContract`] verifies: the
//! model stopping is not enough. Every transition is persisted under
//! `<data_dir>/jobs/<id>.json` before it is announced on
//! [`Runtime::subscribe_jobs`]. A runtime that stops while a job runs leaves that
//! job `interrupted` on the next [`Runtime::recover_jobs`]. Nothing replays on
//! restart: retry continues the saved session, and a tool call that was in flight
//! is recorded as interrupted rather than run again.
//!
//! `wall_secs` bounds the total time a job spends running, across attempts and
//! decision resumptions. `max_turns` and `max_spend` are enforced per job too;
//! exhausting either one fails the job.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use juto_agent::{AgentError, AgentEvent, RunControl};
use juto_ai::{AssistantMessage, ContentBlock, Message, StopReason, timestamp_ms};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use crate::change_graph::ChangeGraph;
use crate::decisions::{Decision, DecisionStatus, continuation_prompt};
use crate::subagents::MainInbox;
use crate::subagents::{ChildOutcome, SubagentStatus};
use crate::workspace::SubagentWorkspace;
use crate::{
    ChildFactory, Exhausted, JobLimits, JobMeter, Runtime, Session, UnattendedOutcome,
    UnattendedRun, reclaim_isolated_workspace,
};

/// Job state transitions a subscriber may fall behind by before it lags.
const EVENT_BUFFER: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobStatus {
    Queued,
    Running,
    /// A tool call needs an operator decision; resuming needs that decision approved.
    Blocked,
    Succeeded,
    Failed,
    /// The runtime stopped while the job was queued or running.
    Interrupted,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Interrupted | Self::Cancelled
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Whether an operator still needs to look at a job. New jobs start `Read`;
/// every transition that ends a run or blocks it sets `Unread`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InboxState {
    Unread,
    Read,
    /// Terminal jobs only. Archiving reclaims the job's isolated workspaces.
    Archived,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobBudget {
    /// Total wall-clock seconds the job may spend running.
    pub wall_secs: u64,
    /// Attempts a job may start, counting the first.
    pub max_attempts: u32,
    /// Model turns one attempt may take. Unset leaves only the runtime's own turn limit.
    #[serde(default)]
    pub max_turns: Option<u32>,
    /// USD the job may spend across all attempts, children included, priced by the model catalog. Unset means no spend limit.
    #[serde(default)]
    pub max_spend: Option<f64>,
}

impl JobBudget {
    fn validate(self) -> Result<(), JobError> {
        if self.wall_secs == 0 {
            return Err(JobError::Invalid("wall_secs must be at least 1".into()));
        }
        if self.max_attempts == 0 {
            return Err(JobError::Invalid("max_attempts must be at least 1".into()));
        }
        if self.max_turns == Some(0) {
            return Err(JobError::Invalid("max_turns must be at least 1".into()));
        }
        if let Some(spend) = self.max_spend
            && !(spend.is_finite() && spend > 0.0)
        {
            return Err(JobError::Invalid(
                "max_spend must be a positive USD amount".into(),
            ));
        }
        Ok(())
    }

    fn wall_ms(self) -> u64 {
        self.wall_secs.saturating_mul(1000)
    }
}

/// What a job must show before it counts as succeeded. The model stopping is
/// not enough, so a job needs at least one clause.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobContract {
    /// A file the job must leave in its workspace, relative to the workspace
    /// root. It must exist and be non-empty.
    pub artifact: Option<String>,
    /// A JSON Schema the assistant's final text must parse as and satisfy.
    pub output_schema: Option<serde_json::Value>,
    /// The workspace must have no changes when the job stops. A verified
    /// no-change result is archived automatically.
    #[serde(default)]
    pub no_change: bool,
}

impl JobContract {
    fn validate(&self) -> Result<(), JobError> {
        if self.artifact.is_none() && self.output_schema.is_none() && !self.no_change {
            return Err(JobError::Invalid(
                "a job needs a completion contract: an artifact, an output schema, or no_change"
                    .into(),
            ));
        }
        if let Some(path) = &self.artifact {
            let relative = !path.is_empty()
                && Path::new(path)
                    .components()
                    .all(|part| matches!(part, Component::Normal(_)));
            if !relative {
                return Err(JobError::Invalid(format!(
                    "artifact `{path}` must be a relative path inside the workspace"
                )));
            }
            if self.no_change {
                return Err(JobError::Invalid(
                    "an artifact contract cannot also require no change".into(),
                ));
            }
        }
        if let Some(schema) = &self.output_schema {
            jsonschema::validator_for(schema)
                .map_err(|error| JobError::Invalid(format!("invalid output_schema: {error}")))?;
        }
        Ok(())
    }
}

/// One isolated workspace a job attempt ran in. Archiving reclaims it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobFork {
    pub path: PathBuf,
    pub repo_root: PathBuf,
    pub thread: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
    pub id: String,
    pub prompt: String,
    pub budget: JobBudget,
    pub contract: JobContract,
    pub status: JobStatus,
    pub inbox: InboxState,
    /// Whether the main agent has been shown this job's terminal result.
    #[serde(default)]
    pub drained: bool,
    #[serde(default)]
    pub attempts: u32,
    /// Wall-clock milliseconds spent running, across attempts and resumptions.
    #[serde(default)]
    pub run_ms: u64,
    /// Model turns the current attempt has taken. Resets when an attempt starts.
    #[serde(default)]
    pub turns: u32,
    /// USD spent across attempts, children included.
    #[serde(default)]
    pub spend_usd: f64,
    pub session_id: Option<String>,
    pub session_path: Option<PathBuf>,
    /// Every attempt's workspace, oldest first. The last one is the current workspace.
    #[serde(default)]
    pub forks: Vec<JobFork>,
    pub decision_id: Option<String>,
    /// The assistant's final text, once a run has produced one.
    pub result: Option<String>,
    pub error: Option<String>,
    /// The process running the job, while it is queued or running.
    pub owner_pid: Option<u32>,
    pub created_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
}

impl Job {
    fn new(id: String, prompt: String, budget: JobBudget, contract: JobContract) -> Self {
        Self {
            id,
            prompt,
            budget,
            contract,
            status: JobStatus::Queued,
            inbox: InboxState::Read,
            drained: false,
            attempts: 0,
            run_ms: 0,
            turns: 0,
            spend_usd: 0.0,
            session_id: None,
            session_path: None,
            forks: Vec::new(),
            decision_id: None,
            result: None,
            error: None,
            owner_pid: Some(std::process::id()),
            created_ms: timestamp_ms(),
            started_ms: None,
            finished_ms: None,
        }
    }
}

#[derive(Debug)]
pub enum JobError {
    NotFound(String),
    /// The request does not fit the job's current state.
    Invalid(String),
    Failed(anyhow::Error),
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "unknown job `{id}`"),
            Self::Invalid(message) => f.write_str(message),
            Self::Failed(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for JobError {}

impl From<anyhow::Error> for JobError {
    fn from(error: anyhow::Error) -> Self {
        Self::Failed(error)
    }
}

/// Serializes read-modify-write cycles over job files within this process.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone)]
struct JobStore {
    dir: PathBuf,
}

impl JobStore {
    fn get(&self, id: &str) -> Result<Option<Job>> {
        let path = self.path(id)?;
        match fs::read(&path) {
            Ok(bytes) => {
                Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
                    format!("job record {} is corrupt", path.display())
                })?))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn list(&self) -> Result<Vec<Job>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut jobs = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let bytes = fs::read(&path)?;
                jobs.push(
                    serde_json::from_slice::<Job>(&bytes)
                        .with_context(|| format!("job record {} is corrupt", path.display()))?,
                );
            }
        }
        jobs.sort_by(|a, b| (a.created_ms, &a.id).cmp(&(b.created_ms, &b.id)));
        Ok(jobs)
    }

    /// Writes through a temporary file so a crash never leaves a torn record.
    /// The caller holds [`LOCK`].
    fn write(&self, job: &Job) -> Result<()> {
        fs::create_dir_all(&self.dir)?;
        let path = self.path(&job.id)?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(job)?)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn path(&self, id: &str) -> Result<PathBuf> {
        let valid = !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !valid {
            anyhow::bail!("invalid job id `{id}`");
        }
        Ok(self.dir.join(format!("{id}.json")))
    }
}

pub(crate) struct JobRegistry {
    store: JobStore,
    /// Run controls of jobs queued or running in this process.
    live: Mutex<HashMap<String, RunControl>>,
    /// Jobs an operator asked to cancel. Kept apart from the run token because
    /// an unattended block also cancels the run tree, and a block is not a cancel.
    cancel_requested: Mutex<HashSet<String>>,
    events: broadcast::Sender<Job>,
}

impl JobRegistry {
    pub(crate) fn open(data_dir: &Path) -> Self {
        Self {
            store: JobStore {
                dir: data_dir.join("jobs"),
            },
            live: Mutex::new(HashMap::new()),
            cancel_requested: Mutex::new(HashSet::new()),
            events: broadcast::channel(EVENT_BUFFER).0,
        }
    }

    fn get(&self, id: &str) -> Result<Job, JobError> {
        self.store
            .get(id)?
            .ok_or_else(|| JobError::NotFound(id.to_owned()))
    }

    fn insert(&self, job: &Job) -> Result<(), JobError> {
        let _guard = lock();
        self.store.write(job)?;
        let _ = self.events.send(job.clone());
        Ok(())
    }

    /// Applies one transition under the lock, persists it, and announces it.
    fn mutate<T>(
        &self,
        id: &str,
        edit: impl FnOnce(&mut Job) -> Result<T, JobError>,
    ) -> Result<(Job, T), JobError> {
        let _guard = lock();
        let mut job = self
            .store
            .get(id)?
            .ok_or_else(|| JobError::NotFound(id.to_owned()))?;
        let value = edit(&mut job)?;
        self.store.write(&job)?;
        let _ = self.events.send(job.clone());
        Ok((job, value))
    }

    fn control(&self, id: &str) -> Result<RunControl> {
        self.live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned()
            .with_context(|| format!("job {id} has no live run control"))
    }

    fn register(&self, id: &str) {
        self.live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(id.to_owned(), RunControl::new());
    }

    fn unregister(&self, id: &str) {
        self.live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
        self.cancel_requested
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
    }

    /// Applies a terminal transition and drops the job's run control under the
    /// live-run lock. A retry registers its control under that same lock, so the
    /// removal can never reach a control registered after the transition.
    fn settle(
        &self,
        id: &str,
        edit: impl FnOnce(&mut Job) -> Result<(), JobError>,
    ) -> Result<(), JobError> {
        let mut live = self
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.mutate(id, edit)?;
        live.remove(id);
        self.cancel_requested
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
        Ok(())
    }

    /// Records an operator cancel and stops the live run of `id`, if it has one.
    fn request_cancel(&self, id: &str) {
        let control = self
            .live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .cloned();
        if let Some(control) = control {
            self.cancel_requested
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(id.to_owned());
            control.cancel();
        }
    }

    fn cancel_requested(&self, id: &str) -> bool {
        self.cancel_requested
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(id)
    }

    fn live_ids(&self) -> HashSet<String> {
        self.live
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect()
    }
}

/// How an attempt or resumption ended.
enum Settled {
    Succeeded(Option<String>),
    Failed {
        error: String,
        result: Option<String>,
    },
    Cancelled,
    /// Waiting on this decision; the job's run has stopped.
    Blocked(String),
}

#[derive(Clone)]
enum Attempt {
    Fresh,
    Resume,
    /// Continue the saved session in the job's last workspace with this prompt.
    Checkpoint(String),
}

impl Runtime {
    /// Queues a job and starts it in the background. Returns once the record is
    /// durable; the run continues after this call returns.
    pub fn start_job(
        &self,
        prompt: impl Into<String>,
        budget: JobBudget,
        contract: JobContract,
    ) -> Result<Job, JobError> {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return Err(JobError::Invalid("a job needs a prompt".into()));
        }
        budget.validate()?;
        contract.validate()?;
        let id = uuid::Uuid::now_v7().to_string();
        let job = Job::new(id.clone(), prompt, budget, contract);
        self.inner.jobs.insert(&job)?;
        self.inner.jobs.register(&id);
        self.spawn_job(id, Attempt::Fresh);
        Ok(job)
    }

    pub fn job(&self, id: &str) -> Result<Job, JobError> {
        self.inner.jobs.get(id)
    }

    pub fn list_jobs(&self) -> Result<Vec<Job>, JobError> {
        Ok(self.inner.jobs.store.list()?)
    }

    /// Every job state change, as it is persisted. A subscriber that falls more
    /// than the buffer behind receives a lag error and should re-list jobs.
    pub fn subscribe_jobs(&self) -> broadcast::Receiver<Job> {
        self.inner.jobs.events.subscribe()
    }

    /// Starts a new attempt for a failed or interrupted job, within its budget.
    pub fn retry_job(&self, id: &str) -> Result<Job, JobError> {
        let (job, attempt) = self.inner.jobs.mutate(id, |job| {
            if job.inbox == InboxState::Archived {
                return Err(JobError::Invalid(format!("job {id} is archived")));
            }
            if !matches!(job.status, JobStatus::Failed | JobStatus::Interrupted) {
                return Err(JobError::Invalid(format!(
                    "job {id} is {}; only failed or interrupted jobs can be retried",
                    job.status.as_str()
                )));
            }
            if job.attempts >= job.budget.max_attempts {
                return Err(JobError::Invalid(format!(
                    "job {id} has used its {} attempt(s)",
                    job.budget.max_attempts
                )));
            }
            if job.run_ms >= job.budget.wall_ms() {
                return Err(JobError::Invalid(format!(
                    "job {id} has used its wall-clock budget"
                )));
            }
            if job.budget.max_spend.is_some_and(|max| job.spend_usd >= max) {
                return Err(JobError::Invalid(format!(
                    "job {id} has used its spend budget"
                )));
            }
            // A job that saved a workspace and session continues from that
            // checkpoint; one that never got that far starts over.
            let attempt = match (&job.session_path, job.forks.is_empty()) {
                (Some(_), false) => {
                    Attempt::Checkpoint(checkpoint_prompt(job.status, job.error.as_deref()))
                }
                _ => Attempt::Fresh,
            };
            job.status = JobStatus::Queued;
            job.owner_pid = Some(std::process::id());
            job.error = None;
            job.result = None;
            job.finished_ms = None;
            Ok(attempt)
        })?;
        self.inner.jobs.register(id);
        self.spawn_job(id.to_owned(), attempt);
        Ok(job)
    }

    /// Continues a blocked job after its decision was approved.
    pub fn resume_job(&self, id: &str) -> Result<Job, JobError> {
        let decisions = self.decision_store();
        let (job, ()) = self.inner.jobs.mutate(id, |job| {
            if job.status != JobStatus::Blocked {
                return Err(JobError::Invalid(format!(
                    "job {id} is {}, not blocked",
                    job.status.as_str()
                )));
            }
            let decision_id = job
                .decision_id
                .clone()
                .ok_or_else(|| JobError::Invalid(format!("blocked job {id} has no decision")))?;
            let decision = decisions
                .get(&decision_id)?
                .ok_or_else(|| JobError::Invalid(format!("unknown decision `{decision_id}`")))?;
            if decision.status != DecisionStatus::Approved {
                return Err(JobError::Invalid(format!(
                    "decision {decision_id} is {:?}; approve it before resuming job {id}",
                    decision.status
                )));
            }
            job.status = JobStatus::Running;
            job.owner_pid = Some(std::process::id());
            job.error = None;
            Ok(())
        })?;
        self.inner.jobs.register(id);
        self.spawn_job(id.to_owned(), Attempt::Resume);
        Ok(job)
    }

    /// Resolves the decision that blocked a job. Approval resumes the job;
    /// rejection fails it without running the blocked action.
    pub fn decide_job(
        &self,
        decision_id: &str,
        approve: bool,
    ) -> Result<(Decision, Job), JobError> {
        let decisions = self.decision_store();
        let decision = decisions
            .get(decision_id)?
            .ok_or_else(|| JobError::NotFound(decision_id.to_owned()))?;
        let job_id = decision.job_id.clone().ok_or_else(|| {
            JobError::Invalid(format!("decision {decision_id} belongs to a session run"))
        })?;
        let current = self.inner.jobs.get(&job_id)?;
        if current.status != JobStatus::Blocked
            || current.decision_id.as_deref() != Some(decision_id)
        {
            return Err(JobError::Invalid(format!(
                "decision {decision_id} is not waiting on job {job_id}"
            )));
        }
        let resolved = decisions
            .resolve(decision_id, approve)
            .map_err(|error| JobError::Invalid(format!("{error:#}")))?;
        if approve {
            let job = self.resume_job(&job_id)?;
            return Ok((resolved, job));
        }
        let (job, ()) = self.inner.jobs.mutate(&job_id, |job| {
            job.status = JobStatus::Failed;
            job.error = Some(format!(
                "the operator rejected the blocked action `{}` ({})",
                resolved.tool, resolved.capability
            ));
            job.finished_ms = Some(timestamp_ms());
            job.owner_pid = None;
            job.inbox = InboxState::Unread;
            Ok(())
        })?;
        self.inner.jobs.unregister(&job_id);
        Ok((resolved, job))
    }

    /// Cancels a queued or running job, or a blocked one (rejecting its pending decision).
    pub fn cancel_job(&self, id: &str) -> Result<Job, JobError> {
        let decisions = self.decision_store();
        let (job, was_blocked) = self.inner.jobs.mutate(id, |job| match job.status {
            JobStatus::Queued | JobStatus::Running => Ok(false),
            JobStatus::Blocked => {
                job.status = JobStatus::Cancelled;
                job.finished_ms = Some(timestamp_ms());
                job.owner_pid = None;
                job.inbox = InboxState::Unread;
                Ok(true)
            }
            _ => Err(JobError::Invalid(format!(
                "job {id} is already {}",
                job.status.as_str()
            ))),
        })?;
        if was_blocked {
            if let Some(decision_id) = &job.decision_id
                && let Ok(Some(decision)) = decisions.get(decision_id)
                && decision.status == DecisionStatus::Pending
            {
                decisions.resolve(decision_id, false)?;
            }
            self.inner.jobs.unregister(id);
        } else {
            self.inner.jobs.request_cancel(id);
        }
        Ok(job)
    }

    /// Changes a job's inbox state. Archiving requires a finished job: the job is
    /// marked archived before its isolated workspaces are reclaimed, so no retry can
    /// start in a fork being torn down. The session transcript is kept.
    pub async fn set_job_inbox(&self, id: &str, state: InboxState) -> Result<Job, JobError> {
        if state == InboxState::Archived {
            let (job, ()) = self.inner.jobs.mutate(id, |job| {
                if !job.status.is_terminal() {
                    return Err(JobError::Invalid(format!(
                        "job {id} is {} and cannot be archived until it finishes",
                        job.status.as_str()
                    )));
                }
                job.inbox = InboxState::Archived;
                Ok(())
            })?;
            // A failed reclaim leaves the job archived with its remaining forks
            // listed; archiving it again resumes the reclaim.
            for fork in job.forks {
                let workspace = SubagentWorkspace {
                    path: fork.path.clone(),
                    node_id: String::new(),
                    thread: fork.thread.clone(),
                    repo_root: fork.repo_root.clone(),
                    subagent_id: fork.thread.clone(),
                };
                let reclaimed = reclaim_isolated_workspace(workspace).await;
                reclaimed
                    .cleanup
                    .map_err(|error| JobError::Failed(anyhow!("{error}")))?;
                self.inner.jobs.mutate(id, |job| {
                    job.forks.retain(|kept| kept.path != fork.path);
                    Ok(())
                })?;
            }
            return self.inner.jobs.get(id);
        }
        let (job, ()) = self.inner.jobs.mutate(id, |job| {
            if job.inbox == InboxState::Archived {
                return Err(JobError::Invalid(format!(
                    "job {id} is archived; archived jobs stay archived"
                )));
            }
            job.inbox = state;
            Ok(())
        })?;
        Ok(job)
    }

    /// Returns the finished jobs the main agent has not yet been shown, and
    /// marks each one drained so it is returned once. Drain state is separate
    /// from the inbox: reading, archiving, or draining one job does not change
    /// the others.
    pub fn drain_job_results(&self) -> Result<Vec<Job>, JobError> {
        let mut drained = Vec::new();
        for job in self.inner.jobs.store.list()? {
            if !job.status.is_terminal() || job.drained {
                continue;
            }
            let (job, fresh) = self.inner.jobs.mutate(&job.id, |job| {
                if job.drained {
                    return Ok(false);
                }
                job.drained = true;
                Ok(true)
            })?;
            if fresh {
                drained.push(job);
            }
        }
        Ok(drained)
    }

    /// Archives a job that succeeded under a verified no-change contract. Its
    /// result needs no operator attention, and archiving reclaims its workspace.
    async fn archive_verified_no_change(&self, id: &str) -> Result<(), JobError> {
        let job = self.inner.jobs.get(id)?;
        if job.status == JobStatus::Succeeded && job.contract.no_change {
            self.set_job_inbox(id, InboxState::Archived).await?;
        }
        Ok(())
    }

    /// Settles jobs left queued or running by a runtime that stopped. A job
    /// owned by another live process is left alone. Returns how many were
    /// interrupted.
    pub fn recover_jobs(&self) -> Result<usize, JobError> {
        let own = std::process::id();
        let live = self.inner.jobs.live_ids();
        let mut interrupted = 0;
        for job in self.inner.jobs.store.list()? {
            if !matches!(job.status, JobStatus::Queued | JobStatus::Running)
                || live.contains(&job.id)
            {
                continue;
            }
            if job
                .owner_pid
                .is_some_and(|pid| pid != own && process_alive(pid))
            {
                continue;
            }
            self.inner.jobs.mutate(&job.id, |job| {
                job.status = JobStatus::Interrupted;
                job.error = Some(
                    "the runtime stopped before this job finished; retry it to run it again"
                        .to_owned(),
                );
                job.owner_pid = None;
                job.finished_ms = Some(timestamp_ms());
                job.inbox = InboxState::Unread;
                Ok(())
            })?;
            interrupted += 1;
        }
        Ok(interrupted)
    }

    fn spawn_job(&self, id: String, attempt: Attempt) {
        let runtime = self.clone();
        tokio::spawn(async move {
            let result = match attempt {
                Attempt::Fresh => runtime.run_fresh(&id).await,
                Attempt::Resume => runtime.run_resumed(&id).await,
                Attempt::Checkpoint(prompt) => runtime.run_checkpoint(&id, prompt).await,
            };
            match result {
                Ok(()) => {
                    if let Err(error) = runtime.archive_verified_no_change(&id).await {
                        eprintln!(
                            "juto: job {id}: could not archive its verified no-change result: {error}"
                        );
                    }
                }
                Err(error) => {
                    // The run could not start or record its progress. Fail the job
                    // rather than leave it looking like it is still running.
                    let settled = Settled::Failed {
                        error: format!("{error:#}"),
                        result: None,
                    };
                    if let Err(record) = runtime.finish(&id, settled) {
                        eprintln!("juto: job {id}: could not record its failure: {record}");
                    }
                }
            }
        });
    }

    async fn run_fresh(&self, id: &str) -> Result<()> {
        let job = self.inner.jobs.get(id)?;
        let control = self.inner.jobs.control(id)?;
        if self.inner.jobs.cancel_requested(id) {
            return Ok(self.finish(id, Settled::Cancelled)?);
        }
        let fork_id = format!("{id}-{}", job.attempts + 1);
        self.inner.jobs.mutate(id, |job| {
            job.status = JobStatus::Running;
            job.attempts += 1;
            job.turns = 0;
            job.started_ms = Some(timestamp_ms());
            job.finished_ms = None;
            job.error = None;
            job.result = None;
            Ok(())
        })?;
        let factory = ChildFactory {
            runtime: self.clone(),
            depth: 0,
            inbox: MainInbox::default(),
            unattended: None,
            meter: Arc::new(JobMeter::new(JobLimits::default())),
        };
        let (runtime, workspace) = factory
            .isolated_runtime(&fork_id)
            .await
            .context("provisioning the job's isolated workspace")?;
        let fork = JobFork {
            path: workspace.path.clone(),
            repo_root: workspace.repo_root.clone(),
            thread: workspace.thread.clone(),
        };
        self.inner.jobs.mutate(id, |job| {
            job.forks.push(fork);
            Ok(())
        })?;
        let mut session = Session::create(&self.inner.data_dir.join("sessions"), &workspace.path)?;
        let session_id = session.header().id.clone();
        let session_path = session.path().to_path_buf();
        self.inner.jobs.mutate(id, |job| {
            job.session_id = Some(session_id);
            job.session_path = Some(session_path);
            Ok(())
        })?;
        self.segment(id, &runtime, &mut session, job.prompt, control)
            .await
    }

    async fn run_resumed(&self, id: &str) -> Result<()> {
        let job = self.inner.jobs.get(id)?;
        let control = self.inner.jobs.control(id)?;
        if self.inner.jobs.cancel_requested(id) {
            return Ok(self.finish(id, Settled::Cancelled)?);
        }
        let decision_id = job
            .decision_id
            .clone()
            .context("blocked job has no decision")?;
        let decision = self
            .decision_store()
            .get(&decision_id)?
            .with_context(|| format!("unknown decision `{decision_id}`"))?;
        let fork = job
            .forks
            .last()
            .context("job has no workspace to resume in")?;
        let session_path = job
            .session_path
            .clone()
            .context("job has no session to resume")?;
        let factory = ChildFactory {
            runtime: self.clone(),
            depth: 0,
            inbox: MainInbox::default(),
            unattended: None,
            meter: Arc::new(JobMeter::new(JobLimits::default())),
        };
        let runtime = factory.runtime_at(fork.path.clone()).await?;
        let mut session = Session::open(&session_path)?;
        self.segment(
            id,
            &runtime,
            &mut session,
            continuation_prompt(&decision),
            control,
        )
        .await
    }

    /// Continues a saved session in the job's last workspace after an interrupt
    /// or failure. The in-flight tool call, if any, is recorded as interrupted
    /// and is not run again.
    async fn run_checkpoint(&self, id: &str, prompt: String) -> Result<()> {
        let job = self.inner.jobs.get(id)?;
        let control = self.inner.jobs.control(id)?;
        if self.inner.jobs.cancel_requested(id) {
            return Ok(self.finish(id, Settled::Cancelled)?);
        }
        let fork = job
            .forks
            .last()
            .context("job has no workspace to continue in")?;
        let session_path = job
            .session_path
            .clone()
            .context("job has no session to continue")?;
        self.inner.jobs.mutate(id, |job| {
            job.status = JobStatus::Running;
            job.attempts += 1;
            job.turns = 0;
            job.finished_ms = None;
            job.error = None;
            job.result = None;
            Ok(())
        })?;
        let factory = ChildFactory {
            runtime: self.clone(),
            depth: 0,
            inbox: MainInbox::default(),
            unattended: None,
            meter: Arc::new(JobMeter::new(JobLimits::default())),
        };
        let runtime = factory.runtime_at(fork.path.clone()).await?;
        let mut session = Session::open(&session_path)?;
        session.append_interrupted(
            id,
            "the run ended before this tool call finished; it was not re-run",
        )?;
        self.segment(id, &runtime, &mut session, prompt, control)
            .await
    }

    /// Runs one stretch of a job's session inside its remaining wall, turn and
    /// spend budgets, and records how it ended.
    async fn segment(
        &self,
        id: &str,
        runtime: &Runtime,
        session: &mut Session,
        prompt: String,
        control: RunControl,
    ) -> Result<()> {
        let job = self.inner.jobs.get(id)?;
        let remaining = job.budget.wall_ms().saturating_sub(job.run_ms);
        if remaining == 0 {
            let error = budget_message(job.budget);
            return Ok(self.finish(
                id,
                Settled::Failed {
                    error,
                    result: last_assistant_text(session),
                },
            )?);
        }
        if job.budget.max_spend.is_some_and(|max| job.spend_usd >= max) {
            return Ok(self.finish(
                id,
                Settled::Failed {
                    error: spend_message(job.budget),
                    result: last_assistant_text(session),
                },
            )?);
        }
        // The turns this attempt may still take. The runtime enforces the cap, so
        // the model never gets the turn beyond it.
        let turn_limit = job
            .budget
            .max_turns
            .map(|max| usize::try_from(max.saturating_sub(job.turns)).unwrap_or(usize::MAX));
        // The job's cap is the one that stopped a run only when it is no looser
        // than the runtime's own limit.
        let job_turns_bind = turn_limit
            .is_some_and(|limit| runtime.config().max_turns.is_none_or(|own| own >= limit));
        let (events, mut received) = mpsc::unbounded_channel::<AgentEvent>();
        // The run's meter does the accounting. This only drains the events, since
        // a closed channel would end the run as aborted.
        let drain = tokio::spawn(async move { while received.recv().await.is_some() {} });
        let started = Instant::now();
        // The budget cancels the run, but the run is awaited rather than dropped:
        // it settles its children before it returns, so the job cannot settle while
        // one of them still runs.
        let (run, timed_out) = {
            let mut run = std::pin::pin!(
                runtime.run_unattended(
                    session,
                    prompt,
                    events,
                    control.clone(),
                    Some(id.to_owned()),
                    JobLimits {
                        turns: turn_limit,
                        spend: job
                            .budget
                            .max_spend
                            .map(|max| (max - job.spend_usd).max(0.0)),
                    },
                )
            );
            match tokio::time::timeout(Duration::from_millis(remaining), run.as_mut()).await {
                Ok(run) => (run, false),
                Err(_elapsed) => {
                    control.cancel();
                    (run.await, true)
                }
            }
        };
        let spent = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        drain.abort();
        let UnattendedRun {
            outcome,
            children,
            spent: segment,
        } = run;
        self.inner.jobs.mutate(id, |job| {
            job.run_ms = job.run_ms.saturating_add(spent);
            job.turns = job
                .turns
                .saturating_add(u32::try_from(segment.turns).unwrap_or(u32::MAX));
            job.spend_usd += segment.cost;
            Ok(())
        })?;
        // The run's meter refuses any turn the budget cannot pay for, so
        // `exhausted` names the budget that stopped it. Spend is also checked
        // against the job's total, since the run only sees this segment's share.
        let turns_over = segment.exhausted == Some(Exhausted::Turns);
        let spend_over = segment.exhausted == Some(Exhausted::Spend)
            || job
                .budget
                .max_spend
                .is_some_and(|max| job.spend_usd + segment.cost >= max);

        let cancelled = self.inner.jobs.cancel_requested(id);
        let main = last_assistant_text(session);
        let stopped = matches!(
            &outcome,
            Ok(UnattendedOutcome::Finished(finished))
                if matches!(finished.stop_reason, StopReason::Stop)
        );
        let result = job_result(main.as_deref(), &children);
        let verdict = if stopped && !cancelled && !timed_out && !spend_over && !turns_over {
            // The contract reads the final message alone, so a silent final turn
            // cannot pass on an earlier turn's text.
            Some(
                self.verify_contract(&job, final_assistant_text(session).as_deref())
                    .await,
            )
        } else {
            None
        };
        let settled = if cancelled {
            if let Ok(UnattendedOutcome::Blocked(decision)) = &outcome {
                self.reject_pending(decision);
            }
            Settled::Cancelled
        } else if timed_out {
            Settled::Failed {
                error: budget_message(job.budget),
                result,
            }
        } else if spend_over {
            Settled::Failed {
                error: spend_message(job.budget),
                result,
            }
        } else if turns_over {
            Settled::Failed {
                error: turns_message(job.budget),
                result,
            }
        } else {
            match outcome {
                Err(error) => Settled::Failed {
                    error: run_error(&error, job.budget, job_turns_bind),
                    result,
                },
                Ok(UnattendedOutcome::Blocked(decision)) => Settled::Blocked(decision.id),
                Ok(UnattendedOutcome::Finished(run)) => match run.stop_reason {
                    StopReason::Stop => match verdict.unwrap_or(Ok(())) {
                        Ok(()) => Settled::Succeeded(result),
                        Err(reason) => Settled::Failed {
                            error: format!("completion contract not met: {reason}"),
                            result,
                        },
                    },
                    other => Settled::Failed {
                        error: format!("the run stopped with {other:?}"),
                        result,
                    },
                },
            }
        };
        Ok(self.finish(id, settled)?)
    }

    /// Checks a finished attempt against the job's contract. Verification reads
    /// the workspace, so it runs off the async runtime.
    async fn verify_contract(&self, job: &Job, result: Option<&str>) -> Result<(), String> {
        let fork = job
            .forks
            .last()
            .map(|fork| fork.path.clone())
            .ok_or_else(|| "the job has no workspace to verify".to_owned())?;
        let contract = job.contract.clone();
        let result = result.map(str::to_owned);
        tokio::task::spawn_blocking(move || check_contract(&contract, &fork, result.as_deref()))
            .await
            .map_err(|error| format!("verification did not finish: {error}"))?
    }

    fn reject_pending(&self, decision: &Decision) {
        let decisions = self.decision_store();
        if let Err(error) = decisions.resolve(&decision.id, false) {
            eprintln!(
                "juto: could not reject pending decision {} of a cancelled job: {error:#}",
                decision.id
            );
        }
    }

    fn finish(&self, id: &str, settled: Settled) -> Result<(), JobError> {
        let (status, error, result, decision_id) = match settled {
            Settled::Succeeded(result) => (JobStatus::Succeeded, None, result, None),
            Settled::Failed { error, result } => (JobStatus::Failed, Some(error), result, None),
            Settled::Cancelled => (JobStatus::Cancelled, None, None, None),
            Settled::Blocked(decision_id) => (JobStatus::Blocked, None, None, Some(decision_id)),
        };
        let blocked = status == JobStatus::Blocked;
        self.inner.jobs.settle(id, |job| {
            job.status = status;
            job.error = error;
            job.result = result;
            if let Some(decision_id) = decision_id {
                job.decision_id = Some(decision_id);
            }
            job.owner_pid = None;
            job.finished_ms = (!blocked).then(timestamp_ms);
            job.inbox = InboxState::Unread;
            job.drained = false;
            Ok(())
        })?;
        Ok(())
    }
}

/// Verifies a finished attempt against its contract. The first clause that
/// fails names the reason.
fn check_contract(contract: &JobContract, fork: &Path, result: Option<&str>) -> Result<(), String> {
    if let Some(path) = &contract.artifact {
        let resolved = fork
            .join(path)
            .canonicalize()
            .ok()
            .filter(|resolved| {
                fork.canonicalize()
                    .is_ok_and(|root| resolved.starts_with(root))
            })
            .ok_or_else(|| format!("artifact `{path}` was not written"))?;
        let metadata =
            fs::metadata(&resolved).map_err(|error| format!("artifact `{path}`: {error}"))?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(format!("artifact `{path}` is empty or not a file"));
        }
    }
    if let Some(schema) = &contract.output_schema {
        let text =
            result.ok_or("the final answer is empty, so it cannot satisfy the output schema")?;
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| format!("the final answer is not JSON: {error}"))?;
        let validator = jsonschema::validator_for(schema)
            .map_err(|error| format!("invalid output_schema: {error}"))?;
        validator.validate(&value).map_err(|error| {
            format!("the final answer does not match the output schema: {error}")
        })?;
    }
    if contract.no_change {
        let changed = ChangeGraph::open(fork)
            .and_then(|graph| graph.paths_changed_since_base())
            .map_err(|error| format!("could not inspect the workspace: {error:#}"))?;
        if !changed.is_empty() {
            return Err(format!(
                "the job changed {} path(s) but its contract requires no change",
                changed.len()
            ));
        }
    }
    Ok(())
}

/// The prompt that continues a saved session. An interrupted run's in-flight
/// tool call was recorded as interrupted and is not re-run; a failed attempt is
/// told why it failed so it can fix that.
fn checkpoint_prompt(status: JobStatus, error: Option<&str>) -> String {
    match status {
        JobStatus::Interrupted => "The runtime stopped before this job finished. A tool call that was in flight was recorded as interrupted and was not re-run. Check whether its effect landed, then continue the task.".to_owned(),
        _ => format!(
            "The previous attempt did not finish: {}. Continue the task and fix that.",
            error.unwrap_or("it ended without a result")
        ),
    }
}

fn budget_message(budget: JobBudget) -> String {
    format!("wall-clock budget of {}s exhausted", budget.wall_secs)
}

fn turns_message(budget: JobBudget) -> String {
    format!(
        "turn budget exhausted: the job allows {} model turns per attempt",
        budget.max_turns.unwrap_or_default()
    )
}

fn spend_message(budget: JobBudget) -> String {
    format!(
        "spend budget of {:.2} USD exhausted",
        budget.max_spend.unwrap_or_default()
    )
}

/// The error a failed run reports. A run stopped by the job's own turn budget
/// names that budget rather than the runtime's limit.
fn run_error(error: &anyhow::Error, budget: JobBudget, job_turns_bind: bool) -> String {
    match error.downcast_ref::<AgentError>() {
        Some(AgentError::TurnLimit(_)) if job_turns_bind => format!(
            "turn budget exhausted: the job allows {} model turns per attempt",
            budget.max_turns.unwrap_or_default()
        ),
        Some(AgentError::SpendLimit(_)) => spend_message(budget),
        _ => format!("{error:#}"),
    }
}

/// The job's result: the main run's final text, then what each child it started
/// reported. The contract is checked against the main text alone.
fn job_result(main: Option<&str>, children: &[ChildOutcome]) -> Option<String> {
    let mut sections: Vec<String> = main.map(str::to_owned).into_iter().collect();
    for child in children {
        let name = child.name.as_deref().unwrap_or(&child.id);
        let (label, body) = match child.status {
            SubagentStatus::Completed => ("completed", child.output.as_deref()),
            SubagentStatus::Failed => ("failed", child.error.as_deref()),
            _ => ("cancelled", child.error.as_deref()),
        };
        sections.push(format!(
            "[child {name}: {label}]\n{}",
            body.unwrap_or_default()
        ));
    }
    (!sections.is_empty()).then(|| sections.join("\n\n"))
}

/// The assistant's latest text, skipping assistant turns that only called tools.
fn last_assistant_text(session: &Session) -> Option<String> {
    let messages = session.messages().ok()?;
    messages.iter().rev().find_map(|message| match message {
        Message::Assistant(assistant) => assistant_text(assistant),
        _ => None,
    })
}

/// The text of the session's final message, only when that message is an
/// assistant turn with text. An empty final turn yields nothing, so an earlier
/// turn's text cannot stand in for it.
fn final_assistant_text(session: &Session) -> Option<String> {
    let messages = session.messages().ok()?;
    match messages.last()? {
        Message::Assistant(assistant) => assistant_text(assistant),
        _ => None,
    }
}

/// The text blocks of one assistant turn, or nothing when it has none.
fn assistant_text(assistant: &AssistantMessage) -> Option<String> {
    let text: String = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    (!text.is_empty()).then_some(text)
}

/// Linux procfs liveness. Platforms without `/proc` treat every owner as gone,
/// so their in-flight jobs are interrupted on recovery.
fn process_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

#[cfg(test)]
mod tests;
