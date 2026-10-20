//! Scheduled automations: a stored prompt plus the settings of a one-off job,
//! started on a fixed-interval schedule. An occurrence is not a new kind of work.
//! It starts the same durable job a chat would (see `jobs.rs`) through
//! `start_job_spec`, with the same budget, completion contract and inbox.
//!
//! # Schedule and time zones
//! Occurrences are fixed UTC intervals. The `k`-th occurrence (k >= 1) falls at
//! `created_ms + k * every_secs * 1000` Unix milliseconds. There is no time zone
//! and no daylight-saving adjustment: an hour is always 3,600,000 ms, so hourly
//! occurrences stay exactly an hour apart across a DST change. Local wall-clock
//! schedules ("09:00 every day") are not supported.
//!
//! # Ticks, restarts and missed occurrences
//! The daemon calls [`Runtime::tick_automations`] about once a second. A tick
//! starts at most one job per automation: the latest occurrence due at `now`.
//! Earlier occurrences that were due but undecided (the daemon was down, or a
//! queued occurrence waited) are never replayed. They are recorded as one
//! `missed` event. A daemon that was down for a day runs the automation once when
//! it comes back, not 1440 times.
//!
//! # Overlap
//! An automation's job is open while it is queued, running or blocked on a
//! decision. When an occurrence is due while a job is open:
//! - `skip`: the occurrence is recorded as `skipped` and the schedule moves on;
//! - `queue`: the schedule holds. After the open job settles, the next tick starts
//!   one job for the latest due occurrence and records the rest as `missed`. At
//!   most one occurrence waits.
//!
//! A run started by hand while a job is open is refused under either policy.
//!
//! # Dedupe
//! A scheduled occurrence's job id is `auto-<automation id>-<instant ms>`, and job
//! creation refuses an id that already exists. An occurrence starts at most one job,
//! even when a crash leaves the schedule cursor behind the job record.
//!
//! # Pause
//! A paused automation never fires. Occurrences that fall during a pause are not
//! replayed. Enabling it records them as `missed` and schedules the next occurrence
//! after the enable time.
//!
//! # Persistence
//! `<data_dir>/automations/<id>.json` holds the definition and schedule cursor,
//! written atomically. `<id>.events.jsonl` is the append-only event log. Deleting an
//! automation removes both files; its jobs stay in the job store.
//!
//! Tick, enable, pause and run-now serialize through one lock, so a pause cannot be
//! overwritten by a concurrent tick.
//!
//! # Tier and permission profile
//! Each automation stores an execution tier and a permission profile, and every
//! occurrence runs under them. The tier is `untrusted` or `workspace`, the most
//! permissive tier an unattended run may use: `attended` and `full` are refused
//! at create and again when a job starts. The profile is `unattended` (the
//! default) or `read_only`. The daemon's configured tier still caps a run, which
//! then uses the lower of the two.

use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::Runtime;
use crate::broker::Priority;
use crate::jobs::{Job, JobBudget, JobContract, JobError, JobOrigin, JobSpec};
use crate::tier::{
    ExecutionTier, PermissionProfile, check_unattended_profile, check_unattended_tier,
};

/// Longest accepted interval: 366 days. It keeps every occurrence far inside `u64`.
pub const MAX_EVERY_SECS: u64 = 366 * 24 * 60 * 60;

/// What an occurrence does when the automation's previous job is still open.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Overlap {
    #[default]
    Skip,
    Queue,
}

/// A persisted automation. Each occurrence starts a job under the stored tier and
/// permission profile, capped by the daemon's configured tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Automation {
    pub id: String,
    /// The workspace the daemon serves. Every occurrence runs there.
    pub project: PathBuf,
    pub prompt: String,
    /// Model selector pinned on each job's session.
    pub model: Option<String>,
    /// Broker priority of each job's main run: `scheduled` or `background`.
    pub priority: Priority,
    /// Execution tier of each job's main run: `untrusted` or `workspace`.
    pub tier: ExecutionTier,
    /// Tool profile of each job's main run.
    pub permission: PermissionProfile,
    pub budget: JobBudget,
    pub contract: JobContract,
    pub every_secs: u64,
    pub overlap: Overlap,
    /// A paused automation has `enabled: false` and never fires.
    pub enabled: bool,
    pub created_ms: u64,
    /// The first occurrence not yet decided: started, skipped or recorded as missed.
    pub next_instant_ms: u64,
}

impl Automation {
    fn every_ms(&self) -> u64 {
        self.every_secs.saturating_mul(1000)
    }
}

/// What a client asks for. [`Runtime::create_automation`] validates it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutomationSpec {
    pub prompt: String,
    pub model: Option<String>,
    /// `scheduled` (the default) or `background`.
    pub priority: Option<Priority>,
    pub budget: JobBudget,
    pub contract: JobContract,
    pub every_secs: u64,
    /// `skip` (the default) or `queue`.
    pub overlap: Option<Overlap>,
    /// `workspace` (the default) or `untrusted`. Stricter tiers are refused.
    pub tier: Option<ExecutionTier>,
    /// `unattended` (the default) or `read_only`.
    pub permission: Option<PermissionProfile>,
}

/// One entry in an automation's event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "event",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum AutomationEvent {
    /// A scheduled occurrence started its job (or found the job already started).
    Fired {
        at_ms: u64,
        instant_ms: u64,
        job_id: String,
    },
    /// A run requested by hand started its job.
    RunNow { at_ms: u64, job_id: String },
    /// Due occurrences that were neither replayed nor started: earlier occurrences
    /// coalesced into a fired one, or the stretch passed while paused.
    Missed {
        at_ms: u64,
        from_ms: u64,
        count: u64,
    },
    /// A due occurrence skipped because the previous job was still open.
    Skipped {
        at_ms: u64,
        instant_ms: u64,
        count: u64,
    },
    /// Starting the occurrence's job failed. The occurrence is not retried.
    Failed {
        at_ms: u64,
        instant_ms: u64,
        error: String,
    },
}

/// The run history of one automation: the jobs it started and its event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutomationHistory {
    /// Jobs this automation started, oldest first.
    pub jobs: Vec<Job>,
    /// The event log, oldest first.
    pub events: Vec<AutomationEvent>,
}

#[derive(Debug)]
pub enum AutomationError {
    NotFound(String),
    Invalid(String),
    Failed(anyhow::Error),
}

impl std::fmt::Display for AutomationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "unknown automation `{id}`"),
            Self::Invalid(message) => f.write_str(message),
            Self::Failed(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for AutomationError {}

impl From<anyhow::Error> for AutomationError {
    fn from(error: anyhow::Error) -> Self {
        Self::Failed(error)
    }
}

impl From<std::io::Error> for AutomationError {
    fn from(error: std::io::Error) -> Self {
        Self::Failed(error.into())
    }
}

impl From<JobError> for AutomationError {
    fn from(error: JobError) -> Self {
        match error {
            JobError::Failed(error) => Self::Failed(error),
            other => Self::Invalid(other.to_string()),
        }
    }
}

/// Serializes automation writes so a tick and a pause cannot overwrite each other.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The latest occurrence at or before `now_ms` on the grid
/// `created_ms + k * every_ms`. Requires `now_ms >= created_ms`.
fn latest_due(created_ms: u64, every_ms: u64, now_ms: u64) -> u64 {
    created_ms + (now_ms - created_ms) / every_ms * every_ms
}

fn occurrence_job_id(automation_id: &str, instant_ms: u64) -> String {
    format!("auto-{automation_id}-{instant_ms}")
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn record_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.json"))
}

fn events_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.events.jsonl"))
}

fn load(dir: &Path, id: &str) -> Result<Automation, AutomationError> {
    if !valid_id(id) {
        return Err(AutomationError::NotFound(id.to_owned()));
    }
    match std::fs::read_to_string(record_path(dir, id)) {
        Ok(text) => {
            let automation: Automation = serde_json::from_str(&text).map_err(|error| {
                AutomationError::Failed(
                    anyhow::Error::from(error).context(format!("automation `{id}` is corrupt")),
                )
            })?;
            if !(1..=MAX_EVERY_SECS).contains(&automation.every_secs) {
                return Err(AutomationError::Failed(anyhow::anyhow!(
                    "automation `{id}` is corrupt: every_secs must be between 1 and {MAX_EVERY_SECS}"
                )));
            }
            Ok(automation)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            Err(AutomationError::NotFound(id.to_owned()))
        }
        Err(error) => Err(error.into()),
    }
}

fn write_record(dir: &Path, automation: &Automation) -> Result<()> {
    std::fs::create_dir_all(dir).context("create automations directory")?;
    let path = record_path(dir, &automation.id);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(automation)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn append_event(dir: &Path, id: &str, event: &AutomationEvent) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut line = serde_json::to_vec(event)?;
    line.push(b'\n');
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(events_path(dir, id))?
        .write_all(&line)?;
    Ok(())
}

fn read_events(dir: &Path, id: &str) -> Result<Vec<AutomationEvent>> {
    let text = match std::fs::read_to_string(events_path(dir, id)) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    text.lines()
        .map(|line| serde_json::from_str(line).context("parse automation event"))
        .collect()
}

/// One stored record: its id, and what loading it produced.
type StoredRecord = (String, Result<Automation, AutomationError>);

impl Runtime {
    fn automations_dir(&self) -> PathBuf {
        self.inner.data_dir.join("automations")
    }

    /// Creates an automation. Its first occurrence is one interval from now.
    pub fn create_automation(&self, spec: AutomationSpec) -> Result<Automation, AutomationError> {
        self.create_automation_at(spec, juto_ai::timestamp_ms())
    }

    pub(crate) fn create_automation_at(
        &self,
        spec: AutomationSpec,
        now_ms: u64,
    ) -> Result<Automation, AutomationError> {
        let AutomationSpec {
            prompt,
            model,
            priority,
            budget,
            contract,
            every_secs,
            overlap,
            tier,
            permission,
        } = spec;
        if prompt.trim().is_empty() {
            return Err(AutomationError::Invalid(
                "an automation needs a prompt".into(),
            ));
        }
        if every_secs == 0 || every_secs > MAX_EVERY_SECS {
            return Err(AutomationError::Invalid(format!(
                "every_secs must be between 1 and {MAX_EVERY_SECS}"
            )));
        }
        let priority = priority.unwrap_or(Priority::Scheduled);
        if !matches!(priority, Priority::Scheduled | Priority::Background) {
            return Err(AutomationError::Invalid(
                "an automation runs as `scheduled` or `background` work".into(),
            ));
        }
        let tier = tier.unwrap_or(ExecutionTier::Workspace);
        check_unattended_tier(tier).map_err(AutomationError::Invalid)?;
        let permission = permission.unwrap_or_default();
        check_unattended_profile(permission).map_err(AutomationError::Invalid)?;
        budget.validate()?;
        contract.validate()?;
        if let Some(selector) = &model {
            self.resolve_model(selector).map_err(|error| {
                AutomationError::Invalid(format!("model `{selector}`: {error:#}"))
            })?;
        }
        let _guard = lock();
        let automation = Automation {
            id: uuid::Uuid::now_v7().to_string(),
            project: self.cwd().to_path_buf(),
            prompt,
            model,
            priority,
            tier,
            permission,
            budget,
            contract,
            every_secs,
            overlap: overlap.unwrap_or_default(),
            enabled: true,
            created_ms: now_ms,
            next_instant_ms: now_ms + every_secs * 1000,
        };
        write_record(&self.automations_dir(), &automation)?;
        Ok(automation)
    }

    pub fn list_automations(&self) -> Result<Vec<Automation>, AutomationError> {
        self.read_records()?
            .into_iter()
            .map(|(_, record)| record)
            .collect()
    }

    /// Every stored record with its id, one result each, so a record that cannot
    /// be read does not hide the others from the scheduler.
    fn read_records(&self) -> Result<Vec<StoredRecord>, AutomationError> {
        let dir = self.automations_dir();
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut records = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path.extension() != Some(OsStr::new("json")) {
                continue;
            }
            if let Some(id) = path.file_stem().and_then(OsStr::to_str) {
                records.push((id.to_owned(), load(&dir, id)));
            }
        }
        records.sort_by_key(|(id, record)| {
            (
                record
                    .as_ref()
                    .map_or(0, |automation| automation.created_ms),
                id.clone(),
            )
        });
        Ok(records)
    }

    pub fn automation(&self, id: &str) -> Result<Automation, AutomationError> {
        load(&self.automations_dir(), id)
    }

    /// Pauses or enables an automation. Enabling skips the occurrences that fell
    /// due during the pause and records them as one `missed` event.
    pub fn set_automation_enabled(
        &self,
        id: &str,
        enabled: bool,
    ) -> Result<Automation, AutomationError> {
        self.set_enabled_at(id, enabled, juto_ai::timestamp_ms())
    }

    pub(crate) fn set_enabled_at(
        &self,
        id: &str,
        enabled: bool,
        now_ms: u64,
    ) -> Result<Automation, AutomationError> {
        let _guard = lock();
        let dir = self.automations_dir();
        let mut automation = load(&dir, id)?;
        if automation.enabled == enabled {
            return Ok(automation);
        }
        if enabled && automation.next_instant_ms <= now_ms {
            let every = automation.every_ms();
            let latest = latest_due(automation.created_ms, every, now_ms);
            let count = (latest - automation.next_instant_ms) / every + 1;
            append_event(
                &dir,
                id,
                &AutomationEvent::Missed {
                    at_ms: now_ms,
                    from_ms: automation.next_instant_ms,
                    count,
                },
            )?;
            automation.next_instant_ms = latest + every;
        }
        automation.enabled = enabled;
        write_record(&dir, &automation)?;
        Ok(automation)
    }

    /// Removes the definition and its event log. Jobs it started stay in the job
    /// store.
    pub fn delete_automation(&self, id: &str) -> Result<(), AutomationError> {
        let _guard = lock();
        let dir = self.automations_dir();
        load(&dir, id)?;
        std::fs::remove_file(record_path(&dir, id))?;
        match std::fs::remove_file(events_path(&dir, id)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    /// Starts one job for `id` now, outside its schedule. The schedule cursor does
    /// not move.
    pub fn run_automation_now(&self, id: &str) -> Result<Job, AutomationError> {
        self.run_now_at(id, juto_ai::timestamp_ms())
    }

    pub(crate) fn run_now_at(&self, id: &str, now_ms: u64) -> Result<Job, AutomationError> {
        let _guard = lock();
        let dir = self.automations_dir();
        let automation = load(&dir, id)?;
        if automation.project != self.cwd().to_path_buf() {
            return Err(AutomationError::Invalid(format!(
                "automation `{id}` belongs to {}, not this daemon's project {}",
                automation.project.display(),
                self.cwd().display()
            )));
        }
        if self.open_job(id)?.is_some() {
            return Err(AutomationError::Invalid(format!(
                "automation `{id}` has a job still open; run it again once that job settles"
            )));
        }
        let job = self.start_job_spec(JobSpec {
            id: None,
            prompt: automation.prompt,
            budget: automation.budget,
            contract: automation.contract,
            model: automation.model,
            priority: Some(automation.priority),
            tier: Some(automation.tier),
            permission: Some(automation.permission),
            origin: Some(JobOrigin {
                automation_id: id.to_owned(),
                instant_ms: now_ms,
            }),
        })?;
        append_event(
            &dir,
            id,
            &AutomationEvent::RunNow {
                at_ms: now_ms,
                job_id: job.id.clone(),
            },
        )?;
        Ok(job)
    }

    pub fn automation_history(&self, id: &str) -> Result<AutomationHistory, AutomationError> {
        let dir = self.automations_dir();
        load(&dir, id)?;
        let mut jobs: Vec<Job> = self
            .list_jobs()?
            .into_iter()
            .filter(|job| {
                job.origin
                    .as_ref()
                    .is_some_and(|origin| origin.automation_id == id)
            })
            .collect();
        jobs.sort_by(|a, b| (a.created_ms, &a.id).cmp(&(b.created_ms, &b.id)));
        Ok(AutomationHistory {
            jobs,
            events: read_events(&dir, id)?,
        })
    }

    /// Starts the job for each enabled automation whose latest due occurrence is at
    /// or before `now_ms`, and returns the jobs started. Safe to call repeatedly:
    /// a second call for the same instant starts nothing.
    pub fn tick_automations(&self, now_ms: u64) -> Result<Vec<Job>, AutomationError> {
        let _guard = lock();
        let dir = self.automations_dir();
        let mut started = Vec::new();
        for (id, record) in self.read_records()? {
            let outcome =
                record.and_then(|mut automation| self.evaluate(&dir, &mut automation, now_ms));
            match outcome {
                Ok(Some(job)) => started.push(job),
                Ok(None) => {}
                // One bad record must not stop the others from firing.
                Err(error) => eprintln!("juto: automation {id}: {error}"),
            }
        }
        Ok(started)
    }

    /// The job of this automation that has not settled yet, if any.
    fn open_job(&self, automation_id: &str) -> Result<Option<Job>, AutomationError> {
        Ok(self.list_jobs()?.into_iter().find(|job| {
            !job.status.is_terminal()
                && job
                    .origin
                    .as_ref()
                    .is_some_and(|origin| origin.automation_id == automation_id)
        }))
    }

    /// Decides the occurrences of one automation due at `now_ms`. Must run under
    /// `LOCK`.
    fn evaluate(
        &self,
        dir: &Path,
        automation: &mut Automation,
        now_ms: u64,
    ) -> Result<Option<Job>, AutomationError> {
        if automation.project != self.cwd().to_path_buf() {
            // Occurrences run in the automation's own project; another daemon's
            // automations are not this daemon's to start.
            return Ok(None);
        }
        if !automation.enabled || now_ms < automation.next_instant_ms {
            return Ok(None);
        }
        let every = automation.every_ms();
        let latest = latest_due(automation.created_ms, every, now_ms);
        let undecided_from = automation.next_instant_ms;
        let due = (latest - undecided_from) / every + 1;
        let id = automation.id.clone();

        if self.open_job(&id)?.is_some() {
            if automation.overlap == Overlap::Queue {
                return Ok(None);
            }
            automation.next_instant_ms = latest + every;
            write_record(dir, automation)?;
            append_event(
                dir,
                &id,
                &AutomationEvent::Skipped {
                    at_ms: now_ms,
                    instant_ms: latest,
                    count: due,
                },
            )?;
            return Ok(None);
        }

        let job_id = occurrence_job_id(&id, latest);
        let outcome = self.start_job_spec(JobSpec {
            id: Some(job_id.clone()),
            prompt: automation.prompt.clone(),
            budget: automation.budget,
            contract: automation.contract.clone(),
            model: automation.model.clone(),
            priority: Some(automation.priority),
            tier: Some(automation.tier),
            permission: Some(automation.permission),
            origin: Some(JobOrigin {
                automation_id: id.clone(),
                instant_ms: latest,
            }),
        });
        automation.next_instant_ms = latest + every;
        write_record(dir, automation)?;
        if due > 1 {
            append_event(
                dir,
                &id,
                &AutomationEvent::Missed {
                    at_ms: now_ms,
                    from_ms: undecided_from,
                    count: due - 1,
                },
            )?;
        }
        match outcome {
            Ok(job) => {
                append_event(
                    dir,
                    &id,
                    &AutomationEvent::Fired {
                        at_ms: now_ms,
                        instant_ms: latest,
                        job_id,
                    },
                )?;
                Ok(Some(job))
            }
            // The job was persisted by an earlier tick whose cursor write was lost.
            Err(JobError::Exists(_)) => {
                append_event(
                    dir,
                    &id,
                    &AutomationEvent::Fired {
                        at_ms: now_ms,
                        instant_ms: latest,
                        job_id,
                    },
                )?;
                Ok(None)
            }
            Err(error) => {
                append_event(
                    dir,
                    &id,
                    &AutomationEvent::Failed {
                        at_ms: now_ms,
                        instant_ms: latest,
                        error: error.to_string(),
                    },
                )?;
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests;
