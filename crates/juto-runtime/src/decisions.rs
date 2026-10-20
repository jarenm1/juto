//! Durable decisions for blocked unattended actions.
//!
//! An unattended run never waits on an approval prompt. When a tool call leaves
//! the run's profile, the run records a [`Decision`], stops, and reports
//! [`UnattendedOutcome::Blocked`]. An operator approval grants that exact action
//! (session, tool, argument digest) once; a rejection leaves it undone. Records
//! live as one JSON file each under `<data_dir>/decisions/`, so a pending
//! decision survives daemon restarts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use juto_agent::RunControl;
use juto_ai::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::publish;

/// Longest string kept in a decision summary before it is cut.
const SUMMARY_STRING_CHARS: usize = 200;
/// Longest decision summary kept on disk and shown to clients.
const SUMMARY_CHARS: usize = 500;
/// Object keys whose values are replaced before a summary is stored.
const SECRET_KEY_MARKERS: &[&str] = &[
    "token",
    "secret",
    "password",
    "passwd",
    "key",
    "credential",
    "authorization",
    "cookie",
];

/// Lifecycle of a blocked action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DecisionStatus {
    /// Waiting for the operator.
    Pending,
    /// The operator allowed the action; one matching call may run.
    Approved,
    /// The operator refused the action; it never runs.
    Rejected,
    /// The approved action was run by a resumed run.
    Consumed,
}

/// A blocked action awaiting, or resolved by, an operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Decision {
    pub id: String,
    pub session_id: String,
    pub job_id: Option<String>,
    pub tool: String,
    pub call_id: String,
    /// Digest of the tool name and arguments; identifies the exact action.
    pub args_digest: String,
    /// Redacted, truncated preview of the arguments. Never the raw arguments.
    pub summary: String,
    /// The profile capability the action needs, e.g. `publish:feat/x`.
    pub capability: String,
    /// Commit the bookmark pointed at when a `publish` decision was recorded.
    /// The push is refused if the bookmark has moved since then.
    pub pinned_commit: Option<String>,
    pub reason: String,
    pub status: DecisionStatus,
    pub created_ms: u64,
    pub resolved_ms: Option<u64>,
}

/// Serializes read-modify-write cycles over the decision files in this process.
static LOCK: Mutex<()> = Mutex::new(());

/// Access to the decision records under one data directory.
#[derive(Debug, Clone)]
pub struct DecisionStore {
    dir: PathBuf,
}

impl DecisionStore {
    pub fn open(data_dir: &Path) -> Self {
        Self {
            dir: data_dir.join("decisions"),
        }
    }

    /// Looks up one decision. Ids that are not plain identifiers never match a file.
    pub fn get(&self, id: &str) -> Result<Option<Decision>> {
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return Ok(None);
        }
        let path = self.dir.join(format!("{id}.json"));
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
                format!("decision record {} is corrupt", path.display())
            })?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// All decisions, oldest first.
    pub fn list(&self) -> Result<Vec<Decision>> {
        let mut decisions = self.read_all()?;
        decisions.sort_by(|a, b| (a.created_ms, &a.id).cmp(&(b.created_ms, &b.id)));
        Ok(decisions)
    }

    /// Records that `call` is blocked for `capability`, pinned to `pinned_commit`.
    /// A pending decision for the same session, exact action, and pin is returned
    /// instead of a duplicate.
    pub fn block(
        &self,
        session_id: &str,
        job_id: Option<&str>,
        call: &ToolCall,
        capability: &str,
        reason: &str,
        pinned_commit: Option<&str>,
    ) -> Result<Decision> {
        let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let digest = args_digest(&call.name, &call.arguments);
        if let Some(existing) = self.read_all()?.into_iter().find(|decision| {
            decision.status == DecisionStatus::Pending
                && decision.session_id == session_id
                && decision.tool == call.name
                && decision.args_digest == digest
                && decision.pinned_commit.as_deref() == pinned_commit
        }) {
            return Ok(existing);
        }
        let decision = Decision {
            id: uuid::Uuid::now_v7().to_string(),
            session_id: session_id.to_owned(),
            job_id: job_id.map(str::to_owned),
            tool: call.name.clone(),
            call_id: call.id.clone(),
            args_digest: digest,
            summary: summarize(&call.arguments),
            capability: capability.to_owned(),
            pinned_commit: pinned_commit.map(str::to_owned),
            reason: reason.to_owned(),
            status: DecisionStatus::Pending,
            created_ms: juto_ai::timestamp_ms(),
            resolved_ms: None,
        };
        self.write(&decision)?;
        Ok(decision)
    }

    /// Resolves a pending decision. Approval grants its exact action once.
    pub fn resolve(&self, id: &str, approve: bool) -> Result<Decision> {
        let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(mut decision) = self.get(id)? else {
            bail!("unknown decision `{id}`");
        };
        if decision.status != DecisionStatus::Pending {
            bail!("decision `{id}` is already {:?}", decision.status);
        }
        decision.status = if approve {
            DecisionStatus::Approved
        } else {
            DecisionStatus::Rejected
        };
        decision.resolved_ms = Some(juto_ai::timestamp_ms());
        self.write(&decision)?;
        Ok(decision)
    }

    /// Consumes an approved grant that matches `call` exactly in `session_id`.
    /// Returns the consumed decision, or `None` when no grant matches.
    pub fn take_grant(&self, session_id: &str, call: &ToolCall) -> Result<Option<Decision>> {
        let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let digest = args_digest(&call.name, &call.arguments);
        let Some(mut grant) = self.read_all()?.into_iter().find(|decision| {
            decision.status == DecisionStatus::Approved
                && decision.session_id == session_id
                && decision.tool == call.name
                && decision.args_digest == digest
        }) else {
            return Ok(None);
        };
        grant.status = DecisionStatus::Consumed;
        self.write(&grant)?;
        Ok(Some(grant))
    }

    fn read_all(&self) -> Result<Vec<Decision>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| format!("reading {}", self.dir.display()));
            }
        };
        let mut decisions = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            let bytes =
                fs::read(&path).with_context(|| format!("reading decision {}", path.display()))?;
            decisions.push(
                serde_json::from_slice(&bytes)
                    .with_context(|| format!("decision record {} is corrupt", path.display()))?,
            );
        }
        Ok(decisions)
    }

    /// Writes a record atomically: a temporary file is renamed over the old one.
    fn write(&self, decision: &Decision) -> Result<()> {
        fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        let path = self.dir.join(format!("{}.json", decision.id));
        let tmp = self.dir.join(format!("{}.json.tmp", decision.id));
        fs::write(&tmp, serde_json::to_vec_pretty(decision)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }
}

/// The profile capability a tool call needs from an operator, if it leaves the
/// unattended profile. Sandboxed tools (`bash`, file tools) are confined by their
/// tier and need no decision; privileged in-process tools do.
pub fn outside_profile(call: &ToolCall) -> Option<String> {
    match call.name.as_str() {
        publish::TOOL_NAME => {
            let bookmark = call
                .arguments
                .get("bookmark")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(format!("publish:{bookmark}"))
        }
        _ => None,
    }
}

/// Digest identifying one exact action: the tool name and its arguments.
pub fn args_digest(tool: &str, arguments: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tool.as_bytes());
    hasher.update([0]);
    hasher.update(serde_json::to_vec(arguments).unwrap_or_default());
    format!("{:x}", hasher.finalize())
}

/// Redacted, truncated JSON preview of tool arguments.
pub fn summarize(arguments: &Value) -> String {
    let text = serde_json::to_string(&redact(arguments)).unwrap_or_default();
    truncate(&text, SUMMARY_CHARS)
}

fn redact(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| {
                    let lowered = key.to_ascii_lowercase();
                    let value = if SECRET_KEY_MARKERS.iter().any(|m| lowered.contains(m)) {
                        Value::String("<redacted>".into())
                    } else {
                        redact(value)
                    };
                    (key.clone(), value)
                })
                .collect::<Map<_, _>>(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
        Value::String(text) => Value::String(truncate(text, SUMMARY_STRING_CHARS)),
        other => other.clone(),
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut cut: String = text.chars().take(max_chars).collect();
    cut.push('…');
    cut
}

/// Per-run state shared by an unattended run tree: its root run and every child
/// it spawns. A call that needs a decision stops the whole tree.
pub(crate) struct UnattendedScope {
    store: DecisionStore,
    cwd: PathBuf,
    session_id: String,
    job_id: Option<String>,
    control: RunControl,
    state: Mutex<ScopeState>,
}

/// What an unattended run tree has recorded, guarded by one lock.
#[derive(Default)]
struct ScopeState {
    /// Set when the root run ends. A later call is refused and records nothing.
    closed: bool,
    /// The first decision that stopped the run tree, until the run ends.
    blocked: Option<Decision>,
    /// Approved `publish` grants consumed by admitted calls that have not reached
    /// the tool yet. The tool checks each grant's pinned commit before it pushes.
    admitted: Vec<Decision>,
}

impl UnattendedScope {
    pub(crate) fn new(
        data_dir: &Path,
        cwd: &Path,
        session_id: String,
        job_id: Option<String>,
        control: RunControl,
    ) -> Self {
        Self {
            store: DecisionStore::open(data_dir),
            cwd: cwd.to_path_buf(),
            session_id,
            job_id,
            control,
            state: Mutex::new(ScopeState::default()),
        }
    }

    /// Admits `call` when an approved grant covers it. Otherwise records a blocked
    /// decision, cancels the run tree, and rejects the call. A `publish` decision is
    /// pinned to the commit its bookmark points at when it is recorded. Any failure
    /// also rejects the call, so the action fails closed. The state lock is held
    /// through the check and the write, so a block cannot land after `finish` has
    /// taken the run's decision.
    pub(crate) async fn admit(
        &self,
        call: &ToolCall,
        capability: &str,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        let pinned_commit = if call.name == publish::TOOL_NAME {
            Some(pin_publish(&self.cwd, call, &cancel).await?)
        } else {
            None
        };
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.closed {
            return Err(format!(
                "unattended run has ended; `{}` was not run",
                call.name
            ));
        }
        match self.store.take_grant(&self.session_id, call) {
            Ok(Some(grant)) => {
                if grant.tool == publish::TOOL_NAME {
                    state.admitted.push(grant);
                }
                return Ok(());
            }
            Ok(None) => {}
            Err(error) => return Err(format!("{error:#}")),
        }
        let reason = format!(
            "`{}` needs operator approval in an unattended run",
            call.name
        );
        let decision = self
            .store
            .block(
                &self.session_id,
                self.job_id.as_deref(),
                call,
                capability,
                &reason,
                pinned_commit.as_deref(),
            )
            .map_err(|error| format!("{error:#}"))?;
        let message = format!(
            "blocked: operator decision {} is required for {capability}",
            decision.id
        );
        state.blocked.get_or_insert(decision);
        self.control.cancel();
        Err(message)
    }

    /// Ends the decision scope when the root run returns. Calls admitted after
    /// this are refused without a record. Returns the first decision recorded.
    pub(crate) fn finish(&self) -> Option<Decision> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.closed = true;
        state.blocked.take()
    }

    /// Takes the grant admitted for the `publish` call with this argument digest,
    /// so the tool can check the pinned commit before it pushes.
    pub(crate) fn take_admitted(&self, digest: &str) -> Option<Decision> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let index = state
            .admitted
            .iter()
            .position(|grant| grant.args_digest == digest)?;
        Some(state.admitted.remove(index))
    }
}

/// Resolves the commit a `publish` call's bookmark points at, or fails the call.
async fn pin_publish(
    cwd: &Path,
    call: &ToolCall,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let bookmark = call
        .arguments
        .get("bookmark")
        .and_then(Value::as_str)
        .ok_or_else(|| "publish needs a bookmark".to_owned())?;
    publish::resolve_bookmark(cwd, bookmark, cancel)
        .await
        .map_err(|error| format!("cannot pin publish of `{bookmark}`: {error}"))
}

/// Shared handle used by the runtime and its hooks.
pub(crate) type SharedScope = Arc<UnattendedScope>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments,
            thought_signature: None,
        }
    }

    #[test]
    fn blocked_action_is_granted_once_after_approval() {
        let dir = tempfile::tempdir().unwrap();
        let store = DecisionStore::open(dir.path());
        let publish = call("publish", json!({"bookmark": "feat/x", "title": "t"}));
        let decision = store
            .block("s1", None, &publish, "publish:feat/x", "reason", None)
            .unwrap();
        assert_eq!(decision.status, DecisionStatus::Pending);
        assert!(store.take_grant("s1", &publish).unwrap().is_none());

        let again = store
            .block("s1", None, &publish, "publish:feat/x", "reason", None)
            .unwrap();
        assert_eq!(
            again.id, decision.id,
            "pending decisions are not duplicated"
        );

        let approved = store.resolve(&decision.id, true).unwrap();
        assert_eq!(approved.status, DecisionStatus::Approved);
        assert!(store.take_grant("s1", &publish).unwrap().is_some());
        assert!(
            store.take_grant("s1", &publish).unwrap().is_none(),
            "grant is single use"
        );
        assert!(store.resolve(&decision.id, false).is_err());
    }

    #[test]
    fn grant_covers_only_the_exact_action() {
        let dir = tempfile::tempdir().unwrap();
        let store = DecisionStore::open(dir.path());
        let approved = call("publish", json!({"bookmark": "feat/x"}));
        let decision = store
            .block("s1", None, &approved, "publish:feat/x", "r", None)
            .unwrap();
        store.resolve(&decision.id, true).unwrap();
        let other_args = call("publish", json!({"bookmark": "feat/y"}));
        assert!(store.take_grant("s1", &other_args).unwrap().is_none());
        assert!(store.take_grant("s2", &approved).unwrap().is_none());
        assert!(store.take_grant("s1", &approved).unwrap().is_some());
    }

    #[test]
    fn rejected_action_is_never_granted() {
        let dir = tempfile::tempdir().unwrap();
        let store = DecisionStore::open(dir.path());
        let publish = call("publish", json!({"bookmark": "feat/x"}));
        let decision = store
            .block("s1", None, &publish, "publish:feat/x", "r", None)
            .unwrap();
        let rejected = store.resolve(&decision.id, false).unwrap();
        assert_eq!(rejected.status, DecisionStatus::Rejected);
        assert!(store.take_grant("s1", &publish).unwrap().is_none());
    }

    #[tokio::test]
    async fn calls_after_the_run_ends_are_refused_without_a_decision() {
        let dir = tempfile::tempdir().unwrap();
        let scope = UnattendedScope::new(
            dir.path(),
            dir.path(),
            "s1".into(),
            None,
            RunControl::new(),
        );
        assert!(scope.finish().is_none());
        let desktop = call("desktop", json!({}));
        let err = scope
            .admit(&desktop, "desktop", CancellationToken::new())
            .await
            .expect_err("an ended scope admits nothing");
        assert!(err.contains("has ended"), "{err}");
        assert!(
            DecisionStore::open(dir.path()).list().unwrap().is_empty(),
            "a late block must not leave a pending decision nobody is told about"
        );
    }

    #[test]
    fn summaries_redact_secret_keys_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let publish = call(
            "publish",
            json!({"bookmark": "feat/x", "apiToken": "hunter2", "nested": {"password": "p"}}),
        );
        let decision = DecisionStore::open(dir.path())
            .block("s1", Some("job-1"), &publish, "publish:feat/x", "r", None)
            .unwrap();
        assert!(!decision.summary.contains("hunter2"));
        assert!(!decision.summary.contains("\"p\""));
        assert!(decision.summary.contains("feat/x"));
        let reopened = DecisionStore::open(dir.path()).list().unwrap();
        assert_eq!(reopened, vec![decision]);
    }

    #[test]
    fn only_privileged_tools_leave_the_profile() {
        assert_eq!(
            outside_profile(&call("publish", json!({"bookmark": "b"}))),
            Some("publish:b".to_string())
        );
        assert_eq!(
            outside_profile(&call("bash", json!({"command": "ls"}))),
            None
        );
    }
}
