//! In-process peer messaging and transcript lookup for subagent URIs.
//!
//! `agent://<id>` addresses one child, `agent://{a, b, c}` addresses a list of
//! children, and `history://<id>` reads a completed child's active session
//! journal. User-supplied URIs never become filesystem paths; history resolves
//! the journal path from the manager's completed result.
//!
//! `agent://main` is write-only: children send messages to the main agent's
//! inbox, and reading it is rejected so only the main agent drains it.

use std::sync::Arc;

use thiserror::Error;

use crate::{
    session::{Session, SessionEntry, SessionError},
    subagents::{
        MainMessage, MainMessageKind, SubagentError, SubagentManager, SubagentResult,
        SubagentStatus, SubagentSummary,
    },
};

/// Errors returned while resolving a peer URI or reading a transcript.
#[derive(Debug, Error)]
pub enum PeerError {
    #[error("invalid peer URI `{0}`")]
    InvalidUri(String),
    #[error("cannot send a message to read-only peer URI `{0}`")]
    ReadOnly(String),
    #[error("peer URI `{0}` accepts messages only and cannot be read")]
    WriteOnly(String),
    #[error("subagent `{agent_id}` has no completed transcript (status: {status:?})")]
    HistoryUnavailable {
        agent_id: String,
        status: SubagentStatus,
    },
    #[error(transparent)]
    Agent(#[from] SubagentError),
    #[error(transparent)]
    Session(#[from] SessionError),
}

/// A child snapshot exposed through an `agent://` URI.
#[derive(Debug, Clone)]
pub struct PeerAgentSnapshot {
    pub summary: SubagentSummary,
    /// Present only when the child completed successfully.
    pub result: Option<SubagentResult>,
}

/// Outcome of a message sent through `agent://<id>`, `agent://{...}`, or `agent://main`.
#[derive(Debug, Clone, Default)]
pub struct PeerSendReport {
    /// Recipients that accepted the message: child IDs, or `main` for the main inbox.
    pub sent: Vec<String>,
    /// Per-child failures. A list send continues when one child is rejected.
    pub failed: Vec<PeerSendFailure>,
}

#[derive(Debug, Clone)]
pub struct PeerSendFailure {
    pub agent_id: String,
    pub error: SubagentError,
}

/// Resource returned by a peer URI read.
#[derive(Debug, Clone)]
pub enum PeerResource {
    Agent(Box<PeerAgentSnapshot>),
    Agents(Vec<PeerAgentSnapshot>),
    History {
        agent_id: String,
        session_id: String,
        entries: Vec<SessionEntry>,
    },
}

/// Peer URI for the main agent's inbox. Write-only: children send to it and
/// the main agent drains it through `agent_inbox`.
pub const MAIN_URI: &str = "agent://main";
/// Sender recorded for a message to the main inbox that has no sender id.
pub const UNKNOWN_SENDER: &str = "unknown";
const MAIN_RECIPIENT: &str = "main";

/// Peer URI adapter over the runtime's subagent manager and durable sessions.
#[derive(Clone)]
pub struct PeerBus {
    subagents: Arc<SubagentManager>,
    /// Sender recorded for messages to `agent://main`; `None` for the main thread.
    sender: Option<String>,
}

impl PeerBus {
    pub fn new(subagents: Arc<SubagentManager>) -> Self {
        Self {
            subagents,
            sender: None,
        }
    }

    /// A bus for a child's own tools: messages to `agent://main` are recorded
    /// with `sender` as their `from`.
    pub fn for_child(subagents: Arc<SubagentManager>, sender: impl Into<String>) -> Self {
        Self {
            subagents,
            sender: Some(sender.into()),
        }
    }

    /// Send a message to one child, a list of children, or the main inbox. A
    /// queued or running child receives a steer, a completed child starts a new
    /// turn, and a failed or cancelled child is reported as failed. A list send
    /// continues past each failure.
    pub fn send(&self, uri: &str, message: impl Into<String>) -> Result<PeerSendReport, PeerError> {
        let target = parse_uri(uri)?;
        let message = message.into();
        let ids = match target {
            PeerTarget::Main => {
                self.subagents.main_inbox().push(MainMessage {
                    from: self
                        .sender
                        .clone()
                        .unwrap_or_else(|| UNKNOWN_SENDER.to_string()),
                    kind: MainMessageKind::Message,
                    body: message,
                });
                return Ok(PeerSendReport {
                    sent: vec![MAIN_RECIPIENT.to_string()],
                    failed: Vec::new(),
                });
            }
            PeerTarget::Agent(id) => vec![id],
            PeerTarget::Children(ids) => ids,
            PeerTarget::History(_) => return Err(PeerError::ReadOnly(uri.to_string())),
        };
        let mut report = PeerSendReport::default();
        for agent_id in ids {
            match self.subagents.send(&agent_id, message.clone()) {
                Ok(()) => report.sent.push(agent_id),
                Err(error) => report.failed.push(PeerSendFailure { agent_id, error }),
            }
        }
        Ok(report)
    }

    /// Read a child's current status/output, a list of child snapshots, or the
    /// active journal entries of a successfully completed child.
    pub fn read(&self, uri: &str) -> Result<PeerResource, PeerError> {
        match parse_uri(uri)? {
            PeerTarget::Agent(agent_id) => {
                Ok(PeerResource::Agent(Box::new(self.snapshot(&agent_id)?)))
            }
            PeerTarget::Children(ids) => {
                let snapshots = ids
                    .iter()
                    .map(|id| self.snapshot(id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PeerResource::Agents(snapshots))
            }
            PeerTarget::Main => Err(PeerError::WriteOnly(uri.to_string())),
            PeerTarget::History(agent_id) => {
                let snapshot = self.snapshot(&agent_id)?;
                let Some(result) = snapshot.result else {
                    return Err(PeerError::HistoryUnavailable {
                        agent_id,
                        status: snapshot.summary.status,
                    });
                };
                let session = Session::open(&result.session_path)?;
                let entries = session.active_entries()?.into_iter().cloned().collect();
                Ok(PeerResource::History {
                    agent_id,
                    session_id: session.header().id.clone(),
                    entries,
                })
            }
        }
    }

    fn snapshot(&self, agent_id: &str) -> Result<PeerAgentSnapshot, PeerError> {
        let summary = self
            .subagents
            .list()
            .into_iter()
            .find(|summary| summary.id == agent_id)
            .ok_or_else(|| PeerError::Agent(SubagentError::Unknown(agent_id.to_string())))?;
        self.snapshot_from_summary(summary)
    }

    fn snapshot_from_summary(
        &self,
        summary: SubagentSummary,
    ) -> Result<PeerAgentSnapshot, PeerError> {
        let result = if summary.status == SubagentStatus::Completed {
            self.subagents.output(&summary.id)?
        } else {
            None
        };
        Ok(PeerAgentSnapshot { summary, result })
    }
}

enum PeerTarget {
    Main,
    Agent(String),
    Children(Vec<String>),
    History(String),
}

fn parse_uri(uri: &str) -> Result<PeerTarget, PeerError> {
    let invalid = || PeerError::InvalidUri(uri.to_string());
    let (scheme, target) = uri.split_once("://").ok_or_else(invalid)?;
    match scheme {
        "agent" if target == MAIN_RECIPIENT => Ok(PeerTarget::Main),
        "agent" if target.starts_with('{') => parse_child_list(target)
            .map(PeerTarget::Children)
            .ok_or_else(invalid),
        "agent" => parse_id(target).map(PeerTarget::Agent).ok_or_else(invalid),
        "history" => parse_id(target)
            .map(PeerTarget::History)
            .ok_or_else(invalid),
        _ => Err(invalid()),
    }
}

/// Parses `{a, b, c}` into distinct child ids in first-seen order. Main is not
/// a child, so `{main}` is rejected; address the inbox as `agent://main`.
fn parse_child_list(target: &str) -> Option<Vec<String>> {
    let inner = target.strip_prefix('{')?.strip_suffix('}')?;
    let mut ids: Vec<String> = Vec::new();
    for part in inner.split(',') {
        let id = parse_id(part.trim())?;
        if id == MAIN_RECIPIENT {
            return None;
        }
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    (!ids.is_empty()).then_some(ids)
}

fn parse_id(id: &str) -> Option<String> {
    let valid = !id.is_empty()
        && id.trim() == id
        && !id
            .chars()
            .any(|c| matches!(c, '/' | '?' | '#' | '{' | '}' | ','));
    valid.then(|| id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use juto_agent::{AgentEvent, RunControl};
    use juto_ai::{Message, StopReason, Usage};
    use std::{path::PathBuf, sync::Arc};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    struct TestFactory {
        sessions: PathBuf,
    }

    #[async_trait]
    impl crate::subagents::SubagentFactory for TestFactory {
        async fn run(
            &self,
            id: &str,
            spec: crate::subagents::TaskSpec,
            control: RunControl,
            _events: mpsc::UnboundedSender<AgentEvent>,
        ) -> Result<SubagentResult, String> {
            if spec.task == "wait" {
                control.token().cancelled().await;
                return Err("cancelled".into());
            }
            if spec.task == "cancel-with-success" {
                control.token().cancelled().await;
            }

            let mut session = Session::create(&self.sessions, PathBuf::from("/work").as_path())
                .map_err(|error| error.to_string())?;
            session
                .append(crate::session::EntryKind::Message {
                    message: Message::user("inspect the parser"),
                })
                .map_err(|error| error.to_string())?;
            session
                .append(crate::session::EntryKind::Message {
                    message: Message::user("the parser rejects empty ids"),
                })
                .map_err(|error| error.to_string())?;
            Ok(SubagentResult {
                id: id.to_string(),
                output: "the parser rejects empty ids".into(),
                data: None,
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
                session_path: session.path().to_path_buf(),
            })
        }
    }

    fn manager(sessions: &std::path::Path) -> Arc<SubagentManager> {
        Arc::new(SubagentManager::new(
            Arc::new(TestFactory {
                sessions: sessions.to_path_buf(),
            }),
            4,
            CancellationToken::new(),
            crate::subagents::MainInbox::default(),
        ))
    }

    #[tokio::test]
    async fn history_uri_returns_active_session_entries_for_completed_agent() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("review parser"))
            .unwrap();
        manager.wait(&id).await.unwrap();
        let bus = PeerBus::new(manager);

        let resource = bus.read(&format!("history://{id}")).unwrap();
        let PeerResource::History {
            agent_id, entries, ..
        } = resource
        else {
            panic!("history URI did not return a transcript");
        };
        assert_eq!(agent_id, id);
        let messages: Vec<_> = entries
            .iter()
            .filter_map(|entry| match &entry.kind {
                crate::session::EntryKind::Message { message } => Some(message.text()),
                _ => None,
            })
            .collect();
        assert_eq!(
            messages,
            ["inspect the parser", "the parser rejects empty ids"]
        );
    }

    #[tokio::test]
    async fn agent_list_sends_to_each_named_child_and_reports_failures() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let finished = manager
            .spawn(crate::subagents::TaskSpec::new("review parser"))
            .unwrap();
        manager.wait(&finished).await.unwrap();
        let active = manager
            .spawn(crate::subagents::TaskSpec::new("wait"))
            .unwrap();
        let mut status = manager.subscribe_status(&active).unwrap();
        while *status.borrow() == SubagentStatus::Queued {
            status.changed().await.unwrap();
        }
        let bus = PeerBus::new(Arc::clone(&manager));

        let uri = format!("agent://{{{active}, {finished}, missing, {active}}}");
        let report = bus.send(&uri, "review this edge case").unwrap();
        assert_eq!(report.sent, [active, finished]);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].agent_id, "missing");
        assert_eq!(
            report.failed[0].error,
            SubagentError::Unknown("missing".into())
        );
        manager.owner().cancel();
    }
    #[tokio::test]
    async fn history_uri_rejects_children_without_a_completed_transcript() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("cancel-with-success"))
            .unwrap();
        let mut status = manager.subscribe_status(&id).unwrap();
        while *status.borrow() == SubagentStatus::Queued {
            status.changed().await.unwrap();
        }
        let bus = PeerBus::new(Arc::clone(&manager));

        assert!(matches!(
            bus.read(&format!("history://{id}")),
            Err(PeerError::HistoryUnavailable {
                status: SubagentStatus::Running,
                ..
            })
        ));
        manager.owner().cancel();
        assert!(matches!(
            manager.wait(&id).await,
            Err(SubagentError::Cancelled)
        ));
        assert!(manager.output(&id).unwrap().is_none());
        assert!(matches!(
            bus.read(&format!("history://{id}")),
            Err(PeerError::HistoryUnavailable {
                status: SubagentStatus::Cancelled,
                ..
            })
        ));
        let PeerResource::Agent(snapshot) = bus.read(&format!("agent://{id}")).unwrap() else {
            panic!("agent URI did not return an agent snapshot");
        };
        assert_eq!(snapshot.summary.status, SubagentStatus::Cancelled);
        assert!(snapshot.result.is_none());
    }

    #[tokio::test]
    async fn peer_snapshot_does_not_pair_running_status_with_completed_output() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("review parser"))
            .unwrap();
        manager.wait(&id).await.unwrap();
        let mut summary = manager
            .list()
            .into_iter()
            .find(|summary| summary.id == id)
            .unwrap();
        summary.status = SubagentStatus::Running;
        let snapshot = PeerBus::new(manager)
            .snapshot_from_summary(summary)
            .unwrap();

        assert_eq!(snapshot.summary.status, SubagentStatus::Running);
        assert!(snapshot.result.is_none());
    }
    #[tokio::test]
    async fn agent_uri_exposes_current_status_and_completed_output() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("review parser"))
            .unwrap();
        manager.wait(&id).await.unwrap();
        let bus = PeerBus::new(manager);

        let resource = bus.read(&format!("agent://{id}")).unwrap();
        let PeerResource::Agent(snapshot) = resource else {
            panic!("agent URI did not return an agent snapshot");
        };
        assert_eq!(snapshot.summary.status, SubagentStatus::Completed);
        assert_eq!(
            snapshot.result.unwrap().output,
            "the parser rejects empty ids"
        );
    }

    #[test]
    fn invalid_and_mismatched_peer_uris_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let bus = PeerBus::new(manager(temp.path()));

        assert!(matches!(
            bus.read("file:///etc/passwd"),
            Err(PeerError::InvalidUri(_))
        ));
        assert!(matches!(
            bus.read("agent://{a, b}/extra"),
            Err(PeerError::InvalidUri(_))
        ));
        assert!(matches!(
            bus.read("agent://{}"),
            Err(PeerError::InvalidUri(_))
        ));
        assert!(matches!(
            bus.send("agent://{a,,b}", "hi"),
            Err(PeerError::InvalidUri(_))
        ));
        assert!(matches!(
            bus.send("agent://{main}", "hi"),
            Err(PeerError::InvalidUri(_))
        ));
        assert!(matches!(
            bus.send("history://unknown", "hello"),
            Err(PeerError::ReadOnly(_))
        ));
        let report = bus.send("agent://unknown", "hello").unwrap();
        assert!(report.sent.is_empty());
        assert_eq!(
            report.failed[0].error,
            SubagentError::Unknown("unknown".into())
        );
    }

    #[tokio::test]
    async fn child_message_to_main_lands_in_main_inbox_with_child_id() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("wait"))
            .unwrap();
        let bus = PeerBus::for_child(manager.clone(), id.clone());

        let report = bus.send(MAIN_URI, "need a decision").unwrap();
        assert_eq!(report.sent, vec![MAIN_RECIPIENT.to_string()]);
        assert!(report.failed.is_empty());
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: id.clone(),
                kind: MainMessageKind::Message,
                body: "need a decision".into(),
            }]
        );

        manager.cancel(&id).unwrap();
        let _ = manager.wait(&id).await;
    }

    #[tokio::test]
    async fn reading_main_is_rejected_and_keeps_pending_messages() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let bus = PeerBus::new(manager.clone());
        bus.send(MAIN_URI, "queued").unwrap();

        assert!(matches!(
            bus.read(MAIN_URI),
            Err(PeerError::WriteOnly(uri)) if uri == MAIN_URI
        ));
        assert_eq!(
            manager.main_inbox().drain(),
            vec![MainMessage {
                from: UNKNOWN_SENDER.into(),
                kind: MainMessageKind::Message,
                body: "queued".into(),
            }]
        );
    }

    #[tokio::test]
    async fn list_send_does_not_reach_main_inbox() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("wait"))
            .unwrap();
        let bus = PeerBus::new(manager.clone());

        let report = bus.send(&format!("agent://{{{id}}}"), "status?").unwrap();
        assert_eq!(report.sent, vec![id.clone()]);
        assert!(manager.main_inbox().drain().is_empty());

        manager.cancel(&id).unwrap();
        let _ = manager.wait(&id).await;
    }
}
