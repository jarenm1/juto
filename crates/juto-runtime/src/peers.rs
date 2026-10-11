//! In-process peer messaging and transcript lookup for subagent URIs.
//!
//! `agent://<id>` addresses one child, `agent://all` broadcasts to all live
//! children, and `history://<id>` reads a completed child's active session
//! journal. User-supplied URIs never become filesystem paths; history resolves
//! the journal path from the manager's completed result.

use std::sync::Arc;

use thiserror::Error;

use crate::{
    session::{Session, SessionEntry, SessionError},
    subagents::{SubagentError, SubagentManager, SubagentResult, SubagentStatus, SubagentSummary},
};

/// Errors returned while resolving a peer URI or reading a transcript.
#[derive(Debug, Error)]
pub enum PeerError {
    #[error("invalid peer URI `{0}`")]
    InvalidUri(String),
    #[error("cannot send a message to read-only peer URI `{0}`")]
    ReadOnly(String),
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

/// Outcome of a message sent through `agent://<id>` or `agent://all`.
#[derive(Debug, Clone, Default)]
pub struct PeerSendReport {
    /// Child IDs whose steering queue accepted the message.
    pub sent: Vec<String>,
    /// Per-child failures. A broadcast continues when one child is terminal.
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

/// Peer URI adapter over the runtime's subagent manager and durable sessions.
#[derive(Clone)]
pub struct PeerBus {
    subagents: Arc<SubagentManager>,
}

impl PeerBus {
    pub fn new(subagents: Arc<SubagentManager>) -> Self {
        Self { subagents }
    }

    /// Send a steering message to one child or every currently live child.
    /// Broadcast failures are returned per recipient, so a terminal child does
    /// not prevent delivery to other children.
    pub fn send(&self, uri: &str, message: impl Into<String>) -> Result<PeerSendReport, PeerError> {
        let target = parse_uri(uri)?;
        let ids = match target {
            PeerTarget::Agent(id) => vec![id],
            PeerTarget::All => self
                .subagents
                .list()
                .into_iter()
                .filter(|child| !child.status.is_terminal())
                .map(|child| child.id)
                .collect(),
            PeerTarget::History(_) => return Err(PeerError::ReadOnly(uri.to_string())),
        };
        let message = message.into();
        let mut report = PeerSendReport::default();
        for agent_id in ids {
            match self.subagents.send(&agent_id, message.clone()) {
                Ok(()) => report.sent.push(agent_id),
                Err(error) => report.failed.push(PeerSendFailure { agent_id, error }),
            }
        }
        Ok(report)
    }

    /// Read a child's current status/output, all child snapshots, or the active
    /// journal entries of a successfully completed child.
    pub fn read(&self, uri: &str) -> Result<PeerResource, PeerError> {
        match parse_uri(uri)? {
            PeerTarget::Agent(agent_id) => {
                Ok(PeerResource::Agent(Box::new(self.snapshot(&agent_id)?)))
            }
            PeerTarget::All => {
                let snapshots = self
                    .subagents
                    .list()
                    .into_iter()
                    .map(|summary| self.snapshot_from_summary(summary))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PeerResource::Agents(snapshots))
            }
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
        let result = self.subagents.output(&summary.id)?;
        Ok(PeerAgentSnapshot { summary, result })
    }
}

enum PeerTarget {
    Agent(String),
    All,
    History(String),
}

fn parse_uri(uri: &str) -> Result<PeerTarget, PeerError> {
    let invalid = || PeerError::InvalidUri(uri.to_string());
    let (scheme, agent_id) = uri.split_once("://").ok_or_else(invalid)?;
    if agent_id.is_empty()
        || agent_id.trim() != agent_id
        || agent_id
            .chars()
            .any(|character| matches!(character, '/' | '?' | '#'))
    {
        return Err(invalid());
    }
    match scheme {
        "agent" if agent_id == "all" => Ok(PeerTarget::All),
        "agent" => Ok(PeerTarget::Agent(agent_id.to_string())),
        "history" => Ok(PeerTarget::History(agent_id.to_string())),
        _ => Err(invalid()),
    }
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
    async fn agent_all_sends_only_to_live_agents() {
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

        let report = bus.send("agent://all", "review this edge case").unwrap();
        assert_eq!(report.sent, [active]);
        assert!(report.failed.is_empty());
        manager.cancel_all();
    }
    #[tokio::test]
    async fn history_uri_rejects_children_without_a_completed_transcript() {
        let temp = tempfile::tempdir().unwrap();
        let manager = manager(temp.path());
        let id = manager
            .spawn(crate::subagents::TaskSpec::new("wait"))
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
        manager.cancel_all();
        assert!(matches!(
            manager.wait(&id).await,
            Err(SubagentError::Cancelled)
        ));
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
            bus.read("agent://all/extra"),
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
}
