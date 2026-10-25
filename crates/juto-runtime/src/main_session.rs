//! The main session: one conversation with the user that never ends.
//!
//! Every request runs in the main journal itself. The model does not get the
//! journal's raw history: it gets the conversation's memory view (see
//! [`crate::memory`]), a line per message or per summary of older messages,
//! followed by the current turn in full. Work that deserves its own context
//! goes to subagents, which report back into this conversation.
//!
//! Sessions written while main routed requests into internal topic threads
//! keep those threads' journals under `<main>/threads/`. Their messages are
//! read back, merged in time order, as the start of the conversation; new
//! messages go to the main journal.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
use juto_agent::{AgentEvent, RunControl, RunOutcome};
use juto_ai::{Context, Message, Usage};
use juto_catalog::Model;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    Runtime,
    session::{EntryKind, Session},
};

/// Journal record of a topic thread from before main became one conversation.
const LEGACY_THREAD: &str = "thread_created";

/// System prompt for the `@tiny` call that summarizes a focus thread's progress.
pub(crate) const SUMMARY_SYSTEM: &str = "You summarize a coding agent's progress for a one-glance status card. Reply with one or two short plain sentences, under 160 characters in total, saying what is done and what is next or blocking. No preamble, no markdown, nothing else.";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LegacyRecord {
    thread: String,
    /// Journal file name inside the main session's thread directory.
    file: String,
}

/// A topic thread journal from before main became one conversation.
struct LegacyThread {
    id: String,
    path: PathBuf,
}

/// A main session: its journal, plus any legacy thread journals whose
/// messages begin the conversation.
pub struct MainSession {
    journal: Session,
    legacy: Vec<LegacyThread>,
}

impl MainSession {
    /// Open an existing session as a main session. Nothing is written here.
    pub fn open(path: &Path) -> Result<Self> {
        Self::load(Session::open(path)?)
    }

    fn load(journal: Session) -> Result<Self> {
        let dir = journal.path().with_extension("").join("threads");
        let mut legacy = Vec::new();
        for entry in journal.active_entries()? {
            if let EntryKind::Custom { custom_type, data } = &entry.kind
                && custom_type == LEGACY_THREAD
            {
                let record: LegacyRecord = serde_json::from_value(data.clone())
                    .context("main session has a malformed thread_created record")?;
                legacy.push(LegacyThread {
                    id: record.thread,
                    path: dir.join(record.file),
                });
            }
        }
        Ok(Self { journal, legacy })
    }

    pub fn id(&self) -> &str {
        &self.journal.header().id
    }

    pub fn path(&self) -> &Path {
        self.journal.path()
    }

    /// Model the next request runs on.
    pub fn current_model(&self) -> Option<String> {
        self.journal.current_model()
    }

    /// Ids and journals of the session's legacy topic threads. A run from
    /// before main became one conversation may have left tool calls without
    /// results there, and children it spawned report to those ids.
    pub fn legacy_threads(&self) -> impl Iterator<Item = (&str, &Path)> {
        self.legacy
            .iter()
            .map(|thread| (thread.id.as_str(), thread.path.as_path()))
    }

    /// The conversation: every message in time order, legacy threads' first,
    /// each once.
    pub fn timeline(&self) -> Result<Vec<Message>> {
        Ok(self
            .history()?
            .into_iter()
            .map(|(_, message)| message)
            .collect())
    }

    /// [`MainSession::timeline`] with each message's journal entry id. The
    /// conversation only grows, so an index into it keeps naming the same
    /// message: memory leaves and `recall` both count this way.
    pub(crate) fn history(&self) -> Result<Vec<(String, Message)>> {
        let mut items: Vec<(u64, String, Message)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for entry in self.journal.active_entries()? {
            seen.insert(entry.id.clone());
            if let EntryKind::Message { message } = &entry.kind {
                items.push((entry.timestamp, entry.id.clone(), message.clone()));
            }
        }
        for thread in &self.legacy {
            let session = Session::open(&thread.path)
                .with_context(|| format!("opening thread {}", thread.id))?;
            for entry in session.active_entries()? {
                // A thread adopted from the main journal copied its entries.
                if !seen.insert(entry.id.clone()) {
                    continue;
                }
                if let EntryKind::Message { message } = &entry.kind {
                    items.push((entry.timestamp, entry.id.clone(), message.clone()));
                }
            }
        }
        items.sort_by_key(|(timestamp, _, _)| *timestamp);
        Ok(items
            .into_iter()
            .map(|(_, id, message)| (id, message))
            .collect())
    }

    /// The journal, for readers that treat it as a plain session.
    pub(crate) fn into_journal(self) -> Session {
        self.journal
    }
}

impl Runtime {
    /// Create a main session on the configured model.
    pub fn new_main_session(&self) -> Result<MainSession> {
        MainSession::load(self.new_session()?)
    }

    /// Record `selector` as the main session's model.
    pub fn switch_main_model(&self, main: &mut MainSession, selector: &str) -> Result<()> {
        self.switch_model(&mut main.journal, selector)
    }

    /// Run `prompt` in the main conversation. The request waits until the
    /// conversation's memory view is built, then sees it before the prompt.
    /// After the run, the new messages join the memory, and their summaries
    /// build in the background.
    pub async fn send(
        &self,
        main: &mut MainSession,
        prompt: impl Into<String>,
        events: mpsc::UnboundedSender<AgentEvent>,
        control: RunControl,
    ) -> Result<RunOutcome> {
        if main.journal.header().cwd != self.inner.cwd {
            bail!("session belongs to a different working directory");
        }
        let selector = main.current_model().unwrap_or_else(|| self.config().model);
        let (model, _) = self.resolve_model(&selector)?;
        let high = self.memory_high(&model);
        let memory = self.main_memory(main.path())?;
        self.sync_memory(&memory, &main.timeline()?, high)?;
        self.settle_memory(&memory, &control.token()).await?;
        let outcome = self
            .run_with_memory(
                &mut main.journal,
                Some(&memory),
                prompt.into(),
                events,
                control,
            )
            .await;
        // Absorb what the run wrote, even a failed run's messages.
        self.sync_memory(&memory, &main.timeline()?, high)?;
        outcome
    }

    /// Bytes the memory view may take: at most `memoryViewBytes`, and half
    /// the model's context budget (about three bytes a token).
    pub(crate) fn memory_high(&self, model: &Model) -> usize {
        let settings = self.config();
        let budget_tokens =
            model.context_window.unwrap_or(128_000) as f64 * settings.compaction_threshold;
        let half_budget_bytes = (budget_tokens * 1.5) as usize;
        settings.memory_view_bytes.min(half_budget_bytes)
    }

    /// One or two lines on where a subagent's work stands, from its task and
    /// its latest reply.
    pub(crate) async fn summarize_subagent(
        &self,
        session: &Session,
        cancel: &CancellationToken,
    ) -> Result<String> {
        let messages = session.messages()?;
        let task = messages
            .iter()
            .find(|message| matches!(message, Message::User { .. }))
            .map(Message::text)
            .unwrap_or_default();
        let reply = messages
            .iter()
            .rev()
            .find(|message| matches!(message, Message::Assistant(_)))
            .map(Message::text)
            .unwrap_or_default();
        let input = format!(
            "Task:\n{}\n\nLatest reply:\n{}",
            excerpt(&task, 1500),
            excerpt(&reply, 3000),
        );
        let (text, _usage) = self.tiny_text(SUMMARY_SYSTEM, input, 80, cancel).await?;
        let summary = text.split_whitespace().collect::<Vec<_>>().join(" ");
        anyhow::ensure!(!summary.is_empty(), "summary model returned no text");
        Ok(excerpt(&summary, 200))
    }

    /// One short tool-free completion on the `@tiny` model chain.
    async fn tiny_text(
        &self,
        system: &str,
        input: String,
        max_tokens: u64,
        cancel: &CancellationToken,
    ) -> Result<(String, Usage)> {
        let (model, mut options) = self.resolve_model("@tiny")?;
        options.max_tokens = Some(max_tokens);
        options.thinking = None;
        let context = Context {
            system_prompt: vec![system.to_owned()],
            messages: vec![Message::user(input)],
            tools: Vec::new(),
        };
        let message = self
            .complete_once(&model, &options, &context, cancel)
            .await?;
        Ok((message.text(), message.usage))
    }
}

/// At most `max` characters of `text`, with an ellipsis when cut.
fn excerpt(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

#[cfg(test)]
#[path = "main_session_tests.rs"]
mod tests;
