//! A focus thread opened in the chat view. The chat view renders its
//! transcript and composer with the same surface as the main thread; this
//! controller owns the thread's state, live stream, and daemon commands.

use crate::chat_request::call;
use gpui::{AppContext, Context, Entity, EventEmitter, ScrollHandle, SharedString, Task};
use juto_agent::AgentEvent;
use juto_daemon::client::{Client, Notification};
use juto_daemon::protocol::{Reply, Request, ServerEvent, SubagentCommand};
use juto_runtime::subagents::{
    SubagentEvent, SubagentStatus, SubagentSummary, USER_TURN_PREFACE, strip_main_turn_header,
};
use tokio::sync::broadcast::error::RecvError;

use crate::chat::{Message, MessageRole, apply_stream_event, view_messages};
use crate::subagent_card::fallback_name;
use crate::vcs_sidebar::VcsSidebar;

/// Why the composer is disabled for `status`, or `None` when a message can be
/// sent. A message to a working thread steers it, a message to a blocked
/// thread answers its question, and either makes it a focus thread.
pub fn message_unavailable_reason(status: &SubagentStatus) -> Option<&'static str> {
    match status {
        SubagentStatus::Completed
        | SubagentStatus::Queued
        | SubagentStatus::Running
        | SubagentStatus::Blocked { .. }
        | SubagentStatus::Verifying
        | SubagentStatus::Reviewing => None,
        SubagentStatus::Failed => Some("The thread failed, so it cannot take messages."),
        SubagentStatus::Cancelled => Some("The thread was cancelled, so it cannot take messages."),
    }
}

/// Pure focus-mode state. `V` is the VCS panel handle, kept generic so the
/// transitions can be tested without GPUI.
pub struct FocusState<V> {
    thread_id: String,
    /// The child's name, shown on its replies.
    label: SharedString,
    status: SubagentStatus,
    vcs: Option<V>,
    /// The main agent released this thread's workspace, so its fork is gone.
    released: bool,
    transcript: Result<Vec<Message>, String>,
    transcript_fresh: bool,
    pending_load: Option<u64>,
    next_load: u64,
    /// The reply being streamed and messages committed since the last transcript read.
    tail: Vec<Message>,
    /// The last tail entry is the reply being streamed.
    streaming: bool,
}

impl<V> FocusState<V> {
    pub fn enter(
        thread_id: String,
        label: SharedString,
        status: SubagentStatus,
        vcs: Option<V>,
        released: bool,
    ) -> Self {
        Self {
            thread_id,
            label,
            status,
            vcs,
            released,
            transcript: Ok(Vec::new()),
            transcript_fresh: false,
            pending_load: None,
            next_load: 0,
            tail: Vec::new(),
            streaming: false,
        }
    }

    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    pub fn status(&self) -> &SubagentStatus {
        &self.status
    }

    pub fn vcs(&self) -> Option<&V> {
        self.vcs.as_ref()
    }

    pub fn transcript(&self) -> &Result<Vec<Message>, String> {
        &self.transcript
    }

    /// Live rows after the committed transcript.
    pub fn tail(&self) -> &[Message] {
        &self.tail
    }

    /// Why the composer is disabled, or `None` when a message can be sent.
    /// A released thread's fork is gone, so it cannot take messages.
    pub fn message_unavailable(&self) -> Option<&'static str> {
        message_unavailable_reason(&self.status).or_else(|| {
            self.released
                .then_some("The workspace was released, so messages are disabled.")
        })
    }

    pub fn can_message(&self) -> bool {
        self.message_unavailable().is_none()
    }

    /// Records a status update. A change invalidates the transcript so a
    /// settled thread reloads it, and any read still in flight is ignored. The
    /// committed rows stay on screen meanwhile.
    pub fn set_status(&mut self, status: SubagentStatus) {
        if status != self.status {
            self.invalidate_transcript();
        }
        self.status = status;
    }

    /// Marks the transcript stale so the next summary read reloads it. Used when
    /// the live stream may have dropped deltas.
    pub fn invalidate_transcript(&mut self) {
        self.transcript_fresh = false;
        self.pending_load = None;
    }

    /// Returns a load token when a transcript read is due: the thread has
    /// settled, the transcript is not fresh, and no read is pending. The daemon
    /// serves a child's transcript only once it settles, so a working thread
    /// keeps its committed rows and shows the live turn from the stream.
    pub fn begin_transcript_load(&mut self) -> Option<u64> {
        if !self.status.is_terminal() || self.transcript_fresh || self.pending_load.is_some() {
            return None;
        }
        self.next_load += 1;
        self.pending_load = Some(self.next_load);
        self.pending_load
    }

    /// Applies a finished read. Returns `false` and changes nothing when the token
    /// is stale.
    pub fn apply_transcript(
        &mut self,
        token: u64,
        transcript: Result<Vec<Message>, String>,
    ) -> bool {
        if self.pending_load != Some(token) {
            return false;
        }
        self.pending_load = None;
        self.transcript_fresh = true;
        self.transcript = transcript.map(|mut rows| {
            label_replies(&mut rows, &self.label);
            rows
        });
        // A reply still streaming is not on disk yet, so it stays on screen.
        let in_flight = if self.streaming {
            self.tail.pop()
        } else {
            None
        };
        self.tail.clear();
        self.tail.extend(in_flight);
        true
    }

    /// Applies one live event from the child. Streamed text accumulates in the
    /// tail, and a committed message replaces the reply being streamed. Returns
    /// whether the view changed.
    pub fn apply_live(&mut self, event: &AgentEvent) -> bool {
        match event {
            AgentEvent::MessageEnd { message } => {
                if self.streaming {
                    self.tail.pop();
                    self.streaming = false;
                }
                let mut committed = view_messages(std::iter::once(message));
                strip_user_preface(&mut committed);
                label_replies(&mut committed, &self.label);
                self.tail.extend(committed);
                true
            }
            event => {
                if matches!(event, AgentEvent::MessageStart) {
                    self.streaming = true;
                }
                let changed = apply_stream_event(&mut self.tail, event);
                label_replies(&mut self.tail, &self.label);
                changed
            }
        }
    }
}

/// Shows user turns as typed. The runtime frames each message the user sends
/// a thread with `USER_TURN_PREFACE` for the model; the row omits it. Main's
/// messages carry `MAIN_TURN_HEADER` instead; the row omits it and reads "Main".
pub fn strip_user_preface(messages: &mut [Message]) {
    for message in messages {
        if message.role != MessageRole::User {
            continue;
        }
        if let Some(typed) = message
            .text
            .strip_prefix(USER_TURN_PREFACE)
            .and_then(|rest| rest.strip_prefix("\n\n"))
        {
            message.text = typed.to_owned().into();
        } else if let Some(sent) = strip_main_turn_header(&message.text) {
            message.text = sent.to_owned().into();
            message.author = Some(MAIN_LABEL.into());
        }
    }
}

/// The author shown on turns main sent a child.
const MAIN_LABEL: &str = "Main";

/// Labels a child's replies with its name. A row that already has an author
/// keeps it.
fn label_replies(messages: &mut [Message], label: &SharedString) {
    for message in messages {
        if message.role == MessageRole::Assistant && message.author.is_none() {
            message.author = Some(label.clone());
        }
    }
}

/// Emitted when the thread should leave focus.
pub enum FocusEvent {
    /// The daemon archived the thread, so it left the task list.
    Archived,
}

pub struct FocusThread {
    state: FocusState<Entity<VcsSidebar>>,
    title: SharedString,
    client: Client,
    tokio_handle: tokio::runtime::Handle,
    scroll: ScrollHandle,
    /// The last rejected send or archive.
    error: Option<SharedString>,
    _updates: Task<()>,
}

impl EventEmitter<FocusEvent> for FocusThread {}

impl FocusThread {
    /// Subscribes to daemon state changes; reconnects reload the authoritative list.
    pub fn new(
        summary: SubagentSummary,
        client: Client,
        tokio_handle: tokio::runtime::Handle,
        cx: &mut Context<Self>,
    ) -> Self {
        let status = summary.status.clone();
        let label: SharedString = summary
            .name
            .clone()
            .unwrap_or_else(|| fallback_name(&summary.id))
            .into();
        let vcs = summary
            .workspace
            .clone()
            .map(|workspace| cx.new(|cx| VcsSidebar::new(workspace, tokio_handle.clone(), cx)));
        let mut notifications = client.subscribe();
        let updates_client = client.clone();
        let updates_handle = tokio_handle.clone();
        let thread_id = summary.id.clone();
        let updates_task = cx.spawn(async move |this, cx| {
            loop {
                let result = call(
                    updates_client.clone(),
                    updates_handle.clone(),
                    Request::ListSubagents,
                )
                .await;
                if let Ok(Reply::Subagents(children)) = result
                    && let Some(child) = children.into_iter().find(|child| child.id == thread_id)
                    && this
                        .update(cx, |this, cx| this.apply_summary(child, cx))
                        .is_err()
                {
                    break;
                }
                loop {
                    match notifications.recv().await {
                        Ok(Notification::Event(ServerEvent::Subagent {
                            event: SubagentEvent::Agent { id, event },
                        })) if id == thread_id => {
                            if this
                                .update(cx, |this, cx| this.apply_live_event(&event, cx))
                                .is_err()
                            {
                                return;
                            }
                        }
                        // Deltas for other children are not shown in this view.
                        Ok(Notification::Event(ServerEvent::Subagent {
                            event: SubagentEvent::Agent { .. },
                        })) => {}
                        Ok(Notification::Event(ServerEvent::Subagent { .. }))
                        | Ok(Notification::Connected(_)) => break,
                        Err(RecvError::Lagged(_)) => {
                            // Dropped deltas leave the live tail stale; the next read reloads it.
                            if this
                                .update(cx, |this, _| this.state.invalidate_transcript())
                                .is_err()
                            {
                                return;
                            }
                            break;
                        }
                        Err(RecvError::Closed) => return,
                        _ => {}
                    }
                }
            }
        });
        Self {
            state: FocusState::enter(summary.id, label, status, vcs, summary.released),
            title: summary.name.unwrap_or(summary.task).into(),
            client,
            tokio_handle,
            scroll: ScrollHandle::new(),
            error: None,
            _updates: updates_task,
        }
    }

    pub fn title(&self) -> SharedString {
        self.title.clone()
    }

    pub fn thread_id(&self) -> &str {
        self.state.thread_id()
    }

    pub fn status(&self) -> &SubagentStatus {
        self.state.status()
    }

    pub fn vcs(&self) -> Option<&Entity<VcsSidebar>> {
        self.state.vcs()
    }

    pub fn scroll(&self) -> &ScrollHandle {
        &self.scroll
    }

    /// The committed transcript followed by live rows.
    pub fn messages(&self) -> Vec<Message> {
        let committed = self.state.transcript().as_deref().unwrap_or_default();
        committed.iter().chain(self.state.tail()).cloned().collect()
    }

    pub fn can_message(&self) -> bool {
        self.state.can_message()
    }

    /// The question a blocked thread is waiting on the user to answer.
    pub fn question(&self) -> Option<SharedString> {
        match self.state.status() {
            SubagentStatus::Blocked { prompt, .. } => Some(prompt.clone().into()),
            _ => None,
        }
    }

    /// What to show above the composer: a rejected command, a failed
    /// transcript read, or why the thread cannot take messages.
    pub fn notice(&self) -> Option<SharedString> {
        self.error
            .clone()
            .or_else(|| {
                self.state
                    .transcript()
                    .as_ref()
                    .err()
                    .map(|e| e.clone().into())
            })
            .or_else(|| self.state.message_unavailable().map(SharedString::from))
    }

    /// Shown in place of an empty transcript.
    pub fn empty_hint(&self) -> &'static str {
        if self.state.status().is_terminal() {
            "No transcript messages."
        } else {
            "No messages yet. Replies stream in here while this thread works."
        }
    }

    fn apply_summary(&mut self, child: SubagentSummary, cx: &mut Context<Self>) {
        self.state.released = child.released;
        self.state.set_status(child.status);
        if let Some(token) = self.state.begin_transcript_load() {
            self.load_transcript(token, cx);
        }
        cx.notify();
    }

    /// Shows one live event from the child, scrolling only when the view changed.
    fn apply_live_event(&mut self, event: &AgentEvent, cx: &mut Context<Self>) {
        if self.state.apply_live(event) {
            self.scroll.scroll_to_bottom();
            cx.notify();
        }
    }

    /// Reads the child's session on the background executor, then converts the
    /// messages on the foreground.
    fn load_transcript(&self, token: u64, cx: &mut Context<Self>) {
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let thread_id = self.state.thread_id().to_owned();
        cx.spawn(async move |this, cx| {
            let transcript = match call(
                client,
                handle,
                Request::SubagentTranscript { id: thread_id },
            )
            .await
            {
                Ok(Reply::Transcript(messages)) => Ok(messages),
                Ok(_) => Err("Unexpected daemon transcript response".to_owned()),
                Err(error) => Err(error.to_string()),
            };
            this.update(cx, |this, cx| {
                let transcript = transcript.map(|messages| {
                    let mut rows = view_messages(&messages);
                    strip_user_preface(&mut rows);
                    rows
                });
                if this.state.apply_transcript(token, transcript) {
                    this.scroll.scroll_to_bottom();
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Sends `text` to the thread: a running thread is steered, a blocked one
    /// takes it as its answer, and a completed one takes a follow-up turn.
    pub fn send(&mut self, text: String, cx: &mut Context<Self>) {
        if !self.state.can_message() {
            return;
        }
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let request = Request::SubagentCommand {
            id: self.state.thread_id().to_owned(),
            command: SubagentCommand::Send { text },
        };
        self.error = None;
        self.scroll.scroll_to_bottom();
        cx.spawn(async move |this, cx| {
            let result = call(client, handle, request).await;
            this.update(cx, |this, cx| {
                if let Err(error) = result {
                    this.error = Some(format!("Message rejected: {error}").into());
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    /// Hides this finished thread from the task list. The daemon keeps its
    /// workspace and transcript, and focus closes once the archive is confirmed.
    pub fn archive(&mut self, cx: &mut Context<Self>) {
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let request = Request::SubagentCommand {
            id: self.state.thread_id().to_owned(),
            command: SubagentCommand::SetArchived { archived: true },
        };
        cx.spawn(async move |this, cx| {
            let result = call(client, handle, request).await;
            this.update(cx, |this, cx| match result {
                Ok(_) => cx.emit(FocusEvent::Archived),
                Err(error) => {
                    this.error = Some(format!("Archive failed: {error}").into());
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }
}

#[cfg(test)]
mod tests {
    use juto_runtime::subagents::BlockerReason;

    use super::*;

    fn all_statuses() -> Vec<SubagentStatus> {
        vec![
            SubagentStatus::Queued,
            SubagentStatus::Running,
            SubagentStatus::Blocked {
                reason: BlockerReason::UserInput,
                prompt: "choose".into(),
            },
            SubagentStatus::Verifying,
            SubagentStatus::Reviewing,
            SubagentStatus::Completed,
            SubagentStatus::Failed,
            SubagentStatus::Cancelled,
        ]
    }

    #[test]
    fn messaging_is_enabled_unless_terminal_without_a_result() {
        for status in all_statuses() {
            let enabled = !matches!(status, SubagentStatus::Failed | SubagentStatus::Cancelled);
            let state: FocusState<()> =
                FocusState::enter("t".into(), "Builder".into(), status.clone(), None, false);
            assert_eq!(state.can_message(), enabled, "status {status:?}");
        }
    }

    #[test]
    fn released_thread_refuses_messages() {
        let state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Completed,
            None,
            true,
        );
        assert!(!state.can_message());
        assert!(state.message_unavailable().is_some());
    }

    #[test]
    fn transcript_loads_once_per_terminal_state() {
        let mut state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Completed,
            None,
            false,
        );
        let token = state
            .begin_transcript_load()
            .expect("terminal thread loads");
        assert!(
            state.begin_transcript_load().is_none(),
            "read already pending"
        );
        assert!(state.apply_transcript(token, Ok(Vec::new())));
        assert!(
            state.begin_transcript_load().is_none(),
            "transcript is fresh"
        );
    }

    #[test]
    fn working_thread_keeps_its_history_until_it_settles() {
        let mut state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Completed,
            None,
            false,
        );
        let first = state.begin_transcript_load().expect("settled thread loads");
        let history = view_messages(&[juto_ai::Message::user("task")]);
        assert!(state.apply_transcript(first, Ok(history)));
        state.set_status(SubagentStatus::Running);
        assert!(
            state.begin_transcript_load().is_none(),
            "the daemon has no transcript for a working thread"
        );
        assert_eq!(state.transcript().as_ref().map(Vec::len), Ok(1));
        state.set_status(SubagentStatus::Completed);
        assert!(
            state.begin_transcript_load().is_some(),
            "reload once the turn settles"
        );
    }

    #[test]
    fn stale_transcript_read_is_ignored() {
        let mut state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Completed,
            None,
            false,
        );
        let stale = state.begin_transcript_load().expect("first load");
        state.set_status(SubagentStatus::Running);
        state.set_status(SubagentStatus::Completed);
        let current = state.begin_transcript_load().expect("reload");
        assert!(!state.apply_transcript(stale, Err("stale".into())));
        assert!(state.apply_transcript(current, Ok(Vec::new())));
        assert!(state.transcript().is_ok());
    }

    fn delta(text: &str) -> AgentEvent {
        AgentEvent::MessageUpdate {
            event: juto_ai::ProviderEvent::TextDelta {
                index: 0,
                delta: text.to_owned(),
            },
        }
    }

    fn assistant(text: &str) -> juto_ai::Message {
        juto_ai::Message::Assistant(juto_ai::AssistantMessage {
            content: vec![juto_ai::ContentBlock::Text {
                text: text.to_owned(),
            }],
            api: "test".into(),
            provider: "test".into(),
            model: "test-model".into(),
            usage: Default::default(),
            stop_reason: juto_ai::StopReason::Stop,
            timestamp: 0,
            response_id: None,
            error: None,
        })
    }

    #[test]
    fn live_deltas_stream_and_message_end_replaces_the_reply() {
        let mut state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Running,
            None,
            false,
        );
        assert!(state.apply_live(&AgentEvent::MessageStart));
        assert!(state.apply_live(&delta("Hel")));
        assert!(state.apply_live(&delta("lo")));
        assert_eq!(state.tail().len(), 1);
        assert_eq!(state.tail()[0].text.to_string(), "Hello");
        assert!(state.apply_live(&AgentEvent::MessageEnd {
            message: assistant("Hello"),
        }));
        assert_eq!(
            state.tail().len(),
            1,
            "the committed reply replaces the stream"
        );
        assert_eq!(state.tail()[0].text.to_string(), "Hello");
    }

    #[test]
    fn transcript_read_keeps_the_reply_in_flight() {
        // A send can start a turn before the summary reports it, so a read begun
        // while the thread looked settled lands mid-stream.
        let mut state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Completed,
            None,
            false,
        );
        let token = state.begin_transcript_load().expect("settled thread loads");
        state.apply_live(&AgentEvent::MessageStart);
        state.apply_live(&delta("Hi"));
        assert!(state.apply_transcript(token, Ok(Vec::new())));
        assert_eq!(state.tail().len(), 1);
        assert_eq!(state.tail()[0].text.to_string(), "Hi");
    }

    #[test]
    fn user_rows_show_the_typed_message() {
        let framed = format!("{USER_TURN_PREFACE}\n\nhello thread");
        let mut rows = view_messages(&[
            juto_ai::Message::user(framed.as_str()),
            juto_ai::Message::user("plain"),
            assistant(&framed),
        ]);
        strip_user_preface(&mut rows);
        let texts: Vec<String> = rows.iter().map(|row| row.text.to_string()).collect();
        assert_eq!(texts, ["hello thread", "plain", framed.as_str()]);
    }

    #[test]
    fn a_main_turn_reads_as_main_without_its_header() {
        let mut rows = view_messages(&[juto_ai::Message::user(
            juto_runtime::subagents::main_turn_text("look at the parser"),
        )]);
        strip_user_preface(&mut rows);
        assert_eq!(rows[0].text.as_ref(), "look at the parser");
        assert_eq!(rows[0].author, Some("Main".into()));
    }

    #[test]
    fn a_child_reply_reads_as_the_child() {
        let mut state: FocusState<()> = FocusState::enter(
            "t".into(),
            "Builder".into(),
            SubagentStatus::Running,
            None,
            false,
        );
        assert!(state.apply_live(&AgentEvent::MessageStart));
        assert!(state.apply_live(&delta("Hi")));
        assert_eq!(
            state.tail().last().and_then(|row| row.author.clone()),
            Some("Builder".into())
        );
    }
}
