//! Focus mode for a focus thread: its transcript, a message composer, and the
//! VCS panel rooted at its workspace.

use std::collections::{HashMap, HashSet};

use crate::chat_request::call;
use gpui::{
    ClickEvent, Context, Entity, EventEmitter, IntoElement, Render, ScrollHandle, SharedString,
    Subscription, Task, Window, div, prelude::*, px, rgb,
};
use juto_agent::AgentEvent;
use juto_daemon::client::{Client, Notification};
use juto_daemon::protocol::{Reply, Request, ServerEvent, SubagentCommand};
use juto_runtime::subagents::{SubagentEvent, SubagentStatus, SubagentSummary};
use tokio::sync::broadcast::error::RecvError;

use crate::chat::{
    DesktopCanvas, Message, MessageRole, apply_stream_event, canvas_body, desktop_artifact_card,
    mount_canvases, view_messages,
};
use crate::chat_input::{ChatInput, SubmitMessage};
use crate::vcs_sidebar::VcsSidebar;
use crate::view_element::{ViewActions, render_view};

/// Why the composer is disabled for `status`, or `None` when a message can be
/// sent. A message to a working thread steers it and makes it a focus thread.
pub fn message_unavailable_reason(status: &SubagentStatus) -> Option<&'static str> {
    match status {
        SubagentStatus::Completed
        | SubagentStatus::Queued
        | SubagentStatus::Running
        | SubagentStatus::Verifying
        | SubagentStatus::Reviewing => None,
        SubagentStatus::Blocked { .. } => {
            Some("The thread is blocked. Answer its request before messaging it.")
        }
        SubagentStatus::Failed => Some("The thread failed, so it cannot take messages."),
        SubagentStatus::Cancelled => Some("The thread was cancelled, so it cannot take messages."),
    }
}

/// Pure focus-mode state. `V` is the VCS panel handle, kept generic so the
/// transitions can be tested without GPUI.
pub struct FocusState<V> {
    thread_id: String,
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
        status: SubagentStatus,
        vcs: Option<V>,
        released: bool,
    ) -> Self {
        Self {
            thread_id,
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

    /// Drops the VCS panel handle. The thread itself keeps running.
    pub fn exit(&mut self) {
        self.vcs = None;
    }

    /// Records a status update. A change invalidates the transcript so the next
    /// summary read reloads it, and any read still in flight is ignored.
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

    /// Returns a load token when a transcript read is due: the transcript is not
    /// fresh and no read is pending. Running threads load too, so their history
    /// shows before the live turn arrives.
    pub fn begin_transcript_load(&mut self) -> Option<u64> {
        if self.transcript_fresh || self.pending_load.is_some() {
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
        self.transcript = transcript;
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
                self.tail.extend(view_messages(std::iter::once(message)));
                true
            }
            event => {
                if matches!(event, AgentEvent::MessageStart) {
                    self.streaming = true;
                }
                apply_stream_event(&mut self.tail, event)
            }
        }
    }
}

/// Emitted when the user leaves focus mode.
pub enum FocusEvent {
    Exit,
    ExpandDesktop(Entity<crate::desktop_viewer::DesktopViewer>),
}

pub struct FocusView {
    state: FocusState<Entity<VcsSidebar>>,
    title: SharedString,
    client: Client,
    tokio_handle: tokio::runtime::Handle,
    desktop_canvases: HashMap<String, DesktopCanvas>,
    input: Entity<ChatInput>,
    transcript_scroll: ScrollHandle,
    error: Option<SharedString>,
    _input_subscription: Subscription,
    _updates: Task<()>,
}

impl EventEmitter<FocusEvent> for FocusView {}

impl FocusView {
    /// Subscribes to daemon state changes; reconnects reload the authoritative list.
    pub fn new(
        summary: SubagentSummary,
        client: Client,
        tokio_handle: tokio::runtime::Handle,
        cx: &mut Context<Self>,
    ) -> Self {
        let status = summary.status.clone();
        let vcs = summary
            .workspace
            .clone()
            .map(|workspace| cx.new(|cx| VcsSidebar::new(workspace, tokio_handle.clone(), cx)));
        let input = cx.new(ChatInput::new);
        let input_subscription = cx.subscribe(&input, |this, _, _: &SubmitMessage, cx| {
            this.submit_message(cx);
        });
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
            state: FocusState::enter(summary.id, status, vcs, summary.released),
            title: summary.name.unwrap_or(summary.task).into(),
            client,
            tokio_handle,
            desktop_canvases: HashMap::new(),
            input,
            transcript_scroll: ScrollHandle::new(),
            error: None,
            _input_subscription: input_subscription,
            _updates: updates_task,
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
            self.transcript_scroll.scroll_to_bottom();
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
                let transcript = transcript.map(|messages| view_messages(&messages));
                if this.state.apply_transcript(token, transcript) {
                    this.transcript_scroll.scroll_to_bottom();
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn submit_message(&mut self, cx: &mut Context<Self>) {
        if !self.state.can_message() || !self.input.read(cx).has_text() {
            return;
        }
        let prompt = self.input.read(cx).text();
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let request = Request::SubagentCommand {
            id: self.state.thread_id().to_owned(),
            command: SubagentCommand::Send {
                text: prompt.to_string(),
            },
        };
        cx.spawn(async move |this, cx| {
            let result = call(client, handle, request).await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(_) => {
                        this.input.update(cx, |input, cx| {
                            input.take_text(cx);
                        });
                        this.error = None;
                    }
                    Err(error) => this.error = Some(format!("Message rejected: {error}").into()),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    fn exit(&mut self, cx: &mut Context<Self>) {
        self.state.exit();
        cx.emit(FocusEvent::Exit);
    }

    /// Hides this thread from the inbox. The daemon keeps its workspace and
    /// transcript, and focus closes once the archive is confirmed.
    fn archive(&mut self, cx: &mut Context<Self>) {
        let client = self.client.clone();
        let handle = self.tokio_handle.clone();
        let request = Request::SubagentCommand {
            id: self.state.thread_id().to_owned(),
            command: SubagentCommand::SetArchived { archived: true },
        };
        cx.spawn(async move |this, cx| {
            let result = call(client, handle, request).await;
            this.update(cx, |this, cx| match result {
                Ok(_) => this.exit(cx),
                Err(error) => {
                    this.error = Some(format!("Archive failed: {error}").into());
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    fn header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .gap_3()
            .px_6()
            .py_3()
            .border_b_1()
            .border_color(rgb(0x27272a))
            .child(
                div()
                    .id("focus-back")
                    .px_3()
                    .py_1p5()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x3f3f46))
                    .bg(rgb(0x27272a))
                    .text_xs()
                    .text_color(rgb(0xd4d4d8))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x3f3f46)).text_color(rgb(0xffffff)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.exit(cx)))
                    .child("← Main"),
            )
            .child(div().flex_1().min_w_0().text_sm().child(self.title.clone()))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x9ca3af))
                    .child(status_label(self.state.status())),
            )
            .child(
                div()
                    .id("focus-archive")
                    .px_3()
                    .py_1p5()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x3f3f46))
                    .bg(rgb(0x27272a))
                    .text_xs()
                    .text_color(rgb(0xd4d4d8))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x3f3f46)).text_color(rgb(0xffffff)))
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.archive(cx)))
                    .child("Archive"),
            )
    }

    /// Mounts a canvas for each desktop artifact in the loaded transcript, so a
    /// finished child's desktop shows in its focus view.
    fn mount_desktop_canvases(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let committed: &[Message] = match self.state.transcript() {
            Ok(messages) => messages,
            Err(_) => &[],
        };
        for messages in [committed, self.state.tail()] {
            mount_canvases(
                cx.entity().downgrade(),
                |view| &mut view.desktop_canvases,
                &mut self.desktop_canvases,
                &self.client,
                &self.tokio_handle,
                messages,
                window,
                cx,
            );
        }
    }

    fn transcript_panel(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut seen_desktops = HashSet::new();
        let tail = self.state.tail();
        let body = match self.state.transcript() {
            Ok(messages) if messages.is_empty() && tail.is_empty() => div()
                .text_sm()
                .text_color(rgb(0x9ca3af))
                .child(empty_transcript_hint(self.state.status())),
            Ok(messages) => {
                div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .children(messages.iter().chain(tail).map(|message| {
                        render_message(message, &self.desktop_canvases, &mut seen_desktops, cx)
                    }))
            }
            Err(err) => div().text_sm().text_color(rgb(0xf87171)).child(err.clone()),
        };
        div()
            .id("focus-transcript")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.transcript_scroll)
            .px_6()
            .py_6()
            .child(body)
    }

    fn composer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let footer = match self.state.message_unavailable() {
            Some(reason) => div()
                .text_sm()
                .text_color(rgb(0x9ca3af))
                .child(reason)
                .into_any_element(),
            None => div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().flex_1().min_w_0().child(self.input.clone()))
                .child(
                    div()
                        .id("focus-send")
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .bg(rgb(0x2563eb))
                        .text_sm()
                        .text_color(rgb(0xffffff))
                        .cursor_pointer()
                        .hover(|s| s.bg(rgb(0x1d4ed8)))
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.submit_message(cx);
                        }))
                        .child("Send"),
                )
                .into_any_element(),
        };
        div()
            .flex()
            .flex_col()
            .gap_2()
            .px_6()
            .py_4()
            .border_t_1()
            .border_color(rgb(0x27272a))
            .when_some(self.error.clone(), |view, error| {
                view.child(div().text_sm().text_color(rgb(0xf87171)).child(error))
            })
            .child(footer)
    }

    fn vcs_panel(&self) -> impl IntoElement {
        let panel = match self.state.vcs() {
            Some(vcs) => vcs.clone().into_any_element(),
            None => div()
                .p_4()
                .text_sm()
                .text_color(rgb(0x9ca3af))
                .child("This thread has no workspace.")
                .into_any_element(),
        };
        div()
            .w(px(380.0))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .bg(rgb(0x18181b))
            .border_l_1()
            .border_color(rgb(0x27272a))
            .child(panel)
    }
}

impl Render for FocusView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.mount_desktop_canvases(window, cx);
        div()
            .flex()
            .flex_1()
            .min_h_0()
            .w_full()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(self.header(cx))
                    .child(self.transcript_panel(cx))
                    .child(self.composer(cx)),
            )
            .child(self.vcs_panel())
    }
}

fn status_label(status: &SubagentStatus) -> String {
    match status {
        SubagentStatus::Queued => "Queued".to_owned(),
        SubagentStatus::Running => "Running".to_owned(),
        SubagentStatus::Verifying => "Verifying".to_owned(),
        SubagentStatus::Reviewing => "Reviewing".to_owned(),
        SubagentStatus::Blocked { prompt, .. } => format!("Blocked: {prompt}"),
        SubagentStatus::Completed => "Completed".to_owned(),
        SubagentStatus::Failed => "Failed".to_owned(),
        SubagentStatus::Cancelled => "Cancelled".to_owned(),
    }
}

fn empty_transcript_hint(status: &SubagentStatus) -> &'static str {
    if status.is_terminal() {
        "No transcript messages."
    } else {
        "No messages yet. Replies stream in here while this thread works."
    }
}

fn render_message(
    message: &Message,
    canvases: &HashMap<String, DesktopCanvas>,
    seen_desktops: &mut HashSet<String>,
    cx: &mut Context<FocusView>,
) -> gpui::Div {
    let author = match message.role {
        MessageRole::User => "You",
        MessageRole::Assistant => "Assistant",
    };
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(div().text_xs().text_color(rgb(0x9ca3af)).child(author))
        .child(div().text_sm().child(message.text.clone()))
        .children(message.desktop_artifacts.iter().map(|artifact| {
            let id = artifact.artifact_id.clone();
            let body = canvas_body(
                canvases,
                artifact,
                seen_desktops,
                cx.listener(move |this, _: &ClickEvent, _, cx| {
                    if let Some(DesktopCanvas::Live(viewer)) = this.desktop_canvases.get(&id) {
                        cx.emit(FocusEvent::ExpandDesktop(viewer.clone()));
                        cx.notify();
                    }
                }),
                cx,
            );
            desktop_artifact_card(artifact, body)
        }))
        .children(
            message
                .views
                .iter()
                .map(|view| render_view(view, &ViewActions::none())),
        )
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

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
    fn messaging_is_enabled_unless_blocked_or_terminal() {
        for status in all_statuses() {
            let enabled = !matches!(
                status,
                SubagentStatus::Blocked { .. } | SubagentStatus::Failed | SubagentStatus::Cancelled
            );
            let state: FocusState<()> = FocusState::enter("t".into(), status.clone(), None, false);
            assert_eq!(state.can_message(), enabled, "status {status:?}");
        }
    }

    #[test]
    fn released_thread_refuses_messages() {
        let state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None, true);
        assert!(!state.can_message());
        assert!(state.message_unavailable().is_some());
    }

    #[test]
    fn exit_releases_the_vcs_handle() {
        let handle = Rc::new(());
        let weak = Rc::downgrade(&handle);
        let mut state =
            FocusState::enter("t".into(), SubagentStatus::Completed, Some(handle), false);
        assert!(state.vcs().is_some());
        state.exit();
        assert!(state.vcs().is_none());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn transcript_loads_once_per_terminal_state() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None, false);
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
    fn running_thread_loads_its_history() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Running, None, false);
        assert!(state.begin_transcript_load().is_some());
    }

    #[test]
    fn resumed_thread_reloads_its_transcript() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None, false);
        let first = state.begin_transcript_load().expect("first load");
        assert!(state.apply_transcript(first, Ok(Vec::new())));
        state.set_status(SubagentStatus::Running);
        assert!(
            state.begin_transcript_load().is_some(),
            "reload after the thread resumes"
        );
    }

    #[test]
    fn stale_transcript_read_is_ignored() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None, false);
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
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Running, None, false);
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
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Running, None, false);
        let token = state.begin_transcript_load().expect("running thread loads");
        state.apply_live(&AgentEvent::MessageStart);
        state.apply_live(&delta("Hi"));
        assert!(state.apply_transcript(token, Ok(Vec::new())));
        assert_eq!(state.tail().len(), 1);
        assert_eq!(state.tail()[0].text.to_string(), "Hi");
    }
}
