//! Focus mode for a focus thread: its transcript, a follow-up composer, and the
//! VCS panel rooted at its workspace.

use std::sync::Arc;

use gpui::{
    ClickEvent, Context, Entity, EventEmitter, IntoElement, Render, ScrollHandle, SharedString,
    Subscription, Task, Window, div, prelude::*, px, rgb,
};
use juto_runtime::subagents::{ChildState, SubagentManager, SubagentStatus, SubagentSummary};
use tokio::sync::watch;

use crate::chat::{Message, MessageRole, view_messages};
use crate::chat_input::{ChatInput, SubmitMessage};
use crate::vcs_sidebar::VcsSidebar;

/// Why the follow-up composer is disabled for `status`, or `None` when enabled.
pub fn follow_up_unavailable_reason(status: &SubagentStatus) -> Option<&'static str> {
    match status {
        SubagentStatus::Completed => None,
        SubagentStatus::Queued
        | SubagentStatus::Running
        | SubagentStatus::Verifying
        | SubagentStatus::Reviewing => {
            Some("The thread is still working. Follow-ups open when it completes.")
        }
        SubagentStatus::Blocked { .. } => {
            Some("The thread is blocked. Answer its request before following up.")
        }
        SubagentStatus::Failed => Some("The thread failed, so it cannot take follow-ups."),
        SubagentStatus::Cancelled => {
            Some("The thread was cancelled, so it cannot take follow-ups.")
        }
    }
}

/// Pure focus-mode state. `V` is the VCS panel handle, kept generic so the
/// transitions can be tested without GPUI.
pub struct FocusState<V> {
    thread_id: String,
    status: SubagentStatus,
    vcs: Option<V>,
    transcript: Result<Vec<Message>, String>,
    transcript_fresh: bool,
    pending_load: Option<u64>,
    next_load: u64,
}

impl<V> FocusState<V> {
    pub fn enter(thread_id: String, status: SubagentStatus, vcs: Option<V>) -> Self {
        Self {
            thread_id,
            status,
            vcs,
            transcript: Ok(Vec::new()),
            transcript_fresh: false,
            pending_load: None,
            next_load: 0,
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

    pub fn can_follow_up(&self) -> bool {
        follow_up_unavailable_reason(&self.status).is_none()
    }

    /// Drops the VCS panel handle. The thread itself keeps running.
    pub fn exit(&mut self) {
        self.vcs = None;
    }

    /// Records a status update. A non-terminal status invalidates the transcript,
    /// so the next terminal state reloads it and any read still in flight is ignored.
    pub fn set_status(&mut self, status: SubagentStatus) {
        if !status.is_terminal() {
            self.transcript_fresh = false;
            self.pending_load = None;
        }
        self.status = status;
    }

    /// Returns a load token when a transcript read is due: the thread is terminal,
    /// the transcript is not already fresh for this terminal state, and no read is pending.
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
        self.transcript = transcript;
        true
    }
}

/// Emitted when the user leaves focus mode.
pub enum FocusEvent {
    Exit,
}

pub struct FocusView {
    state: FocusState<Entity<VcsSidebar>>,
    title: SharedString,
    subagents: Arc<SubagentManager>,
    input: Entity<ChatInput>,
    transcript_scroll: ScrollHandle,
    error: Option<SharedString>,
    _input_subscription: Subscription,
    _updates: Task<()>,
}

impl EventEmitter<FocusEvent> for FocusView {}

impl FocusView {
    /// `updates` is the child's state stream. The view reacts to each change event,
    /// with no polling. Dropping the view cancels the stream task.
    pub fn new(
        summary: SubagentSummary,
        mut updates: watch::Receiver<ChildState>,
        subagents: Arc<SubagentManager>,
        tokio_handle: tokio::runtime::Handle,
        cx: &mut Context<Self>,
    ) -> Self {
        let status = updates.borrow().status.clone();
        let vcs = summary
            .workspace
            .map(|workspace| cx.new(|cx| VcsSidebar::new(workspace, tokio_handle, cx)));
        let input = cx.new(ChatInput::new);
        let input_subscription = cx.subscribe(&input, |this, _, _: &SubmitMessage, cx| {
            this.submit_follow_up(cx);
        });
        let updates_task = cx.spawn(async move |this, cx| {
            loop {
                let child = updates.borrow_and_update().clone();
                if this
                    .update(cx, |this, cx| this.apply_child_state(child, cx))
                    .is_err()
                {
                    break;
                }
                if updates.changed().await.is_err() {
                    break;
                }
            }
        });
        Self {
            state: FocusState::enter(summary.id, status, vcs),
            title: summary.name.unwrap_or(summary.task).into(),
            subagents,
            input,
            transcript_scroll: ScrollHandle::new(),
            error: None,
            _input_subscription: input_subscription,
            _updates: updates_task,
        }
    }

    fn apply_child_state(&mut self, child: ChildState, cx: &mut Context<Self>) {
        self.state.set_status(child.status);
        if let Some(token) = self.state.begin_transcript_load() {
            self.load_transcript(token, cx);
        }
        cx.notify();
    }

    /// Reads the child's session on the background executor, then converts the
    /// messages on the foreground.
    fn load_transcript(&self, token: u64, cx: &mut Context<Self>) {
        let subagents = Arc::clone(&self.subagents);
        let thread_id = self.state.thread_id().to_owned();
        cx.spawn(async move |this, cx| {
            let transcript = cx
                .background_executor()
                .spawn(async move {
                    subagents
                        .transcript(&thread_id)
                        .map_err(|err| err.to_string())
                })
                .await;
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

    fn submit_follow_up(&mut self, cx: &mut Context<Self>) {
        if !self.state.can_follow_up() || !self.input.read(cx).has_text() {
            return;
        }
        let prompt = self.input.read(cx).text();
        match self
            .subagents
            .follow_up_from_user(self.state.thread_id(), prompt.to_string())
        {
            Ok(()) => {
                self.input.update(cx, |input, cx| {
                    input.take_text(cx);
                });
                self.error = None;
            }
            Err(err) => self.error = Some(format!("Follow-up rejected: {err}").into()),
        }
        cx.notify();
    }

    fn exit(&mut self, cx: &mut Context<Self>) {
        self.state.exit();
        cx.emit(FocusEvent::Exit);
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
    }

    fn transcript_panel(&self) -> impl IntoElement {
        let body = match self.state.transcript() {
            Ok(messages) if messages.is_empty() => div()
                .text_sm()
                .text_color(rgb(0x9ca3af))
                .child(empty_transcript_hint(self.state.status())),
            Ok(messages) => div()
                .flex()
                .flex_col()
                .gap_4()
                .children(messages.iter().map(render_message)),
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
        let footer = match follow_up_unavailable_reason(self.state.status()) {
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
                            this.submit_follow_up(cx);
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
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
                    .child(self.transcript_panel())
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
        "The transcript appears when this thread finishes. Live streaming is not shown."
    }
}

fn render_message(message: &Message) -> impl IntoElement {
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
    fn follow_up_is_enabled_only_when_completed() {
        for status in all_statuses() {
            let expected = status == SubagentStatus::Completed;
            let state: FocusState<()> = FocusState::enter("t".into(), status.clone(), None);
            assert_eq!(state.can_follow_up(), expected, "status {status:?}");
            assert_eq!(
                follow_up_unavailable_reason(&status).is_none(),
                expected,
                "status {status:?}"
            );
        }
    }

    #[test]
    fn exit_releases_the_vcs_handle() {
        let handle = Rc::new(());
        let weak = Rc::downgrade(&handle);
        let mut state = FocusState::enter("t".into(), SubagentStatus::Completed, Some(handle));
        assert!(state.vcs().is_some());
        state.exit();
        assert!(state.vcs().is_none());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn transcript_loads_once_per_terminal_state() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None);
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
    fn running_thread_has_no_transcript_load() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Running, None);
        assert!(state.begin_transcript_load().is_none());
    }

    #[test]
    fn next_terminal_state_reloads_transcript() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None);
        let first = state.begin_transcript_load().expect("first load");
        assert!(state.apply_transcript(first, Ok(Vec::new())));
        state.set_status(SubagentStatus::Running);
        assert!(
            state.begin_transcript_load().is_none(),
            "no load while running"
        );
        state.set_status(SubagentStatus::Completed);
        assert!(
            state.begin_transcript_load().is_some(),
            "reload after new turn"
        );
    }

    #[test]
    fn stale_transcript_read_is_ignored() {
        let mut state: FocusState<()> =
            FocusState::enter("t".into(), SubagentStatus::Completed, None);
        let stale = state.begin_transcript_load().expect("first load");
        state.set_status(SubagentStatus::Running);
        state.set_status(SubagentStatus::Completed);
        let current = state.begin_transcript_load().expect("reload");
        assert!(!state.apply_transcript(stale, Err("stale".into())));
        assert!(state.apply_transcript(current, Ok(Vec::new())));
        assert!(state.transcript().is_ok());
    }
}
