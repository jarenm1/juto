//! Chat UI wired to the agent runtime and session persistence.

use std::path::PathBuf;
use std::sync::Arc;

use crate::Quit;
use crate::chat_input::{ChatInput, SubmitMessage};
use gpui::{
    App, ClickEvent, Context, Div, Entity, Focusable, IntoElement, KeyBinding, MouseButton,
    MouseDownEvent, PathPromptOptions, Render, ScrollHandle, SharedString, Subscription, Task,
    Window, actions, deferred, div, prelude::*, px, rgb,
};
use juto_agent::{AgentEvent, RunControl, RunOutcome};
use juto_ai::{Message as AiMessage, ProviderEvent, StopReason};
use juto_runtime::{EntryKind, Runtime, Session};
use tokio::sync::Mutex;

actions!(chat, [CloseAttachmentMenu]);

pub fn init(cx: &mut App) {
    cx.bind_keys([KeyBinding::new(
        "escape",
        CloseAttachmentMenu,
        Some("ChatView"),
    )]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attachment {
    pub path: PathBuf,
    pub label: SharedString,
}

impl Attachment {
    pub fn new(path: PathBuf) -> Self {
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        Self {
            path,
            label: label.into(),
        }
    }
}

/// Appends newly picked paths, keeping the first entry for any path already attached.
pub fn add_attachments(attachments: &mut Vec<Attachment>, paths: Vec<PathBuf>) {
    for path in paths {
        if !attachments.iter().any(|attachment| attachment.path == path) {
            attachments.push(Attachment::new(path));
        }
    }
}

pub fn format_prompt(text: &str, attachments: &[Attachment]) -> String {
    let mut prompt = String::new();
    if !attachments.is_empty() {
        prompt.push_str("Attached files:\n");
        for attachment in attachments {
            prompt.push_str(&format!("- {}\n", attachment.path.display()));
        }
        if !text.trim().is_empty() {
            prompt.push('\n');
        }
    }
    if !text.trim().is_empty() {
        prompt.push_str(text);
    }
    prompt
}

pub fn load_messages(session: &Session) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Ok(entries) = session.active_entries() {
        for entry in entries {
            if let EntryKind::Message { message } = &entry.kind {
                match message {
                    AiMessage::User { .. } => {
                        messages.push(Message {
                            role: MessageRole::User,
                            text: message.text().into(),
                            attachments: Vec::new(),
                        });
                    }
                    AiMessage::Assistant(assistant) => {
                        let text = assistant.text();
                        if !text.is_empty() {
                            messages.push(Message {
                                role: MessageRole::Assistant,
                                text: text.into(),
                                attachments: Vec::new(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    messages
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: MessageRole,
    pub text: SharedString,
    pub attachments: Vec<Attachment>,
}

pub struct ChatView {
    input: Entity<ChatInput>,
    messages: Vec<Message>,
    attachments: Vec<Attachment>,
    attachment_menu_open: bool,
    menu_just_dismissed: bool,
    attachment_error: Option<SharedString>,
    picker: Option<Task<()>>,
    scroll: ScrollHandle,
    runtime: Runtime,
    session: Arc<Mutex<Session>>,
    tokio_handle: tokio::runtime::Handle,
    is_running: bool,
    active_control: Option<RunControl>,
    run_task: Option<Task<()>>,
    _subscriptions: [Subscription; 2],
}

impl Drop for ChatView {
    fn drop(&mut self) {
        if let Some(control) = self.active_control.take() {
            control.cancel();
        }
    }
}

impl ChatView {
    pub fn new(
        runtime: Runtime,
        session: Session,
        tokio_handle: tokio::runtime::Handle,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(ChatInput::new);
        input.read(cx).focus_handle(cx).focus(window);
        let submitted =
            cx.subscribe_in(&input, window, |this, _, _: &SubmitMessage, window, cx| {
                this.send_message(window, cx);
            });
        let changed = cx.observe(&input, |_, _, cx| cx.notify());
        let messages = load_messages(&session);
        Self {
            input,
            messages,
            attachments: Vec::new(),
            attachment_menu_open: false,
            menu_just_dismissed: false,
            attachment_error: None,
            picker: None,
            scroll: ScrollHandle::new(),
            runtime,
            session: Arc::new(Mutex::new(session)),
            tokio_handle,
            is_running: false,
            active_control: None,
            run_task: None,
            _subscriptions: [submitted, changed],
        }
    }

    fn can_send(&self, cx: &App) -> bool {
        let input = self.input.read(cx);
        !self.is_running
            && !input.is_composing()
            && (input.has_text() || !self.attachments.is_empty())
    }

    fn focus_input(&self, window: &mut Window, cx: &App) {
        self.input.read(cx).focus_handle(cx).focus(window);
    }

    fn send_message(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.can_send(cx) {
            return;
        }
        let text = self.input.update(cx, |input, cx| input.take_text(cx));
        let attachments = std::mem::take(&mut self.attachments);
        let prompt = format_prompt(&text, &attachments);

        self.messages.push(Message {
            role: MessageRole::User,
            text: if text.trim().is_empty() {
                SharedString::default()
            } else {
                text
            },
            attachments,
        });
        self.attachment_error = None;
        self.scroll.scroll_to_bottom();
        self.focus_input(window, cx);
        self.is_running = true;

        let control = RunControl::new();
        self.active_control = Some(control.clone());
        let runtime = self.runtime.clone();
        let session = self.session.clone();
        let tokio_handle = self.tokio_handle.clone();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();

        let run_future = tokio_handle.spawn(async move {
            let mut session_guard = session.lock().await;
            runtime
                .run(&mut session_guard, prompt, events_tx, control)
                .await
        });

        self.run_task = Some(cx.spawn_in(window, async move |this, cx| {
            while let Some(event) = events_rx.recv().await {
                let updated = this.update_in(cx, |this, _window, cx| {
                    this.handle_agent_event(event, cx);
                });
                if updated.is_err() {
                    break;
                }
            }

            let outcome = run_future.await;
            this.update_in(cx, |this, window, cx| {
                this.finish_run(outcome, window, cx);
            })
            .ok();
        }));

        cx.notify();
    }

    fn handle_agent_event(&mut self, event: AgentEvent, cx: &mut Context<Self>) {
        match event {
            AgentEvent::MessageStart => {
                self.messages.push(Message {
                    role: MessageRole::Assistant,
                    text: SharedString::default(),
                    attachments: Vec::new(),
                });
                self.scroll.scroll_to_bottom();
                cx.notify();
            }
            AgentEvent::MessageUpdate {
                event: ProviderEvent::TextDelta { delta, .. },
            } => {
                if let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    let mut text = last.text.to_string();
                    text.push_str(&delta);
                    last.text = text.into();
                    self.scroll.scroll_to_bottom();
                    cx.notify();
                }
            }
            AgentEvent::MessageEnd { message } => {
                let text = message.text();
                if let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    last.text = text.into();
                    self.scroll.scroll_to_bottom();
                    cx.notify();
                }
            }
            AgentEvent::Error { error } => {
                let error_text = format!("Error: {error}");
                if let Some(last) = self.messages.last_mut() {
                    if last.role == MessageRole::Assistant && last.text.is_empty() {
                        last.text = error_text.into();
                    } else if last.role == MessageRole::Assistant {
                        let mut text = last.text.to_string();
                        text.push_str(&format!("\n\n{error_text}"));
                        last.text = text.into();
                    } else {
                        self.messages.push(Message {
                            role: MessageRole::Assistant,
                            text: error_text.into(),
                            attachments: Vec::new(),
                        });
                    }
                } else {
                    self.messages.push(Message {
                        role: MessageRole::Assistant,
                        text: error_text.into(),
                        attachments: Vec::new(),
                    });
                }
                self.scroll.scroll_to_bottom();
                cx.notify();
            }
            _ => {}
        }
    }

    fn finish_run(
        &mut self,
        outcome: Result<Result<RunOutcome, anyhow::Error>, tokio::task::JoinError>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.is_running = false;
        self.active_control = None;
        self.run_task = None;
        match outcome {
            Ok(Ok(outcome)) => {
                if outcome.stop_reason == StopReason::Aborted
                    && let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                {
                    let mut text = last.text.to_string();
                    if !text.is_empty() {
                        text.push_str("\n\n[Run cancelled]");
                    } else {
                        text.push_str("[Run cancelled]");
                    }
                    last.text = text.into();
                } else if outcome.stop_reason == StopReason::Error
                    && let Some(last) = self.messages.last_mut()
                    && last.role == MessageRole::Assistant
                    && last.text.is_empty()
                {
                    last.text = "Error: Run failed".into();
                }
            }
            Ok(Err(err)) => {
                let error_msg = format!("Error: {err}");
                if let Some(last) = self.messages.last_mut() {
                    if last.role == MessageRole::Assistant && last.text.is_empty() {
                        last.text = error_msg.into();
                    } else if last.role == MessageRole::Assistant {
                        let mut text = last.text.to_string();
                        text.push_str(&format!("\n\n{error_msg}"));
                        last.text = text.into();
                    } else {
                        self.messages.push(Message {
                            role: MessageRole::Assistant,
                            text: error_msg.into(),
                            attachments: Vec::new(),
                        });
                    }
                } else {
                    self.messages.push(Message {
                        role: MessageRole::Assistant,
                        text: error_msg.into(),
                        attachments: Vec::new(),
                    });
                }
            }
            Err(join_err) => {
                let error_msg = format!("Error: Task panicked or was cancelled: {join_err}");
                self.messages.push(Message {
                    role: MessageRole::Assistant,
                    text: error_msg.into(),
                    attachments: Vec::new(),
                });
            }
        }
        self.scroll.scroll_to_bottom();
        self.focus_input(window, cx);
        cx.notify();
    }

    fn close_attachment_menu(
        &mut self,
        _: &CloseAttachmentMenu,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.attachment_menu_open {
            self.attachment_menu_open = false;
            cx.notify();
        } else {
            cx.propagate();
        }
    }

    fn open_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.attachment_menu_open = false;
        cx.notify();
        if self.picker.is_some() {
            return;
        }
        self.attachment_error = None;
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Attach".into()),
        });
        self.picker = Some(cx.spawn_in(window, async move |this, cx| {
            let result = paths.await;
            this.update_in(cx, |this, window, cx| {
                this.picker = None;
                match result {
                    Ok(Ok(Some(paths))) => add_attachments(&mut this.attachments, paths),
                    Ok(Ok(None)) => {}
                    Ok(Err(error)) => {
                        this.attachment_error =
                            Some(format!("Couldn't open the file picker: {error}").into());
                    }
                    Err(_) => {
                        this.attachment_error =
                            Some("The file picker closed without a response.".into());
                    }
                }
                this.focus_input(window, cx);
                cx.notify();
            })
            .ok();
        }));
    }

    fn header(&self) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .flex_none()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(rgb(0x333333))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().text_size(px(21.)).child("Juto"))
                    .child(div().text_sm().text_color(rgb(0xa0a0a0)).child("Chat")),
            )
    }

    fn message_list(&self) -> impl IntoElement {
        div()
            .id("messages")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .px_6()
            .py_6()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .w_full()
                    .max_w(px(800.))
                    .mx_auto()
                    .when(self.messages.is_empty(), |view| {
                        view.child(
                            div()
                                .py_12()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap_3()
                                .child(div().text_xl().child("Start a conversation"))
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0xa0a0a0))
                                        .child("Type a message below to try the chat view."),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgb(0x808080))
                                        .child("Agent runtime connected."),
                                ),
                        )
                    })
                    .children(
                        self.messages
                            .iter()
                            .enumerate()
                            .map(|(index, message)| message_row(index, message)),
                    ),
            )
    }

    fn attachment_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("attachment-menu")
            .occlude()
            .on_mouse_down_out(cx.listener(|this, _: &MouseDownEvent, _, cx| {
                this.attachment_menu_open = false;
                this.menu_just_dismissed = true;
                cx.notify();
            }))
            .w(px(240.))
            .p_1()
            .rounded_lg()
            .border_1()
            .border_color(rgb(0x3a3a3a))
            .bg(rgb(0x262626))
            .shadow_lg()
            .child(menu_item(
                "attach-files",
                "Files…",
                "Attach one or more files",
                cx.listener(|this, _: &ClickEvent, window, cx| this.open_picker(window, cx)),
            ))
    }

    fn composer(&self, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let can_send = self.can_send(cx);
        let focused = self.input.read(cx).focus_handle(cx).is_focused(window);
        let menu_open = self.attachment_menu_open;

        let attach_button = div()
            .id("attach")
            .flex()
            .items_center()
            .justify_center()
            .size_8()
            .rounded_full()
            .border_1()
            .border_color(rgb(0x444444))
            .text_size(px(20.))
            .text_color(rgb(0xcccccc))
            .cursor_pointer()
            .when(menu_open, |button| button.bg(rgb(0x333333)))
            .hover(|style| style.bg(rgb(0x2e2e2e)))
            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                if this.menu_just_dismissed {
                    this.menu_just_dismissed = false;
                    return;
                }
                this.attachment_menu_open = !this.attachment_menu_open;
                cx.notify();
            }))
            .child("+");

        let send_button = div()
            .id("send-message")
            .flex()
            .items_center()
            .justify_center()
            .size_8()
            .rounded_full()
            .text_size(px(17.))
            .bg(if can_send {
                rgb(0xd4d4d4)
            } else {
                rgb(0x333333)
            })
            .text_color(if can_send {
                rgb(0x181818)
            } else {
                rgb(0x707070)
            })
            .when(can_send, |button| {
                button
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(0xeeeeee)))
            })
            .on_click(cx.listener(|this, _: &ClickEvent, window, cx| this.send_message(window, cx)))
            .child("↑");

        div().flex_none().px_6().pt_2().pb_4().child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .w_full()
                .max_w(px(800.))
                .mx_auto()
                .when_some(self.attachment_error.clone(), |column, error| {
                    column.child(div().text_xs().text_color(rgb(0xb0b0b0)).child(error))
                })
                .child(
                    div()
                        .id("composer")
                        .flex()
                        .flex_col()
                        .gap_2()
                        .p_2()
                        .rounded_xl()
                        .border_1()
                        .border_color(if focused {
                            rgb(0x6a6a6a)
                        } else {
                            rgb(0x3a3a3a)
                        })
                        .bg(rgb(0x222222))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _: &MouseDownEvent, window, cx| {
                                this.focus_input(window, cx)
                            }),
                        )
                        .when(!self.attachments.is_empty(), |card| {
                            card.child(
                                div().flex().flex_wrap().gap_2().px_1().pt_1().children(
                                    self.attachments.iter().enumerate().map(
                                        |(index, attachment)| {
                                            attachment_chip(attachment.label.clone()).child(
                                                div()
                                                    .id(("remove-attachment", index))
                                                    .flex_none()
                                                    .px_1()
                                                    .rounded_sm()
                                                    .text_color(rgb(0x909090))
                                                    .cursor_pointer()
                                                    .hover(|style| {
                                                        style
                                                            .bg(rgb(0x333333))
                                                            .text_color(rgb(0xe7e7e7))
                                                    })
                                                    .on_mouse_down(
                                                        MouseButton::Left,
                                                        cx.listener(
                                                            move |this, _: &MouseDownEvent, window, cx| {
                                                                cx.stop_propagation();
                                                                if index < this.attachments.len() {
                                                                    this.attachments.remove(index);
                                                                    this.focus_input(window, cx);
                                                                    cx.notify();
                                                                }
                                                            },
                                                        ),
                                                    )
                                                    .child("×"),
                                            )
                                        },
                                    ),
                                ),
                            )
                        })
                        .child(self.input.clone())
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(div().relative().child(attach_button).when(
                                    menu_open,
                                    |anchor| {
                                        anchor.child(
                                            deferred(
                                                div()
                                                    .absolute()
                                                    .bottom(px(40.))
                                                    .left_0()
                                                    .child(self.attachment_menu(cx)),
                                            )
                                            .with_priority(1),
                                        )
                                    },
                                ))
                                .child(send_button),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .justify_end()
                        .text_xs()
                        .text_color(rgb(0x808080))
                        .child("Enter to send"),
                ),
        )
    }
}

fn attachment_chip(label: SharedString) -> Div {
    div()
        .flex()
        .items_center()
        .gap_1()
        .max_w(px(260.))
        .pl_2()
        .pr_1()
        .py_1()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x3a3a3a))
        .bg(rgb(0x2a2a2a))
        .text_sm()
        .child(
            div()
                .min_w_0()
                .overflow_hidden()
                .text_ellipsis()
                .whitespace_nowrap()
                .child(label),
        )
}

fn menu_item(
    id: &'static str,
    title: &'static str,
    detail: &'static str,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .flex()
        .flex_col()
        .gap_0p5()
        .px_3()
        .py_2()
        .rounded_md()
        .cursor_pointer()
        .hover(|style| style.bg(rgb(0x333333)))
        .on_click(on_click)
        .child(div().text_sm().child(title))
        .child(div().text_xs().text_color(rgb(0x909090)).child(detail))
}

fn message_row(index: usize, message: &Message) -> impl IntoElement {
    let author = match message.role {
        MessageRole::User => "You",
        MessageRole::Assistant => "Assistant",
    };
    div()
        .id(("message", index))
        .flex_none()
        .w_full()
        .min_w_0()
        .child(
            div()
                .mb_2()
                .text_xs()
                .text_color(rgb(0xa0a0a0))
                .child(author),
        )
        .when(!message.text.is_empty(), |row| {
            row.child(
                div()
                    .w_full()
                    .text_size(px(15.))
                    .child(message.text.clone()),
            )
        })
        .when(!message.attachments.is_empty(), |row| {
            row.child(
                div().mt_2().flex().flex_wrap().gap_2().children(
                    message
                        .attachments
                        .iter()
                        .map(|attachment| attachment_chip(attachment.label.clone()).pr_2()),
                ),
            )
        })
}

impl Render for ChatView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x181818))
            .text_color(rgb(0xe7e7e7))
            .key_context("ChatView")
            .on_action(|_: &Quit, _, cx| cx.quit())
            .on_action(cx.listener(Self::close_attachment_menu))
            .child(self.header())
            .child(self.message_list())
            .child(self.composer(window, cx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(attachments: &[Attachment]) -> Vec<&str> {
        attachments
            .iter()
            .map(|attachment| attachment.label.as_ref())
            .collect()
    }

    #[test]
    fn picking_an_already_attached_path_keeps_one_chip_in_original_order() {
        let mut attachments = Vec::new();
        add_attachments(
            &mut attachments,
            vec!["/work/notes.md".into(), "/work/plan.txt".into()],
        );
        add_attachments(
            &mut attachments,
            vec!["/work/plan.txt".into(), "/work/diagram.png".into()],
        );
        assert_eq!(
            labels(&attachments),
            ["notes.md", "plan.txt", "diagram.png"]
        );
    }

    #[test]
    fn prompt_formatting_combines_attachments_and_text() {
        let attachments = vec![
            Attachment::new("/tmp/alpha.rs".into()),
            Attachment::new("/tmp/beta.txt".into()),
        ];
        let formatted = format_prompt("Explain this code", &attachments);
        assert_eq!(
            formatted,
            "Attached files:\n- /tmp/alpha.rs\n- /tmp/beta.txt\n\nExplain this code"
        );

        let attachments_only = format_prompt("", &attachments);
        assert_eq!(
            attachments_only,
            "Attached files:\n- /tmp/alpha.rs\n- /tmp/beta.txt\n"
        );

        let text_only = format_prompt("Hello world", &[]);
        assert_eq!(text_only, "Hello world");
    }

    #[test]
    fn load_messages_extracts_user_and_assistant_history() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cwd = std::env::current_dir().unwrap();
        let mut session = Session::create(temp_dir.path(), &cwd).unwrap();
        session
            .append(EntryKind::Message {
                message: AiMessage::user("Hello from user"),
            })
            .unwrap();
        let assistant = juto_ai::AssistantMessage {
            content: vec![juto_ai::ContentBlock::Text {
                text: "Hello from assistant".into(),
            }],
            api: "test".into(),
            provider: "test".into(),
            model: "test-model".into(),
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            timestamp: 0,
            response_id: None,
            error: None,
        };
        session
            .append(EntryKind::Message {
                message: AiMessage::Assistant(assistant),
            })
            .unwrap();

        let messages = load_messages(&session);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[0].text.as_ref(), "Hello from user");
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].text.as_ref(), "Hello from assistant");
    }
}
