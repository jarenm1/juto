//! In-memory chat UI. No provider, persistence, or agent execution is connected.

use std::path::PathBuf;

use gpui::{
    App, ClickEvent, Context, Div, Entity, Focusable, KeyBinding, MouseButton, MouseDownEvent,
    PathPromptOptions, ScrollHandle, SharedString, Subscription, Task, Window, actions, deferred,
    div, prelude::*, px, rgb,
};

use crate::Quit;
use crate::chat_input::{ChatInput, SubmitMessage};

actions!(chat, [CloseAttachmentMenu]);

pub fn init(cx: &mut App) {
    cx.bind_keys([KeyBinding::new(
        "escape",
        CloseAttachmentMenu,
        Some("ChatView"),
    )]);
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Attachment {
    path: PathBuf,
    label: SharedString,
}

impl Attachment {
    fn new(path: PathBuf) -> Self {
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
fn add_attachments(attachments: &mut Vec<Attachment>, paths: Vec<PathBuf>) {
    for path in paths {
        if !attachments.iter().any(|attachment| attachment.path == path) {
            attachments.push(Attachment::new(path));
        }
    }
}

struct Message {
    text: SharedString,
    attachments: Vec<Attachment>,
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
    _subscriptions: [Subscription; 2],
}

impl ChatView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(ChatInput::new);
        input.read(cx).focus_handle(cx).focus(window);
        let submitted =
            cx.subscribe_in(&input, window, |this, _, _: &SubmitMessage, window, cx| {
                this.send_message(window, cx);
            });
        let changed = cx.observe(&input, |_, _, cx| cx.notify());
        Self {
            input,
            messages: Vec::new(),
            attachments: Vec::new(),
            attachment_menu_open: false,
            menu_just_dismissed: false,
            attachment_error: None,
            picker: None,
            scroll: ScrollHandle::new(),
            _subscriptions: [submitted, changed],
        }
    }

    fn can_send(&self, cx: &App) -> bool {
        let input = self.input.read(cx);
        !input.is_composing() && (input.has_text() || !self.attachments.is_empty())
    }

    fn focus_input(&self, window: &mut Window, cx: &App) {
        self.input.read(cx).focus_handle(cx).focus(window);
    }

    fn send_message(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.can_send(cx) {
            return;
        }
        let text = self.input.update(cx, |input, cx| input.take_text(cx));
        self.messages.push(Message {
            text: if text.trim().is_empty() {
                SharedString::default()
            } else {
                text
            },
            attachments: std::mem::take(&mut self.attachments),
        });
        self.attachment_error = None;
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
                                        .child("Agent runtime is not connected."),
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
                .child("You"),
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
}
