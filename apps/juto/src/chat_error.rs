//! Window content shown when Juto cannot reach its workspace daemon.

use gpui::{Context, Render, SharedString, Window, div, prelude::*, px, rgb};

use crate::theme;

/// Explains why the chat window could not start. Replaces the chat view so the
/// failure is visible instead of falling back to an in-process runtime.
pub struct StartupError {
    message: SharedString,
}

impl StartupError {
    pub fn new(message: String) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Render for StartupError {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(rgb(theme::BG))
            .text_color(rgb(theme::TEXT))
            .p_8()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .max_w(px(640.))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .child("Juto could not connect to the workspace daemon"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(theme::DANGER))
                            .child(self.message.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_MUTED))
                            .child("Restart Juto after resolving the error above."),
                    ),
            )
    }
}
