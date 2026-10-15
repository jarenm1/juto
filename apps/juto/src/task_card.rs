//! Focus-thread row for the sidebar.
//!
//! Collapsed, a row shows the thread title, its status, and a thin lifecycle
//! stage bar. Blocked threads always show their blocker prompt and a resume
//! field. Expanding a row reveals metadata, steering controls (Pause, Steer,
//! Abort), token usage, the step log, and attached diffs. Clicking the row
//! selects the thread.

use gpui::{
    App, ClickEvent, Context, Div, Entity, FocusHandle, Focusable, IntoElement, KeyBinding, Render,
    Rgba, SharedString, Stateful, Window, actions, div, prelude::*, px, relative, rgb,
};
use juto_runtime::subagents::{BlockerReason, SubagentStatus};

use crate::text_input::TextInput;
use crate::theme;

actions!(
    task_card,
    [
        ToggleTaskCardExpanded,
        SelectTaskCard,
        PauseTaskCard,
        AbortTaskCard,
        SubmitTaskResolution,
    ]
);

pub fn init(cx: &mut App) {
    cx.bind_keys([KeyBinding::new(
        "enter",
        SubmitTaskResolution,
        Some("TaskCard"),
    )]);
}

pub fn status_label(status: &SubagentStatus) -> &'static str {
    match status {
        SubagentStatus::Queued => "Queued",
        SubagentStatus::Running => "Running",
        SubagentStatus::Blocked { .. } => "Blocked",
        SubagentStatus::Verifying => "Verifying",
        SubagentStatus::Reviewing => "Reviewing",
        SubagentStatus::Completed => "Done",
        SubagentStatus::Failed => "Failed",
        SubagentStatus::Cancelled => "Cancelled",
    }
}

/// Color of a status label and its stage-bar fill.
pub fn status_color(status: &SubagentStatus) -> Rgba {
    match status {
        SubagentStatus::Blocked { .. } => rgb(theme::WARNING),
        SubagentStatus::Failed => rgb(theme::DANGER),
        SubagentStatus::Queued | SubagentStatus::Cancelled => rgb(theme::TEXT_FAINT),
        SubagentStatus::Running
        | SubagentStatus::Verifying
        | SubagentStatus::Reviewing
        | SubagentStatus::Completed => rgb(theme::ACCENT),
    }
}

/// Position in the lifecycle queued → running → verifying → reviewing →
/// completed, as a fraction in `0.0..=1.0`. This is a stage indicator, not a
/// work-completion estimate: the runtime reports no progress measure. Blocked
/// threads hold the running position; failed and cancelled threads are over.
pub fn lifecycle_stage(status: &SubagentStatus) -> f32 {
    match status {
        SubagentStatus::Queued => 0.0,
        SubagentStatus::Running | SubagentStatus::Blocked { .. } => 0.25,
        SubagentStatus::Verifying => 0.5,
        SubagentStatus::Reviewing => 0.75,
        SubagentStatus::Completed | SubagentStatus::Failed | SubagentStatus::Cancelled => 1.0,
    }
}

pub fn is_status_active(status: &SubagentStatus) -> bool {
    matches!(
        status,
        SubagentStatus::Running
            | SubagentStatus::Verifying
            | SubagentStatus::Reviewing
            | SubagentStatus::Blocked { .. }
    )
}

/// Token usage statistics for a task card.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenUsageModel {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub total_tokens: u64,
}

#[allow(dead_code)]
impl TokenUsageModel {
    pub fn new(input: u64, output: u64, total: u64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            total_tokens: total,
        }
    }
}

/// Diff line representation for live Zed Delta preview.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffLineKind {
    Context,
    Add,
    Remove,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffLineModel {
    pub kind: DiffLineKind,
    pub old_num: Option<usize>,
    pub new_num: Option<usize>,
    pub content: String,
}

/// A diff hunk anchored to conversational reasoning / turn ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffHunkModel {
    pub header: String,
    pub turn_anchor: Option<String>,
    pub lines: Vec<DiffLineModel>,
}

/// File-level diff attached to a subagent task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffFileModel {
    pub path: String,
    pub additions: usize,
    pub deletions: usize,
    pub hunks: Vec<DiffHunkModel>,
}

/// Tool execution or activity step log entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolLogEntry {
    pub step: usize,
    pub tool_name: String,
    pub summary: String,
    pub status: String,
}

/// Full data model representing a subagent task in Mission Control.
#[derive(Clone, Debug, PartialEq)]
pub struct TaskCardModel {
    pub id: String,
    pub name: Option<String>,
    pub task: String,
    pub agent: String,
    pub model: Option<String>,
    pub workspace: Option<String>,
    pub status: SubagentStatus,
    pub current_step: Option<String>,
    pub elapsed_seconds: u64,
    pub usage: TokenUsageModel,
    pub cost_estimate: Option<f64>,
    pub diffs: Vec<DiffFileModel>,
    pub logs: Vec<ToolLogEntry>,
    pub expanded: bool,
    pub resolution_input: String,
    pub steer_input: String,
    pub show_steer_box: bool,
}

#[allow(dead_code)]
impl TaskCardModel {
    pub fn new(id: impl Into<String>, task: impl Into<String>, agent: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            task: task.into(),
            agent: agent.into(),
            model: None,
            workspace: None,
            status: SubagentStatus::Queued,
            current_step: None,
            elapsed_seconds: 0,
            usage: TokenUsageModel::default(),
            cost_estimate: None,
            diffs: Vec::new(),
            logs: Vec::new(),
            expanded: false,
            resolution_input: String::new(),
            steer_input: String::new(),
            show_steer_box: false,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_status(mut self, status: SubagentStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_workspace(mut self, ws: impl Into<String>) -> Self {
        self.workspace = Some(ws.into());
        self
    }

    pub fn with_current_step(mut self, step: impl Into<String>) -> Self {
        self.current_step = Some(step.into());
        self
    }

    pub fn with_elapsed(mut self, seconds: u64) -> Self {
        self.elapsed_seconds = seconds;
        self
    }

    pub fn with_diff(mut self, diff: DiffFileModel) -> Self {
        self.diffs.push(diff);
        self
    }

    pub fn with_log(mut self, log: ToolLogEntry) -> Self {
        self.logs.push(log);
        self
    }

    pub fn total_diff_stats(&self) -> (usize, usize) {
        let mut adds = 0;
        let mut dels = 0;
        for f in &self.diffs {
            adds += f.additions;
            dels += f.deletions;
        }
        (adds, dels)
    }

    pub fn display_title(&self) -> String {
        if let Some(name) = &self.name {
            name.clone()
        } else if self.task.len() > 38 {
            format!("{}…", &self.task[..35])
        } else {
            self.task.clone()
        }
    }

    pub fn format_duration(&self) -> String {
        let mins = self.elapsed_seconds / 60;
        let secs = self.elapsed_seconds % 60;
        if mins > 0 {
            format!("{}m {:02}s", mins, secs)
        } else {
            format!("{}s", secs)
        }
    }
}

/// Interactive events emitted by a TaskCard.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub enum TaskCardEvent {
    ToggleExpanded(String),
    Selected(String),
    Pause(String),
    Resume { id: String, resolution: String },
    Abort(String),
    Steer { id: String, guidance: String },
}

/// GPUI View wrapper for rendering a TaskCard.
pub struct TaskCard {
    pub model: TaskCardModel,
    pub is_selected: bool,
    focus_handle: FocusHandle,
    pub resolution_input: Entity<TextInput>,
    pub steer_input: Entity<TextInput>,
}

impl gpui::EventEmitter<TaskCardEvent> for TaskCard {}

#[allow(dead_code)]
impl TaskCard {
    pub fn new(model: TaskCardModel, is_selected: bool, cx: &mut Context<Self>) -> Self {
        let resolution_input = cx.new(|cx| {
            let mut input = TextInput::new(cx);
            input.set_placeholder("Reply to resume…", cx);
            if !model.resolution_input.is_empty() {
                input.set_text(&model.resolution_input, cx);
            }
            input
        });
        let steer_input = cx.new(|cx| {
            let mut input = TextInput::new(cx);
            input.set_placeholder("Guidance for this thread…", cx);
            if !model.steer_input.is_empty() {
                input.set_text(&model.steer_input, cx);
            }
            input
        });
        Self {
            model,
            is_selected,
            focus_handle: cx.focus_handle(),
            resolution_input,
            steer_input,
        }
    }

    pub fn id(&self) -> &str {
        &self.model.id
    }

    /// Applies the sidebar's latest model and selection while keeping
    /// row-local state: the steer box toggle and any unsent input text.
    pub fn sync(&mut self, mut model: TaskCardModel, is_selected: bool) {
        model.show_steer_box = self.model.show_steer_box;
        model.resolution_input = std::mem::take(&mut self.model.resolution_input);
        model.steer_input = std::mem::take(&mut self.model.steer_input);
        self.model = model;
        self.is_selected = is_selected;
    }

    pub fn toggle_expand(&mut self, cx: &mut Context<Self>) {
        self.model.expanded = !self.model.expanded;
        cx.emit(TaskCardEvent::ToggleExpanded(self.model.id.clone()));
        cx.notify();
    }

    pub fn select(&mut self, cx: &mut Context<Self>) {
        cx.emit(TaskCardEvent::Selected(self.model.id.clone()));
        cx.notify();
    }

    pub fn pause(&mut self, cx: &mut Context<Self>) {
        cx.emit(TaskCardEvent::Pause(self.model.id.clone()));
        cx.notify();
    }

    pub fn resume(&mut self, cx: &mut Context<Self>) {
        let input_text = self.resolution_input.read(cx).text().to_string();
        let resolution = if !input_text.trim().is_empty() {
            input_text.trim().to_string()
        } else {
            self.model.resolution_input.trim().to_string()
        };
        cx.emit(TaskCardEvent::Resume {
            id: self.model.id.clone(),
            resolution,
        });
        self.resolution_input
            .update(cx, |input, cx| input.set_text("", cx));
        self.model.resolution_input.clear();
        cx.notify();
    }

    pub fn abort(&mut self, cx: &mut Context<Self>) {
        cx.emit(TaskCardEvent::Abort(self.model.id.clone()));
        cx.notify();
    }

    pub fn steer(&mut self, cx: &mut Context<Self>) {
        let input_text = self.steer_input.read(cx).text().to_string();
        let guidance = if !input_text.trim().is_empty() {
            input_text.trim().to_string()
        } else {
            self.model.steer_input.trim().to_string()
        };
        if !guidance.is_empty() {
            cx.emit(TaskCardEvent::Steer {
                id: self.model.id.clone(),
                guidance,
            });
            self.steer_input
                .update(cx, |input, cx| input.set_text("", cx));
            self.model.steer_input.clear();
            self.model.show_steer_box = false;
            cx.notify();
        }
    }

    pub fn toggle_steer_box(&mut self, cx: &mut Context<Self>) {
        self.model.show_steer_box = !self.model.show_steer_box;
        cx.notify();
    }

    /// Title, status, and stage bar. Clicking anywhere here selects the thread.
    fn render_summary(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let status = &self.model.status;
        let color = status_color(status);
        let stage = lifecycle_stage(status);

        div()
            .id(SharedString::from(format!("task-row-{}", self.model.id)))
            .flex()
            .flex_col()
            .gap_2()
            .cursor_pointer()
            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.select(cx)))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .text_sm()
                            .text_color(rgb(theme::TEXT))
                            .child(self.model.display_title()),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_none()
                            .items_center()
                            .gap_1p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(color)
                                    .child(status_label(status)),
                            )
                            .child(
                                div()
                                    .id(SharedString::from(format!("expand-btn-{}", self.model.id)))
                                    .px_1()
                                    .text_xs()
                                    .text_color(rgb(theme::TEXT_FAINT))
                                    .hover(|s| s.text_color(rgb(theme::TEXT)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        cx.stop_propagation();
                                        this.toggle_expand(cx);
                                    }))
                                    .child(if self.model.expanded { "▾" } else { "▸" }),
                            ),
                    ),
            )
            .child(
                div()
                    .h(px(3.))
                    .w_full()
                    .rounded_full()
                    .bg(rgb(theme::BORDER_STRONG))
                    .child(div().h_full().w(relative(stage)).rounded_full().bg(color)),
            )
    }

    fn render_blocker_callout(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let SubagentStatus::Blocked { reason, prompt } = &self.model.status else {
            return div();
        };
        let reason_label = match reason {
            BlockerReason::UserInput => "Input needed".to_string(),
            BlockerReason::Credential => "Provider credential missing".to_string(),
            BlockerReason::Environment => "Environment dependency blocked".to_string(),
            BlockerReason::Approval => "Approval needed".to_string(),
            BlockerReason::Custom(s) => s.clone(),
        };
        let card_id = &self.model.id;

        div()
            .flex()
            .flex_col()
            .gap_2()
            .mt_3()
            .p_2p5()
            .rounded_md()
            .bg(rgb(theme::BG_RAISED))
            .border_l_2()
            .border_color(rgb(theme::WARNING))
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(rgb(theme::WARNING))
                    .child(reason_label),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT))
                    .child(prompt.clone()),
            )
            .child(
                div()
                    .id(SharedString::from(format!("blocker-input-{card_id}")))
                    .w_full()
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .bg(rgb(theme::BG))
                    .border_1()
                    .border_color(rgb(theme::BORDER_STRONG))
                    .overflow_hidden()
                    .text_xs()
                    .child(self.resolution_input.clone()),
            )
            .child(
                div().flex().justify_end().child(
                    div()
                        .id(SharedString::from(format!("resume-btn-{card_id}")))
                        .cursor_pointer()
                        .px_3()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(theme::ACCENT))
                        .hover(|s| s.opacity(0.85))
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(rgb(theme::ON_ACCENT))
                        .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                            this.resume(cx);
                        }))
                        .child("Resume"),
                ),
            )
    }

    fn render_expanded(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_3()
            .mt_3()
            .child(self.render_meta())
            .child(self.render_steering_bar(cx))
            .child(self.render_details())
    }

    fn render_meta(&self) -> impl IntoElement {
        let (adds, dels) = self.model.total_diff_stats();

        div()
            .flex()
            .flex_col()
            .gap_1()
            .text_xs()
            .text_color(rgb(theme::TEXT_MUTED))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1p5()
                    .child(self.model.agent.clone())
                    .child(div().text_color(rgb(theme::TEXT_FAINT)).child("·"))
                    .child(self.model.format_duration())
                    .when(adds > 0 || dels > 0, |row| {
                        row.child(div().text_color(rgb(theme::TEXT_FAINT)).child("·"))
                            .child(
                                div()
                                    .text_color(rgb(theme::DIFF_ADD))
                                    .child(format!("+{adds}")),
                            )
                            .child(
                                div()
                                    .text_color(rgb(theme::DANGER))
                                    .child(format!("-{dels}")),
                            )
                    }),
            )
            .when_some(self.model.workspace.clone(), |meta, workspace| {
                meta.child(
                    div()
                        .truncate()
                        .text_color(rgb(theme::TEXT_FAINT))
                        .child(workspace),
                )
            })
            .when_some(self.model.current_step.clone(), |meta, step| {
                meta.child(div().truncate().child(format!("› {step}")))
            })
    }

    fn render_steering_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let is_active = is_status_active(&self.model.status);
        let is_blocked = self.model.status.is_blocked();
        let card_id = &self.model.id;

        div()
            .flex()
            .flex_col()
            .gap_2()
            .when(is_active, |bar| {
                bar.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1p5()
                        .when(!is_blocked, |row| {
                            row.child(
                                control_button(
                                    format!("pause-btn-{card_id}"),
                                    "Pause",
                                    rgb(theme::TEXT),
                                )
                                .on_click(cx.listener(
                                    |this, _: &ClickEvent, _, cx| {
                                        this.pause(cx);
                                    },
                                )),
                            )
                        })
                        .child(
                            control_button(
                                format!("steer-toggle-btn-{card_id}"),
                                "Steer",
                                rgb(theme::TEXT),
                            )
                            .on_click(cx.listener(
                                |this, _: &ClickEvent, _, cx| {
                                    this.toggle_steer_box(cx);
                                },
                            )),
                        )
                        .child(
                            control_button(
                                format!("abort-btn-{card_id}"),
                                "Abort",
                                rgb(theme::DANGER),
                            )
                            .on_click(cx.listener(
                                |this, _: &ClickEvent, _, cx| {
                                    this.abort(cx);
                                },
                            )),
                        ),
                )
            })
            .when(is_active && self.model.show_steer_box, |bar| {
                bar.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(
                            div()
                                .id(SharedString::from(format!("steer-input-{card_id}")))
                                .flex_1()
                                .min_w_0()
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(theme::BG))
                                .border_1()
                                .border_color(rgb(theme::BORDER_STRONG))
                                .overflow_hidden()
                                .text_xs()
                                .child(self.steer_input.clone()),
                        )
                        .child(
                            div()
                                .id(SharedString::from(format!("send-steer-btn-{card_id}")))
                                .cursor_pointer()
                                .px_2p5()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(theme::ACCENT))
                                .hover(|s| s.opacity(0.85))
                                .text_xs()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(rgb(theme::ON_ACCENT))
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.steer(cx);
                                }))
                                .child("Send"),
                        ),
                )
            })
    }

    fn render_details(&self) -> impl IntoElement {
        let usage = &self.model.usage;
        let diffs = &self.model.diffs;
        let logs = &self.model.logs;
        let mut usage_line = format!(
            "In {} · Out {} · Total {} tok",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
        if let Some(cost) = self.model.cost_estimate {
            usage_line.push_str(&format!(" · ${cost:.4}"));
        }

        div()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_FAINT))
                    .child(usage_line),
            )
            .when(!logs.is_empty(), |details| {
                details.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(section_label("STEPS"))
                        .children(logs.iter().map(|log| {
                            div()
                                .flex()
                                .items_center()
                                .gap_2()
                                .text_xs()
                                .child(
                                    div()
                                        .flex_none()
                                        .text_color(rgb(theme::TEXT_FAINT))
                                        .child(format!("#{}", log.step)),
                                )
                                .child(
                                    div()
                                        .flex_none()
                                        .text_color(rgb(theme::ACCENT))
                                        .child(log.tool_name.clone()),
                                )
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .truncate()
                                        .text_color(rgb(theme::TEXT_MUTED))
                                        .child(log.summary.clone()),
                                )
                                .child(
                                    div()
                                        .flex_none()
                                        .text_color(rgb(theme::TEXT_FAINT))
                                        .child(log.status.clone()),
                                )
                        })),
                )
            })
            .when(!diffs.is_empty(), |details| {
                details.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(section_label("CHANGES"))
                        .children(diffs.iter().map(render_diff_file)),
                )
            })
    }
}

fn section_label(label: &'static str) -> impl IntoElement {
    div()
        .text_xs()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(theme::TEXT_FAINT))
        .child(label)
}

fn control_button(id: String, label: &'static str, text: Rgba) -> Stateful<Div> {
    div()
        .id(SharedString::from(id))
        .cursor_pointer()
        .px_2()
        .py_0p5()
        .rounded_sm()
        .border_1()
        .border_color(rgb(theme::BORDER_STRONG))
        .text_xs()
        .text_color(text)
        .hover(|s| s.bg(rgb(theme::BG_HOVER)))
        .child(label)
}

fn render_diff_file(file_diff: &DiffFileModel) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .rounded_sm()
        .border_1()
        .border_color(rgb(theme::BORDER))
        .overflow_hidden()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap_2()
                .px_2()
                .py_1()
                .bg(rgb(theme::BG_RAISED))
                .text_xs()
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .text_color(rgb(theme::TEXT))
                        .child(file_diff.path.clone()),
                )
                .child(
                    div()
                        .flex()
                        .flex_none()
                        .gap_1p5()
                        .child(
                            div()
                                .text_color(rgb(theme::DIFF_ADD))
                                .child(format!("+{}", file_diff.additions)),
                        )
                        .child(
                            div()
                                .text_color(rgb(theme::DANGER))
                                .child(format!("-{}", file_diff.deletions)),
                        ),
                ),
        )
        .children(file_diff.hunks.iter().map(|hunk| {
            div()
                .flex()
                .flex_col()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_2()
                        .px_2()
                        .py_0p5()
                        .text_xs()
                        .text_color(rgb(theme::TEXT_FAINT))
                        .child(hunk.header.clone())
                        .when_some(hunk.turn_anchor.clone(), |h, anchor| {
                            h.child(div().min_w_0().truncate().child(anchor))
                        }),
                )
                .children(hunk.lines.iter().map(|line| {
                    let (line_bg, line_text, prefix) = match line.kind {
                        DiffLineKind::Add => (rgb(theme::DIFF_ADD_BG), rgb(theme::DIFF_ADD), "+"),
                        DiffLineKind::Remove => {
                            (rgb(theme::DIFF_REMOVE_BG), rgb(theme::DANGER), "-")
                        }
                        DiffLineKind::Context => (rgb(theme::BG), rgb(theme::TEXT_MUTED), " "),
                    };
                    div()
                        .flex()
                        .items_center()
                        .px_2()
                        .bg(line_bg)
                        .text_xs()
                        .text_color(line_text)
                        .child(div().w_4().flex_none().child(prefix))
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .truncate()
                                .child(line.content.clone()),
                        )
                }))
        }))
}

impl Focusable for TaskCard {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for TaskCard {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.is_selected;

        div()
            .id(SharedString::from(format!("task-card-{}", self.model.id)))
            .key_context("TaskCard")
            .relative()
            .flex()
            .flex_col()
            .pl_5()
            .pr_4()
            .py_3()
            .when(is_selected, |row| {
                row.bg(rgb(theme::BG_RAISED)).child(
                    div()
                        .absolute()
                        .left_0()
                        .top_0()
                        .bottom_0()
                        .w(px(3.))
                        .bg(rgb(theme::ACCENT)),
                )
            })
            .when(!is_selected, |row| {
                row.hover(|s| s.bg(rgb(theme::BG_HOVER)))
            })
            .child(self.render_summary(cx))
            .child(self.render_blocker_callout(cx))
            .when(self.model.expanded, |row| {
                row.child(self.render_expanded(cx))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_activity() {
        assert!(!is_status_active(&SubagentStatus::Queued));
        assert!(is_status_active(&SubagentStatus::Running));

        let blocked = SubagentStatus::Blocked {
            reason: BlockerReason::UserInput,
            prompt: "Please clarify".into(),
        };
        assert!(is_status_active(&blocked));
        assert!(blocked.is_blocked());

        assert!(!is_status_active(&SubagentStatus::Completed));
    }

    #[test]
    fn lifecycle_stage_advances_through_the_pipeline() {
        let blocked = SubagentStatus::Blocked {
            reason: BlockerReason::Approval,
            prompt: "Approve?".into(),
        };
        let pipeline = [
            SubagentStatus::Queued,
            SubagentStatus::Running,
            SubagentStatus::Verifying,
            SubagentStatus::Reviewing,
            SubagentStatus::Completed,
        ];
        let stages: Vec<f32> = pipeline.iter().map(lifecycle_stage).collect();
        assert_eq!(stages.first(), Some(&0.0));
        assert_eq!(stages.last(), Some(&1.0));
        assert!(stages.windows(2).all(|pair| pair[0] < pair[1]));
        // Blocking pauses a running thread; it does not move it backward or forward.
        assert_eq!(
            lifecycle_stage(&blocked),
            lifecycle_stage(&SubagentStatus::Running)
        );
        // Terminal failures fill the bar so they never read as "still queued".
        assert_eq!(lifecycle_stage(&SubagentStatus::Failed), 1.0);
        assert_eq!(lifecycle_stage(&SubagentStatus::Cancelled), 1.0);
    }

    #[test]
    fn test_task_card_model_creation_and_stats() {
        let mut model = TaskCardModel::new("task-1", "Implement Auth Provider", "builder")
            .with_name("#1 Auth Session")
            .with_workspace("~/workspaces/juto-auth")
            .with_current_step("cargo test")
            .with_elapsed(125);

        assert_eq!(model.display_title(), "#1 Auth Session");
        assert_eq!(model.format_duration(), "2m 05s");

        model.diffs.push(DiffFileModel {
            path: "src/auth.rs".into(),
            additions: 42,
            deletions: 12,
            hunks: vec![DiffHunkModel {
                header: "@@ -1,5 +1,10 @@".into(),
                turn_anchor: Some("Turn 2: write auth flow".into()),
                lines: vec![
                    DiffLineModel {
                        kind: DiffLineKind::Remove,
                        old_num: Some(2),
                        new_num: None,
                        content: "old code".into(),
                    },
                    DiffLineModel {
                        kind: DiffLineKind::Add,
                        old_num: None,
                        new_num: Some(2),
                        content: "new code".into(),
                    },
                ],
            }],
        });

        let (adds, dels) = model.total_diff_stats();
        assert_eq!(adds, 42);
        assert_eq!(dels, 12);
    }
}
