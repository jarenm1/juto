//! Subagent task card component for GPUI Sidebar Mission Control.
//!
//! Implements the Zed Delta-inspired model and subagent state lifecycle:
//! - Visual status badges: Queued, Running, Blocked, Verifying, Reviewing, Completed, Failed, Cancelled.
//! - Direct integration with `juto_runtime::subagents::{BlockerReason, SubagentStatus}`.
//! - Expandable metrics: duration, token usage (input/output/cache), estimated cost.
//! - Active step & tool execution log.
//! - Unified diff preview pane rendering hunks attached to conversational turn metadata.
//! - Interactive steering controls: Pause, Abort, and Resume/Unblock prompt input.

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, IntoElement, KeyBinding, Render,
    Rgba, SharedString, Window, actions, div, prelude::*, rgb,
};
use juto_runtime::subagents::{BlockerReason, SubagentStatus};

use crate::text_input::TextInput;

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
        SubagentStatus::Queued => "QUEUED",
        SubagentStatus::Running => "RUNNING",
        SubagentStatus::Blocked { .. } => "BLOCKED",
        SubagentStatus::Verifying => "VERIFYING",
        SubagentStatus::Reviewing => "REVIEWING",
        SubagentStatus::Completed => "COMPLETED",
        SubagentStatus::Failed => "FAILED",
        SubagentStatus::Cancelled => "CANCELLED",
    }
}

pub fn status_badge_colors(status: &SubagentStatus) -> (Rgba, Rgba, Rgba) {
    // (background, border, text)
    match status {
        SubagentStatus::Queued => (rgb(0x27272a), rgb(0x3f3f46), rgb(0xa1a1aa)),
        SubagentStatus::Running => (rgb(0x1e3a5f), rgb(0x2563eb), rgb(0x60a5fa)),
        SubagentStatus::Blocked { .. } => (rgb(0x451a03), rgb(0xd97706), rgb(0xfbbf24)),
        SubagentStatus::Verifying => (rgb(0x3b1a45), rgb(0x9333ea), rgb(0xc084fc)),
        SubagentStatus::Reviewing => (rgb(0x143a3a), rgb(0x0d9488), rgb(0x2dd4bf)),
        SubagentStatus::Completed => (rgb(0x064e3b), rgb(0x059669), rgb(0x34d399)),
        SubagentStatus::Failed => (rgb(0x450a0a), rgb(0xdc2626), rgb(0xf87171)),
        SubagentStatus::Cancelled => (rgb(0x27272a), rgb(0x52525b), rgb(0x71717a)),
    }
}

pub fn status_dot_color(status: &SubagentStatus) -> Rgba {
    match status {
        SubagentStatus::Queued => rgb(0x71717a),
        SubagentStatus::Running => rgb(0x3b82f6),
        SubagentStatus::Blocked { .. } => rgb(0xf59e0b),
        SubagentStatus::Verifying => rgb(0xa855f7),
        SubagentStatus::Reviewing => rgb(0x14b8a6),
        SubagentStatus::Completed => rgb(0x10b981),
        SubagentStatus::Failed => rgb(0xef4444),
        SubagentStatus::Cancelled => rgb(0x6b7280),
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
            input.set_placeholder("Provide guidance or confirmation to resume…", cx);
            if !model.resolution_input.is_empty() {
                input.set_text(&model.resolution_input, cx);
            }
            input
        });
        let steer_input = cx.new(|cx| {
            let mut input = TextInput::new(cx);
            input.set_placeholder("Inject live guidance to agent…", cx);
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

    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let status = &self.model.status;
        let (bg, border, text) = status_badge_colors(status);
        let dot_color = status_dot_color(status);
        let is_expanded = self.model.expanded;
        let title = self.model.display_title();
        let agent = self.model.agent.clone();
        let duration = self.model.format_duration();
        let status_label = status_label(status);

        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .min_w_0()
                    .child(
                        // Persona chip
                        div()
                            .px_1p5()
                            .py_0p5()
                            .rounded_xs()
                            .bg(rgb(0x27272a))
                            .border_1()
                            .border_color(rgb(0x3f3f46))
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(rgb(0xd4d4d8))
                            .child(agent),
                    )
                    .child(
                        // Title
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(rgb(0xf4f4f5))
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .child(title),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .flex_none()
                    .child(
                        // Duration
                        div().text_xs().text_color(rgb(0xa1a1aa)).child(duration),
                    )
                    .child(
                        // Status badge
                        div()
                            .flex()
                            .items_center()
                            .gap_1p5()
                            .px_2()
                            .py_0p5()
                            .rounded_sm()
                            .bg(bg)
                            .border_1()
                            .border_color(border)
                            .child(div().size_2().rounded_full().bg(dot_color))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(text)
                                    .child(status_label),
                            ),
                    )
                    .child(
                        // Expand/collapse indicator button
                        div()
                            .id(SharedString::from(format!("expand-btn-{}", self.model.id)))
                            .cursor_pointer()
                            .text_xs()
                            .text_color(rgb(0x71717a))
                            .hover(|s| s.text_color(rgb(0xe4e4e7)))
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.toggle_expand(cx);
                            }))
                            .child(if is_expanded { "▼" } else { "▶" }),
                    ),
            )
    }

    fn render_subline(&self) -> impl IntoElement {
        let (adds, dels) = self.model.total_diff_stats();
        let ws = self.model.workspace.clone();
        let step = self.model.current_step.clone();

        div()
            .flex()
            .flex_col()
            .gap_1()
            .mt_1()
            .when_some(ws, |this, workspace| {
                this.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1()
                        .text_xs()
                        .text_color(rgb(0x71717a))
                        .child("📁")
                        .child(workspace),
                )
            })
            .when_some(step, |this, current_step| {
                this.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1p5()
                        .text_xs()
                        .text_color(rgb(0x93c5fd))
                        .child("⚙")
                        .child(current_step),
                )
            })
            .when(adds > 0 || dels > 0, |this| {
                this.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .text_xs()
                        .child(div().text_color(rgb(0x34d399)).child(format!("+{}", adds)))
                        .child(div().text_color(rgb(0xf87171)).child(format!("-{}", dels))),
                )
            })
    }

    fn render_blocker_callout(&self, cx: &mut Context<Self>) -> impl IntoElement {
        if let SubagentStatus::Blocked { reason, prompt } = &self.model.status {
            let reason_label = match reason {
                BlockerReason::UserInput => "User Input Required".to_string(),
                BlockerReason::Credential => "Provider Credential Missing".to_string(),
                BlockerReason::Environment => "Environment Dependency Blocked".to_string(),
                BlockerReason::Approval => "Operation Approval Required".to_string(),
                BlockerReason::Custom(s) => s.clone(),
            };
            let prompt = prompt.clone();
            let card_id = self.model.id.clone();

            div()
                .flex()
                .flex_col()
                .gap_2()
                .mt_2()
                .p_2p5()
                .rounded_md()
                .bg(rgb(0x2a1a08))
                .border_1()
                .border_color(rgb(0xb45309))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1p5()
                        .text_xs()
                        .font_weight(gpui::FontWeight::BOLD)
                        .text_color(rgb(0xfbbf24))
                        .child("⚠️ ACTION REQUIRED:")
                        .child(reason_label),
                )
                .child(div().text_xs().text_color(rgb(0xfef3c7)).child(prompt))
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .mt_1()
                        .child(
                            div()
                                .id(SharedString::from(format!("blocker-input-{}", card_id)))
                                .flex_1()
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x18181b))
                                .border_1()
                                .border_color(rgb(0x78350f))
                                .text_xs()
                                .child(self.resolution_input.clone()),
                        )
                        .child(
                            div()
                                .id(SharedString::from(format!("resume-btn-{}", card_id)))
                                .cursor_pointer()
                                .px_3()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0xd97706))
                                .hover(|s| s.bg(rgb(0xb45309)))
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(rgb(0x18181b))
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.resume(cx);
                                }))
                                .child("▶ Resume"),
                        ),
                )
        } else {
            div()
        }
    }

    fn render_steering_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let is_active = is_status_active(&self.model.status);
        let is_blocked = self.model.status.is_blocked();
        let card_id = self.model.id.clone();
        let show_steer = self.model.show_steer_box;

        div()
            .flex()
            .flex_col()
            .gap_1p5()
            .mt_2()
            .child(
                div().flex().items_center().justify_between().gap_2().child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1p5()
                        .when(is_active && !is_blocked, |bar| {
                            bar.child(
                                div()
                                    .id(SharedString::from(format!("pause-btn-{}", card_id)))
                                    .cursor_pointer()
                                    .px_2()
                                    .py_0p5()
                                    .rounded_xs()
                                    .bg(rgb(0x27272a))
                                    .border_1()
                                    .border_color(rgb(0x3f3f46))
                                    .text_xs()
                                    .text_color(rgb(0xe4e4e7))
                                    .hover(|s| s.bg(rgb(0x3f3f46)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.pause(cx);
                                    }))
                                    .child("⏸ Pause"),
                            )
                        })
                        .when(is_active, |bar| {
                            bar.child(
                                div()
                                    .id(SharedString::from(format!("steer-toggle-btn-{}", card_id)))
                                    .cursor_pointer()
                                    .px_2()
                                    .py_0p5()
                                    .rounded_xs()
                                    .bg(rgb(0x27272a))
                                    .border_1()
                                    .border_color(rgb(0x3f3f46))
                                    .text_xs()
                                    .text_color(rgb(0x60a5fa))
                                    .hover(|s| s.bg(rgb(0x3f3f46)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.toggle_steer_box(cx);
                                    }))
                                    .child("💬 Steer"),
                            )
                        })
                        .when(is_active, |bar| {
                            bar.child(
                                div()
                                    .id(SharedString::from(format!("abort-btn-{}", card_id)))
                                    .cursor_pointer()
                                    .px_2()
                                    .py_0p5()
                                    .rounded_xs()
                                    .bg(rgb(0x27272a))
                                    .border_1()
                                    .border_color(rgb(0x3f3f46))
                                    .text_xs()
                                    .text_color(rgb(0xf87171))
                                    .hover(|s| s.bg(rgb(0x450a0a)).border_color(rgb(0xdc2626)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                        this.abort(cx);
                                    }))
                                    .child("⏹ Abort"),
                            )
                        }),
                ),
            )
            .when(show_steer, |bar| {
                bar.child(
                    div()
                        .flex()
                        .gap_2()
                        .mt_1()
                        .child(
                            div()
                                .id(SharedString::from(format!("steer-input-{}", card_id)))
                                .flex_1()
                                .px_2()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x18181b))
                                .border_1()
                                .border_color(rgb(0x3b82f6))
                                .text_xs()
                                .child(self.steer_input.clone()),
                        )
                        .child(
                            div()
                                .id(SharedString::from(format!("send-steer-btn-{}", card_id)))
                                .cursor_pointer()
                                .px_2p5()
                                .py_1()
                                .rounded_sm()
                                .bg(rgb(0x2563eb))
                                .hover(|s| s.bg(rgb(0x1d4ed8)))
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(rgb(0xffffff))
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.steer(cx);
                                }))
                                .child("Send"),
                        ),
                )
            })
    }

    fn render_expanded_details(&self) -> impl IntoElement {
        let usage = &self.model.usage;
        let diffs = &self.model.diffs;
        let logs = &self.model.logs;

        div()
            .flex()
            .flex_col()
            .gap_3()
            .mt_3()
            .pt_3()
            .border_t_1()
            .border_color(rgb(0x27272a))
            .child(
                // Metrics Pill Bar
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .child(
                        div()
                            .px_2()
                            .py_1()
                            .rounded_xs()
                            .bg(rgb(0x1c1917))
                            .border_1()
                            .border_color(rgb(0x292524))
                            .text_xs()
                            .text_color(rgb(0xa8a29e))
                            .child(format!("In: {} tok", usage.input_tokens)),
                    )
                    .child(
                        div()
                            .px_2()
                            .py_1()
                            .rounded_xs()
                            .bg(rgb(0x1c1917))
                            .border_1()
                            .border_color(rgb(0x292524))
                            .text_xs()
                            .text_color(rgb(0xa8a29e))
                            .child(format!("Out: {} tok", usage.output_tokens)),
                    )
                    .child(
                        div()
                            .px_2()
                            .py_1()
                            .rounded_xs()
                            .bg(rgb(0x1c1917))
                            .border_1()
                            .border_color(rgb(0x292524))
                            .text_xs()
                            .text_color(rgb(0x38bdf8))
                            .child(format!("Total: {} tok", usage.total_tokens)),
                    )
                    .when_some(self.model.cost_estimate, |bar, cost| {
                        bar.child(
                            div()
                                .px_2()
                                .py_1()
                                .rounded_xs()
                                .bg(rgb(0x1c1917))
                                .border_1()
                                .border_color(rgb(0x292524))
                                .text_xs()
                                .text_color(rgb(0x34d399))
                                .child(format!("${:.4}", cost)),
                        )
                    }),
            )
            .when(!logs.is_empty(), |details| {
                details.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(gpui::FontWeight::BOLD)
                                .text_color(rgb(0x71717a))
                                .child("STEP EXECUTION LOG"),
                        )
                        .children(logs.iter().map(|log| {
                            div()
                                .flex()
                                .items_center()
                                .gap_2()
                                .px_2()
                                .py_1()
                                .rounded_xs()
                                .bg(rgb(0x18181b))
                                .text_xs()
                                .child(
                                    div()
                                        .text_color(rgb(0x71717a))
                                        .child(format!("#{}", log.step)),
                                )
                                .child(
                                    div()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .text_color(rgb(0x38bdf8))
                                        .child(log.tool_name.clone()),
                                )
                                .child(
                                    div()
                                        .flex_1()
                                        .text_color(rgb(0xd4d4d8))
                                        .overflow_hidden()
                                        .whitespace_nowrap()
                                        .child(log.summary.clone()),
                                )
                                .child(div().text_color(rgb(0xa1a1aa)).child(log.status.clone()))
                        })),
                )
            })
            .when(!diffs.is_empty(), |details| {
                details.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::BOLD)
                                        .text_color(rgb(0x71717a))
                                        .child("LIVE DELTAS (ZED MODEL)"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(rgb(0xa1a1aa))
                                        .child(format!("{} file(s)", diffs.len())),
                                ),
                        )
                        .children(diffs.iter().map(|file_diff| {
                            div()
                                .flex()
                                .flex_col()
                                .rounded_sm()
                                .bg(rgb(0x141416))
                                .border_1()
                                .border_color(rgb(0x27272a))
                                .overflow_hidden()
                                .child(
                                    // File diff header
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .px_2()
                                        .py_1()
                                        .bg(rgb(0x1f1f23))
                                        .border_b_1()
                                        .border_color(rgb(0x27272a))
                                        .child(
                                            div()
                                                .text_xs()
                                                .font_weight(gpui::FontWeight::MEDIUM)
                                                .text_color(rgb(0xe4e4e7))
                                                .child(file_diff.path.clone()),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .gap_1p5()
                                                .text_xs()
                                                .child(
                                                    div()
                                                        .text_color(rgb(0x34d399))
                                                        .child(format!("+{}", file_diff.additions)),
                                                )
                                                .child(
                                                    div()
                                                        .text_color(rgb(0xf87171))
                                                        .child(format!("-{}", file_diff.deletions)),
                                                ),
                                        ),
                                )
                                .children(file_diff.hunks.iter().map(|hunk| {
                                    div()
                                        .flex()
                                        .flex_col()
                                        .child(
                                            // Hunk header + turn anchor
                                            div()
                                                .flex()
                                                .items_center()
                                                .justify_between()
                                                .px_2()
                                                .py_0p5()
                                                .bg(rgb(0x1c1917))
                                                .text_xs()
                                                .child(
                                                    div()
                                                        .text_color(rgb(0x78716c))
                                                        .child(hunk.header.clone()),
                                                )
                                                .when_some(
                                                    hunk.turn_anchor.clone(),
                                                    |h, anchor| {
                                                        h.child(
                                                            div()
                                                                .px_1()
                                                                .rounded_xs()
                                                                .bg(rgb(0x292524))
                                                                .text_xs()
                                                                .text_color(rgb(0xf59e0b))
                                                                .child(anchor),
                                                        )
                                                    },
                                                ),
                                        )
                                        .children(hunk.lines.iter().map(|line| {
                                            let (line_bg, line_text, prefix) = match line.kind {
                                                DiffLineKind::Add => {
                                                    (rgb(0x064e3b), rgb(0x6ee7b7), "+")
                                                }
                                                DiffLineKind::Remove => {
                                                    (rgb(0x450a0a), rgb(0xfca5a5), "-")
                                                }
                                                DiffLineKind::Context => {
                                                    (rgb(0x141416), rgb(0xa1a1aa), " ")
                                                }
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
                                                        .whitespace_nowrap()
                                                        .overflow_hidden()
                                                        .child(line.content.clone()),
                                                )
                                        }))
                                }))
                        })),
                )
            })
    }
}

impl Focusable for TaskCard {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for TaskCard {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.is_selected;
        let is_blocked = self.model.status.is_blocked();
        let border_color = if is_selected {
            rgb(0x3b82f6)
        } else if is_blocked {
            rgb(0xb45309)
        } else {
            rgb(0x27272a)
        };

        div()
            .id(SharedString::from(format!("task-card-{}", self.model.id)))
            .key_context("TaskCard")
            .flex()
            .flex_col()
            .p_3()
            .rounded_md()
            .bg(rgb(0x18181b))
            .border_1()
            .border_color(border_color)
            .hover(|s| {
                s.border_color(if is_selected {
                    rgb(0x60a5fa)
                } else {
                    rgb(0x3f3f46)
                })
            })
            .child(self.render_header(cx))
            .child(self.render_subline())
            .child(self.render_blocker_callout(cx))
            .child(self.render_steering_bar(cx))
            .when(self.model.expanded, |card| {
                card.child(self.render_expanded_details())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_labels_and_colors() {
        let queued = SubagentStatus::Queued;
        assert_eq!(status_label(&queued), "QUEUED");
        assert!(!is_status_active(&queued));

        let running = SubagentStatus::Running;
        assert_eq!(status_label(&running), "RUNNING");
        assert!(is_status_active(&running));

        let blocked = SubagentStatus::Blocked {
            reason: BlockerReason::UserInput,
            prompt: "Please clarify".into(),
        };
        assert_eq!(status_label(&blocked), "BLOCKED");
        assert!(is_status_active(&blocked));
        assert!(blocked.is_blocked());

        let completed = SubagentStatus::Completed;
        assert_eq!(status_label(&completed), "COMPLETED");
        assert!(!is_status_active(&completed));
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
