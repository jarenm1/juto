//! GPUI Sidebar "Mission Control" tracking subagent states, Zed live diffs, and steering controls.
//!
//! Specifications from docs/orchestration-plan.md Slice D:
//! - Visual task list displaying badge per SubagentStatus (Queued, Running, Blocked, Verifying, Reviewing, Completed, Failed, Cancelled).
//! - Direct integration with `juto_runtime::subagents::{BlockerReason, SubagentStatus, SubagentSummary}`.
//! - Filter tabs: All, Active, Blocked, Done.
//! - Search/filter bar for tasks.
//! - Expandable task cards: duration, token count, active tool log, live delta preview.
//! - Interactive steering buttons: Pause, Abort, Resume/Unblock, Guidance Injection.
//! - Error banner surfaced with visible failure reason on rejected steering or unblock.
//! - Zed Delta unified diff viewer anchored to conversation turn metadata.

use gpui::{
    App, ClickEvent, Context, FocusHandle, Focusable, IntoElement, KeyBinding, Render,
    ScrollHandle, SharedString, Window, actions, div, prelude::*, rgb,
};
use juto_runtime::subagents::{BlockerReason, SubagentStatus, SubagentSummary};

use crate::task_card::{
    DiffFileModel, DiffHunkModel, DiffLineKind, DiffLineModel, TaskCard, TaskCardEvent,
    TaskCardModel, TokenUsageModel, ToolLogEntry, is_status_active,
};

actions!(
    sidebar,
    [
        CloseSidebar,
        FilterAllTasks,
        FilterActiveTasks,
        FilterBlockedTasks,
        FilterCompletedTasks,
    ]
);

pub fn init(cx: &mut App) {
    cx.bind_keys([KeyBinding::new("escape", CloseSidebar, Some("Sidebar"))]);
}

/// Filter tab for the task list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskFilter {
    All,
    Active,
    Blocked,
    Done,
}

impl TaskFilter {
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Active => "Active",
            Self::Blocked => "Blocked",
            Self::Done => "Done",
        }
    }
}

/// Events emitted by the Sidebar component to the parent ChatView.
#[derive(Clone, Debug, PartialEq)]
pub enum SidebarEvent {
    SelectTask(String),
    PauseTask(String),
    ResumeTask { id: String, resolution: String },
    AbortTask(String),
    SteerTask { id: String, guidance: String },
    Close,
}

/// GPUI Sidebar Mission Control.
pub struct Sidebar {
    focus_handle: Option<FocusHandle>,
    scroll_handle: ScrollHandle,
    tasks: Vec<TaskCardModel>,
    selected_task: Option<String>,
    filter: TaskFilter,
    search_query: String,
    error_message: Option<SharedString>,
}

impl gpui::EventEmitter<SidebarEvent> for Sidebar {}

#[allow(dead_code)]
impl Sidebar {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: Some(cx.focus_handle()),
            scroll_handle: ScrollHandle::new(),
            tasks: Vec::new(),
            selected_task: None,
            filter: TaskFilter::All,
            search_query: String::new(),
            error_message: None,
        }
    }

    /// Sample tasks demonstrating all lifecycle states and Zed live diff cards.
    pub fn sample_tasks() -> Vec<TaskCardModel> {
        vec![
            TaskCardModel::new(
                "subagent-auth-1",
                "Implement OAuth refresh token exchange flow",
                "builder",
            )
            .with_name("#1 Auth Token Refresh")
            .with_status(SubagentStatus::Running)
            .with_workspace("~/workspaces/juto-auth-refresh")
            .with_current_step("cargo test -p juto-ai")
            .with_elapsed(142)
            .with_diff(DiffFileModel {
                path: "crates/juto-ai/src/auth.rs".into(),
                additions: 38,
                deletions: 9,
                hunks: vec![DiffHunkModel {
                    header: "@@ -124,6 +124,15 @@".into(),
                    turn_anchor: Some("Turn 3: refresh token implementation".into()),
                    lines: vec![
                        DiffLineModel {
                            kind: DiffLineKind::Remove,
                            old_num: Some(124),
                            new_num: None,
                            content: "    todo!(\"refresh token\");".into(),
                        },
                        DiffLineModel {
                            kind: DiffLineKind::Add,
                            old_num: None,
                            new_num: Some(124),
                            content: "    let refreshed = client.refresh(&token).await?;".into(),
                        },
                        DiffLineModel {
                            kind: DiffLineKind::Add,
                            old_num: None,
                            new_num: Some(125),
                            content: "    self.store.save(&refreshed).await?;".into(),
                        },
                        DiffLineModel {
                            kind: DiffLineKind::Context,
                            old_num: Some(125),
                            new_num: Some(126),
                            content: "    Ok(refreshed)".into(),
                        },
                    ],
                }],
            })
            .with_log(ToolLogEntry {
                step: 1,
                tool_name: "read".into(),
                summary: "Inspect existing credential store in auth.rs".into(),
                status: "ok".into(),
            })
            .with_log(ToolLogEntry {
                step: 2,
                tool_name: "edit".into(),
                summary: "Add refresh token loop and header injection".into(),
                status: "ok".into(),
            })
            .with_log(ToolLogEntry {
                step: 3,
                tool_name: "bash".into(),
                summary: "Run cargo test --lib auth".into(),
                status: "running".into(),
            }),
            TaskCardModel::new(
                "subagent-audit-2",
                "Adversarial security audit of token storage",
                "reviewer",
            )
            .with_name("#2 Token Storage Security Audit")
            .with_status(SubagentStatus::Blocked {
                reason: BlockerReason::Custom("Secret Encryption Key Ambiguity".into()),
                prompt:
                    "Choose keystore backend: system keyring (recommended) or encrypted file store?"
                        .into(),
            })
            .with_workspace("~/workspaces/juto-auth-refresh")
            .with_current_step("Awaiting human decision on secret keyring")
            .with_elapsed(48)
            .with_log(ToolLogEntry {
                step: 1,
                tool_name: "read".into(),
                summary: "Audit keyring usage and filesystem permissions".into(),
                status: "blocked".into(),
            }),
            TaskCardModel::new(
                "subagent-scout-3",
                "Locate all Model and Role selectors in catalog",
                "scout",
            )
            .with_name("#3 Catalog Model Discovery")
            .with_status(SubagentStatus::Completed)
            .with_current_step("Indexed 128 models and 5 OMP roles")
            .with_elapsed(18)
            .with_log(ToolLogEntry {
                step: 1,
                tool_name: "glob".into(),
                summary: "Scan data/models.json.gz and source.json".into(),
                status: "ok".into(),
            }),
        ]
    }

    pub fn tasks(&self) -> &[TaskCardModel] {
        &self.tasks
    }

    pub fn set_error(&mut self, error: Option<SharedString>, cx: &mut Context<Self>) {
        self.set_error_message(error);
        cx.notify();
    }

    pub fn set_error_message(&mut self, error: Option<SharedString>) {
        self.error_message = error;
    }

    /// Synchronize task cards with real subagent summaries from `SubagentManager::list()`.
    pub fn sync_from_summaries(&mut self, summaries: &[SubagentSummary], cx: &mut Context<Self>) {
        self.apply_summaries(summaries);
        cx.notify();
    }

    pub fn apply_summaries(&mut self, summaries: &[SubagentSummary]) {
        for summary in summaries {
            if let Some(existing) = self.tasks.iter_mut().find(|t| t.id == summary.id) {
                existing.status = summary.status.clone();
                if let Some(name) = &summary.name {
                    existing.name = Some(name.clone());
                }
                if let Some(usage) = &summary.usage {
                    existing.usage = TokenUsageModel {
                        input_tokens: usage.input,
                        output_tokens: usage.output,
                        cache_read_tokens: usage.cache_read,
                        cache_write_tokens: usage.cache_write,
                        total_tokens: usage.total_tokens,
                    };
                }
            } else {
                let mut card =
                    TaskCardModel::new(summary.id.clone(), summary.task.clone(), "builder")
                        .with_status(summary.status.clone());

                if let Some(name) = &summary.name {
                    card.name = Some(name.clone());
                }
                if let Some(usage) = &summary.usage {
                    card.usage = TokenUsageModel {
                        input_tokens: usage.input,
                        output_tokens: usage.output,
                        cache_read_tokens: usage.cache_read,
                        cache_write_tokens: usage.cache_write,
                        total_tokens: usage.total_tokens,
                    };
                }
                self.tasks.push(card);
            }
        }
    }

    pub fn set_tasks(&mut self, tasks: Vec<TaskCardModel>, cx: &mut Context<Self>) {
        self.tasks = tasks;
        cx.notify();
    }

    pub fn upsert_task(&mut self, task: TaskCardModel, cx: &mut Context<Self>) {
        if let Some(pos) = self.tasks.iter().position(|t| t.id == task.id) {
            self.tasks[pos] = task;
        } else {
            self.tasks.push(task);
        }
        cx.notify();
    }

    pub fn update_task_status(&mut self, id: &str, status: SubagentStatus, cx: &mut Context<Self>) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.status = status;
            cx.notify();
        }
    }

    pub fn update_step(&mut self, id: &str, step: impl Into<String>, cx: &mut Context<Self>) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.current_step = Some(step.into());
            cx.notify();
        }
    }

    pub fn add_diff(&mut self, id: &str, diff: DiffFileModel, cx: &mut Context<Self>) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.diffs.push(diff);
            cx.notify();
        }
    }

    pub fn add_log(&mut self, id: &str, log: ToolLogEntry, cx: &mut Context<Self>) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.logs.push(log);
            cx.notify();
        }
    }

    pub fn selected_task(&self) -> Option<&TaskCardModel> {
        self.selected_task
            .as_ref()
            .and_then(|id| self.tasks.iter().find(|t| &t.id == id))
    }

    pub fn set_filter(&mut self, filter: TaskFilter, cx: &mut Context<Self>) {
        self.filter = filter;
        cx.notify();
    }

    pub fn set_search_query(&mut self, query: impl Into<String>, cx: &mut Context<Self>) {
        self.search_query = query.into();
        cx.notify();
    }

    pub fn select_task(&mut self, id: &str, cx: &mut Context<Self>) {
        self.selected_task = Some(id.to_string());
        cx.emit(SidebarEvent::SelectTask(id.to_string()));
        cx.notify();
    }

    pub fn close(&mut self, cx: &mut Context<Self>) {
        cx.emit(SidebarEvent::Close);
        cx.notify();
    }

    // Counts for filter pills
    pub fn filter_counts(&self) -> (usize, usize, usize, usize) {
        let mut total = 0;
        let mut active = 0;
        let mut blocked = 0;
        let mut done = 0;

        for t in &self.tasks {
            total += 1;
            if t.status.is_blocked() {
                blocked += 1;
            } else if is_status_active(&t.status) {
                active += 1;
            } else if t.status.is_terminal() {
                done += 1;
            }
        }
        (total, active, blocked, done)
    }

    // Aggregate statistics across all subagents
    pub fn aggregate_stats(&self) -> (u64, usize, usize) {
        let mut total_tokens = 0;
        let mut total_adds = 0;
        let mut total_dels = 0;

        for t in &self.tasks {
            total_tokens += t.usage.total_tokens;
            let (adds, dels) = t.total_diff_stats();
            total_adds += adds;
            total_dels += dels;
        }

        (total_tokens, total_adds, total_dels)
    }

    fn matches_filter(&self, task: &TaskCardModel) -> bool {
        match self.filter {
            TaskFilter::All => true,
            TaskFilter::Active => is_status_active(&task.status),
            TaskFilter::Blocked => task.status.is_blocked(),
            TaskFilter::Done => task.status.is_terminal(),
        }
    }

    fn matches_search(&self, task: &TaskCardModel) -> bool {
        if self.search_query.trim().is_empty() {
            return true;
        }
        let q = self.search_query.to_lowercase();
        task.task.to_lowercase().contains(&q)
            || task.agent.to_lowercase().contains(&q)
            || task
                .name
                .as_ref()
                .map(|n| n.to_lowercase().contains(&q))
                .unwrap_or(false)
            || task
                .workspace
                .as_ref()
                .map(|w| w.to_lowercase().contains(&q))
                .unwrap_or(false)
            || task.id.to_lowercase().contains(&q)
    }

    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let (total, active, blocked, done) = self.filter_counts();

        div()
            .flex()
            .flex_col()
            .gap_2p5()
            .p_3p5()
            .bg(rgb(0x18181b))
            .border_b_1()
            .border_color(rgb(0x27272a))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(rgb(0xf4f4f5))
                                    .child("⚡ MISSION CONTROL"),
                            )
                            .child(
                                div()
                                    .px_1p5()
                                    .py_0p5()
                                    .rounded_full()
                                    .bg(if blocked > 0 {
                                        rgb(0x451a03)
                                    } else if active > 0 {
                                        rgb(0x1e3a5f)
                                    } else {
                                        rgb(0x27272a)
                                    })
                                    .border_1()
                                    .border_color(if blocked > 0 {
                                        rgb(0xd97706)
                                    } else if active > 0 {
                                        rgb(0x2563eb)
                                    } else {
                                        rgb(0x3f3f46)
                                    })
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(if blocked > 0 {
                                        rgb(0xfbbf24)
                                    } else if active > 0 {
                                        rgb(0x60a5fa)
                                    } else {
                                        rgb(0xa1a1aa)
                                    })
                                    .child(format!("{}", total)),
                            ),
                    )
                    .child(
                        div()
                            .id("close-sidebar-btn")
                            .cursor_pointer()
                            .text_xs()
                            .text_color(rgb(0x71717a))
                            .hover(|s| s.text_color(rgb(0xe4e4e7)))
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                this.close(cx);
                            }))
                            .child("✕"),
                    ),
            )
            .child(
                // Filter Tab Pills
                div()
                    .flex()
                    .gap_1p5()
                    .child(self.filter_pill(TaskFilter::All, format!("All ({})", total), cx))
                    .child(self.filter_pill(TaskFilter::Active, format!("Active ({})", active), cx))
                    .child(self.filter_pill(
                        TaskFilter::Blocked,
                        format!("Blocked ({})", blocked),
                        cx,
                    ))
                    .child(self.filter_pill(TaskFilter::Done, format!("Done ({})", done), cx)),
            )
            .when_some(self.error_message.clone(), |hdr, err| {
                hdr.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .p_2()
                        .rounded_xs()
                        .bg(rgb(0x450a0a))
                        .border_1()
                        .border_color(rgb(0xdc2626))
                        .text_xs()
                        .text_color(rgb(0xfca5a5))
                        .child(err)
                        .child(
                            div()
                                .id("dismiss-sidebar-error-btn")
                                .cursor_pointer()
                                .text_color(rgb(0xf87171))
                                .hover(|s| s.text_color(rgb(0xffffff)))
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.set_error(None, cx);
                                }))
                                .child("✕"),
                        ),
                )
            })
    }

    fn filter_pill(
        &self,
        filter: TaskFilter,
        label: String,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let is_active = self.filter == filter;
        let pill_id = format!("filter-pill-{:?}", filter);

        div()
            .id(SharedString::from(pill_id))
            .cursor_pointer()
            .px_2()
            .py_0p5()
            .rounded_xs()
            .bg(if is_active {
                rgb(0x27272a)
            } else {
                rgb(0x18181b)
            })
            .border_1()
            .border_color(if is_active {
                rgb(0x3f3f46)
            } else {
                rgb(0x27272a)
            })
            .text_xs()
            .font_weight(if is_active {
                gpui::FontWeight::BOLD
            } else {
                gpui::FontWeight::NORMAL
            })
            .text_color(if is_active {
                rgb(0xf4f4f5)
            } else {
                rgb(0x71717a)
            })
            .hover(|s| s.bg(rgb(0x27272a)).text_color(rgb(0xe4e4e7)))
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                this.set_filter(filter, cx);
            }))
            .child(label)
    }

    fn render_summary_footer(&self) -> impl IntoElement {
        let (total_tokens, adds, dels) = self.aggregate_stats();

        div()
            .flex()
            .items_center()
            .justify_between()
            .px_3p5()
            .py_2()
            .bg(rgb(0x18181b))
            .border_t_1()
            .border_color(rgb(0x27272a))
            .text_xs()
            .text_color(rgb(0x71717a))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(format!("Tokens: {} total", total_tokens)),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1p5()
                    .child(div().text_color(rgb(0x34d399)).child(format!("+{}", adds)))
                    .child(div().text_color(rgb(0xf87171)).child(format!("-{}", dels))),
            )
    }

    fn render_empty_state(&self) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .p_6()
            .gap_2()
            .text_center()
            .child(
                div()
                    .text_sm()
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(rgb(0x71717a))
                    .child("No subagents active"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x52525b))
                    .child("When child runs are spawned or queued, Mission Control streams status badges, blocker taxonomy, live Zed deltas, and steering controls here."),
            )
    }
}

impl Focusable for Sidebar {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.focus_handle
            .clone()
            .unwrap_or_else(|| cx.focus_handle())
    }
}

impl Render for Sidebar {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let matching_tasks: Vec<TaskCardModel> = self
            .tasks
            .iter()
            .filter(|t| self.matches_filter(t) && self.matches_search(t))
            .cloned()
            .collect();

        let is_empty = matching_tasks.is_empty();
        let selected_id = self.selected_task.clone();

        div()
            .id("sidebar-mission-control")
            .key_context("Sidebar")
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x141416))
            .border_l_1()
            .border_color(rgb(0x27272a))
            .child(self.render_header(cx))
            .child(
                div()
                    .id("sidebar-task-list-container")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .p_3()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2p5()
                            .when(is_empty, |list| list.child(self.render_empty_state()))
                            .when(!is_empty, |list| {
                                list.children(matching_tasks.into_iter().map(|task_model| {
                                    let task_id = task_model.id.clone();
                                    let is_selected = selected_id.as_deref() == Some(&task_id);
                                    let card_view =
                                        cx.new(|cx| TaskCard::new(task_model, is_selected, cx));

                                    // Forward task card events
                                    cx.subscribe(
                                        &card_view,
                                        |this, _, event: &TaskCardEvent, cx| match event {
                                            TaskCardEvent::Selected(id) => {
                                                this.select_task(id, cx);
                                            }
                                            TaskCardEvent::Pause(id) => {
                                                cx.emit(SidebarEvent::PauseTask(id.clone()));
                                            }
                                            TaskCardEvent::Resume { id, resolution } => {
                                                cx.emit(SidebarEvent::ResumeTask {
                                                    id: id.clone(),
                                                    resolution: resolution.clone(),
                                                });
                                            }
                                            TaskCardEvent::Abort(id) => {
                                                cx.emit(SidebarEvent::AbortTask(id.clone()));
                                            }
                                            TaskCardEvent::Steer { id, guidance } => {
                                                cx.emit(SidebarEvent::SteerTask {
                                                    id: id.clone(),
                                                    guidance: guidance.clone(),
                                                });
                                            }
                                            TaskCardEvent::ToggleExpanded(id) => {
                                                if let Some(t) =
                                                    this.tasks.iter_mut().find(|t| &t.id == id)
                                                {
                                                    t.expanded = !t.expanded;
                                                    cx.notify();
                                                }
                                            }
                                        },
                                    )
                                    .detach();

                                    card_view
                                }))
                            }),
                    ),
            )
            .child(self.render_summary_footer())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_counts_and_matching() {
        let tasks = Sidebar::sample_tasks();
        assert_eq!(tasks.len(), 3);

        let mut sidebar = Sidebar {
            focus_handle: None,
            scroll_handle: ScrollHandle::new(),
            tasks,
            selected_task: None,
            filter: TaskFilter::All,
            search_query: String::new(),
            error_message: None,
        };

        let (total, active, blocked, done) = sidebar.filter_counts();
        assert_eq!(total, 3);
        assert_eq!(active, 1);
        assert_eq!(blocked, 1);
        assert_eq!(done, 1);

        // Test search
        sidebar.search_query = "auth".into();
        let matches: Vec<_> = sidebar
            .tasks
            .iter()
            .filter(|t| sidebar.matches_search(t))
            .collect();
        assert_eq!(matches.len(), 2);

        // Test aggregate stats
        let (_tokens, adds, dels) = sidebar.aggregate_stats();
        assert_eq!(adds, 38);
        assert_eq!(dels, 9);
    }

    #[test]
    fn runtime_summary_synchronization_and_status_transitions() {
        let mut sidebar = Sidebar {
            focus_handle: None,
            scroll_handle: ScrollHandle::new(),
            tasks: Vec::new(),
            selected_task: None,
            filter: TaskFilter::All,
            search_query: String::new(),
            error_message: None,
        };

        // Truthful initial empty state
        assert_eq!(sidebar.tasks().len(), 0);
        let (total, active, blocked, done) = sidebar.filter_counts();
        assert_eq!((total, active, blocked, done), (0, 0, 0, 0));

        // Step 1: Subagent queued
        let summary1 = SubagentSummary {
            id: "child-42".into(),
            name: Some("#42 Refactor".into()),
            task: "Refactor session store".into(),
            status: SubagentStatus::Queued,
            stop_reason: None,
            usage: None,
            error: None,
        };
        sidebar.apply_summaries(&[summary1]);
        assert_eq!(sidebar.tasks().len(), 1);
        assert_eq!(sidebar.filter_counts(), (1, 0, 0, 0));

        // Step 2: Transitions to Running
        let summary2 = SubagentSummary {
            id: "child-42".into(),
            name: Some("#42 Refactor".into()),
            task: "Refactor session store".into(),
            status: SubagentStatus::Running,
            stop_reason: None,
            usage: None,
            error: None,
        };
        sidebar.apply_summaries(&[summary2]);
        assert_eq!(sidebar.filter_counts(), (1, 1, 0, 0));

        // Step 3: Transitions to Blocked (UserInput)
        let summary3 = SubagentSummary {
            id: "child-42".into(),
            name: Some("#42 Refactor".into()),
            task: "Refactor session store".into(),
            status: SubagentStatus::Blocked {
                reason: BlockerReason::UserInput,
                prompt: "Confirm schema change".into(),
            },
            stop_reason: None,
            usage: None,
            error: None,
        };
        sidebar.apply_summaries(&[summary3]);
        assert_eq!(sidebar.filter_counts(), (1, 0, 1, 0));
        assert!(sidebar.tasks()[0].status.is_blocked());

        // Step 4: Transitions to Verifying after unblock
        let summary4 = SubagentSummary {
            id: "child-42".into(),
            name: Some("#42 Refactor".into()),
            task: "Refactor session store".into(),
            status: SubagentStatus::Verifying,
            stop_reason: None,
            usage: None,
            error: None,
        };
        sidebar.apply_summaries(&[summary4]);
        assert_eq!(sidebar.filter_counts(), (1, 1, 0, 0));

        // Step 5: Transitions to Reviewing
        let summary5 = SubagentSummary {
            id: "child-42".into(),
            name: Some("#42 Refactor".into()),
            task: "Refactor session store".into(),
            status: SubagentStatus::Reviewing,
            stop_reason: None,
            usage: None,
            error: None,
        };
        sidebar.apply_summaries(&[summary5]);
        assert_eq!(sidebar.filter_counts(), (1, 1, 0, 0));

        // Step 6: Completed
        let summary6 = SubagentSummary {
            id: "child-42".into(),
            name: Some("#42 Refactor".into()),
            task: "Refactor session store".into(),
            status: SubagentStatus::Completed,
            stop_reason: None,
            usage: None,
            error: None,
        };
        sidebar.apply_summaries(&[summary6]);
        assert_eq!(sidebar.filter_counts(), (1, 0, 0, 1));
        assert!(sidebar.tasks()[0].status.is_terminal());
    }

    #[test]
    fn error_banner_management() {
        let mut sidebar = Sidebar {
            focus_handle: None,
            scroll_handle: ScrollHandle::new(),
            tasks: Vec::new(),
            selected_task: None,
            filter: TaskFilter::All,
            search_query: String::new(),
            error_message: None,
        };

        assert!(sidebar.error_message.is_none());
        sidebar.set_error_message(Some("Steer rejected: child already completed".into()));
        assert_eq!(
            sidebar.error_message.as_ref().map(|s| s.as_ref()),
            Some("Steer rejected: child already completed")
        );

        sidebar.set_error_message(None);
        assert!(sidebar.error_message.is_none());
    }
}
