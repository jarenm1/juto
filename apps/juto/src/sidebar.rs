//! Left sidebar listing the main session's internal threads and its focus
//! threads (subagents that own an orchestrated task).
//!
//! - Internal threads: one row per thread with its label and whether it is the
//!   current, an active, or a dormant thread.
//! - One row per focus thread: title, status, and lifecycle stage bar (see
//!   `task_card`). Clicking a row selects it and opens focus mode.
//! - Direct integration with `juto_runtime::subagents::{BlockerReason, SubagentStatus, SubagentSummary}`.
//! - Filter tabs: All, Active, Blocked, Done.
//! - Search/filter over title, task, agent, workspace, and id.
//! - Blocked rows always expose their prompt and a resume field; expanding a
//!   row reveals steering (Pause, Steer, Abort), usage, step log, and diffs.
//! - Error banner surfaced with visible failure reason on rejected steering or unblock.

use std::collections::HashMap;

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, IntoElement, KeyBinding, Render,
    ScrollHandle, SharedString, Subscription, Window, actions, div, prelude::*, px, rgb,
};
use juto_runtime::subagents::{BlockerReason, MainMessage, SubagentStatus, SubagentSummary};
use juto_runtime::{ThreadInfo, ThreadStatus};

use crate::task_card::{
    DiffFileModel, DiffHunkModel, DiffLineKind, DiffLineModel, TaskCard, TaskCardEvent,
    TaskCardModel, TokenUsageModel, ToolLogEntry, is_status_active,
};
use crate::theme;

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
    DismissInbox(MainMessage),
    Close,
}

/// Left sidebar listing focus threads.
pub struct Sidebar {
    focus_handle: Option<FocusHandle>,
    scroll_handle: ScrollHandle,
    tasks: Vec<TaskCardModel>,
    selected_task: Option<String>,
    filter: TaskFilter,
    search_query: String,
    error_message: Option<SharedString>,
    /// Live row views by task id, with the subscription forwarding their events.
    cards: HashMap<String, (Entity<TaskCard>, Subscription)>,
    threads: ThreadList,
    inbox: Vec<MainMessage>,
}

/// The main session's internal threads, in creation order.
#[derive(Default)]
struct ThreadList {
    items: Vec<ThreadInfo>,
    /// Thread the latest request was routed to.
    current: Option<String>,
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
            cards: HashMap::new(),
            threads: ThreadList::default(),
            inbox: Vec::new(),
        }
    }

    /// Shows the session inbox as the daemon last reported it.
    pub fn set_inbox(&mut self, inbox: Vec<MainMessage>, cx: &mut Context<Self>) {
        self.inbox = inbox;
        cx.notify();
    }

    fn render_inbox(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let messages = self.inbox.clone();
        div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .child(format!("Inbox · {}", messages.len()))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_FAINT))
                    .child("Dismissal does not remove the agent’s copy."),
            )
            .children(messages.into_iter().enumerate().map(|(index, message)| {
                let dismiss = message.clone();
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .p_2()
                    .border_1()
                    .border_color(rgb(theme::TEXT_FAINT))
                    .child(format!("{} · {:?}", message.from, message.kind))
                    .child(div().text_sm().child(message.body))
                    .child(
                        div()
                            .id(("dismiss-inbox", index))
                            .cursor_pointer()
                            .text_xs()
                            .child("Dismiss")
                            .on_click(cx.listener(move |_, _, _, cx| {
                                cx.emit(SidebarEvent::DismissInbox(dismiss.clone()));
                            })),
                    )
            }))
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

    /// Replace the thread list; the most recently routed thread is current.
    pub fn set_threads(&mut self, threads: Vec<ThreadInfo>, cx: &mut Context<Self>) {
        self.threads.current = threads
            .iter()
            .max_by_key(|thread| thread.last_routed)
            .map(|thread| thread.id.clone());
        self.threads.items = threads;
        cx.notify();
    }

    /// Add a new thread or replace an existing one's label and status.
    pub fn upsert_thread(&mut self, thread: ThreadInfo, cx: &mut Context<Self>) {
        match self.threads.items.iter_mut().find(|t| t.id == thread.id) {
            Some(existing) => *existing = thread,
            None => self.threads.items.push(thread),
        }
        cx.notify();
    }

    /// Mark the thread the latest request was routed to.
    pub fn set_current_thread(&mut self, id: &str, cx: &mut Context<Self>) {
        self.threads.current = Some(id.to_owned());
        cx.notify();
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

    /// Focus threads and blocked workers are visible. Once a worker has been
    /// shown, keep updating its card so resolving a blocker clears the prompt.
    pub fn apply_summaries(&mut self, summaries: &[SubagentSummary]) {
        for summary in summaries {
            if !summary.focus
                && !summary.status.is_blocked()
                && !self.tasks.iter().any(|task| task.id == summary.id)
            {
                continue;
            }
            let workspace = summary
                .workspace
                .as_ref()
                .map(|workspace| workspace.display().to_string());
            if let Some(existing) = self.tasks.iter_mut().find(|t| t.id == summary.id) {
                existing.status = summary.status.clone();
                if let Some(name) = &summary.name {
                    existing.name = Some(name.clone());
                }
                if let Some(workspace) = workspace {
                    existing.workspace = Some(workspace);
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
                if let Some(workspace) = workspace {
                    card = card.with_workspace(workspace);
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

    /// Drops the row highlight without emitting a selection event, e.g. when
    /// focus mode closes.
    pub fn clear_selection(&mut self, cx: &mut Context<Self>) {
        self.selected_task = None;
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

    fn render_threads(&self) -> impl IntoElement {
        let current = self.threads.current.as_deref();
        div()
            .flex()
            .flex_col()
            .gap_1()
            .px_5()
            .pt_5()
            .pb_2()
            .border_b_1()
            .border_color(rgb(theme::BORDER))
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(rgb(theme::TEXT_MUTED))
                    .child("THREADS"),
            )
            .children(self.threads.items.iter().map(|thread| {
                let is_current = current == Some(thread.id.as_str());
                let dormant = thread.status == ThreadStatus::Dormant;
                let (dot, state) = if is_current {
                    (theme::ACCENT, "current")
                } else if dormant {
                    (theme::BORDER_STRONG, "dormant")
                } else {
                    (theme::TEXT_MUTED, "active")
                };
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .py_0p5()
                    .child(div().flex_none().size(px(6.)).rounded_full().bg(rgb(dot)))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .text_sm()
                            .text_color(rgb(if dormant {
                                theme::TEXT_FAINT
                            } else {
                                theme::TEXT
                            }))
                            .child(thread.label.clone()),
                    )
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_FAINT))
                            .child(state),
                    )
            }))
    }
    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let (total, active, blocked, done) = self.filter_counts();

        div()
            .flex()
            .flex_col()
            .gap_2p5()
            .px_5()
            .pt_5()
            .pb_2()
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .text_color(rgb(theme::TEXT_MUTED))
                    .child("FOCUS THREADS"),
            )
            .when(total > 0, |hdr| {
                hdr.child(
                    div()
                        .flex()
                        .gap_3()
                        .child(self.filter_tab(TaskFilter::All, total, cx))
                        .child(self.filter_tab(TaskFilter::Active, active, cx))
                        .child(self.filter_tab(TaskFilter::Blocked, blocked, cx))
                        .child(self.filter_tab(TaskFilter::Done, done, cx)),
                )
            })
            .when_some(self.error_message.clone(), |hdr, err| {
                hdr.child(
                    div()
                        .flex()
                        .items_start()
                        .justify_between()
                        .gap_2()
                        .p_2()
                        .rounded_sm()
                        .bg(rgb(theme::BG_RAISED))
                        .border_l_2()
                        .border_color(rgb(theme::DANGER))
                        .text_xs()
                        .text_color(rgb(theme::TEXT))
                        .child(div().flex_1().min_w_0().child(err))
                        .child(
                            div()
                                .id("dismiss-sidebar-error-btn")
                                .flex_none()
                                .cursor_pointer()
                                .text_color(rgb(theme::TEXT_FAINT))
                                .hover(|s| s.text_color(rgb(theme::TEXT)))
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.set_error(None, cx);
                                }))
                                .child("✕"),
                        ),
                )
            })
    }

    fn filter_tab(
        &self,
        filter: TaskFilter,
        count: usize,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let is_active = self.filter == filter;

        div()
            .id(SharedString::from(format!("filter-pill-{:?}", filter)))
            .flex()
            .gap_1()
            .cursor_pointer()
            .text_xs()
            .text_color(if is_active {
                rgb(theme::TEXT)
            } else {
                rgb(theme::TEXT_FAINT)
            })
            .hover(|s| s.text_color(rgb(theme::TEXT)))
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                this.set_filter(filter, cx);
            }))
            .child(filter.label())
            .child(
                div()
                    .text_color(if is_active {
                        rgb(theme::ACCENT)
                    } else {
                        rgb(theme::TEXT_FAINT)
                    })
                    .child(count.to_string()),
            )
    }

    fn render_summary_footer(&self) -> impl IntoElement {
        let (total_tokens, adds, dels) = self.aggregate_stats();

        div()
            .flex()
            .items_center()
            .justify_between()
            .px_5()
            .py_2p5()
            .border_t_1()
            .border_color(rgb(theme::BORDER))
            .text_xs()
            .text_color(rgb(theme::TEXT_FAINT))
            .child(format!("{total_tokens} tokens"))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1p5()
                    .child(
                        div()
                            .text_color(rgb(theme::DIFF_ADD))
                            .child(format!("+{adds}")),
                    )
                    .child(
                        div()
                            .text_color(rgb(theme::DANGER))
                            .child(format!("-{dels}")),
                    ),
            )
    }

    fn render_empty_state(&self) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_1p5()
            .px_5()
            .py_2()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(theme::TEXT_MUTED))
                    .child("No focus threads"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_FAINT))
                    .child("Threads the agent starts for orchestrated tasks appear here."),
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

impl Sidebar {
    /// Returns the live card for `task`, creating it on first sight. Cards
    /// persist across renders so click state, focus, and typed resume/steer
    /// text survive re-renders.
    fn card_for(
        &mut self,
        task: TaskCardModel,
        is_selected: bool,
        cx: &mut Context<Self>,
    ) -> Entity<TaskCard> {
        if let Some((card, _)) = self.cards.get(&task.id) {
            card.update(cx, |card, _| card.sync(task, is_selected));
            return card.clone();
        }
        let id = task.id.clone();
        let card = cx.new(|cx| TaskCard::new(task, is_selected, cx));
        let subscription = cx.subscribe(&card, Self::on_card_event);
        self.cards.insert(id, (card.clone(), subscription));
        card
    }

    fn on_card_event(
        &mut self,
        _: Entity<TaskCard>,
        event: &TaskCardEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            TaskCardEvent::Selected(id) => self.select_task(id, cx),
            TaskCardEvent::Pause(id) => cx.emit(SidebarEvent::PauseTask(id.clone())),
            TaskCardEvent::Resume { id, resolution } => cx.emit(SidebarEvent::ResumeTask {
                id: id.clone(),
                resolution: resolution.clone(),
            }),
            TaskCardEvent::Abort(id) => cx.emit(SidebarEvent::AbortTask(id.clone())),
            TaskCardEvent::Steer { id, guidance } => cx.emit(SidebarEvent::SteerTask {
                id: id.clone(),
                guidance: guidance.clone(),
            }),
            TaskCardEvent::ToggleExpanded(id) => {
                if let Some(task) = self.tasks.iter_mut().find(|t| &t.id == id) {
                    task.expanded = !task.expanded;
                    cx.notify();
                }
            }
        }
    }
}

impl Render for Sidebar {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let tasks = &self.tasks;
        self.cards
            .retain(|id, _| tasks.iter().any(|task| &task.id == id));

        let matching_tasks: Vec<TaskCardModel> = self
            .tasks
            .iter()
            .filter(|t| self.matches_filter(t) && self.matches_search(t))
            .cloned()
            .collect();
        let cards: Vec<Entity<TaskCard>> = matching_tasks
            .into_iter()
            .map(|task| {
                let is_selected = self.selected_task.as_deref() == Some(task.id.as_str());
                self.card_for(task, is_selected, cx)
            })
            .collect();

        div()
            .id("sidebar-mission-control")
            .key_context("Sidebar")
            .flex()
            .flex_col()
            .size_full()
            .when(!self.threads.items.is_empty(), |sidebar| {
                sidebar.child(self.render_threads())
            })
            .child(self.render_header(cx))
            .child(
                div()
                    .id("sidebar-task-list-container")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .child(self.render_inbox(cx))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_0p5()
                            .when(cards.is_empty(), |list| {
                                list.child(self.render_empty_state())
                            })
                            .children(cards),
                    ),
            )
            .when(!self.tasks.is_empty(), |sidebar| {
                sidebar.child(self.render_summary_footer())
            })
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
            cards: HashMap::new(),
            threads: ThreadList::default(),
            inbox: Vec::new(),
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
            cards: HashMap::new(),
            threads: ThreadList::default(),
            inbox: Vec::new(),
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
            focus: true,
            workspace: None,
            released: false,
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
            focus: true,
            workspace: None,
            released: false,
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
            focus: true,
            workspace: None,
            released: false,
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
            focus: true,
            workspace: None,
            released: false,
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
            focus: true,
            workspace: None,
            released: false,
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
            focus: true,
            workspace: None,
            released: false,
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
            cards: HashMap::new(),
            threads: ThreadList::default(),
            inbox: Vec::new(),
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

    #[test]
    fn focus_and_blocked_worker_summaries_become_cards() {
        let mut sidebar = Sidebar {
            focus_handle: None,
            scroll_handle: ScrollHandle::new(),
            tasks: Vec::new(),
            selected_task: None,
            filter: TaskFilter::All,
            search_query: String::new(),
            error_message: None,
            cards: HashMap::new(),
            threads: ThreadList::default(),
            inbox: Vec::new(),
        };
        let summary = |id: &str, focus: bool| SubagentSummary {
            id: id.into(),
            name: None,
            task: "task".into(),
            status: SubagentStatus::Running,
            stop_reason: None,
            usage: None,
            error: None,
            focus,
            workspace: None,
            released: false,
        };

        sidebar.apply_summaries(&[summary("worker-1", false), summary("focus-1", true)]);

        let ids: Vec<&str> = sidebar.tasks().iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, vec!["focus-1"]);

        let mut worker = summary("worker-1", false);
        worker.status = SubagentStatus::Blocked {
            reason: BlockerReason::UserInput,
            prompt: "Choose a target".into(),
        };
        sidebar.apply_summaries(&[worker.clone()]);
        assert!(sidebar.tasks()[1].status.is_blocked());
        worker.status = SubagentStatus::Running;
        sidebar.apply_summaries(&[worker]);
        assert!(!sidebar.tasks()[1].status.is_blocked());
    }

    #[test]
    fn focus_summary_workspace_is_carried_into_card() {
        use std::path::PathBuf;

        let mut sidebar = Sidebar {
            focus_handle: None,
            scroll_handle: ScrollHandle::new(),
            tasks: Vec::new(),
            selected_task: None,
            filter: TaskFilter::All,
            search_query: String::new(),
            error_message: None,
            cards: HashMap::new(),
            threads: ThreadList::default(),
            inbox: Vec::new(),
        };
        let summary = SubagentSummary {
            id: "focus-1".into(),
            name: None,
            task: "task".into(),
            status: SubagentStatus::Running,
            stop_reason: None,
            usage: None,
            error: None,
            focus: true,
            workspace: Some(PathBuf::from("/tmp/juto-focus-1")),
            released: false,
        };

        sidebar.apply_summaries(&[summary]);

        assert_eq!(
            sidebar.tasks()[0].workspace.as_deref(),
            Some("/tmp/juto-focus-1")
        );
    }
}
