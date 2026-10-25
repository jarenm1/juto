//! Left sidebar: the main session's internal threads, then every task the
//! user may care about, grouped by what it needs from them.
//!
//! - Internal threads: one row per thread with its label and whether it is the
//!   current, an active, or a dormant thread.
//! - Tasks: every subagent main started and every background job, in three
//!   sections. **Needs you** holds blocked subagents (waiting on the user's
//!   answer) and jobs waiting on a decision; **Running** holds queued and
//!   working ones; **Done** holds finished ones until the user archives them.
//! - Subagent rows are `task_card` rows: title, status, and lifecycle stage
//!   bar. Clicking one opens it in the chat surface. Blocked rows always expose
//!   their prompt and an answer field; expanding a row reveals steering (Pause,
//!   Steer, Abort), usage, step log, and diffs.
//! - Job rows show their status, prompt, result, and the actions `jobs_panel`
//!   offers (approve, deny, follow up, archive).
//! - Error banner surfaced with visible failure reason on rejected steering or unblock.

use std::collections::HashMap;

use gpui::{
    App, ClickEvent, Context, Entity, FocusHandle, Focusable, IntoElement, KeyBinding, Render,
    ScrollHandle, SharedString, Subscription, Window, actions, div, prelude::*, rgb,
};
use juto_runtime::jobs::{Job, JobStatus};
use juto_runtime::subagents::{BlockerReason, SubagentStatus, SubagentSummary};

use crate::jobs_panel::{self, JobActionHandler};
use crate::task_card::{
    DiffFileModel, DiffHunkModel, DiffLineKind, DiffLineModel, TaskCard, TaskCardEvent,
    TaskCardModel, TokenUsageModel, ToolLogEntry,
};
use crate::theme;

actions!(sidebar, [CloseSidebar]);

pub fn init(cx: &mut App) {
    cx.bind_keys([KeyBinding::new("escape", CloseSidebar, Some("Sidebar"))]);
}

/// Sidebar section a task belongs to, in display order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    NeedsYou,
    Running,
    Done,
}

impl Section {
    pub fn label(self) -> &'static str {
        match self {
            Self::NeedsYou => "Needs you",
            Self::Running => "Running",
            Self::Done => "Done",
        }
    }

    /// A blocked subagent waits on the user; a finished one is done.
    pub fn of_subagent(status: &SubagentStatus) -> Self {
        if status.is_blocked() {
            Self::NeedsYou
        } else if status.is_terminal() {
            Self::Done
        } else {
            Self::Running
        }
    }

    /// A blocked job waits on the user's decision; a finished one is done.
    pub fn of_job(job: &Job) -> Self {
        if job.status == JobStatus::Blocked {
            Self::NeedsYou
        } else if job.status.is_terminal() {
            Self::Done
        } else {
            Self::Running
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
    ArchiveFinished,
    Close,
}

/// Left sidebar listing the session's threads and its tasks.
pub struct Sidebar {
    focus_handle: Option<FocusHandle>,
    scroll_handle: ScrollHandle,
    tasks: Vec<TaskCardModel>,
    selected_task: Option<String>,
    search_query: String,
    error_message: Option<SharedString>,
    /// Live row views by task id, with the subscription forwarding their events.
    cards: HashMap<String, (Entity<TaskCard>, Subscription)>,
    /// Background jobs as the daemon last reported them, archived ones included.
    jobs: Vec<Job>,
    /// Routes job row clicks to the chat view, which owns the daemon requests.
    job_actions: Option<JobActionHandler>,
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
            search_query: String::new(),
            error_message: None,
            cards: HashMap::new(),
            jobs: Vec::new(),
            job_actions: None,
        }
    }

    /// Shows `jobs` in the task sections; `actions` runs their row buttons.
    pub fn set_jobs(&mut self, jobs: Vec<Job>, actions: JobActionHandler, cx: &mut Context<Self>) {
        self.jobs = jobs;
        self.job_actions = Some(actions);
        cx.notify();
    }

    /// Jobs shown in `section`, newest first. Archived jobs are hidden.
    fn jobs_in(&self, section: Section) -> Vec<&Job> {
        jobs_panel::visible(&self.jobs)
            .into_iter()
            .filter(|job| Section::of_job(job) == section)
            .collect()
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

    /// Every subagent main started is listed until the user archives it.
    /// Nested children belong to their parent's run and never reach this list.
    pub fn apply_summaries(&mut self, summaries: &[SubagentSummary]) {
        self.tasks.retain(|task| {
            summaries
                .iter()
                .any(|summary| summary.id == task.id && !summary.archived)
        });
        for summary in summaries {
            if summary.archived {
                continue;
            }
            let workspace = summary
                .workspace
                .as_ref()
                .map(|workspace| workspace.display().to_string());
            if let Some(existing) = self.tasks.iter_mut().find(|t| t.id == summary.id) {
                existing.status = summary.status.clone();
                existing.focus = summary.focus;
                existing.summary = summary.summary.clone();
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
                card.focus = summary.focus;
                card.summary = summary.summary.clone();

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

    /// How many subagents and jobs each section holds, in [`Section`] order.
    pub fn section_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        let sections = self
            .tasks
            .iter()
            .map(|task| Section::of_subagent(&task.status))
            .chain(
                jobs_panel::visible(&self.jobs)
                    .into_iter()
                    .map(Section::of_job),
            );
        for section in sections {
            counts[section as usize] += 1;
        }
        counts
    }

    /// Subagents and jobs waiting on the user.
    pub fn needs_you_count(&self) -> usize {
        self.section_counts()[Section::NeedsYou as usize]
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
        let done = self.section_counts()[Section::Done as usize];

        div()
            .flex()
            .flex_col()
            .gap_2p5()
            .px_5()
            .pt_5()
            .pb_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(rgb(theme::TEXT_MUTED))
                            .child("TASKS"),
                    )
                    .when(done > 0, |row| {
                        row.child(
                            div()
                                .id("archive-finished-btn")
                                .text_xs()
                                .cursor_pointer()
                                .text_color(rgb(theme::TEXT_MUTED))
                                .hover(|s| s.text_color(rgb(theme::TEXT)))
                                .on_click(cx.listener(|_, _: &ClickEvent, _, cx| {
                                    cx.emit(SidebarEvent::ArchiveFinished);
                                }))
                                .child(format!("Archive {done} done")),
                        )
                    }),
            )
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

    /// A section heading with its count. "Needs you" is drawn in the warning
    /// color so it stands out from the work the agents own.
    fn render_section_heading(section: Section, count: usize) -> impl IntoElement {
        let color = match section {
            Section::NeedsYou => theme::WARNING,
            Section::Running | Section::Done => theme::TEXT_FAINT,
        };
        div()
            .flex()
            .items_center()
            .gap_1p5()
            .px_5()
            .pt_3()
            .pb_1()
            .text_xs()
            .font_weight(gpui::FontWeight::MEDIUM)
            .text_color(rgb(color))
            .child(section.label())
            .child(
                div()
                    .text_color(rgb(theme::TEXT_FAINT))
                    .child(count.to_string()),
            )
    }

    /// The job rows of `section`, inset to line up with the subagent rows.
    fn render_job_rows(&self, section: Section) -> Option<impl IntoElement> {
        let handler = self.job_actions.as_ref()?;
        let jobs = self.jobs_in(section);
        (!jobs.is_empty()).then(|| {
            div().flex().flex_col().gap_1().px_3().py_1().children(
                jobs.into_iter()
                    .map(|job| jobs_panel::job_row(job, handler)),
            )
        })
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
                    .child("No tasks"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_FAINT))
                    .child("Work the agent hands off and background jobs appear here."),
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

        let mut sections: Vec<(Section, Vec<Entity<TaskCard>>)> = Vec::new();
        for section in [Section::NeedsYou, Section::Running, Section::Done] {
            // Focus threads, the ones the user is working with, lead their section.
            let mut matching: Vec<TaskCardModel> = self
                .tasks
                .iter()
                .filter(|t| Section::of_subagent(&t.status) == section && self.matches_search(t))
                .cloned()
                .collect();
            matching.sort_by_key(|task| !task.focus);
            let cards = matching
                .into_iter()
                .map(|task| {
                    let is_selected = self.selected_task.as_deref() == Some(task.id.as_str());
                    self.card_for(task, is_selected, cx)
                })
                .collect();
            sections.push((section, cards));
        }
        let counts = self.section_counts();
        let empty = counts.iter().all(|count| *count == 0);

        div()
            .id("sidebar-mission-control")
            .key_context("Sidebar")
            .flex()
            .flex_col()
            .size_full()
            .child(self.render_header(cx))
            .child(
                div()
                    .id("sidebar-task-list-container")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .when(empty, |list| list.child(self.render_empty_state()))
                    .children(
                        sections
                            .into_iter()
                            .filter(|(section, _)| counts[*section as usize] > 0)
                            .map(|(section, cards)| {
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_0p5()
                                    .child(Self::render_section_heading(
                                        section,
                                        counts[section as usize],
                                    ))
                                    .children(cards)
                                    .children(self.render_job_rows(section))
                            }),
                    ),
            )
            .when(!self.tasks.is_empty(), |sidebar| {
                sidebar.child(self.render_summary_footer())
            })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::jobs_panel::test_job;
    use juto_runtime::jobs::InboxState;

    fn sidebar(tasks: Vec<TaskCardModel>) -> Sidebar {
        Sidebar {
            focus_handle: None,
            scroll_handle: ScrollHandle::new(),
            tasks,
            selected_task: None,
            search_query: String::new(),
            error_message: None,
            cards: HashMap::new(),
            jobs: Vec::new(),
            job_actions: None,
        }
    }

    fn summary(id: &str, status: SubagentStatus) -> SubagentSummary {
        SubagentSummary {
            id: id.into(),
            name: None,
            task: "task".into(),
            status,
            stop_reason: None,
            usage: None,
            error: None,
            focus: false,
            summary: None,
            workspace: None,
            released: false,
            archived: false,
        }
    }

    fn blocked() -> SubagentStatus {
        SubagentStatus::Blocked {
            reason: BlockerReason::UserInput,
            prompt: "Merge now?".into(),
        }
    }

    fn ids(sidebar: &Sidebar) -> Vec<&str> {
        sidebar.tasks().iter().map(|t| t.id.as_str()).collect()
    }

    #[test]
    fn sample_tasks_fill_each_section_and_search_narrows_rows() {
        let mut sidebar = sidebar(Sidebar::sample_tasks());
        assert_eq!(sidebar.section_counts(), [1, 1, 1]);

        sidebar.search_query = "auth".into();
        let matches = sidebar
            .tasks
            .iter()
            .filter(|t| sidebar.matches_search(t))
            .count();
        assert_eq!(matches, 2);

        let (_tokens, adds, dels) = sidebar.aggregate_stats();
        assert_eq!((adds, dels), (38, 9));
    }

    #[test]
    fn a_task_moves_between_sections_as_its_status_changes() {
        let mut sidebar = sidebar(Vec::new());
        let steps = [
            (SubagentStatus::Queued, [0, 1, 0]),
            (SubagentStatus::Running, [0, 1, 0]),
            (blocked(), [1, 0, 0]),
            (SubagentStatus::Verifying, [0, 1, 0]),
            (SubagentStatus::Reviewing, [0, 1, 0]),
            (SubagentStatus::Completed, [0, 0, 1]),
        ];
        for (status, counts) in steps {
            sidebar.apply_summaries(&[summary("child-42", status.clone())]);
            assert_eq!(ids(&sidebar), ["child-42"], "one row through {status:?}");
            assert_eq!(sidebar.section_counts(), counts, "sections for {status:?}");
        }
        assert_eq!(sidebar.needs_you_count(), 0);
    }

    #[test]
    fn every_task_is_listed_until_archived() {
        let mut sidebar = sidebar(Vec::new());
        let mut focus = summary("focus-1", SubagentStatus::Running);
        focus.focus = true;
        let mut worker = summary("worker-1", SubagentStatus::Completed);
        sidebar.apply_summaries(&[focus.clone(), worker.clone()]);
        assert_eq!(
            ids(&sidebar),
            ["focus-1", "worker-1"],
            "workers are tasks too, and finished ones stay until archived"
        );

        worker.archived = true;
        sidebar.apply_summaries(&[focus, worker]);
        assert_eq!(ids(&sidebar), ["focus-1"]);
    }

    #[test]
    fn blocked_tasks_and_jobs_awaiting_a_decision_need_the_user() {
        let mut sidebar = sidebar(Vec::new());
        sidebar.apply_summaries(&[
            summary("asking", blocked()),
            summary("working", SubagentStatus::Running),
        ]);
        let mut decision = test_job("pr-7", JobStatus::Blocked, InboxState::Unread);
        decision.decision_id = Some("d1".into());
        sidebar.jobs = vec![
            decision,
            test_job("scan", JobStatus::Running, InboxState::Unread),
            test_job("old", JobStatus::Succeeded, InboxState::Archived),
            test_job("new", JobStatus::Succeeded, InboxState::Unread),
        ];

        assert_eq!(sidebar.section_counts(), [2, 2, 1]);
        assert_eq!(sidebar.needs_you_count(), 2);
        let needs_you: Vec<&str> = sidebar
            .jobs_in(Section::NeedsYou)
            .into_iter()
            .map(|job| job.id.as_str())
            .collect();
        assert_eq!(needs_you, ["pr-7"]);
    }

    #[test]
    fn summary_workspace_is_carried_into_card() {
        let mut sidebar = sidebar(Vec::new());
        let mut task = summary("focus-1", SubagentStatus::Running);
        task.workspace = Some(PathBuf::from("/tmp/juto-focus-1"));
        sidebar.apply_summaries(&[task]);
        assert_eq!(
            sidebar.tasks()[0].workspace.as_deref(),
            Some("/tmp/juto-focus-1")
        );
    }
}
