//! Background jobs in the chat view: the strip that lists them, the actions offered on each,
//! and the daemon request each action sends. Job state lives in the daemon; this module only
//! presents it and turns clicks into requests.

use std::rc::Rc;

use gpui::{App, ClickEvent, Div, SharedString, Window, div, prelude::*, px, rgb};
use juto_daemon::protocol::Request;
use juto_runtime::jobs::{InboxState, Job, JobBudget, JobContract, JobStatus};

use crate::theme;

/// Wall-clock budget for a job started from the composer.
pub const BACKGROUND_WALL_SECS: u64 = 3600;
/// Attempts a job started from the composer may make, counting the first.
pub const BACKGROUND_MAX_ATTEMPTS: u32 = 1;

/// What the user can do with a job from the strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobAction {
    MarkRead,
    Archive,
    /// Seeds the composer with the job's result so the user can ask a follow-up.
    FollowUp,
    Approve,
    Deny,
}

/// Calls a job action for the job with the given id.
pub type JobActionHandler = Rc<dyn Fn(JobAction, &str, &mut Window, &mut App)>;

/// The request that starts a background job for `prompt`. A job started from the composer
/// must report and leave the workspace unchanged, so its contract is `no_change`.
pub fn background_request(prompt: String) -> Request {
    Request::StartJob {
        prompt,
        budget: JobBudget {
            wall_secs: BACKGROUND_WALL_SECS,
            max_attempts: BACKGROUND_MAX_ATTEMPTS,
            max_turns: None,
            max_spend: None,
        },
        contract: JobContract {
            no_change: true,
            ..Default::default()
        },
    }
}

/// Jobs the strip shows: every job the user has not archived, newest first.
pub fn visible(jobs: &[Job]) -> Vec<&Job> {
    let mut shown: Vec<&Job> = jobs
        .iter()
        .filter(|job| job.inbox != InboxState::Archived)
        .collect();
    shown.sort_by_key(|job| std::cmp::Reverse(job.created_ms));
    shown
}

/// The status as the strip words it. A blocked job says what it needs.
pub fn status_label(job: &Job) -> &'static str {
    match job.status {
        JobStatus::Queued => "Queued",
        JobStatus::Running => "Running",
        JobStatus::Blocked => "Needs a decision",
        JobStatus::Succeeded => "Done",
        JobStatus::Failed => "Failed",
        JobStatus::Interrupted => "Interrupted",
        JobStatus::Cancelled => "Cancelled",
    }
}

/// The actions offered on `job`, in display order.
pub fn available_actions(job: &Job) -> Vec<JobAction> {
    let mut actions = Vec::new();
    if job.status == JobStatus::Blocked && job.decision_id.is_some() {
        actions.push(JobAction::Approve);
        actions.push(JobAction::Deny);
    }
    if job.status.is_terminal() {
        if job.result.is_some() {
            actions.push(JobAction::FollowUp);
        }
        if job.inbox == InboxState::Unread {
            actions.push(JobAction::MarkRead);
        }
        actions.push(JobAction::Archive);
    }
    actions
}

/// The request an action sends, or `None` for actions the view handles locally.
pub fn action_request(action: JobAction, job: &Job) -> Option<Request> {
    match action {
        JobAction::MarkRead => Some(Request::SetJobInbox {
            job_id: job.id.clone(),
            state: InboxState::Read,
        }),
        JobAction::Archive => Some(Request::SetJobInbox {
            job_id: job.id.clone(),
            state: InboxState::Archived,
        }),
        JobAction::Approve | JobAction::Deny => {
            job.decision_id
                .clone()
                .map(|decision_id| Request::ResolveDecision {
                    decision_id,
                    approve: action == JobAction::Approve,
                })
        }
        JobAction::FollowUp => None,
    }
}

/// The composer text that follows up on a job, seeded with its result on one line.
pub fn follow_up_text(job: &Job) -> String {
    let result = job.result.as_deref().unwrap_or("(no result)");
    let flat = result.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("Follow up on background job {}: {flat}", job.id)
}

fn action_label(action: JobAction) -> &'static str {
    match action {
        JobAction::MarkRead => "Mark read",
        JobAction::Archive => "Archive",
        JobAction::FollowUp => "Follow up",
        JobAction::Approve => "Approve",
        JobAction::Deny => "Deny",
    }
}

/// `text` shortened to at most `limit` characters, with an ellipsis when cut.
fn shorten(text: &str, limit: usize) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

fn action_button(job: &Job, action: JobAction, handler: &JobActionHandler) -> impl IntoElement {
    let handler = handler.clone();
    let job_id = job.id.clone();
    let color = match action {
        JobAction::Approve => theme::ACCENT,
        JobAction::Deny => theme::DANGER,
        _ => theme::TEXT,
    };
    let id = SharedString::from(format!("job-{}-{}", job.id, action_label(action)));
    div()
        .id(id)
        .px_2()
        .py_0p5()
        .rounded_sm()
        .border_1()
        .border_color(rgb(theme::BORDER_STRONG))
        .text_xs()
        .text_color(rgb(color))
        .cursor_pointer()
        .hover(|style| style.bg(rgb(theme::BG_HOVER)))
        .on_click(move |_: &ClickEvent, window, cx| handler(action, &job_id, window, cx))
        .child(action_label(action))
}

fn job_row(job: &Job, handler: &JobActionHandler) -> impl IntoElement {
    let mut row = div()
        .flex()
        .flex_col()
        .gap_1()
        .px_3()
        .py_2()
        .rounded_md()
        .border_1()
        .border_color(rgb(theme::BORDER))
        .bg(rgb(theme::BG_RAISED))
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .text_xs()
                .child(
                    div()
                        .text_color(rgb(theme::TEXT_MUTED))
                        .child(status_label(job)),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .text_color(rgb(theme::TEXT))
                        .child(shorten(&job.prompt, 80)),
                ),
        );
    if let Some(result) = job.result.as_deref().filter(|_| job.status.is_terminal()) {
        row = row.child(
            div()
                .text_xs()
                .text_color(rgb(theme::TEXT_MUTED))
                .child(shorten(result, 160)),
        );
    }
    if let Some(error) = job.error.as_deref() {
        row = row.child(
            div()
                .text_xs()
                .text_color(rgb(theme::DANGER))
                .child(shorten(error, 160)),
        );
    }
    let actions = available_actions(job);
    if !actions.is_empty() {
        row = row.child(
            div().flex().gap_2().children(
                actions
                    .into_iter()
                    .map(|action| action_button(job, action, handler)),
            ),
        );
    }
    row
}

/// The strip above the composer. Empty (and taking no space) when nothing is shown.
pub fn render_strip(jobs: &[Job], handler: &JobActionHandler) -> Div {
    let shown = visible(jobs);
    if shown.is_empty() {
        return div();
    }
    div()
        .flex()
        .flex_col()
        .gap_2()
        .flex_none()
        .px_6()
        .pb_2()
        .w_full()
        .max_w(px(800.))
        .mx_auto()
        .children(shown.into_iter().map(|job| job_row(job, handler)))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn job(id: &str, status: JobStatus, inbox: InboxState) -> Job {
        Job {
            id: id.into(),
            prompt: "summarize the repo".into(),
            budget: JobBudget {
                wall_secs: 60,
                max_attempts: 1,
                max_turns: None,
                max_spend: None,
            },
            contract: JobContract {
                no_change: true,
                ..Default::default()
            },
            status,
            inbox,
            drained: false,
            attempts: 1,
            run_ms: 0,
            turns: 0,
            spend_usd: 0.0,
            session_id: None,
            session_path: None::<PathBuf>,
            forks: Vec::new(),
            decision_id: None,
            result: None,
            error: None,
            owner_pid: None,
            created_ms: 0,
            started_ms: None,
            finished_ms: None,
        }
    }

    #[test]
    fn archived_jobs_leave_the_strip_and_the_rest_are_newest_first() {
        let mut old = job("old", JobStatus::Succeeded, InboxState::Unread);
        old.created_ms = 1;
        let mut new = job("new", JobStatus::Running, InboxState::Read);
        new.created_ms = 2;
        let archived = job("gone", JobStatus::Succeeded, InboxState::Archived);
        let jobs = [old, archived, new];
        let ids: Vec<&str> = visible(&jobs).iter().map(|job| job.id.as_str()).collect();
        assert_eq!(ids, vec!["new", "old"]);
    }

    #[test]
    fn a_blocked_job_offers_its_decision_and_a_finished_one_its_result() {
        let mut blocked = job("b", JobStatus::Blocked, InboxState::Unread);
        blocked.decision_id = Some("d1".into());
        assert_eq!(
            available_actions(&blocked),
            vec![JobAction::Approve, JobAction::Deny]
        );
        let request = action_request(JobAction::Approve, &blocked).expect("a decision request");
        assert!(matches!(
            request,
            Request::ResolveDecision { ref decision_id, approve: true } if decision_id == "d1"
        ));

        let mut done = job("d", JobStatus::Succeeded, InboxState::Unread);
        done.result = Some("ok".into());
        assert_eq!(
            available_actions(&done),
            vec![JobAction::FollowUp, JobAction::MarkRead, JobAction::Archive]
        );
    }

    #[test]
    fn a_job_without_a_decision_offers_no_decision_action() {
        let blocked = job("b", JobStatus::Blocked, InboxState::Unread);
        assert!(available_actions(&blocked).is_empty());
        assert!(action_request(JobAction::Approve, &blocked).is_none());
    }

    #[test]
    fn follow_up_seeds_the_composer_with_the_result_on_one_line() {
        let mut done = job("abc", JobStatus::Succeeded, InboxState::Read);
        done.result = Some("first line\nsecond  line".into());
        assert_eq!(
            follow_up_text(&done),
            "Follow up on background job abc: first line second line"
        );
    }

    #[test]
    fn background_jobs_are_report_only_with_a_fixed_budget() {
        let Request::StartJob {
            prompt,
            budget,
            contract,
        } = background_request("count the tests".into())
        else {
            panic!("expected a start request");
        };
        assert_eq!(prompt, "count the tests");
        assert_eq!(budget.wall_secs, BACKGROUND_WALL_SECS);
        assert_eq!(budget.max_attempts, BACKGROUND_MAX_ATTEMPTS);
        assert!(contract.no_change);
        assert!(contract.artifact.is_none() && contract.output_schema.is_none());
    }

    #[test]
    fn shorten_cuts_long_text_and_marks_it() {
        assert_eq!(shorten("short", 10), "short");
        assert_eq!(shorten("abcdef", 3), "abc…");
    }
}
