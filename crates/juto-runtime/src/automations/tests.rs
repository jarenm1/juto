use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::broker::Priority;
use crate::jobs::tests::{Answer, Stalled, budget, native_repo, no_change, runtime};
use crate::jobs::{JobOrigin, JobStatus};
use tempfile::{TempDir, tempdir};

/// 2024-03-30T12:00:00Z.
const T0: u64 = 1_711_800_000_000;
const MINUTE: u64 = 60_000;

fn spec(every_secs: u64) -> AutomationSpec {
    AutomationSpec {
        prompt: "tidy the notes".into(),
        model: None,
        priority: None,
        budget: budget(60),
        contract: no_change(),
        every_secs,
        overlap: None,
        tier: None,
        permission: None,
    }
}

fn fixture() -> (TempDir, PathBuf, PathBuf) {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data = root.path().join("data");
    (root, repo, data)
}

/// Polls the stored job until `done` holds.
async fn wait_for(runtime: &Runtime, id: &str, done: impl Fn(&Job) -> bool) -> Job {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let job = runtime.job(id).expect("job exists");
            if done(&job) {
                return job;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("job reached the expected state in time")
}

/// Simulates a crash after a job was persisted but before the schedule cursor was.
fn rewind_cursor(data: &Path, id: &str, next_instant_ms: u64) {
    let path = data.join("automations").join(format!("{id}.json"));
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read record"))
            .expect("parse record");
    value["nextInstantMs"] = next_instant_ms.into();
    std::fs::write(&path, value.to_string()).expect("write record");
}

/// Simulates a hand-edited record: one field is replaced on disk.
fn rewrite_record(data: &Path, id: &str, field: &str, value: serde_json::Value) {
    let path = data.join("automations").join(format!("{id}.json"));
    let mut record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read record"))
            .expect("parse record");
    record[field] = value;
    std::fs::write(&path, record.to_string()).expect("write record");
}

fn origin(automation: &Automation, instant_ms: u64) -> Option<JobOrigin> {
    Some(JobOrigin {
        automation_id: automation.id.clone(),
        instant_ms,
    })
}

#[tokio::test]
async fn a_due_tick_starts_exactly_one_job() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");
    assert_eq!(automation.next_instant_ms, T0 + MINUTE);

    assert!(
        runtime
            .tick_automations(T0 + MINUTE - 1)
            .expect("tick")
            .is_empty(),
        "nothing is due before the first interval"
    );

    let started = runtime.tick_automations(T0 + MINUTE).expect("tick");
    assert_eq!(started.len(), 1, "one due occurrence, one job");
    let first = &started[0];
    assert_eq!(first.id, occurrence_job_id(&automation.id, T0 + MINUTE));
    assert_eq!(first.origin, origin(&automation, T0 + MINUTE));
    assert_eq!(first.priority, Some(Priority::Scheduled));
    wait_for(&runtime, &first.id, |job| job.status.is_terminal()).await;
    assert_eq!(runtime.list_jobs().expect("list").len(), 1);
}

#[tokio::test]
async fn a_repeat_tick_for_the_same_occurrence_starts_no_second_job() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");

    let first = runtime.tick_automations(T0 + MINUTE).expect("tick");
    assert_eq!(first.len(), 1);
    wait_for(&runtime, &first[0].id, |job| job.status.is_terminal()).await;

    assert!(
        runtime
            .tick_automations(T0 + MINUTE)
            .expect("tick")
            .is_empty(),
        "a repeat tick for the same occurrence starts no second job"
    );
    assert!(
        runtime
            .tick_automations(T0 + MINUTE + 30_000)
            .expect("tick")
            .is_empty(),
        "the next occurrence is not due yet"
    );

    let second = runtime.tick_automations(T0 + 2 * MINUTE).expect("tick");
    assert_eq!(second.len(), 1);
    assert_ne!(
        second[0].id, first[0].id,
        "distinct occurrences, distinct jobs"
    );
    wait_for(&runtime, &second[0].id, |job| job.status.is_terminal()).await;

    let history = runtime.automation_history(&automation.id).expect("history");
    assert_eq!(history.jobs.len(), 2);
    assert_eq!(
        history.events,
        vec![
            AutomationEvent::Fired {
                at_ms: T0 + MINUTE,
                instant_ms: T0 + MINUTE,
                job_id: first[0].id.clone(),
            },
            AutomationEvent::Fired {
                at_ms: T0 + 2 * MINUTE,
                instant_ms: T0 + 2 * MINUTE,
                job_id: second[0].id.clone(),
            },
        ]
    );
}

#[tokio::test]
async fn a_scheduled_job_carries_the_stored_tier_and_permission_profile() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Stalled));
    let automation = runtime
        .create_automation_at(
            AutomationSpec {
                tier: Some(ExecutionTier::Untrusted),
                permission: Some(PermissionProfile::ReadOnly),
                ..spec(60)
            },
            T0,
        )
        .expect("create");
    assert_eq!(automation.tier, ExecutionTier::Untrusted);
    assert_eq!(automation.permission, PermissionProfile::ReadOnly);

    let started = runtime.tick_automations(T0 + MINUTE).expect("tick");
    assert_eq!(started.len(), 1);
    let job = runtime.job(&started[0].id).expect("job");
    assert_eq!(job.tier, Some(ExecutionTier::Untrusted));
    assert_eq!(job.permission, Some(PermissionProfile::ReadOnly));
    assert_eq!(
        runtime.automation(&automation.id).expect("load").tier,
        ExecutionTier::Untrusted,
        "the stored tier survives a reload"
    );
    runtime.cancel_job(&job.id).expect("cancel");
    wait_for(&runtime, &job.id, |job| job.status.is_terminal()).await;
}

#[tokio::test]
async fn a_definition_without_a_tier_runs_workspace_and_unattended() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");
    assert_eq!(automation.tier, ExecutionTier::Workspace);
    assert_eq!(automation.permission, PermissionProfile::Unattended);
}

#[tokio::test]
async fn a_tier_above_the_unattended_ceiling_is_refused_at_create_and_at_start() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    for tier in [ExecutionTier::Attended, ExecutionTier::Full] {
        let error = runtime
            .create_automation_at(
                AutomationSpec {
                    tier: Some(tier),
                    ..spec(60)
                },
                T0,
            )
            .expect_err("refused");
        match error {
            AutomationError::Invalid(message) => {
                assert!(message.contains(tier.as_str()), "{message}");
                assert!(message.contains("`untrusted` or `workspace`"), "{message}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }
    assert!(runtime.list_automations().expect("list").is_empty());

    let automation = runtime.create_automation_at(spec(60), T0).expect("create");
    rewrite_record(&data, &automation.id, "tier", "full".into());
    assert!(
        runtime
            .tick_automations(T0 + MINUTE)
            .expect("tick")
            .is_empty(),
        "a record edited past the ceiling starts no job"
    );
    assert!(runtime.list_jobs().expect("list").is_empty());
    assert!(matches!(
        runtime
            .automation_history(&automation.id)
            .expect("history")
            .events
            .as_slice(),
        [AutomationEvent::Failed { .. }]
    ));
}

#[tokio::test]
async fn a_restart_does_not_start_a_second_job_for_an_occurrence_it_already_started() {
    let (_root, repo, data) = fixture();
    let before = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = before.create_automation_at(spec(60), T0).expect("create");
    let first = before.tick_automations(T0 + MINUTE).expect("tick");
    assert_eq!(first.len(), 1);
    wait_for(&before, &first[0].id, |job| job.status.is_terminal()).await;
    drop(before);

    // The job was persisted, but the crash lost the schedule cursor write.
    rewind_cursor(&data, &automation.id, T0 + MINUTE);

    let after = runtime(&repo, &data, Arc::new(Answer("done")));
    assert!(
        after
            .tick_automations(T0 + MINUTE + 1)
            .expect("tick")
            .is_empty(),
        "the occurrence's job already exists, so none starts"
    );
    assert_eq!(
        after.list_jobs().expect("list").len(),
        1,
        "no duplicate job"
    );
    assert_eq!(
        after
            .automation(&automation.id)
            .expect("load")
            .next_instant_ms,
        T0 + 2 * MINUTE,
        "the cursor moves past the occurrence"
    );

    let next = after.tick_automations(T0 + 2 * MINUTE).expect("tick");
    assert_eq!(next.len(), 1, "the next occurrence still runs");
    assert_eq!(after.list_jobs().expect("list").len(), 2);
}

#[tokio::test]
async fn missed_occurrences_run_once_and_are_recorded_as_missed() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");

    // The daemon was down for three intervals: 1, 2 and 3 minutes are missed.
    let started = runtime.tick_automations(T0 + 4 * MINUTE).expect("tick");
    assert_eq!(
        started.len(),
        1,
        "a missed stretch runs once, not four times"
    );
    assert_eq!(started[0].origin, origin(&automation, T0 + 4 * MINUTE));
    wait_for(&runtime, &started[0].id, |job| job.status.is_terminal()).await;

    let history = runtime.automation_history(&automation.id).expect("history");
    assert!(
        history.events.contains(&AutomationEvent::Missed {
            at_ms: T0 + 4 * MINUTE,
            from_ms: T0 + MINUTE,
            count: 3,
        }),
        "the three earlier occurrences are recorded: {:?}",
        history.events
    );

    assert_eq!(
        runtime
            .tick_automations(T0 + 5 * MINUTE)
            .expect("tick")
            .len(),
        1,
        "the schedule resumes on the grid"
    );
}

#[tokio::test]
async fn overlap_skip_records_the_occurrence_and_moves_on() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Stalled));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");

    let started = runtime.tick_automations(T0 + MINUTE).expect("tick");
    assert_eq!(started.len(), 1);
    wait_for(&runtime, &started[0].id, |job| {
        job.status == JobStatus::Running
    })
    .await;

    assert!(
        runtime
            .tick_automations(T0 + 2 * MINUTE)
            .expect("tick")
            .is_empty(),
        "the open job blocks the next occurrence"
    );
    let skipped = runtime.automation(&automation.id).expect("load");
    assert_eq!(
        skipped.next_instant_ms,
        T0 + 3 * MINUTE,
        "the schedule moved on"
    );
    assert!(
        runtime
            .automation_history(&automation.id)
            .expect("history")
            .events
            .contains(&AutomationEvent::Skipped {
                at_ms: T0 + 2 * MINUTE,
                instant_ms: T0 + 2 * MINUTE,
                count: 1,
            })
    );

    runtime.cancel_job(&started[0].id).expect("cancel");
    wait_for(&runtime, &started[0].id, |job| job.status.is_terminal()).await;
    let after = runtime.tick_automations(T0 + 3 * MINUTE).expect("tick");
    assert_eq!(
        after.len(),
        1,
        "once the job settled, the next occurrence runs"
    );
    assert_eq!(after[0].origin, origin(&automation, T0 + 3 * MINUTE));
    runtime.cancel_job(&after[0].id).expect("cancel");
}

#[tokio::test]
async fn overlap_queue_holds_one_occurrence_until_the_open_job_settles() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Stalled));
    let automation = runtime
        .create_automation_at(
            AutomationSpec {
                overlap: Some(Overlap::Queue),
                ..spec(60)
            },
            T0,
        )
        .expect("create");

    let started = runtime.tick_automations(T0 + MINUTE).expect("tick");
    assert_eq!(started.len(), 1);
    wait_for(&runtime, &started[0].id, |job| {
        job.status == JobStatus::Running
    })
    .await;

    assert!(
        runtime
            .tick_automations(T0 + 2 * MINUTE)
            .expect("tick")
            .is_empty()
    );
    assert!(
        runtime
            .tick_automations(T0 + 3 * MINUTE)
            .expect("tick")
            .is_empty()
    );
    assert_eq!(
        runtime
            .automation(&automation.id)
            .expect("load")
            .next_instant_ms,
        T0 + 2 * MINUTE,
        "the queued occurrence keeps its place"
    );

    runtime.cancel_job(&started[0].id).expect("cancel");
    wait_for(&runtime, &started[0].id, |job| job.status.is_terminal()).await;
    let after = runtime.tick_automations(T0 + 3 * MINUTE).expect("tick");
    assert_eq!(after.len(), 1, "one job for the latest due occurrence");
    assert_eq!(after[0].origin, origin(&automation, T0 + 3 * MINUTE));
    assert!(
        runtime
            .automation_history(&automation.id)
            .expect("history")
            .events
            .contains(&AutomationEvent::Missed {
                at_ms: T0 + 3 * MINUTE,
                from_ms: T0 + 2 * MINUTE,
                count: 1,
            }),
        "the coalesced earlier occurrence is recorded"
    );
    assert_eq!(runtime.list_jobs().expect("list").len(), 2);
    runtime.cancel_job(&after[0].id).expect("cancel");
}

#[tokio::test]
async fn a_paused_automation_never_fires_and_enabling_skips_the_paused_stretch() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");
    runtime
        .set_enabled_at(&automation.id, false, T0)
        .expect("pause");

    for minutes in 1..=3 {
        assert!(
            runtime
                .tick_automations(T0 + minutes * MINUTE)
                .expect("tick")
                .is_empty(),
            "a paused automation fires nothing"
        );
    }
    assert!(runtime.list_jobs().expect("list").is_empty());

    let resumed = runtime
        .set_enabled_at(&automation.id, true, T0 + 3 * MINUTE)
        .expect("enable");
    assert_eq!(
        resumed.next_instant_ms,
        T0 + 4 * MINUTE,
        "the occurrences during the pause are not replayed"
    );
    assert!(
        runtime
            .automation_history(&automation.id)
            .expect("history")
            .events
            .contains(&AutomationEvent::Missed {
                at_ms: T0 + 3 * MINUTE,
                from_ms: T0 + MINUTE,
                count: 3,
            })
    );
    assert!(
        runtime
            .tick_automations(T0 + 3 * MINUTE)
            .expect("tick")
            .is_empty()
    );
    let started = runtime.tick_automations(T0 + 4 * MINUTE).expect("tick");
    assert_eq!(started.len(), 1, "enabled again, the next occurrence runs");
}

#[tokio::test]
async fn run_now_starts_a_job_without_moving_the_schedule_and_refuses_while_one_is_open() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Stalled));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");

    let run = runtime
        .run_now_at(&automation.id, T0 + 10_000)
        .expect("run now");
    assert_eq!(run.origin, origin(&automation, T0 + 10_000));
    assert_eq!(run.priority, Some(Priority::Scheduled));
    wait_for(&runtime, &run.id, |job| job.status == JobStatus::Running).await;

    assert!(
        matches!(
            runtime.run_now_at(&automation.id, T0 + 20_000),
            Err(AutomationError::Invalid(_))
        ),
        "a second run while one is open is refused"
    );
    assert_eq!(
        runtime
            .automation(&automation.id)
            .expect("load")
            .next_instant_ms,
        T0 + MINUTE,
        "run-now leaves the schedule alone"
    );

    runtime.cancel_job(&run.id).expect("cancel");
    wait_for(&runtime, &run.id, |job| job.status.is_terminal()).await;
    let again = runtime
        .run_now_at(&automation.id, T0 + 30_000)
        .expect("run now once the first run settled");
    runtime.cancel_job(&again.id).expect("cancel");
}

#[test]
fn an_hourly_schedule_is_utc_arithmetic_across_a_daylight_saving_change() {
    // 2024-03-31T01:00:00Z, when Central European clocks move forward.
    const SWITCH: u64 = 1_711_846_800_000;
    const HOUR: u64 = 3_600_000;
    let created = T0;

    assert_eq!(latest_due(created, HOUR, SWITCH), SWITCH);
    assert_eq!(latest_due(created, HOUR, SWITCH - 1), SWITCH - HOUR);
    assert_eq!(latest_due(created, HOUR, SWITCH + HOUR - 1), SWITCH);
    assert_eq!(
        latest_due(created, HOUR, SWITCH + HOUR),
        SWITCH + HOUR,
        "the occurrence after the change is exactly one UTC hour later"
    );
    assert_eq!((SWITCH - created) % HOUR, 0, "the change lies on the grid");
}

#[tokio::test]
async fn a_definition_that_does_not_fit_is_refused_and_nothing_is_stored() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let refused = |result: Result<Automation, AutomationError>| {
        assert!(
            matches!(result, Err(AutomationError::Invalid(_))),
            "got {result:?}"
        );
    };

    refused(runtime.create_automation_at(spec(0), T0));
    refused(runtime.create_automation_at(spec(MAX_EVERY_SECS + 1), T0));
    refused(runtime.create_automation_at(
        AutomationSpec {
            prompt: "   ".into(),
            ..spec(60)
        },
        T0,
    ));
    refused(runtime.create_automation_at(
        AutomationSpec {
            priority: Some(Priority::Attended),
            ..spec(60)
        },
        T0,
    ));
    refused(runtime.create_automation_at(
        AutomationSpec {
            contract: crate::jobs::JobContract::default(),
            ..spec(60)
        },
        T0,
    ));
    refused(runtime.create_automation_at(
        AutomationSpec {
            model: Some("nope/missing".into()),
            ..spec(60)
        },
        T0,
    ));
    assert!(runtime.list_automations().expect("list").is_empty());
}

#[tokio::test]
async fn deleting_removes_the_definition_and_keeps_its_jobs() {
    let (_root, repo, data) = fixture();
    let runtime = runtime(&repo, &data, Arc::new(Answer("done")));
    let automation = runtime.create_automation_at(spec(60), T0).expect("create");
    let started = runtime.tick_automations(T0 + MINUTE).expect("tick");
    wait_for(&runtime, &started[0].id, |job| job.status.is_terminal()).await;

    runtime.delete_automation(&automation.id).expect("delete");

    assert!(matches!(
        runtime.automation(&automation.id),
        Err(AutomationError::NotFound(_))
    ));
    assert!(runtime.list_automations().expect("list").is_empty());
    assert!(
        runtime
            .tick_automations(T0 + 2 * MINUTE)
            .expect("tick")
            .is_empty(),
        "a deleted automation never fires"
    );
    assert_eq!(
        runtime.list_jobs().expect("list").len(),
        1,
        "the job it started stays in the job store"
    );
}
