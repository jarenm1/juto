use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::*;
use async_trait::async_trait;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, CredentialStore, Message, Provider, ProviderError,
    ProviderEvent, ProviderStream, StreamOptions, ToolCall, Usage,
};
use juto_catalog::{Model, ModelCost};
use tempfile::tempdir;
use tokio::sync::broadcast::error::RecvError;
use tokio_util::sync::CancellationToken;

fn git(dir: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn native_repo(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create repo dir");
    git(dir, &["init", "-q"]);
    git(dir, &["config", "user.name", "Jobs Test"]);
    git(dir, &["config", "user.email", "jobs@example.test"]);
    std::fs::write(dir.join("README.md"), "# Initial\n").expect("write base file");
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "-q", "-m", "initial"]);
}

fn jj(dir: &Path, args: &[&str]) {
    let out = std::process::Command::new("jj")
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run jj");
    assert!(
        out.status.success(),
        "jj {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A git repo colocated with jj, with bookmark `feat/x`, so `publish` can pin it.
fn publishable_repo(dir: &Path) {
    native_repo(dir);
    jj(dir, &["git", "init", "--colocate"]);
    jj(dir, &["bookmark", "create", "feat/x", "-r", "@-"]);
}

fn mock_config() -> crate::RuntimeConfig {
    crate::RuntimeConfig {
        model: "mock/test".to_string(),
        custom_models: vec![Model {
            id: "test".to_string(),
            name: "Mock Model".to_string(),
            api: "openai-completions".to_string(),
            provider: "mock".to_string(),
            base_url: "http://localhost".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: Some(8192),
            max_tokens: Some(1024),
            kind: "chat".to_string(),
            supports_tools: true,
            compat: serde_json::Value::Null,
        }],
        ..crate::RuntimeConfig::default()
    }
}

fn runtime(cwd: &Path, data_dir: &Path, provider: Arc<dyn Provider>) -> Runtime {
    let credentials =
        Arc::new(CredentialStore::open(data_dir.join("credentials.json")).expect("credentials"));
    Runtime::with_provider(cwd, data_dir, mock_config(), credentials, provider, None)
        .expect("runtime")
}

fn no_change() -> JobContract {
    JobContract {
        no_change: true,
        ..Default::default()
    }
}

fn artifact(path: &str) -> JobContract {
    JobContract {
        artifact: Some(path.into()),
        ..Default::default()
    }
}

fn budget(wall_secs: u64) -> JobBudget {
    JobBudget {
        wall_secs,
        max_attempts: 1,
    }
}

fn assistant(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "mock".to_string(),
        provider: "mock".to_string(),
        model: "mock/test".to_string(),
        usage: Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 2,
            cost: 0.0,
        },
        stop_reason,
        timestamp: timestamp_ms(),
        response_id: None,
        error: None,
    }
}

fn one_message(message: AssistantMessage) -> Result<ProviderStream, ProviderError> {
    let events: Vec<Result<ProviderEvent, ProviderError>> = vec![
        Ok(ProviderEvent::Start),
        Ok(ProviderEvent::Done { message }),
    ];
    Ok(Box::pin(futures::stream::iter(events)))
}

fn text_answer(text: &str) -> AssistantMessage {
    assistant(
        vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        StopReason::Stop,
    )
}

fn tool_call(name: &str, arguments: serde_json::Value) -> AssistantMessage {
    assistant(
        vec![ContentBlock::ToolCall(ToolCall {
            id: format!("{name}-call"),
            name: name.to_string(),
            arguments,
            thought_signature: None,
        })],
        StopReason::ToolUse,
    )
}

/// Answers every turn with the same text.
struct Answer(&'static str);

#[async_trait]
impl Provider for Answer {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        one_message(text_answer(self.0))
    }
}

/// Plays a job's main thread, which cannot write: it delegates the change to a
/// `builder` child, waits for it, then reports completion. The builder writes
/// `job.txt` on its first turn.
#[derive(Default)]
struct DelegatingWriter;

#[async_trait]
impl Provider for DelegatingWriter {
    async fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let is_builder = context
            .system_prompt
            .iter()
            .any(|part| part.contains("You are a builder"));
        let message = if is_builder {
            match context.messages.last() {
                Some(Message::ToolResult { .. }) => text_answer("builder finished"),
                _ => tool_call(
                    "write",
                    serde_json::json!({ "path": "job.txt", "content": "from the job" }),
                ),
            }
        } else {
            match context.messages.last() {
                Some(Message::ToolResult {
                    tool_name, details, ..
                }) if tool_name == "task" => {
                    let id = details[0]["id"].as_str().expect("child id").to_owned();
                    tool_call("wait", serde_json::json!({ "ids": [id] }))
                }
                Some(Message::ToolResult { .. }) => text_answer("wrote it"),
                Some(Message::User { content, .. })
                    if content.iter().any(|block| {
                        matches!(block, ContentBlock::Text { text } if text.starts_with("[subagent "))
                    }) =>
                {
                    text_answer("wrote it")
                }
                _ => tool_call(
                    "task",
                    serde_json::json!({
                        "tasks": [{
                            "name": "writer",
                            "agent": "builder",
                            "task": "write job.txt",
                        }]
                    }),
                ),
            }
        };
        one_message(message)
    }
}

/// Asks for `publish` on its first turn, then records the result it receives.
#[derive(Default)]
struct PublishThenAnswer {
    results: std::sync::Mutex<Vec<String>>,
}

#[async_trait]
impl Provider for PublishThenAnswer {
    async fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        if let Some(Message::ToolResult { content, .. }) = context.messages.last() {
            let text: String = content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            self.results.lock().unwrap().push(text);
            one_message(text_answer("done"))
        } else {
            one_message(tool_call(
                "publish",
                serde_json::json!({ "bookmark": "feat/x", "title": "Add x" }),
            ))
        }
    }
}

/// Never produces a response, so only the wall budget or a cancel can end a run.
struct Stalled;

#[async_trait]
impl Provider for Stalled {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        Ok(Box::pin(futures::stream::pending()))
    }
}

/// Counts provider calls; used to prove a job never reached the model.
#[derive(Default)]
struct Counting {
    calls: AtomicUsize,
}

#[async_trait]
impl Provider for Counting {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        one_message(text_answer("should not run"))
    }
}

/// Waits on job events until `done` holds for the job, re-checking the stored
/// state first so an event that landed before the wait is not missed.
async fn settle(
    runtime: &Runtime,
    events: &mut broadcast::Receiver<Job>,
    id: &str,
    done: impl Fn(&Job) -> bool,
) -> Job {
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            let job = runtime.job(id).expect("job exists");
            if done(&job) {
                return job;
            }
            match events.recv().await {
                Ok(_) | Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => panic!("job events closed"),
            }
        }
    })
    .await
    .expect("job settled in time")
}

fn terminal_or_blocked(job: &Job) -> bool {
    job.status.is_terminal() || job.status == JobStatus::Blocked
}

#[tokio::test]
async fn a_job_runs_in_its_own_fork_and_leaves_the_repo_untouched() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(DelegatingWriter));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("write the file", budget(60), artifact("job.txt"))
        .expect("start");
    let job = settle(&runtime, &mut events, &started.id, terminal_or_blocked).await;

    assert_eq!(job.status, JobStatus::Succeeded, "error: {:?}", job.error);
    assert_eq!(job.result.as_deref(), Some("wrote it"));
    assert_eq!(job.attempts, 1);
    assert_eq!(job.forks.len(), 1);
    let fork = &job.forks[0].path;
    assert_eq!(
        std::fs::read_to_string(fork.join("job.txt")).expect("file in fork"),
        "from the job"
    );
    assert!(
        !repo.join("job.txt").exists(),
        "a job must never write into the user's working copy"
    );
    assert!(job.session_path.expect("session path").exists());
}

#[tokio::test]
async fn a_blocked_job_resumes_after_approval_and_runs_the_action_once() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    publishable_repo(&repo);
    let data_dir = root.path().join("data");
    let provider = Arc::new(PublishThenAnswer::default());
    let runtime = runtime(&repo, &data_dir, provider.clone());
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("publish feat/x", budget(60), no_change())
        .expect("start");
    let blocked = settle(&runtime, &mut events, &started.id, terminal_or_blocked).await;
    assert_eq!(
        blocked.status,
        JobStatus::Blocked,
        "error: {:?}",
        blocked.error
    );
    let decision_id = blocked.decision_id.clone().expect("decision recorded");
    let decision = runtime
        .decision_store()
        .get(&decision_id)
        .expect("read decision")
        .expect("decision exists");
    assert_eq!(decision.job_id.as_deref(), Some(started.id.as_str()));
    assert_eq!(decision.capability, "publish:feat/x");

    assert!(
        matches!(runtime.resume_job(&started.id), Err(JobError::Invalid(_))),
        "a job must not resume before its decision is approved"
    );

    let (resolved, _) = runtime.decide_job(&decision_id, true).expect("approve");
    assert_eq!(resolved.status, DecisionStatus::Approved);
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Succeeded, "error: {:?}", job.error);
    let results = provider.results.lock().unwrap().clone();
    assert_eq!(results.len(), 1, "the approved action runs exactly once");
    assert!(
        !results[0].contains("blocked"),
        "the granted call must run: {}",
        results[0]
    );
    let consumed = runtime
        .decision_store()
        .get(&decision_id)
        .expect("read decision")
        .expect("decision exists");
    assert_eq!(consumed.status, DecisionStatus::Consumed);
}

#[tokio::test]
async fn a_rejected_block_fails_the_job_without_running_the_action() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    publishable_repo(&repo);
    let data_dir = root.path().join("data");
    let provider = Arc::new(PublishThenAnswer::default());
    let runtime = runtime(&repo, &data_dir, provider.clone());
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("publish feat/x", budget(60), no_change())
        .expect("start");
    let blocked = settle(&runtime, &mut events, &started.id, terminal_or_blocked).await;
    let decision_id = blocked.decision_id.expect("decision recorded");

    let (resolved, job) = runtime.decide_job(&decision_id, false).expect("reject");
    assert_eq!(resolved.status, DecisionStatus::Rejected);
    assert_eq!(job.status, JobStatus::Failed);
    assert!(
        job.error
            .as_deref()
            .is_some_and(|error| error.contains("rejected")),
        "error: {:?}",
        job.error
    );
    assert!(
        provider.results.lock().unwrap().is_empty(),
        "a rejected action must not run"
    );
}

#[tokio::test]
async fn a_job_over_its_wall_budget_fails() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Stalled));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("stall", budget(1), no_change())
        .expect("start");
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Failed);
    assert!(
        job.error
            .as_deref()
            .is_some_and(|error| error.contains("wall-clock budget")),
        "error: {:?}",
        job.error
    );
    assert!(job.run_ms >= 1000, "run_ms: {}", job.run_ms);
}

#[tokio::test]
async fn cancelling_a_running_job_settles_it_cancelled() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Stalled));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("stall", budget(60), no_change())
        .expect("start");
    settle(&runtime, &mut events, &started.id, |job| {
        job.status == JobStatus::Running
    })
    .await;
    runtime.cancel_job(&started.id).expect("cancel");
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Cancelled, "error: {:?}", job.error);
    assert!(matches!(
        runtime.cancel_job(&started.id),
        Err(JobError::Invalid(_))
    ));
}

#[tokio::test]
async fn a_restart_interrupts_unfinished_jobs_and_retry_runs_them_again() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let store = JobStore {
        dir: data_dir.join("jobs"),
    };
    // A job whose owner process is gone, and one whose owner (PID 1) is alive.
    let mut orphaned = Job::new(
        "orphaned".into(),
        "answer".into(),
        JobBudget {
            wall_secs: 60,
            max_attempts: 2,
        },
        no_change(),
    );
    orphaned.status = JobStatus::Running;
    orphaned.attempts = 1;
    orphaned.owner_pid = Some(u32::MAX);
    store.write(&orphaned).expect("write orphan");
    let mut foreign = Job::new("foreign".into(), "answer".into(), budget(60), no_change());
    foreign.status = JobStatus::Running;
    foreign.owner_pid = Some(1);
    store.write(&foreign).expect("write foreign");

    let runtime = runtime(&repo, &data_dir, Arc::new(Answer("recovered")));
    assert_eq!(runtime.recover_jobs().expect("recover"), 1);
    let interrupted = runtime.job("orphaned").expect("orphan");
    assert_eq!(interrupted.status, JobStatus::Interrupted);
    assert!(interrupted.error.is_some());
    assert_eq!(
        runtime.job("foreign").expect("foreign").status,
        JobStatus::Running,
        "a job owned by another live process is left alone"
    );

    let mut events = runtime.subscribe_jobs();
    runtime.retry_job("orphaned").expect("retry");
    let job = settle(&runtime, &mut events, "orphaned", |job| {
        job.status.is_terminal()
    })
    .await;
    assert_eq!(job.status, JobStatus::Succeeded, "error: {:?}", job.error);
    assert_eq!(job.attempts, 2);
    assert_eq!(job.result.as_deref(), Some("recovered"));

    assert!(
        matches!(runtime.retry_job("orphaned"), Err(JobError::Invalid(_))),
        "a succeeded job is not retried"
    );
}

#[tokio::test]
async fn archiving_reclaims_the_forks_and_keeps_the_transcript() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(DelegatingWriter));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("write", budget(60), artifact("job.txt"))
        .expect("start");
    let done = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;
    let fork = done.forks[0].path.clone();
    let session_path = done.session_path.clone().expect("session path");
    assert!(fork.exists());

    let archived = runtime
        .set_job_inbox(&started.id, InboxState::Archived)
        .await
        .expect("archive");
    assert_eq!(archived.inbox, InboxState::Archived);
    assert!(archived.forks.is_empty());
    assert!(!fork.exists(), "archiving reclaims the isolated workspace");
    assert!(session_path.exists(), "the transcript outlives the fork");
    assert!(matches!(
        runtime.set_job_inbox(&started.id, InboxState::Read).await,
        Err(JobError::Invalid(_))
    ));
    assert!(matches!(
        runtime.retry_job(&started.id),
        Err(JobError::Invalid(_))
    ));
}

/// Archiving must reject a retry that lands while its forks are still being
/// reclaimed; otherwise the retry restarts work in a fork being torn down.
#[tokio::test]
async fn a_retry_cannot_land_while_archiving_reclaims_the_forks() {
    use futures::FutureExt;
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Answer("done")));
    let mut events = runtime.subscribe_jobs();

    let retryable = JobBudget {
        max_attempts: 2,
        ..budget(60)
    };
    let started = runtime
        .start_job("write", retryable, artifact("job.txt"))
        .expect("start");
    let failed = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;
    assert_eq!(failed.status, JobStatus::Failed);
    assert!(
        !failed.forks.is_empty(),
        "the failed job still owns its fork"
    );

    let mut archive = Box::pin(runtime.set_job_inbox(&started.id, InboxState::Archived));
    assert!(
        archive.as_mut().now_or_never().is_none(),
        "archiving waits on the fork reclaim"
    );
    assert!(
        matches!(runtime.retry_job(&started.id), Err(JobError::Invalid(_))),
        "a retry must not land while the job is being archived"
    );
    let archived = archive.await.expect("archive");
    assert_eq!(archived.inbox, InboxState::Archived);
    assert!(archived.forks.is_empty());
}

#[tokio::test]
async fn a_job_that_cannot_be_isolated_fails_before_the_model_runs() {
    let root = tempdir().expect("tempdir");
    // A plain directory, not a repository: no fork can be provisioned.
    let cwd = root.path().join("plain");
    std::fs::create_dir_all(&cwd).expect("create dir");
    let data_dir = root.path().join("data");
    let provider = Arc::new(Counting::default());
    let runtime = runtime(&cwd, &data_dir, provider.clone());
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("anything", budget(60), no_change())
        .expect("start");
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Failed);
    assert!(
        job.error
            .as_deref()
            .is_some_and(|error| error.contains("isolated workspace")),
        "error: {:?}",
        job.error
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_job_needs_a_prompt_and_a_budget() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let runtime = runtime(&repo, &root.path().join("data"), Arc::new(Answer("x")));

    assert!(matches!(
        runtime.start_job("   ", budget(60), no_change()),
        Err(JobError::Invalid(_))
    ));
    assert!(matches!(
        runtime.start_job("work", budget(0), no_change()),
        Err(JobError::Invalid(_))
    ));
    assert!(runtime.list_jobs().expect("list").is_empty());
}

#[test]
fn a_completion_contract_must_name_a_check_inside_the_workspace() {
    assert!(JobContract::default().validate().is_err());
    assert!(
        artifact("/etc/passwd").validate().is_err(),
        "an artifact must be a relative path inside the workspace"
    );
    assert!(artifact("out/result.txt").validate().is_ok());
    assert!(no_change().validate().is_ok());
}

#[tokio::test]
async fn a_job_without_a_completion_contract_is_refused_before_it_runs() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Answer("done")));

    assert!(matches!(
        runtime.start_job("work", budget(60), JobContract::default()),
        Err(JobError::Invalid(_))
    ));
    assert!(runtime.list_jobs().expect("list").is_empty());
}

#[tokio::test]
async fn a_model_stop_without_the_promised_artifact_fails_the_job() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Answer("all done")));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("write the file", budget(60), artifact("job.txt"))
        .expect("start");
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Failed);
    let error = job.error.expect("failure reason");
    assert!(error.contains("completion contract not met"), "{error}");
}

/// Answers its first turn with JSON plus a tool call, then stops on an empty
/// turn. The job's final answer is the empty turn.
#[derive(Default)]
struct JsonThenEmptyStop {
    calls: AtomicUsize,
}

#[async_trait]
impl Provider for JsonThenEmptyStop {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            one_message(assistant(
                vec![
                    ContentBlock::Text {
                        text: r#"{"ok":true}"#.to_string(),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "read-call".to_string(),
                        name: "read".to_string(),
                        arguments: serde_json::json!({"path": "missing.txt"}),
                        thought_signature: None,
                    }),
                ],
                StopReason::ToolUse,
            ))
        } else {
            one_message(assistant(Vec::new(), StopReason::Stop))
        }
    }
}

#[tokio::test]
async fn a_silent_final_turn_fails_the_output_schema_contract() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(JsonThenEmptyStop::default()));
    let mut events = runtime.subscribe_jobs();
    let contract = JobContract {
        output_schema: Some(serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"]
        })),
        ..Default::default()
    };

    let started = runtime
        .start_job("answer", budget(60), contract)
        .expect("start");
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Failed, "the final turn is empty");
    let error = job.error.expect("failure reason");
    assert!(error.contains("final answer is empty"), "{error}");
}

#[tokio::test]
async fn a_no_change_job_that_edits_its_workspace_fails_its_contract() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(DelegatingWriter));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("claim nothing changed", budget(60), no_change())
        .expect("start");
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.status.is_terminal()
    })
    .await;

    assert_eq!(job.status, JobStatus::Failed, "error: {:?}", job.error);
    assert_ne!(
        job.inbox,
        InboxState::Archived,
        "a failed job is kept for review"
    );
}

#[tokio::test]
async fn a_verified_no_change_job_is_archived_and_its_forks_reclaimed() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Answer("nothing to change")));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("look and report", budget(60), no_change())
        .expect("start");
    // Archiving marks the job first and reclaims its forks after, so wait for both.
    let job = settle(&runtime, &mut events, &started.id, |job| {
        job.inbox == InboxState::Archived && job.forks.is_empty()
    })
    .await;

    assert_eq!(job.status, JobStatus::Succeeded, "error: {:?}", job.error);
    assert!(job.forks.is_empty(), "archiving reclaims the job's forks");
    assert!(job.session_path.expect("transcript kept").exists());
}

#[tokio::test]
async fn drained_results_are_returned_once_even_after_a_restart() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(DelegatingWriter));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("write the file", budget(60), artifact("job.txt"))
        .expect("start");
    settle(&runtime, &mut events, &started.id, terminal_or_blocked).await;

    let first = runtime.drain_job_results().expect("drain");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].id, started.id);
    assert!(runtime.drain_job_results().expect("drain again").is_empty());
    drop(runtime);

    let restarted = runtime_for_restart(&repo, &data_dir);
    assert!(
        restarted
            .drain_job_results()
            .expect("drain after restart")
            .is_empty(),
        "a drained result is not delivered again after a restart"
    );
    assert_eq!(
        restarted.job(&started.id).expect("job").status,
        JobStatus::Succeeded
    );
}

fn runtime_for_restart(cwd: &Path, data_dir: &Path) -> Runtime {
    runtime(cwd, data_dir, Arc::new(Answer("unused")))
}

#[tokio::test]
async fn a_settled_job_has_no_run_control_left_for_a_retry_to_lose() {
    let root = tempdir().expect("tempdir");
    let repo = root.path().join("repo");
    native_repo(&repo);
    let data_dir = root.path().join("data");
    let runtime = runtime(&repo, &data_dir, Arc::new(Stalled));
    let mut events = runtime.subscribe_jobs();

    let started = runtime
        .start_job("stall", budget(60), no_change())
        .expect("start");
    settle(&runtime, &mut events, &started.id, |job| {
        job.status == JobStatus::Running
    })
    .await;

    // Hold the live-run map, as a retry does while it registers its control.
    let live = runtime
        .inner
        .jobs
        .live
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let finisher = {
        let runtime = runtime.clone();
        let id = started.id.clone();
        std::thread::spawn(move || {
            runtime.finish(
                &id,
                Settled::Failed {
                    error: "attempt failed".to_owned(),
                    result: None,
                },
            )
        })
    };
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !runtime
            .job(&started.id)
            .expect("job exists")
            .status
            .is_terminal(),
        "a job must not show as settled while its run control is still registered"
    );
    drop(live);
    finisher.join().expect("finish thread").expect("finish");
    assert!(
        runtime.inner.jobs.control(&started.id).is_err(),
        "a settled job keeps no run control"
    );
}
