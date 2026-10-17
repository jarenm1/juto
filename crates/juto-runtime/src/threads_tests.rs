use std::{collections::VecDeque, sync::Arc};

use async_trait::async_trait;
use juto_agent::RunControl;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, CredentialStore, Message, Provider, ProviderError,
    ProviderEvent, ProviderStream, StopReason, StreamOptions, ToolCall, Usage, timestamp_ms,
};
use juto_catalog::{Model, ModelCost};
use parking_lot::Mutex;
use serde_json::Value;
use tempfile::{TempDir, tempdir};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::{Runtime, RuntimeConfig};

/// Answers each call by its system prompt: judge calls and query calls pop
/// scripted replies, label calls echo the latest request, and agent turns
/// record the user texts in their context and reply `done: <request>`.
#[derive(Default)]
struct Script {
    judge: Mutex<VecDeque<String>>,
    queries: Mutex<VecDeque<String>>,
    judge_prompts: Mutex<Vec<String>>,
    turns: Mutex<Vec<Vec<String>>>,
}

impl Script {
    fn judge(&self, reply: &str) {
        self.judge.lock().push_back(reply.to_owned());
    }
    fn query(&self, reply: &str) {
        self.queries.lock().push_back(reply.to_owned());
    }
    /// User texts the most recent agent turn was given as context.
    fn last_turn(&self) -> Vec<String> {
        self.turns
            .lock()
            .last()
            .cloned()
            .expect("an agent turn ran")
    }
}

#[async_trait]
impl Provider for Script {
    async fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let system = context.system_prompt.join("\n");
        let user_texts: Vec<String> = context
            .messages
            .iter()
            .filter(|message| matches!(message, Message::User { .. }))
            .map(Message::text)
            .collect();
        let reply = if system.contains("discrete decision classifier") {
            self.judge_prompts
                .lock()
                .push(user_texts.last().cloned().unwrap_or_default());
            self.judge
                .lock()
                .pop_front()
                .expect("unexpected judge call")
        } else if system == QUERY_SYSTEM {
            self.queries
                .lock()
                .pop_front()
                .expect("unexpected query call")
        } else if system == LABEL_SYSTEM {
            let input = user_texts.last().cloned().unwrap_or_default();
            input
                .split("Latest request:\n")
                .nth(1)
                .and_then(|rest| rest.lines().next())
                .unwrap_or("")
                .to_owned()
        } else {
            let request = user_texts.last().cloned().unwrap_or_default();
            let after_tool = matches!(context.messages.last(), Some(Message::ToolResult { .. }));
            self.turns.lock().push(user_texts);
            if request == SPAWN_REQUEST && !after_tool {
                // Queue one child through the main thread's non-blocking `task`.
                let call = ContentBlock::ToolCall(ToolCall {
                    id: "call-task".into(),
                    name: "task".into(),
                    arguments: serde_json::json!({"tasks": [{"task": "child job"}]}),
                    thought_signature: None,
                });
                return Ok(reply_stream(vec![call], StopReason::ToolUse));
            }
            format!("done: {request}")
        };
        Ok(reply_stream(
            vec![ContentBlock::Text { text: reply }],
            StopReason::Stop,
        ))
    }
}

/// A request the script answers by spawning one subagent.
const SPAWN_REQUEST: &str = "spawn a child";

fn reply_stream(content: Vec<ContentBlock>, stop_reason: StopReason) -> ProviderStream {
    let message = AssistantMessage {
        content,
        api: "mock".into(),
        provider: "mock".into(),
        model: "mock/test".into(),
        usage: Usage::default(),
        stop_reason,
        timestamp: timestamp_ms(),
        response_id: None,
        error: None,
    };
    let events: Vec<Result<ProviderEvent, ProviderError>> = vec![
        Ok(ProviderEvent::Start),
        Ok(ProviderEvent::Done { message }),
    ];
    Box::pin(futures::stream::iter(events))
}

struct Harness {
    _dir: TempDir,
    runtime: Runtime,
    script: Arc<Script>,
}

fn harness(max_active_threads: usize) -> Harness {
    let dir = tempdir().expect("tempdir");
    let cwd = dir.path().join("work");
    std::fs::create_dir_all(&cwd).expect("cwd");
    let data_dir = dir.path().join("data");
    let config = RuntimeConfig {
        model: "mock/test".into(),
        // Bind the judge to the scripted model so routing never depends on
        // TypeSafe or OpenRouter credentials in the environment.
        model_roles: [("judge".to_string(), "mock/test".to_string())].into(),
        max_active_threads,
        custom_models: vec![Model {
            id: "test".into(),
            name: "Mock Model".into(),
            api: "openai-completions".into(),
            provider: "mock".into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: Some(8192),
            max_tokens: Some(1024),
            kind: "chat".into(),
            supports_tools: true,
            compat: Value::Null,
        }],
        ..RuntimeConfig::default()
    };
    let credentials =
        Arc::new(CredentialStore::open(data_dir.join("credentials.json")).expect("credentials"));
    let script = Arc::new(Script::default());
    let runtime =
        Runtime::with_provider(&cwd, &data_dir, config, credentials, script.clone(), None)
            .expect("runtime");
    Harness {
        _dir: dir,
        runtime,
        script,
    }
}

/// Send one request and return its route plus every thread event it emitted.
async fn send(
    runtime: &Runtime,
    main: &mut MainSession,
    prompt: &str,
) -> (Route, Vec<ThreadEvent>, RunOutcome) {
    let (events, mut receiver) = mpsc::unbounded_channel();
    let outcome = runtime
        .send(main, None, prompt, events, RunControl::new())
        .await
        .expect("send");
    let mut thread_events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        if let MainEvent::Thread(event) = event {
            thread_events.push(event);
        }
    }
    let route = thread_events
        .iter()
        .find_map(|event| match event {
            ThreadEvent::ThreadRouted { route } => Some(route.clone()),
            _ => None,
        })
        .expect("a routing event");
    (route, thread_events, outcome)
}

fn decision(key: &str) -> String {
    format!(r#"{{"key":"{key}","confidence":0.9,"rationale":"scripted"}}"#)
}

#[tokio::test]
async fn each_thread_runs_with_only_its_own_history() {
    let Harness {
        _dir,
        runtime,
        script,
    } = harness(6);
    let mut main = runtime.new_main_session().expect("main session");

    // The first request needs no judge: there is nothing to choose from.
    let (auth, _, outcome) = send(&runtime, &mut main, "Fix the flaky auth test").await;
    assert_eq!(auth.kind, RouteKind::New);
    assert_eq!(outcome.stop_reason, StopReason::Stop);

    script.judge(&decision("new"));
    let (notes, _, _) = send(&runtime, &mut main, "Write release notes for v2").await;
    assert_eq!(notes.kind, RouteKind::New);
    assert_ne!(notes.thread, auth.thread);

    // Options list the most recent thread first, so the auth thread is
    // `thread-2` now.
    script.judge(&decision("thread-2"));
    let (follow_up, _, _) = send(&runtime, &mut main, "Also check the retry path").await;
    assert_eq!(follow_up.kind, RouteKind::Continue);
    assert_eq!(follow_up.thread, auth.thread);

    // The model saw the auth thread's history and nothing from the notes thread.
    let context = script.last_turn();
    assert!(context.iter().any(|text| text == "Fix the flaky auth test"));
    assert!(!context.iter().any(|text| text.contains("release notes")));

    // Each completed turn relabels its thread from the latest exchange.
    assert_eq!(
        main.thread(&auth.thread).expect("auth thread").label,
        "Also check the retry path"
    );

    // The merged history interleaves routing records with each thread's
    // messages in the order they happened.
    let rows: Vec<String> = main
        .timeline()
        .expect("timeline")
        .into_iter()
        .map(|item| match item {
            TimelineItem::Routed { route } => format!("route {:?}", route.kind),
            TimelineItem::Message { message, .. } => message.text(),
        })
        .collect();
    assert_eq!(
        rows,
        [
            "route New",
            "Fix the flaky auth test",
            "done: Fix the flaky auth test",
            "route New",
            "Write release notes for v2",
            "done: Write release notes for v2",
            "route Continue",
            "Also check the retry path",
            "done: Also check the retry path",
        ]
    );

    // Threads, labels, and routing times survive reopening the journal.
    let reopened = MainSession::open(main.path()).expect("reopen");
    assert_eq!(reopened.threads(), main.threads());
}

#[tokio::test]
async fn a_dormant_thread_resumes_through_history_lookup() {
    let Harness {
        _dir,
        runtime,
        script,
    } = harness(1);
    let mut main = runtime.new_main_session().expect("main session");

    let (migration, _, _) = send(
        &runtime,
        &mut main,
        "Plan the database migration to postgres",
    )
    .await;

    script.judge(&decision("new"));
    let (onboarding, events, _) = send(&runtime, &mut main, "Design the onboarding screen").await;
    // Only one thread may stay active: the older one goes dormant.
    assert!(events.iter().any(|event| matches!(
        event,
        ThreadEvent::ThreadUpdated { thread }
            if thread.id == migration.thread && thread.status == ThreadStatus::Dormant
    )));

    script.judge(&decision("lookup"));
    script.query("database migration postgres");
    let (resumed, _, _) = send(
        &runtime,
        &mut main,
        "Back to that migration: add a rollback step",
    )
    .await;
    assert_eq!(resumed.kind, RouteKind::Resume);
    assert_eq!(resumed.thread, migration.thread);
    assert_eq!(
        resumed.query.as_deref(),
        Some("database migration postgres")
    );

    // The judge was offered only the active thread, never the dormant one.
    let offered = script
        .judge_prompts
        .lock()
        .last()
        .cloned()
        .expect("judge ran");
    assert!(offered.contains("Design the onboarding screen"));
    assert!(!offered.contains("database migration"));

    // The resumed request runs with the migration thread's history.
    assert!(
        script
            .last_turn()
            .iter()
            .any(|text| text == "Plan the database migration to postgres")
    );
    let status = |id: &str| main.thread(id).expect("thread").status;
    assert_eq!(status(&migration.thread), ThreadStatus::Active);
    assert_eq!(status(&onboarding.thread), ThreadStatus::Dormant);
}

#[tokio::test]
async fn a_failed_router_continues_the_recent_thread_and_says_why() {
    let Harness {
        _dir,
        runtime,
        script,
    } = harness(6);
    let mut main = runtime.new_main_session().expect("main session");
    let (first, _, _) = send(&runtime, &mut main, "Profile the slow build").await;

    script.judge("this is not a decision");
    let (second, _, outcome) = send(&runtime, &mut main, "and the linker step?").await;
    assert_eq!(second.thread, first.thread);
    assert_eq!(second.kind, RouteKind::Continue);
    let note = second.note.expect("degraded routing is noted");
    assert!(note.contains("router judge failed"), "{note}");
    assert_eq!(outcome.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn cancelled_routing_records_nothing() {
    let Harness {
        _dir,
        runtime,
        script,
    } = harness(6);
    let mut main = runtime.new_main_session().expect("main session");
    send(&runtime, &mut main, "Audit the config loader").await;
    let before = main.timeline().expect("timeline").len();

    script.judge(&decision("thread-1"));
    let control = RunControl::new();
    control.cancel();
    let (events, _receiver) = mpsc::unbounded_channel();
    let error = runtime
        .send(&mut main, None, "keep going", events, control)
        .await
        .expect_err("cancelled routing fails");
    assert!(error.to_string().contains("cancelled"), "{error:#}");
    assert_eq!(main.timeline().expect("timeline").len(), before);
    assert!(script.turns.lock().len() == 1, "no agent turn ran");
}

#[tokio::test]
async fn a_pre_thread_session_is_adopted_and_shown_once() {
    let Harness {
        _dir,
        runtime,
        script,
    } = harness(6);
    // A session from before threads: messages live in the journal itself.
    let mut legacy = runtime.new_session().expect("session");
    let (events, _receiver) = mpsc::unbounded_channel();
    runtime
        .run(
            &mut legacy,
            "What does the scheduler do?",
            events,
            RunControl::new(),
        )
        .await
        .expect("legacy run");
    let path = legacy.path().to_path_buf();
    drop(legacy);

    let mut main = MainSession::open(&path).expect("open as main");
    assert!(main.threads().is_empty());

    script.judge(&decision("thread-1"));
    let (route, _, _) = send(&runtime, &mut main, "Now make it fair").await;
    assert_eq!(route.kind, RouteKind::Continue);
    assert_eq!(main.threads().len(), 1);
    assert!(
        script
            .last_turn()
            .iter()
            .any(|text| text == "What does the scheduler do?")
    );

    // The adopted thread copies the old messages; the merged history shows
    // them once.
    let texts: Vec<String> = main
        .timeline()
        .expect("timeline")
        .into_iter()
        .filter_map(|item| match item {
            TimelineItem::Message { message, .. } => Some(message.text()),
            TimelineItem::Routed { .. } => None,
        })
        .collect();
    assert_eq!(
        texts,
        [
            "What does the scheduler do?",
            "done: What does the scheduler do?",
            "Now make it fair",
            "done: Now make it fair",
        ]
    );
}

#[tokio::test]
async fn a_subagent_report_returns_to_the_thread_that_spawned_it() {
    let Harness {
        _dir,
        runtime,
        script,
    } = harness(6);
    let mut main = runtime.new_main_session().expect("main session");
    let (spawner, _, _) = send(&runtime, &mut main, SPAWN_REQUEST).await;

    // Let the child settle: it completes, or is cancelled when the spawning
    // run ends. Either way its report is addressed to the spawning thread.
    let manager = runtime.subagent_manager(&RunControl::new());
    let children = manager.list();
    assert_eq!(children.len(), 1, "one child was queued");
    let _ = manager.wait(&children[0].id).await;

    script.judge(&decision("new"));
    let (other, _, _) = send(&runtime, &mut main, "Unrelated question").await;
    assert_ne!(other.thread, spawner.thread);
    let has_report = |texts: &[String]| texts.iter().any(|text| text.starts_with("[subagent"));
    assert!(
        !has_report(&script.last_turn()),
        "another thread must not receive the report"
    );

    // A report that missed the spawning run waits for that thread, and the
    // host can run it there directly.
    let inbox = manager.main_inbox();
    if let Some((origin, reports)) = inbox.take_next_group() {
        assert_eq!(origin.as_deref(), Some(spawner.thread.as_str()));
        let prompt = reports
            .iter()
            .map(|report| {
                format!(
                    "[subagent {} {:?}]\n{}",
                    report.from, report.kind, report.body
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let (events, _receiver) = mpsc::unbounded_channel();
        runtime
            .send(
                &mut main,
                Some(&spawner.thread),
                prompt,
                events,
                RunControl::new(),
            )
            .await
            .expect("deliver report");
    }
    let spawner_messages = Session::open(&main.thread(&spawner.thread).unwrap().session_path)
        .expect("spawner journal")
        .messages()
        .expect("messages")
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>();
    assert!(
        has_report(&spawner_messages),
        "the spawning thread receives its child's report"
    );
}
