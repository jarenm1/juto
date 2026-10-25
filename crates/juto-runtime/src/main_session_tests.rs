use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use juto_agent::RunControl;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, CredentialStore, Message, Provider, ProviderError,
    ProviderEvent, ProviderStream, StopReason, StreamOptions, ToolCall, Usage, timestamp_ms,
};
use juto_catalog::{Model, ModelCost};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};
use tokio::sync::mpsc;

use super::*;
use crate::{
    RuntimeConfig,
    memory::{COMPACTOR_SYSTEM, NODE},
};

/// Agent turns record what they were given and reply `done: <request>`;
/// compactor calls reply with a short line naming their task.
#[derive(Default)]
struct Script {
    /// Each agent request's messages as text, in order.
    turns: Mutex<Vec<Vec<String>>>,
    /// Each compactor request's task.
    compactions: Mutex<Vec<String>>,
}

impl Script {
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
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let texts: Vec<String> = context.messages.iter().map(Message::text).collect();
        if context
            .system_prompt
            .iter()
            .any(|part| part == COMPACTOR_SYSTEM)
        {
            let task = texts.first().cloned().unwrap_or_default();
            let head = task.lines().next().unwrap_or_default().to_owned();
            self.compactions.lock().push(task);
            return Ok(reply(
                vec![text(&format!("summary — {head}"))],
                StopReason::Stop,
            ));
        }
        let request = context
            .messages
            .iter()
            .rev()
            .find(|message| matches!(message, Message::User { .. }))
            .map(Message::text)
            .unwrap_or_default();
        let after_tool = matches!(context.messages.last(), Some(Message::ToolResult { .. }));
        self.turns.lock().push(texts);
        if !after_tool && let Some(arguments) = tool_request(&request) {
            let (name, arguments) = arguments;
            let call = ContentBlock::ToolCall(ToolCall {
                id: format!("call-{name}"),
                name: name.into(),
                arguments,
                thought_signature: None,
            });
            return Ok(reply(vec![call], StopReason::ToolUse));
        }
        Ok(reply(
            vec![text(&format!("done: {request}"))],
            StopReason::Stop,
        ))
    }
}

const ZOOM_REQUEST: &str = "zoom into the first message";
const SPAWN_REQUEST: &str = "spawn a child";

fn tool_request(request: &str) -> Option<(&'static str, Value)> {
    match request {
        ZOOM_REQUEST => Some(("zoom", json!({"id": 0}))),
        SPAWN_REQUEST => Some(("task", json!({"tasks": [{"task": "child job"}]}))),
        _ => None,
    }
}

fn text(text: &str) -> ContentBlock {
    ContentBlock::Text { text: text.into() }
}

fn reply(content: Vec<ContentBlock>, stop_reason: StopReason) -> ProviderStream {
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
    Box::pin(futures::stream::iter(vec![
        Ok(ProviderEvent::Start),
        Ok(ProviderEvent::Done { message }),
    ]))
}

struct Harness {
    dir: TempDir,
    runtime: Runtime,
    script: Arc<Script>,
}

fn config(context_window: u64, memory_view_bytes: usize) -> RuntimeConfig {
    RuntimeConfig {
        memory_view_bytes,
        model: "mock/test".into(),
        custom_models: vec![Model {
            id: "test".into(),
            name: "Mock Model".into(),
            api: "openai-completions".into(),
            provider: "mock".into(),
            base_url: "http://localhost".into(),
            reasoning: false,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: Some(context_window),
            max_tokens: Some(1024),
            kind: "chat".into(),
            supports_tools: true,
            compat: Value::Null,
        }],
        ..RuntimeConfig::default()
    }
}

fn runtime_at(dir: &Path, config: RuntimeConfig, script: Arc<Script>) -> Runtime {
    let cwd = dir.join("work");
    std::fs::create_dir_all(&cwd).expect("cwd");
    let data_dir = dir.join("data");
    let credentials =
        Arc::new(CredentialStore::open(data_dir.join("credentials.json")).expect("credentials"));
    Runtime::with_provider(&cwd, &data_dir, config, credentials, script, None).expect("runtime")
}

fn harness(context_window: u64) -> Harness {
    harness_with(config(context_window, 128_000))
}

fn harness_with(config: RuntimeConfig) -> Harness {
    let dir = tempdir().expect("tempdir");
    let script = Arc::new(Script::default());
    let runtime = runtime_at(dir.path(), config, script.clone());
    Harness {
        dir,
        runtime,
        script,
    }
}

async fn send(runtime: &Runtime, main: &mut MainSession, prompt: &str) -> RunOutcome {
    let (events, _receiver) = mpsc::unbounded_channel();
    runtime
        .send(main, prompt, events, RunControl::new())
        .await
        .expect("send")
}

#[tokio::test]
async fn every_request_sees_the_whole_conversation_then_the_new_prompt() {
    let Harness {
        dir: _dir,
        runtime,
        script,
    } = harness(32_000);
    let mut main = runtime.new_main_session().expect("main session");
    send(&runtime, &mut main, "Fix the flaky auth test").await;
    send(&runtime, &mut main, "Write release notes for v2").await;
    send(
        &runtime,
        &mut main,
        "what is the earliest message you can see",
    )
    .await;

    // One conversation: the first request is the first line of the view, and
    // the prompt follows the view in full.
    let turn = script.last_turn();
    assert_eq!(turn.len(), 2, "{turn:?}");
    assert_eq!(
        turn[0],
        "<chat>\n0+1|user: Fix the flaky auth test\n1+1|assistant: done: Fix the flaky auth test\n2+1|user: Write release notes for v2\n3+1|assistant: done: Write release notes for v2\n</chat>"
    );
    assert_eq!(turn[1], "what is the earliest message you can see");

    // Short messages are their own lines: no compactor call was needed.
    assert!(script.compactions.lock().is_empty());

    // Everything went to the main journal, in order.
    let texts: Vec<String> = main
        .timeline()
        .expect("timeline")
        .iter()
        .map(Message::text)
        .collect();
    assert_eq!(texts.len(), 6);
    assert_eq!(texts[4], "what is the earliest message you can see");
}

#[tokio::test]
async fn long_messages_are_summarized_and_zoom_reads_them_whole() {
    let Harness {
        dir: _dir,
        runtime,
        script,
    } = harness(32_000);
    let mut main = runtime.new_main_session().expect("main session");
    let long = format!(
        "Here is the stack trace:\n{}",
        "frame at parse_id\n".repeat(80)
    );
    assert!(long.len() > NODE);
    send(&runtime, &mut main, &long).await;
    send(&runtime, &mut main, ZOOM_REQUEST).await;

    // The long request reached the next turn as its compactor line.
    let compactions = script.compactions.lock().clone();
    assert!(
        compactions
            .iter()
            .any(|task| task.starts_with("Compress message 0 ")),
        "{compactions:?}"
    );
    let turns = script.turns.lock().clone();
    let zoom_turn = turns
        .iter()
        .find(|turn| turn.last().is_some_and(|text| text == ZOOM_REQUEST))
        .expect("the zoom request ran");
    assert!(
        zoom_turn[0].contains("0+1|summary — Compress message 0 "),
        "{}",
        zoom_turn[0]
    );
    assert!(!zoom_turn[0].contains("frame at parse_id"));

    // Zooming into line 0+1 returns the original message.
    let after_zoom = script.last_turn();
    let result = after_zoom.last().expect("a tool result");
    assert!(result.starts_with("message 0 (user"), "{result}");
    assert!(result.contains(&"frame at parse_id\n".repeat(80)));
}

#[tokio::test]
async fn a_turn_that_outgrows_the_budget_goes_on_from_the_view() {
    // An 8k-token window: the budget is about 7.4k tokens.
    let Harness {
        dir: _dir,
        runtime,
        script,
    } = harness(8_192);
    let mut main = runtime.new_main_session().expect("main session");
    let huge = format!(
        "Review this log:\n{}",
        "error: retry budget exhausted\n".repeat(900)
    );
    let outcome = send(&runtime, &mut main, &huge).await;
    assert_eq!(outcome.stop_reason, StopReason::Stop);

    // The prompt alone was over budget, so the request went on from the view:
    // the prompt as its compactor line, then the note saying so.
    let turn = script.last_turn();
    assert_eq!(turn.len(), 2, "{turn:?}");
    assert!(
        turn[0].contains("0+1|summary — Compress message 0 "),
        "{}",
        turn[0]
    );
    assert!(
        turn[1].contains("outgrew the context budget"),
        "{}",
        turn[1]
    );
}

#[tokio::test]
async fn a_long_conversation_fades_with_age_within_the_budget() {
    let Harness {
        dir: _dir,
        runtime,
        script,
    } = harness_with(config(32_000, 8_000));
    let mut main = runtime.new_main_session().expect("main session");
    let high = runtime.memory_high(&runtime.resolve_model("mock/test").unwrap().0);
    let mut sent = 0;
    while sent < 40 {
        send(
            &runtime,
            &mut main,
            &format!("request {sent}: {}", "detail ".repeat(60)),
        )
        .await;
        sent += 1;
    }
    let turn = script.last_turn();
    let view = &turn[0];
    let lines: Vec<&str> = view.lines().filter(|line| line.contains('|')).collect();
    // Within budget, give or take each line's `id+n|` prefix.
    assert!(
        view.len() <= high + 16 * (lines.len() + 1),
        "view of {} bytes over {high}",
        view.len()
    );
    let n = |line: &str| -> usize {
        line.split('|')
            .next()
            .unwrap()
            .split('+')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };
    // Old lines cover more messages than recent ones; the newest are leaves.
    assert!(n(lines[0]) > 1, "{view}");
    assert_eq!(n(lines[lines.len() - 1]), 1);
    // The lines still cover every message, oldest first.
    let mut next = 0;
    for line in &lines {
        let (id, count) = line.split('|').next().unwrap().split_once('+').unwrap();
        assert_eq!(id.parse::<usize>().unwrap(), next, "{view}");
        next += count.parse::<usize>().unwrap();
    }
    assert_eq!(next, main.timeline().unwrap().len() - 2);
}

#[tokio::test]
async fn the_view_survives_a_restart() {
    let Harness {
        dir,
        runtime,
        script,
    } = harness(32_000);
    let mut main = runtime.new_main_session().expect("main session");
    let long = format!("Plan:\n{}", "step: migrate the schema\n".repeat(40));
    send(&runtime, &mut main, &long).await;
    send(&runtime, &mut main, "next").await;
    let before = script.last_turn()[0].clone();
    let compactions = script.compactions.lock().len();
    let path = main.path().to_path_buf();
    drop(main);
    drop(runtime);

    // A new process: the summary is read back, not written again.
    let runtime = runtime_at(dir.path(), config(32_000, 128_000), script.clone());
    let mut main = MainSession::open(&path).expect("reopen");
    send(&runtime, &mut main, "and after the restart").await;
    let after = script.last_turn()[0].clone();
    assert!(
        after.starts_with(before.trim_end_matches("\n</chat>")),
        "{before}\n---\n{after}"
    );
    assert_eq!(script.compactions.lock().len(), compactions);
}

#[tokio::test]
async fn legacy_thread_messages_begin_the_conversation() {
    let Harness {
        dir: _dir,
        runtime,
        script,
    } = harness(32_000);
    // A session from when main routed requests into topic threads.
    let mut journal = runtime.new_session().expect("session");
    let thread_dir = journal.path().with_extension("").join("threads");
    let mut thread = Session::create(&thread_dir, runtime.cwd()).expect("thread");
    thread
        .append(EntryKind::Message {
            message: Message::user("What does the scheduler do?"),
        })
        .unwrap();
    let file = thread
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    journal
        .append(EntryKind::Custom {
            custom_type: "thread_created".into(),
            data: json!({"thread": thread.header().id, "file": file, "label": "scheduler"}),
        })
        .unwrap();
    let path = journal.path().to_path_buf();
    drop(journal);

    let mut main = MainSession::open(&path).expect("open");
    assert_eq!(main.legacy_threads().count(), 1);
    send(&runtime, &mut main, "Now make it fair").await;
    assert_eq!(
        script.last_turn(),
        [
            "<chat>\n0+1|user: What does the scheduler do?\n</chat>",
            "Now make it fair"
        ]
    );
    let texts: Vec<String> = main.timeline().unwrap().iter().map(Message::text).collect();
    assert_eq!(
        texts,
        [
            "What does the scheduler do?",
            "Now make it fair",
            "done: Now make it fair"
        ]
    );
}

#[tokio::test]
async fn a_subagent_report_joins_the_conversation() {
    let Harness {
        dir: _dir,
        runtime,
        script,
    } = harness(32_000);
    let mut main = runtime.new_main_session().expect("main session");
    send(&runtime, &mut main, SPAWN_REQUEST).await;
    let manager = runtime.subagent_manager();
    let children = manager.list();
    assert_eq!(children.len(), 1, "one child was queued");
    let _ = manager.wait(&children[0].id).await;

    // A report that missed the spawning run is addressed to the main session
    // and enters its next run.
    send(&runtime, &mut main, "what came back?").await;
    let has_report = |texts: &[String]| texts.iter().any(|text| text.contains("[subagent"));
    assert!(has_report(&script.last_turn()), "{:?}", script.last_turn());
    let timeline: Vec<String> = main.timeline().unwrap().iter().map(Message::text).collect();
    assert!(has_report(&timeline));
}
