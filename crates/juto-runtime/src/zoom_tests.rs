use crate::session::EntryKind;
use juto_agent::{Agent, AgentConfig, RunControl, ToolRegistry};
use juto_ai::{
    AssistantMessage, ContentBlock, Context, Message, Provider, ProviderError, ProviderEvent,
    ProviderStream, StopReason, StreamOptions, ToolCall, Usage, timestamp_ms,
};
use juto_catalog::{Model, ModelCost};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::*;

fn user(text: &str) -> EntryKind {
    EntryKind::Message {
        message: Message::user(text),
    }
}

fn text_of(output: &ToolOutput) -> String {
    output
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn zoom_call(path: &Path, args: Value) -> ToolOutput {
    read(path, serde_json::from_value(args).unwrap()).unwrap()
}

/// Expansion tokens `zz-NNN` in the order they appear in a page.
fn tokens(text: &str) -> Vec<String> {
    text.match_indices("zz-")
        .map(|(at, _)| text[at..at + 6].to_owned())
        .collect()
}

/// The newest entry's id and its span's message texts, read back from the journal.
fn latest_compaction(session: &Session) -> (String, Vec<String>) {
    let entries = session.active_entries().unwrap();
    let compaction = entries.last().unwrap();
    let EntryKind::Compaction {
        span_start_id: Some(start),
        span_end_id: Some(end),
        ..
    } = &compaction.kind
    else {
        panic!("latest entry is not a compaction with a span");
    };
    let texts = session
        .entries_in_span(start, end)
        .unwrap()
        .iter()
        .map(|entry| match &entry.kind {
            EntryKind::Message { message } => message.text(),
            other => panic!("span returned a non-message entry: {other:?}"),
        })
        .collect();
    (compaction.id.clone(), texts)
}

/// The offset a `more: zoom` hint continues from, if the page has one.
fn more_offset(text: &str) -> Option<usize> {
    let hint = text
        .lines()
        .find_map(|line| line.strip_prefix("more: zoom "))?;
    let args: Value = serde_json::from_str(hint).unwrap();
    Some(usize::try_from(args["offset"].as_u64().unwrap()).unwrap())
}

#[test]
fn zoom_returns_the_span_messages_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..6 {
        session.append(user(&format!("zz-{i:03}"))).unwrap();
    }
    session.compact("summary".into(), None, 2).unwrap();

    let (id, expected) = latest_compaction(&session);
    assert_eq!(expected, ["zz-000", "zz-001", "zz-002", "zz-003"]);

    let out = zoom_call(session.path(), json!({ "id": id }));
    assert!(!out.is_error, "{}", text_of(&out));
    assert_eq!(
        tokens(&text_of(&out)),
        expected
            .iter()
            .map(|t| t[..6].to_owned())
            .collect::<Vec<_>>()
    );
}

#[test]
fn unknown_and_non_compaction_ids_are_tool_errors() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    session.append(user("zz-000")).unwrap();
    session.append(user("zz-001")).unwrap();
    let first = session.active_entries().unwrap()[0].id.clone();
    session.compact("summary".into(), None, 2).unwrap();
    let no_span = session.active_entries().unwrap().last().unwrap().id.clone();

    let unknown = zoom_call(session.path(), json!({ "id": "no-such-id" }));
    assert!(unknown.is_error);
    let message = zoom_call(session.path(), json!({ "id": first }));
    assert!(message.is_error);
    let spanless = zoom_call(session.path(), json!({ "id": no_span }));
    assert!(spanless.is_error);
}

#[test]
fn pages_cover_the_span_once_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..300 {
        session
            .append(user(&format!("zz-{i:03} {}", "x".repeat(400))))
            .unwrap();
    }
    session.compact("summary".into(), None, 0).unwrap();

    let (id, expected) = latest_compaction(&session);
    assert_eq!(expected.len(), 300);
    let expected_tokens: Vec<String> = expected.iter().map(|t| t[..6].to_owned()).collect();

    for limit in [None, Some(40)] {
        let mut seen = Vec::new();
        let mut pages = 0;
        let mut offset = Some(0);
        while let Some(from) = offset {
            let mut args = json!({ "id": id, "offset": from });
            if let Some(limit) = limit {
                args["limit"] = json!(limit);
            }
            let out = zoom_call(session.path(), args);
            let text = text_of(&out);
            assert!(!out.is_error, "{text}");
            assert!(
                text.len() <= OUTPUT_MAX_BYTES,
                "page is {} bytes",
                text.len()
            );
            seen.extend(tokens(&text));
            offset = more_offset(&text);
            pages += 1;
        }
        assert_eq!(seen, expected_tokens);
        assert!(pages > 1);
    }
}

#[test]
fn oversized_message_is_cut_and_points_at_recall() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    session.append(user(&"é".repeat(60_000))).unwrap();
    session.compact("summary".into(), None, 0).unwrap();

    let (id, _) = latest_compaction(&session);
    let out = zoom_call(session.path(), json!({ "id": id }));
    let text = text_of(&out);
    assert!(!out.is_error, "{text}");
    assert!(
        text.len() <= OUTPUT_MAX_BYTES,
        "page is {} bytes",
        text.len()
    );
    assert!(text.contains("more: recall {\"id\": 0"), "{text}");
}

/// One scripted model turn: a `zoom` call with these JSON arguments, or the final answer.
enum Step {
    Call(Value),
    Answer,
}

/// Replays one `Step` per model turn. The agent checks each emitted call
/// against the tool's JSON Schema and then calls `execute`, so the test drives
/// the same JSON path a model call takes.
struct ScriptedProvider {
    steps: Mutex<VecDeque<Step>>,
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl Provider for ScriptedProvider {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let step = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Step::Answer);
        let (content, stop_reason) = match step {
            Step::Call(arguments) => {
                let n = self.calls.fetch_add(1, Ordering::Relaxed);
                (
                    vec![ContentBlock::ToolCall(ToolCall {
                        id: format!("call-{n}"),
                        name: "zoom".into(),
                        arguments,
                        thought_signature: None,
                    })],
                    StopReason::ToolUse,
                )
            }
            Step::Answer => (
                vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                StopReason::Stop,
            ),
        };
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
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

fn zoom_model() -> Model {
    Model {
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
    }
}

/// The text one `zoom` page must return, written out literally: the header,
/// each message's index, id and body, and the `more:` hint when the span continues.
fn expected_page(
    id: &str,
    ids: &[String],
    from: usize,
    through: usize,
    next: Option<usize>,
) -> String {
    let mut out = format!(
        "span of compaction {id}: messages {from}..={through} of {}\n",
        ids.len()
    );
    for (k, entry) in ids.iter().enumerate().skip(from).take(through + 1 - from) {
        out.push_str(&format!("message {k} (user, id {entry})\nzz-00{k}\n"));
    }
    if let Some(next) = next {
        out.push_str(&format!(
            "more: zoom {{\"id\":\"{id}\",\"offset\":{next}}}\n"
        ));
    }
    out
}

/// Drives `zoom` through the agent's JSON tool-call path: a session with a
/// known four-message span, then exact-text checks on each result.
#[tokio::test]
async fn zoom_json_calls_through_the_agent_return_the_exact_span() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..6 {
        session.append(user(&format!("zz-{i:03}"))).unwrap();
    }
    // Six messages with two kept: the span is the first four, zz-000..=zz-003.
    session.compact("summary".into(), None, 2).unwrap();
    let id = session.active_entries().unwrap().last().unwrap().id.clone();
    let all: Vec<String> = session
        .active_entries()
        .unwrap()
        .iter()
        .filter(|entry| matches!(entry.kind, EntryKind::Message { .. }))
        .map(|entry| entry.id.clone())
        .collect();
    assert_eq!(all.len(), 6);
    let ids = all[..4].to_vec();

    let steps = VecDeque::from([
        Step::Call(json!({ "id": id })),
        Step::Call(json!({ "id": "no-such-id" })),
        Step::Call(json!({ "id": id, "offset": 0, "limit": 3 })),
        Step::Call(json!({ "id": id, "offset": 3 })),
        Step::Call(json!({ "id": id, "offset": 0, "limit": 1 })),
        Step::Call(json!({ "id": id, "offset": 1, "limit": 1 })),
        Step::Call(json!({ "id": id, "offset": 2, "limit": 1 })),
        Step::Call(json!({ "id": id, "offset": 3, "limit": 1 })),
        Step::Answer,
    ]);
    let provider = Arc::new(ScriptedProvider {
        steps: Mutex::new(steps),
        calls: AtomicUsize::new(0),
    });
    let mut tools = ToolRegistry::new();
    tools
        .register(Arc::new(ZoomTool::new(session.path().to_path_buf())))
        .unwrap();
    let config = AgentConfig {
        cwd: dir.path().to_path_buf(),
        session_id: "zoom-smoke".into(),
        ..AgentConfig::for_model(zoom_model())
    };
    let mut agent = Agent::new(
        config,
        provider,
        tools,
        Context {
            system_prompt: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
        },
    );
    let (events, _events_rx) = mpsc::unbounded_channel();
    agent
        .run(
            vec![Message::user("zoom the summary")],
            events,
            RunControl::new(),
        )
        .await
        .unwrap();

    let got: Vec<(String, bool, String)> = agent
        .context()
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult {
                tool_name,
                is_error,
                ..
            } => Some((tool_name.clone(), *is_error, message.text())),
            _ => None,
        })
        .collect();
    let zoom = |text: String| ("zoom".to_string(), false, text);
    let want = vec![
        zoom(expected_page(&id, &ids, 0, 3, None)),
        (
            "zoom".to_string(),
            true,
            "no entry no-such-id on the main thread's current branch".to_string(),
        ),
        zoom(expected_page(&id, &ids, 0, 2, Some(3))),
        zoom(expected_page(&id, &ids, 3, 3, None)),
        zoom(expected_page(&id, &ids, 0, 0, Some(1))),
        zoom(expected_page(&id, &ids, 1, 1, Some(2))),
        zoom(expected_page(&id, &ids, 2, 2, Some(3))),
        zoom(expected_page(&id, &ids, 3, 3, None)),
    ];
    assert_eq!(got, want);
}
