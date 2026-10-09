//! Public-seam regression tests for the agent lifecycle: pairing, ordering,
//! retry visibility, steering/follow-up delivery, pause, and cancellation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, Message, Provider, ProviderError, ProviderEvent,
    ProviderStream, StopReason, StreamOptions, ToolCall, ToolDefinition,
};
use juto_catalog::{Model, ModelCost};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::{
    Agent, AgentConfig, AgentError, AgentEvent, AgentHooks, RunControl, RunOutcome, Tool,
    ToolConcurrency, ToolContext, ToolOutput, ToolRegistry, ToolTier,
};

fn test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "test".into(),
        api: "test-api".into(),
        provider: "test-provider".into(),
        base_url: String::new(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: Some(128_000),
        max_tokens: Some(4096),
        kind: "chat".into(),
        supports_tools: true,
        compat: Value::Null,
    }
}

fn assistant(model: &Model, content: Vec<ContentBlock>, stop: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Default::default(),
        stop_reason: stop,
        timestamp: 1,
        response_id: None,
        error: None,
    }
}

fn call(id: &str, name: &str, arguments: Value) -> ContentBlock {
    ContentBlock::ToolCall(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
        thought_signature: None,
    })
}

enum Script {
    /// `stream` returns this error before any response is built.
    ConnectError(ProviderError),
    /// Events to yield; may end with `Err` or without `Done`.
    Events(Vec<Result<ProviderEvent, ProviderError>>),
}

/// Deterministic provider: one scripted response per `stream` call, recording
/// how many messages the context held at each request.
struct ScriptedProvider {
    scripts: Mutex<VecDeque<Script>>,
    contexts: Mutex<Vec<usize>>,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Script>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into()),
            contexts: Mutex::new(Vec::new()),
        })
    }

    async fn stream_calls(&self) -> usize {
        self.contexts.lock().await.len()
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        self.contexts.lock().await.push(context.messages.len());
        let script = self
            .scripts
            .lock()
            .await
            .pop_front()
            .unwrap_or(Script::ConnectError(ProviderError::Protocol(
                "scripted provider ran out of responses".into(),
            )));
        match script {
            Script::ConnectError(error) => Err(error),
            Script::Events(events) => Ok(futures::stream::iter(events).boxed()),
        }
    }
}

fn text_done_script(model: &Model, text: &str) -> Script {
    Script::Events(vec![
        Ok(ProviderEvent::Start),
        Ok(ProviderEvent::TextDelta {
            index: 0,
            delta: text.into(),
        }),
        Ok(ProviderEvent::BlockEnd { index: 0 }),
        Ok(ProviderEvent::Done {
            message: assistant(
                model,
                vec![ContentBlock::Text { text: text.into() }],
                StopReason::Stop,
            ),
        }),
    ])
}

/// Records the interleaving of tool starts/ends for concurrency assertions.
struct ToolLog {
    running: AtomicUsize,
    max_concurrent: AtomicUsize,
    order: Mutex<Vec<String>>,
}

impl ToolLog {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            running: AtomicUsize::new(0),
            max_concurrent: AtomicUsize::new(0),
            order: Mutex::new(Vec::new()),
        })
    }
}

struct LoggedTool {
    name: &'static str,
    concurrency: ToolConcurrency,
    log: Arc<ToolLog>,
    /// When set the tool parks on the context cancel token instead of finishing.
    wait_cancel: bool,
    on_start: Mutex<Option<RunControl>>,
    /// Optional side channel used to steer/cancel from inside execution.
    steer_on_start: Mutex<Option<Message>>,
    cancel_on_start: bool,
}

impl LoggedTool {
    fn shared(name: &'static str, log: Arc<ToolLog>) -> Arc<dyn Tool> {
        Arc::new(Self {
            name,
            concurrency: ToolConcurrency::Shared,
            log,
            wait_cancel: false,
            on_start: Mutex::new(None),
            steer_on_start: Mutex::new(None),
            cancel_on_start: false,
        })
    }
}

#[async_trait]
impl Tool for LoggedTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.into(),
            description: self.name.into(),
            parameters: json!({"type": "object"}),
        }
    }

    fn concurrency(&self) -> ToolConcurrency {
        self.concurrency
    }

    async fn execute(&self, _arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let running = self.log.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.log.max_concurrent.fetch_max(running, Ordering::SeqCst);
        self.log.order.lock().await.push(format!("+{}", self.name));
        {
            let control = self.on_start.lock().await.take();
            if let Some(control) = control {
                if let Some(message) = self.steer_on_start.lock().await.take() {
                    control.steer(message);
                }
                if self.cancel_on_start {
                    control.cancel();
                }
            }
        }
        if self.wait_cancel {
            ctx.cancel.cancelled().await;
        } else {
            tokio::task::yield_now().await;
        }
        self.log.order.lock().await.push(format!("-{}", self.name));
        self.log.running.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolOutput::text(format!("{} done", self.name)))
    }
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echoes".into(),
            parameters: json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": false,
            }),
        }
    }

    async fn execute(&self, arguments: Value, _ctx: ToolContext) -> Result<ToolOutput, String> {
        Ok(ToolOutput::text(
            arguments
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ))
    }
}

/// Run one agent to completion, returning the outcome and every event.
/// Same as `run_with_context` without returning the context.
#[allow(dead_code)]
async fn run(
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    config: Option<AgentConfig>,
    control: &RunControl,
    prompts: Vec<Message>,
) -> (Result<RunOutcome, AgentError>, Vec<AgentEvent>) {
    let (outcome, events, _) = run_with_context(provider, tools, config, control, prompts).await;
    (outcome, events)
}

/// Same as `run` but returns the agent's final context too.
async fn run_with_context(
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    config: Option<AgentConfig>,
    control: &RunControl,
    prompts: Vec<Message>,
) -> (Result<RunOutcome, AgentError>, Vec<AgentEvent>, Context) {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        config.unwrap_or_else(|| AgentConfig::for_model(test_model())),
        provider,
        tools,
        Context::default(),
    );
    let outcome = agent.run(prompts, tx, control.clone()).await;
    let context = agent.into_context();
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    (outcome, events, context)
}

fn message_ends(events: &[AgentEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MessageEnd { .. }))
        .count()
}

fn agent_ends(events: &[AgentEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, AgentEvent::AgentEnd { .. }))
        .count()
}

fn tool_results(context: &Context) -> Vec<&Message> {
    context
        .messages
        .iter()
        .filter(|m| matches!(m, Message::ToolResult { .. }))
        .collect()
}

#[tokio::test]
async fn registry_rejects_duplicates_and_bad_schema() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool)).unwrap();
    assert!(matches!(
        registry.register(Arc::new(EchoTool)),
        Err(AgentError::DuplicateTool(name)) if name == "echo"
    ));

    struct BadSchema;
    #[async_trait]
    impl Tool for BadSchema {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "bad".into(),
                description: "bad".into(),
                parameters: json!({"type": "object", "properties": {"x": {"type": "nonsense"}}}),
            }
        }
        async fn execute(&self, _: Value, _: ToolContext) -> Result<ToolOutput, String> {
            unreachable!()
        }
    }
    assert!(matches!(
        registry.register(Arc::new(BadSchema)),
        Err(AgentError::InvalidSchema { .. })
    ));

    let mut retained = ToolRegistry::new();
    retained.register(Arc::new(EchoTool)).unwrap();
    retained
        .register(LoggedTool::shared("other", ToolLog::new()))
        .unwrap();
    retained.retain_names(&["echo".to_string()]);
    assert_eq!(retained.definitions().len(), 1);
    assert_eq!(retained.definitions()[0].name, "echo");
}

#[tokio::test]
async fn simple_stop_commits_messages_once() {
    let model = test_model();
    let provider = ScriptedProvider::new(vec![text_done_script(&model, "hi")]);
    let (outcome, events, context) = run_with_context(
        provider,
        ToolRegistry::new(),
        None,
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;

    let outcome = outcome.unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Stop);
    assert_eq!(outcome.turns, 1);
    assert!(matches!(events.first(), Some(AgentEvent::AgentStart)));
    assert_eq!(agent_ends(&events), 1);
    // prompt + assistant = 2 committed messages, 2 MessageEnd events.
    assert_eq!(context.messages.len(), 2);
    assert_eq!(message_ends(&events), 2);
}

#[tokio::test]
async fn tool_loop_runs_to_second_turn() {
    let model = test_model();
    let provider = ScriptedProvider::new(vec![
        Script::Events(vec![
            Ok(ProviderEvent::Start),
            Ok(ProviderEvent::Done {
                message: assistant(
                    &model,
                    vec![call("c1", "echo", json!({"text": "ping"}))],
                    StopReason::ToolUse,
                ),
            }),
        ]),
        text_done_script(&model, "done"),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool)).unwrap();
    let (outcome, events, context) = run_with_context(
        provider,
        tools,
        None,
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;

    let outcome = outcome.unwrap();
    assert_eq!(outcome.turns, 2);
    assert_eq!(agent_ends(&events), 1);
    // prompt, assistant(tool call), tool result, assistant(text) = 4.
    assert_eq!(context.messages.len(), 4);
    assert_eq!(message_ends(&events), 4);
    let results = tool_results(&context);
    assert_eq!(results.len(), 1);
    match results[0] {
        Message::ToolResult {
            tool_call_id,
            is_error,
            ..
        } => {
            assert_eq!(tool_call_id, "c1");
            assert!(!is_error);
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn shared_tools_overlap_and_exclusive_is_a_barrier() {
    let model = test_model();
    let log = ToolLog::new();
    let mut tools = ToolRegistry::new();
    tools
        .register(LoggedTool::shared("s1", log.clone()))
        .unwrap();
    tools
        .register(LoggedTool::shared("s2", log.clone()))
        .unwrap();
    tools
        .register(Arc::new(LoggedTool {
            name: "x1",
            concurrency: ToolConcurrency::Exclusive,
            log: log.clone(),
            wait_cancel: false,
            on_start: Mutex::new(None),
            steer_on_start: Mutex::new(None),
            cancel_on_start: false,
        }) as Arc<dyn Tool>)
        .unwrap();
    tools
        .register(LoggedTool::shared("s3", log.clone()))
        .unwrap();

    let provider = ScriptedProvider::new(vec![
        Script::Events(vec![Ok(ProviderEvent::Done {
            message: assistant(
                &model,
                vec![
                    call("c1", "s1", json!({})),
                    call("c2", "s2", json!({})),
                    call("c3", "x1", json!({})),
                    call("c4", "s3", json!({})),
                ],
                StopReason::ToolUse,
            ),
        })]),
        text_done_script(&model, "done"),
    ]);
    let (outcome, _events, context) = run_with_context(
        provider,
        tools,
        None,
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;
    outcome.unwrap();

    assert!(
        log.max_concurrent.load(Ordering::SeqCst) >= 2,
        "shared tools did not overlap"
    );
    let order = log.order.lock().await.clone();
    let pos = |name: &str, marker: char| {
        order
            .iter()
            .position(|e| e == &format!("{marker}{name}"))
            .unwrap()
    };
    // Exclusive x1 starts only after both shared tools finished.
    assert!(pos("x1", '+') > pos("s1", '-'));
    assert!(pos("x1", '+') > pos("s2", '-'));
    // s3 starts only after the exclusive tool finished.
    assert!(pos("s3", '+') > pos("x1", '-'));
    // Results are committed in original call order c1..c4.
    let ids: Vec<String> = context
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, ["c1", "c2", "c3", "c4"]);
}

#[tokio::test]
async fn retry_only_before_visible_output() {
    let model = test_model();
    // First request fails with a retryable error before any content; second
    // streams partial text then fails — must NOT retry a third time.
    let provider = ScriptedProvider::new(vec![
        Script::ConnectError(ProviderError::Http {
            status: 500,
            message: "boom".into(),
            retry_after: None,
        }),
        Script::Events(vec![
            Ok(ProviderEvent::Start),
            Ok(ProviderEvent::TextDelta {
                index: 0,
                delta: "partial".into(),
            }),
            Err(ProviderError::Transport("connection dropped".into())),
        ]),
    ]);
    let mut config = AgentConfig::for_model(model);
    config.retry_delay = Duration::ZERO;
    let (outcome, events, _context) = run_with_context(
        provider,
        ToolRegistry::new(),
        Some(config),
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;

    assert!(matches!(
        outcome,
        Err(AgentError::Provider(ProviderError::Transport(_)))
    ));
    let retries = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::Retry { .. }))
        .count();
    assert_eq!(retries, 1, "only the pre-content failure may be retried");
    assert_eq!(agent_ends(&events), 1);
}

#[tokio::test]
async fn retry_exhaustion_returns_error() {
    let provider = ScriptedProvider::new(vec![
        Script::ConnectError(ProviderError::Transport("down".into())),
        Script::ConnectError(ProviderError::Transport("down".into())),
        Script::ConnectError(ProviderError::Transport("down".into())),
    ]);
    let mut config = AgentConfig::for_model(test_model());
    config.retry_delay = Duration::ZERO;
    config.max_retries = 2;
    let provider_ref = provider.clone();
    let (outcome, events, _context) = run_with_context(
        provider,
        ToolRegistry::new(),
        Some(config),
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;
    assert!(matches!(outcome, Err(AgentError::Provider(_))));
    assert_eq!(provider_ref.stream_calls().await, 3);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, AgentEvent::Retry { .. }))
            .count(),
        2
    );
}

#[tokio::test]
async fn cancelled_batch_pairs_every_call() {
    let model = test_model();
    let log = ToolLog::new();
    let mut tools = ToolRegistry::new();
    let control = RunControl::new();
    let cancelling = Arc::new(LoggedTool {
        name: "slow",
        concurrency: ToolConcurrency::Shared,
        log: log.clone(),
        wait_cancel: true,
        on_start: Mutex::new(Some(control.clone())),
        steer_on_start: Mutex::new(None),
        cancel_on_start: true,
    });
    tools.register(cancelling as Arc<dyn Tool>).unwrap();
    tools
        .register(LoggedTool::shared("never", log.clone()))
        .unwrap();

    let provider = ScriptedProvider::new(vec![Script::Events(vec![Ok(ProviderEvent::Done {
        message: assistant(
            &model,
            vec![
                call("c1", "slow", json!({})),
                call("c2", "never", json!({})),
                call("c3", "missing", json!({})),
            ],
            StopReason::ToolUse,
        ),
    })])]);

    let (outcome, _events, context) =
        run_with_context(provider, tools, None, &control, vec![Message::user("go")]).await;
    let outcome = outcome.unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Aborted);
    // Every committed tool call has exactly one tool result.
    let results = tool_results(&context);
    assert_eq!(results.len(), 3);
    let ids: Vec<String> = results
        .iter()
        .map(|m| match m {
            Message::ToolResult { tool_call_id, .. } => tool_call_id.clone(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(ids, ["c1", "c2", "c3"]);
}

#[tokio::test]
async fn steering_injects_before_next_model_call() {
    let model = test_model();
    let mut tools = ToolRegistry::new();
    let control = RunControl::new();
    let steering = Arc::new(LoggedTool {
        name: "steer",
        concurrency: ToolConcurrency::Shared,
        log: ToolLog::new(),
        wait_cancel: false,
        on_start: Mutex::new(Some(control.clone())),
        steer_on_start: Mutex::new(Some(Message::user("mid-run note"))),
        cancel_on_start: false,
    });
    tools.register(steering as Arc<dyn Tool>).unwrap();

    let provider = ScriptedProvider::new(vec![
        Script::Events(vec![Ok(ProviderEvent::Done {
            message: assistant(
                &model,
                vec![call("c1", "steer", json!({}))],
                StopReason::ToolUse,
            ),
        })]),
        text_done_script(&model, "done"),
    ]);
    let provider_ref = provider.clone();
    let (outcome, _events, context) =
        run_with_context(provider, tools, None, &control, vec![Message::user("go")]).await;
    outcome.unwrap();

    // The steering message is committed between the tool result and the next
    // assistant message, and the second request saw it.
    assert!(
        context
            .messages
            .iter()
            .any(|m| matches!(m, Message::User { content, .. }
        if matches!(&content[..], [ContentBlock::Text { text }] if text == "mid-run note")))
    );
    let ctx_sizes = provider_ref.contexts.lock().await.clone();
    assert_eq!(ctx_sizes.len(), 2);
    assert_eq!(ctx_sizes[1], context.messages.len() - 1);
}

#[tokio::test]
async fn follow_up_keeps_run_alive() {
    let model = test_model();
    let control = RunControl::new();
    control.follow_up(Message::user("one more"));
    let provider = ScriptedProvider::new(vec![
        text_done_script(&model, "first"),
        text_done_script(&model, "second"),
    ]);
    let provider_ref = provider.clone();
    let (outcome, _events, context) = run_with_context(
        provider,
        ToolRegistry::new(),
        None,
        &control,
        vec![Message::user("go")],
    )
    .await;
    let outcome = outcome.unwrap();
    assert_eq!(outcome.turns, 2);
    assert_eq!(provider_ref.stream_calls().await, 2);
    // user, assistant, follow-up user, assistant.
    assert_eq!(context.messages.len(), 4);
}

#[tokio::test]
async fn pause_gates_before_model_call() {
    let model = test_model();
    let control = RunControl::new();
    control.pause();
    let provider = ScriptedProvider::new(vec![text_done_script(&model, "hi")]);
    let provider_ref = provider.clone();
    let control_ref = control.clone();
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        AgentConfig::for_model(test_model()),
        provider,
        ToolRegistry::new(),
        Context::default(),
    );
    let handle = tokio::spawn(async move {
        agent
            .run(vec![Message::user("go")], tx, control_ref.clone())
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        provider_ref.stream_calls().await,
        0,
        "paused run reached the model"
    );
    control.resume();
    let outcome = handle.await.unwrap().unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Stop);
    assert_eq!(provider_ref.stream_calls().await, 1);
}

#[tokio::test]
async fn before_tool_rejection_pairs_error_result() {
    struct RejectAll;
    #[async_trait]
    impl AgentHooks for RejectAll {
        async fn before_tool(
            &self,
            _call: &ToolCall,
            _tier: ToolTier,
            _cancel: CancellationToken,
        ) -> Result<(), String> {
            Err("denied".into())
        }
    }

    let model = test_model();
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool)).unwrap();
    let provider = ScriptedProvider::new(vec![
        Script::Events(vec![Ok(ProviderEvent::Done {
            message: assistant(
                &model,
                vec![call("c1", "echo", json!({"text": "x"}))],
                StopReason::ToolUse,
            ),
        })]),
        text_done_script(&model, "done"),
    ]);
    let (tx, _rx) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        AgentConfig::for_model(model),
        provider,
        tools,
        Context::default(),
    );
    agent.set_hooks(Arc::new(RejectAll));
    let outcome = agent
        .run(vec![Message::user("go")], tx, RunControl::new())
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Stop);
    let results = tool_results(agent.context());
    assert_eq!(results.len(), 1);
    match results[0] {
        Message::ToolResult {
            is_error, content, ..
        } => {
            assert!(is_error);
            assert!(
                content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Text { text } if text.contains("denied")))
            );
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn schema_violation_and_unknown_tool_pair_error_results() {
    let model = test_model();
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool)).unwrap();
    let provider = ScriptedProvider::new(vec![
        Script::Events(vec![Ok(ProviderEvent::Done {
            message: assistant(
                &model,
                vec![
                    call("c1", "echo", json!({"wrong": 1})),
                    call("c2", "ghost", json!({})),
                ],
                StopReason::ToolUse,
            ),
        })]),
        text_done_script(&model, "done"),
    ]);
    let (outcome, _events, context) = run_with_context(
        provider,
        tools,
        None,
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;
    outcome.unwrap();
    let results = tool_results(&context);
    assert_eq!(results.len(), 2);
    for result in results {
        match result {
            Message::ToolResult { is_error, .. } => assert!(is_error),
            _ => unreachable!(),
        }
    }
}

#[tokio::test]
async fn turn_limit_errors_after_exactly_max_turns() {
    let model = test_model();
    let provider = ScriptedProvider::new(vec![
        Script::Events(vec![Ok(ProviderEvent::Done {
            message: assistant(
                &model,
                vec![call("c1", "echo", json!({"text": "x"}))],
                StopReason::ToolUse,
            ),
        })]),
        Script::Events(vec![Ok(ProviderEvent::Done {
            message: assistant(
                &model,
                vec![call("c2", "echo", json!({"text": "x"}))],
                StopReason::ToolUse,
            ),
        })]),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(EchoTool)).unwrap();
    let mut config = AgentConfig::for_model(model);
    config.max_turns = Some(2);
    let (outcome, events, _context) = run_with_context(
        provider,
        tools,
        Some(config),
        &RunControl::new(),
        vec![Message::user("go")],
    )
    .await;
    assert!(matches!(outcome, Err(AgentError::TurnLimit(2))));
    assert_eq!(agent_ends(&events), 1);
}

#[tokio::test]
async fn cancel_before_start_aborts_cleanly() {
    let control = RunControl::new();
    control.cancel();
    let model = test_model();
    let provider = ScriptedProvider::new(vec![text_done_script(&model, "never")]);
    let (outcome, events, context) = run_with_context(
        provider,
        ToolRegistry::new(),
        None,
        &control,
        vec![Message::user("go")],
    )
    .await;
    let outcome = outcome.unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Aborted);
    assert_eq!(agent_ends(&events), 1);
    // The prompt still commits before the loop notices cancellation.
    assert_eq!(context.messages.len(), 1);
    assert_eq!(message_ends(&events), 1);
}
