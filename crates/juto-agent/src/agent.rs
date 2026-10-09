//! Agent execution loop. Behavioral reference: oh-my-pi 579da1d6
//! `packages/agent/src/agent-loop.ts` — scoped to the frozen contract:
//! steering/follow-up boundaries, pause gate, shared/exclusive tool
//! scheduling, cancellation pairing, and pre-content retries.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, Message, Provider, ProviderError, ProviderEvent,
    StopReason, ToolCall, Usage, timestamp_ms,
};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::control::{self, RunControl};
use crate::error::AgentError;
use crate::events::{AgentConfig, AgentEvent, AgentHooks, RunOutcome};
use crate::tool::{ToolContext, ToolOutput, ToolRegistry};

/// Longest backoff a retry sleep may reach; provider `retry_after` hints may
/// still exceed it.
const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

/// Result of a single provider request inside one turn.
enum AttemptOutcome {
    /// The stream delivered `Done`; the message is ready to commit.
    Done(Box<AssistantMessage>),
    /// The run's cancellation token fired (connect, read, or provider abort).
    Aborted,
    /// The request failed. `visible` is true once any content event was
    /// forwarded — after that the run must not retry, so consumers never see
    /// a re-streamed response after partial output.
    Failed(ProviderError, bool),
}

/// One planned call in a tool batch.
enum Planned {
    /// Fails before reaching the tool (unknown name, schema violation).
    Immediate(ToolOutput),
    /// Schema-valid; goes through the hook gate then `Tool::execute`.
    Executable(Arc<dyn crate::tool::Tool>),
}

/// Why a batch member never reached its tool.
enum Skipped {
    Cancelled,
    AssistantStopped(StopReason),
}

/// Agent: one model, one tool set, one context. `run` drives the full
/// tool-call loop; hosts interact only through `AgentEvent`s and `RunControl`.
pub struct Agent {
    config: AgentConfig,
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    context: Context,
    hooks: Option<Arc<dyn AgentHooks>>,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        mut context: Context,
    ) -> Self {
        // The registry is authoritative for what this agent can run; keep the
        // declared context in sync so providers never see phantom tools.
        context.tools = tools.definitions();
        Self {
            config,
            provider,
            tools,
            context,
            hooks: None,
        }
    }

    /// Attach the host hook seam (approvals, telemetry).
    pub fn set_hooks(&mut self, hooks: Arc<dyn AgentHooks>) {
        self.hooks = Some(hooks);
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    pub fn into_context(self) -> Context {
        self.context
    }

    /// Run the loop: commit `prompts`, then iterate model call → tool batch
    /// until the model produces no tool calls and no follow-up is queued.
    ///
    /// Emits `AgentStart` first and `AgentEnd` exactly once, including on
    /// returned `Err`s; `MessageEnd` fires exactly once per message appended
    /// to `self.context.messages`.
    pub async fn run(
        &mut self,
        prompts: Vec<Message>,
        events: mpsc::UnboundedSender<AgentEvent>,
        control: RunControl,
    ) -> Result<RunOutcome, AgentError> {
        let token = control.token();
        let (mut steering, mut follow_ups) = control.take_queues().await;
        let mut usage = Usage::default();
        let mut turns = 0usize;

        emit(&events, AgentEvent::AgentStart);
        for message in prompts {
            self.commit(message, &events);
        }

        let outcome: Result<RunOutcome, AgentError> = loop {
            // Pause gate parks the loop at boundaries only; cancellation
            // releases a parked run immediately.
            control.wait_if_paused().await;
            if token.is_cancelled() || events.is_closed() {
                break Ok(self.finish(StopReason::Aborted, usage, turns, &events));
            }

            // Steering injects at this boundary and forces another turn.
            for message in control::drain(&mut steering) {
                self.commit(message, &events);
            }

            turns += 1;
            if let Some(max) = self.config.max_turns {
                if turns > max {
                    let error = AgentError::TurnLimit(max);
                    emit(
                        &events,
                        AgentEvent::Error {
                            error: error.to_string(),
                        },
                    );
                    self.finish(StopReason::Error, usage, turns, &events);
                    break Err(error);
                }
            }
            emit(&events, AgentEvent::TurnStart { turn: turns });

            if let Some(hooks) = &self.hooks {
                if let Err(reason) = hooks.before_model(&self.config.model, &self.context).await {
                    let error = AgentError::HookRejected(reason);
                    emit(
                        &events,
                        AgentEvent::Error {
                            error: error.to_string(),
                        },
                    );
                    self.finish(StopReason::Error, usage, turns, &events);
                    break Err(error);
                }
            }

            let assistant = match self.stream_turn(&events, &control, &token).await {
                TurnStream::Done(message) => message,
                TurnStream::Aborted => {
                    break Ok(self.finish(StopReason::Aborted, usage, turns, &events));
                }
                TurnStream::Failed(error) => {
                    emit(
                        &events,
                        AgentEvent::Error {
                            error: error.to_string(),
                        },
                    );
                    self.finish(StopReason::Error, usage, turns, &events);
                    break Err(AgentError::Provider(error));
                }
            };

            usage.add(&assistant.usage);
            let calls: Vec<ToolCall> = assistant.tool_calls().cloned().collect();
            let assistant_stop = assistant.stop_reason;
            let assistant_error = assistant.error.clone();
            self.commit(Message::Assistant(*assistant), &events);

            match assistant_stop {
                StopReason::Aborted => {
                    self.commit_skipped(&calls, Skipped::Cancelled, &events);
                    break Ok(self.finish(StopReason::Aborted, usage, turns, &events));
                }
                StopReason::Length | StopReason::Error => {
                    self.commit_skipped(&calls, Skipped::AssistantStopped(assistant_stop), &events);
                    if assistant_stop == StopReason::Error {
                        let detail = assistant_error
                            .unwrap_or_else(|| "assistant message reported an error".to_string());
                        emit(
                            &events,
                            AgentEvent::Error {
                                error: detail.clone(),
                            },
                        );
                        self.finish(StopReason::Error, usage, turns, &events);
                        break Err(AgentError::Provider(ProviderError::Protocol(detail)));
                    }
                    break Ok(self.finish(StopReason::Length, usage, turns, &events));
                }
                _ => {}
            }

            if calls.is_empty() {
                // Stop boundary: steering still forces another turn; only
                // then do follow-ups keep the run alive.
                let mut pending = control::drain(&mut steering);
                pending.extend(control::drain(&mut follow_ups));
                if pending.is_empty() {
                    let reason = if assistant_stop == StopReason::ToolUse {
                        StopReason::Stop
                    } else {
                        assistant_stop
                    };
                    break Ok(self.finish(reason, usage, turns, &events));
                }
                for message in pending {
                    self.commit(message, &events);
                }
                continue;
            }

            self.execute_batch(&calls, &events, &control, &token).await;
            if token.is_cancelled() || events.is_closed() {
                break Ok(self.finish(StopReason::Aborted, usage, turns, &events));
            }
        };

        outcome
    }

    /// Emit `AgentEnd` and build the outcome. Called on every exit path so the
    /// terminal event fires exactly once.
    fn finish(
        &self,
        stop_reason: StopReason,
        usage: Usage,
        turns: usize,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> RunOutcome {
        emit(
            events,
            AgentEvent::AgentEnd {
                usage: usage.clone(),
                stop_reason,
            },
        );
        RunOutcome {
            usage,
            stop_reason,
            turns,
        }
    }

    /// One turn's worth of provider requests: connect, stream, and retry
    /// failures that produced no visible output.
    async fn stream_turn(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        control: &RunControl,
        token: &CancellationToken,
    ) -> TurnStream {
        let mut retries = 0usize;
        loop {
            control.wait_if_paused().await;
            if token.is_cancelled() {
                return TurnStream::Aborted;
            }
            match self.stream_attempt(events, token).await {
                AttemptOutcome::Done(message) => return TurnStream::Done(message),
                AttemptOutcome::Aborted => return TurnStream::Aborted,
                AttemptOutcome::Failed(error, visible) => {
                    if visible || !error.is_retryable() || retries >= self.config.max_retries {
                        return TurnStream::Failed(error);
                    }
                    retries += 1;
                    let mut delay = self
                        .config
                        .retry_delay
                        .saturating_mul(1u32 << retries.saturating_sub(1).min(6));
                    if delay > MAX_RETRY_DELAY {
                        delay = MAX_RETRY_DELAY;
                    }
                    if let ProviderError::Http {
                        retry_after: Some(secs),
                        ..
                    } = &error
                    {
                        let hinted = Duration::from_secs(*secs);
                        if hinted > delay {
                            delay = hinted;
                        }
                    }
                    emit(
                        events,
                        AgentEvent::Retry {
                            attempt: retries,
                            delay_ms: delay.as_millis() as u64,
                            error: error.to_string(),
                        },
                    );
                    tokio::select! {
                        biased;
                        () = token.cancelled() => return TurnStream::Aborted,
                        () = tokio::time::sleep(delay) => {}
                    }
                }
            }
        }
    }

    /// Connect and consume one provider stream, forwarding every event.
    async fn stream_attempt(
        &self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        token: &CancellationToken,
    ) -> AttemptOutcome {
        let mut stream = match tokio::select! {
            biased;
            () = token.cancelled() => return AttemptOutcome::Aborted,
            result = self.provider.stream(
                &self.config.model,
                &self.context,
                &self.config.stream_options,
                token.clone(),
            ) => result,
        } {
            Ok(stream) => stream,
            Err(ProviderError::Aborted) => return AttemptOutcome::Aborted,
            Err(error) => return AttemptOutcome::Failed(error, false),
        };

        // `visible` flips on the first content event (anything past `start`);
        // `message_started` gates the lazy `MessageStart` emission.
        let mut visible = false;
        let mut message_started = false;
        loop {
            let item = tokio::select! {
                biased;
                () = token.cancelled() => return AttemptOutcome::Aborted,
                item = stream.next() => item,
            };
            match item {
                None => {
                    return AttemptOutcome::Failed(
                        ProviderError::Protocol(
                            "provider stream ended without a `done` event".into(),
                        ),
                        visible,
                    );
                }
                Some(Err(ProviderError::Aborted)) => return AttemptOutcome::Aborted,
                Some(Err(error)) => return AttemptOutcome::Failed(error, visible),
                Some(Ok(event)) => {
                    if !matches!(event, ProviderEvent::Start) {
                        visible = true;
                        if !message_started {
                            message_started = true;
                            emit(events, AgentEvent::MessageStart);
                        }
                    }
                    match event {
                        ProviderEvent::Done { message } => {
                            emit(
                                events,
                                AgentEvent::MessageUpdate {
                                    event: ProviderEvent::Done {
                                        message: message.clone(),
                                    },
                                },
                            );
                            return AttemptOutcome::Done(Box::new(message));
                        }
                        event => emit(events, AgentEvent::MessageUpdate { event }),
                    }
                }
            }
        }
    }

    /// Append a message to the context and emit its single `MessageEnd`.
    fn commit(&mut self, message: Message, events: &mpsc::UnboundedSender<AgentEvent>) {
        self.context.messages.push(message.clone());
        emit(events, AgentEvent::MessageEnd { message });
    }

    /// Commit error tool results for calls the run could not execute so
    /// call/result pairing survives aborts and truncated output.
    fn commit_skipped(
        &mut self,
        calls: &[ToolCall],
        skipped: Skipped,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        for call in calls {
            emit(
                events,
                AgentEvent::ToolExecutionStart { call: call.clone() },
            );
            let text = match &skipped {
                Skipped::Cancelled => "Tool execution cancelled.".to_string(),
                Skipped::AssistantStopped(reason) => format!(
                    "Skipped: the assistant stopped with stop_reason `{reason:?}` before this tool call could run."
                ),
            };
            let output = ToolOutput::error(text);
            let result = tool_result_message(call, &output);
            emit(
                events,
                AgentEvent::ToolExecutionEnd {
                    call_id: call.id.clone(),
                    result: result.clone(),
                },
            );
            self.commit(result, events);
        }
    }

    /// Run one assistant-message tool batch: shared calls concurrent,
    /// exclusive calls serialized between them, results committed in call
    /// order. A cancelled batch still commits a result for every call.
    async fn execute_batch(
        &mut self,
        calls: &[ToolCall],
        events: &mpsc::UnboundedSender<AgentEvent>,
        control: &RunControl,
        token: &CancellationToken,
    ) {
        // Plan: resolve + schema-validate each call up front so unknown or
        // malformed calls produce results without touching scheduling.
        let mut planned = Vec::with_capacity(calls.len());
        for call in calls {
            planned.push(match self.tools.get(&call.name) {
                None => {
                    Planned::Immediate(ToolOutput::error(format!("Unknown tool `{}`.", call.name)))
                }
                Some(entry) => match ToolRegistry::validate(entry, &call.arguments) {
                    Err(message) => Planned::Immediate(ToolOutput::error(message)),
                    Ok(()) => Planned::Executable(Arc::clone(ToolRegistry::tool_of(entry))),
                },
            });
        }

        // Segments in call order: consecutive shared calls run together;
        // exclusive calls and pre-failed calls form singleton barriers.
        let mut index = 0usize;
        while index < calls.len() {
            let exclusive_here = matches!(&planned[index], Planned::Executable(tool)
                if tool.concurrency() == crate::tool::ToolConcurrency::Exclusive);
            let end = if exclusive_here {
                index + 1
            } else {
                let mut end = index;
                while end < calls.len()
                    && !matches!(&planned[end], Planned::Executable(tool)
                        if tool.concurrency() == crate::tool::ToolConcurrency::Exclusive)
                {
                    end += 1;
                }
                end
            };

            control.wait_if_paused().await;
            let cancelled = token.is_cancelled();

            // Dispatch this segment (or mark it cancelled). Every call that
            // runs gets a `ToolExecutionStart` here, in call order.
            let segment: Vec<(usize, ToolOutput)> = if cancelled {
                Vec::new()
            } else {
                let mut futures = FuturesUnordered::new();
                for i in index..end {
                    emit(
                        events,
                        AgentEvent::ToolExecutionStart {
                            call: calls[i].clone(),
                        },
                    );
                    if let Planned::Executable(tool) = &planned[i] {
                        let call = calls[i].clone();
                        let tool = Arc::clone(tool);
                        let hooks = self.hooks.clone();
                        let ctx = ToolContext {
                            cwd: self.config.cwd.clone(),
                            session_id: self.config.session_id.clone(),
                            cancel: token.clone(),
                        };
                        let cancel = token.clone();
                        futures.push(async move {
                            let output = execute_one(&call, tool, hooks, ctx, cancel).await;
                            (i, output)
                        });
                    }
                }
                let mut results: Vec<(usize, ToolOutput)> = Vec::new();
                loop {
                    tokio::select! {
                        biased;
                        () = token.cancelled() => break,
                        next = futures.next() => match next {
                            Some(done) => results.push(done),
                            None => break,
                        },
                    }
                }
                results
            };

            // Emit ends and commit results strictly in call order. Members of
            // a cancelled segment (and every call after it) still get a
            // paired error result; results already completed before the
            // cancel are kept.
            let cancelled_now = cancelled || token.is_cancelled();
            for i in index..end {
                if cancelled {
                    emit(
                        events,
                        AgentEvent::ToolExecutionStart {
                            call: calls[i].clone(),
                        },
                    );
                }
                let output = match &planned[i] {
                    Planned::Immediate(output) if !cancelled_now => output.clone(),
                    Planned::Executable(_) => segment
                        .iter()
                        .find(|(done, _)| *done == i)
                        .map(|(_, output)| output.clone())
                        .unwrap_or_else(|| ToolOutput::error("Tool execution cancelled.")),
                    _ => ToolOutput::error("Tool execution cancelled."),
                };
                if !cancelled_now {
                    if let (Planned::Immediate(_), Some(hooks)) = (&planned[i], &self.hooks) {
                        hooks.after_tool(&calls[i], &output).await;
                    }
                }
                let result = tool_result_message(&calls[i], &output);
                emit(
                    events,
                    AgentEvent::ToolExecutionEnd {
                        call_id: calls[i].id.clone(),
                        result: result.clone(),
                    },
                );
                self.commit(result, events);
            }

            index = end;
        }
    }
}

/// Outcome of `stream_turn`.
enum TurnStream {
    Done(Box<AssistantMessage>),
    Aborted,
    Failed(ProviderError),
}

/// Hook gate + tool execute + after-hook for one call.
async fn execute_one(
    call: &ToolCall,
    tool: Arc<dyn crate::tool::Tool>,
    hooks: Option<Arc<dyn AgentHooks>>,
    ctx: ToolContext,
    cancel: CancellationToken,
) -> ToolOutput {
    if let Some(hooks) = &hooks {
        match hooks.before_tool(call, tool.tier(), cancel.clone()).await {
            Err(reason) => {
                let output = ToolOutput::error(format!("Tool call rejected: {reason}"));
                hooks.after_tool(call, &output).await;
                return output;
            }
            Ok(()) => {}
        }
    }
    let mut output = match tool.execute(call.arguments.clone(), ctx).await {
        Ok(output) => output,
        Err(error) => ToolOutput::error(error),
    };
    if output.content.is_empty() {
        output.content.push(ContentBlock::Text {
            text: String::new(),
        });
    }
    if let Some(hooks) = &hooks {
        hooks.after_tool(call, &output).await;
    }
    output
}

/// Build the `ToolResult` message committed for one call.
fn tool_result_message(call: &ToolCall, output: &ToolOutput) -> Message {
    Message::ToolResult {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: output.content.clone(),
        is_error: output.is_error,
        details: output.details.clone(),
        timestamp: timestamp_ms(),
    }
}

/// Best-effort event emission; a closed channel is observed via
/// `events.is_closed()` at boundaries rather than aborting mid-operation.
fn emit(events: &mpsc::UnboundedSender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
}
