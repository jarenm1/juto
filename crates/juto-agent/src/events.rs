//! Event stream, hook seam, run configuration, and run outcome.

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use juto_ai::{Context, Message, ProviderEvent, StopReason, StreamOptions, ToolCall, Usage};
use juto_catalog::Model;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::tool::ToolTier;

/// Immutable snapshot stream consumed by the host (persistence, UI, metrics).
///
/// Wire format is tagged `type` in snake_case. `MessageEnd` fires exactly once
/// per committed context message and is the only event the session layer
/// persists; `ToolExecutionEnd` carries the same `ToolResult` message but does
/// not imply a second `MessageEnd` beyond its own single emission.
#[derive(Debug, Clone, Serialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "snake_case"
)]
pub enum AgentEvent {
    /// Emitted once before the run's prompts are committed.
    AgentStart,
    /// Emitted before each model request (retries stay inside one turn).
    TurnStart { turn: usize },
    /// Emitted lazily before the first streamed content event of a response.
    /// A retry that saw no content emits no `MessageStart`/`MessageEnd` pair.
    MessageStart,
    /// Raw provider event as observed; `Done` is forwarded before `MessageEnd`.
    MessageUpdate { event: ProviderEvent },
    /// The message was committed to `Context`; exactly once per message.
    MessageEnd { message: Message },
    /// A tool call (or its validation) started.
    ToolExecutionStart { call: ToolCall },
    /// A tool call settled; `result` is the committed `ToolResult` message.
    ToolExecutionEnd { call_id: String, result: Message },
    /// A failed pre-content model request will be retried after `delay_ms`.
    Retry {
        attempt: usize,
        delay_ms: u64,
        error: String,
    },
    /// Terminal event: exactly once per `run`, including error paths.
    AgentEnd {
        usage: Usage,
        stop_reason: StopReason,
    },
    /// Non-fatal or fatal error report for the host; fatal ones precede `AgentEnd`.
    Error { error: String },
}

/// Final run summary returned by `Agent::run`.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub usage: Usage,
    pub stop_reason: StopReason,
    pub turns: usize,
}

/// Host hook seam: approvals, context checks, telemetry. Real host integration
/// point (not a plugin loader); every method defaults to allowing the run.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Gate before each model request. `Err` aborts the run with
    /// `AgentError::HookRejected`.
    async fn before_model(&self, _model: &Model, _context: &Context) -> Result<(), String> {
        Ok(())
    }

    /// Gate before each tool execution. `Err` is not fatal: it produces an
    /// `is_error` tool result so the call/result pairing is preserved and the
    /// model sees the rejection.
    async fn before_tool(
        &self,
        _call: &ToolCall,
        _tier: ToolTier,
        _cancel: CancellationToken,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Observe a finished tool execution; also called for validation and
    /// rejection results, never for calls skipped by cancellation.
    async fn after_tool(&self, _call: &ToolCall, _result: &crate::tool::ToolOutput) {}
}

/// Per-run configuration. No default turn cap: the loop runs until the model
/// stops calling tools and no follow-ups are queued.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: Model,
    pub stream_options: StreamOptions,
    pub cwd: PathBuf,
    pub session_id: String,
    /// Optional bound on model turns; `None` = unbounded.
    pub max_turns: Option<usize>,
    /// Retries for a failed request *before any streamed content*. `0`
    /// disables retry; each retry doubles the delay up to 30s, honouring
    /// provider `retry_after` hints when larger.
    pub max_retries: usize,
    pub retry_delay: Duration,
}

impl AgentConfig {
    /// Sensible defaults for a given model: unlimited turns, 3 retries.
    pub fn for_model(model: Model) -> Self {
        Self {
            model,
            stream_options: StreamOptions::default(),
            cwd: PathBuf::new(),
            session_id: String::new(),
            max_turns: None,
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
        }
    }
}
