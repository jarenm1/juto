//! Error type for agent registration and run failures.

use juto_ai::ProviderError;

/// Errors surfaced by `ToolRegistry` and `Agent::run`.
///
/// Run-level errors are also reported through `AgentEvent::Error` and a
/// terminal `AgentEvent::AgentEnd` before `run` returns them, so consumers
/// that only watch the event stream see the same failure once.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("duplicate tool name: {0}")]
    DuplicateTool(String),

    #[error("tool `{tool}` has an invalid JSON Schema: {message}")]
    InvalidSchema { tool: String, message: String },

    #[error("host hook rejected the model call: {0}")]
    HookRejected(String),

    #[error("turn limit reached ({0} turns) without the model finishing")]
    TurnLimit(usize),
}
