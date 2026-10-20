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

    /// The context stays over the window budget even after host compaction,
    /// so the model request was not sent.
    #[error("context is about {estimated} tokens, over the {limit}-token budget after compaction")]
    ContextOverflow { estimated: u64, limit: u64 },

    #[error("context compaction failed: {0}")]
    CompactionFailed(String),

    #[error("spend limit reached ({0:.2} USD) before the model finished")]
    SpendLimit(f64),
}
