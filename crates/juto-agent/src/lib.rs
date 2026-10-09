//! Agent execution: the tool-call loop, `Tool`/`ToolRegistry` interface,
//! host hooks, run control, and the event stream the session layer persists.
//! Behavioral reference: oh-my-pi 579da1d6; see licenses/OMP-MIT.txt.

mod agent;
mod control;
mod error;
mod events;
mod tool;

pub use agent::Agent;
pub use control::RunControl;
pub use error::AgentError;
pub use events::{AgentConfig, AgentEvent, AgentHooks, RunOutcome};
pub use tool::{Tool, ToolConcurrency, ToolContext, ToolOutput, ToolRegistry, ToolTier};

#[cfg(test)]
mod tests;
