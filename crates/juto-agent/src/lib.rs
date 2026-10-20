//! Agent execution: the tool-call loop, `Tool`/`ToolRegistry` interface,
//! host hooks, run control, and the event stream the session layer persists.
//! Behavioral reference: oh-my-pi 579da1d6; see licenses/OMP-MIT.txt.

mod agent;
mod budget;
mod control;
mod error;
mod events;
mod tool;
mod view;

pub use agent::Agent;
pub use budget::{IMAGE_TOKEN_ESTIMATE, estimate_tokens};
pub use control::{BackgroundJob, BlockingWait, RunControl};
pub use error::AgentError;
pub use events::{AgentConfig, AgentEvent, AgentHooks, CompactError, RunOutcome};
pub use tool::{Tool, ToolConcurrency, ToolContext, ToolOutput, ToolRegistry, ToolTier};
pub use view::{
    MAX_CHART_POINTS, MAX_TIMELINE_TICKS, VIEW_DETAILS_TYPE, VIEW_TAGS, ViewChild, ViewNode,
    parse_chart_labels, parse_chart_values, parse_table_columns, validate_view, view_details,
    view_from_details,
};

#[cfg(test)]
mod tests;
