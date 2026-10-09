//! Tool interface and registry. Behavioral reference: oh-my-pi 579da1d6
//! (`packages/agent` tool execution + `coding-agent` tool defs).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use juto_ai::{ContentBlock, ToolDefinition};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::AgentError;

/// Side-effect classification of a tool; hosts use it for approval policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolTier {
    /// Reads state only (read, glob, grep, ...).
    #[default]
    Read,
    /// Mutates files or durable state.
    Write,
    /// Runs processes or otherwise reaches the environment.
    Exec,
}

/// Scheduling class of a tool within one assistant tool-call batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolConcurrency {
    /// May run concurrently with other Shared calls in the same batch.
    #[default]
    Shared,
    /// Runs alone: waits for every earlier call, blocks every later call.
    Exclusive,
}

/// Per-call execution context handed to [`Tool::execute`].
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Working directory for relative paths and spawned processes.
    pub cwd: PathBuf,
    /// Owning session/run identifier.
    pub session_id: String,
    /// Cancelled when the run is aborted; cooperative tools should stop early.
    pub cancel: CancellationToken,
}

/// Result of one tool execution, converted to a `ToolResult` message by the agent.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: Vec<ContentBlock>,
    /// Structured metadata for the host (diffs, truncation, ...); opaque to the agent.
    pub details: Value,
    /// Marks the result as an error for the model and the journal.
    pub is_error: bool,
}

impl ToolOutput {
    /// Plain-text successful output.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            details: Value::Null,
            is_error: false,
        }
    }

    /// Plain-text error output.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            details: Value::Null,
            is_error: true,
        }
    }
}

/// A runnable tool. Implementations must be honest about tier/concurrency;
/// approvals are applied by the host through `AgentHooks`, not inside tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name, description, and JSON Schema for `arguments`.
    fn definition(&self) -> ToolDefinition;

    /// Side-effect classification. Default: read-only.
    fn tier(&self) -> ToolTier {
        ToolTier::Read
    }

    /// Batch scheduling class. Default: may run concurrently.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Shared
    }
    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String>;
}

pub(crate) struct RegisteredTool {
    tool: Arc<dyn Tool>,
    definition: ToolDefinition,
    validator: jsonschema::Validator,
}

/// Registry of available tools. Cloning shares the same registered set
/// snapshot semantics: clones see the same map and later registrations on the
/// clone do not affect earlier clones (cheap `Arc`-based copy).
#[derive(Clone, Default)]
pub struct ToolRegistry {
    inner: Arc<BTreeMap<String, Arc<RegisteredTool>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool; compiles its parameter schema once. Duplicate names
    /// and uncompilable schemas fail registration.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), AgentError> {
        let definition = tool.definition();
        if definition.name.is_empty() {
            return Err(AgentError::InvalidSchema {
                tool: String::new(),
                message: "tool name must not be empty".to_string(),
            });
        }
        if self.inner.contains_key(&definition.name) {
            return Err(AgentError::DuplicateTool(definition.name));
        }
        let validator = jsonschema::validator_for(&definition.parameters).map_err(|err| {
            AgentError::InvalidSchema {
                tool: definition.name.clone(),
                message: err.to_string(),
            }
        })?;
        let map = Arc::make_mut(&mut self.inner);
        map.insert(
            definition.name.clone(),
            Arc::new(RegisteredTool {
                definition,
                tool,
                validator,
            }),
        );
        Ok(())
    }

    /// Tool definitions in registration-name order, for `Context::tools`.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.inner
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    /// Drop every tool whose name is not in `names`; unknown names are ignored.
    pub fn retain_names(&mut self, names: &[String]) {
        Arc::make_mut(&mut self.inner).retain(|name, _| names.iter().any(|n| n == name));
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&Arc<RegisteredTool>> {
        self.inner.get(name)
    }

    /// Validate `arguments` against the tool's compiled JSON Schema.
    pub(crate) fn validate(entry: &RegisteredTool, arguments: &Value) -> Result<(), String> {
        if let Some(error) = arguments.get("__parseError").and_then(Value::as_str) {
            return Err(error.to_owned());
        }
        match entry.validator.validate(arguments) {
            Ok(()) => Ok(()),
            Err(err) => Err(format!(
                "invalid arguments for `{}`: {err}",
                entry.definition.name
            )),
        }
    }

    pub(crate) fn tool_of(entry: &RegisteredTool) -> &Arc<dyn Tool> {
        &entry.tool
    }
}
