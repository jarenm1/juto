//! Context-size estimation shared by the agent's window guard and the runtime's
//! compaction trigger. The estimate is a byte-based heuristic, not a tokenizer.

use juto_ai::{ContentBlock, Message, ToolDefinition};
use serde_json::Value;

/// Token estimate for one image in context accounting.
pub const IMAGE_TOKEN_ESTIMATE: u64 = 4096;

/// Estimated tokens for `messages`, counting per-message overhead, text and
/// thinking bytes at roughly three bytes per token, tool-call arguments, and a
/// fixed cost per image.
pub fn estimate_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(|message| {
            let content = match message {
                Message::User { content, .. }
                | Message::Developer { content, .. }
                | Message::ToolResult { content, .. } => content,
                Message::Assistant(message) => &message.content,
            };
            8 + content
                .iter()
                .map(|block| match block {
                    ContentBlock::Text { text } => (text.len() as u64).div_ceil(3),
                    ContentBlock::Thinking { thinking, .. } => (thinking.len() as u64).div_ceil(3),
                    ContentBlock::Image { .. } => IMAGE_TOKEN_ESTIMATE,
                    ContentBlock::ToolCall(call) => {
                        (json_weight(&call.arguments) + call.name.len() as u64).div_ceil(3)
                    }
                    ContentBlock::RedactedThinking { data } => (data.len() as u64).div_ceil(3),
                })
                .sum::<u64>()
        })
        .sum()
}

/// Estimated tokens for one provider request: the system prompt and the tool
/// definitions it advertises count alongside `messages`, using the same
/// bytes-per-token heuristic as `estimate_tokens`.
pub fn estimate_request_tokens(
    system_prompt: &[String],
    tools: &[ToolDefinition],
    messages: &[Message],
) -> u64 {
    let system: u64 = system_prompt
        .iter()
        .map(|text| (text.len() as u64).div_ceil(3))
        .sum();
    let schemas: u64 = tools
        .iter()
        .map(|tool| {
            (tool.name.len() as u64 + tool.description.len() as u64 + json_weight(&tool.parameters))
                .div_ceil(3)
        })
        .sum();
    system + schemas + estimate_tokens(messages)
}

fn json_weight(value: &Value) -> u64 {
    match value {
        Value::Null | Value::Bool(_) => 5,
        Value::Number(_) => 24,
        Value::String(value) => value.len() as u64 + 2,
        Value::Array(items) => items.iter().map(json_weight).sum::<u64>() + items.len() as u64 + 2,
        Value::Object(items) => {
            items
                .iter()
                .map(|(key, value)| key.len() as u64 + json_weight(value) + 4)
                .sum::<u64>()
                + 2
        }
    }
}
