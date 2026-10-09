use std::{
    pin::Pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use futures::Stream;
use juto_catalog::Model;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    Image {
        data: String,
        mime_type: String,
    },
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    #[default]
    Stop,
    Length,
    ToolUse,
    Aborted,
    Error,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(default)]
    pub input: u64,
    #[serde(default)]
    pub output: u64,
    #[serde(default)]
    pub cache_read: u64,
    #[serde(default)]
    pub cache_write: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: f64,
}

impl Usage {
    pub fn price(&mut self, model: &Model) {
        self.total_tokens = self.input + self.output + self.cache_read + self.cache_write;
        self.cost = (self.input as f64 * model.cost.input
            + self.output as f64 * model.cost.output
            + self.cache_read as f64 * model.cost.cache_read
            + self.cache_write as f64 * model.cost.cache_write)
            / 1_000_000.0;
    }
    pub fn add(&mut self, other: &Self) {
        self.input += other.input;
        self.output += other.output;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
        self.total_tokens += other.total_tokens;
        self.cost += other.cost;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub stop_reason: StopReason,
    #[serde(default = "timestamp_ms")]
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl AssistantMessage {
    pub fn new(model: &Model) -> Self {
        Self {
            content: Vec::new(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            timestamp: timestamp_ms(),
            response_id: None,
            error: None,
        }
    }
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(
    tag = "role",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum Message {
    User {
        content: Vec<ContentBlock>,
        #[serde(default = "timestamp_ms")]
        timestamp: u64,
    },
    Developer {
        content: Vec<ContentBlock>,
        #[serde(default = "timestamp_ms")]
        timestamp: u64,
    },
    Assistant(AssistantMessage),
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<ContentBlock>,
        is_error: bool,
        #[serde(default)]
        details: Value,
        #[serde(default = "timestamp_ms")]
        timestamp: u64,
    },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: timestamp_ms(),
        }
    }
    pub fn developer(text: impl Into<String>) -> Self {
        Self::Developer {
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: timestamp_ms(),
        }
    }
    pub fn text(&self) -> String {
        let content = match self {
            Self::User { content, .. }
            | Self::Developer { content, .. }
            | Self::ToolResult { content, .. } => content,
            Self::Assistant(message) => &message.content,
        };
        content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(default)]
    pub system_prompt: Vec<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u64>,
    pub thinking: Option<String>,
    pub service_tier: Option<String>,
    pub timeout: Duration,
}
impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            thinking: None,
            service_tier: None,
            timeout: Duration::from_secs(180),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderEvent {
    Start,
    TextDelta {
        index: usize,
        delta: String,
    },
    ThinkingDelta {
        index: usize,
        delta: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallDelta {
        index: usize,
        delta: String,
    },
    BlockEnd {
        index: usize,
    },
    Done {
        message: AssistantMessage,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("provider HTTP {status}: {message}")]
    Http {
        status: u16,
        message: String,
        retry_after: Option<u64>,
    },
    #[error("authentication: {0}")]
    Authentication(String),
    #[error("transport: {0}")]
    Transport(String),
    #[error("provider protocol: {0}")]
    Protocol(String),
    #[error("unsupported provider API: {0}")]
    Unsupported(String),
    #[error("request aborted")]
    Aborted,
}
impl ProviderError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Http {
                status: 429 | 500..=599,
                ..
            } | Self::Transport(_)
        )
    }
}

pub type ProviderStream = Pin<Box<dyn Stream<Item = Result<ProviderEvent, ProviderError>> + Send>>;

#[async_trait]
pub trait Provider: Send + Sync {
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError>;
}
