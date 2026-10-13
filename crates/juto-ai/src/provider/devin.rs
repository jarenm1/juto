//! Devin (`devin-agent`) transport: Codeium Cascade over the Connect protocol.
//!
//! Ported from OMP `packages/ai/src/providers/devin.ts` @ 579da1d6 (see
//! `licenses/OMP-MIT.txt`). Three unary/streaming RPCs on `model.base_url`
//! (default `https://server.codeium.com`):
//!
//! 1. `AuthService/GetUserJwt` — exchange the session token for a user JWT and
//!    an optional per-user API server URL.
//! 2. `ApiServerService/AssignModel` — router models only (`compat.modelRouter`)
//!    resolve to a concrete model uid plus an assignment JWT.
//! 3. `ApiServerService/GetChatMessage` — Connect streaming chat: one gzipped
//!    request frame, then framed protobuf `GetChatMessageResponse` deltas with
//!    a JSON end-of-stream trailer.
//!
//! Requests are hand-encoded proto3 (see `proto.rs`); only the fields the
//! transport consumes are modeled, matching upstream's pruned generated
//! schemas. Devin's own protobuf vendoring (~100 `.proto` files) is not
//! needed: Connect over HTTP/1.1 carries plain protobuf payloads.

use std::collections::HashMap;

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use futures::StreamExt;
use juto_catalog::Model;
use serde_json::Value;
use sha2::Digest;
use tokio_util::sync::CancellationToken;

use super::connect::{
    CONNECT_COMPRESSED_FLAG, CONNECT_END_STREAM_FLAG, ConnectFrameDecoder, frame_connect_message,
};
use super::proto::{ProtoMessage, ProtoWriter};
use super::{ProviderError, transport_error};
use crate::{ContentBlock, Context, Credential, ProviderEvent, StopReason, ToolCall};

const DEVIN_DEFAULT_BASE_URL: &str = "https://server.codeium.com";
const CHAT_MESSAGE_PATH: &str = "/exa.api_server_pb.ApiServerService/GetChatMessage";
const ASSIGN_MODEL_PATH: &str = "/exa.api_server_pb.ApiServerService/AssignModel";
const AUTH_PATH: &str = "/exa.auth_pb.AuthService/GetUserJwt";

const DEVIN_DEFAULT_STOP_PATTERNS: &[&str] = &[
    "<|user|>",
    "<|bot|>",
    "<|context_request|>",
    "<|endoftext|>",
    "<|end_of_turn|>",
];

/// Hard upper bound on a single Connect frame payload (upstream limit).
const MAX_CONNECT_FRAME_PAYLOAD: usize = 16 * 1024 * 1024;
/// An `invalid_argument` trailer with history this large is likely context
/// overflow rather than an intermittent error (upstream heuristic).
const LARGE_HISTORY_RECOVERY_BYTES: usize = 512 * 1024;
const MAX_DEVIN_ERROR_DETAIL_CHARS: usize = 4096;

/// Session token as the wire format carries it: the scheme prefix is required.
const DEVIN_SESSION_TOKEN_PREFIX: &str = "devin-session-token$";

/// Released Devin CLI request identity. The backend gates behavior on this
/// tuple: `ide_type: "chisel"` unlocks router assignment and the CLI model
/// surface the older Windsurf identity does not reach (upstream `wire/devin.ts`).
const IDE_NAME: &str = "devin-cli";
const IDE_TYPE: &str = "chisel";
const IDE_VERSION: &str = "3000.11.3";
const EXTENSION_NAME: &str = "chisel";
const EXTENSION_VERSION: &str = "3000.11.3";

// Proto enum values consumed by the transport (devin-proto.ts).
const CHAT_REQUEST_TYPE_CASCADE: u64 = 5;
const CHAT_SOURCE_USER: u64 = 1;
const CHAT_SOURCE_SYSTEM: u64 = 2;
const CHAT_SOURCE_TOOL: u64 = 4;
const PLANNER_MODE_DEFAULT: u64 = 1;
const CACHE_CONTROL_EPHEMERAL: u64 = 1;
const STOP_REASON_MAX_TOKENS: u64 = 3;

fn gzip(data: &[u8]) -> Result<Vec<u8>, ProviderError> {
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(data)
        .and_then(|()| encoder.finish())
        .map_err(|error| ProviderError::Transport(format!("gzip failed: {error}")))
}

fn gunzip(data: &[u8]) -> Result<Vec<u8>, ProviderError> {
    use std::io::Read;
    let mut out = Vec::new();
    GzDecoder::new(data)
        .read_to_end(&mut out)
        .map_err(|error| ProviderError::Protocol(format!("gzip decode failed: {error}")))?;
    Ok(out)
}

fn devin_os() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "windows",
        _ => "linux",
    }
}

fn normalize_session_token(api_key: &str) -> String {
    if api_key.is_empty() || api_key.starts_with(DEVIN_SESSION_TOKEN_PREFIX) {
        api_key.to_string()
    } else {
        format!("{DEVIN_SESSION_TOKEN_PREFIX}{api_key}")
    }
}

/// `Metadata` for `GetUserJwt` when authenticating by session token
/// (upstream `devinCliMetadata`).
fn cli_metadata(api_key: &str) -> ProtoWriter {
    let mut metadata = ProtoWriter::new();
    metadata.string(1, IDE_NAME);
    metadata.string(7, IDE_VERSION);
    metadata.string(28, IDE_TYPE);
    metadata.string(12, EXTENSION_NAME);
    metadata.string(2, EXTENSION_VERSION);
    metadata.string(3, &normalize_session_token(api_key));
    metadata.string(4, "en");
    metadata.string(5, devin_os());
    metadata
}

/// `Metadata` for `AssignModel`/`GetChatMessage` (upstream `devinWireMetadata`):
/// raw API key + the user JWT minted by `GetUserJwt`.
fn wire_metadata(api_key: &str, user_jwt: &str) -> ProtoWriter {
    let mut metadata = ProtoWriter::new();
    metadata.string(1, IDE_NAME);
    metadata.string(7, IDE_VERSION);
    metadata.string(28, IDE_TYPE);
    metadata.string(12, EXTENSION_NAME);
    metadata.string(2, EXTENSION_VERSION);
    metadata.string(3, api_key);
    metadata.string(4, "en");
    metadata.string(5, devin_os());
    metadata.string(21, user_jwt);
    metadata
}

/// SHA-256 leading-128-bit UUID shape (upstream `deterministicUuid`): stable
/// message ids across history rebuilds without a persisted seed map.
fn deterministic_uuid(seed: &str) -> String {
    let hex = format!("{:x}", sha2::Sha256::digest(seed.as_bytes()));
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Per-turn wire state shared by `AssignModel` and `GetChatMessage`; the
/// assignment JWT is bound to `cascade_id`, so both calls must agree on it.
struct DevinTurn {
    api_key: String,
    user_jwt: String,
    cascade_id: String,
}

struct ModelAssignment {
    assignment_jwt: String,
    model_uid: String,
}

/// `ChatMessagePrompt` for one user/developer turn (field numbers from the
/// pruned upstream schema; empty fields are omitted by `ProtoWriter`).
fn user_prompt(prompt: String, images: Vec<ProtoWriter>, message_id: &str) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.string(1, message_id);
    writer.varint_field(2, CHAT_SOURCE_USER);
    writer.string(3, &prompt);
    writer.repeated_message(10, &images);
    writer
}

/// `ImageData` from an image content block.
fn image_data(data: &str, mime_type: &str) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.string(1, data);
    writer.string(2, mime_type);
    writer
}

/// Tool-call ids are replayed to routed backends that accept only
/// `[a-zA-Z0-9_-]` up to 40 chars (upstream `transformMessages`). Calls and
/// results both pass through here so pairs stay matched. Overlong ids keep a
/// hash suffix so distinct ids sharing a long prefix do not collide.
fn wire_tool_call_id(id: &str) -> String {
    const MAX_LEN: usize = 40;
    let clean: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if clean.len() <= MAX_LEN {
        return clean;
    }
    let hex = format!("{:x}", sha2::Sha256::digest(id.as_bytes()));
    format!("{}_{}", &clean[..MAX_LEN - 9], &hex[..8])
}

/// Map one tool call to `ChatToolCall` (id, name, arguments_json).
fn chat_tool_call(call: &ToolCall) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.string(1, &wire_tool_call_id(&call.id));
    writer.string(2, &call.name);
    let arguments = serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
    writer.string(3, &arguments);
    writer
}

/// Map juto `Message` history onto Cascade `ChatMessagePrompt`s
/// (upstream `buildChatMessagePrompts`): USER for user/developer turns,
/// SYSTEM for assistant replies, TOOL for tool results. `message_id` seeds
/// are `cascade_id\0index\0role` — prompt text excluded so ids stay stable
/// across content edits.
fn build_chat_message_prompts(
    messages: &[crate::Message],
    cascade_id: &str,
    model: &Model,
) -> Vec<ProtoWriter> {
    let mut prompts = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        match message {
            crate::Message::User { content, .. } | crate::Message::Developer { content, .. } => {
                let mut prompt = String::new();
                let mut images = Vec::new();
                for block in content {
                    match block {
                        ContentBlock::Text { text } => prompt.push_str(text),
                        ContentBlock::Image { data, mime_type } => {
                            images.push(image_data(data, mime_type))
                        }
                        _ => {}
                    }
                }
                let role = if matches!(message, crate::Message::User { .. }) {
                    "user"
                } else {
                    "developer"
                };
                let seed = format!("{cascade_id}\0{index}\0{role}");
                prompts.push(user_prompt(prompt, images, &deterministic_uuid(&seed)));
            }
            crate::Message::Assistant(assistant) => {
                let is_native = assistant.api == model.api
                    && assistant.provider == model.provider
                    && assistant.model == model.id;
                let mut prompt_text = String::new();
                let mut thinking_text = String::new();
                let mut signature = String::new();
                let mut tool_calls = Vec::new();
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text { text } => prompt_text.push_str(text),
                        ContentBlock::Thinking {
                            thinking,
                            thinking_signature,
                        } => {
                            thinking_text.push_str(thinking);
                            if is_native
                                && signature.is_empty()
                                && let Some(value) = thinking_signature
                            {
                                signature = value.clone();
                            }
                        }
                        ContentBlock::ToolCall(call) => tool_calls.push(chat_tool_call(call)),
                        _ => {}
                    }
                }
                if prompt_text.is_empty()
                    && thinking_text.is_empty()
                    && signature.is_empty()
                    && tool_calls.is_empty()
                {
                    continue;
                }
                let message_id = if is_native && let Some(id) = &assistant.response_id {
                    id.clone()
                } else {
                    format!(
                        "bot-{}",
                        deterministic_uuid(&format!("{cascade_id}\0{index}\0assistant"))
                    )
                };
                let mut writer = ProtoWriter::new();
                writer.string(1, &message_id);
                writer.varint_field(2, CHAT_SOURCE_SYSTEM);
                writer.string(3, &prompt_text);
                writer.repeated_message(6, &tool_calls);
                writer.string(11, &thinking_text);
                writer.string(12, &signature);
                prompts.push(writer);
            }
            crate::Message::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let mut result_text = String::new();
                let mut images = Vec::new();
                for block in content {
                    match block {
                        ContentBlock::Text { text } => result_text.push_str(text),
                        ContentBlock::Image { data, mime_type } => {
                            images.push(image_data(data, mime_type))
                        }
                        _ => {}
                    }
                }
                let seed = format!("{cascade_id}\0{index}\0tool\0{tool_call_id}");
                let mut writer = ProtoWriter::new();
                writer.string(1, &deterministic_uuid(&seed));
                writer.varint_field(2, CHAT_SOURCE_TOOL);
                writer.string(3, &result_text);
                writer.string(7, &wire_tool_call_id(tool_call_id));
                writer.bool(9, *is_error);
                writer.repeated_message(10, &images);
                prompts.push(writer);
            }
        }
    }
    prompts
}

/// The prompt the router scores: the current user/developer turn on its own
/// (upstream `buildRouterPrompt`), message_id left empty.
fn build_router_prompt(messages: &[crate::Message]) -> Option<ProtoWriter> {
    for message in messages.iter().rev() {
        match message {
            crate::Message::User { content, .. } | crate::Message::Developer { content, .. } => {
                let mut prompt = String::new();
                let mut images = Vec::new();
                for block in content {
                    match block {
                        ContentBlock::Text { text } => prompt.push_str(text),
                        ContentBlock::Image { data, mime_type } => {
                            images.push(image_data(data, mime_type))
                        }
                        _ => {}
                    }
                }
                return Some(user_prompt(prompt, images, ""));
            }
            _ => {}
        }
    }
    None
}

fn prompt_cache_options() -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.varint_field(1, CACHE_CONTROL_EPHEMERAL);
    writer
}

/// `ChatToolChoice` oneof `optionName = "auto"`.
fn tool_choice_auto() -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.string(1, "auto");
    writer
}

fn completion_configuration(model: &Model, options: &crate::StreamOptions) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.varint_field(1, 1); // num_completions
    writer.varint_field(2, options.max_tokens.or(model.max_tokens).unwrap_or(64_000));
    writer.varint_field(3, 200); // max_newlines
    let temperature = options.temperature.map(f64::from).unwrap_or(0.4);
    writer.double(5, temperature);
    writer.double(6, temperature); // first_temperature
    writer.varint_field(7, 50); // top_k
    writer.double(8, 1.0); // top_p
    writer.repeated_string(
        9,
        &DEVIN_DEFAULT_STOP_PATTERNS
            .iter()
            .map(|pattern| pattern.to_string())
            .collect::<Vec<_>>(),
    );
    writer.double(11, 1.0); // fim_eot_prob_threshold
    writer
}

fn chat_tool_definition(tool: &crate::ToolDefinition) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.string(1, &tool.name);
    writer.string(2, &tool.description);
    let schema = serde_json::to_string(&tool.parameters).unwrap_or_else(|_| "{}".into());
    writer.string(3, &schema);
    writer
}

/// `GetChatMessageRequest` for one Cascade turn. Auth rides inside
/// `Metadata.api_key`; `assignment` is present only for router models and
/// supplies both the resolved uid and its JWT.
fn build_chat_request(
    model: &Model,
    context: &Context,
    options: &crate::StreamOptions,
    turn: &DevinTurn,
    assignment: Option<&ModelAssignment>,
    prompts: Vec<ProtoWriter>,
) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.message(1, Some(&wire_metadata(&turn.api_key, &turn.user_jwt)));
    let system_prompt = context
        .system_prompt
        .iter()
        .filter(|prompt| !prompt.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n");
    writer.string(2, &system_prompt);
    writer.repeated_message(3, &prompts);
    let chat_model_uid = assignment
        .map(|assignment| assignment.model_uid.clone())
        .unwrap_or_else(|| model.id.clone());
    writer.string(21, &chat_model_uid);
    writer.varint_field(7, CHAT_REQUEST_TYPE_CASCADE);
    writer.message(8, Some(&completion_configuration(model, options)));
    writer.repeated_message(
        10,
        &context
            .tools
            .iter()
            .map(chat_tool_definition)
            .collect::<Vec<_>>(),
    );
    let parallel = model
        .compat
        .pointer("/supportsParallelToolCalls")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    writer.bool(11, !parallel);
    writer.message(12, Some(&tool_choice_auto()));
    writer.message(13, Some(&prompt_cache_options()));
    writer.string(16, &turn.cascade_id);
    writer.varint_field(20, PLANNER_MODE_DEFAULT);
    writer.string(22, &uuid::Uuid::new_v4().to_string());
    if let Some(assignment) = assignment {
        writer.string(26, &assignment.assignment_jwt);
    }
    writer
}

fn build_assign_request(
    model: &Model,
    turn: &DevinTurn,
    prompt: Option<ProtoWriter>,
) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.message(1, Some(&wire_metadata(&turn.api_key, "")));
    writer.string(2, &model.id);
    writer.string(3, &turn.cascade_id);
    writer.message(5, prompt.as_ref());
    writer
}

fn build_auth_request(metadata: &ProtoWriter) -> ProtoWriter {
    let mut writer = ProtoWriter::new();
    writer.message(1, Some(metadata));
    writer
}

/// Strip credential material from server-echoed text before it reaches a
/// transcript. Empty secrets are skipped: `replace("")` would insert everywhere.
fn redact_secrets(text: String, secrets: &[&str]) -> String {
    secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .fold(text, |text, secret| text.replace(secret, "<redacted>"))
}

/// Decode a unary Connect response: edges variously return bare protobuf or a
/// gzipped protobuf body (upstream `decodeDevinUnaryMessage`), so a failed
/// direct decode falls back to gunzip.
fn decode_unary(payload: &[u8]) -> Option<ProtoMessage> {
    ProtoMessage::parse(payload).ok().or_else(|| {
        gunzip(payload)
            .ok()
            .and_then(|raw| ProtoMessage::parse(&raw).ok())
    })
}

/// Extract a bounded human error without leaking proxy HTML or binary
/// protobuf (upstream `devinErrorDetail`).
fn devin_error_detail(content_type: Option<&str>, payload: &[u8]) -> Option<String> {
    if content_type.is_some_and(|value| value.to_ascii_lowercase().contains("text/html")) {
        return None;
    }
    let text = String::from_utf8(payload.to_vec()).ok()?;
    let trimmed = text.trim();
    let mut detail = trimmed.to_string();
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
        if let Some(Value::Object(error)) = map.get("error")
            && let Some(Value::String(message)) = error.get("message")
        {
            detail = message.trim().to_string();
        } else if let Some(Value::String(error)) = map.get("error") {
            detail = error.trim().to_string();
        } else if let Some(Value::String(message)) = map.get("message") {
            detail = message.trim().to_string();
        }
    }
    let normalized: String = detail.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty()
        || normalized
            .to_lowercase()
            .trim_start()
            .starts_with("<!doctype html")
        || normalized.to_lowercase().trim_start().starts_with("<html")
        // Untrustworthy diagnostics: C0/C1 controls or DEL beyond whitespace.
        || normalized.chars().any(|c| (c < ' ' && !c.is_whitespace()) || c == '\x7f')
    {
        return None;
    }
    let mut truncated: String = normalized
        .chars()
        .take(MAX_DEVIN_ERROR_DETAIL_CHARS)
        .collect();
    if normalized.chars().count() > MAX_DEVIN_ERROR_DETAIL_CHARS {
        truncated.push('…');
    }
    Some(truncated)
}

/// Build the HTTP error for a failed Devin RPC (upstream `createDevinHttpError`):
/// `Devin <op> error <status>` plus a sanitized detail when the body is safe
/// to surface. The stored credential is redacted from the detail.
async fn devin_http_error_async(
    operation: &str,
    response: reqwest::Response,
    credential: Option<&Credential>,
) -> ProviderError {
    let status = response.status();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok());
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let mut bytes = Vec::new();
    // Bounded: a stalled error body must not hang the failure path; partial bytes suffice.
    const ERROR_BODY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    let _ = tokio::time::timeout(ERROR_BODY_TIMEOUT, async {
        let mut body = response.bytes_stream();
        while let Some(Ok(chunk)) = body.next().await {
            let retain = chunk.len().min(32_768 - usize::min(bytes.len(), 32_768));
            bytes.extend_from_slice(&chunk[..retain]);
            if bytes.len() == 32_768 {
                break;
            }
        }
    })
    .await;
    let mut message = match devin_error_detail(content_type.as_deref(), &bytes) {
        Some(detail) => format!("Devin {operation} error {status}: {detail}"),
        None => format!("Devin {operation} error {status}"),
    };
    if let Some(credential) = credential
        && !credential.expose_secret().is_empty()
    {
        message = message.replace(credential.expose_secret(), "<redacted>");
    }
    ProviderError::Http {
        status: status.as_u16(),
        message,
        retry_after,
    }
}

/// One `GetUserJwt` attempt: raw protobuf body, `application/proto`. `metadata`
/// carries the key: session token (`cli_metadata`) or raw key (`wire_metadata`).
async fn request_auth(
    client: &reqwest::Client,
    metadata: &ProtoWriter,
    base_url: &str,
    timeout: std::time::Duration,
) -> Result<(reqwest::StatusCode, Vec<u8>), ProviderError> {
    let request = build_auth_request(metadata);
    let response = tokio::time::timeout(
        timeout,
        client
            .post(format!("{base_url}{AUTH_PATH}"))
            .header("content-type", "application/proto")
            .header("connect-protocol-version", "1")
            .body(request.finish())
            .send(),
    )
    .await
    .map_err(|_| ProviderError::Transport("auth request timeout".into()))?
    .map_err(transport_error)?;
    let status = response.status();
    let payload = tokio::time::timeout(timeout, response.bytes())
        .await
        .map_err(|_| ProviderError::Transport("auth response timeout".into()))?
        .map_err(transport_error)?
        .to_vec();
    Ok((status, payload))
}

/// Parse a Connect end-of-stream JSON trailer and return its structured error
/// when it carries `{ error: { code, message } }` (upstream
/// `readConnectTrailerError`, including bounded `details` summaries).
fn read_trailer_error(text: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(text.trim()).ok()?;
    let error = parsed.get("error")?.as_object()?;
    let code = error.get("code").and_then(Value::as_str).unwrap_or("");
    let message = error.get("message").and_then(Value::as_str).unwrap_or("");
    if code.is_empty() && message.is_empty() {
        return None;
    }
    let mut formatted = format!(
        "Devin stream error{}: {message}",
        if code.is_empty() {
            String::new()
        } else {
            format!(" {code}")
        }
    );
    if let Some(detail) = error.get("details").and_then(summarize_trailer_details) {
        formatted.push_str(&format!(" [details: {detail}]"));
    }
    Some(formatted)
}

const MAX_TRAILER_EVIDENCE_CHARS: usize = 2000;

fn truncate_evidence(text: &str) -> String {
    let mut out: String = text.chars().take(MAX_TRAILER_EVIDENCE_CHARS).collect();
    if text.chars().count() > MAX_TRAILER_EVIDENCE_CHARS {
        out.push('…');
    }
    out
}

/// Summarize Connect error `details` entries (`{type, value, debug}` records)
/// into bounded evidence — the trailer is the only server-side signal for
/// opaque rejections.
fn summarize_trailer_details(details: &Value) -> Option<String> {
    let entries = details.as_array().filter(|entries| !entries.is_empty())?;
    let mut summary = String::new();
    for entry in entries {
        let Some(record) = entry.as_object() else {
            continue;
        };
        let type_ = record
            .get("type")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty());
        let value = record
            .get("value")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty());
        let debug = record.get("debug").map(|d| {
            if d.is_string() {
                d.as_str().unwrap_or_default().to_string()
            } else {
                d.to_string()
            }
        });
        let evidence = debug.as_deref().or(value);
        let part = match (type_, evidence) {
            (Some(t), Some(e)) => Some(format!(
                "{}: {}",
                truncate_evidence(t),
                truncate_evidence(e)
            )),
            (Some(t), None) => Some(truncate_evidence(t)),
            (None, Some(e)) => Some(truncate_evidence(e)),
            (None, None) => None,
        };
        if let Some(part) = part {
            let next = if summary.is_empty() {
                part
            } else {
                format!("{summary}; {part}")
            };
            if next.chars().count() > MAX_TRAILER_EVIDENCE_CHARS {
                return Some(truncate_evidence(&next));
            }
            summary = next;
        }
    }
    if summary.is_empty() {
        None
    } else {
        Some(summary)
    }
}

pub(super) async fn stream_devin(
    client: &reqwest::Client,
    model: &Model,
    context: &Context,
    options: &crate::StreamOptions,
    credential: Option<&Credential>,
    cancel: CancellationToken,
) -> Result<crate::ProviderStream, ProviderError> {
    let base_url = if model.base_url.is_empty() {
        DEVIN_DEFAULT_BASE_URL
    } else {
        model.base_url.trim_end_matches('/')
    }
    .to_string();
    let api_key = credential
        .map(|credential| credential.expose_secret().to_string())
        .unwrap_or_default();

    // --- GetUserJwt: session-token metadata first, raw-key retry on 401. ---
    let session_key = normalize_session_token(&api_key);
    let session_metadata = cli_metadata(&session_key);
    let (status, payload) = tokio::select! {
        _ = cancel.cancelled() => return Err(ProviderError::Aborted),
        result = request_auth(client, &session_metadata, &base_url, options.timeout) => result?,
    };
    let mut auth_status = status;
    let mut auth_payload = payload;
    let mut wire_api_key = session_key.clone();
    if auth_status.as_u16() == 401 && api_key != session_key {
        let raw_metadata = wire_metadata(&api_key, "");
        let retry = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Aborted),
            result = request_auth(client, &raw_metadata, &base_url, options.timeout) => result?,
        };
        auth_status = retry.0;
        auth_payload = retry.1;
        wire_api_key = api_key.clone();
    }
    if !auth_status.is_success() {
        // Redact the session token first: the raw key is a substring of it.
        let detail = devin_error_detail(None, &auth_payload)
            .map(|detail| redact_secrets(detail, &[&session_key, &api_key]));
        return Err(ProviderError::Http {
            status: auth_status.as_u16(),
            message: detail
                .map(|detail| format!("Devin auth error {auth_status}: {detail}"))
                .unwrap_or_else(|| format!("Devin auth error {auth_status}")),
            retry_after: None,
        });
    }
    let decoded = decode_unary(&auth_payload).ok_or_else(|| {
        ProviderError::Protocol("Devin auth error: undecodable GetUserJwt response".into())
    })?;
    let user_jwt = decoded.string(1);
    if user_jwt.is_empty() {
        return Err(ProviderError::Protocol(
            "Devin auth error: GetUserJwt returned an empty user JWT".into(),
        ));
    }
    let custom_base = decoded.string(2);
    let chat_base_url = if custom_base.trim().is_empty() {
        base_url.clone()
    } else {
        custom_base.trim().trim_end_matches('/').to_string()
    };
    let turn = DevinTurn {
        api_key: wire_api_key,
        user_jwt,
        cascade_id: uuid::Uuid::new_v4().to_string(),
    };

    // --- AssignModel for router models (`compat.modelRouter`). The router uid
    // is never a legal chat_model_uid: a failed assignment must fail the turn.
    let model_router = model
        .compat
        .pointer("/modelRouter")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut assignment: Option<ModelAssignment> = None;
    if model_router {
        let request = build_assign_request(model, &turn, build_router_prompt(&context.messages));
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Aborted),
            result = tokio::time::timeout(options.timeout, client
                .post(format!("{chat_base_url}{ASSIGN_MODEL_PATH}"))
                .header("content-type", "application/proto")
                .header("connect-protocol-version", "1")
                .body(request.finish())
                .send()) => result
                .map_err(|_| ProviderError::Transport("AssignModel request timeout".into()))?
                .map_err(transport_error)?,
        };
        if !response.status().is_success() {
            return Err(devin_http_error_async("AssignModel", response, credential).await);
        }
        let payload = tokio::time::timeout(options.timeout, response.bytes())
            .await
            .map_err(|_| ProviderError::Transport("AssignModel response timeout".into()))?
            .map_err(transport_error)?;
        let parsed = decode_unary(&payload)
            .and_then(|message| message.message(1))
            .ok_or_else(|| {
                ProviderError::Protocol(
                    "Devin AssignModel error: response carried no assignment".into(),
                )
            })?;
        let assignment_jwt = parsed.string(1);
        let model_uid = parsed.string(2);
        if assignment_jwt.is_empty() || model_uid.is_empty() {
            return Err(ProviderError::Protocol(
                "Devin AssignModel error: response carried no assignment JWT and model uid".into(),
            ));
        }
        assignment = Some(ModelAssignment {
            assignment_jwt,
            model_uid,
        });
    }

    // --- GetChatMessage: one gzipped Connect request frame, framed response. ---
    let prompts = build_chat_message_prompts(&context.messages, &turn.cascade_id, model);
    let request = build_chat_request(model, context, options, &turn, assignment.as_ref(), prompts);
    let req_bytes = request.finish();
    let frame = frame_connect_message(&gzip(&req_bytes)?, CONNECT_COMPRESSED_FLAG);
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(ProviderError::Aborted),
        result = tokio::time::timeout(options.timeout, client
            .post(format!("{chat_base_url}{CHAT_MESSAGE_PATH}"))
            .header("content-type", "application/connect+proto")
            .header("connect-protocol-version", "1")
            .header("connect-content-encoding", "gzip")
            .header("accept-encoding", "identity")
            .header("user-agent", "connect-go/1.18.1 (go1.26.3)")
            .header("connect-accept-encoding", "gzip")
            .body(frame)
            .send()) => result
            .map_err(|_| ProviderError::Transport("request timeout".into()))?
            .map_err(transport_error)?,
    };
    if !response.status().is_success() {
        return Err(devin_http_error_async("API", response, credential).await);
    }

    let model = model.clone();
    let timeout = options.timeout;
    let cascade_id = turn.cascade_id.clone();
    // Owned history for the context-overflow probe inside the 'static stream.
    let history = context.messages.clone();
    Ok(Box::pin(async_stream::try_stream! {
        yield ProviderEvent::Start;
        let mut output = crate::AssistantMessage::new(&model);
        let mut closed: Vec<bool> = Vec::new();
        let mut current_text: Option<usize> = None;
        let mut current_thinking: Option<usize> = None;
        let mut tool_indexes: HashMap<String, usize> = HashMap::new();
        let mut tool_partial: HashMap<usize, String> = HashMap::new();
        let mut active_tool_call: Option<String> = None;
        let mut latest_stop: u64 = 0;
        let mut saw_token = false;

        macro_rules! end_text {
            ($pending:expr) => {
                if let Some(index) = current_text.take() {
                    if !closed[index] {
                        closed[index] = true;
                        $pending.push(ProviderEvent::BlockEnd { index });
                    }
                }
            };
        }
        macro_rules! end_thinking {
            ($pending:expr) => {
                if let Some(index) = current_thinking.take() {
                    if !closed[index] {
                        closed[index] = true;
                        $pending.push(ProviderEvent::BlockEnd { index });
                    }
                }
            };
        }

        let mut body = response.bytes_stream();
        let mut frames = ConnectFrameDecoder::new(MAX_CONNECT_FRAME_PAYLOAD);
        let mut pending: Vec<ProviderEvent> = Vec::new();
        loop {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => Err(ProviderError::Aborted),
                result = tokio::time::timeout(timeout, body.next()) => result
                    .map_err(|_| ProviderError::Transport("stream idle timeout".into())),
            }?;
            match chunk {
                None => break,
                Some(chunk) => {
                    let chunk = chunk.map_err(transport_error)?;
                    for frame in frames.decode(&chunk)? {
                        if frame.flags & CONNECT_END_STREAM_FLAG != 0 {
                            let trailer_bytes = if frame.flags & CONNECT_COMPRESSED_FLAG != 0 {
                                gunzip(&frame.payload)?
                            } else {
                                frame.payload
                            };
                            let text = String::from_utf8_lossy(&trailer_bytes).trim().to_string();
                            if let Some(mut error) = read_trailer_error(&text) {
                                // Large shrinkable history + an opaque
                                // invalid_argument rejection is likely context
                                // overflow rather than an intermittent error.
                                if !saw_token && error.contains("invalid_argument") && error.to_lowercase().contains("internal error") {
                                    let mut tail = 0usize;
                                    for message in history.iter().rev() {
                                        if matches!(message, crate::Message::User { .. } | crate::Message::Developer { .. }) {
                                            tail += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    let shrinkable = build_chat_message_prompts(
                                        &history[..history.len().saturating_sub(tail)],
                                        &cascade_id,
                                        &model,
                                    );
                                    let mut probe = ProtoWriter::new();
                                    probe.repeated_message(3, &shrinkable);
                                    if probe.finish().len() >= LARGE_HISTORY_RECOVERY_BYTES {
                                        error = format!("{error} (likely context overflow: history alone exceeds {} bytes)", LARGE_HISTORY_RECOVERY_BYTES);
                                    }
                                }
                                Err::<(), ProviderError>(ProviderError::Protocol(error))?;
                            }
                            continue;
                        }
                        let raw = if frame.flags & CONNECT_COMPRESSED_FLAG != 0 {
                            gunzip(&frame.payload)?
                        } else {
                            frame.payload
                        };
                        let message = ProtoMessage::parse(&raw).map_err(|error| {
                            ProviderError::Protocol(format!("invalid GetChatMessageResponse: {error}"))
                        })?;
                        if output.response_id.is_none() && !message.string(1).is_empty() {
                            output.response_id = Some(message.string(1));
                        }

                        let delta_thinking = message.string(9);
                        if !delta_thinking.is_empty() {
                            saw_token = true;
                            end_text!(pending);
                            let index = match current_thinking {
                                Some(index) => index,
                                None => {
                                    let index = output.content.len();
                                    output.content.push(ContentBlock::Thinking {
                                        thinking: String::new(),
                                        thinking_signature: None,
                                    });
                                    closed.push(false);
                                    current_thinking = Some(index);
                                    index
                                }
                            };
                            if let ContentBlock::Thinking { thinking, thinking_signature } = &mut output.content[index] {
                                thinking.push_str(&delta_thinking);
                                let signature = message.string(10);
                                if !signature.is_empty() {
                                    *thinking_signature = Some(signature);
                                }
                            }
                            pending.push(ProviderEvent::ThinkingDelta { index, delta: delta_thinking });
                        }

                        let delta_text = message.string(3);
                        if !delta_text.is_empty() {
                            saw_token = true;
                            end_thinking!(pending);
                            let index = match current_text {
                                Some(index) => index,
                                None => {
                                    let index = output.content.len();
                                    output.content.push(ContentBlock::Text { text: String::new() });
                                    closed.push(false);
                                    current_text = Some(index);
                                    index
                                }
                            };
                            if let ContentBlock::Text { text } = &mut output.content[index] {
                                text.push_str(&delta_text);
                            }
                            pending.push(ProviderEvent::TextDelta { index, delta: delta_text });
                        }

                        for tc in message.repeated_message(6) {
                            let tc_id = {
                                let id = tc.string(1);
                                if id.is_empty() {
                                    active_tool_call.clone().unwrap_or_default()
                                } else {
                                    id
                                }
                            };
                            if tc_id.is_empty() {
                                continue;
                            }
                            saw_token = true;
                            end_text!(pending);
                            end_thinking!(pending);
                            let index = match tool_indexes.get(&tc_id).copied() {
                                Some(index) => index,
                                None => {
                                    let index = output.content.len();
                                    let name = tc.string(2);
                                    output.content.push(ContentBlock::ToolCall(ToolCall {
                                        id: tc_id.clone(),
                                        name: name.clone(),
                                        arguments: serde_json::json!({}),
                                        thought_signature: None,
                                    }));
                                    closed.push(false);
                                    tool_indexes.insert(tc_id.clone(), index);
                                    tool_partial.insert(index, String::new());
                                    pending.push(ProviderEvent::ToolCallStart {
                                        index,
                                        id: tc_id.clone(),
                                        name,
                                    });
                                    index
                                }
                            };
                            let name = tc.string(2);
                            if !name.is_empty()
                                && let ContentBlock::ToolCall(call) = &mut output.content[index]
                            {
                                call.name = name;
                            }
                            active_tool_call = Some(tc_id.clone());
                            let args = tc.string(3);
                            if args.is_empty() {
                                continue;
                            }
                            let previous = tool_partial.get(&index).cloned().unwrap_or_default();
                            let accumulated = if args.starts_with(&previous) {
                                args
                            } else {
                                format!("{previous}{args}")
                            };
                            let delta = accumulated[previous.len()..].to_string();
                            tool_partial.insert(index, accumulated);
                            pending.push(ProviderEvent::ToolCallDelta { index, delta });
                        }

                        let stop = message.varint(5);
                        if stop != 0 {
                            latest_stop = stop;
                        }
                        if let Some(usage) = message.message(7) {
                            output.usage.input = usage.varint(2);
                            output.usage.output = usage.varint(3);
                            output.usage.cache_write = usage.varint(4);
                            output.usage.cache_read = usage.varint(5);
                            output.usage.total_tokens = output.usage.input
                                + output.usage.output
                                + output.usage.cache_read
                                + output.usage.cache_write;
                        }

                        for event in pending.drain(..) {
                            yield event;
                        }
                    }
                }
            }
        }

        end_text!(pending);
        end_thinking!(pending);
        for index in tool_indexes.values() {
            if !closed[*index] {
                closed[*index] = true;
                pending.push(ProviderEvent::BlockEnd { index: *index });
            }
        }
        for event in pending.drain(..) {
            yield event;
        }

        // Final authoritative argument parse for each tool call.
        for index in tool_indexes.values() {
            if let ContentBlock::ToolCall(call) = &mut output.content[*index]
                && let Some(raw) = tool_partial.get(index)
            {
                call.arguments = serde_json::from_str(raw)
                    .unwrap_or_else(|_| serde_json::json!({ "raw": raw }));
            }
        }

        output.stop_reason = if !tool_indexes.is_empty() {
            StopReason::ToolUse
        } else if latest_stop == STOP_REASON_MAX_TOKENS {
            StopReason::Length
        } else {
            StopReason::Stop
        };
        output.usage.price(&model);
        yield ProviderEvent::Done { message: output };
    }))
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use std::sync::Arc;

    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    use super::*;
    use crate::{Context, Message, StreamOptions};

    /// Raw-HTTP stub for the three Devin RPCs. Responses are fixed per path;
    /// every request's head + body is captured for assertions.
    type CapturedRequests = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    async fn spawn_stub(
        auth_response: Vec<u8>,
        assign_response: Vec<u8>,
        chat_frames: Vec<Vec<u8>>,
    ) -> (String, CapturedRequests) {
        spawn_stub_with_first_auth(200, auth_response, assign_response, chat_frames).await
    }

    /// Like `spawn_stub`, but the first `GetUserJwt` answers `first_auth_status`.
    async fn spawn_stub_with_first_auth(
        first_auth_status: u16,
        auth_response: Vec<u8>,
        assign_response: Vec<u8>,
        chat_frames: Vec<Vec<u8>>,
    ) -> (String, CapturedRequests) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests: CapturedRequests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let mut auth_calls = 0usize;
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let mut reader = BufReader::new(stream);
                let mut head = String::new();
                let mut line = String::new();
                let mut content_length = 0usize;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                        break;
                    }
                    head.push_str(&line);
                    if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                    if line.trim().is_empty() {
                        break;
                    }
                }
                let mut body = vec![0u8; content_length];
                let _ = reader.read_exact(&mut body).await;
                seen.lock().push((head.clone(), body));
                let mut status = 200u16;
                let (content_type, payload): (&str, Vec<u8>) = if head.contains(AUTH_PATH) {
                    auth_calls += 1;
                    if auth_calls == 1 {
                        status = first_auth_status;
                    }
                    ("application/proto", auth_response.clone())
                } else if head.contains(ASSIGN_MODEL_PATH) {
                    ("application/proto", assign_response.clone())
                } else {
                    ("application/connect+proto", chat_frames.concat())
                };
                let response = format!(
                    "HTTP/1.1 {status} stub\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    payload.len()
                );
                let stream = reader.get_mut();
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.write_all(&payload).await;
            }
        });
        (format!("http://127.0.0.1:{port}"), requests)
    }

    fn jwt_response(jwt: &str, custom_base: &str) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.string(1, jwt);
        w.string(2, custom_base);
        w.finish()
    }

    fn assign_response(jwt: &str, uid: &str) -> Vec<u8> {
        let mut inner = ProtoWriter::new();
        inner.string(1, jwt);
        inner.string(2, uid);
        let mut w = ProtoWriter::new();
        w.message(1, Some(&inner));
        w.finish()
    }

    /// `GetChatMessageResponse` frame payload: message id + optional deltas.
    fn chat_response(text: &str, stop: u64) -> Vec<u8> {
        let mut w = ProtoWriter::new();
        w.string(1, "resp-1");
        w.string(3, text);
        w.varint_field(5, stop);
        w.finish()
    }

    fn usage_response(input: u64, output: u64) -> Vec<u8> {
        let mut usage = ProtoWriter::new();
        usage.varint_field(2, input);
        usage.varint_field(3, output);
        let mut w = ProtoWriter::new();
        w.message(7, Some(&usage));
        w.finish()
    }

    /// A chat response carrying a tool call delta (id, name, args json).
    fn tool_response(id: &str, name: &str, args: &str) -> Vec<u8> {
        let mut tc = ProtoWriter::new();
        tc.string(1, id);
        tc.string(2, name);
        tc.string(3, args);
        let mut w = ProtoWriter::new();
        w.string(1, "resp-1");
        w.repeated_message(6, vec![&tc]);
        w.finish()
    }

    fn trailer_error_frame(code: &str, message: &str) -> Vec<u8> {
        let json = format!("{{\"error\":{{\"code\":\"{code}\",\"message\":\"{message}\"}}}}");
        frame_connect_message(json.as_bytes(), CONNECT_END_STREAM_FLAG)
    }

    fn devin_model(base_url: &str) -> Model {
        serde_json::from_value(serde_json::json!({
            "id": "swe-1-6",
            "name": "SWE-1.6",
            "api": "devin-agent",
            "provider": "devin",
            "baseUrl": base_url,
            "reasoning": true,
            "input": ["text"],
            "supportsTools": true,
            "cost": {"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0},
            "contextWindow": 200000,
            "maxTokens": 128000,
            "compat": {"supportsParallelToolCalls": true}
        }))
        .unwrap()
    }

    /// Parse the request head's first line + captured body into
    /// (path, decoded-body). The chat body is one Connect frame: flags +
    /// length + gzipped protobuf.
    fn request_body(request: &(String, Vec<u8>)) -> (String, Vec<u8>) {
        let path = request
            .0
            .lines()
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_string();
        let mut body = request.1.clone();
        if body.len() > 5 && body[0] & CONNECT_COMPRESSED_FLAG != 0 {
            let _flags = body[0];
            body = gunzip(&body[5..]).unwrap();
        }
        (path, body)
    }

    /// First (and only) `metadata` field of a request body.
    fn request_metadata(body: &[u8]) -> ProtoMessage {
        ProtoMessage::parse(body).unwrap().message(1).unwrap()
    }

    #[tokio::test]
    async fn streams_text_with_jwt_and_usage() {
        let frames = vec![
            frame_connect_message(&chat_response("Hello", 0), 0),
            frame_connect_message(&chat_response(" world", 2), 0),
            frame_connect_message(&usage_response(11, 7), 0),
            frame_connect_message(b"{}", CONNECT_END_STREAM_FLAG),
        ];
        let (base_url, requests) = spawn_stub(jwt_response("jwt-abc", ""), vec![], frames).await;
        let model = devin_model(&base_url);
        let context = Context {
            system_prompt: vec!["sys".into()],
            messages: vec![Message::user("hi")],
            tools: vec![],
        };
        let client = reqwest::Client::new();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::CredentialStore::open(dir.path().join("auth.json")).unwrap();
        store.set_api_key("devin", "devin-key-1").unwrap();
        let credential = store.resolve_provider("devin").await.unwrap();
        let stream = stream_devin(
            &client,
            &model,
            &context,
            &StreamOptions::default(),
            credential.as_ref(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let events: Vec<_> = stream.collect().await;
        let errors: Vec<_> = events.iter().filter(|e| e.is_err()).collect();
        assert!(errors.is_empty(), "stream errors: {errors:?}");
        let done = events.iter().find_map(|event| match event {
            Ok(ProviderEvent::Done { message }) => Some(message),
            _ => None,
        });
        let message = done.expect("done event");
        assert_eq!(message.text(), "Hello world");
        assert_eq!(message.response_id.as_deref(), Some("resp-1"));
        assert_eq!(message.usage.input, 11);
        assert_eq!(message.usage.output, 7);
        assert_eq!(message.stop_reason, StopReason::Stop);

        // Two requests: GetUserJwt then GetChatMessage.
        let requests = requests.lock();
        assert_eq!(requests.len(), 2);
        let (auth_path, auth_body) = request_body(&requests[0]);
        assert_eq!(auth_path, AUTH_PATH);
        // Session token prefix is applied to the auth attempt.
        assert_eq!(
            request_metadata(&auth_body).string(3),
            format!("{DEVIN_SESSION_TOKEN_PREFIX}devin-key-1")
        );
        let (chat_path, chat_body) = request_body(&requests[1]);
        assert_eq!(chat_path, CHAT_MESSAGE_PATH);
        let metadata = request_metadata(&chat_body);
        assert_eq!(metadata.string(21), "jwt-abc");
        assert_eq!(metadata.string(1), IDE_NAME);
        let chat = ProtoMessage::parse(&chat_body).unwrap();
        assert_eq!(chat.string(2), "sys");
        assert_eq!(chat.string(21), "swe-1-6");
        assert_eq!(chat.varint(7), CHAT_REQUEST_TYPE_CASCADE);
        assert_eq!(chat.varint(20), PLANNER_MODE_DEFAULT);
        let prompts = chat.repeated_message(3);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].string(3), "hi");
        assert_eq!(prompts[0].varint(2), CHAT_SOURCE_USER);
    }

    /// A 401 on the session-token `GetUserJwt` retries with the raw key. The
    /// retry must send that key unprefixed: the session prefix belongs only to
    /// the first attempt (upstream `devinWireMetadata` vs `devinCliMetadata`).
    #[tokio::test]
    async fn auth_401_retries_with_unprefixed_raw_key() {
        let frames = vec![
            frame_connect_message(&chat_response("ok", 0), 0),
            frame_connect_message(b"{}", CONNECT_END_STREAM_FLAG),
        ];
        let (base_url, requests) =
            spawn_stub_with_first_auth(401, jwt_response("jwt-abc", ""), vec![], frames).await;
        let model = devin_model(&base_url);
        let context = Context {
            system_prompt: vec![],
            messages: vec![Message::user("hi")],
            tools: vec![],
        };
        let client = reqwest::Client::new();
        let dir = tempfile::tempdir().unwrap();
        let store = crate::CredentialStore::open(dir.path().join("auth.json")).unwrap();
        store.set_api_key("devin", "devin-key-1").unwrap();
        let credential = store.resolve_provider("devin").await.unwrap();
        let stream = stream_devin(
            &client,
            &model,
            &context,
            &StreamOptions::default(),
            credential.as_ref(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let events: Vec<_> = stream.collect().await;
        assert!(events.iter().all(|event| event.is_ok()), "stream errors");

        let requests = requests.lock();
        assert_eq!(requests.len(), 3, "auth, auth retry, chat");
        let (_, first_auth) = request_body(&requests[0]);
        let (_, retry_auth) = request_body(&requests[1]);
        assert_eq!(
            request_metadata(&first_auth).string(3),
            format!("{DEVIN_SESSION_TOKEN_PREFIX}devin-key-1")
        );
        assert_eq!(request_metadata(&retry_auth).string(3), "devin-key-1");
    }

    #[tokio::test]
    async fn router_models_assign_then_chat() {
        let frames = vec![
            frame_connect_message(&chat_response("ok", 0), 0),
            frame_connect_message(b"{}", CONNECT_END_STREAM_FLAG),
        ];
        let (base_url, requests) = spawn_stub(
            jwt_response("jwt-abc", ""),
            assign_response("assign-jwt", "MODEL_SWE_16"),
            frames,
        )
        .await;
        let mut model = devin_model(&base_url);
        model.id = "adaptive".into();
        model.compat = serde_json::json!({"modelRouter": true});
        let context = Context {
            system_prompt: vec![],
            messages: vec![Message::user("hi")],
            tools: vec![],
        };
        let client = reqwest::Client::new();
        let stream = stream_devin(
            &client,
            &model,
            &context,
            &StreamOptions::default(),
            None,
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let _ = stream.collect::<Vec<_>>().await;
        let requests = requests.lock();
        assert_eq!(requests.len(), 3);
        let (assign_path, assign_body) = request_body(&requests[1]);
        assert_eq!(assign_path, ASSIGN_MODEL_PATH);
        let assign = ProtoMessage::parse(&assign_body).unwrap();
        assert_eq!(assign.string(2), "adaptive");
        assert!(!assign.string(3).is_empty());
        let (_, chat_body) = request_body(&requests[2]);
        let chat = ProtoMessage::parse(&chat_body).unwrap();
        assert_eq!(chat.string(21), "MODEL_SWE_16");
        assert_eq!(chat.string(26), "assign-jwt");
    }

    #[tokio::test]
    async fn trailer_errors_fail_the_stream() {
        let frames = vec![trailer_error_frame("invalid_argument", "internal error")];
        let (base_url, _) = spawn_stub(jwt_response("jwt-abc", ""), vec![], frames).await;
        let model = devin_model(&base_url);
        let context = Context {
            system_prompt: vec![],
            messages: vec![Message::user("hi")],
            tools: vec![],
        };
        let client = reqwest::Client::new();
        let stream = stream_devin(
            &client,
            &model,
            &context,
            &StreamOptions::default(),
            None,
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let events: Vec<_> = stream.collect().await;
        let error = events
            .iter()
            .find_map(|event| event.as_ref().err().map(|e| e.to_string()));
        assert!(error.unwrap().contains("invalid_argument"));
    }

    #[tokio::test]
    async fn tool_call_deltas_accumulate_and_parse() {
        let frames = vec![
            frame_connect_message(&tool_response("call-1", "read", "{\"path\":"), 0),
            frame_connect_message(&tool_response("", "", "\"a.rs\"}"), 0),
            frame_connect_message(b"{}", CONNECT_END_STREAM_FLAG),
        ];
        let (base_url, _) = spawn_stub(jwt_response("jwt-abc", ""), vec![], frames).await;
        let model = devin_model(&base_url);
        let context = Context {
            system_prompt: vec![],
            messages: vec![Message::user("hi")],
            tools: vec![],
        };
        let client = reqwest::Client::new();
        let stream = stream_devin(
            &client,
            &model,
            &context,
            &StreamOptions::default(),
            None,
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let events: Vec<_> = stream.collect().await;
        let done = events.iter().find_map(|event| match event {
            Ok(ProviderEvent::Done { message }) => Some(message),
            _ => None,
        });
        let message = done.expect("done event");
        assert_eq!(message.stop_reason, StopReason::ToolUse);
        let call = message.tool_calls().next().unwrap();
        assert_eq!(call.id, "call-1");
        assert_eq!(call.name, "read");
        assert_eq!(call.arguments, serde_json::json!({"path": "a.rs"}));
    }

    #[test]
    fn deterministic_uuid_is_stable_and_shaped() {
        let a = deterministic_uuid("seed");
        let b = deterministic_uuid("seed");
        assert_eq!(a, b);
        assert_eq!(a.len(), 36);
        assert!(a.chars().nth(8) == Some('-'));
        assert_ne!(a, deterministic_uuid("other"));
    }

    #[test]
    fn trailer_parser_extracts_code_and_details() {
        let error = read_trailer_error(
            r#"{"error":{"code":"unavailable","message":"busy","details":[{"type":"reason","value":"quota"}]}}"#,
        )
        .unwrap();
        assert!(error.contains("unavailable"));
        assert!(error.contains("busy"));
        assert!(error.contains("reason: quota"));
        assert!(read_trailer_error("").is_none());
        assert!(read_trailer_error("{}").is_none());
        assert!(read_trailer_error("not json").is_none());
    }
}
