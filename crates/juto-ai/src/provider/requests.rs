//! Borrowed-history wire conversion. Provider dialects do not mutate or clone Context.
use super::*;
use crate::{AssistantMessage, ContentBlock, CredentialKind, Message};
use serde_json::json;

fn flag(model: &Model, key: &str, default: bool) -> bool {
    model
        .compat
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}
fn same_model(model: &Model, message: &AssistantMessage) -> bool {
    model.provider == message.provider && model.api == message.api && model.id == message.model
}
fn base<'a>(model: &'a Model, default: &'a str) -> &'a str {
    if model.base_url.is_empty() {
        default
    } else {
        model.base_url.trim_end_matches('/')
    }
}
fn call_id(id: &str) -> String {
    let id = id.split(['|', '\n']).next().unwrap_or(id);
    if !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return id.to_owned();
    }
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(id.as_bytes());
    let mut output = String::from("call_");
    use std::fmt::Write as _;
    for byte in &hash[..16] {
        write!(output, "{byte:02x}").expect("writing String cannot fail");
    }
    output
}
fn text(content: &[ContentBlock]) -> String {
    let mut result = String::new();
    for block in content {
        if let ContentBlock::Text { text } = block {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(text);
        }
    }
    result
}
fn content(message: &Message) -> &[ContentBlock] {
    match message {
        Message::User { content, .. }
        | Message::Developer { content, .. }
        | Message::ToolResult { content, .. } => content,
        Message::Assistant(message) => &message.content,
    }
}
fn data_uri(data: &str, mime: &str) -> String {
    format!("data:{mime};base64,{data}")
}
fn chat_parts(model: &Model, blocks: &[ContentBlock]) -> Value {
    if blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::Image { .. }))
    {
        Value::Array(blocks.iter().filter_map(|block|match block {
            ContentBlock::Text{text}=>Some(json!({"type":"text","text":text})),
            ContentBlock::Image{data,mime_type} if model.input.iter().any(|input|input=="image")=>Some(json!({"type":"image_url","image_url":{"url":data_uri(data,mime_type)}})),
            ContentBlock::Image{..}=>Some(json!({"type":"text","text":"[image omitted: model does not support vision]"})),
            _=>None,
        }).collect())
    } else {
        json!(text(blocks))
    }
}
fn chat_messages(model: &Model, context: &Context, ollama: bool) -> Vec<Value> {
    let mut messages = Vec::with_capacity(context.messages.len() + 1);
    if !context.system_prompt.is_empty() {
        messages.push(json!({"role":"system","content":context.system_prompt.join("\n\n")}));
    }
    for message in &context.messages {
        match message {
            Message::User { content, .. } | Message::Developer { content, .. } => {
                let role = if matches!(message, Message::Developer { .. })
                    && flag(model, "supportsDeveloperRole", true)
                    && !ollama
                {
                    "developer"
                } else {
                    "user"
                };
                let mut wire = json!({"role":role,"content":if ollama {json!(text(content))}else{chat_parts(model,content)}});
                if ollama {
                    let images: Vec<_> = content
                        .iter()
                        .filter_map(|block| {
                            if let ContentBlock::Image { data, .. } = block {
                                Some(data)
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !images.is_empty() {
                        wire["images"] = json!(images);
                    }
                }
                messages.push(wire);
            }
            Message::Assistant(assistant) => {
                let mut wire = json!({"role":"assistant","content":text(&assistant.content)});
                let calls:Vec<_>=assistant.tool_calls().map(|call|{
                    if ollama {json!({"function":{"name":call.name,"arguments":call.arguments}})}
                    else {json!({"id":call_id(&call.id),"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}})}
                }).collect();
                if !calls.is_empty() {
                    wire["tool_calls"] = json!(calls);
                }
                if same_model(model, assistant) {
                    let thinking: String = assistant
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                            _ => None,
                        })
                        .collect();
                    if !thinking.is_empty() {
                        wire[if ollama {
                            "thinking"
                        } else {
                            "reasoning_content"
                        }] = json!(thinking);
                    }
                }
                messages.push(wire);
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } => {
                let mut wire = json!({"role":"tool","content":text(content)});
                if ollama {
                    wire["tool_name"] = json!(tool_name);
                } else {
                    wire["tool_call_id"] = json!(call_id(tool_call_id));
                }
                messages.push(wire);
                let images: Vec<_> = content
                    .iter()
                    .filter(|block| matches!(block, ContentBlock::Image { .. }))
                    .cloned()
                    .collect();
                if !images.is_empty() {
                    if ollama {
                        messages.push(json!({"role":"user","content":"Images from tool result","images":images.iter().filter_map(|block|if let ContentBlock::Image{data,..}=block{Some(data)}else{None}).collect::<Vec<_>>()}));
                    } else {
                        messages.push(json!({"role":"user","content":chat_parts(model,&images)}));
                    }
                }
            }
        }
    }
    messages
}
fn response_input(model: &Model, context: &Context) -> Vec<Value> {
    let mut input = Vec::with_capacity(context.messages.len());
    for message in &context.messages {
        match message {
            Message::User { content, .. } | Message::Developer { content, .. } => {
                let parts: Vec<_> = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => {
                            Some(json!({"type":"input_text","text":text}))
                        }
                        ContentBlock::Image { data, mime_type } => {
                            Some(json!({"type":"input_image","image_url":data_uri(data,mime_type)}))
                        }
                        _ => None,
                    })
                    .collect();
                input.push(json!({"role":if matches!(message,Message::Developer{..}){"developer"}else{"user"},"content":parts}));
            }
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                ContentBlock::Text{text}=>input.push(json!({"role":"assistant","content":[{"type":"output_text","text":text}]})),
                ContentBlock::ToolCall(call)=>input.push(json!({"type":"function_call","call_id":call_id(&call.id),"name":call.name,"arguments":call.arguments.to_string()})),
                ContentBlock::Thinking{thinking,thinking_signature:Some(signature)} if same_model(model,assistant)=>input.push(json!({"type":"reasoning","summary":[{"type":"summary_text","text":thinking}],"encrypted_content":signature})),
                _=>{},
            }
                }
            }
            Message::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                input.push(json!({"type":"function_call_output","call_id":call_id(tool_call_id),"output":text(content)}));
                let images: Vec<_> = content
                    .iter()
                    .filter_map(|block| {
                        if let ContentBlock::Image { data, mime_type } = block {
                            Some(json!({"type":"input_image","image_url":data_uri(data,mime_type)}))
                        } else {
                            None
                        }
                    })
                    .collect();
                if !images.is_empty() {
                    input.push(json!({"role":"user","content":images}));
                }
            }
        }
    }
    input
}
fn anthropic_parts(blocks: &[ContentBlock], signed: bool) -> Vec<Value> {
    blocks.iter().filter_map(|block|match block {
        ContentBlock::Text{text}=>Some(json!({"type":"text","text":text})),
        ContentBlock::Image{data,mime_type}=>Some(json!({"type":"image","source":{"type":"base64","media_type":mime_type,"data":data}})),
        ContentBlock::ToolCall(call)=>Some(json!({"type":"tool_use","id":call_id(&call.id),"name":call.name,"input":call.arguments})),
        ContentBlock::Thinking{thinking,thinking_signature:Some(signature)} if signed=>Some(json!({"type":"thinking","thinking":thinking,"signature":signature})),
        ContentBlock::RedactedThinking{data} if signed=>Some(json!({"type":"redacted_thinking","data":data})),
        ContentBlock::Thinking{thinking,..}=>Some(json!({"type":"text","text":thinking})),
        _=>None,
    }).collect()
}
/// SDK `OutputConfig.effort` accepts only low|medium|high|xhigh|max. Config's
/// `minimal` maps to `low`, matching the budget table on the enabled-thinking path.
fn anthropic_effort(effort: &str) -> &str {
    if effort == "minimal" { "low" } else { effort }
}
fn anthropic_messages(model: &Model, context: &Context) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();
    for message in &context.messages {
        let (role, parts) = match message {
            Message::Assistant(assistant) => (
                "assistant",
                anthropic_parts(&assistant.content, same_model(model, assistant)),
            ),
            Message::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => (
                "user",
                vec![
                    json!({"type":"tool_result","tool_use_id":call_id(tool_call_id),"content":anthropic_parts(content,false),"is_error":is_error}),
                ],
            ),
            _ => ("user", anthropic_parts(content(message), false)),
        };
        if parts.is_empty() {
            continue;
        }
        if messages
            .last()
            .is_some_and(|message| message["role"] == role)
        {
            messages.last_mut().expect("checked above")["content"]
                .as_array_mut()
                .expect("array constructed here")
                .extend(parts);
        } else {
            messages.push(json!({"role":role,"content":parts}));
        }
    }
    messages
}

fn ephemeral(block: &mut Value) {
    block["cache_control"] = json!({"type":"ephemeral"});
}
/// Thinking blocks and empty text blocks cannot carry cache_control.
fn cacheable(block: &Value) -> bool {
    match block["type"].as_str() {
        Some("thinking" | "redacted_thinking") => false,
        Some("text") => block["text"].as_str().is_some_and(|text| !text.is_empty()),
        _ => true,
    }
}
/// Anthropic Messages body with two of the four allowed cache breakpoints: the
/// last cacheable caller system block (tools precede system in the cached prefix,
/// so it caches both) and the last cacheable block of the final message (caches
/// the conversation). The OAuth identity block is inserted after the markers are
/// placed, so it never takes one.
fn anthropic_body(model: &Model, context: &Context, max_tokens: u64, oauth: bool) -> Value {
    let mut system: Vec<Value> = context
        .system_prompt
        .iter()
        .map(|text| json!({"type":"text","text":text}))
        .collect();
    if let Some(block) = system.iter_mut().rev().find(|block| cacheable(block)) {
        ephemeral(block);
    }
    if oauth {
        system.insert(
            0,
            json!({"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude."}),
        );
    }
    let mut messages = anthropic_messages(model, context);
    if let Some(block) = messages
        .last_mut()
        .and_then(|message| message["content"].as_array_mut())
        .and_then(|blocks| blocks.iter_mut().rev().find(|block| cacheable(block)))
    {
        ephemeral(block);
    }
    json!({"model":model.id,"messages":messages,"system":system,"max_tokens":max_tokens,"stream":true})
}

fn gemini_parts(model: &Model, message: &Message) -> Vec<Value> {
    let signed = matches!(message,Message::Assistant(assistant) if same_model(model,assistant));
    content(message).iter().filter_map(|block|match block {
        ContentBlock::Text{text}=>Some(json!({"text":text})),
        ContentBlock::Image{data,mime_type}=>Some(json!({"inlineData":{"mimeType":mime_type,"data":data}})),
        ContentBlock::Thinking{thinking,thinking_signature}=>{
            let mut part=json!({"text":thinking,"thought":true});if signed&&let Some(signature)=thinking_signature{part["thoughtSignature"]=json!(signature);}Some(part)
        }
        ContentBlock::ToolCall(call)=>{
            let mut part=json!({"functionCall":{"name":call.name,"args":call.arguments,"id":call_id(&call.id)}});
            if signed&&call.thought_signature.is_some(){part["thoughtSignature"]=json!(call.thought_signature);}
            else if flag(model,"requiresSkipThoughtSignature",false)||flag(model,"requiresSkipThoughtSignatureOnFirstFunctionCall",false){part["thoughtSignature"]=json!("skip_thought_signature_validator");}
            Some(part)
        }
        _=>None,
    }).collect()
}
fn gemini_messages(model: &Model, context: &Context) -> Vec<Value> {
    context.messages.iter().map(|message|match message {
        Message::ToolResult{tool_call_id,tool_name,content,is_error,..}=>json!({"role":"user","parts":[{"functionResponse":{"id":call_id(tool_call_id),"name":tool_name,"response":if *is_error{json!({"error":text(content)})}else{json!({"output":text(content)})}}}]}),
        Message::Assistant(_)=>json!({"role":"model","parts":gemini_parts(model,message)}),
        _=>json!({"role":"user","parts":gemini_parts(model,message)}),
    }).collect()
}

pub(super) fn build(
    client: &reqwest::Client,
    api: Api,
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    credential: Option<&Credential>,
) -> Result<reqwest::RequestBuilder, ProviderError> {
    let max_tokens = options.max_tokens.or(model.max_tokens).unwrap_or(4096);
    let oauth = credential.is_some_and(|credential| credential.kind == CredentialKind::OAuth);
    let mut body = match api {
        Api::Chat | Api::Ollama => {
            json!({"model":model.id,"messages":chat_messages(model,context,api==Api::Ollama),"stream":true})
        }
        Api::Responses | Api::Codex | Api::Azure => {
            json!({"model":model.id,"input":response_input(model,context),"instructions":context.system_prompt.join("\n\n"),"stream":true,"store":false})
        }
        Api::Anthropic => anthropic_body(model, context, max_tokens, oauth),
        Api::Gemini => {
            json!({"contents":gemini_messages(model,context),"systemInstruction":{"parts":context.system_prompt.iter().map(|text|json!({"text":text})).collect::<Vec<_>>()},"generationConfig":{"maxOutputTokens":max_tokens}})
        }
        Api::Devin => unreachable!("devin-agent builds requests in devin.rs"),
    };
    if let Some(temperature) = options.temperature {
        match api {
            Api::Gemini => body["generationConfig"]["temperature"] = json!(temperature),
            Api::Ollama => body["options"] = json!({"temperature":temperature}),
            _ if flag(model, "supportsSamplingParams", true) => {
                body["temperature"] = json!(temperature)
            }
            _ => {}
        }
    }
    match api {
        Api::Chat => {
            let field = model
                .compat
                .get("maxTokensField")
                .and_then(Value::as_str)
                .unwrap_or("max_tokens");
            body[field] = json!(max_tokens);
            if flag(model, "supportsUsageInStreaming", true) {
                body["stream_options"] = json!({"include_usage":true});
            }
        }
        Api::Responses | Api::Azure => body["max_output_tokens"] = json!(max_tokens),
        Api::Ollama => {
            if body.get("options").is_none() {
                body["options"] = json!({});
            }
            body["options"]["num_predict"] = json!(max_tokens);
            if let Some(window) = model.context_window {
                body["options"]["num_ctx"] = json!(window);
            }
        }
        _ => {}
    }
    if let Some(effort) = options
        .thinking
        .as_deref()
        .filter(|effort| !matches!(*effort, "off" | "inherit" | "auto"))
        && model.reasoning
    {
        match api {
            Api::Chat if flag(model, "supportsReasoningEffort", true) => {
                body["reasoning_effort"] = json!(effort)
            }
            Api::Responses | Api::Codex | Api::Azure => {
                body["reasoning"] = json!({"effort":effort,"summary":"auto"})
            }
            Api::Anthropic => {
                if flag(model, "supportsAdaptiveThinking", false) {
                    body["thinking"] = json!({"type":"adaptive"});
                    body["output_config"] = json!({"effort":anthropic_effort(effort)});
                } else {
                    let budget = match effort {
                        "minimal" | "low" => 1024,
                        "medium" => 4096,
                        _ => 8192,
                    };
                    if max_tokens > 1024 {
                        body["thinking"] =
                            json!({"type":"enabled","budget_tokens":budget.min(max_tokens-1)});
                    }
                }
            }
            Api::Gemini => {
                body["generationConfig"]["thinkingConfig"] = json!({"includeThoughts":true,"thinkingBudget":match effort{"minimal"|"low"=>1024,"medium"=>4096,_=>8192}})
            }
            Api::Ollama => body["think"] = json!(true),
            Api::Devin => unreachable!("devin-agent builds requests in devin.rs"),
            _ => {}
        }
    }
    // Anthropic rejects `temperature` alongside `thinking` before API version 2026-10-21;
    // the pinned `anthropic-version` is 2023-06-01, so drop it on both thinking paths.
    if body.get("thinking").is_some() {
        body.as_object_mut().expect("object").remove("temperature");
    }
    if let Some(tier) = &options.service_tier
        && matches!(api, Api::Chat | Api::Responses | Api::Codex)
    {
        body["service_tier"] = json!(tier);
    }
    if model.supports_tools && !context.tools.is_empty() {
        match api {
        Api::Chat|Api::Ollama=>body["tools"]=json!(context.tools.iter().map(|tool|json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.parameters}})).collect::<Vec<_>>()),
        Api::Responses|Api::Codex|Api::Azure=>body["tools"]=json!(context.tools.iter().map(|tool|json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.parameters})).collect::<Vec<_>>()),
        Api::Anthropic=>body["tools"]=json!(context.tools.iter().map(|tool|json!({"name":tool.name,"description":tool.description,"input_schema":tool.parameters})).collect::<Vec<_>>()),
        Api::Gemini=>body["tools"]=json!([{"functionDeclarations":context.tools.iter().map(|tool|json!({"name":tool.name,"description":tool.description,"parameters":tool.parameters})).collect::<Vec<_>>() }]),
        Api::Devin => unreachable!("devin-agent builds requests in devin.rs"),
    }
    }
    let url = match api {
        Api::Chat => format!(
            "{}/chat/completions",
            base(model, "https://api.openai.com/v1")
        ),
        Api::Responses => format!("{}/responses", base(model, "https://api.openai.com/v1")),
        Api::Codex => {
            let base = base(model, "https://chatgpt.com/backend-api");
            format!(
                "{base}/{}",
                if base.ends_with("/codex") {
                    "responses"
                } else {
                    "codex/responses"
                }
            )
        }
        Api::Azure => format!(
            "{}/responses?api-version={}",
            base(model, "https://api.openai.com/v1"),
            model
                .compat
                .get("apiVersion")
                .and_then(Value::as_str)
                .unwrap_or("2025-04-01-preview")
        ),
        Api::Anthropic => format!("{}/messages", base(model, "https://api.anthropic.com/v1")),
        Api::Gemini => {
            let model_id = model.id.strip_prefix("models/").unwrap_or(&model.id);
            let encoded: String = model_id
                .bytes()
                .map(|byte| {
                    if byte.is_ascii_alphanumeric() || b"-_.".contains(&byte) {
                        (byte as char).to_string()
                    } else {
                        format!("%{byte:02X}")
                    }
                })
                .collect();
            format!(
                "{}/models/{encoded}:streamGenerateContent?alt=sse",
                base(model, "https://generativelanguage.googleapis.com/v1beta")
            )
        }
        Api::Ollama => format!("{}/api/chat", base(model, "http://localhost:11434")),
        Api::Devin => unreachable!("devin-agent builds requests in devin.rs"),
    };
    let mut request = client
        .post(url)
        .header("content-type", "application/json")
        .header(
            "accept",
            if api == Api::Ollama {
                "application/x-ndjson"
            } else {
                "text/event-stream"
            },
        );
    if let Some(credential) = credential {
        request = match api {
            Api::Gemini => request.header("x-goog-api-key", credential.expose_secret()),
            Api::Anthropic if !oauth => request.header("x-api-key", credential.expose_secret()),
            Api::Azure => request.header("api-key", credential.expose_secret()),
            _ => request.bearer_auth(credential.expose_secret()),
        };
    }
    if api == Api::Anthropic {
        request = request.header("anthropic-version", "2023-06-01");
        if oauth {
            request = request
                .header(
                    "anthropic-beta",
                    "oauth-2025-04-20,interleaved-thinking-2025-05-14",
                )
                .header("user-agent", "claude-cli/2.1.280 (external, cli)")
                .header("x-app", "cli");
        }
    }
    if api == Api::Codex {
        if !oauth {
            return Err(ProviderError::Authentication(
                "Codex requires OAuth login to openai-codex".into(),
            ));
        }
        let account = credential
            .and_then(|credential| credential.account_id.as_deref())
            .ok_or_else(|| {
                ProviderError::Authentication(
                    "Codex OAuth credential lacks account identity".into(),
                )
            })?;
        request = request
            .header("chatgpt-account-id", account)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "omp");
    }
    Ok(request.json(&body))
}

#[cfg(test)]
mod tests {
    //! Anthropic request shapes pinned to `@anthropic-ai/sdk` tag `sdk-v1.1.0`
    //! (commit 5badc765f17d373405f300ee8e69e7f490b1f172): `MessageCreateParamsBase`,
    //! `MessageParam`, `TextBlockParam`, `ImageBlockParam` / `Base64ImageSource`,
    //! `ToolUseBlockParam`, `ToolResultBlockParam`, `ThinkingBlockParam`,
    //! `RedactedThinkingBlockParam`, `Tool`, `ThinkingConfigEnabled`, `OutputConfig`.
    use super::*;
    use crate::{
        AssistantMessage, ContentBlock, Context, Message, StreamOptions, ToolCall, ToolDefinition,
    };

    fn anthropic_model() -> Model {
        serde_json::from_value(json!({
            "id": "claude-test",
            "name": "Claude Test",
            "api": "anthropic-messages",
            "provider": "anthropic",
            "baseUrl": "https://api.anthropic.com/v1",
            "reasoning": false,
            "input": ["text"],
            "supportsTools": true,
            "cost": {"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0},
            "contextWindow": 200000,
            "maxTokens": 8192,
            "compat": {}
        }))
        .unwrap()
    }

    fn text_message(text: &str) -> Message {
        Message::User {
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: 0,
        }
    }

    fn assistant_message(text: &str) -> Message {
        let mut message = AssistantMessage::new(&anthropic_model());
        message
            .content
            .push(ContentBlock::Text { text: text.into() });
        Message::Assistant(message)
    }

    fn read_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: format!("{name} a file"),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"}}}),
        }
    }

    /// JSON Pointers of every `cache_control` key in `value`, sorted.
    fn markers(value: &Value) -> Vec<String> {
        fn walk(value: &Value, path: &str, out: &mut Vec<String>) {
            match value {
                Value::Object(map) => {
                    for (key, child) in map {
                        let child_path = format!("{path}/{key}");
                        if key == "cache_control" {
                            out.push(child_path.clone());
                        }
                        walk(child, &child_path, out);
                    }
                }
                Value::Array(items) => {
                    for (index, child) in items.iter().enumerate() {
                        walk(child, &format!("{path}/{index}"), out);
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(value, "", &mut out);
        out.sort();
        out
    }

    fn without_markers(value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .filter(|(key, _)| key.as_str() != "cache_control")
                    .map(|(key, child)| (key.clone(), without_markers(child)))
                    .collect(),
            ),
            Value::Array(items) => Value::Array(items.iter().map(without_markers).collect()),
            other => other.clone(),
        }
    }

    fn request_body(api: Api, context: &Context) -> Value {
        let request = build(
            &reqwest::Client::new(),
            api,
            &anthropic_model(),
            context,
            &StreamOptions::default(),
            None,
        )
        .unwrap()
        .build()
        .unwrap();
        serde_json::from_slice(request.body().unwrap().as_bytes().unwrap()).unwrap()
    }

    #[test]
    fn anthropic_marks_last_system_block_and_final_message_block() {
        let context = Context {
            system_prompt: vec!["base".into(), "rules".into()],
            messages: vec![
                text_message("first"),
                assistant_message("reply"),
                text_message("second"),
            ],
            tools: Vec::new(),
        };
        let body = anthropic_body(&anthropic_model(), &context, 4096, false);
        assert_eq!(
            markers(&body),
            vec![
                "/messages/2/content/0/cache_control",
                "/system/1/cache_control"
            ]
        );
        assert_eq!(
            body["system"][1]["cache_control"],
            json!({"type":"ephemeral"})
        );
    }

    #[test]
    fn oauth_identity_block_does_not_take_the_system_marker() {
        let context = Context {
            system_prompt: vec!["base".into(), "rules".into()],
            messages: vec![text_message("hi")],
            tools: Vec::new(),
        };
        let body = anthropic_body(&anthropic_model(), &context, 4096, true);
        assert_eq!(
            body["system"][0]["text"],
            "You are Claude Code, Anthropic's official CLI for Claude."
        );
        assert_eq!(body["system"].as_array().unwrap().len(), 3);
        assert_eq!(
            markers(&body),
            vec![
                "/messages/0/content/0/cache_control",
                "/system/2/cache_control"
            ]
        );
    }

    #[test]
    fn non_anthropic_bodies_carry_no_cache_control() {
        let context = Context {
            system_prompt: vec!["base".into()],
            messages: vec![text_message("hi"), assistant_message("reply")],
            tools: vec![read_tool("read")],
        };
        for api in [
            Api::Chat,
            Api::Responses,
            Api::Azure,
            Api::Gemini,
            Api::Ollama,
        ] {
            assert!(markers(&request_body(api, &context)).is_empty());
        }
    }

    #[test]
    fn consecutive_turns_share_byte_identical_tools_and_system() {
        let tools = vec![read_tool("grep"), read_tool("read")];
        let turn1 = Context {
            system_prompt: vec!["base".into(), "rules".into()],
            messages: vec![text_message("first")],
            tools: tools.clone(),
        };
        let turn2 = Context {
            system_prompt: vec!["base".into(), "rules".into()],
            messages: vec![
                text_message("first"),
                assistant_message("reply"),
                text_message("second"),
            ],
            tools,
        };
        let first = request_body(Api::Anthropic, &turn1);
        let second = request_body(Api::Anthropic, &turn2);
        assert_eq!(first["tools"].as_array().unwrap().len(), 2);
        assert_eq!(first["tools"].to_string(), second["tools"].to_string());
        assert_eq!(first["system"].to_string(), second["system"].to_string());

        let earlier = without_markers(&first["messages"]);
        let later = without_markers(&second["messages"]);
        let (earlier, later) = (earlier.as_array().unwrap(), later.as_array().unwrap());
        assert_eq!(later[..earlier.len()], earlier[..]);
    }

    #[test]
    fn final_message_ending_in_signed_thinking_marks_preceding_text() {
        let mut reply = AssistantMessage::new(&anthropic_model());
        reply.content.push(ContentBlock::Text {
            text: "answer".into(),
        });
        reply.content.push(ContentBlock::Thinking {
            thinking: "hmm".into(),
            thinking_signature: Some("sig".into()),
        });
        let context = Context {
            system_prompt: vec!["base".into()],
            messages: vec![text_message("hi"), Message::Assistant(reply)],
            tools: Vec::new(),
        };
        let body = anthropic_body(&anthropic_model(), &context, 4096, false);
        assert_eq!(
            markers(&body),
            vec![
                "/messages/1/content/0/cache_control",
                "/system/0/cache_control"
            ]
        );
        assert!(
            body["messages"][1]["content"][1]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(body["messages"][1]["content"][1]["type"], "thinking");
        assert_eq!(body["messages"][1]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn final_message_ending_in_empty_text_skips_the_empty_block() {
        let mut reply = AssistantMessage::new(&anthropic_model());
        reply.content.push(ContentBlock::Text {
            text: "answer".into(),
        });
        reply.content.push(ContentBlock::Text {
            text: String::new(),
        });
        let context = Context {
            system_prompt: vec!["base".into()],
            messages: vec![text_message("hi"), Message::Assistant(reply)],
            tools: Vec::new(),
        };
        let body = anthropic_body(&anthropic_model(), &context, 4096, false);
        assert_eq!(
            markers(&body),
            vec![
                "/messages/1/content/0/cache_control",
                "/system/0/cache_control"
            ]
        );
        assert!(
            body["messages"][1]["content"][1]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn oauth_identity_block_takes_no_marker_when_caller_prompt_is_empty() {
        let context = Context {
            system_prompt: Vec::new(),
            messages: vec![text_message("hi")],
            tools: Vec::new(),
        };
        let body = anthropic_body(&anthropic_model(), &context, 4096, true);
        assert_eq!(body["system"].as_array().unwrap().len(), 1);
        assert_eq!(markers(&body), vec!["/messages/0/content/0/cache_control"]);
    }

    const MESSAGE_CREATE_PARAMS: &[&str] = &[
        "max_tokens",
        "messages",
        "model",
        "cache_control",
        "container",
        "diagnostics",
        "inference_geo",
        "metadata",
        "output_config",
        "service_tier",
        "speed",
        "stop_sequences",
        "stream",
        "system",
        "thinking",
        "tool_choice",
        "tools",
    ];

    fn sdk_anthropic_model(compat: Value) -> Model {
        serde_json::from_value(json!({
            "id": "claude-test",
            "name": "Claude test",
            "api": "anthropic-messages",
            "provider": "anthropic",
            "baseUrl": "https://api.anthropic.com/v1",
            "reasoning": true,
            "maxTokens": 8192,
            "compat": compat,
        }))
        .expect("valid model row")
    }

    fn full_context(model: &Model) -> Context {
        let mut assistant = AssistantMessage::new(model);
        assistant.content = vec![
            ContentBlock::Thinking {
                thinking: "pondering".into(),
                thinking_signature: Some("sig-1".into()),
            },
            ContentBlock::RedactedThinking {
                data: "opaque".into(),
            },
            ContentBlock::Text {
                text: "reading".into(),
            },
            ContentBlock::ToolCall(ToolCall {
                id: "toolu_1".into(),
                name: "read_file".into(),
                arguments: json!({"path": "a.txt"}),
                thought_signature: None,
            }),
        ];
        Context {
            system_prompt: vec!["be terse".into()],
            messages: vec![
                Message::User {
                    content: vec![
                        ContentBlock::Text {
                            text: "look".into(),
                        },
                        ContentBlock::Image {
                            data: "aGk=".into(),
                            mime_type: "image/png".into(),
                        },
                    ],
                    timestamp: 0,
                },
                Message::Assistant(assistant),
                Message::ToolResult {
                    tool_call_id: "toolu_1".into(),
                    tool_name: "read_file".into(),
                    content: vec![ContentBlock::Text {
                        text: "hello".into(),
                    }],
                    is_error: false,
                    details: Value::Null,
                    timestamp: 0,
                },
            ],
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: "Read a file".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                }),
            }],
        }
    }

    fn sdk_anthropic_body_for_test(
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Value {
        let request = build(
            &reqwest::Client::new(),
            Api::Anthropic,
            model,
            context,
            options,
            None,
        )
        .expect("anthropic request builds")
        .build()
        .expect("request assembles");
        let bytes = request
            .body()
            .and_then(|body| body.as_bytes())
            .expect("json body");
        serde_json::from_slice(bytes).expect("json body parses")
    }

    /// Every key we send must be declared by the SDK type, and every required key present.
    fn assert_fields(value: &Value, allowed: &[&str], required: &[&str], shape: &str) {
        let object = value
            .as_object()
            .unwrap_or_else(|| panic!("{shape} must be an object"));
        for key in object.keys() {
            assert!(
                allowed.contains(&key.as_str()),
                "{shape} sends `{key}`, which the SDK type does not declare"
            );
        }
        for key in required {
            assert!(
                object.contains_key(*key),
                "{shape} lacks SDK-required `{key}`"
            );
        }
    }

    fn assert_block(block: &Value) {
        match block["type"].as_str() {
            Some("text") => assert_fields(
                block,
                &["type", "text", "cache_control", "citations"],
                &["type", "text"],
                "TextBlockParam",
            ),
            Some("image") => {
                assert_fields(
                    block,
                    &["type", "source", "cache_control", "transformations"],
                    &["type", "source"],
                    "ImageBlockParam",
                );
                let source = &block["source"];
                assert_fields(
                    source,
                    &["type", "media_type", "data"],
                    &["type", "media_type", "data"],
                    "Base64ImageSource",
                );
                assert!(matches!(
                    source["media_type"].as_str(),
                    Some("image/jpeg" | "image/png" | "image/gif" | "image/webp")
                ));
            }
            Some("tool_use") => assert_fields(
                block,
                &[
                    "type",
                    "id",
                    "name",
                    "input",
                    "cache_control",
                    "caller",
                    "toolset_name",
                ],
                &["type", "id", "name", "input"],
                "ToolUseBlockParam",
            ),
            Some("tool_result") => {
                assert_fields(
                    block,
                    &[
                        "type",
                        "tool_use_id",
                        "content",
                        "is_error",
                        "cache_control",
                    ],
                    &["type", "tool_use_id"],
                    "ToolResultBlockParam",
                );
                for part in block["content"]
                    .as_array()
                    .expect("tool_result content array")
                {
                    assert_block(part);
                }
            }
            Some("thinking") => assert_fields(
                block,
                &["type", "thinking", "signature"],
                &["type", "thinking", "signature"],
                "ThinkingBlockParam",
            ),
            Some("redacted_thinking") => assert_fields(
                block,
                &["type", "data"],
                &["type", "data"],
                "RedactedThinkingBlockParam",
            ),
            other => panic!("block type {other:?} is not a SDK content block param"),
        }
    }

    #[test]
    fn top_level_keys_are_sdk_message_create_params() {
        let model = sdk_anthropic_model(json!({}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            thinking: Some("medium".into()),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        assert_fields(
            &body,
            MESSAGE_CREATE_PARAMS,
            &["model", "messages", "max_tokens"],
            "request body",
        );
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn message_system_and_tool_shapes_match_sdk_params() {
        let model = sdk_anthropic_model(json!({}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        for message in body["messages"].as_array().expect("messages array") {
            assert_fields(
                message,
                &["role", "content"],
                &["role", "content"],
                "MessageParam",
            );
            assert!(matches!(
                message["role"].as_str(),
                Some("user" | "assistant")
            ));
            for block in message["content"].as_array().expect("content array") {
                assert_block(block);
            }
        }
        for block in body["system"].as_array().expect("system array") {
            assert_block(block);
        }
        for tool in body["tools"].as_array().expect("tools array") {
            assert_fields(
                tool,
                &["name", "description", "input_schema"],
                &["name", "input_schema"],
                "Tool",
            );
            assert_eq!(tool["input_schema"]["type"], "object");
        }
    }

    #[test]
    fn temperature_is_sent_to_anthropic_when_sampling_params_are_allowed() {
        let model = sdk_anthropic_model(json!({}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            temperature: Some(0.5),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        assert_eq!(body["temperature"], json!(0.5));
    }

    #[test]
    fn temperature_is_omitted_when_model_disallows_sampling_params() {
        let model = sdk_anthropic_model(json!({"supportsSamplingParams": false}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            temperature: Some(0.2),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn temperature_is_omitted_with_enabled_thinking() {
        let model = sdk_anthropic_model(json!({}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            thinking: Some("high".into()),
            temperature: Some(0.2),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn temperature_is_omitted_with_adaptive_thinking() {
        let model = sdk_anthropic_model(json!({"supportsAdaptiveThinking": true}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            thinking: Some("high".into()),
            temperature: Some(0.2),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn enabled_thinking_budget_is_below_max_tokens() {
        let model = sdk_anthropic_model(json!({}));
        let options = StreamOptions {
            max_tokens: Some(4096),
            thinking: Some("high".into()),
            ..Default::default()
        };
        let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
        assert_fields(
            &body["thinking"],
            &["type", "budget_tokens", "display"],
            &["type", "budget_tokens"],
            "ThinkingConfigEnabled",
        );
        assert_eq!(body["thinking"]["type"], "enabled");
        let budget = body["thinking"]["budget_tokens"]
            .as_u64()
            .expect("integer budget");
        assert!((1024..4096).contains(&budget));
    }

    #[test]
    fn adaptive_effort_is_an_sdk_output_config_value() {
        for (configured, sent) in [("minimal", "low"), ("xhigh", "xhigh"), ("max", "max")] {
            let model = sdk_anthropic_model(json!({"supportsAdaptiveThinking": true}));
            let options = StreamOptions {
                max_tokens: Some(4096),
                thinking: Some(configured.into()),
                ..Default::default()
            };
            let body = sdk_anthropic_body_for_test(&model, &full_context(&model), &options);
            assert_eq!(body["thinking"], json!({"type": "adaptive"}));
            assert_fields(
                &body["output_config"],
                &["effort", "format"],
                &[],
                "OutputConfig",
            );
            assert_eq!(body["output_config"]["effort"], sent);
        }
    }
}
