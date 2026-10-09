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
fn gemini_parts(model: &Model, message: &Message) -> Vec<Value> {
    let signed = matches!(message,Message::Assistant(assistant) if same_model(model,assistant));
    content(message).iter().filter_map(|block|match block {
        ContentBlock::Text{text}=>Some(json!({"text":text})),
        ContentBlock::Image{data,mime_type}=>Some(json!({"inlineData":{"mimeType":mime_type,"data":data}})),
        ContentBlock::Thinking{thinking,thinking_signature}=>{
            let mut part=json!({"text":thinking,"thought":true});if signed{if let Some(signature)=thinking_signature{part["thoughtSignature"]=json!(signature);}}Some(part)
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
        Api::Anthropic => {
            json!({"model":model.id,"messages":anthropic_messages(model,context),"system":context.system_prompt.iter().map(|text|json!({"type":"text","text":text})).collect::<Vec<_>>(),"max_tokens":max_tokens,"stream":true})
        }
        Api::Gemini => {
            json!({"contents":gemini_messages(model,context),"systemInstruction":{"parts":context.system_prompt.iter().map(|text|json!({"text":text})).collect::<Vec<_>>()},"generationConfig":{"maxOutputTokens":max_tokens}})
        }
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
    {
        if model.reasoning {
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
                        body["output_config"] = json!({"effort":effort});
                    } else {
                        let budget = match effort {
                            "minimal" | "low" => 1024,
                            "medium" => 4096,
                            _ => 8192,
                        };
                        if max_tokens > 1024 {
                            body["thinking"] =
                                json!({"type":"enabled","budget_tokens":budget.min(max_tokens-1)});
                            body.as_object_mut().expect("object").remove("temperature");
                        }
                    }
                }
                Api::Gemini => {
                    body["generationConfig"]["thinkingConfig"] = json!({"includeThoughts":true,"thinkingBudget":match effort{"minimal"|"low"=>1024,"medium"=>4096,_=>8192}})
                }
                Api::Ollama => body["think"] = json!(true),
                _ => {}
            }
        }
    }
    if let Some(tier) = &options.service_tier {
        if matches!(api, Api::Chat | Api::Responses | Api::Codex) {
            body["service_tier"] = json!(tier);
        }
    }
    if model.supports_tools && !context.tools.is_empty() {
        match api {
        Api::Chat|Api::Ollama=>body["tools"]=json!(context.tools.iter().map(|tool|json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.parameters}})).collect::<Vec<_>>()),
        Api::Responses|Api::Codex|Api::Azure=>body["tools"]=json!(context.tools.iter().map(|tool|json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.parameters})).collect::<Vec<_>>()),
        Api::Anthropic=>body["tools"]=json!(context.tools.iter().map(|tool|json!({"name":tool.name,"description":tool.description,"input_schema":tool.parameters})).collect::<Vec<_>>()),
        Api::Gemini=>body["tools"]=json!([{"functionDeclarations":context.tools.iter().map(|tool|json!({"name":tool.name,"description":tool.description,"parameters":tool.parameters})).collect::<Vec<_>>() }]),
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
        Api::Anthropic => {
            if oauth {
                body["system"].as_array_mut().expect("array").insert(0,json!({"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude."}));
            }
            format!("{}/messages", base(model, "https://api.anthropic.com/v1"))
        }
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
