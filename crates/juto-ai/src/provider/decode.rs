//! Protocol-terminal events, not socket EOF alone, decide successful completion.
use super::*;
use crate::{AssistantMessage, ContentBlock, StopReason, ToolCall, Usage};
use serde_json::json;
use std::collections::HashMap;

pub(super) struct Decoder {
    api: Api,
    output: AssistantMessage,
    slots: HashMap<(u8, u64, u64), usize>,
    raw_args: HashMap<usize, String>,
    closed: Vec<bool>,
    saw_finish: bool,
    terminal: bool,
}
impl Decoder {
    pub fn new(api: Api, model: &Model) -> Self {
        Self {
            api,
            output: AssistantMessage::new(model),
            slots: HashMap::new(),
            raw_args: HashMap::new(),
            closed: Vec::new(),
            saw_finish: false,
            terminal: false,
        }
    }
    fn block(&mut self, key: (u8, u64, u64), block: ContentBlock) -> usize {
        if let Some(index) = self.slots.get(&key) {
            return *index;
        }
        let index = self.output.content.len();
        self.output.content.push(block);
        self.closed.push(false);
        self.slots.insert(key, index);
        index
    }
    fn text(
        &mut self,
        key: (u8, u64, u64),
        delta: &str,
        thinking: bool,
        events: &mut Vec<ProviderEvent>,
    ) {
        if delta.is_empty() {
            return;
        }
        let block = if thinking {
            ContentBlock::Thinking {
                thinking: String::new(),
                thinking_signature: None,
            }
        } else {
            ContentBlock::Text {
                text: String::new(),
            }
        };
        let index = self.block(key, block);
        match &mut self.output.content[index] {
            ContentBlock::Text { text } => text.push_str(delta),
            ContentBlock::Thinking { thinking, .. } => thinking.push_str(delta),
            _ => return,
        };
        events.push(if thinking {
            ProviderEvent::ThinkingDelta {
                index,
                delta: delta.to_owned(),
            }
        } else {
            ProviderEvent::TextDelta {
                index,
                delta: delta.to_owned(),
            }
        });
    }
    fn snapshot_text(
        &mut self,
        key: (u8, u64, u64),
        text: &str,
        thinking: bool,
        events: &mut Vec<ProviderEvent>,
    ) {
        if let Some(index) = self.slots.get(&key).copied() {
            match &mut self.output.content[index] {
                ContentBlock::Text { text: current } => {
                    current.clear();
                    current.push_str(text);
                }
                ContentBlock::Thinking {
                    thinking: current, ..
                } => {
                    current.clear();
                    current.push_str(text);
                }
                _ => {}
            }
        } else {
            self.text(key, text, thinking, events);
        }
    }
    fn signature(&mut self, key: (u8, u64, u64), signature: &str, append: bool) {
        if let Some(index) = self.slots.get(&key).copied() {
            match &mut self.output.content[index] {
                ContentBlock::Thinking {
                    thinking_signature, ..
                } => {
                    if append {
                        thinking_signature
                            .get_or_insert_default()
                            .push_str(signature);
                    } else {
                        *thinking_signature = Some(signature.to_owned());
                    }
                }
                ContentBlock::ToolCall(call) => call.thought_signature = Some(signature.to_owned()),
                _ => {}
            }
        }
    }
    fn tool(
        &mut self,
        key: (u8, u64, u64),
        id: Option<&str>,
        name: Option<&str>,
        events: &mut Vec<ProviderEvent>,
    ) -> usize {
        let existing = self.slots.get(&key).copied();
        let index = match existing {
            Some(index) => index,
            None => {
                let id = id
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("call_{:032x}", rand::random::<u128>()));
                let name = name.unwrap_or_default().to_owned();
                let index = self.block(
                    key,
                    ContentBlock::ToolCall(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: json!({}),
                        thought_signature: None,
                    }),
                );
                events.push(ProviderEvent::ToolCallStart { index, id, name });
                index
            }
        };
        if let ContentBlock::ToolCall(call) = &mut self.output.content[index] {
            if let Some(id) = id.filter(|id| !id.is_empty()) {
                call.id = id.to_owned();
            }
            if let Some(name) = name.filter(|name| !name.is_empty()) {
                call.name = name.to_owned();
            }
        }
        index
    }
    fn argument_delta(&mut self, index: usize, delta: &str, events: &mut Vec<ProviderEvent>) {
        self.raw_args.entry(index).or_default().push_str(delta);
        events.push(ProviderEvent::ToolCallDelta {
            index,
            delta: delta.to_owned(),
        });
    }
    fn argument_snapshot(&mut self, index: usize, args: &str) {
        let raw = self.raw_args.entry(index).or_default();
        raw.clear();
        raw.push_str(args);
    }
    fn close(&mut self, index: usize, events: &mut Vec<ProviderEvent>) {
        if !self.closed[index] {
            self.closed[index] = true;
            events.push(ProviderEvent::BlockEnd { index });
        }
    }
    pub fn close_blocks(&mut self, events: &mut Vec<ProviderEvent>) {
        for index in 0..self.closed.len() {
            self.close(index, events);
        }
    }
    pub fn feed(
        &mut self,
        event: Option<&str>,
        payload: &str,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<bool, ProviderError> {
        if payload.trim() == "[DONE]" {
            self.terminal = true;
            return Ok(true);
        }
        let value: Value = serde_json::from_str(payload)
            .map_err(|error| ProviderError::Protocol(format!("invalid streamed JSON: {error}")))?;
        // Anthropic error frames are typed events handled in `anthropic`, which keeps their type.
        if let Some(error) = value.get("error").filter(|_| self.api != Api::Anthropic) {
            return Err(ProviderError::Protocol(
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| error.as_str())
                    .unwrap_or("provider stream error")
                    .to_owned(),
            ));
        }
        match self.api {
            Api::Chat => self.chat(&value, events)?,
            Api::Responses | Api::Codex | Api::Azure => self.responses(&value, events)?,
            Api::Anthropic => self.anthropic(event, &value, events)?,
            Api::Gemini => self.gemini(&value, events)?,
            Api::Ollama => self.ollama(&value, events)?,
            // Devin streams Connect frames through devin.rs, never SSE payloads.
            Api::Devin => unreachable!("devin-agent does not decode SSE payloads"),
        }
        Ok(self.terminal)
    }
    fn chat(
        &mut self,
        value: &Value,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        if let Some(id) = string(value, "id") {
            self.output.response_id = Some(id.to_owned());
        }
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            let prompt = number(usage, "prompt_tokens");
            let cached = usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| number(usage, "prompt_cache_hit_tokens"));
            self.output.usage = Usage {
                input: prompt.saturating_sub(cached),
                output: number(usage, "completion_tokens"),
                cache_read: cached,
                ..Usage::default()
            };
        }
        for choice in value
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|choice| number(choice, "index") == 0)
        {
            let delta = &choice["delta"];
            if let Some(text) = string(delta, "content") {
                self.text((0, 0, 0), text, false, events);
            }
            if let Some(thinking) =
                string(delta, "reasoning_content").or_else(|| string(delta, "reasoning"))
            {
                self.text((1, 0, 0), thinking, true, events);
            }
            if let Some(refusal) = string(delta, "refusal").filter(|refusal| !refusal.is_empty()) {
                return Err(ProviderError::Protocol(format!("model refused: {refusal}")));
            }
            for call in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let index = self.tool(
                    (2, number(call, "index"), 0),
                    string(call, "id"),
                    string(&call["function"], "name"),
                    events,
                );
                if let Some(args) = string(&call["function"], "arguments") {
                    self.argument_delta(index, args, events);
                } else if let Some(args) = call
                    .pointer("/function/arguments")
                    .filter(|args| args.is_object())
                {
                    self.argument_snapshot(index, &args.to_string());
                }
            }
            if let Some(reason) = string(choice, "finish_reason") {
                self.output.stop_reason = reason_to_stop(reason)?;
                self.saw_finish = true;
            }
        }
        Ok(())
    }
    fn response_item(
        &mut self,
        index: u64,
        item: &Value,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        match string(item, "type") {
            Some("message") => {
                for (part_index, part) in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if let Some(text) = string(part, "text") {
                        self.snapshot_text((3, index, part_index as u64), text, false, events);
                    }
                    if let Some(refusal) = string(part, "refusal") {
                        return Err(ProviderError::Protocol(format!("model refused: {refusal}")));
                    }
                }
            }
            Some("function_call") => {
                let target = self.tool(
                    (4, index, 0),
                    string(item, "call_id"),
                    string(item, "name"),
                    events,
                );
                if let Some(args) = string(item, "arguments")
                    && !args.is_empty()
                {
                    self.argument_snapshot(target, args);
                }
            }
            Some("reasoning") => {
                for (part_index, part) in item
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if let Some(text) = string(part, "text") {
                        self.snapshot_text((5, index, part_index as u64), text, true, events);
                    }
                }
                if let Some(signature) = string(item, "encrypted_content") {
                    let key = (5, index, 0);
                    self.block(
                        key,
                        ContentBlock::Thinking {
                            thinking: String::new(),
                            thinking_signature: None,
                        },
                    );
                    self.signature(key, signature, false);
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn responses(
        &mut self,
        value: &Value,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        let index = number(value, "output_index");
        let part = number(value, "content_index");
        match string(value, "type").unwrap_or_default() {
            "response.created" | "response.in_progress" => {
                if let Some(id) = value.pointer("/response/id").and_then(Value::as_str) {
                    self.output.response_id = Some(id.to_owned());
                }
            }
            "response.output_text.delta" => self.text(
                (3, index, part),
                string(value, "delta").unwrap_or_default(),
                false,
                events,
            ),
            "response.reasoning_summary_text.delta" => self.text(
                (5, index, number(value, "summary_index")),
                string(value, "delta").unwrap_or_default(),
                true,
                events,
            ),
            "response.output_item.added" | "response.output_item.done" => {
                self.response_item(index, &value["item"], events)?
            }
            "response.function_call_arguments.delta" => {
                let target = self.tool((4, index, 0), None, None, events);
                self.argument_delta(target, string(value, "delta").unwrap_or_default(), events);
            }
            "response.function_call_arguments.done" => {
                let target = self.tool(
                    (4, index, 0),
                    string(value, "call_id"),
                    string(value, "name"),
                    events,
                );
                if let Some(args) = string(value, "arguments") {
                    self.argument_snapshot(target, args);
                }
            }
            "response.completed" | "response.done" | "response.incomplete" => {
                let response = &value["response"];
                if let Some(error) = response.get("error").filter(|error| !error.is_null()) {
                    return Err(ProviderError::Protocol(
                        error["message"]
                            .as_str()
                            .unwrap_or("response failed")
                            .to_owned(),
                    ));
                }
                if let Some(id) = string(response, "id") {
                    self.output.response_id = Some(id.to_owned());
                }
                for (index, item) in response
                    .get("output")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    self.response_item(index as u64, item, events)?;
                }
                let usage = &response["usage"];
                let input = number(usage, "input_tokens");
                let cached = usage
                    .pointer("/input_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                self.output.usage = Usage {
                    input: input.saturating_sub(cached),
                    cache_read: cached,
                    output: number(usage, "output_tokens"),
                    ..Usage::default()
                };
                self.output.stop_reason = if string(response, "status") == Some("incomplete")
                    || string(value, "type") == Some("response.incomplete")
                {
                    if response
                        .pointer("/incomplete_details/reason")
                        .and_then(Value::as_str)
                        == Some("max_output_tokens")
                    {
                        StopReason::Length
                    } else {
                        return Err(ProviderError::Protocol("response incomplete".into()));
                    }
                } else if self.output.tool_calls().next().is_some() {
                    StopReason::ToolUse
                } else {
                    StopReason::Stop
                };
                self.saw_finish = true;
                self.terminal = true;
            }
            "error" | "response.failed" => {
                return Err(ProviderError::Protocol(
                    value
                        .pointer("/response/error/message")
                        .or_else(|| value.pointer("/error/message"))
                        .and_then(Value::as_str)
                        .unwrap_or("response failed")
                        .to_owned(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
    fn anthropic(
        &mut self,
        event: Option<&str>,
        value: &Value,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        let wire = number(value, "index");
        let key = (6, wire, 0);
        // The SDK routes on the SSE event name, so an `error` frame is an error without a `type`.
        let kind = if event == Some("error") {
            "error"
        } else {
            string(value, "type").unwrap_or_default()
        };
        match kind {
            "message_start" => {
                if let Some(id) = value.pointer("/message/id").and_then(Value::as_str) {
                    self.output.response_id = Some(id.to_owned());
                }
                let usage = &value["message"]["usage"];
                self.output.usage.input = number(usage, "input_tokens");
                self.output.usage.output = number(usage, "output_tokens");
                self.output.usage.cache_read = number(usage, "cache_read_input_tokens");
                self.output.usage.cache_write = number(usage, "cache_creation_input_tokens");
            }
            "content_block_start" => {
                let block = &value["content_block"];
                match string(block, "type") {
                    Some("text") => {
                        self.block(
                            key,
                            ContentBlock::Text {
                                text: String::new(),
                            },
                        );
                        self.text(
                            key,
                            string(block, "text").unwrap_or_default(),
                            false,
                            events,
                        );
                    }
                    Some("thinking") => {
                        self.block(
                            key,
                            ContentBlock::Thinking {
                                thinking: String::new(),
                                thinking_signature: None,
                            },
                        );
                        self.text(
                            key,
                            string(block, "thinking").unwrap_or_default(),
                            true,
                            events,
                        );
                        if let Some(signature) = string(block, "signature") {
                            self.signature(key, signature, false);
                        }
                    }
                    Some("redacted_thinking") => {
                        self.block(
                            key,
                            ContentBlock::RedactedThinking {
                                data: string(block, "data").unwrap_or_default().to_owned(),
                            },
                        );
                    }
                    Some("tool_use") => {
                        let target =
                            self.tool(key, string(block, "id"), string(block, "name"), events);
                        if let Some(input) = block.get("input").filter(|input| {
                            input.as_object().is_some_and(|input| !input.is_empty())
                        }) {
                            self.argument_snapshot(target, &input.to_string());
                        }
                    }
                    other => {
                        return Err(ProviderError::Protocol(format!(
                            "unsupported Anthropic content block `{}`",
                            other.unwrap_or("<missing type>")
                        )));
                    }
                }
            }
            "content_block_delta" => {
                let delta = &value["delta"];
                match string(delta, "type") {
                    Some("text_delta") => self.text(
                        key,
                        string(delta, "text").unwrap_or_default(),
                        false,
                        events,
                    ),
                    Some("thinking_delta") => self.text(
                        key,
                        string(delta, "thinking").unwrap_or_default(),
                        true,
                        events,
                    ),
                    Some("signature_delta") => {
                        self.signature(key, string(delta, "signature").unwrap_or_default(), true)
                    }
                    Some("input_json_delta") => {
                        let target = self.slots.get(&key).copied().ok_or_else(|| {
                            ProviderError::Protocol("tool delta preceded its start".into())
                        })?;
                        self.argument_delta(
                            target,
                            string(delta, "partial_json").unwrap_or_default(),
                            events,
                        );
                    }
                    other => {
                        return Err(ProviderError::Protocol(format!(
                            "unsupported Anthropic content delta `{}`",
                            other.unwrap_or("<missing type>")
                        )));
                    }
                }
            }
            "content_block_stop" => {
                if let Some(index) = self.slots.get(&key).copied() {
                    self.close(index, events);
                }
            }
            "message_delta" => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.output.stop_reason = anthropic_stop_reason(reason, &value["delta"])?;
                    self.saw_finish = true;
                }
                // Counters are cumulative: a null or absent field keeps the value already seen.
                let usage = &value["usage"];
                if let Some(output) = usage["output_tokens"].as_u64() {
                    self.output.usage.output = output;
                }
                if let Some(input) = usage["input_tokens"].as_u64() {
                    self.output.usage.input = input;
                }
                if let Some(read) = usage["cache_read_input_tokens"].as_u64() {
                    self.output.usage.cache_read = read;
                }
                if let Some(write) = usage["cache_creation_input_tokens"].as_u64() {
                    self.output.usage.cache_write = write;
                }
            }
            "message_stop" => self.terminal = true,
            "error" => {
                let error = &value["error"];
                return Err(ProviderError::Protocol(format!(
                    "Anthropic stream error ({}): {}",
                    string(error, "type").unwrap_or("unknown"),
                    string(error, "message").unwrap_or("no message"),
                )));
            }
            // `ping` and unknown event types carry no message state and are ignored.
            _ => {}
        }
        Ok(())
    }
    fn gemini(
        &mut self,
        value: &Value,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        if let Some(reason) = value
            .pointer("/promptFeedback/blockReason")
            .and_then(Value::as_str)
        {
            return Err(ProviderError::Protocol(format!(
                "Gemini blocked prompt: {reason}"
            )));
        }
        if let Some(id) = string(value, "responseId") {
            self.output.response_id = Some(id.to_owned());
        }
        for candidate in value
            .get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|candidate| number(candidate, "index") == 0)
        {
            for (part_index, part) in candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                let thinking = part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let key = (if thinking { 8 } else { 7 }, 0, 0);
                if let Some(text) = string(part, "text") {
                    self.text(key, text, thinking, events);
                    if let Some(signature) = string(part, "thoughtSignature") {
                        self.signature(key, signature, false);
                    }
                }
                if let Some(call) = part.get("functionCall") {
                    let key = (9, part_index as u64, self.output.content.len() as u64);
                    let target = self.tool(key, string(call, "id"), string(call, "name"), events);
                    if let Some(args) = call.get("args") {
                        self.argument_snapshot(target, &args.to_string());
                        events.push(ProviderEvent::ToolCallDelta {
                            index: target,
                            delta: args.to_string(),
                        });
                    }
                    if let Some(signature) = string(part, "thoughtSignature") {
                        self.signature(key, signature, false);
                    }
                }
            }
            if let Some(reason) = string(candidate, "finishReason") {
                self.output.stop_reason = match reason {
                    "STOP" => {
                        if self.output.tool_calls().next().is_some() {
                            StopReason::ToolUse
                        } else {
                            StopReason::Stop
                        }
                    }
                    "MAX_TOKENS" => StopReason::Length,
                    _ => {
                        return Err(ProviderError::Protocol(format!(
                            "Gemini finish reason: {reason}"
                        )));
                    }
                };
                self.saw_finish = true;
            }
        }
        if let Some(usage) = value.get("usageMetadata") {
            let prompt = number(usage, "promptTokenCount");
            let cached = number(usage, "cachedContentTokenCount");
            self.output.usage = Usage {
                input: prompt.saturating_sub(cached),
                cache_read: cached,
                output: number(usage, "candidatesTokenCount") + number(usage, "thoughtsTokenCount"),
                ..Usage::default()
            };
        }
        Ok(())
    }
    fn ollama(
        &mut self,
        value: &Value,
        events: &mut Vec<ProviderEvent>,
    ) -> Result<(), ProviderError> {
        let message = &value["message"];
        if let Some(text) = string(message, "content") {
            self.text((10, 0, 0), text, false, events);
        }
        if let Some(thinking) = string(message, "thinking") {
            self.text((11, 0, 0), thinking, true, events);
        }
        for (index, call) in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let target = self.tool(
                (12, index as u64, 0),
                string(call, "id"),
                string(&call["function"], "name"),
                events,
            );
            if let Some(args) = call.pointer("/function/arguments") {
                let raw = if let Some(raw) = args.as_str() {
                    raw.to_owned()
                } else {
                    args.to_string()
                };
                self.argument_snapshot(target, &raw);
                events.push(ProviderEvent::ToolCallDelta {
                    index: target,
                    delta: raw,
                });
            }
        }
        if value.get("done").and_then(Value::as_bool) == Some(true) {
            self.output.stop_reason = match string(value, "done_reason").unwrap_or("stop") {
                "stop" => {
                    if self.output.tool_calls().next().is_some() {
                        StopReason::ToolUse
                    } else {
                        StopReason::Stop
                    }
                }
                "length" => StopReason::Length,
                other => {
                    return Err(ProviderError::Protocol(format!(
                        "Ollama done reason: {other}"
                    )));
                }
            };
            let prompt = number(value, "prompt_eval_count");
            let cached = number(value, "prompt_eval_cached_count");
            self.output.usage = Usage {
                input: prompt.saturating_sub(cached),
                cache_read: cached,
                output: number(value, "eval_count"),
                ..Usage::default()
            };
            self.saw_finish = true;
            self.terminal = true;
        }
        Ok(())
    }
    pub fn finish(mut self, model: &Model) -> Result<AssistantMessage, ProviderError> {
        let complete = self.saw_finish
            && match self.api {
                Api::Chat | Api::Gemini => true,
                _ => self.terminal,
            };
        if !complete {
            return Err(ProviderError::Protocol(
                "stream ended without its protocol completion event".into(),
            ));
        }
        for (index, raw) in self.raw_args {
            if let ContentBlock::ToolCall(call) = &mut self.output.content[index] {
                call.arguments = serde_json::from_str(&raw).unwrap_or_else(
                    |_| json!({"__parseError":"model supplied invalid JSON tool arguments"}),
                );
            }
        }
        if self.output.tool_calls().any(|call| call.name.is_empty()) {
            return Err(ProviderError::Protocol("tool call has no name".into()));
        }
        if !self.output.content.iter().any(|block| match block {
            ContentBlock::Text { text } => !text.is_empty(),
            ContentBlock::Thinking { thinking, .. } => !thinking.is_empty(),
            ContentBlock::ToolCall(_) => true,
            _ => false,
        }) {
            return Err(ProviderError::Protocol(
                "model completed without visible content or tool calls".into(),
            ));
        }
        self.output.usage.price(model);
        Ok(self.output)
    }
}
/// Anthropic `StopReason` values from `@anthropic-ai/sdk` sdk-v1.1.0. Unmapped values are errors,
/// never silent completions.
fn anthropic_stop_reason(reason: &str, delta: &Value) -> Result<StopReason, ProviderError> {
    match reason {
        "end_turn" | "stop_sequence" => Ok(StopReason::Stop),
        "max_tokens" | "model_context_window_exceeded" => Ok(StopReason::Length),
        "tool_use" => Ok(StopReason::ToolUse),
        "refusal" => {
            let details = &delta["stop_details"];
            Err(ProviderError::Protocol(format!(
                "model refused the request (category: {}): {}",
                string(details, "category").unwrap_or("none"),
                string(details, "explanation").unwrap_or("no explanation"),
            )))
        }
        "pause_turn" => Err(ProviderError::Protocol(
            "Anthropic pause_turn needs server-tool continuation, which juto does not implement"
                .to_owned(),
        )),
        other => Err(ProviderError::Protocol(format!(
            "unsupported Anthropic stop reason `{other}`"
        ))),
    }
}

/// SSE frames skipped before decoding, as `@anthropic-ai/sdk` streaming does: `ping` and frames
/// with no event name. Other dialects carry their type in the payload, so they are never skipped.
pub(super) fn frame_is_control(api: Api, event: Option<&str>) -> bool {
    api == Api::Anthropic && matches!(event, None | Some("ping"))
}
fn string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}
fn number(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}
fn reason_to_stop(reason: &str) -> Result<StopReason, ProviderError> {
    match reason {
        "stop" | "end_turn" | "stop_sequence" => Ok(StopReason::Stop),
        "length" | "max_tokens" => Ok(StopReason::Length),
        "tool_calls" | "function_call" | "tool_use" => Ok(StopReason::ToolUse),
        _ => Err(ProviderError::Protocol(format!(
            "provider stop reason: {reason}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    //! Anthropic stream contract, pinned to `@anthropic-ai/sdk` tag `sdk-v1.1.0`
    //! (commit 5badc765f17d373405f300ee8e69e7f490b1f172): `RawMessageStreamEvent`,
    //! `RawContentBlockDelta`, `ContentBlock`, `Usage`, `MessageDeltaUsage`, `StopReason`,
    //! `RefusalStopDetails`. Fixtures under tests/fixtures/anthropic/ are hand-written to those
    //! types, not captured from the live API.
    use super::*;
    use crate::sse::SseDecoder;
    use serde_json::json;

    #[test]
    fn anthropic_message_start_reports_cache_read_and_write_tokens() {
        let model: Model = serde_json::from_value(json!({
            "id": "claude-test",
            "name": "Claude Test",
            "api": "anthropic-messages",
            "provider": "anthropic",
            "baseUrl": "https://api.anthropic.com/v1",
            "supportsTools": true,
            "cost": {"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0},
            "maxTokens": 8192,
            "compat": {}
        }))
        .unwrap();
        let mut decoder = Decoder::new(Api::Anthropic, &model);
        let mut events = Vec::new();
        for payload in [
            r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":12,"cache_read_input_tokens":1500,"cache_creation_input_tokens":300,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
            r#"{"type":"message_stop"}"#,
        ] {
            decoder.feed(None, payload, &mut events).unwrap();
        }
        let message = decoder.finish(&model).unwrap();
        assert_eq!(message.usage.input, 12);
        assert_eq!(message.usage.cache_read, 1500);
        assert_eq!(message.usage.cache_write, 300);
        assert_eq!(message.usage.output, 7);
    }

    #[test]
    fn anthropic_message_delta_updates_cache_usage() {
        let model: Model = serde_json::from_value(json!({
            "id": "claude-test",
            "name": "Claude Test",
            "api": "anthropic-messages",
            "provider": "anthropic",
            "baseUrl": "https://api.anthropic.com/v1",
            "supportsTools": true,
            "cost": {"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0},
            "maxTokens": 8192,
            "compat": {}
        }))
        .unwrap();
        let mut decoder = Decoder::new(Api::Anthropic, &model);
        let mut events = Vec::new();
        for payload in [
            r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":12,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"cache_read_input_tokens":900,"cache_creation_input_tokens":250}}"#,
            r#"{"type":"message_stop"}"#,
        ] {
            decoder.feed(None, payload, &mut events).unwrap();
        }
        let message = decoder.finish(&model).unwrap();
        assert_eq!(message.usage.cache_read, 900);
        assert_eq!(message.usage.cache_write, 250);
        assert_eq!(message.usage.output, 7);
    }

    const TEXT: &str = include_str!("../../tests/fixtures/anthropic/text_stream.sse");
    const THINKING: &str = include_str!("../../tests/fixtures/anthropic/thinking_signature.sse");
    const TOOL: &str = include_str!("../../tests/fixtures/anthropic/tool_input_json.sse");
    const ERROR: &str = include_str!("../../tests/fixtures/anthropic/error_event.sse");

    const UNKNOWN_BLOCK: &str = r#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}

"#;

    const CITATIONS: &str = r#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":null}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":"x","document_index":0,"document_title":null,"start_char_index":0,"end_char_index":1,"file_id":null}}}

"#;

    const REFUSAL: &str = r#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"cyber","explanation":"Blocked by policy"}},"usage":{"input_tokens":null,"output_tokens":3,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}}

"#;

    const CONTEXT_WINDOW: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":null}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Partial"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"model_context_window_exceeded","stop_sequence":null},"usage":{"input_tokens":null,"output_tokens":0,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}}

event: message_stop
data: {"type":"message_stop"}

"#;

    const ERROR_WITHOUT_TYPE: &str = r#"event: error
data: {"error":{"type":"overloaded_error","message":"Overloaded"}}

"#;

    const NULL_DELTA_USAGE: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_03","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":0,"cache_creation_input_tokens":4,"cache_read_input_tokens":9}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":null}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":null,"output_tokens":12,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}}

event: message_stop
data: {"type":"message_stop"}

"#;

    fn anthropic_model() -> Model {
        serde_json::from_value(json!({
            "id": "claude-test",
            "name": "Claude test",
            "api": "anthropic-messages",
            "provider": "anthropic",
            "baseUrl": "https://api.anthropic.com/v1",
            "reasoning": true,
            "maxTokens": 8192,
        }))
        .expect("valid model row")
    }

    /// Runs SSE bytes through the transport's frame filter and the Anthropic decoder.
    fn run(stream: &str) -> Result<(Vec<ProviderEvent>, AssistantMessage), ProviderError> {
        let model = anthropic_model();
        let mut sse = SseDecoder::new();
        let mut decoder = Decoder::new(Api::Anthropic, &model);
        let mut events = Vec::new();
        let mut frames = sse.feed(stream.as_bytes())?;
        frames.extend(sse.finish()?);
        for frame in frames {
            if frame.data.is_empty() || frame_is_control(Api::Anthropic, frame.event.as_deref()) {
                continue;
            }
            if decoder.feed(frame.event.as_deref(), &frame.data, &mut events)? {
                break;
            }
        }
        decoder.close_blocks(&mut events);
        let message = decoder.finish(&model)?;
        Ok((events, message))
    }

    fn error_of(stream: &str) -> String {
        match run(stream) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("stream should have been rejected"),
        }
    }

    fn kinds(events: &[ProviderEvent]) -> Vec<String> {
        events
            .iter()
            .map(|event| match event {
                ProviderEvent::TextDelta { index, delta } => format!("text[{index}]:{delta}"),
                ProviderEvent::ThinkingDelta { index, delta } => {
                    format!("thinking[{index}]:{delta}")
                }
                ProviderEvent::ToolCallStart { index, name, .. } => {
                    format!("tool_start[{index}]:{name}")
                }
                ProviderEvent::ToolCallDelta { index, delta } => {
                    format!("tool_delta[{index}]:{delta}")
                }
                ProviderEvent::BlockEnd { index } => format!("end[{index}]"),
                _ => "other".to_owned(),
            })
            .collect()
    }

    #[test]
    fn text_stream_maps_text_deltas_usage_and_stop() {
        let (events, message) = run(TEXT).expect("text stream decodes");
        assert_eq!(
            kinds(&events),
            ["text[0]:Hello", "text[0]:, world", "end[0]"]
        );
        assert!(matches!(
            &message.content[..],
            [ContentBlock::Text { text }] if text == "Hello, world"
        ));
        assert_eq!(message.usage.input, 25);
        assert_eq!(message.usage.output, 12);
        assert_eq!(message.usage.cache_read, 0);
        assert_eq!(message.usage.cache_write, 0);
        assert!(matches!(message.stop_reason, StopReason::Stop));
    }

    #[test]
    fn thinking_signature_is_carried_onto_the_block() {
        let (events, message) = run(THINKING).expect("thinking stream decodes");
        assert_eq!(
            kinds(&events),
            [
                "thinking[0]:Let me think.",
                "end[0]",
                "text[1]:Done.",
                "end[1]",
            ]
        );
        match &message.content[..] {
            [
                ContentBlock::Thinking {
                    thinking_signature, ..
                },
                ContentBlock::Text { .. },
            ] => {
                assert_eq!(thinking_signature.as_deref(), Some("EqQBsig"));
            }
            _ => panic!("expected thinking then text blocks"),
        }
    }

    #[test]
    fn tool_input_json_deltas_assemble_valid_arguments_and_usage() {
        let (events, message) = run(TOOL).expect("tool stream decodes");
        assert_eq!(
            kinds(&events),
            [
                "tool_start[0]:get_weather",
                "tool_delta[0]:{\"city\":",
                "tool_delta[0]: \"Paris\"}",
                "end[0]",
            ]
        );
        match &message.content[..] {
            [ContentBlock::ToolCall(call)] => {
                assert_eq!(call.id, "toolu_01A");
                assert_eq!(call.arguments, json!({"city": "Paris"}));
            }
            _ => panic!("expected one tool call"),
        }
        assert_eq!(message.usage.input, 30);
        assert_eq!(message.usage.output, 17);
        assert_eq!(message.usage.cache_read, 9);
        assert_eq!(message.usage.cache_write, 4);
        assert!(matches!(message.stop_reason, StopReason::ToolUse));
    }

    #[test]
    fn error_event_surfaces_type_and_message() {
        let error = error_of(ERROR);
        assert!(error.contains("overloaded_error"), "{error}");
        assert!(error.contains("Overloaded"), "{error}");
    }

    #[test]
    fn error_frame_without_payload_type_is_routed_by_event_name() {
        let error = error_of(ERROR_WITHOUT_TYPE);
        assert!(error.contains("overloaded_error"), "{error}");
        assert!(error.contains("Overloaded"), "{error}");
    }

    #[test]
    fn unknown_content_block_type_is_rejected_by_name() {
        assert!(error_of(UNKNOWN_BLOCK).contains("server_tool_use"));
    }

    #[test]
    fn citations_delta_is_rejected_by_name() {
        assert!(error_of(CITATIONS).contains("citations_delta"));
    }

    #[test]
    fn refusal_stop_reason_is_an_error_with_details() {
        let error = error_of(REFUSAL);
        assert!(error.contains("cyber"), "{error}");
        assert!(error.contains("Blocked by policy"), "{error}");
    }

    #[test]
    fn context_window_stop_reason_maps_to_length() {
        let (_, message) = run(CONTEXT_WINDOW).expect("stream decodes");
        assert!(matches!(message.stop_reason, StopReason::Length));
    }

    #[test]
    fn null_message_delta_usage_keeps_the_previously_seen_values() {
        let (_, message) = run(NULL_DELTA_USAGE).expect("stream decodes");
        assert_eq!(message.usage.input, 25);
        assert_eq!(message.usage.output, 12);
        assert_eq!(message.usage.cache_read, 9);
        assert_eq!(message.usage.cache_write, 4);
        assert!(matches!(message.stop_reason, StopReason::Stop));
    }

    #[test]
    fn ping_and_unnamed_frames_are_control_frames_for_anthropic_only() {
        assert!(frame_is_control(Api::Anthropic, Some("ping")));
        assert!(frame_is_control(Api::Anthropic, None));
        assert!(!frame_is_control(Api::Anthropic, Some("message_start")));
        assert!(!frame_is_control(Api::Chat, Some("ping")));
    }
}
