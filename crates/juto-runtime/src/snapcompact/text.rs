//! Text preparation for archive frames: transcript serialization and glyph normalization.

use std::collections::{HashMap, HashSet};

use juto_ai::{ContentBlock, Message, ToolCall};
use serde_json::Value;

use super::font::{self, Font};

/// Stands in for a line break in archived text: one solid cell.
pub const NEWLINE_GLYPH: char = '█';

const TOOL_RESULT_MAX_CHARS: usize = 2000;
const TOOL_ARG_MAX_CHARS: usize = 500;
const TOOL_CALL_MAX_CHARS: usize = 2000;
/// Share of a truncated text's budget kept from its start; the rest comes from its end.
const HEAD_RATIO: f64 = 0.6;

const USER: &str = "¶user: ";
const ASSISTANT: &str = "¶ai: ";
const CALL: &str = "¶call: ";
const NOTE: &str = "¶note: ";

/// Serializes messages oldest-first as `¶`-marked blocks. Thinking is omitted.
/// Each tool output is attached to its call and truncated.
pub fn serialize_conversation(messages: &[Message]) -> String {
    let outputs = tool_outputs(messages);
    let mut attached = HashSet::new();
    let mut transcript = Transcript::default();
    for message in messages {
        match message {
            Message::User { content, .. } => push_body(&mut transcript, USER, content),
            Message::Developer { content, .. } => push_body(&mut transcript, NOTE, content),
            Message::Assistant(assistant) => {
                let mut pending = String::new();
                for block in &assistant.content {
                    match block {
                        ContentBlock::Text { text } => pending.push_str(text),
                        ContentBlock::ToolCall(call) => {
                            flush_text(&mut transcript, &mut pending);
                            let mut line = call_line(call);
                            if let Some(output) = outputs.get(call.id.as_str()) {
                                line = format!(
                                    "{line}\n<out>\n{}\n</out>",
                                    truncate(output, TOOL_RESULT_MAX_CHARS)
                                );
                                attached.insert(call.id.as_str());
                            }
                            transcript.push(CALL, &line);
                        }
                        _ => {}
                    }
                }
                flush_text(&mut transcript, &mut pending);
            }
            Message::ToolResult { tool_call_id, .. } => {
                if attached.contains(tool_call_id.as_str()) {
                    continue;
                }
                if let Some(output) = outputs.get(tool_call_id.as_str()) {
                    let body =
                        format!("<out>\n{}\n</out>", truncate(output, TOOL_RESULT_MAX_CHARS));
                    transcript.push(CALL, &body);
                }
            }
        }
    }
    let files = file_operations(messages);
    let mut result = transcript.render();
    if !files.is_empty() {
        result.push_str("\n\nFILES (tool-call paths, not confirmation of success):\n");
        result.push_str(&files.join("\n"));
    }
    result
}

fn file_operations(messages: &[Message]) -> Vec<String> {
    let mut files = std::collections::BTreeSet::new();
    for message in messages {
        if let Message::Assistant(assistant) = message {
            for call in assistant.tool_calls() {
                let operation = match call.name.as_str() {
                    "read" => "read",
                    "write" | "edit" | "apply_patch" => "modified",
                    _ => continue,
                };
                for key in ["path", "file_path", "file"] {
                    if let Some(path) = call.arguments.get(key).and_then(Value::as_str) {
                        files.insert(format!("{operation}: {path}"));
                    }
                }
            }
        }
    }
    files.into_iter().collect()
}

/// Output text keyed by tool-call id. Empty outputs are dropped.
fn tool_outputs(messages: &[Message]) -> HashMap<&str, String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                let text = block_text(content);
                (!text.trim().is_empty()).then_some((tool_call_id.as_str(), text))
            }
            _ => None,
        })
        .collect()
}

fn block_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Image { .. } => Some("[image]"),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn push_body(transcript: &mut Transcript, prefix: &'static str, content: &[ContentBlock]) {
    let body = block_text(content);
    if !body.trim().is_empty() {
        transcript.push(prefix, &body);
    }
}

fn flush_text(transcript: &mut Transcript, pending: &mut String) {
    let text = pending.trim();
    if !text.is_empty() {
        transcript.push(ASSISTANT, text);
    }
    pending.clear();
}

/// `name(key=value, ...)//intent`. Each value is capped; the whole line is capped.
fn call_line(call: &ToolCall) -> String {
    let mut intent = String::new();
    let args = match &call.arguments {
        Value::Object(map) => map
            .iter()
            .filter_map(|(key, value)| {
                if key == "intent" {
                    if let Some(text) = value.as_str() {
                        intent = format!("//{text}");
                    }
                    return None;
                }
                Some(format!(
                    "{key}={}",
                    truncate(&value.to_string(), TOOL_ARG_MAX_CHARS)
                ))
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => truncate(&other.to_string(), TOOL_ARG_MAX_CHARS),
    };
    truncate(
        &format!("{}({args}){intent}", call.name),
        TOOL_CALL_MAX_CHARS,
    )
}

/// Keeps the start and end of text longer than `max` characters, with an elision marker between.
pub fn truncate(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        return text.to_owned();
    }
    let head = (max as f64 * HEAD_RATIO).round() as usize;
    let tail = max - head;
    let head_text: String = chars[..head].iter().collect();
    let tail_text: String = chars[chars.len() - tail..].iter().collect();
    format!("{head_text} […{}ch elided…] {tail_text}", chars.len() - max)
}

#[derive(Default)]
struct Transcript {
    parts: Vec<(&'static str, String)>,
}

impl Transcript {
    /// Appends `body` under `prefix`. Consecutive blocks with the same prefix share one marker.
    fn push(&mut self, prefix: &'static str, body: &str) {
        match self.parts.last_mut() {
            Some((last, text)) if *last == prefix => {
                if !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(body);
            }
            _ => self.parts.push((prefix, body.to_owned())),
        }
    }

    fn render(&self) -> String {
        self.parts
            .iter()
            .map(|(prefix, body)| format!("{prefix}{body}"))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Makes `text` renderable with the bundled font. Whitespace runs become one space, or one
/// `NEWLINE_GLYPH` when the run contains a line break. Common symbols fold to ASCII;
/// unsupported Unicode is escaped by code point so identifiers never silently become `?`.
pub fn normalize(text: &str) -> String {
    let font = font::bundled();
    let stripped = strip_ansi(text);
    let mut out = String::with_capacity(stripped.len());
    let mut run: Option<char> = None;
    for ch in stripped.chars() {
        if ch.is_whitespace() {
            let mark = if is_line_break(ch) {
                NEWLINE_GLYPH
            } else {
                ' '
            };
            run = Some(if run == Some(NEWLINE_GLYPH) {
                NEWLINE_GLYPH
            } else {
                mark
            });
            continue;
        }
        if let Some(mark) = run.take() {
            out.push(mark);
        }
        push_cell(&mut out, ch, font);
    }
    out.trim_matches(|c| c == ' ' || c == NEWLINE_GLYPH)
        .to_owned()
}

fn push_cell(out: &mut String, ch: char, font: &Font) {
    if ch == NEWLINE_GLYPH || font.glyph(ch).is_some() {
        out.push(ch);
    } else if let Some(folded) = fold(ch) {
        out.push_str(&folded);
    } else if !ch.is_control() {
        use std::fmt::Write;
        write!(out, "\\u{{{:X}}}", ch as u32).expect("writing to a string");
    }
}

fn is_line_break(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{2028}' | '\u{2029}'
    )
}

/// ASCII stand-ins for characters outside the bundled font.
fn fold(ch: char) -> Option<String> {
    let text = match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' => "'",
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => "\"",
        '\u{2010}'..='\u{2015}' | '\u{2212}' => "-",
        '\u{2026}' => "...",
        '\u{2022}' | '\u{2023}' | '\u{25E6}' => "*",
        '\u{2190}' | '\u{21D0}' => "<-",
        '\u{2192}' | '\u{21D2}' | '\u{27F6}' => "->",
        '\u{2194}' | '\u{21D4}' => "<->",
        '\u{2248}' => "~",
        '\u{2260}' => "!=",
        '\u{2264}' => "<=",
        '\u{2265}' => ">=",
        '\u{2713}' | '\u{2714}' | '\u{2611}' | '\u{2705}' => "[OK]",
        '\u{2717}' | '\u{2718}' | '\u{2612}' | '\u{2716}' | '\u{274C}' | '\u{274E}' => "[FAIL]",
        '\u{26A0}' => "[WARN]",
        '\u{2139}' => "[INFO]",
        '\u{1F680}' => "[RUN]",
        '\u{1F6A8}' => "[ALERT]",
        '\u{2502}' | '\u{2503}' | '\u{2551}' => "|",
        '\u{2500}' | '\u{2501}' | '\u{2550}' => "-",
        '\u{2500}'..='\u{257F}' => "+",
        '\u{FF01}'..='\u{FF5E}' => return char::from_u32(ch as u32 - 0xFEE0).map(String::from),
        _ => return None,
    };
    Some(text.to_owned())
}

/// Removes CSI, OSC, and two-character escape sequences.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('[') => {
                for next in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&next) {
                        break;
                    }
                }
            }
            Some(']') => {
                let mut prev = '\0';
                for next in chars.by_ref() {
                    if next == '\u{07}' || (prev == '\u{1b}' && next == '\\') {
                        break;
                    }
                    prev = next;
                }
            }
            Some('(' | ')' | '*' | '+') => {
                chars.next();
            }
            _ => {}
        }
    }
    out
}
