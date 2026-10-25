//! Read-only access to the main conversation's original messages.
//!
//! Messages are numbered by [`Session::path_view`]: every message on the
//! active path, counted from the root. A compaction hides the prefix from the
//! model but never renumbers it, so a number cited before a compaction still
//! names the same message afterward. A main session with legacy topic threads
//! is numbered over its merged conversation ([`MainSession::history`]). Either
//! way the numbers match the main memory's leaves, so `id+1` in `<chat>` is
//! `recall {"id": id}`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use juto_agent::{Tool, ToolContext, ToolOutput};
use juto_ai::{ContentBlock, Message, ToolCall, ToolDefinition};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    main_session::MainSession,
    session::{CompactionMark, PathMessage, PathView, Session},
};

/// Byte budget of one page of list lines, and of one page of a message body.
const PAGE_BYTES: usize = 16 * 1024;
/// Character budget of the text on one list line.
const LINE_CHARS: usize = 160;
/// Character budget of a tool call's arguments on one list line.
const ARGS_CHARS: usize = 80;

const DESCRIPTION: &str = "Read the main conversation's original messages, including those summarized in <chat> or compacted. Without `id`, lists from `from` (default 0), up to `count` and about 16 KB; follow `more:`. With `id` alone, returns that message whole; `at` continues it by byte offset. Ids match <chat>'s.";

/// Arguments of one `recall` call. Every field is optional.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    from: Option<usize>,
    count: Option<usize>,
    #[serde(rename = "id")]
    index: Option<usize>,
    at: Option<usize>,
}

/// Read-only `recall` over the session file at `session`.
pub(crate) struct RecallTool {
    session: PathBuf,
}

impl RecallTool {
    pub(crate) fn new(session: PathBuf) -> Self {
        Self { session }
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall".into(),
            description: DESCRIPTION.into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "from": {"type": "integer", "minimum": 0, "description": "First index to list. Defaults to 0."},
                    "count": {"type": "integer", "minimum": 1, "description": "Most lines to list."},
                    "id": {"type": "integer", "minimum": 0, "description": "Return this message index in full."},
                    "at": {"type": "integer", "minimum": 0, "description": "Byte offset into the message body; needs id."}
                },
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: Value, _ctx: ToolContext) -> Result<ToolOutput, String> {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(args) => args,
            Err(error) => {
                return Ok(ToolOutput::error(format!(
                    "invalid recall arguments: {error}"
                )));
            }
        };
        let path = self.session.clone();
        tokio::task::spawn_blocking(move || read(&path, args))
            .await
            .map_err(|error| error.to_string())?
    }
}

/// The messages `recall` numbers.
pub(crate) enum History {
    /// A main session's conversation merged with its legacy topic threads.
    Merged(Vec<(String, Message)>),
    /// A session's own active path.
    Journal(Box<Session>),
}

impl History {
    pub(crate) fn open(path: &Path) -> anyhow::Result<Self> {
        let main = MainSession::open(path)?;
        Ok(if main.legacy_threads().next().is_none() {
            Self::Journal(Box::new(main.into_journal()))
        } else {
            Self::Merged(main.history()?)
        })
    }

    pub(crate) fn view(&self) -> anyhow::Result<PathView<'_>> {
        Ok(match self {
            Self::Merged(messages) => PathView {
                messages: messages
                    .iter()
                    .enumerate()
                    .map(|(index, (id, message))| PathMessage { index, id, message })
                    .collect(),
                nodes: Vec::new(),
                compaction: None,
            },
            Self::Journal(session) => session.path_view()?,
        })
    }
}

/// Answers one `recall` call from the session at `path`. Mistakes in the call
/// are tool errors; only a session that cannot be read is an `Err`.
fn read(path: &Path, args: Args) -> Result<ToolOutput, String> {
    // Models that fill every optional field send `id` and `at` alongside
    // `from`/`count`; paging the list is the request that still makes sense.
    let listing = args.from.is_some() || args.count.is_some();
    let ignored = listing && (args.index.is_some() || args.at.is_some());
    let (index, at) = if listing {
        (None, None)
    } else {
        (args.index, args.at)
    };
    if at.is_some() && index.is_none() {
        return Ok(ToolOutput::error(
            "`at` pages one message, so it needs `id`",
        ));
    }
    if args.count == Some(0) {
        return Ok(ToolOutput::error("`count` must be at least 1"));
    }
    let history = History::open(path).map_err(|error| format!("{error:#}"))?;
    let view = history.view().map_err(|error| format!("{error:#}"))?;
    let total = view.messages.len();
    if total == 0 {
        return Ok(ToolOutput::error("the main thread has no messages yet"));
    }
    let mut output = match index {
        Some(index) if index >= total => ToolOutput::error(format!(
            "index {index} is out of range; the main thread has messages 0..={}",
            total - 1
        )),
        Some(index) => detail(&view, index, at.unwrap_or(0)),
        None => list(&view, args.from.unwrap_or(0), args.count),
    };
    if ignored {
        output.content.insert(
            0,
            ContentBlock::Text {
                text: "(`id` and `at` ignored: `from`/`count` list messages)\n".into(),
            },
        );
    }
    Ok(output)
}

/// Message `index` of the conversation at `path` in full, from byte `at`.
pub(crate) fn read_message(path: &Path, index: usize, at: usize) -> Result<ToolOutput, String> {
    read(
        path,
        Args {
            index: Some(index),
            at: Some(at),
            ..Args::default()
        },
    )
}

/// Index lines from `from`, stopping at the page budget or after `count` lines.
fn list(view: &PathView<'_>, from: usize, count: Option<usize>) -> ToolOutput {
    let total = view.messages.len();
    if from >= total {
        return ToolOutput::error(format!(
            "from {from} is past the last message; the main thread has messages 0..={}",
            total - 1
        ));
    }
    let limit = count.map_or(total, |count| from.saturating_add(count).min(total));
    let mut lines = String::new();
    let mut next = from;
    while next < limit {
        let mut chunk = String::new();
        if let Some(mark) = view.compaction.filter(|mark| mark.at == next) {
            chunk.push_str(&marker(&mark));
        }
        chunk.push_str(&index_line(&view.messages[next]));
        chunk.push('\n');
        if next > from && lines.len() + chunk.len() > PAGE_BYTES {
            break;
        }
        lines.push_str(&chunk);
        next += 1;
    }
    if next == total
        && let Some(mark) = view.compaction.filter(|mark| mark.at == total)
    {
        lines.push_str(&marker(&mark));
    }
    let mut out = format!("main thread: messages {from}..={} of {total}\n", next - 1);
    out.push_str(&lines);
    if next < total {
        out.push_str(&format!("more: recall {{\"from\": {next}}}\n"));
    }
    ToolOutput::text(out)
}

/// One message in full, starting at byte `at` of its body.
fn detail(view: &PathView<'_>, index: usize, at: usize) -> ToolOutput {
    let entry = view.messages[index];
    let body = full_body(entry.message);
    let Some((start, end)) = page_bounds(&body, at) else {
        return ToolOutput::error(format!(
            "at {at} is past the end of message {index} ({} bytes)",
            body.len()
        ));
    };
    let mut out = format!(
        "message {index} ({}, id {})\n{}",
        role(entry.message),
        entry.id,
        &body[start..end]
    );
    if start > 0 || end < body.len() {
        out.push_str(&format!("\n[bytes {start}..{end} of {}]", body.len()));
        if end < body.len() {
            out.push_str(&format!(" more: recall {{\"id\": {index}, \"at\": {end}}}"));
        }
    }
    ToolOutput::text(out)
}

/// Byte range of the page of `body` that starts at `at`, moved onto char
/// boundaries; `None` when `at` is past the end.
fn page_bounds(body: &str, at: usize) -> Option<(usize, usize)> {
    if at > body.len() {
        return None;
    }
    let start = floor_boundary(body, at);
    let end = floor_boundary(body, start + PAGE_BYTES);
    Some((start, end))
}

pub(crate) fn floor_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// Line marking the compaction boundary: messages before `first_visible` are
/// summarized, those up to `at` are kept, and newer ones follow the marker.
fn marker(mark: &CompactionMark<'_>) -> String {
    format!(
        "-- compaction: 0..{} summarized; {}..{} kept --\n",
        mark.first_visible, mark.first_visible, mark.at
    )
}

/// `index|role: text` on one line, with the text flattened and clipped.
fn index_line(entry: &PathMessage<'_>) -> String {
    let text = match entry.message {
        Message::ToolResult {
            tool_name, content, ..
        } => format!("{tool_name}: {}", render(content, false)),
        Message::User { content, .. } | Message::Developer { content, .. } => {
            render(content, false)
        }
        Message::Assistant(message) => render(&message.content, false),
    };
    format!(
        "{}|{}: {}",
        entry.index,
        role(entry.message),
        clip_chars(&flatten(&text), LINE_CHARS)
    )
}

/// Everything in a message, for the detail view.
pub(crate) fn full_body(message: &Message) -> String {
    match message {
        Message::ToolResult {
            tool_name,
            tool_call_id,
            content,
            is_error,
            ..
        } => {
            let kind = if *is_error { "error result" } else { "result" };
            format!(
                "{kind} {tool_name} (call {tool_call_id})\n{}",
                render(content, true)
            )
        }
        Message::User { content, .. } | Message::Developer { content, .. } => render(content, true),
        Message::Assistant(message) => render(&message.content, true),
    }
}

/// Text of content blocks; thinking is never shown. `full` keeps tool-call
/// arguments whole and puts each block on its own line. Otherwise blocks run
/// together and tool-call arguments are clipped.
fn render(blocks: &[ContentBlock], full: bool) -> String {
    let parts: Vec<String> = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.clone()),
            ContentBlock::Image { mime_type, .. } => Some(format!("[image {mime_type}]")),
            ContentBlock::ToolCall(call) => Some(render_call(call, full)),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => None,
        })
        .collect();
    parts.join(if full { "\n" } else { " " })
}

fn render_call(call: &ToolCall, full: bool) -> String {
    if full {
        let arguments = serde_json::to_string_pretty(&call.arguments).unwrap_or_default();
        format!("tool call {} ({})\n{arguments}", call.name, call.id)
    } else {
        let arguments = serde_json::to_string(&call.arguments).unwrap_or_default();
        format!("tool:{}({})", call.name, clip_chars(&arguments, ARGS_CHARS))
    }
}

pub(crate) fn role(message: &Message) -> &'static str {
    match message {
        Message::User { .. } => "user",
        Message::Developer { .. } => "developer",
        Message::Assistant(_) => "assistant",
        Message::ToolResult { is_error: true, .. } => "result(error)",
        Message::ToolResult { .. } => "result",
    }
}

fn flatten(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// At most `max` characters; clipped text ends in an ellipsis.
fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut clipped: String = text.chars().take(max - 1).collect();
    clipped.push('…');
    clipped
}

/// Header of the snapshot a child starts with.
const CHILD_HEADER: &str = "Main thread snapshot (context only — do your task, not the user's last message)\nOriginal messages: recall {\"id\": N}; indices refer to the current main branch.\n";

/// Capture once per task call, not once per child: siblings share exact bytes.
pub(crate) fn snapshot(path: &Path) -> anyhow::Result<String> {
    let history = History::open(path)?;
    Ok(render_snapshot(CHILD_HEADER, &history.view()?))
}

/// `header`, then the view's summaries and its newest visible messages, in
/// about 12 KB.
fn render_snapshot(header: &str, view: &PathView<'_>) -> String {
    const BUDGET: usize = 12 * 1024;
    let mut out = String::from(header);
    let per_node = BUDGET / 2 / view.nodes.len().max(1);
    for node in &view.nodes {
        out.push_str("[Earlier conversation summary]\n");
        let end = floor_boundary(node.summary, per_node);
        out.push_str(&node.summary[..end]);
        if end < node.summary.len() {
            out.push_str("\n[summary clipped]");
        }
        out.push('\n');
    }
    // Reserve a small omission notice, then select a contiguous newest suffix.
    let budget = BUDGET - out.len() - 80;
    let mut lines = Vec::new();
    let mut bytes = 0;
    for entry in view.visible().iter().rev() {
        let line = index_line(entry);
        if bytes + line.len() + 1 > budget {
            break;
        }
        bytes += line.len() + 1;
        lines.push(line);
    }
    if lines.len() < view.visible().len() {
        out.push_str("[Older visible messages omitted; use recall]\n");
    }
    for line in lines.iter().rev() {
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
#[path = "recall_tests.rs"]
mod tests;
