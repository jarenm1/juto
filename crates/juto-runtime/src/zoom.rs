use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use juto_agent::{Tool, ToolContext, ToolOutput};
use juto_ai::ToolDefinition;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::recall::{History, floor_boundary, full_body, role};
use crate::session::{EntryKind, Session};
use crate::tools::OUTPUT_MAX_BYTES;

/// Bytes kept back from the output cap for the span header and the `more:` hint.
const RESERVE_BYTES: usize = 1024;
/// Message budget of one page.
const PAGE_BYTES: usize = OUTPUT_MAX_BYTES - RESERVE_BYTES;

const DESCRIPTION: &str = "Expand a main-thread compaction summary back to the original messages it replaced. `id` is the compaction id named in the `[Earlier conversation summary id=...]` marker. Returns the span's messages in order, each with its `recall` index, up to about 50 KB; follow the `more:` hint with `offset` to continue. A single message longer than a page is cut and points at `recall`. Read-only; never calls a model.";

/// Arguments of one `zoom` call.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    id: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

/// Read-only `zoom` over the compactions of the session file at `session`.
/// Indices are `recall`'s, numbered over `history`.
pub(crate) struct ZoomTool {
    session: PathBuf,
    history: PathBuf,
}

impl ZoomTool {
    pub(crate) fn new(session: PathBuf, history: PathBuf) -> Self {
        Self { session, history }
    }
}

#[async_trait]
impl Tool for ZoomTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "zoom".into(),
            description: DESCRIPTION.into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Compaction entry id from the summary marker."},
                    "offset": {"type": "integer", "minimum": 0, "description": "First message of the span to return, counted within the span. Defaults to 0."},
                    "limit": {"type": "integer", "minimum": 1, "description": "Most messages to return."}
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: Value, _ctx: ToolContext) -> Result<ToolOutput, String> {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(args) => args,
            Err(error) => {
                return Ok(ToolOutput::error(format!(
                    "invalid zoom arguments: {error}"
                )));
            }
        };
        let path = self.session.clone();
        let history = self.history.clone();
        tokio::task::spawn_blocking(move || read(&path, &history, args))
            .await
            .map_err(|error| error.to_string())?
    }
}

/// Answers one `zoom` call from the session at `path`, citing `recall`
/// indices over `history`. Mistakes in the call are tool errors; only a
/// session that cannot be read is an `Err`.
fn read(path: &Path, history: &Path, args: Args) -> Result<ToolOutput, String> {
    if args.limit == Some(0) {
        return Ok(ToolOutput::error("`limit` must be at least 1"));
    }
    let session = Session::open(path).map_err(|error| error.to_string())?;
    let entries = session
        .active_entries()
        .map_err(|error| error.to_string())?;
    let Some(entry) = entries.into_iter().find(|entry| entry.id == args.id) else {
        return Ok(ToolOutput::error(format!(
            "no entry {} on the main thread's current branch",
            args.id
        )));
    };
    let (start, end) = match &entry.kind {
        EntryKind::Compaction {
            span_start_id: Some(start),
            span_end_id: Some(end),
            ..
        } => (start.as_str(), end.as_str()),
        EntryKind::Compaction { .. } => {
            return Ok(ToolOutput::error(format!(
                "compaction {} recorded no span: it replaced no messages, or it predates span recording",
                args.id
            )));
        }
        _ => {
            return Ok(ToolOutput::error(format!(
                "entry {} is not a compaction",
                args.id
            )));
        }
    };
    let span = match session.entries_in_span(start, end) {
        Ok(span) => span,
        Err(error) => return Ok(ToolOutput::error(error.to_string())),
    };
    let history = History::open(history).map_err(|error| format!("{error:#}"))?;
    let view = history.view().map_err(|error| format!("{error:#}"))?;
    let index_of: HashMap<&str, usize> = view
        .messages
        .iter()
        .map(|message| (message.id, message.index))
        .collect();
    let mut spanned = Vec::with_capacity(span.len());
    for &entry in &span {
        let EntryKind::Message { message } = &entry.kind else {
            return Ok(ToolOutput::error(format!(
                "span of {} holds a non-message entry",
                args.id
            )));
        };
        // Context the runtime seeded into a thread has no `recall` index.
        let index = index_of.get(entry.id.as_str()).copied();
        spanned.push((index, entry.id.as_str(), message));
    }

    let total = spanned.len();
    let offset = args.offset.unwrap_or(0);
    if offset >= total {
        return Ok(ToolOutput::error(format!(
            "offset {offset} is past the span; compaction {} replaced messages 0..={}",
            args.id,
            total.saturating_sub(1)
        )));
    }
    let limit = args
        .limit
        .map_or(total, |limit| offset.saturating_add(limit).min(total));
    let mut body = String::new();
    let mut next = offset;
    while next < limit {
        let (index, id, message) = spanned[next];
        let header = match index {
            Some(index) => format!("message {index} ({}, id {id})\n", role(message)),
            None => format!("context ({}, id {id})\n", role(message)),
        };
        let text = full_body(message);
        if body.len() + header.len() + text.len() < PAGE_BYTES {
            body.push_str(&header);
            body.push_str(&text);
            body.push('\n');
            next += 1;
        } else {
            if next == offset {
                // One message larger than a page: cut it here; `recall` reads the rest.
                let room = PAGE_BYTES.saturating_sub(body.len() + header.len());
                let cut = floor_boundary(&text, room);
                body.push_str(&header);
                body.push_str(&text[..cut]);
                body.push_str(&format!("\n[bytes 0..{cut} of {}]", text.len()));
                if let Some(index) = index {
                    body.push_str(&format!(" more: recall {{\"id\": {index}, \"at\": {cut}}}"));
                }
                body.push('\n');
                next += 1;
            }
            break;
        }
    }

    let mut out = format!(
        "span of compaction {}: messages {offset}..={} of {total}\n",
        args.id,
        next - 1
    );
    out.push_str(&body);
    if next < total {
        out.push_str(&format!(
            "more: zoom {}\n",
            json!({"id": args.id, "offset": next})
        ));
    }
    Ok(ToolOutput::text(out))
}

#[cfg(test)]
#[path = "zoom_tests.rs"]
mod tests;
