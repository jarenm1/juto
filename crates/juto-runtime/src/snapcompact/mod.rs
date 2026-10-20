//! Snapcompact: archives compacted history as bitmap frames that the model reads directly,
//! instead of replacing it with an LLM-written summary.
//!
//! Ported from oh-my-pi's snapcompact (MIT; see `licenses/OMP-MIT.txt`). Scope of this port:
//! 1-bit black-on-white ink, a single-column grid, and the ASCII and Latin-1 subset of the
//! public-domain X.org `misc-fixed` 8x13 font embedded in `font.rs`. Other characters are
//! folded or escaped before rendering (see `text::normalize`).

mod font;
mod render;
mod text;

#[cfg(test)]
mod tests;

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

pub use render::Shape;
pub use text::serialize_conversation;

use render::render_page;
use text::{NEWLINE_GLYPH, normalize};

/// Frame ceiling before provider and context-window limits apply.
const MAX_FRAMES: usize = 80;
/// An archive may use at most this fraction of the context window.
const WINDOW_SHARE_DIVISOR: u64 = 4;

/// Compacted history stored in the session journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Archive {
    /// Normalized transcript, oldest first, with `█` for line breaks. Kept so a later
    /// compaction can re-paginate it together with newer history.
    pub source: String,
    /// PNG pages, oldest first.
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    /// Base64-encoded PNG.
    pub png: String,
}

impl Archive {
    /// The archived transcript as plain text, with line breaks restored.
    pub fn text(&self) -> String {
        self.source.replace(NEWLINE_GLYPH, "\n")
    }
}

/// Frames an archive may use: the provider's image limit, `MAX_FRAMES`, and a share of the
/// context window. Unknown providers get a conservative limit of 5.
pub fn frame_budget(provider: &str, context_window: Option<u64>) -> usize {
    let provider_limit = match provider {
        "anthropic" | "amazon-bedrock" | "openrouter" => 90,
        "openai" | "openai-codex" => 200,
        p if p.starts_with("google") => 200,
        _ => 5,
    };
    let window_limit = context_window.map_or(MAX_FRAMES, |window| {
        ((window / WINDOW_SHARE_DIVISOR) / juto_agent::IMAGE_TOKEN_ESTIMATE) as usize
    });
    provider_limit.min(MAX_FRAMES).min(window_limit)
}

/// Appends history and renders it only if the entire normalized source fits the budget.
/// Returning `None` requests LLM summarization rather than dropping archived history.
pub fn build_archive(
    previous: Option<&Archive>,
    transcript: &str,
    shape: Shape,
    max_frames: usize,
) -> Result<Option<Archive>> {
    let mut source = previous.map_or_else(String::new, |archive| archive.source.clone());
    let fresh = normalize(transcript);
    if !fresh.is_empty() {
        if !source.is_empty() {
            source.push(NEWLINE_GLYPH);
        }
        source.push_str(&fresh);
    }

    let cells: Vec<char> = source.chars().collect();
    if cells.len() > max_frames.saturating_mul(shape.capacity()) {
        return Ok(None);
    }

    let frames = cells
        .chunks(shape.capacity())
        .map(|page| {
            render_page(page, shape).map(|png| Frame {
                png: STANDARD.encode(png),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(Archive { source, frames }))
}

/// The text placed before an archive's frames: reading guide and file operations.
pub fn lead_in(archive: &Archive, shape: Shape) -> String {
    let mut paragraphs = vec![
        format!(
            "Resume prior conversation. The earlier turns are archived in the {} image(s) that follow, oldest first. Read them as the conversation so far, then continue from the most recent turn.",
            archive.frames.len()
        ),
        "Markers: `¶user:` user message; `¶ai:` assistant text (reasoning is not archived); `¶call:` tool call as name(args)//intent, followed by its <out>…</out> output; `¶note:` other context, such as an earlier summary.".to_owned(),
        format!(
            "Layout: each image holds up to {} characters per row and {} rows, read left to right, top to bottom. A solid black cell is a line break. Words are not wrapped and may split across rows.",
            shape.cols(),
            shape.rows()
        ),
    ];
    let files: std::collections::BTreeSet<_> = archive
        .text()
        .lines()
        .filter(|line| line.starts_with("read: ") || line.starts_with("modified: "))
        .map(str::to_owned)
        .collect();
    if !files.is_empty() {
        paragraphs.push(format!(
            "FILES (tool-call paths; not proof of success)\n{}",
            files.into_iter().collect::<Vec<_>>().join("\n")
        ));
    }
    paragraphs.push(
        "If an exact earlier detail matters and the images are unclear, re-derive it from the workspace instead of guessing."
            .to_owned(),
    );
    paragraphs.join("\n\n")
}
