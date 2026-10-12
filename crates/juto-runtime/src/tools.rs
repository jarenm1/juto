//! Standard file/shell tools: read, write, edit, glob, grep, bash.
//!
//! Behavioral reference: oh-my-pi `packages/coding-agent/src/tools`
//! (read.ts, write.ts, edit/, glob.ts, grep.ts, bash.ts) at
//! 579da1d661c5cb8d43bc2ddd429ab72e67165ad8; see licenses/OMP-MIT.txt.
//!
//! Approvals are enforced by agent hooks via [`Tool::tier`]; these tools
//! perform the real operation and return true errors. Approval is not a
//! sandbox: absolute paths are allowed, matching source behavior.
//!
//! Requires the `libc` crate (workspace dependency) for process-group kill
//! on Unix; see `kill_process_group`.

use std::{
    collections::VecDeque,
    io::{self, BufRead, Read as _},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use globset::{Candidate, GlobBuilder};
use juto_agent::{
    AgentError, Tool, ToolConcurrency, ToolContext, ToolOutput, ToolRegistry, ToolTier,
};
use juto_ai::ContentBlock;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

use crate::change_graph::{ChangeGraph, EditProvenance, JoinOutcome, ReconcileOutcome};
use crate::workspace::{ResourceLimits, ScopedChild, ScopedStdio, SubagentWorkspace};

// ---------------------------------------------------------------------------
// Bounds (source: streaming-output.ts, glob.ts, grep.ts, tool-timeouts.ts)
// ---------------------------------------------------------------------------

/// Default line window for `read`.
const READ_DEFAULT_LIMIT: usize = 2000;
/// Hard cap on lines returned by `read` (DEFAULT_MAX_LINES is 3000; the read
/// window is capped lower like the source default).
const READ_MAX_LINES: usize = 2000;
/// Inline byte cap for file output (DEFAULT_MAX_BYTES upstream).
const OUTPUT_MAX_BYTES: usize = 50 * 1024;
/// Per match-line byte cap (DEFAULT_MAX_COLUMN upstream).
const MATCH_LINE_MAX_BYTES: usize = 512;
/// Directory listing cap.
const DIR_LIST_LIMIT: usize = 2000;
/// Default glob result count (DEFAULT_LIMIT upstream).
const GLOB_DEFAULT_LIMIT: usize = 200;
/// Hard glob cap. Upstream hard-caps at 200; we allow a larger explicit page
/// while still bounding every call.
const GLOB_MAX_LIMIT: usize = 2000;
/// Files surfaced per `grep` response (DEFAULT_FILE_LIMIT upstream).
const GREP_FILE_LIMIT: usize = 20;
/// Per-file match cap in multi-file searches (MULTI_FILE_PER_FILE_MATCHES).
const GREP_PER_FILE_MATCHES: usize = 20;
/// Per-file match cap when `path` names a single file (SINGLE_FILE_MATCHES).
const GREP_SINGLE_FILE_MATCHES: usize = 200;
/// Only the first bytes of a larger file are searched (NATIVE_GREP_MAX_FILE_BYTES).
const GREP_MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
/// Prefix inspected to classify a file as binary.
const BINARY_SNIFF_BYTES: usize = 8192;
/// Bash timeout bounds in seconds (TOOL_TIMEOUTS.bash upstream).
const BASH_DEFAULT_TIMEOUT: f64 = 300.0;
const BASH_MIN_TIMEOUT: f64 = 1.0;
const BASH_MAX_TIMEOUT: f64 = 3600.0;
/// Rolling tail retained from merged bash stdout/stderr.
const BASH_OUTPUT_MAX_BYTES: usize = 50 * 1024;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn output(text: String, details: Value) -> ToolOutput {
    ToolOutput {
        content: vec![ContentBlock::Text { text }],
        details,
        is_error: false,
    }
}

fn error_output(text: String, details: Value) -> ToolOutput {
    ToolOutput {
        content: vec![ContentBlock::Text { text }],
        details,
        is_error: true,
    }
}

fn check_cancel(cancel: &CancellationToken) -> Result<(), String> {
    if cancel.is_cancelled() {
        Err("cancelled".to_string())
    } else {
        Ok(())
    }
}

/// Resolve `raw` against `base`; absolute paths are used verbatim.
fn resolve_path(base: &Path, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(raw)
    }
}

fn effective_cwd<'a>(tool_cwd: &'a Path, ctx: &'a ToolContext) -> &'a Path {
    if ctx.cwd.as_os_str().is_empty() {
        tool_cwd
    } else {
        ctx.cwd.as_path()
    }
}

/// Best-effort relative display path: relative to `cwd` when possible.
fn display_path(cwd: &Path, path: &Path) -> String {
    match path.strip_prefix(cwd) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().into_owned(),
        _ => path.to_string_lossy().into_owned(),
    }
}

/// Read up to `max_bytes` from a file, returning the prefix and whether the
/// file is larger.
fn read_file_prefix(path: &Path, max_bytes: u64) -> io::Result<(Vec<u8>, bool)> {
    let file = std::fs::File::open(path)?;
    let total = file.metadata()?.len();
    let mut buf = Vec::with_capacity(total.min(max_bytes) as usize);
    let mut take = file.take(max_bytes);
    take.read_to_end(&mut buf)?;
    Ok((buf, total > max_bytes))
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes[..bytes.len().min(BINARY_SNIFF_BYTES)].contains(&0)
}

/// Truncate a string to `max` bytes at a char boundary, flagging truncation.
fn truncate_bytes(text: &str, max: usize) -> (String, bool) {
    if text.len() <= max {
        return (text.to_string(), false);
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

// ---------------------------------------------------------------------------
// read
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ReadParams {
    path: String,
    /// 1-based first line to return.
    #[serde(default = "one")]
    offset: usize,
    #[serde(default)]
    limit: Option<usize>,
}

fn one() -> usize {
    1
}

struct ReadTool {
    cwd: PathBuf,
}

impl ReadTool {
    /// List a directory: sorted entries, directories suffixed with `/`.
    fn read_dir(path: &Path, cwd: &Path, cancel: &CancellationToken) -> Result<ToolOutput, String> {
        let mut entries: Vec<String> = Vec::new();
        let read_dir =
            std::fs::read_dir(path).map_err(|e| format!("{}: {e}", display_path(cwd, path)))?;
        for entry in read_dir {
            let entry = entry.map_err(|e| e.to_string())?;
            let file_type = entry.file_type().map_err(|e| e.to_string())?;
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if file_type.is_dir() {
                name.push('/');
            }
            entries.push(name);
        }
        check_cancel(cancel)?;
        entries.sort();
        let total = entries.len();
        let truncated = total > DIR_LIST_LIMIT;
        entries.truncate(DIR_LIST_LIMIT);
        let mut text = entries.join("\n");
        if truncated {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&format!(
                "... ({total} entries; listing truncated at {DIR_LIST_LIMIT})"
            ));
        }
        Ok(output(
            text,
            json!({
                "path": display_path(cwd, path),
                "kind": "directory",
                "entries": total,
                "truncated": truncated,
            }),
        ))
    }

    /// Stream the file once, counting lines and retaining the requested window.
    /// Memory stays O(window) regardless of file size.
    fn read_file(
        path: &Path,
        cwd: &Path,
        params: &ReadParams,
        cancel: &CancellationToken,
    ) -> Result<ToolOutput, String> {
        let shown = display_path(cwd, path);
        let limit = params
            .limit
            .map_or(READ_DEFAULT_LIMIT, |l| l.clamp(1, READ_MAX_LINES));
        let start = params.offset.max(1);
        let mut file = std::fs::File::open(path).map_err(|e| format!("{shown}: {e}"))?;
        let mut prefix = [0u8; BINARY_SNIFF_BYTES];
        let prefix_len = file
            .read(&mut prefix)
            .map_err(|e| format!("{shown}: {e}"))?;
        if prefix[..prefix_len].contains(&0) {
            return Err(format!(
                "{shown}: binary file (NUL byte in first {BINARY_SNIFF_BYTES} bytes)"
            ));
        }
        let cursor = io::Cursor::new(prefix[..prefix_len].to_vec());
        let reader = io::BufReader::new(std::io::Read::chain(cursor, file));

        let mut total_lines = 0usize;
        let mut window: Vec<String> = Vec::new();
        let mut window_bytes = 0usize;
        let mut byte_limited = false;
        for line in reader.lines() {
            let line = line.map_err(|e| {
                if e.kind() == io::ErrorKind::InvalidData {
                    format!("{shown}: not valid UTF-8")
                } else {
                    format!("{shown}: {e}")
                }
            })?;
            total_lines += 1;
            if total_lines >= start && total_lines < start + limit {
                if window_bytes + line.len() + 8 <= OUTPUT_MAX_BYTES {
                    window_bytes += line.len() + 8;
                    window.push(line);
                } else {
                    byte_limited = true;
                }
            }
            if total_lines.is_multiple_of(65536) {
                check_cancel(cancel)?;
            }
        }
        if start > 1 && start > total_lines {
            return Err(format!(
                "{shown}: offset {start} beyond end of file ({total_lines} lines)"
            ));
        }
        let line_limited = total_lines > start - 1 + window.len();
        let truncated = byte_limited || line_limited;
        let mut body = String::new();
        for (i, line) in window.iter().enumerate() {
            body.push_str(&format!("{}: {line}\n", start + i));
        }
        if truncated {
            if window.is_empty() {
                body.push_str(&format!(
                    "... ({total_lines} lines total; no lines fit the byte cap at offset {start})\n"
                ));
            } else {
                let shown_last = start + window.len() - 1;
                body.push_str(&format!("... ({total_lines} lines total; showing {start}-{shown_last}; pass offset for more)\n"));
            }
        }
        Ok(output(
            body,
            json!({
                "path": shown,
                "kind": "file",
                "offset": start,
                "lines": window.len(),
                "totalLines": total_lines,
                "truncated": truncated,
            }),
        ))
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "read".to_string(),
            description: "Read a file as numbered lines (offset/limit windowed, bounded output) \
				or list a directory. Relative paths resolve against the session cwd; absolute \
				paths are allowed."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory path; relative paths resolve against the session cwd."},
                    "offset": {"type": "integer", "minimum": 1, "description": "1-based first line to return (default 1)."},
                    "limit": {"type": "integer", "minimum": 1, "description": format!("Maximum lines to return (default {READ_DEFAULT_LIMIT}, max {READ_MAX_LINES}).")},
                },
                "required": ["path"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Read
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: ReadParams = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid read arguments: {e}"))?;
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let path = resolve_path(&cwd, &params.path);
        let cancel = ctx.cancel.clone();
        tokio::task::spawn_blocking(move || {
            let meta = std::fs::metadata(&path)
                .map_err(|e| format!("{}: {e}", display_path(&cwd, &path)))?;
            if meta.is_dir() {
                Self::read_dir(&path, &cwd, &cancel)
            } else if meta.is_file() {
                Self::read_file(&path, &cwd, &params, &cancel)
            } else {
                Err(format!(
                    "{}: not a regular file or directory",
                    display_path(&cwd, &path)
                ))
            }
        })
        .await
        .map_err(|e| format!("read task failed: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// write
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WriteParams {
    path: String,
    content: String,
}

struct WriteTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for WriteTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "write".to_string(),
            description: "Write a UTF-8 file through the native Git workspace, creating parents. \
                Requires a path inside the session's workspace. Returns before/after \
                snapshot trees and the operation's diff; records a durable change-graph node."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination path; relative paths resolve against the session cwd."},
                    "content": {"type": "string", "description": "Exact bytes to write (UTF-8 text)."},
                },
                "required": ["path", "content"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Write
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: WriteParams = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid write arguments: {e}"))?;
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let path = resolve_path(&cwd, &params.path);
        let cancel = ctx.cancel.clone();
        let provenance = crate::change_graph::EditProvenance {
            session_id: ctx.session_id,
            tool: "write".to_string(),
            selection_mode: None,
            path: Some(params.path.clone()),
        };
        tokio::task::spawn_blocking(move || {
            let details =
                crate::vcs::mutate(&cwd, &path, &cancel, &provenance, |_| Ok(params.content))?;
            Ok(output(
                format!("Wrote {}", display_path(&cwd, &path)),
                details,
            ))
        })
        .await
        .map_err(|e| format!("write task failed: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// edit
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditParams {
    path: String,
    selection: crate::contextual::Selection,
    replacement: String,
}

struct EditTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for EditTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "edit".to_string(),
            description: "Contextual edit through the native Git workspace. Prefer AST queries \
                for supported Rust, Python, JavaScript and TypeScript files; use inclusive line \
                ranges or exact text otherwise. AST query must uniquely capture @edit (or the \
                supplied capture). Replacement is verbatim, including newlines. Returns snapshot \
                trees and the operation's diff; records a durable change-graph node; never \
                silently falls back between modes."
                .to_string(),
            parameters: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": {"type": "string"},
                    "replacement": {"type": "string"},
                    "selection": {
                        "oneOf": [
                            {"type": "object", "additionalProperties": false, "properties": {
                                "mode": {"const": "ast"}, "query": {"type": "string"},
                                "capture": {"type": "string", "default": "edit"}
                            }, "required": ["mode", "query"]},
                            {"type": "object", "additionalProperties": false, "properties": {
                                "mode": {"const": "range"}, "start_line": {"type": "integer", "minimum": 1},
                                "end_line": {"type": "integer", "minimum": 1}
                            }, "required": ["mode", "start_line", "end_line"]},
                            {"type": "object", "additionalProperties": false, "properties": {
                                "mode": {"const": "text"}, "old_string": {"type": "string", "minLength": 1},
                                "replace_all": {"type": "boolean", "default": false}
                            }, "required": ["mode", "old_string"]}
                        ]
                    }
                },
                "required": ["path", "selection", "replacement"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Write
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: EditParams = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid edit arguments: {e}"))?;
        let cancel = ctx.cancel.clone();
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let path = resolve_path(&cwd, &params.path);
        let mode = match &params.selection {
            crate::contextual::Selection::Ast { .. } => "ast",
            crate::contextual::Selection::Range { .. } => "range",
            crate::contextual::Selection::Text { .. } => "text",
        };
        let provenance = crate::change_graph::EditProvenance {
            session_id: ctx.session_id,
            tool: "edit".to_string(),
            selection_mode: Some(mode.to_string()),
            path: Some(params.path.clone()),
        };
        tokio::task::spawn_blocking(move || {
            let mut replacements = 0;
            let mut details = crate::vcs::mutate(&cwd, &path, &cancel, &provenance, |original| {
                let text = original.ok_or_else(|| "edit: file does not exist".to_string())?;
                let ranges = crate::contextual::resolve(&path, text, &params.selection)?;
                replacements = ranges.len();
                let mut updated = String::with_capacity(text.len());
                let mut cursor = 0;
                for range in ranges {
                    updated.push_str(&text[cursor..range.start]);
                    updated.push_str(&params.replacement);
                    cursor = range.end;
                }
                updated.push_str(&text[cursor..]);
                Ok(updated)
            })?;
            details["replacements"] = json!(replacements);
            Ok(output(
                format!(
                    "Edited {} ({replacements} replacements)",
                    display_path(&cwd, &path)
                ),
                details,
            ))
        })
        .await
        .map_err(|e| format!("edit task failed: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// glob
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GlobParams {
    pattern: String,
    /// Search root; relative paths resolve against the session cwd.
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    hidden: bool,
    #[serde(default = "yes")]
    gitignore: bool,
    #[serde(default)]
    limit: Option<usize>,
}

fn yes() -> bool {
    true
}

struct GlobTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for GlobTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "glob".to_string(),
            description: "Find paths by glob pattern under a directory, honoring .gitignore. \
				Patterns without a slash match file names at any depth; patterns with a slash \
				match relative paths (* stays within a segment, ** crosses directories). \
				Results are sorted and bounded."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. `**/*.rs` or `*.toml`."},
                    "path": {"type": "string", "description": "Search root directory (default session cwd)."},
                    "hidden": {"type": "boolean", "description": "Include hidden files (default false)."},
                    "gitignore": {"type": "boolean", "description": "Respect ignore files (default true)."},
                    "limit": {"type": "integer", "minimum": 1, "description": format!("Maximum results (default {GLOB_DEFAULT_LIMIT}, max {GLOB_MAX_LIMIT}).")},
                },
                "required": ["pattern"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Read
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: GlobParams = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid glob arguments: {e}"))?;
        let glob = GlobBuilder::new(&params.pattern)
            .literal_separator(true)
            .build()
            .map_err(|e| format!("invalid glob pattern {:?}: {e}", params.pattern))?
            .compile_matcher();
        // Basename-only patterns match file names at any depth; path patterns
        // match the path relative to the search root.
        let basename_mode = !params.pattern.contains('/');
        let limit = params
            .limit
            .map_or(GLOB_DEFAULT_LIMIT, |l| l.clamp(1, GLOB_MAX_LIMIT));
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let root = params
            .path
            .as_deref()
            .map_or_else(|| cwd.clone(), |p| resolve_path(&cwd, p));
        let cancel = ctx.cancel.clone();
        tokio::task::spawn_blocking(move || {
            if !root.is_dir() {
                return Err(format!("{}: not a directory", display_path(&cwd, &root)));
            }
            let mut entries = Vec::new();
            let mut walker = ignore::WalkBuilder::new(&root);
            walker
                .hidden(!params.hidden)
                .git_ignore(params.gitignore)
                .git_global(params.gitignore)
                .git_exclude(params.gitignore)
                .parents(params.gitignore)
                .require_git(false)
                .follow_links(false)
                .sort_by_file_name(|a, b| a.cmp(b));
            let mut seen: u64 = 0;
            for entry in walker.build() {
                seen += 1;
                if seen.is_multiple_of(256) && cancel.is_cancelled() {
                    return Err("cancelled".to_string());
                }
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(_) => continue,
                };
                if entry.path() == root {
                    continue;
                }
                let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
                let candidate_text = if basename_mode {
                    entry.file_name().to_string_lossy().into_owned()
                } else {
                    entry
                        .path()
                        .strip_prefix(&root)
                        .unwrap_or_else(|_| entry.path())
                        .to_string_lossy()
                        .into_owned()
                };
                if !glob.is_match_candidate(&Candidate::new(&candidate_text)) {
                    continue;
                }
                let mut shown = display_path(&cwd, entry.path());
                if is_dir && !shown.ends_with('/') {
                    shown.push('/');
                }
                entries.push(shown);
            }
            entries.sort();
            entries.dedup();
            let total = entries.len();
            let truncated = total > limit;
            entries.truncate(limit);
            let mut text = entries.join("\n");
            if truncated {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&format!(
                    "... ({total} matches; truncated at limit {limit})"
                ));
            }
            if text.is_empty() {
                text = "(no matches)".to_string();
            }
            Ok(output(
                text,
                json!({
                    "pattern": params.pattern,
                    "root": display_path(&cwd, &root),
                    "matches": entries.len(),
                    "total": total,
                    "truncated": truncated,
                    "limit": limit,
                }),
            ))
        })
        .await
        .map_err(|e| format!("glob task failed: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// grep
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GrepParams {
    pattern: String,
    /// File or directory to search; relative paths resolve against the cwd.
    #[serde(default)]
    path: Option<String>,
    /// Case-sensitive when true or absent; `false` searches case-insensitively.
    #[serde(default)]
    case: Option<bool>,
    #[serde(default = "yes")]
    gitignore: bool,
    /// Number of files to skip before reporting (pagination).
    #[serde(default)]
    skip: Option<usize>,
}

struct GrepTool {
    cwd: PathBuf,
}

struct FileMatches {
    lines: Vec<String>,
    matches: usize,
    over_cap: bool,
    size_limited: bool,
    binary: bool,
}

impl GrepTool {
    fn search_file(
        regex: &regex::Regex,
        path: &Path,
        cwd: &Path,
        cap: usize,
    ) -> Result<FileMatches, String> {
        let (bytes, size_limited) = read_file_prefix(path, GREP_MAX_FILE_BYTES)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if looks_binary(&bytes) {
            return Ok(FileMatches {
                lines: Vec::new(),
                matches: 0,
                over_cap: false,
                size_limited,
                binary: true,
            });
        }
        let shown = display_path(cwd, path);
        let mut lines = Vec::new();
        let mut matches = 0usize;
        let mut over_cap = false;
        for (index, raw) in bytes.split(|b| *b == b'\n').enumerate() {
            let line = String::from_utf8_lossy(raw);
            if !regex.is_match(&line) {
                continue;
            }
            matches += 1;
            if lines.len() < cap {
                let (mut text, cut) =
                    truncate_bytes(line.trim_end_matches('\r'), MATCH_LINE_MAX_BYTES);
                if cut {
                    text.push_str("...");
                }
                lines.push(format!("{shown}:{}: {text}", index + 1));
            } else {
                over_cap = true;
            }
        }
        Ok(FileMatches {
            lines,
            matches,
            over_cap,
            size_limited,
            binary: false,
        })
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "grep".to_string(),
            description: "Search file contents with a regex under a file or directory, honoring \
				.gitignore. Deterministic output (`path:line: text`), bounded per file and per \
				response; `skip` paginates additional files."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Rust regex searched per line."},
                    "path": {"type": "string", "description": "File or directory to search (default session cwd)."},
                    "case": {"type": "boolean", "description": "Case-sensitive search (default true)."},
                    "gitignore": {"type": "boolean", "description": "Respect ignore files (default true)."},
                    "skip": {"type": "integer", "minimum": 0, "description": "Files to skip before reporting (pagination)."},
                },
                "required": ["pattern"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Read
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: GrepParams = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid grep arguments: {e}"))?;
        let regex = regex::RegexBuilder::new(&params.pattern)
            .case_insensitive(params.case == Some(false))
            .build()
            .map_err(|e| format!("invalid regex {:?}: {e}", params.pattern))?;
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let root = params
            .path
            .as_deref()
            .map_or_else(|| cwd.clone(), |p| resolve_path(&cwd, p));
        let skip = params.skip.unwrap_or(0);
        let cancel = ctx.cancel.clone();
        tokio::task::spawn_blocking(move || {
            let meta = std::fs::metadata(&root)
                .map_err(|e| format!("{}: {e}", display_path(&cwd, &root)))?;
            if meta.is_file() {
                // Single explicit file: wider per-file cap, ignore rules bypassed.
                let result = Self::search_file(&regex, &root, &cwd, GREP_SINGLE_FILE_MATCHES)?;
                if result.binary {
                    return Err(format!(
                        "{}: binary file (no text matches reported)",
                        display_path(&cwd, &root)
                    ));
                }
                let mut text = String::new();
                for line in &result.lines {
                    text.push_str(line);
                    text.push('\n');
                }
                if result.over_cap {
                    text.push_str(&format!(
                        "... ({} matches total; capped at {})",
                        result.matches, GREP_SINGLE_FILE_MATCHES
                    ));
                }
                return Ok(output(
                    if text.is_empty() {
                        "(no matches)".to_string()
                    } else {
                        text
                    },
                    json!({
                        "pattern": params.pattern,
                        "root": display_path(&cwd, &root),
                        "files": 1,
                        "matches": result.matches,
                        "truncated": result.over_cap || result.size_limited,
                    }),
                ));
            }

            // Collect candidate files deterministically, then paginate.
            let mut files = Vec::new();
            let mut walker = ignore::WalkBuilder::new(&root);
            walker
                .hidden(true)
                .git_ignore(params.gitignore)
                .git_global(params.gitignore)
                .git_exclude(params.gitignore)
                .parents(params.gitignore)
                .require_git(false)
                .follow_links(false)
                .sort_by_file_name(|a, b| a.cmp(b));
            for entry in walker.build() {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(_) => continue,
                };
                if entry.file_type().is_some_and(|t| t.is_file()) {
                    files.push(entry.into_path());
                }
            }
            files.sort();
            let total_files = files.len();
            let page: Vec<PathBuf> = files.into_iter().skip(skip).take(GREP_FILE_LIMIT).collect();
            let remaining = total_files.saturating_sub(skip + page.len());
            let mut text = String::new();
            let mut matches = 0usize;
            let mut binary_files = 0usize;
            let mut truncated = false;
            for file in &page {
                if cancel.is_cancelled() {
                    return Err("cancelled".to_string());
                }
                let result = match Self::search_file(&regex, file, &cwd, GREP_PER_FILE_MATCHES) {
                    Ok(result) => result,
                    Err(_) => continue, // unreadable mid-walk: skip, don't fail the search
                };
                if result.binary {
                    binary_files += 1;
                    continue;
                }
                matches += result.matches;
                for line in &result.lines {
                    text.push_str(line);
                    text.push('\n');
                }
                if result.over_cap {
                    truncated = true;
                    text.push_str(&format!(
                        "  ... (more matches in {})\n",
                        display_path(&cwd, file)
                    ));
                }
                if result.size_limited {
                    truncated = true;
                }
            }
            if remaining > 0 {
                truncated = true;
                text.push_str(&format!(
                    "... ({remaining} files not searched; pass skip={} for the next page)",
                    skip + page.len()
                ));
            }
            if text.is_empty() {
                text = "(no matches)".to_string();
            }
            Ok(output(
                text,
                json!({
                    "pattern": params.pattern,
                    "root": display_path(&cwd, &root),
                    "files": page.len(),
                    "filesSkipped": skip,
                    "filesRemaining": remaining,
                    "binaryFilesSkipped": binary_files,
                    "matches": matches,
                    "truncated": truncated,
                }),
            ))
        })
        .await
        .map_err(|e| format!("grep task failed: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// bash
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BashParams {
    command: String,
    /// Seconds; 0 disables the deadline. Clamped to [1, 3600].
    #[serde(default)]
    timeout: Option<f64>,
    /// Working directory for the command; default session cwd.
    #[serde(default)]
    cwd: Option<String>,
}

/// Rolling tail buffer: keeps the most recent `cap` bytes, counting the total
/// seen and bytes dropped.
struct TailBuffer {
    buf: VecDeque<u8>,
    cap: usize,
    total: usize,
    dropped: usize,
}

impl TailBuffer {
    fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::new(),
            cap,
            total: 0,
            dropped: 0,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.total += bytes.len();
        if bytes.len() >= self.cap {
            let keep = &bytes[bytes.len() - self.cap..];
            self.dropped += self.buf.len() + bytes.len() - self.cap;
            self.buf.clear();
            self.buf.extend(keep.iter().copied());
            return;
        }
        self.buf.extend(bytes.iter().copied());
        while self.buf.len() > self.cap {
            self.buf.pop_front();
            self.dropped += 1;
        }
    }

    fn finish(&mut self) -> (String, usize, usize) {
        let bytes: Vec<u8> = self.buf.drain(..).collect();
        (
            String::from_utf8_lossy(&bytes).into_owned(),
            self.total,
            self.dropped,
        )
    }
}

fn spawn_drain<R>(mut reader: R, sink: Arc<Mutex<TailBuffer>>) -> tokio::task::JoinHandle<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if let Ok(mut guard) = sink.lock() {
                        guard.push(&chunk[..n]);
                    }
                }
            }
        }
    })
}

/// SIGKILL the child's whole process group. The child is spawned with
/// `process_group(0)`, so its pgid equals its pid; this also kills pipelines
/// and background grandchildren the shell started.
#[cfg(unix)]
fn kill_process_group(pid: u32) {
    // SAFETY: killpg on a process group we spawned; an already-reaped pgid
    // makes this a harmless no-op (ESRCH).
    unsafe {
        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}
struct ProcessGroupGuard {
    pid: Option<u32>,
    active: bool,
}

impl ProcessGroupGuard {
    fn new(pid: Option<u32>) -> Self {
        Self { pid, active: true }
    }
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if self.active
            && let Some(pid) = self.pid
        {
            kill_process_group(pid);
        }
    }
}

struct BashTool {
    cwd: PathBuf,
    /// When set, the command runs under the workspace's resource governance
    /// (a systemd user scope with CPU/memory caps) instead of a bare spawn.
    limits: Option<ResourceLimits>,
}

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "bash".to_string(),
            description: "Run a shell command (sh -c) in the session working directory. Merged \
				stdout/stderr is bounded to a rolling tail; the process runs in its own process \
				group and is killed on timeout or cancellation."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command line passed to `sh -c`."},
                    "timeout": {"type": "number", "description": format!("Timeout in seconds; 0 disables the deadline; nonzero values are clamped to {BASH_MIN_TIMEOUT}-{BASH_MAX_TIMEOUT} (default {BASH_DEFAULT_TIMEOUT}).")},
                    "cwd": {"type": "string", "description": "Working directory for the command (default session cwd)."},
                },
                "required": ["command"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Exec
    }

    /// Non-pty commands run alongside each other, matching source.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Shared
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: BashParams = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid bash arguments: {e}"))?;
        check_cancel(&ctx.cancel)?;
        let base_cwd = effective_cwd(&self.cwd, &ctx);
        let cwd = params
            .cwd
            .as_deref()
            .map_or_else(|| base_cwd.to_path_buf(), |p| resolve_path(base_cwd, p));
        if !cwd.is_dir() {
            return Err(format!("{}: not a directory", cwd.display()));
        }
        let timeout = match params.timeout {
            Some(raw) if raw <= 0.0 => None,
            Some(raw) => Some(Duration::from_secs_f64(
                raw.clamp(BASH_MIN_TIMEOUT, BASH_MAX_TIMEOUT),
            )),
            None => Some(Duration::from_secs_f64(BASH_DEFAULT_TIMEOUT)),
        };

        let mut command = tokio::process::Command::new("sh");
        command
            .arg("-c")
            .arg(&params.command)
            .current_dir(&cwd)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        // Under governance the command runs inside a systemd user scope, so the
        // caps apply to the whole tree. `enforce_scope` is the caller's choice:
        // with it off, a host without `systemd-run` degrades to a bare spawn
        // rather than failing the tool.
        // The governed path goes through `spawn_scoped`, whose guard stops the
        // transient unit on drop: a sandboxed command runs as a service that
        // outlives its client, so killing the child alone would leave it running.
        let mut child = match &self.limits {
            Some(limits) => {
                let workspace = SubagentWorkspace {
                    path: cwd.clone(),
                    node_id: String::new(),
                    thread: String::new(),
                    repo_root: cwd.clone(),
                    subagent_id: String::new(),
                };
                workspace
                    .spawn_scoped(
                        &mut command,
                        limits.clone(),
                        ScopedStdio {
                            stdin: std::process::Stdio::null(),
                            stdout: std::process::Stdio::piped(),
                            stderr: std::process::Stdio::piped(),
                        },
                    )
                    .await
                    .map_err(|e| format!("failed to spawn `sh`: {e}"))?
            }
            None => ScopedChild::unscoped(
                command
                    .spawn()
                    .map_err(|e| format!("failed to spawn `sh`: {e}"))?,
            ),
        };
        let pid = child.id();
        let mut guard = ProcessGroupGuard::new(pid);

        // Drain both pipes concurrently into a shared rolling tail so slow
        // readers can never deadlock the child on a full pipe.
        let sink = Arc::new(Mutex::new(TailBuffer::new(BASH_OUTPUT_MAX_BYTES)));
        let stdout = child.child_mut().stdout.take().expect("stdout piped");
        let stderr = child.child_mut().stderr.take().expect("stderr piped");
        let out_task = spawn_drain(stdout, Arc::clone(&sink));
        let err_task = spawn_drain(stderr, Arc::clone(&sink));

        let start = Instant::now();
        let (finished, timed_out, cancelled) = {
            let wait = child.wait();
            tokio::pin!(wait);
            match timeout {
                Some(deadline) => {
                    tokio::select! {
                        result = &mut wait => (Some(result.map_err(|e| e.to_string())?), false, false),
                        () = tokio::time::sleep(deadline) => (None, true, false),
                        () = ctx.cancel.cancelled() => (None, false, true),
                    }
                }
                None => {
                    tokio::select! {
                        result = &mut wait => (Some(result.map_err(|e| e.to_string())?), false, false),
                        () = ctx.cancel.cancelled() => (None, false, true),
                    }
                }
            }
        };

        let exit_status = if timed_out || cancelled {
            if let Some(pid) = pid {
                kill_process_group(pid);
            }
            guard.disarm();
            // Reap the leader; the group has been signalled.
            child.wait().await.map_err(|e| e.to_string())?
        } else {
            guard.disarm();
            finished.expect("status present when not timed out or cancelled")
        };

        // Readers finish once the pipes close (exited or killed).
        let _ = out_task.await;
        let _ = err_task.await;
        let (text_output, total_bytes, dropped_bytes) = {
            let mut guard = sink.lock().unwrap_or_else(|e| e.into_inner());
            guard.finish()
        };
        let truncated = dropped_bytes > 0;

        let exit_code = exit_status.code();
        let mut summary = String::new();
        if truncated {
            summary.push_str(&format!("[output truncated: {dropped_bytes} of {total_bytes} bytes dropped; showing last {} bytes]\n", BASH_OUTPUT_MAX_BYTES.min(total_bytes)));
        }
        summary.push_str(&text_output);
        if !text_output.is_empty() && !text_output.ends_with('\n') {
            summary.push('\n');
        }
        if cancelled {
            summary.push_str("[cancelled; process group killed]");
        } else if timed_out {
            summary.push_str(&format!(
                "[timed out after {}s; process group killed]",
                timeout.map_or(0.0, |d| d.as_secs_f64())
            ));
        } else if let Some(code) = exit_code {
            summary.push_str(&format!("[exit code {code}]"));
        } else {
            summary.push_str(&format!("[{exit_status}]"));
        }
        let is_error = cancelled || timed_out || exit_code.is_none_or(|c| c != 0);
        let details = json!({
            "command": params.command,
            "cwd": display_path(base_cwd, &cwd),
            "exitCode": exit_code,
            "timedOut": timed_out,
            "cancelled": cancelled,
            "truncated": truncated,
            "outputBytes": total_bytes,
            "durationMs": start.elapsed().as_millis() as u64,
        });
        Ok(if is_error {
            error_output(summary, details)
        } else {
            output(summary, details)
        })
    }
}

// ---------------------------------------------------------------------------
// Desktop remote-sharing artifact tool
// ---------------------------------------------------------------------------

/// Standard descriptor version for desktop artifacts.
pub const DESKTOP_SESSION_ARTIFACT_VERSION: u32 = 1;

fn default_artifact_version() -> u32 {
    DESKTOP_SESSION_ARTIFACT_VERSION
}

/// Typed descriptor for an agent-mounted desktop session artifact.
///
/// Carries identifiers only. Bearer tokens, SDP payloads, and raw local socket
/// paths are omitted by design so durable transcripts remain safe to share.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DesktopSessionArtifact {
    #[serde(default = "default_artifact_version")]
    pub version: u32,
    pub artifact_id: String,
    pub title: String,
    pub host_id: String,
    pub environment_id: String,
    pub grant_id: String,
}

#[derive(Deserialize)]
struct DesktopArtifactArgs {
    title: String,
    host_id: String,
    environment_id: String,
    grant_id: String,
    #[serde(default)]
    artifact_id: Option<String>,
}

/// Tool enabling agents to register and mount a desktop sharing artifact in chat.
pub struct DesktopArtifactTool;

#[async_trait]
impl Tool for DesktopArtifactTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "desktop_artifact".to_string(),
            description: "Creates and mounts a remote desktop session artifact in the chat. Records durable identifier metadata (host_id, environment_id, grant_id, title) without embedding credentials or live stream state.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Human-readable label for the desktop sharing artifact."
                    },
                    "host_id": {
                        "type": "string",
                        "description": "Opaque identifier of the host machine or cluster."
                    },
                    "environment_id": {
                        "type": "string",
                        "description": "Opaque identifier of the isolated desktop environment."
                    },
                    "grant_id": {
                        "type": "string",
                        "description": "Opaque view/control grant identifier."
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": "Optional explicit artifact identifier; generated if omitted."
                    }
                },
                "required": ["title", "host_id", "environment_id", "grant_id"],
                "additionalProperties": false
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Write
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Shared
    }

    async fn execute(&self, arguments: Value, _ctx: ToolContext) -> Result<ToolOutput, String> {
        let args: DesktopArtifactArgs = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid desktop_artifact arguments: {e}"))?;

        if args.title.trim().is_empty() {
            return Err("desktop artifact title must not be empty".to_string());
        }
        if args.host_id.trim().is_empty() {
            return Err("desktop artifact host_id must not be empty".to_string());
        }
        if args.environment_id.trim().is_empty() {
            return Err("desktop artifact environment_id must not be empty".to_string());
        }
        if args.grant_id.trim().is_empty() {
            return Err("desktop artifact grant_id must not be empty".to_string());
        }

        let artifact_id = args
            .artifact_id
            .filter(|id| !id.trim().is_empty())
            .unwrap_or_else(|| format!("desk-{}", uuid::Uuid::now_v7()));

        let artifact = DesktopSessionArtifact {
            version: DESKTOP_SESSION_ARTIFACT_VERSION,
            artifact_id: artifact_id.clone(),
            title: args.title.clone(),
            host_id: args.host_id.clone(),
            environment_id: args.environment_id.clone(),
            grant_id: args.grant_id.clone(),
        };

        let details = json!({
            "type": "desktop_session_artifact",
            "artifact": artifact
        });

        let summary = format!(
            "Desktop sharing artifact mounted: \"{}\" [artifact_id: {}, env: {}, host: {}, grant: {}]",
            args.title, artifact_id, args.environment_id, args.host_id, args.grant_id
        );

        Ok(output(summary, details))
    }
}
// ---------------------------------------------------------------------------
// desktop environment tool
// ---------------------------------------------------------------------------

/// Tool enabling an agent to own and drive an isolated headless desktop.
pub struct DesktopTool {
    supervisor: Arc<juto_desktop::DesktopSupervisor>,
    signaling: Arc<crate::SignalingHandle>,
    data_dir: PathBuf,
}

/// Registers the desktop environment tool against a supervisor.
pub fn register_desktop_tools(
    registry: &mut ToolRegistry,
    supervisor: Arc<juto_desktop::DesktopSupervisor>,
    signaling: Arc<crate::SignalingHandle>,
    data_dir: PathBuf,
) -> Result<(), AgentError> {
    registry.register(Arc::new(DesktopTool {
        supervisor,
        signaling,
        data_dir,
    }))?;
    Ok(())
}

#[derive(Deserialize)]
struct DesktopArgs {
    action: String,
    #[serde(default)]
    env_id: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    fps: Option<u32>,
    #[serde(default)]
    scale: Option<u32>,
    #[serde(default)]
    app_id: Option<String>,
    #[serde(default)]
    argv: Option<Vec<String>>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    env: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    events: Option<Vec<juto_desktop::protocol::InputEvent>>,
    #[serde(default)]
    snapshot_path: Option<String>,
}

#[async_trait]
impl Tool for DesktopTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "desktop".to_string(),
            description: "Owns an isolated headless desktop environment: create it, launch \
                          applications into it, observe lossless snapshots, inject input, and \
                          destroy it. The desktop has no host display; applications run in a \
                          per-environment systemd scope with enforced CPU/memory/task limits."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "launch", "observe", "input", "describe", "attach", "destroy"],
                        "description": "Operation to perform."
                    },
                    "env_id": {
                        "type": "string",
                        "description": "Environment identifier returned by `create`. Required for every action except `create`."
                    },
                    "title": {
                        "type": "string",
                        "description": "create: human-readable label for the environment."
                    },
                    "width": { "type": "integer", "description": "create: output width in pixels (default 1280)." },
                    "height": { "type": "integer", "description": "create: output height in pixels (default 720)." },
                    "fps": { "type": "integer", "description": "create: virtual output refresh cap (default 30)." },
                    "scale": { "type": "integer", "description": "create: integer output scale, 1 or 2 (default 1)." },
                    "app_id": {
                        "type": "string",
                        "description": "launch: identifier for the launched application."
                    },
                    "argv": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "launch: executable and arguments. Never shell-interpolated."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "launch: working directory inside the session workspace (defaults to the workspace root)."
                    },
                    "env": {
                        "type": "object",
                        "additionalProperties": { "type": "string" },
                        "description": "launch: explicit environment overrides for the application."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "observe: how long to wait for a fresh frame (default 5000)."
                    },
                    "snapshot_path": {
                        "type": "string",
                        "description": "observe: where to write the lossless PNG snapshot (defaults to the runtime data directory)."
                    },
                    "events": {
                        "type": "array",
                        "description": "input: ordered input events, e.g. {\"kind\":\"pointerMoveAbs\",\"x\":10,\"y\":20}, {\"kind\":\"pointerButton\",\"button\":272,\"pressed\":true}, {\"kind\":\"key\",\"keycode\":30,\"pressed\":true}, {\"kind\":\"pointerScroll\",\"axis\":1,\"value\":-1}.",
                        "items": { "type": "object" }
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Exec
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Shared
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let args: DesktopArgs = serde_json::from_value(arguments)
            .map_err(|e| format!("invalid desktop arguments: {e}"))?;
        check_cancel(&ctx.cancel)?;

        let supervisor = self.supervisor.clone();
        let signaling = self.signaling.clone();
        let data_dir = self.data_dir.clone();
        let workspace = effective_cwd(Path::new(""), &ctx).to_path_buf();
        let cancel = ctx.cancel.clone();

        tokio::task::spawn_blocking(move || {
            run_desktop_action(
                &supervisor,
                &signaling,
                &data_dir,
                &workspace,
                args,
                &cancel,
            )
        })
        .await
        .map_err(|e| format!("desktop task failed: {e}"))?
    }
}

fn run_desktop_action(
    supervisor: &juto_desktop::DesktopSupervisor,
    signaling: &crate::SignalingHandle,
    data_dir: &Path,
    workspace: &Path,
    args: DesktopArgs,
    cancel: &CancellationToken,
) -> Result<ToolOutput, String> {
    use juto_desktop::supervisor::{EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits};

    if cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }

    let env_id = || -> Result<String, String> {
        args.env_id
            .clone()
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| "env_id is required for this action".to_string())
    };
    let env = || -> Result<juto_desktop::DesktopEnvironment, String> {
        let id = env_id()?;
        supervisor.get_environment(&id).map_err(|e| e.to_string())
    };

    match args.action.as_str() {
        "create" => {
            let width = args.width.unwrap_or(1280);
            let height = args.height.unwrap_or(720);
            let fps = args.fps.unwrap_or(30);
            let scale = args.scale.unwrap_or(1);
            let title = args
                .title
                .clone()
                .unwrap_or_else(|| "agent desktop".to_string());
            let id = format!("desk-{}", uuid::Uuid::now_v7());
            let config = EnvironmentConfig {
                env_id: id.clone(),
                owner_principal: "agent:session".to_string(),
                output: OutputConfig {
                    width,
                    height,
                    scale,
                    fps,
                },
                render_profile: RenderProfile::Software,
                limits: SystemLimits::default(),
                workspace_dir: workspace.to_path_buf(),
                lease_ttl_secs: 6,
            };
            let env = supervisor
                .create_environment(config)
                .map_err(|e| e.to_string())?;
            let grant = env.issue_grant(
                "agent:session".to_string(),
                vec![
                    juto_desktop::protocol::GrantRole::Owner,
                    juto_desktop::protocol::GrantRole::View,
                    juto_desktop::protocol::GrantRole::Control,
                ],
                "session".to_string(),
                1800,
            );
            let host_id = host_identifier();
            let summary = format!(
                "Desktop environment created: {id} \"{title}\" ({width}x{height}@{fps}, scale {scale}). \
                 Socket: {}. Owner grant: {}.",
                env.socket_path().display(),
                grant.grant_id
            );
            Ok(output(
                summary,
                json!({
                    "action": "create",
                    "envId": id,
                    "title": title,
                    "hostId": host_id,
                    "grantId": grant.grant_id,
                    "socketPath": env.socket_path().to_string_lossy(),
                    "output": env.output(),
                    "limits": env.limits(),
                }),
            ))
        }
        "launch" => {
            let env = env()?;
            let app_id = args
                .app_id
                .clone()
                .filter(|id| !id.trim().is_empty())
                .ok_or_else(|| "app_id is required for launch".to_string())?;
            let argv = args
                .argv
                .clone()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| "argv is required for launch".to_string())?;
            let cwd = args.cwd.as_ref().map(PathBuf::from);
            let overrides: Vec<(String, String)> =
                args.env.clone().unwrap_or_default().into_iter().collect();
            let handle = env
                .launch(&app_id, &argv, cwd.as_deref(), &overrides)
                .map_err(|e| e.to_string())?;
            let summary = format!(
                "Launched `{}` (pid {}) in environment {}.",
                handle.app_id,
                handle.pid,
                env.env_id()
            );
            Ok(output(
                summary,
                json!({
                    "action": "launch",
                    "envId": env.env_id(),
                    "appId": handle.app_id,
                    "pid": handle.pid,
                    "unit": handle.unit,
                }),
            ))
        }
        "observe" => {
            let env = env()?;
            let timeout = Duration::from_millis(args.timeout_ms.unwrap_or(5000).max(1));
            let (info, png) = env.snapshot_png(timeout).map_err(|e| e.to_string())?;
            let path = match args.snapshot_path.as_ref() {
                Some(raw) => resolve_path(workspace, raw),
                None => data_dir.join("desktop-snapshots").join(format!(
                    "{}-{}.png",
                    env.env_id(),
                    info.frame_id
                )),
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            std::fs::write(&path, &png)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            let status = env.get_status();
            let summary = format!(
                "Observed frame {} of {} ({}x{}, generation {}); snapshot written to {} ({} bytes).",
                info.frame_id,
                env.env_id(),
                info.width,
                info.height,
                info.generation,
                path.display(),
                png.len()
            );
            Ok(output(
                summary,
                json!({
                    "action": "observe",
                    "envId": env.env_id(),
                    "frameId": info.frame_id,
                    "generation": info.generation,
                    "timestampUs": info.timestamp_us,
                    "width": info.width,
                    "height": info.height,
                    "snapshotPath": path.to_string_lossy(),
                    "snapshotBytes": png.len(),
                    "windows": status.mapped_windows,
                    "apps": env.apps(),
                }),
            ))
        }
        "input" => {
            let env = env()?;
            let events = args
                .events
                .clone()
                .filter(|e| !e.is_empty())
                .ok_or_else(|| "events are required for input".to_string())?;
            let owner = env.get_status().owner_principal;
            let lease = match env.get_status().active_control_lease {
                Some(lease) if lease.holder == owner => lease,
                _ => env
                    .set_control(&owner, "acquire", false)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| "failed to acquire control".to_string())?,
            };
            let batch = juto_desktop::protocol::InputBatch {
                seq: 0,
                client_timestamp_us: 0,
                events,
            };
            let watermark = env
                .dispatch_input(&owner, lease.epoch, env.generation(), batch)
                .map_err(|e| e.to_string())?;
            Ok(output(
                format!(
                    "Input accepted for {} (lease epoch {}, watermark {}).",
                    env.env_id(),
                    lease.epoch,
                    watermark
                ),
                json!({
                    "action": "input",
                    "envId": env.env_id(),
                    "epoch": lease.epoch,
                    "appliedWatermark": watermark,
                }),
            ))
        }
        "describe" => {
            let env = env()?;
            let status = env.get_status();
            let summary = format!(
                "Environment {} is {} at {}x{}@{} (generation {}), {} window(s), {} app(s).",
                status.env_id,
                if status.is_ready { "ready" } else { "stopped" },
                status.output.width,
                status.output.height,
                status.output.fps,
                status.generation,
                status.mapped_windows.len(),
                env.apps().len()
            );
            Ok(output(
                summary,
                json!({
                    "action": "describe",
                    "status": status,
                    "apps": env.apps(),
                }),
            ))
        }
        "attach" => {
            let env = env()?;
            let server = signaling.get().map_err(|e| e.to_string())?;
            let grant = server.issue_token(
                env.env_id(),
                "remote-viewer",
                vec![
                    juto_desktop::protocol::GrantRole::View,
                    juto_desktop::protocol::GrantRole::Control,
                ],
            );
            let url = server.viewer_url(&grant.token);
            let summary = format!(
                "Remote viewer ready for {}: {url} (one-time token, expires in 60 seconds). \
                 Open it on another device on this network; TLS termination and public \
                 reachability belong to the deployment gateway.",
                env.env_id()
            );
            Ok(output(
                summary,
                json!({
                    "action": "attach",
                    "envId": env.env_id(),
                    "viewerUrl": url,
                    "token": grant.token,
                    "expiresAt": grant.expires_at,
                    "transport": "webrtc",
                }),
            ))
        }
        "destroy" => {
            let id = env_id()?;
            supervisor
                .destroy_environment(&id)
                .map_err(|e| e.to_string())?;
            Ok(output(
                format!("Desktop environment {id} destroyed."),
                json!({ "action": "destroy", "envId": id }),
            ))
        }
        other => Err(format!("unknown desktop action `{other}`")),
    }
}

pub(crate) fn host_identifier() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.trim().is_empty())
        .unwrap_or_else(|| "local-host".to_string())
}

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------

/// Construct the standard tool set rooted at `cwd`.
///
/// Registers `read`, `write`, `edit`, `glob`, `grep`, and `bash`. Approval
/// policy is enforced by agent hooks via each tool's [`Tool::tier`]; these
/// tools perform real filesystem/process work and return true errors.
/// The change-graph operations an agent needs to work across machines.
///
/// The same operations exist in `juto-cli changes`, but an agent cannot rely on
/// that binary being installed, so the cross-machine workflow — fetch another
/// machine's work, reconcile it, publish your own — would otherwise be
/// unreachable from a tool call.
struct VcsTool {
    cwd: PathBuf,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum VcsAction {
    /// Counts, heads, and whether a reconcile is pending.
    Status,
    /// Import another machine's graph.
    Fetch,
    /// Fold imported heads into the local thread.
    Reconcile,
    /// Publish the local graph.
    Push,
    /// Check that every node's snapshot ref is present and correct.
    Verify,
    /// Freeze a point as a named checkpoint.
    Mark,
    /// Revert a thread to an earlier point.
    Rewind,
    /// Project marked points into Git commits on a branch.
    Export,
    /// Push an exported branch to a remote.
    Publish,
    /// Fetch an exported branch from a remote into a local branch.
    FetchBranch,
    /// Merge two threads' trees into one node with two parents.
    Join,
    /// Record a resolved merge: the merge result with the conflicted paths
    /// taken from the workspace, as a two-parent node.
    Resolve,
    /// List the paths where the workspace differs from its head.
    Dirty,
    /// Capture the workspace as a new point on a thread.
    Capture,
    /// Print the change graph: nodes, heads, and checkpoints.
    Log,
    /// Reclaim unreachable nodes, release their refs, and pack loose refs.
    Gc,
}

#[derive(Deserialize)]
struct VcsParams {
    action: VcsAction,
    /// Remote name or path; defaults to `origin`.
    remote: Option<String>,
    /// Thread for `reconcile` and `rewind`; defaults to `main`.
    thread: Option<String>,
    /// Point to freeze (`mark`) or revert to (`rewind`): a node id or a
    /// checkpoint name. `mark` defaults to the thread head.
    point: Option<String>,
    /// Checkpoint name for `mark`; defaults to the point's id.
    name: Option<String>,
    /// Git commit message for `mark`; defaults to the checkpoint name.
    message: Option<String>,
    /// Whether `mark` records a verification attestation. It never runs or
    /// invents verification.
    verified: Option<bool>,
    /// Git branch for `export` and `publish`.
    branch: Option<String>,
    /// Left and right points for `join`: node ids or checkpoint names.
    left: Option<String>,
    right: Option<String>,
    /// How many of the most recent nodes `log` returns; 0 means all.
    limit: Option<usize>,
}

#[async_trait]
impl Tool for VcsTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "vcs".to_string(),
            description: "Inspect and synchronize Juto's native change graph. `status` reports \
                counts, heads, and whether an imported head still needs reconciling; `log` \
                prints the nodes, heads, checkpoints, and base anchor; `dirty` \
                lists paths edited outside the capture tools, since a `bash` command is not \
                captured automatically, and `capture` records them as a new point; `fetch` \
                imports another machine's graph; `reconcile` \
                folds imported heads into the local thread; `push` publishes the local graph; \
                `verify` checks that every node's snapshot ref is present and correct; `mark` \
                freezes a point as a named checkpoint with a commit message; `rewind` reverts a \
                thread to an earlier \
                point; `export` projects marked points into Git commits on a branch; `publish` \
                and `fetch_branch` move that branch to and from a remote; `join` merges two \
                threads into a two-parent node; `resolve` records a conflict resolved by hand \
                the same way, taking the conflicted paths from the workspace; `gc` reclaims \
                unreachable nodes, releases their refs, and packs loose refs."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "status",
                            "fetch",
                            "reconcile",
                            "push",
                            "verify",
                            "mark",
                            "rewind",
                            "export",
                            "publish",
                            "fetch_branch",
                            "join",
                            "resolve",
                            "dirty",
                            "capture",
                            "log",
                            "gc"
                        ],
                        "description": "Which change-graph operation to run."
                    },
                    "remote": {
                        "type": "string",
                        "description": "Remote name or path for `fetch` and `push`; defaults to `origin`."
                    },
                    "thread": {
                        "type": "string",
                        "description": "Thread for `reconcile` and `rewind`; defaults to `main`."
                    },
                    "point": {
                        "type": "string",
                        "description": "Point to freeze (`mark`) or revert to (`rewind`): a node id or a checkpoint name. `mark` defaults to the thread head."
                    },
                    "name": {
                        "type": "string",
                        "description": "Checkpoint name for `mark`; defaults to the point's id."
                    },
                    "message": {
                        "type": "string",
                        "description": "Git commit message for `mark`; defaults to the checkpoint name."
                    },
                    "verified": {
                        "type": "boolean",
                        "description": "Whether `mark` records a verification attestation. It never runs or invents verification."
                    },
                    "branch": {
                        "type": "string",
                        "description": "Git branch for `export` and `publish`."
                    },
                    "left": {
                        "type": "string",
                        "description": "Left point for `join`: a node id or checkpoint name."
                    },
                    "right": {
                        "type": "string",
                        "description": "Right point for `join`: a node id or checkpoint name."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many of the most recent nodes `log` returns; 0 means all. Defaults to 20."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Write
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let VcsParams {
            action,
            remote,
            thread,
            point,
            name,
            message,
            verified,
            branch,
            left,
            right,
            limit,
        } = serde_json::from_value(arguments).map_err(|e| format!("invalid vcs arguments: {e}"))?;
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let remote = remote.unwrap_or_else(|| "origin".to_string());
        let thread = thread.unwrap_or_else(|| "main".to_string());
        let session_id = ctx.session_id;
        let label = action_label(&action);

        // The graph shells out to git, so keep it off the async runtime.
        let details = tokio::task::spawn_blocking(move || -> Result<Value, String> {
            let graph = ChangeGraph::open(&cwd).map_err(|e| format!("{e:#}"))?;
            match action {
                VcsAction::Status => {
                    let state = graph.state().map_err(|e| format!("{e:#}"))?;
                    // A head key containing '/' is an imported remote head: a
                    // local thread name never contains one.
                    let remote_heads: Vec<&String> =
                        state.heads.keys().filter(|key| key.contains('/')).collect();
                    Ok(json!({
                        "thread": graph.thread(),
                        "workspace": graph.workspace(),
                        "sparse": graph.sparse(),
                        "nodes": state.nodes.len(),
                        "heads": state.heads,
                        "remote_heads": remote_heads,
                        "needs_reconcile": !remote_heads.is_empty(),
                        "pending_threads": graph
                            .pending_reconcile_threads()
                            .map_err(|e| format!("{e:#}"))?,
                        "checkpoints": state.checkpoints,
                        "workspace_heads": state.workspace_heads.len(),
                    }))
                }
                VcsAction::Fetch => {
                    let report = graph.fetch_graph(&remote).map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(report).map_err(|e| e.to_string())
                }
                VcsAction::Push => {
                    let report = graph.push_graph(&remote).map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(report).map_err(|e| e.to_string())
                }
                VcsAction::Verify => {
                    let report = graph.verify().map_err(|e| format!("{e:#}"))?;
                    let ok = report.is_ok();
                    let mut value = serde_json::to_value(&report).map_err(|e| e.to_string())?;
                    if let Some(object) = value.as_object_mut() {
                        object.insert("ok".to_string(), Value::Bool(ok));
                    }
                    Ok(value)
                }
                VcsAction::Reconcile => {
                    let outcomes = graph.reconcile(&thread).map_err(|e| format!("{e:#}"))?;
                    // `ReconcileOutcome` carries a `ChangeNode`, so map it rather
                    // than deriving a wire shape for an internal type.
                    Ok(Value::Array(
                        outcomes
                            .into_iter()
                            .map(|outcome| match outcome {
                                ReconcileOutcome::Joined {
                                    remote,
                                    node,
                                    base,
                                    tree,
                                } => json!({
                                    "status": "joined",
                                    "remote": remote,
                                    "node": node,
                                    "base": base,
                                    "tree": tree,
                                }),
                                ReconcileOutcome::Adopted { remote, node } => json!({
                                    "status": "adopted",
                                    "remote": remote,
                                    "node": node,
                                }),
                                ReconcileOutcome::Conflicted {
                                    remote,
                                    base,
                                    conflicts,
                                } => json!({
                                    "status": "conflicted",
                                    "remote": remote,
                                    "base": base,
                                    "conflicts": conflicts,
                                }),
                            })
                            .collect(),
                    ))
                }
                VcsAction::Mark => {
                    // Default to the thread head: an agent checkpointing "where
                    // I am now" should not have to look up the node id first.
                    let id = match point {
                        Some(point) => point,
                        None => graph
                            .state()
                            .map_err(|e| format!("{e:#}"))?
                            .heads
                            .get(&thread)
                            .cloned()
                            .ok_or_else(|| format!("thread {thread:?} has no head"))?,
                    };
                    let name = name.unwrap_or_else(|| id.clone());
                    let message = message.unwrap_or_else(|| name.clone());
                    let node = graph
                        .mark(&id, &name, &message, verified.unwrap_or(false))
                        .map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(node).map_err(|e| e.to_string())
                }
                VcsAction::Rewind => {
                    let to = point.ok_or_else(|| "`rewind` needs a `point`".to_string())?;
                    let node = graph.rewind(&thread, &to).map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(node).map_err(|e| e.to_string())
                }
                VcsAction::Export => {
                    let checkpoint = point.ok_or_else(|| "`export` needs a `point`".to_string())?;
                    let branch = branch.ok_or_else(|| "`export` needs a `branch`".to_string())?;
                    let result = graph
                        .export(&checkpoint, &branch)
                        .map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(result).map_err(|e| e.to_string())
                }
                VcsAction::Publish => {
                    let branch = branch.ok_or_else(|| "`publish` needs a `branch`".to_string())?;
                    graph.push(&branch, &remote).map_err(|e| format!("{e:#}"))?;
                    Ok(json!({ "branch": branch, "remote": remote, "pushed": true }))
                }
                VcsAction::FetchBranch => {
                    let branch =
                        branch.ok_or_else(|| "`fetch_branch` needs a `branch`".to_string())?;
                    graph
                        .fetch_branch(&branch, &remote)
                        .map_err(|e| format!("{e:#}"))?;
                    Ok(json!({ "branch": branch, "remote": remote, "fetched": true }))
                }
                VcsAction::Join => {
                    let left = left.ok_or_else(|| "`join` needs a `left`".to_string())?;
                    let right = right.ok_or_else(|| "`join` needs a `right`".to_string())?;
                    match graph
                        .join(&thread, &left, &right)
                        .map_err(|e| format!("{e:#}"))?
                    {
                        JoinOutcome::Joined { node, base, tree } => Ok(json!({
                            "status": "joined",
                            "node": node,
                            "base": base,
                            "tree": tree,
                        })),
                        JoinOutcome::Conflicted { base, conflicts } => Ok(json!({
                            "status": "conflicted",
                            "base": base,
                            "conflicts": conflicts,
                        })),
                    }
                }
                VcsAction::Resolve => {
                    let left = left.ok_or_else(|| "`resolve` needs a `left`".to_string())?;
                    let right = right.ok_or_else(|| "`resolve` needs a `right`".to_string())?;
                    let node = graph
                        .resolve(
                            &thread,
                            &left,
                            &right,
                            EditProvenance {
                                session_id,
                                tool: "vcs".to_string(),
                                selection_mode: None,
                                path: None,
                            },
                        )
                        .map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(node).map_err(|e| e.to_string())
                }
                VcsAction::Dirty => {
                    let changed = graph.dirty_paths().map_err(|e| format!("{e:#}"))?;
                    Ok(json!({
                        "dirty": !changed.is_empty(),
                        "changed": changed,
                    }))
                }
                VcsAction::Capture => {
                    let node = graph
                        .capture(
                            &thread,
                            EditProvenance {
                                session_id,
                                tool: "vcs".to_string(),
                                selection_mode: None,
                                path: None,
                            },
                            None,
                        )
                        .map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(node).map_err(|e| e.to_string())
                }
                VcsAction::Log => {
                    let state = graph.state().map_err(|e| format!("{e:#}"))?;
                    let total = state.nodes.len();
                    let limit = limit.unwrap_or(20);
                    let nodes: Vec<&_> = if limit == 0 || limit >= total {
                        state.nodes.iter().collect()
                    } else {
                        state.nodes[total - limit..].iter().collect()
                    };
                    Ok(json!({
                        "nodes": nodes,
                        "total": total,
                        "heads": state.heads,
                        "checkpoints": state.checkpoints,
                        "base_commit": state.base_commit,
                        "base_tree": state.base_tree,
                    }))
                }
                VcsAction::Gc => {
                    let report = graph.gc().map_err(|e| format!("{e:#}"))?;
                    serde_json::to_value(report).map_err(|e| e.to_string())
                }
            }
        })
        .await
        .map_err(|e| format!("vcs task failed: {e}"))??;

        let summary = format!("vcs {label}: {details}");
        Ok(output(summary, details))
    }
}

/// A short label for a [`VcsAction`], for the tool's human-readable summary.
fn action_label(action: &VcsAction) -> &'static str {
    match action {
        VcsAction::Status => "status",
        VcsAction::Fetch => "fetch",
        VcsAction::Reconcile => "reconcile",
        VcsAction::Push => "push",
        VcsAction::Verify => "verify",
        VcsAction::Mark => "mark",
        VcsAction::Rewind => "rewind",
        VcsAction::Export => "export",
        VcsAction::Publish => "publish",
        VcsAction::FetchBranch => "fetch_branch",
        VcsAction::Join => "join",
        VcsAction::Resolve => "resolve",
        VcsAction::Dirty => "dirty",
        VcsAction::Capture => "capture",
        VcsAction::Log => "log",
        VcsAction::Gc => "gc",
    }
}

/// Build the builtin tool registry with no resource governance.
///
/// Approval policy is enforced by agent hooks via each tool's [`Tool::tier`];
/// these tools perform real filesystem/process work and return true errors.
pub fn builtin_tools(cwd: &Path) -> Result<ToolRegistry, AgentError> {
    builtin_tools_with_limits(cwd, None)
}

/// Build the builtin tool registry, running `bash` under `limits` when set.
///
/// Governance wraps the command in a systemd user scope so the CPU and memory
/// caps apply to the whole process tree. Whether a missing `systemd-run` is a
/// hard failure or a fallback to a bare spawn is `ResourceLimits::enforce_scope`.
pub fn builtin_tools_with_limits(
    cwd: &Path,
    limits: Option<ResourceLimits>,
) -> Result<ToolRegistry, AgentError> {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ReadTool {
        cwd: cwd.to_path_buf(),
    }))?;
    registry.register(Arc::new(WriteTool {
        cwd: cwd.to_path_buf(),
    }))?;
    registry.register(Arc::new(EditTool {
        cwd: cwd.to_path_buf(),
    }))?;
    registry.register(Arc::new(GlobTool {
        cwd: cwd.to_path_buf(),
    }))?;
    registry.register(Arc::new(GrepTool {
        cwd: cwd.to_path_buf(),
    }))?;
    registry.register(Arc::new(BashTool {
        cwd: cwd.to_path_buf(),
        limits,
    }))?;
    registry.register(Arc::new(VcsTool {
        cwd: cwd.to_path_buf(),
    }))?;
    registry.register(Arc::new(DesktopArtifactTool))?;
    Ok(registry)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_graph::EditProvenance;

    fn ctx(dir: &Path) -> ToolContext {
        ToolContext {
            cwd: dir.to_path_buf(),
            session_id: "test".to_string(),
            cancel: CancellationToken::new(),
        }
    }

    fn text(out: &ToolOutput) -> String {
        out.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn registry_has_builtin_tools_with_schemas() {
        let dir = tempfile::tempdir().unwrap();
        let registry = builtin_tools(dir.path()).expect("builtin tools");
        let mut names: Vec<String> = registry
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "bash",
                "desktop_artifact",
                "edit",
                "glob",
                "grep",
                "read",
                "vcs",
                "write"
            ]
        );
        for def in registry.definitions() {
            assert_eq!(def.parameters["type"], "object", "{}", def.name);
            assert!(def.parameters["required"].is_array(), "{}", def.name);
        }
    }

    #[test]
    fn tiers_and_concurrency() {
        assert!(matches!(
            ReadTool {
                cwd: PathBuf::new()
            }
            .tier(),
            ToolTier::Read
        ));
        assert!(matches!(
            GlobTool {
                cwd: PathBuf::new()
            }
            .tier(),
            ToolTier::Read
        ));
        assert!(matches!(
            GrepTool {
                cwd: PathBuf::new()
            }
            .tier(),
            ToolTier::Read
        ));
        assert!(matches!(
            WriteTool {
                cwd: PathBuf::new()
            }
            .tier(),
            ToolTier::Write
        ));
        assert!(matches!(
            EditTool {
                cwd: PathBuf::new()
            }
            .tier(),
            ToolTier::Write
        ));
        assert!(matches!(
            BashTool {
                cwd: PathBuf::new(),
                limits: None,
            }
            .tier(),
            ToolTier::Exec
        ));
        assert!(matches!(
            WriteTool {
                cwd: PathBuf::new()
            }
            .concurrency(),
            ToolConcurrency::Exclusive
        ));
        assert!(matches!(
            EditTool {
                cwd: PathBuf::new()
            }
            .concurrency(),
            ToolConcurrency::Exclusive
        ));
        assert!(matches!(
            BashTool {
                cwd: PathBuf::new(),
                limits: None,
            }
            .concurrency(),
            ToolConcurrency::Shared
        ));
        assert!(matches!(DesktopArtifactTool.tier(), ToolTier::Write));
        assert!(matches!(
            DesktopArtifactTool.concurrency(),
            ToolConcurrency::Shared
        ));
    }

    #[tokio::test]
    async fn read_numbers_lines_and_windows() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("f.txt"),
            (1..=10).map(|n| format!("line{n}\n")).collect::<String>(),
        )
        .unwrap();
        let tool = ReadTool {
            cwd: dir.path().to_path_buf(),
        };
        let out = tool
            .execute(
                json!({"path": "f.txt", "offset": 3, "limit": 2}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert_eq!(
            text(&out),
            "3: line3\n4: line4\n... (10 lines total; showing 3-4; pass offset for more)\n"
        );
        assert_eq!(out.details["totalLines"], 10);
        assert_eq!(out.details["truncated"], true);

        let err = tool
            .execute(json!({"path": "f.txt", "offset": 99}), ctx(dir.path()))
            .await
            .unwrap_err();
        assert!(err.contains("beyond end of file"), "{err}");
    }

    #[tokio::test]
    async fn read_lists_directories_and_rejects_binary() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("b.txt"), "x").unwrap();
        std::fs::write(dir.path().join("a.txt"), "y").unwrap();
        let tool = ReadTool {
            cwd: dir.path().to_path_buf(),
        };
        let out = tool
            .execute(json!({"path": "."}), ctx(dir.path()))
            .await
            .unwrap();
        let body = text(&out);
        let lines: Vec<&str> = body.lines().collect();
        assert!(lines.starts_with(&["a.txt", "b.txt"]), "{lines:?}");
        assert!(lines.contains(&"sub/"));

        std::fs::write(dir.path().join("bin"), b"a\0b\0").unwrap();
        let err = tool
            .execute(json!({"path": "bin"}), ctx(dir.path()))
            .await
            .unwrap_err();
        assert!(err.contains("binary"), "{err}");
    }

    /// A plain Git repository with a deterministic identity and one base commit.
    fn git_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(&["init", "-q"]);
        git(&["config", "user.name", "Tool Test"]);
        git(&["config", "user.email", "tools@example.test"]);
        std::fs::write(dir.path().join(".gitignore"), "").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "base"]);
        dir
    }

    #[tokio::test]
    async fn write_creates_parents_and_overwrites() {
        let dir = git_dir();
        let tool = WriteTool {
            cwd: dir.path().to_path_buf(),
        };
        tool.execute(
            json!({"path": "deep/nested/f.txt", "content": "v1"}),
            ctx(dir.path()),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("deep/nested/f.txt")).unwrap(),
            "v1"
        );
        tool.execute(
            json!({"path": "deep/nested/f.txt", "content": "v2"}),
            ctx(dir.path()),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("deep/nested/f.txt")).unwrap(),
            "v2"
        );
        // No temp files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path().join("deep/nested"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".juto-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[tokio::test]
    async fn edit_requires_unique_match_unless_replace_all() {
        let dir = git_dir();
        std::fs::write(dir.path().join("f.txt"), "foo bar foo").unwrap();
        let tool = EditTool {
            cwd: dir.path().to_path_buf(),
        };

        let err = tool
            .execute(
                json!({"path": "f.txt", "selection": {"mode": "text", "old_string": "foo"}, "replacement": "x"}),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("2 times"), "{err}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "foo bar foo"
        );

        let err = tool
            .execute(
                json!({"path": "f.txt", "selection": {"mode": "text", "old_string": "nope"}, "replacement": "x"}),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("not found"), "{err}");

        let out = tool.execute(json!({"path": "f.txt", "selection": {"mode": "text", "old_string": "foo", "replace_all": true}, "replacement": "x"}), ctx(dir.path())).await.unwrap();
        assert_eq!(out.details["replacements"], 2);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "x bar x"
        );

        let err = tool
            .execute(
                json!({"path": "f.txt", "selection": {"mode": "text", "old_string": ""}, "replacement": "x"}),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[tokio::test]
    async fn contextual_edits_preserve_other_code_and_report_only_the_operation() {
        let dir = git_dir();
        let tool = EditTool {
            cwd: dir.path().to_path_buf(),
        };
        let path = dir.path().join("f.rs");
        std::fs::write(&path, "fn first() { 1; }\nfn second() { 2; }\n").unwrap();
        std::fs::write(dir.path().join("unrelated.txt"), "human edit\n").unwrap();
        let ast = tool.execute(json!({
            "path": "f.rs", "selection": {"mode": "ast",
            "query": "(function_item name: (identifier) @name (#eq? @name \"second\")) @edit"},
            "replacement": "fn second() { 3; }"
        }), ctx(dir.path())).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn first() { 1; }\nfn second() { 3; }\n"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("unrelated.txt")).unwrap(),
            "human edit\n"
        );
        assert_ne!(ast.details["before_tree"], ast.details["after_tree"]);
        let diff = ast.details["diff"].as_str().unwrap();
        assert!(diff.contains("-fn second() { 2; }"), "{diff}");
        assert!(diff.contains("+fn second() { 3; }"), "{diff}");
        assert!(!diff.contains("unrelated.txt"), "{diff}");
        let range = tool
            .execute(
                json!({
                    "path": "f.rs", "selection": {"mode": "range", "start_line": 1, "end_line": 1},
                    "replacement": "fn first() { 4; }\n"
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn first() { 4; }\nfn second() { 3; }\n"
        );
        // Each edit records its own durable node; the earlier receipt still
        // identifies the exact tree it produced.
        assert_ne!(range.details["graph_node"], ast.details["graph_node"]);
        assert!(ast.details["graph_node"].is_string());
        assert!(range.details["graph_node"].is_string());
        // Selecting the earlier tool receipt must publish that exact edit, not
        // the later workspace state, and preserve the pre-existing human file.
        for (key, value) in [
            ("user.name", "Edit Graph Test"),
            ("user.email", "edit-graph@example.test"),
        ] {
            assert!(
                std::process::Command::new("git")
                    .args(["config", key, value])
                    .current_dir(dir.path())
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let graph = crate::change_graph::ChangeGraph::open(dir.path()).unwrap();
        let node = ast.details["graph_node"].as_str().unwrap();
        graph
            .mark(node, "checkpoint", "feat: selected AST edit", true)
            .unwrap();
        let exported = graph.export(node, "feature/selected-edit").unwrap();
        for (file, expected) in [
            ("f.rs", "fn first() { 1; }\nfn second() { 3; }\n"),
            ("unrelated.txt", "human edit\n"),
        ] {
            let shown = std::process::Command::new("git")
                .args(["show", &format!("{}:{file}", exported.tip)])
                .current_dir(dir.path())
                .output()
                .unwrap();
            assert!(shown.status.success());
            assert_eq!(String::from_utf8(shown.stdout).unwrap(), expected);
        }
        let err = tool
            .execute(
                json!({
                    "path": "f.rs", "selection": {"mode": "ast", "query": "(function_item) @edit"},
                    "replacement": ""
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(
            err.contains("ambig") || err.contains("unique") || err.contains("2"),
            "{err}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn first() { 4; }\nfn second() { 3; }\n"
        );
    }

    #[tokio::test]
    async fn mutations_fail_closed_without_git_and_outside_workspace() {
        let plain = tempfile::tempdir().unwrap();
        let tool = WriteTool {
            cwd: plain.path().to_path_buf(),
        };
        assert!(
            tool.execute(
                json!({"path": "f.txt", "content": "bad"}),
                ctx(plain.path())
            )
            .await
            .is_err()
        );
        assert!(!plain.path().join("f.txt").exists());
        let repo = git_dir();
        let tool = WriteTool {
            cwd: repo.path().to_path_buf(),
        };
        let outside = plain.path().join("outside.txt");
        assert!(
            tool.execute(json!({"path": outside, "content": "bad"}), ctx(repo.path()))
                .await
                .is_err()
        );
        assert!(!outside.exists());
        assert!(
            tool.execute(
                json!({"path": ".git/config", "content": "bad"}),
                ctx(repo.path())
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn range_edits_preserve_utf8_crlf_and_reject_phantom_lines() {
        let dir = git_dir();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "α\r\nβ\r\nlast").unwrap();
        let tool = EditTool {
            cwd: dir.path().to_path_buf(),
        };
        tool.execute(json!({"path":"f.txt","selection":{"mode":"range","start_line":2,"end_line":2},"replacement":"γ\r\n"}),ctx(dir.path())).await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "α\r\nγ\r\nlast");
        tool.execute(json!({"path":"f.txt","selection":{"mode":"range","start_line":3,"end_line":3},"replacement":"end\n"}),ctx(dir.path())).await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "α\r\nγ\r\nend\n");
        for (start, end) in [(0, 1), (2, 1), (4, 4)] {
            assert!(tool.execute(json!({"path":"f.txt","selection":{"mode":"range","start_line":start,"end_line":end},"replacement":"bad"}),ctx(dir.path())).await.is_err());
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "α\r\nγ\r\nend\n");
        }
    }

    #[tokio::test]
    async fn ast_edits_use_each_language_grammar_and_fail_without_fallback() {
        let dir = git_dir();
        let tool = EditTool {
            cwd: dir.path().to_path_buf(),
        };
        for (path, source, query, expected) in [
            (
                "f.py",
                "def old():\n    return 1\n",
                "(function_definition name: (identifier) @edit)",
                "def new():\n    return 1\n",
            ),
            (
                "f.js",
                "function old() { return 1; }\n",
                "(function_declaration name: (identifier) @edit)",
                "function new() { return 1; }\n",
            ),
            (
                "f.ts",
                "function old(): number { return 1; }\n",
                "(function_declaration name: (identifier) @edit)",
                "function new(): number { return 1; }\n",
            ),
            (
                "f.tsx",
                "const old = <div />;\n",
                "(variable_declarator name: (identifier) @edit)",
                "const new = <div />;\n",
            ),
        ] {
            std::fs::write(dir.path().join(path), source).unwrap();
            tool.execute(
                json!({"path":path,"selection":{"mode":"ast","query":query},"replacement":"new"}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
            assert_eq!(
                std::fs::read_to_string(dir.path().join(path)).unwrap(),
                expected
            );
        }
        for (path, source, query) in [
            ("bad.rs", "fn broken( {", "(function_item) @edit"),
            ("plain.txt", "old", "(_) @edit"),
            ("query.rs", "fn old() {}", "(not_a_node) @edit"),
            ("capture.rs", "fn old() {}", "(function_item) @other"),
        ] {
            std::fs::write(dir.path().join(path), source).unwrap();
            assert!(tool.execute(json!({"path":path,"selection":{"mode":"ast","query":query},"replacement":"bad"}),ctx(dir.path())).await.is_err());
            assert_eq!(
                std::fs::read_to_string(dir.path().join(path)).unwrap(),
                source
            );
        }
    }

    #[tokio::test]
    async fn write_noop_and_cancellation_do_not_create_edits() {
        let dir = git_dir();
        let tool = WriteTool {
            cwd: dir.path().to_path_buf(),
        };
        let first = tool
            .execute(json!({"path":"f.txt","content":"kept"}), ctx(dir.path()))
            .await
            .unwrap();
        let noop = tool
            .execute(json!({"path":"f.txt","content":"kept"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(noop.details["no_op"], true);
        assert_eq!(noop.details["after_tree"], first.details["after_tree"]);
        let context = ctx(dir.path());
        context.cancel.cancel();
        assert!(
            tool.execute(json!({"path":"f.txt","content":"bad"}), context)
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "kept"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn edits_preserve_executable_mode_and_reject_symlinks() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let dir = git_dir();
        let path = dir.path().join("script");
        std::fs::write(&path, "echo old\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let tool = WriteTool {
            cwd: dir.path().to_path_buf(),
        };
        tool.execute(
            json!({"path":"script","content":"echo new\n"}),
            ctx(dir.path()),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o755
        );
        symlink(&path, dir.path().join("link")).unwrap();
        assert!(
            tool.execute(json!({"path":"link","content":"bad"}), ctx(dir.path()))
                .await
                .is_err()
        );
        assert_eq!(std::fs::read_to_string(path).unwrap(), "echo new\n");
    }

    #[tokio::test]
    async fn glob_matches_sorted_respects_gitignore_and_hidden() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        std::fs::write(dir.path().join("src/b.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/nested/c.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/ignored.rs"), "").unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(dir.path().join(".hidden.rs"), "").unwrap();
        let tool = GlobTool {
            cwd: dir.path().to_path_buf(),
        };

        let out = tool
            .execute(
                json!({"pattern": "**/*.rs", "path": "src"}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        let body = text(&out);
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(
            lines,
            ["src/a.rs", "src/b.rs", "src/nested/c.rs"],
            "{lines:?}"
        );

        // Basename pattern matches at any depth.
        let out = tool
            .execute(json!({"pattern": "*.rs", "path": "src"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(text(&out).lines().count(), 3);

        // gitignore=false surfaces the ignored file.
        let out = tool
            .execute(
                json!({"pattern": "ignored.rs", "path": "src", "gitignore": false}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert!(text(&out).contains("ignored.rs"));

        // Hidden excluded by default, included on request.
        let out = tool
            .execute(json!({"pattern": ".hidden.rs"}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(!text(&out).contains(".hidden.rs"));
        let out = tool
            .execute(
                json!({"pattern": ".hidden.rs", "hidden": true}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert!(text(&out).contains(".hidden.rs"));

        let err = tool
            .execute(json!({"pattern": "["}), ctx(dir.path()))
            .await
            .unwrap_err();
        assert!(err.contains("invalid glob"), "{err}");
    }

    #[tokio::test]
    async fn glob_limit_reports_truncation() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..5 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "").unwrap();
        }
        let tool = GlobTool {
            cwd: dir.path().to_path_buf(),
        };
        let out = tool
            .execute(json!({"pattern": "*.txt", "limit": 2}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(out.details["total"], 5);
        assert_eq!(out.details["matches"], 2);
        assert_eq!(out.details["truncated"], true);
    }

    #[tokio::test]
    async fn grep_reports_path_line_text_and_case() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "Hello world\nbye\n").unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/b.txt"), "say hello again\n").unwrap();
        let tool = GrepTool {
            cwd: dir.path().to_path_buf(),
        };

        // Case-sensitive by default: only sub/b.txt matches.
        let out = tool
            .execute(json!({"pattern": "hello"}), ctx(dir.path()))
            .await
            .unwrap();
        let body = text(&out);
        assert!(body.contains("sub/b.txt:1: say hello again"), "{body}");
        assert!(!body.contains("Hello world"), "{body}");

        let out = tool
            .execute(json!({"pattern": "hello", "case": false}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(
            text(&out).contains("a.txt:1: Hello world"),
            "{}",
            text(&out)
        );

        // Single-file search.
        let out = tool
            .execute(json!({"pattern": "bye", "path": "a.txt"}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(text(&out).contains("a.txt:2: bye"));

        let err = tool
            .execute(json!({"pattern": "("}), ctx(dir.path()))
            .await
            .unwrap_err();
        assert!(err.contains("invalid regex"), "{err}");
    }

    #[tokio::test]
    async fn grep_skips_binary_and_honors_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("bin.dat"), b"hit\0hit").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "hit\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "hit\n").unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        let tool = GrepTool {
            cwd: dir.path().to_path_buf(),
        };
        let out = tool
            .execute(json!({"pattern": "hit"}), ctx(dir.path()))
            .await
            .unwrap();
        let body = text(&out);
        assert!(body.contains("kept.txt:1: hit"), "{body}");
        assert!(!body.contains("ignored.txt"), "{body}");
        assert!(!body.contains("bin.dat"), "{body}");
        assert_eq!(out.details["binaryFilesSkipped"], 1);
    }

    #[tokio::test]
    async fn bash_captures_output_and_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: None,
        };
        let out = tool
            .execute(json!({"command": "echo hi; echo err >&2"}), ctx(dir.path()))
            .await
            .unwrap();
        let body = text(&out);
        assert!(body.contains("hi"), "{body}");
        assert!(body.contains("err"), "{body}");
        assert!(!out.is_error);
        assert_eq!(out.details["exitCode"], 0);

        let out = tool
            .execute(json!({"command": "exit 7"}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(out.is_error);
        assert_eq!(out.details["exitCode"], 7);
        assert!(text(&out).contains("exit code 7"));
    }

    #[tokio::test]
    async fn governed_bash_runs_under_a_systemd_scope() {
        if !crate::workspace::is_systemd_run_available() {
            eprintln!("skipping governed-bash test: systemd-run unavailable");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: Some(ResourceLimits::default().with_enforce_scope(false)),
        };
        // A scoped command runs in its own cgroup, so its cgroup path names a
        // scope rather than the test process's.
        let out = tool
            .execute(json!({"command": "cat /proc/self/cgroup"}), ctx(dir.path()))
            .await
            .unwrap();
        let body = text(&out);
        assert!(
            body.contains(".scope"),
            "expected a systemd scope cgroup, got: {body}"
        );
    }

    /// Live proof that a timed-out sandboxed command leaves no unit behind.
    ///
    /// A sandboxed command runs as a transient service that outlives its client,
    /// so killing the child alone would leave `sleep` running.
    #[tokio::test]
    async fn a_timed_out_sandboxed_bash_command_leaves_no_unit() {
        if !crate::workspace::is_systemd_run_available() {
            eprintln!("skipping sandboxed-bash test: systemd-run unavailable");
            return;
        }
        let Some(home) = std::env::var_os("HOME")
            .filter(|home| !home.is_empty())
            .map(PathBuf::from)
        else {
            eprintln!("skipping sandboxed-bash test: HOME unset");
            return;
        };
        // `PrivateTmp=yes` hides /tmp, so the workspace must live under $HOME.
        let dir = match tempfile::Builder::new()
            .prefix("juto-bash-sandbox-")
            .tempdir_in(&home)
        {
            Ok(dir) => dir,
            Err(err) => {
                eprintln!("skipping sandboxed-bash test: {err}");
                return;
            }
        };

        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: Some(
                ResourceLimits::default().with_sandbox(crate::workspace::Sandbox::default()),
            ),
        };
        let out = tool
            .execute(
                json!({"command": "sleep 47", "timeout": 1}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert_eq!(out.details["timedOut"], true, "{:?}", out.details);

        // The guard stops the unit on drop; the stop is fire-and-forget, so poll
        // for the payload itself. Scanning `juto-*` units would also match
        // unrelated ones such as the desktop subsystem's cleanup slices.
        let mut leftover = String::new();
        for _ in 0..50 {
            let ps = std::process::Command::new("pgrep")
                .args(["-f", "sleep 47"])
                .output()
                .expect("pgrep");
            leftover = String::from_utf8_lossy(&ps.stdout).trim().to_string();
            if leftover.is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert!(
            leftover.is_empty(),
            "a timed-out sandboxed command must not survive: {leftover}"
        );
    }

    #[test]
    fn vcs_tool_description_covers_every_action() {
        let tool = VcsTool {
            cwd: PathBuf::from("."),
        };
        let description = tool.definition().description;
        for action in [
            "status",
            "fetch",
            "reconcile",
            "push",
            "verify",
            "mark",
            "rewind",
            "export",
            "publish",
            "fetch_branch",
            "join",
            "resolve",
            "dirty",
            "capture",
            "log",
            "gc",
        ] {
            assert!(
                description.contains(action),
                "the vcs tool description must mention `{action}`: {description}"
            );
        }
    }

    #[tokio::test]
    async fn vcs_tool_reports_status_and_verifies() {
        let dir = git_dir();
        let graph = ChangeGraph::open(dir.path()).expect("init graph");
        let tool = VcsTool {
            cwd: dir.path().to_path_buf(),
        };

        let out = tool
            .execute(json!({"action": "status"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(out.details["nodes"], 0, "{:?}", out.details);
        assert_eq!(out.details["needs_reconcile"], false);
        assert!(!out.is_error);

        let out = tool
            .execute(json!({"action": "verify"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(out.details["ok"], true, "{:?}", out.details);

        // `mark` defaults to the thread head, and `rewind` returns to it.
        graph
            .capture(
                "main",
                EditProvenance {
                    session_id: "test".into(),
                    tool: "test".into(),
                    selection_mode: None,
                    path: None,
                },
                None,
            )
            .expect("capture");
        let out = tool
            .execute(
                json!({"action": "mark", "name": "c1", "message": "feat: c1"}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert!(out.details["id"].is_string(), "{:?}", out.details);
        assert_eq!(
            out.details["mark"]["message"], "feat: c1",
            "the commit message must be the one the caller gave: {:?}",
            out.details
        );

        let out = tool
            .execute(json!({"action": "rewind", "point": "c1"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(
            out.details["provenance"]["tool"], "rewind",
            "{:?}",
            out.details
        );

        // `export` projects the marked point into a Git commit on a branch.
        let out = tool
            .execute(
                json!({"action": "export", "point": "c1", "branch": "juto-test"}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert!(out.details["tip"].is_string(), "{:?}", out.details);
        assert!(
            out.details["commits"]
                .as_object()
                .is_some_and(|commits| !commits.is_empty()),
            "{:?}",
            out.details
        );

        // `join` merges two threads into a two-parent node. Each capture must
        // change the workspace: capturing an unchanged one is a no-op that does
        // not create a thread head.
        for (thread, content) in [("left", "left\n"), ("right", "right\n")] {
            std::fs::write(dir.path().join(format!("{thread}.txt")), content).expect("write");
            graph
                .capture(
                    thread,
                    EditProvenance {
                        session_id: "test".into(),
                        tool: "test".into(),
                        selection_mode: None,
                        path: None,
                    },
                    None,
                )
                .expect("capture thread");
        }
        let heads = graph.state().expect("state").heads;
        let out = tool
            .execute(
                json!({
                    "action": "join",
                    "thread": "integration",
                    "left": heads["left"],
                    "right": heads["right"],
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert_eq!(out.details["status"], "joined", "{:?}", out.details);
        assert_eq!(
            out.details["node"]["parents"].as_array().map(Vec::len),
            Some(2),
            "{:?}",
            out.details
        );

        // `resolve` refuses a workspace with uncaptured edits outside the
        // conflict, rather than materializing over them.
        std::fs::write(dir.path().join("stray.txt"), "stray\n").expect("write");
        assert!(
            tool.execute(
                json!({
                    "action": "resolve",
                    "thread": "integration",
                    "left": heads["left"],
                    "right": heads["right"],
                }),
                ctx(dir.path()),
            )
            .await
            .is_err(),
            "resolve must refuse a stray uncaptured edit"
        );
        std::fs::remove_file(dir.path().join("stray.txt")).expect("remove");

        // `resolve` records the workspace as a two-parent node, the way an
        // agent records a conflict it resolved by hand.
        let out = tool
            .execute(
                json!({
                    "action": "resolve",
                    "thread": "integration",
                    "left": heads["left"],
                    "right": heads["right"],
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert_eq!(
            out.details["parents"].as_array().map(Vec::len),
            Some(2),
            "{:?}",
            out.details
        );

        // `dirty` reports an uncaptured edit and `capture` records it, so an
        // agent can close the loop a `bash` command leaves open.
        std::fs::write(dir.path().join("shell.txt"), "shell\n").expect("write");
        let out = tool
            .execute(json!({"action": "dirty"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(out.details["dirty"], true, "{:?}", out.details);
        assert_eq!(
            out.details["changed"],
            json!(["shell.txt"]),
            "{:?}",
            out.details
        );

        let out = tool
            .execute(
                json!({"action": "capture", "thread": "integration"}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert!(
            out.details["id"].as_str().is_some(),
            "capture must return the new node: {:?}",
            out.details
        );
        let out = tool
            .execute(json!({"action": "dirty"}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(out.details["dirty"], false, "{:?}", out.details);

        // `log` prints the graph, bounded by `limit`.
        let out = tool
            .execute(json!({"action": "log", "limit": 1}), ctx(dir.path()))
            .await
            .unwrap();
        assert_eq!(
            out.details["nodes"].as_array().map(Vec::len),
            Some(1),
            "{:?}",
            out.details
        );
        assert!(
            out.details["total"].as_u64().is_some_and(|total| total > 1),
            "{:?}",
            out.details
        );
        assert!(
            out.details["heads"].as_object().is_some(),
            "{:?}",
            out.details
        );
        assert!(
            out.details["base_commit"]
                .as_str()
                .is_some_and(|c| !c.is_empty()),
            "log must report the graph's base anchor: {:?}",
            out.details
        );

        // `gc` reclaims and reports what it kept.
        let out = tool
            .execute(json!({"action": "gc"}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(
            out.details["retained_nodes"].as_u64().is_some(),
            "{:?}",
            out.details
        );

        // An unknown action is rejected rather than silently ignored.
        assert!(
            tool.execute(json!({"action": "nope"}), ctx(dir.path()))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn bash_runs_in_cwd_and_honors_cwd_param() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: None,
        };
        let out = tool
            .execute(json!({"command": "pwd", "cwd": "sub"}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(text(&out).contains("/sub"), "{}", text(&out));
        let err = tool
            .execute(
                json!({"command": "true", "cwd": "missing"}),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("not a directory"), "{err}");
    }

    #[tokio::test]
    async fn bash_timeout_kills_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let sentinel = dir.path().join("sentinel");
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: None,
        };
        // Grandchild survives the shell if only the leader is killed.
        let command = format!("(sleep 2; touch {}) & wait", sentinel.display());
        let start = Instant::now();
        let out = tool
            .execute(json!({"command": command, "timeout": 1}), ctx(dir.path()))
            .await
            .unwrap();
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "timeout should kill quickly"
        );
        assert!(out.is_error);
        assert_eq!(out.details["timedOut"], true);
        // If the group was killed, the background subshell never creates the file.
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(!sentinel.exists(), "process-group child leaked past kill");
    }

    #[tokio::test]
    async fn bash_cancellation_kills_command() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: None,
        };
        let mut context = ctx(dir.path());
        context.cancel = cancel.clone();
        let spawned =
            tokio::spawn(
                async move { tool.execute(json!({"command": "sleep 60"}), context).await },
            );
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
        let out = spawned.await.unwrap().unwrap();
        assert!(out.is_error);
        assert_eq!(out.details["cancelled"], true);
    }

    #[tokio::test]
    async fn bash_output_is_bounded_with_truncation_marker() {
        let dir = tempfile::tempdir().unwrap();
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
            limits: None,
        };
        let out = tool
            .execute(
                json!({"command": "yes 0123456789 | head -c 200000"}),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(out.details["truncated"], true);
        let body = text(&out);
        assert!(body.contains("truncated"), "{body}");
        assert!(body.len() < 60 * 1024, "bounded: {} bytes", body.len());
    }

    #[tokio::test]
    async fn tools_resolve_against_ctx_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("x.txt"), "other\n").unwrap();
        let tool = ReadTool {
            cwd: dir.path().to_path_buf(),
        };
        // ctx.cwd overrides the tool's baked cwd.
        let out = tool
            .execute(json!({"path": "x.txt"}), ctx(other.path()))
            .await
            .unwrap();
        assert!(text(&out).contains("1: other"));
    }

    #[tokio::test]
    async fn desktop_artifact_tool_validates_and_generates_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let tool = DesktopArtifactTool;

        // Missing required field rejected
        let err = tool
            .execute(
                json!({
                    "title": "My Game",
                    "host_id": "host-1",
                    "environment_id": "env-1"
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("invalid desktop_artifact arguments"), "{err}");

        // Empty field rejected
        let err = tool
            .execute(
                json!({
                    "title": "",
                    "host_id": "host-1",
                    "environment_id": "env-1",
                    "grant_id": "grant-1"
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("title must not be empty"), "{err}");

        // Successful execution with explicit artifact_id
        let out = tool
            .execute(
                json!({
                    "title": "Interactive Game Desktop",
                    "host_id": "host-us-central",
                    "environment_id": "env-prod-99",
                    "grant_id": "grant-view-play",
                    "artifact_id": "custom-art-123"
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap();

        assert!(!out.is_error);
        assert_eq!(out.details["type"], "desktop_session_artifact");
        let art: DesktopSessionArtifact =
            serde_json::from_value(out.details["artifact"].clone()).unwrap();
        assert_eq!(art.version, DESKTOP_SESSION_ARTIFACT_VERSION);
        assert_eq!(art.artifact_id, "custom-art-123");
        assert_eq!(art.title, "Interactive Game Desktop");
        assert_eq!(art.host_id, "host-us-central");
        assert_eq!(art.environment_id, "env-prod-99");
        assert_eq!(art.grant_id, "grant-view-play");

        // Successful execution generating uuid v7 artifact_id
        let out2 = tool
            .execute(
                json!({
                    "title": "Auto ID Game",
                    "host_id": "host-2",
                    "environment_id": "env-2",
                    "grant_id": "grant-2"
                }),
                ctx(dir.path()),
            )
            .await
            .unwrap();
        let art2: DesktopSessionArtifact =
            serde_json::from_value(out2.details["artifact"].clone()).unwrap();
        assert!(
            art2.artifact_id.starts_with("desk-"),
            "generated: {}",
            art2.artifact_id
        );
    }

    #[tokio::test]
    async fn desktop_artifact_resolver_validates_host_environment_and_grant() {
        use crate::desktop_artifact::{DesktopArtifactResolver, ResolveError, ViewerRoute};

        let supervisor = Arc::new(juto_desktop::DesktopSupervisor::new());
        let signaling = Arc::new(crate::SignalingHandle::new(supervisor.clone()));
        let data_dir = tempfile::tempdir().unwrap();
        let tool = DesktopTool {
            supervisor: supervisor.clone(),
            signaling: signaling.clone(),
            data_dir: data_dir.path().to_path_buf(),
        };
        let ctx = ctx(data_dir.path());

        let created = tool
            .execute(
                json!({"action": "create", "title": "artifact"}),
                ctx.clone(),
            )
            .await
            .unwrap();
        let env_id = created.details["envId"].as_str().unwrap().to_string();
        let grant_id = created.details["grantId"].as_str().unwrap().to_string();
        let host_id = created.details["hostId"].as_str().unwrap().to_string();

        let resolver =
            DesktopArtifactResolver::new(supervisor.clone(), signaling.clone(), host_id.clone());
        let artifact = DesktopSessionArtifact {
            version: DESKTOP_SESSION_ARTIFACT_VERSION,
            artifact_id: "art-1".to_string(),
            title: "artifact".to_string(),
            host_id: host_id.clone(),
            environment_id: env_id.clone(),
            grant_id: grant_id.clone(),
        };

        // A valid descriptor resolves to the same-host route.
        let route = resolver.resolve("agent:session", &artifact).unwrap();
        assert!(matches!(route, ViewerRoute::Local { .. }), "{route:?}");

        // The remote route issues a one-time token.
        let route = resolver.resolve_remote("agent:session", &artifact).unwrap();
        match route {
            ViewerRoute::Remote { url, token, .. } => {
                assert!(url.starts_with("http://127.0.0.1:"), "{url}");
                assert!(token.starts_with("attach-"), "{token}");
            }
            other => panic!("expected a remote route, got {other:?}"),
        }

        // Wrong host, environment, grant, and principal are distinguished.
        let mut wrong_host = artifact.clone();
        wrong_host.host_id = "other-host".to_string();
        assert!(matches!(
            resolver.resolve("agent:session", &wrong_host),
            Err(ResolveError::Unsupported(_))
        ));

        let mut wrong_env = artifact.clone();
        wrong_env.environment_id = "missing".to_string();
        assert!(matches!(
            resolver.resolve("agent:session", &wrong_env),
            Err(ResolveError::Unavailable(_))
        ));

        let mut wrong_grant = artifact.clone();
        wrong_grant.grant_id = "missing".to_string();
        assert!(matches!(
            resolver.resolve("agent:session", &wrong_grant),
            Err(ResolveError::Unauthorized(_))
        ));

        assert!(matches!(
            resolver.resolve("human:other", &artifact),
            Err(ResolveError::Unauthorized(_))
        ));

        // An expired grant is reported as expired, not unauthorized.
        let env = supervisor.get_environment(&env_id).unwrap();
        let expired = env.issue_grant(
            "agent:session".to_string(),
            vec![juto_desktop::protocol::GrantRole::View],
            "session".to_string(),
            0,
        );
        let mut expired_artifact = artifact.clone();
        expired_artifact.grant_id = expired.grant_id.clone();
        assert!(matches!(
            resolver.resolve("agent:session", &expired_artifact),
            Err(ResolveError::Expired(_))
        ));

        // A destroyed environment is reported as stopped.
        env.destroy();
        assert!(matches!(
            resolver.resolve("agent:session", &artifact),
            Err(ResolveError::Stopped(_))
        ));
    }

    #[tokio::test]
    async fn desktop_tool_creates_launches_observes_and_destroys() {
        let supervisor = Arc::new(juto_desktop::DesktopSupervisor::new());
        let data_dir = tempfile::tempdir().unwrap();
        let tool = DesktopTool {
            supervisor: supervisor.clone(),
            signaling: Arc::new(crate::SignalingHandle::new(supervisor.clone())),
            data_dir: data_dir.path().to_path_buf(),
        };
        let ctx = ctx(data_dir.path());

        let created = tool
            .execute(
                json!({"action": "create", "title": "smoke", "width": 640, "height": 480, "fps": 30}),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(!created.is_error);
        let env_id = created.details["envId"].as_str().unwrap().to_string();
        assert!(env_id.starts_with("desk-"));
        assert!(
            created.details["grantId"]
                .as_str()
                .unwrap()
                .starts_with("grant-")
        );
        assert_eq!(created.details["output"]["width"], 640);

        let launched = tool
            .execute(
                json!({"action": "launch", "env_id": env_id, "app_id": "sleeper",
                       "argv": ["sleep", "60"]}),
                ctx.clone(),
            )
            .await
            .unwrap();
        assert!(launched.details["pid"].as_u64().unwrap() > 0);
        assert!(
            launched.details["unit"]
                .as_str()
                .unwrap()
                .ends_with(".scope")
        );

        let observed = tool
            .execute(
                json!({"action": "observe", "env_id": env_id, "timeout_ms": 5000}),
                ctx.clone(),
            )
            .await
            .unwrap();
        let snapshot = observed.details["snapshotPath"].as_str().unwrap();
        let bytes = std::fs::read(snapshot).unwrap();
        assert_eq!(
            &bytes[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
        assert!(bytes.len() > 100, "png is {} bytes", bytes.len());
        assert_eq!(observed.details["width"], 640);

        let described = tool
            .execute(json!({"action": "describe", "env_id": env_id}), ctx.clone())
            .await
            .unwrap();
        assert_eq!(described.details["apps"].as_array().unwrap().len(), 1);

        let attached = tool
            .execute(json!({"action": "attach", "env_id": env_id}), ctx.clone())
            .await
            .unwrap();
        let url = attached.details["viewerUrl"].as_str().unwrap();
        assert!(url.starts_with("http://127.0.0.1:"), "{url}");
        assert!(
            attached.details["token"]
                .as_str()
                .unwrap()
                .starts_with("attach-")
        );
        assert_eq!(attached.details["transport"], "webrtc");

        tool.execute(json!({"action": "destroy", "env_id": env_id}), ctx.clone())
            .await
            .unwrap();
        assert!(supervisor.get_environment(&env_id).is_err());

        // Unknown action and missing env_id are rejected.
        assert!(
            tool.execute(json!({"action": "frobnicate"}), ctx.clone())
                .await
                .is_err()
        );
        assert!(
            tool.execute(json!({"action": "describe"}), ctx.clone())
                .await
                .is_err()
        );
    }
}
