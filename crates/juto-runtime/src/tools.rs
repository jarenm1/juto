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
    io::{self, BufRead, Read as _, Write as _},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use globset::{Candidate, GlobBuilder};
use juto_agent::{
    AgentError, Tool, ToolConcurrency, ToolContext, ToolOutput, ToolRegistry, ToolTier,
};
use juto_ai::ContentBlock;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

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

/// Decode file bytes as UTF-8, rejecting binary data with an explicit error.
fn decode_file(cwd: &Path, path: &Path, bytes: &[u8]) -> Result<String, String> {
    let shown = display_path(cwd, path);
    if looks_binary(bytes) {
        return Err(format!(
            "{shown}: binary file (NUL byte in first {BINARY_SNIFF_BYTES} bytes)"
        ));
    }
    String::from_utf8(bytes.to_vec()).map_err(|e| format!("{shown}: not valid UTF-8 ({e})"))
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

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `contents` to `path` via a same-directory temp file + rename so a
/// failure never truncates the destination. Creates parent directories.
fn atomic_write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&parent)?;
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(
        ".{file_name}.juto-{}-{unique}.tmp",
        std::process::id()
    ));
    let result = (|| -> io::Result<()> {
        {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(contents)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
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
            description: "Write content to a file, creating parent directories. Overwrites \
				atomically so a failed write never truncates the destination."
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
        tokio::task::spawn_blocking(move || {
            atomic_write(&path, params.content.as_bytes())
                .map_err(|e| format!("{}: {e}", display_path(&cwd, &path)))?;
            Ok(output(
                format!(
                    "Wrote {} bytes to {}",
                    params.content.len(),
                    display_path(&cwd, &path)
                ),
                json!({"path": display_path(&cwd, &path), "bytes": params.content.len()}),
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
struct EditParams {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

struct EditTool {
    cwd: PathBuf,
}

#[async_trait]
impl Tool for EditTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "edit".to_string(),
            description: "Replace exact text in a file. old_string must match exactly once; \
				set replace_all to replace every occurrence. The file is updated atomically."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit; relative paths resolve against the session cwd."},
                    "old_string": {"type": "string", "description": "Exact text to find (must be non-empty)."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."},
                },
                "required": ["path", "old_string", "new_string"],
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
        if params.old_string.is_empty() {
            return Err("edit: old_string must not be empty".to_string());
        }
        check_cancel(&ctx.cancel)?;
        let cwd = effective_cwd(&self.cwd, &ctx).to_path_buf();
        let path = resolve_path(&cwd, &params.path);
        tokio::task::spawn_blocking(move || {
            let shown = display_path(&cwd, &path);
            let bytes = std::fs::read(&path).map_err(|e| format!("{shown}: {e}"))?;
            let text = decode_file(&cwd, &path, &bytes)?;
            let matches = text.matches(&params.old_string).count();
            if matches == 0 {
                return Err(format!("{shown}: old_string not found"));
            }
            if matches > 1 && !params.replace_all {
                return Err(format!(
                    "{shown}: old_string matches {matches} times; make it unique or set replace_all"
                ));
            }
            let updated = text.replace(&params.old_string, &params.new_string);
            atomic_write(&path, updated.as_bytes()).map_err(|e| format!("{shown}: {e}"))?;
            Ok(output(
                format!(
                    "Edited {shown} ({matches} replacement{})",
                    if matches == 1 { "" } else { "s" }
                ),
                json!({"path": shown, "replacements": matches}),
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
        let mut child = command
            .spawn()
            .map_err(|e| format!("failed to spawn `sh`: {e}"))?;
        let pid = child.id();
        let mut guard = ProcessGroupGuard::new(pid);

        // Drain both pipes concurrently into a shared rolling tail so slow
        // readers can never deadlock the child on a full pipe.
        let sink = Arc::new(Mutex::new(TailBuffer::new(BASH_OUTPUT_MAX_BYTES)));
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
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
// registry
// ---------------------------------------------------------------------------

/// Construct the standard tool set rooted at `cwd`.
///
/// Registers `read`, `write`, `edit`, `glob`, `grep`, and `bash`. Approval
/// policy is enforced by agent hooks via each tool's [`Tool::tier`]; these
/// tools perform real filesystem/process work and return true errors.
pub fn builtin_tools(cwd: &Path) -> Result<ToolRegistry, AgentError> {
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
    }))?;
    Ok(registry)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn registry_has_six_tools_with_schemas() {
        let dir = tempfile::tempdir().unwrap();
        let registry = builtin_tools(dir.path()).expect("builtin tools");
        let mut names: Vec<String> = registry
            .definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        names.sort();
        assert_eq!(names, ["bash", "edit", "glob", "grep", "read", "write"]);
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
                cwd: PathBuf::new()
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
                cwd: PathBuf::new()
            }
            .concurrency(),
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

    #[tokio::test]
    async fn write_creates_parents_and_overwrites() {
        let dir = tempfile::tempdir().unwrap();
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
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "foo bar foo").unwrap();
        let tool = EditTool {
            cwd: dir.path().to_path_buf(),
        };

        let err = tool
            .execute(
                json!({"path": "f.txt", "old_string": "foo", "new_string": "x"}),
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
                json!({"path": "f.txt", "old_string": "nope", "new_string": "x"}),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("not found"), "{err}");

        let out = tool.execute(json!({"path": "f.txt", "old_string": "foo", "new_string": "x", "replace_all": true}), ctx(dir.path())).await.unwrap();
        assert_eq!(out.details["replacements"], 2);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "x bar x"
        );

        let err = tool
            .execute(
                json!({"path": "f.txt", "old_string": "", "new_string": "x"}),
                ctx(dir.path()),
            )
            .await
            .unwrap_err();
        assert!(err.contains("empty"), "{err}");
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
    async fn bash_runs_in_cwd_and_honors_cwd_param() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        let tool = BashTool {
            cwd: dir.path().to_path_buf(),
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
}
