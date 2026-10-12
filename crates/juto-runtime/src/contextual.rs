//! Contextual edit selections: exact text, line range, or tree-sitter AST query.
//!
//! [`resolve`] turns a [`Selection`] into the byte ranges that a caller replaces
//! with the replacement string. Ranges are sorted, non-overlapping, and always
//! fall on UTF-8 boundaries because they are derived from `&str` indices or
//! tree-sitter node byte ranges.

use std::ops::Range;
use std::path::Path;

use serde::Deserialize;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

/// How to locate the text to replace inside a file.
///
/// Deserialized from a JSON object tagged by `mode` (`text`, `range`, or
/// `ast`). Unknown fields are rejected so a typo cannot silently change the
/// selection.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Selection {
    /// Replace an exact substring.
    Text {
        /// Literal text to find. Must be non-empty.
        old_string: String,
        /// Replace every occurrence instead of requiring a unique match.
        #[serde(default)]
        replace_all: bool,
    },
    /// Replace an inclusive, 1-based range of whole lines.
    Range {
        /// First line to replace (1-based, inclusive).
        start_line: usize,
        /// Last line to replace (1-based, inclusive).
        end_line: usize,
    },
    /// Replace the node(s) captured by a tree-sitter query.
    Ast {
        /// Tree-sitter query source. The grammar is chosen from the file extension.
        query: String,
        /// Capture name whose node is replaced. Defaults to `edit`.
        #[serde(default = "default_capture")]
        capture: String,
    },
}

fn default_capture() -> String {
    "edit".to_string()
}

/// Resolve `selection` against `text`, returning the byte ranges to replace.
///
/// The returned ranges are sorted ascending and non-overlapping. `path` is used
/// only to pick the AST grammar from its extension.
pub(crate) fn resolve(
    path: &Path,
    text: &str,
    selection: &Selection,
) -> Result<Vec<Range<usize>>, String> {
    match selection {
        Selection::Text {
            old_string,
            replace_all,
        } => resolve_text(text, old_string, *replace_all),
        Selection::Range {
            start_line,
            end_line,
        } => resolve_range(text, *start_line, *end_line),
        Selection::Ast { query, capture } => resolve_ast(path, text, query, capture),
    }
}

fn resolve_text(
    text: &str,
    old_string: &str,
    replace_all: bool,
) -> Result<Vec<Range<usize>>, String> {
    if old_string.is_empty() {
        return Err("text: old_string must not be empty".to_string());
    }
    let ranges: Vec<Range<usize>> = text
        .match_indices(old_string)
        .map(|(start, matched)| start..start + matched.len())
        .collect();
    if ranges.is_empty() {
        return Err("text: old_string not found".to_string());
    }
    if ranges.len() > 1 && !replace_all {
        return Err(format!(
            "text: old_string matches {} times; make it unique or set replace_all",
            ranges.len()
        ));
    }
    Ok(ranges)
}

fn resolve_range(
    text: &str,
    start_line: usize,
    end_line: usize,
) -> Result<Vec<Range<usize>>, String> {
    if start_line == 0 {
        return Err("range: start_line is 1-based and must be >= 1".to_string());
    }
    if end_line < start_line {
        return Err(format!(
            "range: end_line ({end_line}) must be >= start_line ({start_line})"
        ));
    }
    let mut start = None;
    let mut end = 0;
    let mut count = 0;
    for line in text.split_inclusive('\n') {
        count += 1;
        if count == start_line {
            start = Some(end);
        }
        end += line.len();
        if count == end_line {
            return Ok(std::iter::once(start.expect("validated line ordering")..end).collect());
        }
    }
    Err(format!(
        "range: requested lines {start_line}..={end_line} exceed file's {count} lines"
    ))
}

fn resolve_ast(
    path: &Path,
    text: &str,
    query_source: &str,
    capture: &str,
) -> Result<Vec<Range<usize>>, String> {
    let language = language_for(path)?;
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| format!("ast: failed to load grammar: {e}"))?;
    let tree = parser
        .parse(text, None)
        .ok_or_else(|| "ast: parser produced no tree".to_string())?;
    if tree.root_node().has_error() {
        return Err("ast: source contains parse errors".to_string());
    }
    let query =
        Query::new(&language, query_source).map_err(|e| format!("ast: invalid query: {e}"))?;
    let capture_index = query
        .capture_index_for_name(capture)
        .ok_or_else(|| format!("ast: query does not capture '{capture}'"))?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), text.as_bytes());
    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut saw_capture = false;
    while let Some(matched) = matches.next() {
        for cap in matched.captures() {
            if cap.index == capture_index {
                saw_capture = true;
                let range = cap.node.byte_range();
                if range.start < range.end {
                    ranges.push(range);
                }
            }
        }
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    ranges.dedup();
    match ranges.len() {
        0 if !saw_capture => Err(format!(
            "ast: query produced no match for capture '{capture}'"
        )),
        0 => Err(format!("ast: capture '{capture}' matched only empty nodes")),
        1 => Ok(ranges),
        n => Err(format!(
            "ast: capture '{capture}' matched {n} distinct nodes; the query must be unambiguous"
        )),
    }
}

/// Select the tree-sitter grammar for `path` from its extension.
///
/// Unknown or missing extensions are an explicit error; there is no fallback.
fn language_for(path: &Path) -> Result<Language, String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);
    let language = match extension.as_deref() {
        Some("rs") => tree_sitter_rust::LANGUAGE.into(),
        Some("py" | "pyi") => tree_sitter_python::LANGUAGE.into(),
        Some("js" | "jsx" | "mjs" | "cjs") => tree_sitter_javascript::LANGUAGE.into(),
        Some("ts" | "mts" | "cts") => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Some("tsx") => tree_sitter_typescript::LANGUAGE_TSX.into(),
        Some(other) => {
            return Err(format!(
                "ast: unsupported file extension '.{other}'; supported: rs, py, pyi, js, jsx, mjs, cjs, ts, mts, cts, tsx"
            ));
        }
        None => {
            return Err("ast: file has no extension; cannot determine a grammar".to_string());
        }
    };
    Ok(language)
}
