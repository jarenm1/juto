//! Pure helpers ported from oh-my-pi's browser tool at 579da1d6 (MIT, see
//! licenses/OMP-MIT.txt): selector parsing (`tern/selectors.ts`,
//! `aria/aria-snapshot.ts`), ARIA snapshot post-processing and diffs
//! (`snapshot-plus.ts`), Markdown section filters (`readable.ts`), key
//! resolution (`tern/keys.ts`), and allowed-domain matching.

use std::collections::HashMap;

use regex::Regex;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

/// Playwright-only engines no backend parses (`PLAYWRIGHT_ONLY_SELECTOR_RE`).
fn playwright_only(selector: &str) -> bool {
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r":has-text\(|:text\(|:text-is\(|:text-matches\(|:visible\b|:hidden\b|:nth-match\(|:near\(|:above\(|:below\(|:right-of\(|:left-of\(").unwrap()
    });
    RE.is_match(selector)
}

const ARIA_REF_PREFIXES: [&str; 3] = ["aria-ref=", "aria-ref/", "ariaref/"];
const LEGACY_PREFIXES: [&str; 4] = ["p-aria/", "p-text/", "p-xpath/", "p-pierce/"];
const PREFIXED_ENGINES: [&str; 9] = [
    "text",
    "xpath",
    "pierce",
    "label",
    "placeholder",
    "testid",
    "alt",
    "title",
    "role",
];

fn is_ref_id(id: &str) -> bool {
    id.len() > 1 && id.starts_with('e') && id[1..].bytes().all(|b| b.is_ascii_digit())
}

/// Recognize a snapshot-ref selector (`aria-ref=e5`, `aria-ref/e5`,
/// `ariaref/e5`, `e5`, `@e5`) and return the bare ref.
pub(crate) fn parse_aria_ref(selector: &str) -> Option<String> {
    let trimmed = selector.trim();
    for prefix in ARIA_REF_PREFIXES {
        if let Some(id) = trimmed.strip_prefix(prefix) {
            let id = id.trim();
            return is_ref_id(id).then(|| id.to_string());
        }
    }
    let bare = trimmed.strip_prefix('@').unwrap_or(trimmed);
    is_ref_id(bare).then(|| bare.to_string())
}

fn parse_aria_query(query: &str) -> Result<Value, String> {
    if query.len() > 10_000 {
        return Err(format!("Selector {query} is too long"));
    }
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r#"\[\s*(\w+)\s*=\s*(?:"((?:\\.|[^"])*)"|'((?:\\.|[^'])*)')\s*\]"#).unwrap()
    });
    let mut name: Option<String> = None;
    let mut role: Option<String> = None;
    for caps in RE.captures_iter(query) {
        let value = caps
            .get(2)
            .or_else(|| caps.get(3))
            .map_or("", |m| m.as_str())
            .to_string();
        match &caps[1] {
            "name" => name = Some(value),
            "role" => role = Some(value),
            other => return Err(format!("Unknown aria attribute \"{other}\" in selector")),
        }
    }
    let default_name = RE.replace_all(query, "").to_string();
    if !default_name.is_empty() && name.is_none() {
        name = Some(default_name);
    }
    let mut out = json!({ "engine": "aria" });
    if let Some(name) = name {
        out["name"] = Value::String(name);
    }
    if let Some(role) = role {
        out["role"] = Value::String(role);
    }
    Ok(out)
}

fn legacy_to_modern(selector: &str) -> String {
    if let Some(rest) = selector.strip_prefix("p-aria/") {
        static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
            Regex::new(r#"\[\s*name\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\]]+))\s*\]"#).unwrap()
        });
        let name = RE.captures(rest).and_then(|c| {
            c.get(1)
                .or_else(|| c.get(2))
                .or_else(|| c.get(3))
                .map(|m| m.as_str().trim().to_string())
        });
        return format!("aria/{}", name.unwrap_or_else(|| rest.to_string()));
    }
    selector["p-".len()..].to_string()
}

/// Parse a browser selector string into the page kit's JSON selector
/// (`parseTernSelector`).
pub(crate) fn parse_selector(selector: &str) -> Result<Value, String> {
    let engine_prefixed = selector.starts_with("aria/")
        || selector.starts_with("p-")
        || ARIA_REF_PREFIXES.iter().any(|p| selector.starts_with(p))
        || PREFIXED_ENGINES
            .iter()
            .any(|e| selector.starts_with(&format!("{e}/")));
    if !engine_prefixed && playwright_only(selector) {
        return Err(format!(
            "Playwright-only selector {} is not supported by the browser tool. Use a puppeteer text selector (\"text/Allow all\"), an aria selector (\"aria/Name\"), CSS, or \"xpath/...\".",
            json!(selector)
        ));
    }
    if selector.starts_with("p-") {
        if !LEGACY_PREFIXES.iter().any(|p| selector.starts_with(p)) {
            return Err(format!(
                "Unsupported selector prefix. Use CSS or puppeteer query handlers (aria/, text/, xpath/, pierce/). Got: {selector}"
            ));
        }
        return parse_selector(&legacy_to_modern(selector));
    }
    if let Some(reference) = parse_aria_ref(selector) {
        return Ok(json!({ "engine": "ariaRef", "ref": reference }));
    }
    if ARIA_REF_PREFIXES
        .iter()
        .any(|p| selector.trim().starts_with(p))
    {
        return Err(format!(
            "Invalid ARIA ref selector {}; expected e.g. \"aria-ref=e5\".",
            json!(selector)
        ));
    }
    if let Some(query) = selector.strip_prefix("aria/") {
        return parse_aria_query(query);
    }
    for engine in PREFIXED_ENGINES {
        if let Some(query) = selector
            .strip_prefix(engine)
            .and_then(|r| r.strip_prefix('/'))
        {
            return Ok(json!({ "engine": engine, "query": query }));
        }
    }
    Ok(json!({ "engine": "css", "query": selector }))
}

// ---------------------------------------------------------------------------
// ARIA snapshot post-processing (snapshot-plus.ts)
// ---------------------------------------------------------------------------

const INTERACTIVE_ROLES: [&str; 16] = [
    "button",
    "link",
    "textbox",
    "combobox",
    "listbox",
    "checkbox",
    "radio",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "slider",
    "switch",
    "option",
    "spinbutton",
    "searchbox",
];

/// Snapshot post-processing switches.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SnapshotOptions {
    pub interactive: bool,
    pub compact: bool,
    pub urls: bool,
}

struct SnapshotLine {
    text: String,
    indent: usize,
    role: Option<String>,
    name: String,
    children: Vec<SnapshotLine>,
    contains_interactive: bool,
}

fn line_indent(line: &str) -> usize {
    let mut width = 0;
    for c in line.chars() {
        match c {
            ' ' => width += 1,
            '\t' => width += 2,
            _ => break,
        }
    }
    width
}

fn parse_snapshot_line(text: &str) -> SnapshotLine {
    static DASH: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"^-[ \t]*").unwrap());
    static ROLE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"(?i)^([a-z][\w-]*)\b").unwrap());
    static ATTRS: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\[[^\]]*\]").unwrap());
    static COLON: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r":\s*$").unwrap());
    let body = DASH.replace(text.trim_start(), "").to_string();
    let role = ROLE.captures(&body).map(|c| c[1].to_ascii_lowercase());
    let rest = match &role {
        Some(role) => &body[role.len()..],
        None => body.as_str(),
    };
    let stripped = ATTRS.replace_all(rest, "");
    let stripped = COLON.replace(&stripped, "");
    let mut name = stripped.trim().to_string();
    if name == "\"\"" || name == "''" {
        name.clear();
    }
    let interactive = role
        .as_deref()
        .is_some_and(|r| INTERACTIVE_ROLES.contains(&r));
    SnapshotLine {
        text: text.to_string(),
        indent: line_indent(text),
        role,
        name,
        children: Vec::new(),
        contains_interactive: interactive,
    }
}

fn parse_snapshot(snapshot: &str) -> Vec<SnapshotLine> {
    // Build the tree with a stack of open nodes, then mark interactive containment.
    let mut roots: Vec<SnapshotLine> = Vec::new();
    let mut stack: Vec<SnapshotLine> = Vec::new();
    fn attach(roots: &mut Vec<SnapshotLine>, stack: &mut [SnapshotLine], node: SnapshotLine) {
        match stack.last_mut() {
            Some(parent) => parent.children.push(node),
            None => roots.push(node),
        }
    }
    for text in snapshot.split('\n') {
        if text.trim().is_empty() {
            continue;
        }
        let node = parse_snapshot_line(text);
        while stack.last().is_some_and(|top| top.indent >= node.indent) {
            let done = stack.pop().unwrap();
            attach(&mut roots, &mut stack, done);
        }
        stack.push(node);
    }
    while let Some(done) = stack.pop() {
        attach(&mut roots, &mut stack, done);
    }
    fn mark(node: &mut SnapshotLine) -> bool {
        for child in &mut node.children {
            if mark(child) {
                node.contains_interactive = true;
            }
        }
        node.contains_interactive
    }
    for root in &mut roots {
        mark(root);
    }
    roots
}

fn append_href(text: &str, role: Option<&str>, hrefs: &HashMap<String, String>) -> String {
    if role != Some("link") || text.contains("[href=") {
        return text.to_string();
    }
    static REF: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\[ref=(e\d+)\]").unwrap());
    let Some(href) = REF.captures(text).and_then(|c| hrefs.get(&c[1])) else {
        return text.to_string();
    };
    let suffix = format!("[href={}]", json!(href));
    let trimmed = text.trim_end();
    match trimmed.strip_suffix(':') {
        Some(head) => format!("{head} {suffix}:"),
        None => format!("{text} {suffix}"),
    }
}

fn render_snapshot(
    nodes: &[SnapshotLine],
    options: SnapshotOptions,
    hrefs: &HashMap<String, String>,
    indent: usize,
    out: &mut Vec<String>,
) {
    for node in nodes {
        if options.interactive && !node.contains_interactive {
            continue;
        }
        let promote = options.compact
            && matches!(node.role.as_deref(), Some("generic" | "none" | "group"))
            && node.name.is_empty()
            && !node.contains_interactive;
        if promote {
            render_snapshot(&node.children, options, hrefs, indent, out);
            continue;
        }
        let trimmed = node.text.trim_start();
        out.push(format!(
            "{}{}",
            " ".repeat(indent),
            append_href(trimmed, node.role.as_deref(), hrefs)
        ));
        if let Some(first) = node.children.first() {
            let step = first.indent.saturating_sub(node.indent).max(2);
            render_snapshot(&node.children, options, hrefs, indent + step, out);
        }
    }
}

/// Apply interactive, compact, and URL decorations to ARIA snapshot YAML.
pub(crate) fn post_process_snapshot(
    snapshot: &str,
    options: SnapshotOptions,
    hrefs: &HashMap<String, String>,
) -> String {
    if !options.interactive && !options.compact && !options.urls {
        return snapshot.to_string();
    }
    let roots = parse_snapshot(snapshot);
    let Some(first) = roots.first() else {
        return snapshot.to_string();
    };
    let empty = HashMap::new();
    let mut out = Vec::new();
    render_snapshot(
        &roots,
        options,
        if options.urls { hrefs } else { &empty },
        first.indent,
        &mut out,
    );
    out.join("\n")
}

/// One retained snapshot baseline.
#[derive(Debug, Clone)]
pub(crate) struct SnapshotBaseline {
    url: String,
    revision: u64,
    snapshot: String,
}

fn unified_replacement(old: &str, new: &str) -> String {
    const CONTEXT: usize = 3;
    let old_lines: Vec<&str> = old.split('\n').collect();
    let new_lines: Vec<&str> = new.split('\n').collect();
    let mut prefix = 0;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < old_lines.len() - prefix
        && suffix < new_lines.len() - prefix
        && old_lines[old_lines.len() - 1 - suffix] == new_lines[new_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let context_start = prefix.saturating_sub(CONTEXT);
    let old_change_end = old_lines.len() - suffix;
    let new_change_end = new_lines.len() - suffix;
    let context_end = CONTEXT.min(suffix);
    let old_end = old_change_end + context_end;
    let new_end = new_change_end + context_end;
    let mut lines = vec![format!(
        "@@ -{},{} +{},{} @@",
        context_start + 1,
        old_end - context_start,
        context_start + 1,
        new_end - context_start
    )];
    for line in &old_lines[context_start..prefix] {
        lines.push(format!(" {line}"));
    }
    for line in &old_lines[prefix..old_change_end] {
        lines.push(format!("-{line}"));
    }
    for line in &new_lines[prefix..new_change_end] {
        lines.push(format!("+{line}"));
    }
    for line in &old_lines[old_change_end..old_change_end + context_end] {
        lines.push(format!(" {line}"));
    }
    lines.join("\n")
}

/// Update one keyed baseline and return the revisioned result
/// (`diffAriaSnapshot`).
pub(crate) fn diff_snapshot(
    baselines: &mut HashMap<String, SnapshotBaseline>,
    key: &str,
    url: &str,
    snapshot: &str,
) -> Value {
    let Some(previous) = baselines.get(key).cloned() else {
        baselines.insert(
            key.to_string(),
            SnapshotBaseline {
                url: url.to_string(),
                revision: 1,
                snapshot: snapshot.to_string(),
            },
        );
        return json!({ "status": "full", "revision": 1, "snapshot": snapshot });
    };
    if previous.url == url && previous.snapshot == snapshot {
        return json!({ "status": "unchanged", "revision": previous.revision });
    }
    let revision = previous.revision + 1;
    baselines.insert(
        key.to_string(),
        SnapshotBaseline {
            url: url.to_string(),
            revision,
            snapshot: snapshot.to_string(),
        },
    );
    if previous.url != url {
        return json!({ "status": "full", "revision": revision, "snapshot": snapshot });
    }
    let delta = unified_replacement(&previous.snapshot, snapshot);
    if delta.len() >= snapshot.len() {
        return json!({ "status": "full", "revision": revision, "snapshot": snapshot });
    }
    json!({ "status": "delta", "revision": revision, "baseRevision": previous.revision, "delta": delta })
}

// ---------------------------------------------------------------------------
// Readable Markdown filters (readable.ts)
// ---------------------------------------------------------------------------

fn markdown_heading(line: &str) -> Option<(usize, &str)> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &line[hashes..];
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    let title = rest.trim();
    (!title.is_empty()).then_some((hashes, title))
}

/// Reduce Markdown to its heading lines.
pub(crate) fn markdown_outline(markdown: &str) -> String {
    markdown
        .split('\n')
        .filter(|line| markdown_heading(line).is_some())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Keep complete sections whose heading contains `filter` (case-insensitive).
pub(crate) fn filter_markdown_sections(markdown: &str, filter: &str) -> String {
    let needle = filter.trim().to_lowercase();
    if needle.is_empty() {
        return markdown.to_string();
    }
    let lines: Vec<&str> = markdown.split('\n').collect();
    let mut keep = vec![false; lines.len()];
    for index in 0..lines.len() {
        let Some((level, title)) = markdown_heading(lines[index]) else {
            continue;
        };
        if !title.to_lowercase().contains(&needle) {
            continue;
        }
        let mut end = index + 1;
        while end < lines.len() {
            if markdown_heading(lines[end]).is_some_and(|(next, _)| next <= level) {
                break;
            }
            end += 1;
        }
        keep[index..end].fill(true);
    }
    lines
        .iter()
        .zip(keep)
        .filter_map(|(line, keep)| keep.then_some(*line))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Strip Markdown decoration (`markdownToPlainText`).
pub(crate) fn markdown_to_plain_text(markdown: &str) -> String {
    static RULES: std::sync::LazyLock<[(Regex, &'static str); 5]> =
        std::sync::LazyLock::new(|| {
            [
                (Regex::new(r"(?m)^#{1,6}\s+").unwrap(), ""),
                (Regex::new(r"!\[([^\]]*)\]\([^)]*\)").unwrap(), "$1"),
                (Regex::new(r"\[([^\]]+)\]\([^)]*\)").unwrap(), "$1"),
                (Regex::new(r"(?m)^\s*[-*+]\s+").unwrap(), ""),
                (Regex::new(r"[*_`~]").unwrap(), ""),
            ]
        });
    let mut text = markdown.to_string();
    for (re, replacement) in RULES.iter() {
        text = re.replace_all(&text, *replacement).into_owned();
    }
    text.trim().to_string()
}

// ---------------------------------------------------------------------------
// Keys (tern/keys.ts, adapted to CDP Input.dispatchKeyEvent)
// ---------------------------------------------------------------------------

/// CDP modifier bits.
pub(crate) const MOD_ALT: u8 = 1;
pub(crate) const MOD_CTRL: u8 = 2;
pub(crate) const MOD_META: u8 = 4;
pub(crate) const MOD_SHIFT: u8 = 8;

/// A resolved key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Key {
    pub key: String,
    pub code: Option<String>,
    pub key_code: u32,
    pub modifier: u8,
}

fn named_key_code(name: &str) -> Option<u32> {
    Some(match name {
        "Backspace" => 8,
        "Tab" => 9,
        "Enter" => 13,
        "Shift" => 16,
        "Control" => 17,
        "Alt" => 18,
        "CapsLock" => 20,
        "Escape" => 27,
        "PageUp" => 33,
        "PageDown" => 34,
        "End" => 35,
        "Home" => 36,
        "ArrowLeft" => 37,
        "ArrowUp" => 38,
        "ArrowRight" => 39,
        "ArrowDown" => 40,
        "Insert" => 45,
        "Delete" => 46,
        "Meta" => 91,
        "ContextMenu" => 93,
        _ => {
            let n: u32 = name.strip_prefix('F')?.parse().ok()?;
            if (1..=24).contains(&n) {
                111 + n
            } else {
                return None;
            }
        }
    })
}

/// US-layout virtual key code and DOM code of one printable character.
fn char_key(c: char) -> (u32, Option<String>) {
    if c.is_ascii_alphabetic() {
        let upper = c.to_ascii_uppercase();
        return (upper as u32, Some(format!("Key{upper}")));
    }
    if c.is_ascii_digit() {
        return (c as u32, Some(format!("Digit{c}")));
    }
    let (code, name) = match c {
        ' ' => (32, "Space"),
        ')' => (48, "Digit0"),
        '!' => (49, "Digit1"),
        '@' => (50, "Digit2"),
        '#' => (51, "Digit3"),
        '$' => (52, "Digit4"),
        '%' => (53, "Digit5"),
        '^' => (54, "Digit6"),
        '&' => (55, "Digit7"),
        '*' => (56, "Digit8"),
        '(' => (57, "Digit9"),
        ';' | ':' => (186, "Semicolon"),
        '=' | '+' => (187, "Equal"),
        ',' | '<' => (188, "Comma"),
        '-' | '_' => (189, "Minus"),
        '.' | '>' => (190, "Period"),
        '/' | '?' => (191, "Slash"),
        '`' | '~' => (192, "Backquote"),
        '[' | '{' => (219, "BracketLeft"),
        '\\' | '|' => (220, "Backslash"),
        ']' | '}' => (221, "BracketRight"),
        '\'' | '"' => (222, "Quote"),
        _ => return (0, None),
    };
    (code, Some(name.to_string()))
}

/// Resolve one Puppeteer key name.
pub(crate) fn resolve_key(name: &str) -> Result<Key, String> {
    if name.is_empty() {
        return Err("Key name must be a non-empty string".into());
    }
    let modifier = match name {
        "Shift" | "ShiftLeft" | "ShiftRight" => MOD_SHIFT,
        "Control" | "ControlLeft" | "ControlRight" => MOD_CTRL,
        "Alt" | "AltLeft" | "AltRight" => MOD_ALT,
        "Meta" | "MetaLeft" | "MetaRight" => MOD_META,
        _ => 0,
    };
    if modifier != 0 {
        let base = name.trim_end_matches("Left").trim_end_matches("Right");
        return Ok(Key {
            key: base.to_string(),
            code: Some(if name == base {
                format!("{base}Left")
            } else {
                name.to_string()
            }),
            key_code: named_key_code(base).unwrap_or(0),
            modifier,
        });
    }
    let named = |key: &str, code: &str, key_code: u32| Key {
        key: key.to_string(),
        code: Some(code.to_string()),
        key_code,
        modifier: 0,
    };
    match name {
        "\r" | "\n" => return Ok(named("Enter", "Enter", 13)),
        "\t" => return Ok(named("Tab", "Tab", 9)),
        " " | "Space" => return Ok(named(" ", "Space", 32)),
        "NumpadEnter" => return Ok(named("Enter", "NumpadEnter", 13)),
        "NumpadAdd" => return Ok(named("+", "NumpadAdd", 107)),
        "NumpadSubtract" => return Ok(named("-", "NumpadSubtract", 109)),
        "NumpadMultiply" => return Ok(named("*", "NumpadMultiply", 106)),
        "NumpadDivide" => return Ok(named("/", "NumpadDivide", 111)),
        "NumpadDecimal" => return Ok(named(".", "NumpadDecimal", 110)),
        _ => {}
    }
    if let Some(code) = named_key_code(name) {
        return Ok(named(name, name, code));
    }
    if let Some(letter) = name.strip_prefix("Key")
        && letter.len() == 1
        && letter.as_bytes()[0].is_ascii_uppercase()
    {
        return Ok(named(
            &letter.to_ascii_lowercase(),
            name,
            letter.as_bytes()[0] as u32,
        ));
    }
    for (prefix, base) in [("Digit", 48u32), ("Numpad", 96u32)] {
        if let Some(digit) = name.strip_prefix(prefix)
            && digit.len() == 1
            && digit.as_bytes()[0].is_ascii_digit()
        {
            return Ok(named(
                digit,
                name,
                base + (digit.as_bytes()[0] - b'0') as u32,
            ));
        }
    }
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        let (key_code, code) = char_key(c);
        return Ok(Key {
            key: name.to_string(),
            code,
            key_code,
            modifier: 0,
        });
    }
    Err(format!("Unknown key {}", json!(name)))
}

/// Split `Control+Shift+a` into its keys; a lone `+` is one key.
pub(crate) fn split_key_combo(combo: &str) -> Vec<&str> {
    if combo.len() <= 1 || !combo.contains('+') {
        return vec![combo];
    }
    let mut parts: Vec<&str> = combo.split('+').collect();
    if parts.last() == Some(&"") {
        parts.pop();
        if let Some(last) = parts.last_mut() {
            *last = "+";
        }
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// One CDP `Input.dispatchKeyEvent` payload.
pub(crate) fn key_event(key: &Key, down: bool, mods: u8) -> Value {
    let shifted = mods & MOD_SHIFT != 0;
    let mut value = key.key.clone();
    if shifted && value.len() == 1 && value.as_bytes()[0].is_ascii_lowercase() {
        value = value.to_ascii_uppercase();
    }
    // Text is produced for printable keys and Enter, unless a shortcut
    // modifier turns the press into a command.
    let text = if mods & (MOD_CTRL | MOD_META | MOD_ALT) != 0 {
        None
    } else if value == "Enter" {
        Some("\r".to_string())
    } else if value.chars().count() == 1 {
        Some(value.clone())
    } else {
        None
    };
    let mut event = json!({
        "type": if down { if text.is_some() { "keyDown" } else { "rawKeyDown" } } else { "keyUp" },
        "key": value,
        "windowsVirtualKeyCode": key.key_code,
        "nativeVirtualKeyCode": key.key_code,
        "modifiers": mods,
        "location": if key.code.as_deref().is_some_and(|c| c.starts_with("Numpad")) { 3 } else if key.code.as_deref().is_some_and(|c| c.ends_with("Right")) && key.modifier != 0 { 2 } else if key.modifier != 0 { 1 } else { 0 },
    });
    if let Some(code) = &key.code {
        event["code"] = Value::String(code.clone());
    }
    if down && let Some(text) = text {
        event["text"] = Value::String(text.clone());
        event["unmodifiedText"] = Value::String(text);
    }
    // Chromium's headless shell maps editing shortcuts only through commands.
    if down && mods & (MOD_CTRL | MOD_META) != 0 {
        let command = match key.key.to_ascii_lowercase().as_str() {
            "a" => Some("selectAll"),
            "c" => Some("copy"),
            "x" => Some("cut"),
            "v" => Some("paste"),
            "z" if shifted => Some("redo"),
            "z" => Some("undo"),
            _ => None,
        };
        if let Some(command) = command {
            event["commands"] = json!([command]);
        }
    }
    event
}

// ---------------------------------------------------------------------------
// Allowed domains
// ---------------------------------------------------------------------------

/// Whether `host` is allowed by exact hosts or `*.example.com` patterns
/// (which also allow the bare domain).
pub(crate) fn host_allowed(host: &str, allowed: &[String]) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    allowed.iter().any(|pattern| {
        let pattern = pattern.trim().trim_end_matches('.').to_ascii_lowercase();
        match pattern.strip_prefix("*.") {
            Some(domain) => host == domain || host.ends_with(&format!(".{domain}")),
            None => host == pattern,
        }
    })
}

/// Host of a URL, or `None` for schemes without one (`data:`, `about:`).
pub(crate) fn url_host(url: &str) -> Option<&str> {
    let (scheme, rest) = url.split_once("://")?;
    if !matches!(scheme, "http" | "https" | "ws" | "wss") {
        return None;
    }
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit('@').next()?;
    if let Some(v6) = authority.strip_prefix('[') {
        return v6.split(']').next();
    }
    authority.split(':').next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_parse_like_upstream() {
        assert_eq!(
            parse_selector("#go").unwrap(),
            json!({"engine": "css", "query": "#go"})
        );
        assert_eq!(
            parse_selector("text/Allow all").unwrap(),
            json!({"engine": "text", "query": "Allow all"})
        );
        assert_eq!(
            parse_selector("aria/Submit[role=\"button\"]").unwrap(),
            json!({"engine": "aria", "name": "Submit", "role": "button"})
        );
        assert_eq!(
            parse_selector("e12").unwrap(),
            json!({"engine": "ariaRef", "ref": "e12"})
        );
        assert_eq!(
            parse_selector("aria-ref=e3").unwrap(),
            json!({"engine": "ariaRef", "ref": "e3"})
        );
        assert_eq!(
            parse_selector("p-aria/[name=\"Next\"]").unwrap(),
            json!({"engine": "aria", "name": "Next"})
        );
        assert_eq!(
            parse_selector("role/button[name=\"Go\" exact]").unwrap(),
            json!({"engine": "role", "query": "button[name=\"Go\" exact]"})
        );
        assert!(parse_selector("button:has-text(\"Go\")").is_err());
        assert!(parse_selector("aria-ref=x1").is_err());
        assert!(parse_selector("p-css/div").is_err());
        assert!(parse_selector("aria/x[foo=\"y\"]").is_err());
    }

    #[test]
    fn snapshot_compact_interactive_and_urls() {
        let snapshot = "- generic [ref=e1]:\n  - generic [ref=e2]:\n    - text: hello\n  - link \"Docs\" [ref=e3]:\n    - /url: /docs\n- heading \"Title\" [level=1] [ref=e4]";
        let compact = post_process_snapshot(
            snapshot,
            SnapshotOptions {
                compact: true,
                ..Default::default()
            },
            &HashMap::new(),
        );
        assert_eq!(
            compact,
            "- generic [ref=e1]:\n  - text: hello\n  - link \"Docs\" [ref=e3]:\n    - /url: /docs\n- heading \"Title\" [level=1] [ref=e4]"
        );
        let interactive = post_process_snapshot(
            snapshot,
            SnapshotOptions {
                interactive: true,
                ..Default::default()
            },
            &HashMap::new(),
        );
        assert_eq!(
            interactive,
            "- generic [ref=e1]:\n  - link \"Docs\" [ref=e3]:"
        );
        let hrefs = HashMap::from([("e3".to_string(), "https://x.test/docs".to_string())]);
        let urls = post_process_snapshot(
            snapshot,
            SnapshotOptions {
                urls: true,
                ..Default::default()
            },
            &hrefs,
        );
        assert!(urls.contains("- link \"Docs\" [ref=e3] [href=\"https://x.test/docs\"]:"));
    }

    #[test]
    fn snapshot_diff_revisions() {
        let mut baselines = HashMap::new();
        let base = (1..=20)
            .map(|i| format!("- item {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let first = diff_snapshot(&mut baselines, "k", "u", &base);
        assert_eq!(first["status"], "full");
        let same = diff_snapshot(&mut baselines, "k", "u", &base);
        assert_eq!(same, json!({"status": "unchanged", "revision": 1}));
        let changed = base.replace("- item 10", "- item ten");
        let delta = diff_snapshot(&mut baselines, "k", "u", &changed);
        assert_eq!(delta["status"], "delta");
        assert_eq!(delta["revision"], 2);
        assert_eq!(delta["baseRevision"], 1);
        assert_eq!(
            delta["delta"],
            "@@ -7,7 +7,7 @@\n - item 7\n - item 8\n - item 9\n-- item 10\n+- item ten\n - item 11\n - item 12\n - item 13"
        );
        let navigated = diff_snapshot(&mut baselines, "k", "other", &changed);
        assert_eq!(navigated["status"], "full");
        assert_eq!(navigated["revision"], 3);
    }

    #[test]
    fn markdown_sections_and_outline() {
        let md = "# A\nintro\n## A.1 Install\nsteps\n## A.2\nmore\n# B Install\nlast";
        assert_eq!(
            markdown_outline(md),
            "# A\n## A.1 Install\n## A.2\n# B Install"
        );
        assert_eq!(
            filter_markdown_sections(md, "install"),
            "## A.1 Install\nsteps\n# B Install\nlast"
        );
        assert_eq!(
            markdown_to_plain_text("## Head\n- [link](http://x) *b*"),
            "Head\nlink b"
        );
    }

    #[test]
    fn keys_resolve_and_combos_split() {
        assert_eq!(
            split_key_combo("Control+Shift+a"),
            vec!["Control", "Shift", "a"]
        );
        assert_eq!(split_key_combo("Control++"), vec!["Control", "+"]);
        assert_eq!(split_key_combo("+"), vec!["+"]);
        let enter = resolve_key("Enter").unwrap();
        assert_eq!(enter.key_code, 13);
        let down = key_event(&enter, true, 0);
        assert_eq!(down["type"], "keyDown");
        assert_eq!(down["text"], "\r");
        let a = resolve_key("KeyA").unwrap();
        assert_eq!((a.key.as_str(), a.key_code), ("a", 65));
        let shifted = key_event(&a, true, MOD_SHIFT);
        assert_eq!(shifted["key"], "A");
        assert_eq!(shifted["text"], "A");
        let select_all = key_event(&a, true, MOD_CTRL);
        assert_eq!(select_all["type"], "rawKeyDown");
        assert_eq!(select_all["commands"], json!(["selectAll"]));
        assert!(select_all.get("text").is_none());
        assert_eq!(resolve_key("ControlRight").unwrap().modifier, MOD_CTRL);
        assert_eq!(resolve_key("F5").unwrap().key_code, 116);
        assert!(resolve_key("NotAKey").is_err());
    }

    #[test]
    fn allowed_domains_match_hosts_and_wildcards() {
        let allowed = vec!["example.com".to_string(), "*.test.dev".to_string()];
        assert!(host_allowed("example.com", &allowed));
        assert!(!host_allowed("www.example.com", &allowed));
        assert!(host_allowed("test.dev", &allowed));
        assert!(host_allowed("a.b.test.dev", &allowed));
        assert!(!host_allowed("eviltest.dev", &allowed));
        assert_eq!(
            url_host("https://u:p@Example.com:8080/x?y"),
            Some("Example.com")
        );
        assert_eq!(url_host("http://[::1]:9/"), Some("::1"));
        assert_eq!(url_host("data:text/html,hi"), None);
    }
}
