//! Main memory: one conversation that never ends, kept as a binary tree of
//! one-line summaries, and the view of it every main request sees.
//!
//! Design after UniiChat's `Memory.ts` (github.com/VictorTaelin/uniichat).
//!
//! Leaf `(0, i)` is message `i` of the main conversation
//! ([`MainSession::history`](crate::MainSession)), shown as `role: text`.
//! Node `(l, i)` merges `(l-1, 2i)` and `(l-1, 2i+1)`, so it covers messages
//! `i·2^l .. (i+1)·2^l`. A node whose source fits in [`NODE`] bytes is its
//! source and needs no model call; any other is written by the compactor
//! model and appended to `nodes.jsonl`. Every node is built in the
//! background as messages arrive, oldest first.
//!
//! The view is a list of nodes covering the conversation, oldest first. Each
//! new message appends its leaf. Past `high` bytes, adjacent siblings whose
//! parent is built merge, most due first (oldest and lowest), down to half,
//! so detail fades with age and the view changes only at its end between
//! batches: a request reads most of its input from the provider's prompt
//! cache. The view is kept in `view.json`, so a restart shows the same lines.
//!
//! Nothing is lost: `zoom` opens any line into the two it was merged from,
//! down to the original message.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use juto_agent::{Tool, ToolContext, ToolOutput};
use juto_ai::{Context, Message, ToolDefinition};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::{
    MainSession, Runtime, recall,
    recall::{floor_boundary, full_body, role},
};

/// Most bytes of one node's text.
pub(crate) const NODE: usize = 512;
/// The text of a view line whose node is not built yet.
pub(crate) const WAIT: &str = "(not summarized yet: zoom it)";
/// Compactor calls building at once per conversation.
const JOBS: usize = 4;
/// Attempts at a line that fits before the shortest one is clipped.
const TRIES: usize = 3;
/// Attempts after a failed compactor call before the source is clipped.
const ERROR_TRIES: usize = 3;
/// Bytes of a leaf's source the compactor reads: its head and its tail.
const INPUT_HEAD: usize = 16 * 1024;
const INPUT_TAIL: usize = 8 * 1024;

const NODES_FILE: &str = "nodes.jsonl";
const VIEW_FILE: &str = "view.json";

/// System prompt of every compactor call.
pub(crate) const COMPACTOR_SYSTEM: &str = "You compress an AI agent's conversation into its long-term memory. Reply with exactly one line of plain text: no preamble, no quotes, no markdown headings. Keep what later work needs: decisions, facts, names, paths, numbers, errors, and open questions.";

/// What replaces a main turn's history when it outgrows the context budget.
const CONDENSED: &str = "[The turn so far outgrew the context budget, so it is now in <chat> above as lines, like the rest of the conversation. Continue the turn; zoom into lines for detail.]";

/// How the main system prompt explains the view.
pub(crate) const GUIDE: &str = "Your conversation with the user never ends; there is no new chat. The first message holds it as <chat>, one line per entry, oldest first: `id+n|text`. `id+1` is message id, whole when short, else summarized. A larger n summarizes messages id to id+n-1; older parts are covered by fewer, broader lines. `zoom {\"id\": id, \"n\": n}` opens a line into the two lines it was made from, down to the original message (n = 1). Messages after </chat> are the current turn, in full.";

/// Node `(l, i)`: messages `i·2^l .. (i+1)·2^l`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct Part {
    pub l: u32,
    pub i: usize,
}

impl Part {
    pub(crate) fn leaf(i: usize) -> Self {
        Self { l: 0, i }
    }
    /// First message covered.
    pub(crate) fn first(self) -> usize {
        self.i << self.l
    }
    /// Messages covered.
    pub(crate) fn len(self) -> usize {
        1 << self.l
    }
    /// One past the last message covered.
    pub(crate) fn end(self) -> usize {
        self.first() + self.len()
    }
    pub(crate) fn halves(self) -> Option<[Part; 2]> {
        (self.l > 0).then(|| {
            [
                Part {
                    l: self.l - 1,
                    i: 2 * self.i,
                },
                Part {
                    l: self.l - 1,
                    i: 2 * self.i + 1,
                },
            ]
        })
    }
    fn parent(self) -> Part {
        Part {
            l: self.l + 1,
            i: self.i / 2,
        }
    }
    /// The `id+n` a view line and `zoom` name it by.
    pub(crate) fn reference(self) -> String {
        format!("{}+{}", self.first(), self.len())
    }
    /// The node `zoom {id, n}` names, if `n` is a power of two and `id` a
    /// multiple of it.
    pub(crate) fn named(id: usize, n: usize) -> Option<Part> {
        (n.is_power_of_two() && id.is_multiple_of(n)).then(|| Part {
            l: n.trailing_zeros(),
            i: id / n,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct NodeRecord {
    l: u32,
    i: usize,
    text: String,
}

#[derive(Serialize, Deserialize)]
struct ViewRecord {
    parts: Vec<(u32, usize)>,
}

struct State {
    /// Each message's source: `role: text`.
    leaves: Vec<String>,
    /// Nodes the compactor wrote.
    written: HashMap<Part, String>,
    view: Vec<Part>,
    /// Nodes being written.
    busy: HashSet<Part>,
    /// Bytes past which the view merges down to half.
    high: usize,
}

/// One main conversation's tree and view, under its memory directory.
pub(crate) struct Memory {
    dir: PathBuf,
    state: Mutex<State>,
    changed: Notify,
}

impl Memory {
    /// Open the memory kept in `dir`, creating it if absent. A line a crash
    /// cut short is skipped; an unreadable view starts over from the leaves.
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut written = HashMap::new();
        match fs::read_to_string(dir.join(NODES_FILE)) {
            Ok(text) => {
                for line in text.lines() {
                    if let Ok(record) = serde_json::from_str::<NodeRecord>(line) {
                        written.insert(
                            Part {
                                l: record.l,
                                i: record.i,
                            },
                            record.text,
                        );
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("reading memory nodes"),
        }
        let view = fs::read(dir.join(VIEW_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ViewRecord>(&bytes).ok())
            .map(|record| {
                record
                    .parts
                    .into_iter()
                    .map(|(l, i)| Part { l, i })
                    .collect::<Vec<_>>()
            })
            .filter(|parts| contiguous(parts))
            .unwrap_or_default();
        Ok(Self {
            dir: dir.to_path_buf(),
            state: Mutex::new(State {
                leaves: Vec::new(),
                written,
                view,
                busy: HashSet::new(),
                high: usize::MAX,
            }),
            changed: Notify::new(),
        })
    }

    /// Take the conversation's messages and the view's byte budget: new
    /// messages join the view as leaves, and the view merges if it is over.
    pub(crate) fn sync(&self, messages: &[Message], high: usize) -> Result<()> {
        let mut state = self.state.lock();
        if messages.len() < view_end(&state.view) {
            // The conversation is shorter than the view: it was rewritten.
            // Lines past its end, and nodes over them, no longer hold.
            let count = messages.len();
            state.view.clear();
            state.written.retain(|part, _| part.end() <= count);
        }
        state.leaves = messages.iter().map(leaf_source).collect();
        state.high = high;
        let end = view_end(&state.view);
        let count = state.leaves.len();
        state.view.extend((end..count).map(Part::leaf));
        let changed = fit(&mut state, false) || end < count;
        if changed {
            self.save(&state)?;
        }
        drop(state);
        self.changed.notify_waiters();
        Ok(())
    }

    /// Whether a request may read the view: every line built and the view
    /// within its budget. Merges as far as built parents allow first.
    fn settled(&self) -> Result<bool> {
        let mut state = self.state.lock();
        if fit(&mut state, true) {
            self.save(&state)?;
        }
        let all_built = state.view.iter().all(|&part| text(&state, part).is_some());
        Ok(all_built && view_size(&state) <= state.high)
    }

    /// The view as the first message of a main request: `<chat>`, a line per
    /// node, `</chat>`. `None` before the conversation has a message.
    pub(crate) fn view_text(&self) -> Option<String> {
        let state = self.state.lock();
        if state.view.is_empty() {
            return None;
        }
        let mut out = String::from("<chat>\n");
        for &part in &state.view {
            out.push_str(&line(part, text(&state, part).as_deref().unwrap_or(WAIT)));
            out.push('\n');
        }
        out.push_str("</chat>");
        Some(out)
    }

    /// `zoom {id, n}`: message `id` whole when `n` is 1, else the two lines
    /// node `id+n` was made from.
    pub(crate) fn zoom(&self, id: usize, n: usize) -> std::result::Result<Zoomed, String> {
        let state = self.state.lock();
        let count = state.leaves.len();
        let Some(part) = Part::named(id, n).filter(|part| part.end() <= count) else {
            return Err(format!(
                "no line {id}+{n}: n is a power of 2, id a multiple of n, and id+n at most {count}"
            ));
        };
        Ok(match part.halves() {
            None => Zoomed::Message(id),
            Some(halves) => Zoomed::Lines(
                halves
                    .iter()
                    .map(|&half| line(half, text(&state, half).as_deref().unwrap_or(WAIT)))
                    .collect(),
            ),
        })
    }

    /// Nodes to write, up to [`JOBS`] at once, oldest first: marked busy and
    /// returned with their sources.
    fn take_jobs(&self) -> Vec<(Part, String)> {
        let mut state = self.state.lock();
        let free = JOBS.saturating_sub(state.busy.len());
        if free == 0 {
            return Vec::new();
        }
        let count = state.leaves.len();
        let mut jobs = Vec::new();
        // Oldest first: by the last message a node covers, a merge before
        // the leaf it ends on.
        let mut candidates: Vec<Part> = Vec::new();
        let mut l = 0;
        while (1usize << l) <= count {
            let size = 1usize << l;
            for i in 0..count / size {
                candidates.push(Part { l, i });
            }
            l += 1;
        }
        candidates.sort_by_key(|part| (part.end(), std::cmp::Reverse(part.l)));
        for part in candidates {
            if jobs.len() == free {
                break;
            }
            if state.written.contains_key(&part) || state.busy.contains(&part) {
                continue;
            }
            let Some(source) = source(&state, part) else {
                continue;
            };
            if source.len() <= NODE {
                continue;
            }
            let source = source.into_owned();
            state.busy.insert(part);
            jobs.push((part, source));
        }
        jobs
    }

    /// Keep a written node: appended to `nodes.jsonl`, then the view merges
    /// if it can.
    fn store(&self, part: Part, text: String) -> Result<()> {
        let mut state = self.state.lock();
        state.busy.remove(&part);
        let record = serde_json::to_string(&NodeRecord {
            l: part.l,
            i: part.i,
            text: text.clone(),
        })?;
        let path = self.dir.join(NODES_FILE);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        file.write_all(format!("{record}\n").as_bytes())
            .with_context(|| format!("writing {}", path.display()))?;
        state.written.insert(part, text);
        if fit(&mut state, false) {
            self.save(&state)?;
        }
        drop(state);
        self.changed.notify_waiters();
        Ok(())
    }

    /// Give up a node's job; it is taken again on the next pump.
    fn release(&self, part: Part) {
        self.state.lock().busy.remove(&part);
        self.changed.notify_waiters();
    }

    /// `view.json`, written whole and moved into place.
    fn save(&self, state: &State) -> Result<()> {
        let record = ViewRecord {
            parts: state.view.iter().map(|part| (part.l, part.i)).collect(),
        };
        let path = self.dir.join(VIEW_FILE);
        let staging = self.dir.join(format!("{VIEW_FILE}.new"));
        fs::write(&staging, serde_json::to_vec(&record)?)
            .with_context(|| format!("writing {}", staging.display()))?;
        fs::rename(&staging, &path).with_context(|| format!("replacing {}", path.display()))
    }
}

/// What `zoom` opens a line into.
pub(crate) enum Zoomed {
    /// The original message at this index.
    Message(usize),
    /// The two lines under the node.
    Lines(Vec<String>),
}

/// A message as the tree reads it: `role: text`, thinking left out. A tool
/// result's body already opens with its kind and tool.
fn leaf_source(message: &Message) -> String {
    match message {
        Message::ToolResult { .. } => full_body(message),
        _ => format!("{}: {}", role(message), full_body(message)),
    }
}

/// The node's text: written, or its source when that fits. `None`: not
/// built.
fn text(state: &State, part: Part) -> Option<Cow<'_, str>> {
    if let Some(text) = state.written.get(&part) {
        return Some(Cow::Borrowed(text));
    }
    source(state, part).filter(|source| source.len() <= NODE)
}

/// What a node is made from: its message, or its halves' lines. `None` when
/// a half is not built.
fn source(state: &State, part: Part) -> Option<Cow<'_, str>> {
    match part.halves() {
        None => state
            .leaves
            .get(part.i)
            .map(|leaf| Cow::Borrowed(leaf.as_str())),
        Some([a, b]) => {
            let a = text(state, a)?;
            let b = text(state, b)?;
            Some(Cow::Owned(format!("{}\n{}", flat(&a), flat(&b))))
        }
    }
}

/// One view line: `id+n|text` on one line.
fn line(part: Part, text: &str) -> String {
    format!("{}|{}", part.reference(), flat(text))
}

/// Whitespace runs holding a newline become one space.
fn flat(text: &str) -> Cow<'_, str> {
    if !text.contains('\n') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    for (k, piece) in text.split('\n').enumerate() {
        let piece = if k == 0 {
            piece.trim_end()
        } else {
            piece.trim()
        };
        if piece.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(piece);
    }
    Cow::Owned(out)
}

/// Bytes of the view's built lines.
fn view_size(state: &State) -> usize {
    state
        .view
        .iter()
        .filter_map(|&part| text(state, part))
        .map(|text| text.len())
        .sum()
}

fn view_end(view: &[Part]) -> usize {
    view.last().map_or(0, |part| part.end())
}

/// Whether `parts` cover messages from 0 without gaps.
fn contiguous(parts: &[Part]) -> bool {
    let mut end = 0;
    for part in parts {
        if part.first() != end {
            return false;
        }
        end = part.end();
    }
    true
}

/// Merge the view if it is over `high`: the most due pairs whose parent is
/// built, down to half. Between turns only a whole batch applies, so the
/// view's old lines change once per batch; `force` (a request about to read
/// it) applies whatever the built parents allow. Returns whether it changed.
fn fit(state: &mut State, force: bool) -> bool {
    let size = view_size(state);
    if size <= state.high {
        return false;
    }
    let target = state.high / 2;
    let (merged, merged_size) = merge(state, target);
    if merged.len() == state.view.len() || (!force && merged_size > target) {
        return false;
    }
    state.view = merged;
    true
}

/// The view with its most due sibling pairs merged until it is at most
/// `target` bytes or no pair's parent is built. A pair is due by its age in
/// halves from the view's end, `(end + 1) / 2^l - i`: old and low first, the
/// oldest of equals.
fn merge(state: &State, target: usize) -> (Vec<Part>, usize) {
    let end = view_end(&state.view) as f64;
    let mut view = state.view.clone();
    let mut size = view_size(state);
    while size > target {
        let mut best: Option<(f64, usize)> = None;
        for k in 0..view.len().saturating_sub(1) {
            let (a, b) = (view[k], view[k + 1]);
            if a.l != b.l || !a.i.is_multiple_of(2) || b.i != a.i + 1 {
                continue;
            }
            if text(state, a.parent()).is_none() {
                continue;
            }
            let due = (end + 1.0) / f64::from(1u32 << a.l.min(31)) - a.i as f64;
            if best.is_none_or(|(top, _)| due > top) {
                best = Some((due, k));
            }
        }
        let Some((_, k)) = best else {
            break;
        };
        let (a, b) = (view[k], view[k + 1]);
        let up = a.parent();
        let bytes = |part: Part| text(state, part).map_or(0, |text| text.len());
        size = size + bytes(up) - bytes(a) - bytes(b);
        view[k] = up;
        view.remove(k + 1);
    }
    (view, size)
}

/// A compactor line too long for a node: its size and where the limit cuts.
fn over(line: &str) -> String {
    let cut = floor_boundary(line, NODE);
    format!(
        "Too long: your line was {} bytes, over the {NODE}-byte limit. Write the whole line again, cutting just enough of the least valuable items to fit before this cut:\n{}| ← LIMIT",
        line.len(),
        &line[..cut]
    )
}

/// What the compactor reads of a source: all of it, or its head and tail.
fn compactor_input(source: &str) -> String {
    if source.len() <= INPUT_HEAD + INPUT_TAIL {
        return source.to_owned();
    }
    let head = floor_boundary(source, INPUT_HEAD);
    let tail = floor_boundary(source, source.len() - INPUT_TAIL);
    format!(
        "{}\n[… {} bytes omitted …]\n{}",
        &source[..head],
        tail - head,
        &source[tail..]
    )
}

/// The compactor's task for `part`.
fn task(part: Part, source: &str) -> String {
    let input = compactor_input(source);
    match part.halves() {
        None => format!(
            "Compress message {} into one line of at most {NODE} bytes (about 70 words).\n<input>\n{input}\n</input>",
            part.first()
        ),
        Some([a, b]) => format!(
            "Merge lines {} and {}, adjacent in the conversation, into one line of at most {NODE} bytes (about 70 words) covering messages {} to {}.\n<input>\n{input}\n</input>",
            a.reference(),
            b.reference(),
            part.first(),
            part.end() - 1
        ),
    }
}

/// The first `NODE` bytes of `text`, on one line.
fn clip(text: &str) -> String {
    let text = flat(text);
    text[..floor_boundary(&text, NODE)].to_owned()
}

impl Runtime {
    /// The memory of the main conversation at `main_path`, shared by every
    /// run on it in this process.
    pub(crate) fn main_memory(&self, main_path: &Path) -> Result<Arc<Memory>> {
        let mut memories = self.inner.memories.lock();
        if let Some(memory) = memories.get(main_path) {
            return Ok(memory.clone());
        }
        let memory = Arc::new(Memory::open(&memory_dir(main_path))?);
        memories.insert(main_path.to_path_buf(), memory.clone());
        Ok(memory)
    }

    /// Bring `memory` up to the conversation's messages and start building
    /// the nodes it lacks.
    pub(crate) fn sync_memory(
        &self,
        memory: &Arc<Memory>,
        messages: &[Message],
        high: usize,
    ) -> Result<()> {
        memory.sync(messages, high)?;
        self.pump_memory(memory);
        Ok(())
    }

    /// Wait until a request may read the view: every line built and the view
    /// within budget.
    pub(crate) async fn settle_memory(
        &self,
        memory: &Arc<Memory>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        loop {
            let changed = memory.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if memory.settled()? {
                return Ok(());
            }
            self.pump_memory(memory);
            tokio::select! {
                _ = cancel.cancelled() => anyhow::bail!("cancelled while the memory was being summarized"),
                _ = &mut changed => {}
            }
        }
    }

    /// A main turn's history once it outgrows the context budget: its
    /// messages so far join the memory, and the turn goes on from the view.
    pub(crate) async fn condensed_view(
        &self,
        memory: &Arc<Memory>,
        main_path: &Path,
        high: usize,
        cancel: &CancellationToken,
    ) -> Result<Vec<Message>> {
        let messages = MainSession::open(main_path)?.timeline()?;
        self.sync_memory(memory, &messages, high)?;
        self.settle_memory(memory, cancel).await?;
        Ok(memory
            .view_text()
            .map(Message::developer)
            .into_iter()
            .chain([Message::developer(CONDENSED)])
            .collect())
    }

    /// Start a job for each node the memory can build now.
    fn pump_memory(&self, memory: &Arc<Memory>) {
        for (part, source) in memory.take_jobs() {
            let runtime = self.clone();
            let memory = memory.clone();
            tokio::spawn(async move {
                let text = runtime.write_node(part, &source).await;
                match memory.store(part, text) {
                    Ok(()) => runtime.pump_memory(&memory),
                    Err(error) => {
                        eprintln!("juto: memory node {}: {error:#}", part.reference());
                        memory.release(part);
                    }
                }
            });
        }
    }

    /// Node `part`'s line, from the compactor. A line too long is asked again
    /// with its size and cut, and the shortest kept; with no line at all, the
    /// source is clipped. Never fails: a node always builds, so no turn waits
    /// forever on an unavailable model.
    async fn write_node(&self, part: Part, source: &str) -> String {
        let selector = self
            .config()
            .compactor
            .unwrap_or_else(|| "@tiny".to_owned());
        let Ok((model, mut options)) = self.resolve_model(&selector) else {
            return clip(source);
        };
        options.max_tokens = Some(400);
        options.thinking = None;
        let cancel = CancellationToken::new();
        let first = Message::user(task(part, source));
        let mut lines: Vec<String> = Vec::new();
        let mut errors = 0;
        while lines.len() < TRIES && errors < ERROR_TRIES {
            let mut messages = vec![first.clone()];
            if let Some(last) = lines.last() {
                messages.push(Message::user(over(last)));
            }
            let context = Context {
                system_prompt: vec![COMPACTOR_SYSTEM.to_owned()],
                messages,
                tools: Vec::new(),
            };
            match self
                .complete_once(&model, &options, &context, &cancel)
                .await
            {
                Ok(reply) => {
                    let line = flat(reply.text().trim()).into_owned();
                    if line.is_empty() {
                        break;
                    }
                    let fits = line.len() <= NODE;
                    lines.push(line);
                    if fits {
                        break;
                    }
                }
                Err(_) => {
                    errors += 1;
                    tokio::time::sleep(Duration::from_millis(200 * errors as u64)).await;
                }
            }
        }
        match lines.into_iter().min_by_key(String::len) {
            Some(line) if line.len() <= NODE => line,
            Some(line) => clip(&line),
            None => clip(source),
        }
    }
}

const ZOOM_DESCRIPTION: &str = "Open a line of <chat> into the two lines it was made from. `id` and `n` are the line's `id+n`. With n = 1, returns message id in full; `at` continues a long message by byte offset.";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ZoomArgs {
    id: usize,
    #[serde(default = "one")]
    n: usize,
    #[serde(default)]
    at: usize,
}

fn one() -> usize {
    1
}

/// `zoom` over the main conversation's memory tree.
pub(crate) struct ZoomTool {
    memory: Arc<Memory>,
    main_path: PathBuf,
}

impl ZoomTool {
    pub(crate) fn new(memory: Arc<Memory>, main_path: PathBuf) -> Self {
        Self { memory, main_path }
    }
}

#[async_trait]
impl Tool for ZoomTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "zoom".into(),
            description: ZOOM_DESCRIPTION.into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0, "description": "The line's first message: the id of id+n."},
                    "n": {"type": "integer", "minimum": 1, "description": "The line's message count: the n of id+n. Defaults to 1."},
                    "at": {"type": "integer", "minimum": 0, "description": "Byte offset into a long message (n = 1)."}
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: Value, _ctx: ToolContext) -> Result<ToolOutput, String> {
        let args: ZoomArgs = match serde_json::from_value(arguments) {
            Ok(args) => args,
            Err(error) => {
                return Ok(ToolOutput::error(format!(
                    "invalid zoom arguments: {error}"
                )));
            }
        };
        match self.memory.zoom(args.id, args.n) {
            Err(error) => Ok(ToolOutput::error(error)),
            Ok(Zoomed::Lines(lines)) => Ok(ToolOutput::text(lines.join("\n"))),
            Ok(Zoomed::Message(index)) => {
                let path = self.main_path.clone();
                tokio::task::spawn_blocking(move || recall::read_message(&path, index, args.at))
                    .await
                    .map_err(|error| error.to_string())?
            }
        }
    }
}

/// Where a main session keeps its memory: beside its journal.
fn memory_dir(main_path: &Path) -> PathBuf {
    main_path.with_extension("").join("memory")
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
