//! Append-only JSONL session trees. Navigation and compaction are durable records.
//! Behavioral reference: OMP 579da1d6; Juto's journal is not byte-compatible with OMP.
use fs2::FileExt;
use juto_ai::{ContentBlock, Message, timestamp_ms};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

use crate::snapcompact::Archive;

pub const SESSION_VERSION: u32 = 1;
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid session file {path}: {reason}")]
    Invalid { path: PathBuf, reason: String },
    #[error("session {path} is corrupt: {reason}")]
    Corrupt { path: PathBuf, reason: String },
    #[error("unknown branch target {0}")]
    UnknownTarget(String),
    #[error("span {start}..{end} is not a message range on the active path")]
    InvalidSpan { start: String, end: String },
}
pub type Result<T, E = SessionError> = std::result::Result<T, E>;
fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> SessionError + '_ {
    move |source| SessionError::Io {
        path: path.to_path_buf(),
        source,
    }
}
fn corrupt(path: &Path, reason: impl Into<String>) -> SessionError {
    SessionError::Corrupt {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    #[serde(rename = "type")]
    marker: String,
    pub id: String,
    pub cwd: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
    pub version: u32,
    pub created: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntry {
    pub id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub timestamp: u64,
    #[serde(flatten)]
    pub kind: EntryKind,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntryKind {
    Message {
        message: Message,
    },
    ModelChange {
        model: String,
    },
    Compaction {
        summary: String,
        first_kept_id: Option<String>,
        /// Ids of the first and last message this compaction replaced, inclusive,
        /// on the active path. Absent when it replaced nothing and in journals
        /// written before spans were recorded.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        span_start_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        span_end_id: Option<String>,
        /// Page images of the discarded history; absent for text summaries and old journals.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        archive: Option<Archive>,
        /// Depth of summarization: 0 summarizes original messages, 1 summarizes
        /// level-0 summaries, and so on. Absent in journals written before nodes.
        #[serde(default)]
        level: u32,
    },
    Branch {
        target: Option<String>,
    },
    Custom {
        #[serde(rename = "customType")]
        custom_type: String,
        #[serde(default)]
        data: serde_json::Value,
    },
}
struct Journal {
    header: SessionHeader,
    entries: Vec<SessionEntry>,
    index: HashMap<String, usize>,
    leaf: Option<String>,
    size: u64,
    valid_size: u64,
    needs_newline: bool,
}
pub struct Session {
    path: PathBuf,
    file: File,
    journal: Journal,
}

/// One message on the active path, numbered from the root.
#[derive(Debug, Clone, Copy)]
pub struct PathMessage<'a> {
    pub index: usize,
    pub id: &'a str,
    pub message: &'a Message,
}

/// The latest compaction on the active path.
#[derive(Debug, Clone, Copy)]
pub struct CompactionMark<'a> {
    /// Id of the compaction entry. The marker names it so `zoom` can expand the span.
    pub id: &'a str,
    pub summary: &'a str,
    pub archive: Option<&'a Archive>,
    /// Index of the first message after the compaction entry.
    pub at: usize,
    /// Index of the first message the model still sees; equal to `at` when the
    /// compaction kept nothing.
    pub first_visible: usize,
}

/// A summary node on the active path, covering the original messages `start..end`.
#[derive(Debug, Clone, Copy)]
pub struct NodeRef<'a> {
    pub entry_id: &'a str,
    pub summary: &'a str,
    pub archive: Option<&'a Archive>,
    pub level: u32,
    pub start: usize,
    pub end: usize,
}

/// The active path as recall addresses it.
#[derive(Debug, Clone)]
pub struct PathView<'a> {
    pub messages: Vec<PathMessage<'a>>,
    /// Summary nodes in order; together they cover `messages[..raw_start()]`.
    pub nodes: Vec<NodeRef<'a>>,
    /// The latest compaction entry on the path.
    pub compaction: Option<CompactionMark<'a>>,
}

impl<'a> PathView<'a> {
    /// Index of the first message that no node covers.
    pub fn raw_start(&self) -> usize {
        self.nodes.last().map_or(0, |node| node.end)
    }
    /// Messages the model sees: the raw tail after the nodes.
    pub fn visible(&self) -> &[PathMessage<'a>] {
        &self.messages[self.raw_start()..]
    }
}

/// History discarded by a compaction and the preceding summary or frame archive.
#[derive(Debug)]
pub struct CompactionPlan {
    pub previous: Vec<PriorCompaction>,
    pub discarded: Vec<Message>,
}

#[derive(Debug, Clone)]
pub struct PriorCompaction {
    pub summary: String,
    pub archive: Option<Archive>,
}

/// Messages to summarize into one new node, and where the node sits on the path.
#[derive(Debug, Clone)]
pub struct NodePlan {
    pub input: Vec<Message>,
    pub level: u32,
    pub span_start_id: String,
    pub span_end_id: String,
    /// Entry ids of the live nodes the new node replaces.
    pub covers: Vec<String>,
}

impl NodePlan {
    pub fn into_node(self, summary: String, archive: Option<Archive>) -> NewNode {
        NewNode {
            summary,
            archive,
            level: self.level,
            span_start_id: self.span_start_id,
            span_end_id: self.span_end_id,
            covers: self.covers,
        }
    }
}

/// A summary node to append: a summary of the messages from `span_start_id` through
/// `span_end_id`, replacing the live nodes named in `covers`.
#[derive(Debug, Clone)]
pub struct NewNode {
    pub summary: String,
    pub archive: Option<Archive>,
    pub level: u32,
    pub span_start_id: String,
    pub span_end_id: String,
    pub covers: Vec<String>,
}

impl Session {
    pub fn create(directory: &Path, cwd: &Path) -> Result<Self> {
        Self::create_from(directory, cwd, None, &[])
    }
    fn create_from(
        directory: &Path,
        cwd: &Path,
        parent_session: Option<String>,
        entries: &[&SessionEntry],
    ) -> Result<Self> {
        std::fs::create_dir_all(directory).map_err(io_err(directory))?;
        let header = SessionHeader {
            marker: "session".into(),
            id: Uuid::now_v7().to_string(),
            cwd: cwd.to_path_buf(),
            parent_session,
            version: SESSION_VERSION,
            created: timestamp_ms(),
        };
        let path = directory.join(format!("{}_{}.jsonl", header.created, header.id));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&path).map_err(io_err(&path))?;
        {
            let mut writer = BufWriter::new(&mut file);
            serde_json::to_writer(&mut writer, &header)
                .map_err(|error| corrupt(&path, error.to_string()))?;
            writer.write_all(b"\n").map_err(io_err(&path))?;
            for entry in entries {
                serde_json::to_writer(&mut writer, entry)
                    .map_err(|error| corrupt(&path, error.to_string()))?;
                writer.write_all(b"\n").map_err(io_err(&path))?;
            }
            writer.flush().map_err(io_err(&path))?;
        }
        file.sync_all().map_err(io_err(&path))?;
        File::open(directory)
            .and_then(|directory| directory.sync_all())
            .map_err(io_err(directory))?;
        let journal = read_journal(&path, &mut file)?;
        Ok(Self {
            path,
            file,
            journal,
        })
    }
    /// Open validates without creating, truncating or rewriting anything. A torn
    /// final append is repaired only by the next writer under the exclusive lock.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(io_err(path))?;
        FileExt::lock_shared(&file).map_err(io_err(path))?;
        let result = read_journal(path, &mut file);
        let unlock = FileExt::unlock(&file).map_err(io_err(path));
        let journal = result?;
        unlock?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            journal,
        })
    }
    pub fn header(&self) -> &SessionHeader {
        &self.journal.header
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn append(&mut self, kind: EntryKind) -> Result<String> {
        FileExt::lock_exclusive(&self.file).map_err(io_err(&self.path))?;
        let result = self.append_locked(kind);
        let unlock = FileExt::unlock(&self.file).map_err(io_err(&self.path));
        let id = result?;
        unlock?;
        Ok(id)
    }
    fn append_locked(&mut self, kind: EntryKind) -> Result<String> {
        self.refresh_locked()?;
        let target = match &kind {
            EntryKind::Branch { target } => resolve_target(&self.journal, target.as_deref())?,
            _ => None,
        };
        let entry = SessionEntry {
            id: Uuid::now_v7().to_string(),
            parent_id: self.journal.leaf.clone(),
            timestamp: timestamp_ms(),
            kind,
        };
        let mut bytes =
            serde_json::to_vec(&entry).map_err(|error| corrupt(&self.path, error.to_string()))?;
        bytes.push(b'\n');
        self.file
            .set_len(self.journal.valid_size)
            .map_err(io_err(&self.path))?;
        self.file
            .seek(SeekFrom::Start(self.journal.valid_size))
            .map_err(io_err(&self.path))?;
        if self.journal.needs_newline {
            self.file.write_all(b"\n").map_err(io_err(&self.path))?;
        }
        self.file.write_all(&bytes).map_err(io_err(&self.path))?;
        self.file.sync_data().map_err(io_err(&self.path))?;
        self.journal.size =
            self.journal.valid_size + bytes.len() as u64 + u64::from(self.journal.needs_newline);
        self.journal.valid_size = self.journal.size;
        self.journal.needs_newline = false;
        self.journal.leaf = if matches!(entry.kind, EntryKind::Branch { .. }) {
            target
        } else {
            Some(entry.id.clone())
        };
        let id = entry.id.clone();
        self.journal
            .index
            .insert(entry.id.clone(), self.journal.entries.len());
        self.journal.entries.push(entry);
        Ok(id)
    }

    /// Records an interrupted run. Every tool call on the active branch without
    /// a result gets an error `ToolResult`, then one `run_interrupted` marker is
    /// appended. A repeat call with nothing left unpaired is a no-op.
    pub fn append_interrupted(&mut self, run_id: &str, reason: &str) -> Result<()> {
        FileExt::lock_exclusive(&self.file).map_err(io_err(&self.path))?;
        let result = self.append_interrupted_locked(run_id, reason);
        let unlock = FileExt::unlock(&self.file).map_err(io_err(&self.path));
        result?;
        unlock
    }
    fn append_interrupted_locked(&mut self, run_id: &str, reason: &str) -> Result<()> {
        self.refresh_locked()?;
        let mut calls = Vec::new();
        let mut answered = HashSet::new();
        let mut marked = false;
        for entry in self.active_entries()? {
            match &entry.kind {
                EntryKind::Message {
                    message: Message::Assistant(assistant),
                } => calls.extend(
                    assistant
                        .tool_calls()
                        .map(|call| (call.id.clone(), call.name.clone())),
                ),
                EntryKind::Message {
                    message: Message::ToolResult { tool_call_id, .. },
                } => {
                    answered.insert(tool_call_id.clone());
                }
                EntryKind::Custom { custom_type, data }
                    if custom_type == "run_interrupted" && data["run_id"] == run_id =>
                {
                    marked = true;
                }
                _ => {}
            }
        }
        let unpaired: Vec<_> = calls
            .into_iter()
            .filter(|(id, _)| !answered.contains(id))
            .collect();
        if unpaired.is_empty() && marked {
            return Ok(());
        }
        for (id, name) in unpaired {
            self.append_locked(EntryKind::Message {
                message: Message::ToolResult {
                    tool_call_id: id,
                    tool_name: name,
                    content: vec![ContentBlock::Text {
                        text: format!("Tool call interrupted: {reason}"),
                    }],
                    is_error: true,
                    details: serde_json::Value::Null,
                    timestamp: timestamp_ms(),
                },
            })?;
        }
        if !marked {
            self.append_locked(EntryKind::Custom {
                custom_type: "run_interrupted".into(),
                data: serde_json::json!({ "run_id": run_id, "reason": reason }),
            })?;
        }
        Ok(())
    }
    fn refresh_locked(&mut self) -> Result<()> {
        let size = self.file.metadata().map_err(io_err(&self.path))?.len();
        if size != self.journal.size {
            let next = read_journal(&self.path, &mut self.file)?;
            if next.header.id != self.journal.header.id
                || next.header.cwd != self.journal.header.cwd
            {
                return Err(corrupt(&self.path, "journal identity changed"));
            }
            self.journal = next;
        }
        Ok(())
    }
    pub fn active_entries(&self) -> Result<Vec<&SessionEntry>> {
        let mut entries = Vec::new();
        let mut current = self.journal.leaf.as_deref();
        while let Some(id) = current {
            let index = *self
                .journal
                .index
                .get(id)
                .ok_or_else(|| corrupt(&self.path, "missing tree parent"))?;
            let entry = &self.journal.entries[index];
            entries.push(entry);
            current = entry.parent_id.as_deref();
        }
        entries.reverse();
        Ok(entries)
    }
    /// The active path as recall addresses it: every message entry numbered from the
    /// root, the summary nodes covering a prefix of it, and the latest compaction.
    /// Numbers stay put when a compaction is appended and change only when the active
    /// path changes (a branch).
    pub fn path_view(&self) -> Result<PathView<'_>> {
        let entries = self.active_entries()?;
        let mut messages: Vec<PathMessage<'_>> = Vec::new();
        let mut nodes: Vec<NodeRef<'_>> = Vec::new();
        let mut latest = None;
        for entry in entries {
            match &entry.kind {
                EntryKind::Message { message } => messages.push(PathMessage {
                    index: messages.len(),
                    id: entry.id.as_str(),
                    message,
                }),
                EntryKind::Compaction {
                    summary,
                    first_kept_id,
                    span_start_id,
                    span_end_id,
                    archive,
                    level,
                } => {
                    let (start, end) = match (span_start_id, span_end_id) {
                        (Some(first), Some(last)) => {
                            match (
                                message_index(&messages, first),
                                message_index(&messages, last),
                            ) {
                                (Some(start), Some(end)) if start <= end => (start, end + 1),
                                _ => {
                                    return Err(corrupt(
                                        &self.path,
                                        "compaction span does not match the active path",
                                    ));
                                }
                            }
                        }
                        _ => {
                            // Journals from before nodes: the entry replaced the whole
                            // prefix before its first kept message.
                            let floor = match first_kept_id {
                                Some(id) => message_index(&messages, id).ok_or_else(|| {
                                    corrupt(
                                        &self.path,
                                        "compaction references a missing retained entry",
                                    )
                                })?,
                                None => messages.len(),
                            };
                            (0, floor)
                        }
                    };
                    // A summary supersedes the nodes it contains.
                    nodes.retain(|node| !(start <= node.start && node.end <= end));
                    nodes.push(NodeRef {
                        entry_id: entry.id.as_str(),
                        summary: summary.as_str(),
                        archive: archive.as_ref(),
                        level: *level,
                        start,
                        end,
                    });
                    latest = Some((
                        entry.id.as_str(),
                        summary.as_str(),
                        archive.as_ref(),
                        messages.len(),
                    ));
                }
                _ => {}
            }
        }
        nodes.sort_by_key(|node| (node.start, node.end));
        let mut cursor = 0;
        for node in &nodes {
            if node.start != cursor {
                return Err(corrupt(
                    &self.path,
                    "compaction summaries do not cover a contiguous prefix",
                ));
            }
            cursor = node.end;
        }
        let compaction = latest.map(|(id, summary, archive, at)| CompactionMark {
            id,
            summary,
            archive,
            at,
            first_visible: cursor,
        });
        Ok(PathView {
            messages,
            nodes,
            compaction,
        })
    }
    pub fn messages(&self) -> Result<Vec<Message>> {
        let view = self.path_view()?;
        let visible = view.visible();
        let mut messages = Vec::with_capacity(view.nodes.len() + visible.len());
        for node in &view.nodes {
            if let Some(archive) = node.archive {
                // Anthropic system/developer prompts cannot carry images. User content
                // supports frames on both supported vision wire formats.
                let mut content = Vec::with_capacity(archive.frames.len() + 1);
                content.push(ContentBlock::Text {
                    text: format!(
                        "[Earlier conversation archive id={} — originals: recall]\n{}",
                        node.entry_id, node.summary
                    ),
                });
                content.extend(archive.frames.iter().map(|frame| ContentBlock::Image {
                    data: frame.png.clone(),
                    mime_type: "image/png".into(),
                }));
                messages.push(Message::User {
                    content,
                    timestamp: timestamp_ms(),
                });
            } else {
                messages.push(Message::developer(format!(
                    "[Earlier conversation summary id={} — originals: recall]\n{}",
                    node.entry_id, node.summary
                )));
            }
        }
        messages.extend(visible.iter().map(|entry| entry.message.clone()));
        Ok(messages)
    }
    pub fn current_model(&self) -> Option<String> {
        self.active_entries()
            .ok()?
            .into_iter()
            .rev()
            .find_map(|entry| match &entry.kind {
                EntryKind::ModelChange { model } => Some(model.clone()),
                _ => None,
            })
    }
    pub fn branch(&mut self, target: Option<&str>) -> Result<()> {
        self.append(EntryKind::Branch {
            target: target.map(str::to_owned),
        })?;
        Ok(())
    }
    pub fn fork(&self, directory: &Path) -> Result<Self> {
        Self::create_from(
            directory,
            &self.header().cwd,
            Some(self.header().id.clone()),
            &self.active_entries()?,
        )
    }
    pub fn compaction_plan(&self, keep_recent_messages: usize) -> Result<CompactionPlan> {
        let view = self.path_view()?;
        let visible = view.visible();
        let start = kept_start(visible, keep_recent_messages);
        Ok(CompactionPlan {
            previous: view
                .nodes
                .iter()
                .map(|node| PriorCompaction {
                    summary: node.summary.to_owned(),
                    archive: node.archive.cloned(),
                })
                .collect(),
            discarded: visible[..start]
                .iter()
                .map(|entry| entry.message.clone())
                .collect(),
        })
    }

    /// Replace the whole prefix of history (live nodes and raw messages before the kept
    /// tail) with one node one level above the highest node it replaces. Archives use
    /// this because an archive holds the whole discarded history.
    pub fn compact(
        &mut self,
        summary: String,
        archive: Option<Archive>,
        keep_recent_messages: usize,
    ) -> Result<()> {
        let node = {
            let view = self.path_view()?;
            let visible = view.visible();
            let cut = kept_start(visible, keep_recent_messages);
            if cut == 0 {
                return Ok(());
            }
            NewNode {
                summary,
                archive,
                level: view
                    .nodes
                    .iter()
                    .map(|node| node.level + 1)
                    .max()
                    .unwrap_or(0),
                span_start_id: view.messages[0].id.to_owned(),
                span_end_id: visible[cut - 1].id.to_owned(),
                covers: view
                    .nodes
                    .iter()
                    .map(|node| node.entry_id.to_owned())
                    .collect(),
            }
        };
        self.append_node(node)
    }
    /// Summary request for the oldest raw messages no node covers, keeping
    /// `keep_recent_messages` raw. `None` when there is nothing to summarize.
    pub fn raw_summary_plan(&self, keep_recent_messages: usize) -> Result<Option<NodePlan>> {
        let view = self.path_view()?;
        let visible = view.visible();
        let cut = kept_start(visible, keep_recent_messages);
        if cut == 0 {
            return Ok(None);
        }
        Ok(Some(NodePlan {
            input: visible[..cut]
                .iter()
                .map(|entry| entry.message.clone())
                .collect(),
            level: 0,
            span_start_id: visible[0].id.to_owned(),
            span_end_id: visible[cut - 1].id.to_owned(),
            covers: Vec::new(),
        }))
    }
    /// Summary request merging the adjacent pair of nodes with the lowest level; ties go
    /// to the earliest pair. `None` when no pair of nodes is adjacent and non-empty.
    pub fn merge_plan(&self) -> Result<Option<NodePlan>> {
        let view = self.path_view()?;
        let Some(pair) = view
            .nodes
            .windows(2)
            .filter(|pair| pair[1].end > pair[0].start)
            .min_by_key(|pair| pair[0].level.max(pair[1].level))
        else {
            return Ok(None);
        };
        Ok(Some(NodePlan {
            input: pair
                .iter()
                .map(|node| Message::user(node.summary))
                .collect(),
            level: pair[0].level.max(pair[1].level) + 1,
            span_start_id: view.messages[pair[0].start].id.to_owned(),
            span_end_id: view.messages[pair[1].end - 1].id.to_owned(),
            covers: vec![pair[0].entry_id.to_owned(), pair[1].entry_id.to_owned()],
        }))
    }
    /// Append a summary node. The live nodes it replaces must be exactly `covers`, and
    /// the nodes that remain must still tile a prefix with it.
    pub fn append_node(&mut self, node: NewNode) -> Result<()> {
        let first_kept_id = {
            let view = self.path_view()?;
            let invalid = || SessionError::InvalidSpan {
                start: node.span_start_id.clone(),
                end: node.span_end_id.clone(),
            };
            let start = message_index(&view.messages, &node.span_start_id).ok_or_else(invalid)?;
            let end = message_index(&view.messages, &node.span_end_id).ok_or_else(invalid)?;
            if start > end {
                return Err(invalid());
            }
            let end = end + 1;
            let mut inside: Vec<&str> = view
                .nodes
                .iter()
                .filter(|n| start <= n.start && n.end <= end)
                .map(|n| n.entry_id)
                .collect();
            let mut expected: Vec<&str> = node.covers.iter().map(String::as_str).collect();
            inside.sort_unstable();
            expected.sort_unstable();
            if inside != expected {
                return Err(invalid());
            }
            let mut ranges: Vec<(usize, usize)> = view
                .nodes
                .iter()
                .filter(|n| !(start <= n.start && n.end <= end))
                .map(|n| (n.start, n.end))
                .collect();
            ranges.push((start, end));
            ranges.sort_unstable();
            let mut cursor = 0;
            for (from, to) in ranges {
                if from != cursor {
                    return Err(invalid());
                }
                cursor = to;
            }
            view.messages.get(end).map(|message| message.id.to_owned())
        };
        self.append(EntryKind::Compaction {
            summary: node.summary,
            first_kept_id,
            span_start_id: Some(node.span_start_id),
            span_end_id: Some(node.span_end_id),
            archive: node.archive,
            level: node.level,
        })?;
        Ok(())
    }
    /// Message entries on the active path from `start` through `end`, inclusive,
    /// in order. Compaction, model and custom entries inside the span replaced no
    /// history, so they are skipped.
    pub fn entries_in_span(&self, start: &str, end: &str) -> Result<Vec<&SessionEntry>> {
        let messages: Vec<&SessionEntry> = self
            .active_entries()?
            .into_iter()
            .filter(|entry| matches!(entry.kind, EntryKind::Message { .. }))
            .collect();
        let from = messages.iter().position(|entry| entry.id == start);
        let to = messages.iter().position(|entry| entry.id == end);
        match (from, to) {
            (Some(from), Some(to)) if from <= to => Ok(messages[from..=to].to_vec()),
            _ => Err(SessionError::InvalidSpan {
                start: start.to_owned(),
                end: end.to_owned(),
            }),
        }
    }
}

fn message_index(messages: &[PathMessage<'_>], id: &str) -> Option<usize> {
    messages.iter().position(|message| message.id == id)
}
/// Move the retained boundary back until every retained tool result has its call.
fn kept_start(messages: &[PathMessage<'_>], keep_recent_messages: usize) -> usize {
    let mut start = messages.len().saturating_sub(keep_recent_messages);
    let mut calls = HashMap::new();
    let mut groups = Vec::new();
    for (index, entry) in messages.iter().enumerate() {
        match entry.message {
            Message::Assistant(assistant) => {
                for call in assistant.tool_calls() {
                    calls.insert(call.id.as_str(), index);
                }
            }
            Message::ToolResult { tool_call_id, .. } => {
                if let Some(call) = calls.get(tool_call_id.as_str()) {
                    groups.push((*call, index));
                }
            }
            _ => {}
        }
    }
    loop {
        let previous = start;
        for (call, result) in &groups {
            if *result >= start && *call < start {
                start = *call;
            }
        }
        if previous == start {
            return start;
        }
    }
}

fn resolve_target(journal: &Journal, target: Option<&str>) -> Result<Option<String>> {
    let mut current = target;
    while let Some(id) = current {
        let entry = &journal.entries[*journal
            .index
            .get(id)
            .ok_or_else(|| SessionError::UnknownTarget(id.to_owned()))?];
        if let EntryKind::Branch { target } = &entry.kind {
            current = target.as_deref();
        } else {
            return Ok(Some(id.to_owned()));
        }
    }
    Ok(None)
}
fn read_journal(path: &Path, file: &mut File) -> Result<Journal> {
    file.seek(SeekFrom::Start(0)).map_err(io_err(path))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(io_err(path))?;
    let first_end = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .unwrap_or(bytes.len());
    let header: SessionHeader =
        serde_json::from_slice(&bytes[..first_end]).map_err(|error| SessionError::Invalid {
            path: path.to_path_buf(),
            reason: format!("invalid header: {error}"),
        })?;
    if header.marker != "session" || header.id.is_empty() || header.version != SESSION_VERSION {
        return Err(SessionError::Invalid {
            path: path.to_path_buf(),
            reason: "unsupported session header".into(),
        });
    }
    let mut journal = Journal {
        header,
        entries: Vec::new(),
        index: HashMap::new(),
        leaf: None,
        size: bytes.len() as u64,
        valid_size: first_end as u64 + u64::from(first_end < bytes.len()),
        needs_newline: first_end == bytes.len(),
    };
    let mut offset = journal.valid_size as usize;
    while offset < bytes.len() {
        let end = bytes[offset..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|end| offset + end)
            .unwrap_or(bytes.len());
        let terminated = end < bytes.len();
        let raw = &bytes[offset..end];
        let entry: SessionEntry = match serde_json::from_slice(raw) {
            Ok(entry) => entry,
            Err(error) => {
                let incomplete_utf8 = std::str::from_utf8(raw)
                    .err()
                    .is_some_and(|error| error.error_len().is_none());
                if !terminated && (error.is_eof() || incomplete_utf8) {
                    break;
                }
                return Err(corrupt(path, format!("invalid record: {error}")));
            }
        };
        if entry.id.is_empty() || journal.index.contains_key(&entry.id) {
            return Err(corrupt(path, "empty or duplicate entry id"));
        }
        if entry
            .parent_id
            .as_ref()
            .is_some_and(|parent| !journal.index.contains_key(parent))
        {
            return Err(corrupt(path, "record references an unknown parent"));
        }
        let leaf = match &entry.kind {
            EntryKind::Branch { target } => resolve_target(&journal, target.as_deref())
                .map_err(|_| corrupt(path, "unknown branch target"))?,
            _ => Some(entry.id.clone()),
        };
        journal
            .index
            .insert(entry.id.clone(), journal.entries.len());
        journal.entries.push(entry);
        journal.leaf = leaf;
        offset = end + usize::from(terminated);
        journal.valid_size = offset as u64;
        journal.needs_newline = !terminated;
    }
    Ok(journal)
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
