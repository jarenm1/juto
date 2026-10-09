//! Append-only JSONL session trees. Navigation and compaction are durable records.
//! Behavioral reference: OMP 579da1d6; Juto's journal is not byte-compatible with OMP.
use fs2::FileExt;
use juto_ai::{Message, timestamp_ms};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

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
    fn visible_messages(&self) -> Result<(Option<&str>, Vec<&SessionEntry>)> {
        let entries = self.active_entries()?;
        let boundary = entries
            .iter()
            .rposition(|entry| matches!(entry.kind, EntryKind::Compaction { .. }));
        if let Some(boundary) = boundary {
            let EntryKind::Compaction {
                summary,
                first_kept_id,
            } = &entries[boundary].kind
            else {
                unreachable!()
            };
            let floor = match first_kept_id {
                Some(id) => entries[..boundary]
                    .iter()
                    .position(|entry| entry.id == *id)
                    .ok_or_else(|| {
                        corrupt(&self.path, "compaction references a missing retained entry")
                    })?,
                None => boundary,
            };
            let visible = entries
                .into_iter()
                .enumerate()
                .filter_map(|(index, entry)| {
                    if index >= floor
                        && index != boundary
                        && matches!(entry.kind, EntryKind::Message { .. })
                    {
                        Some(entry)
                    } else {
                        None
                    }
                })
                .collect();
            Ok((Some(summary), visible))
        } else {
            Ok((
                None,
                entries
                    .into_iter()
                    .filter(|entry| matches!(entry.kind, EntryKind::Message { .. }))
                    .collect(),
            ))
        }
    }
    pub fn messages(&self) -> Result<Vec<Message>> {
        let (summary, entries) = self.visible_messages()?;
        let mut messages = Vec::with_capacity(entries.len() + usize::from(summary.is_some()));
        if let Some(summary) = summary {
            messages.push(Message::developer(format!(
                "[Earlier conversation summary]\n{summary}"
            )));
        }
        for entry in entries {
            if let EntryKind::Message { message } = &entry.kind {
                messages.push(message.clone());
            }
        }
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
    pub fn compact(&mut self, summary: String, keep_recent_messages: usize) -> Result<()> {
        let (_, messages) = self.visible_messages()?;
        let mut start = messages.len().saturating_sub(keep_recent_messages);
        let mut calls = HashMap::new();
        let mut groups = Vec::new();
        for (index, entry) in messages.iter().enumerate() {
            if let EntryKind::Message { message } = &entry.kind {
                match message {
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
        }
        loop {
            let previous = start;
            for (call, result) in &groups {
                if *result >= start && *call < start {
                    start = *call;
                }
            }
            if previous == start {
                break;
            }
        }
        let first_kept_id = messages.get(start).map(|entry| entry.id.clone());
        self.append(EntryKind::Compaction {
            summary,
            first_kept_id,
        })?;
        Ok(())
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
