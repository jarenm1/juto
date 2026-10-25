//! On-disk form of the main inbox: the undrained messages, each with the
//! origin it is addressed to, and the archive its digests point into. The
//! file is replaced atomically, so a crash leaves either the old or the new
//! state, never a torn one. Files written before the user's notice list was
//! removed still carry a `user` field; it is ignored on load.

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::Path,
};

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};

use crate::subagents::{ArchivedReport, Queued};

/// On-disk format version. A file with any other version is refused, so no
/// message is discarded by a reader that does not understand it.
const FORMAT_VERSION: u32 = 1;

/// Contents of an inbox file, as read back on bind.
pub(crate) struct Loaded {
    pub(crate) queued: Vec<Queued>,
    pub(crate) archive: HashMap<String, ArchivedReport>,
}

#[derive(Serialize)]
struct StoredRef<'a> {
    version: u32,
    queued: &'a [Queued],
    archive: &'a HashMap<String, ArchivedReport>,
}

#[derive(Deserialize)]
struct Stored {
    version: u32,
    queued: Vec<Queued>,
    archive: HashMap<String, ArchivedReport>,
}

/// Read the inbox at `path`. A missing file is an empty inbox. A file that
/// cannot be parsed, or that has an unknown version, is an error: binding
/// fails instead of starting over and losing its messages.
pub(crate) fn load(path: &Path) -> Result<Loaded> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return Ok(Loaded {
                queued: Vec::new(),
                archive: HashMap::new(),
            });
        }
        Err(source) => {
            return Err(source).with_context(|| format!("reading main inbox {}", path.display()));
        }
    };
    let stored: Stored = serde_json::from_slice(&bytes)
        .with_context(|| format!("main inbox {} is corrupt", path.display()))?;
    if stored.version != FORMAT_VERSION {
        bail!(
            "main inbox {} has unsupported format version {}",
            path.display(),
            stored.version
        );
    }
    Ok(Loaded {
        queued: stored.queued,
        archive: stored.archive,
    })
}

/// Write the inbox to `path` atomically: a temporary file is fsynced and
/// renamed over the target, then the directory is fsynced so the rename
/// itself is durable.
pub(crate) fn save(
    path: &Path,
    queued: &[Queued],
    archive: &HashMap<String, ArchivedReport>,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(&StoredRef {
        version: FORMAT_VERSION,
        queued,
        archive,
    })
    .context("encoding main inbox")?;
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory).with_context(|| format!("creating {}", directory.display()))?;
    let temporary = path.with_extension("json.tmp");
    write_synced(&temporary, &bytes)
        .with_context(|| format!("writing main inbox {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("replacing main inbox {}", path.display()))?;
    File::open(directory)
        .and_then(|dir| dir.sync_all())
        .with_context(|| format!("syncing {}", directory.display()))
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::subagents::{MainMessage, MainMessageKind};

    fn queued(origin: Option<&str>, body: &str) -> Queued {
        Queued {
            origin: origin.map(str::to_owned),
            message: MainMessage {
                from: "child".into(),
                name: None,
                kind: MainMessageKind::Completed,
                body: body.into(),
                digest_of: None,
            },
        }
    }

    #[test]
    fn missing_file_loads_as_empty_inbox() {
        let root = tempdir().expect("tempdir");
        let loaded = load(&root.path().join("absent.json")).expect("load");
        assert!(loaded.queued.is_empty());
        assert!(loaded.archive.is_empty());
    }

    #[test]
    fn saved_queue_and_origins_round_trip() {
        let root = tempdir().expect("tempdir");
        let path = root.path().join("inbox").join("main.json");
        let queue = vec![queued(Some("thread-a"), "one"), queued(None, "two")];
        save(&path, &queue, &HashMap::new()).expect("save");
        let loaded = load(&path).expect("load");
        assert_eq!(loaded.queued.len(), 2);
        assert_eq!(loaded.queued[0].origin.as_deref(), Some("thread-a"));
        assert_eq!(loaded.queued[0].message, queue[0].message);
        assert_eq!(loaded.queued[1].origin, None);
    }

    #[test]
    fn a_file_with_the_retired_user_notices_still_loads() {
        let root = tempdir().expect("tempdir");
        let path = root.path().join("main.json");
        fs::write(
            &path,
            br#"{"version":1,"queued":[],"archive":{},"user":[{"from":"c","kind":"completed","body":"b"}]}"#,
        )
        .expect("write");
        let loaded = load(&path).expect("load");
        assert!(loaded.queued.is_empty());
    }

    #[test]
    fn unknown_format_version_is_refused() {
        let root = tempdir().expect("tempdir");
        let path = root.path().join("main.json");
        fs::write(
            &path,
            br#"{"version":99,"queued":[],"archive":{},"user":[]}"#,
        )
        .expect("write");
        let error = load(&path).err().expect("refused");
        assert!(error.to_string().contains("unsupported format version 99"));
    }
}
