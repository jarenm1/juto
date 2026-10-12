//! Native Git-backed mutation path for file edits.
//!
//! [`mutate`] is the single write path shared by the `write`/`edit` tools and
//! the contextual-edit tools. It runs synchronously (callers wrap it in
//! `spawn_blocking`) and performs one transaction:
//!
//! 1. discover the workspace root through the change graph (a plain Git
//!    repository or a native linked fork directory);
//! 2. reject targets outside the canonical workspace root, through symlinks,
//!    or inside `.git`/`.jj`/native metadata;
//! 3. take a cross-process advisory lock (`.juto-mutation.lock`) in the
//!    workspace root;
//! 4. capture the workspace tree before the edit;
//! 5. reject ignored paths;
//! 6. run the caller's transform, then atomically replace the file while
//!    preserving its permissions;
//! 7. capture the tree again, record a durable change-graph node, and return a
//!    path-scoped diff between the two trees.
//!
//! No Git commit is created: live snapshots are trees/blobs only, and the
//! path-scoped diff is computed from the two trees, so unrelated preexisting
//! edits are never attributed to this operation.
//!
//! Failures are honest: a Git failure is a true error, and an error after the
//! file has been written explicitly reports that the file was applied and that
//! bookkeeping failed. Nothing is rolled back and success is never fabricated.

use std::{
    io::Write as _,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use fs2::FileExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::change_graph::{ChangeGraph, EditProvenance};

/// Maximum bytes of diff output returned inline; longer diffs are truncated at
/// a char boundary and flagged.
const DIFF_MAX_BYTES: usize = 64 * 1024;

/// Name of the per-workspace advisory lock file, created in the workspace root.
///
/// It lives at the workspace root (not inside `.git`/`.jj`) so it serializes
/// mutations within one working copy while the native snapshot capture excludes
/// it from every tree.
const LOCK_FILE_NAME: &str = ".juto-mutation.lock";

/// Native metadata names that must never be edited as ordinary files.
const METADATA_NAMES: &[&str] = &[".git", ".jj", ".juto-workspace.json", LOCK_FILE_NAME];

/// Monotonic counter for same-directory temporary file names.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Apply `transform` to `path` through a native, atomic, locked mutation.
///
/// `cwd` is the caller's working directory; a relative `path` resolves against
/// it. `transform` receives the current file contents (`None` when the file
/// does not exist) and returns the new contents. `provenance` identifies the
/// originating session/tool/selection and is echoed in the receipt.
/// The returned JSON describes the applied change:
///
/// ```json
/// {
///   "path": "src/lib.rs",
///   "before_tree": "<hex tree id>",
///   "after_tree": "<hex tree id>",
///   "graph_node": "<change node id>",
///   "diff": "<git diff, possibly truncated>",
///   "diff_truncated": false,
///   "changed": true,
///   "no_op": false,
///   "applied": true,
///   "cancelled_after_apply": false,
///   "provenance": {"session_id": "...", "tool": "...", "selection_mode": null, "path": null}
/// }
/// ```
///
/// A transform that returns the current contents is an honest no-op: nothing is
/// written, `no_op` is `true`, and the diff is empty.
pub fn mutate(
    cwd: &Path,
    path: &Path,
    cancel: &CancellationToken,
    provenance: &EditProvenance,
    transform: impl FnOnce(Option<&str>) -> Result<String, String>,
) -> Result<Value, String> {
    if cancel.is_cancelled() {
        return Err("cancelled before mutation".to_string());
    }

    let graph = ChangeGraph::open(cwd)
        .map_err(|e| format!("{}: cannot open change graph: {e:#}", cwd.display()))?;
    // Canonicalize so containment checks compare like-for-like against the
    // canonical working directory. The lock file resolves to the same inode as
    // the graph's own workspace path, so mutual exclusion is unaffected.
    let workspace = graph.workspace();
    let root = std::fs::canonicalize(workspace)
        .map_err(|e| format!("cannot resolve workspace {}: {e}", workspace.display()))?;
    let target = validate_target(&root, cwd, path)?;
    let rel = target
        .strip_prefix(&root)
        .map_err(|_| format!("{} is outside {}", target.display(), root.display()))?;
    let display = rel.to_string_lossy().replace('\\', "/");

    // The lock covers the entire transaction: read, transform, write, snapshot,
    // and diff. It is held while the before/after trees are captured.
    let _lock = MutationLock::acquire(&root)?;
    let mut provenance = provenance.clone();
    provenance.path = Some(display.clone());

    let before_tree = graph.snapshot_tree().map_err(|e| {
        format!("{display}: cannot capture the workspace tree before mutation: {e:#}")
    })?;
    let before_text = read_optional(&target)?;

    let tracked = graph
        .path_tracked(&before_tree, &display)
        .map_err(|e| format!("{display}: cannot check whether the path is tracked: {e:#}"))?;
    if !tracked
        && graph
            .path_ignored(&display)
            .map_err(|e| format!("{display}: cannot check ignore rules: {e:#}"))?
    {
        return Err(format!(
            "{display}: path is ignored; refusing an untracked mutation"
        ));
    }

    let new_text = transform(before_text.as_deref())?;

    if before_text.as_deref() == Some(new_text.as_str()) {
        return Ok(json!({
            "path": display,
            "before_tree": before_tree,
            "after_tree": before_tree,
            "graph_node": null,
            "diff": "",
            "diff_truncated": false,
            "changed": false,
            "no_op": true,
            "applied": false,
            "cancelled_after_apply": false,
            "provenance": provenance,
        }));
    }

    if cancel.is_cancelled() {
        return Err("cancelled before mutation".to_string());
    }

    atomic_write_preserving(&target, new_text.as_bytes())?;

    // The file is now applied. Bookkeeping must complete even if cancellation
    // was requested in the meantime; we report the applied change rather than
    // claiming a clean cancellation.

    let after_tree = graph.snapshot_tree().map_err(|e| {
        format!(
            "{display}: file applied but the workspace snapshot failed (before_tree={before_tree}, after_tree=unavailable): {e:#}"
        )
    })?;

    // A newly created path may be ignored by Git; the capture silently omits it.
    // Fail honestly rather than record a node for a change Git cannot see.
    if !graph
        .path_tracked(&after_tree, &display)
        .map_err(|e| format!(
            "{display}: file applied but the tracked-path check failed (before_tree={before_tree}, after_tree={after_tree}): {e:#}"
        ))?
    {
        return Err(format!(
            "{display}: file applied but the path is ignored by Git (before_tree={before_tree}, after_tree={after_tree})"
        ));
    }

    let raw_diff = graph
        .diff_trees(&before_tree, &after_tree, &display)
        .map_err(|e| {
            format!(
                "{display}: file applied but the diff failed (before_tree={before_tree}, after_tree={after_tree}): {e:#}"
            )
        })?;

    if raw_diff.trim().is_empty() {
        return Err(format!(
            "{display}: file applied but Git reports no tracked change (before_tree={before_tree}, after_tree={after_tree})"
        ));
    }

    let node = graph
        .record_edit(&before_tree, &after_tree, provenance.clone())
        .map_err(|e| {
            format!(
                "{display}: file applied but change graph bookkeeping failed (before_tree={before_tree}, after_tree={after_tree}): {e:#}"
            )
        })?
        .ok_or_else(|| {
            format!(
                "{display}: file applied but change graph reports no changed tree (before_tree={before_tree}, after_tree={after_tree})"
            )
        })?;
    let (diff, diff_truncated) = bound_text(raw_diff, DIFF_MAX_BYTES);
    let cancelled_after_apply = cancel.is_cancelled();

    Ok(json!({
        "path": display,
        "before_tree": before_tree,
        "after_tree": after_tree,
        "graph_node": node.id,
        "diff": diff,
        "diff_truncated": diff_truncated,
        "changed": true,
        "no_op": false,
        "applied": true,
        "cancelled_after_apply": cancelled_after_apply,
        "provenance": provenance,
    }))
}

// ---------------------------------------------------------------------------
// Workspace discovery and target validation
// ---------------------------------------------------------------------------

/// Resolve `path` against `cwd` and require it to be a writable file inside the
/// canonical workspace `root`.
///
/// Rejects lexical traversal outside the root, `.git`/`.jj`/native metadata
/// components, symlinked components, and directories. For a new file, the
/// nearest existing ancestor is validated and must canonicalize inside `root`.
fn validate_target(root: &Path, cwd: &Path, path: &Path) -> Result<PathBuf, String> {
    let canon_cwd = std::fs::canonicalize(cwd)
        .map_err(|e| format!("cannot resolve working directory {}: {e}", cwd.display()))?;
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        canon_cwd.join(path)
    };
    let lex = normalize_lexical(&abs);

    for comp in lex.components() {
        if let Component::Normal(name) = comp
            && METADATA_NAMES.iter().any(|meta| name == *meta)
        {
            return Err(format!(
                "refusing to edit repository metadata path {}",
                lex.display()
            ));
        }
    }

    let rel = lex.strip_prefix(root).map_err(|_| {
        format!(
            "path {} is outside the workspace {}",
            lex.display(),
            root.display()
        )
    })?;
    if rel.as_os_str().is_empty() {
        return Err(format!(
            "path {} is the workspace root itself",
            lex.display()
        ));
    }

    // Walk the existing prefix: reject symlinks and find the deepest existing
    // ancestor (which may be the target itself).
    let mut cur = root.to_path_buf();
    let mut existing_ancestor: Option<PathBuf> = None;
    for comp in rel.components() {
        cur.push(comp);
        match std::fs::symlink_metadata(&cur) {
            Ok(md) => {
                if md.file_type().is_symlink() {
                    return Err(format!(
                        "refusing to edit through symlink {}",
                        cur.display()
                    ));
                }
                existing_ancestor = Some(cur.clone());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => return Err(format!("cannot inspect {}: {e}", cur.display())),
        }
    }

    if let Some(anc) = &existing_ancestor {
        if anc == &lex {
            let md = std::fs::symlink_metadata(&lex)
                .map_err(|e| format!("cannot inspect {}: {e}", lex.display()))?;
            if md.is_dir() {
                return Err(format!(
                    "refusing to write a file over directory {}",
                    lex.display()
                ));
            }
        }
        // Defense in depth: the existing ancestor must resolve inside the root.
        let canon = std::fs::canonicalize(anc)
            .map_err(|e| format!("cannot canonicalize {}: {e}", anc.display()))?;
        if !canon.starts_with(root) {
            return Err(format!(
                "path {} resolves outside the workspace {}",
                lex.display(),
                root.display()
            ));
        }
    }

    Ok(lex)
}

/// Lexically resolve `.` and `..` without touching the filesystem.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(c) => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Cross-process lock
// ---------------------------------------------------------------------------

/// Exclusive advisory lock over a workspace's mutation transaction.
pub(crate) struct MutationLock {
    file: std::fs::File,
}

impl MutationLock {
    /// Acquire the workspace mutation lock, creating `.juto-mutation.lock` in
    /// `root` if needed. The lock file is excluded from every native snapshot.
    pub(crate) fn acquire(root: &Path) -> Result<Self, String> {
        let path = root.join(LOCK_FILE_NAME);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("cannot open mutation lock {}: {e}", path.display()))?;
        file.lock_exclusive()
            .map_err(|e| format!("cannot lock {}: {e}", path.display()))?;
        Ok(Self { file })
    }
}

impl Drop for MutationLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Read a UTF-8 text file, returning `None` when it does not exist.
fn read_optional(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read(path) {
        Ok(bytes) => {
            if bytes.contains(&0) {
                return Err(format!(
                    "{}: binary file (NUL byte); only UTF-8 text is supported",
                    path.display()
                ));
            }
            let text = String::from_utf8(bytes)
                .map_err(|e| format!("{}: not valid UTF-8 ({e})", path.display()))?;
            Ok(Some(text))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("cannot read {}: {e}", path.display())),
    }
}

/// Atomically replace `target` with `contents`, preserving existing permissions.
///
/// Writes to a same-directory temporary file created with `create_new`, flushes
/// and syncs it, applies the destination's permissions, then renames over the
/// target. A failure never truncates the destination.
fn atomic_write_preserving(target: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = target
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", target.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;

    let existing_perms = std::fs::metadata(target).ok().map(|m| m.permissions());
    let file_name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(
        ".{file_name}.juto-{}-{unique}.tmp",
        std::process::id()
    ));

    let result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("cannot create temp file {}: {e}", tmp.display()))?;
        file.write_all(contents)
            .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        file.flush()
            .map_err(|e| format!("cannot flush {}: {e}", tmp.display()))?;
        file.sync_all()
            .map_err(|e| format!("cannot sync {}: {e}", tmp.display()))?;
        if let Some(perms) = existing_perms {
            std::fs::set_permissions(&tmp, perms)
                .map_err(|e| format!("cannot set permissions on {}: {e}", tmp.display()))?;
        }
        drop(file);
        std::fs::rename(&tmp, target)
            .map_err(|e| format!("cannot replace {}: {e}", target.display()))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    } else if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    result
}

/// Truncate `text` to `max` bytes at a char boundary, flagging truncation.
fn bound_text(text: String, max: usize) -> (String, bool) {
    if text.len() <= max {
        return (text, false);
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}
