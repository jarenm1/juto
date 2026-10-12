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
//!    workspace root and re-validate the target, so a directory component
//!    swapped for a symlink while waiting for the lock cannot redirect the
//!    write outside the workspace;
//! 4. capture the workspace tree before the edit;
//! 5. reject ignored paths;
//! 6. run the caller's transform, then atomically replace the file while
//!    preserving its permissions;
//! 7. capture the tree again, record a durable change-graph node, and return a
//!    path-scoped diff between the two trees.
//!
//! No Git commit is created: live snapshots are trees/blobs only, and the
//! returned diff is computed from the two trees and scoped to the mutated path,
//! so edits to other paths are never attributed to this operation. An external
//! (non-juto) writer that touches the *same* path between the atomic replace and
//! the after-snapshot is folded into `after_tree`: the diff stays path-scoped
//! and honest, but it describes the file as it stands at snapshot time.
//!
//! Failures are honest: a Git failure is a true error, and an error after the
//! file has been written explicitly reports that the file was applied and that
//! bookkeeping failed. Nothing is rolled back and success is never fabricated.

use std::{
    io::{Read as _, Write as _},
    os::unix::ffi::OsStrExt,
    os::unix::fs::OpenOptionsExt,
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
    // Git paths are raw bytes: a backslash is an ordinary filename byte and a
    // non-UTF-8 name must never be lossily rewritten. `display` is for humans
    // (messages and provenance) only.
    let rel_bytes = rel.as_os_str().as_bytes().to_vec();
    let display = rel.to_string_lossy().replace('\\', "/");

    // A partial fork holds only its sparse prefixes. An edit outside them would
    // be written to disk but preserved from the baseline by the next capture, so
    // it could never be recorded. Refuse it before the file is touched.
    if let Some(sparse) = graph.repository().sparse()
        && !sparse
            .iter()
            .any(|prefix| crate::git_transfer::path_under(&rel_bytes, prefix.as_bytes()))
    {
        return Err(format!(
            "{display}: path is outside this workspace's sparse spec ({})",
            sparse.join(", ")
        ));
    }

    // The lock covers the entire transaction: read, transform, write, snapshot,
    // and diff. It is held while the before/after trees are captured.
    let _lock = MutationLock::acquire(&root)?;
    // Re-validate after the lock. The check above fails fast, but a racing
    // non-juto process can swap a directory component for a symlink pointing
    // outside the workspace while this mutation waits for the lock; the write
    // below would then follow it.
    let target = validate_target(&root, cwd, path)?;
    let mut provenance = provenance.clone();
    provenance.path = Some(display.clone());

    let before_tree = graph.snapshot_tree().map_err(|e| {
        format!("{display}: cannot capture the workspace tree before mutation: {e:#}")
    })?;
    let before_text = read_optional(&target)?;

    let tracked = graph
        .path_tracked(&before_tree, &rel_bytes)
        .map_err(|e| format!("{display}: cannot check whether the path is tracked: {e:#}"))?;
    if !tracked
        && graph
            .path_ignored(&rel_bytes)
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
        .path_tracked(&after_tree, &rel_bytes)
        .map_err(|e| format!(
            "{display}: file applied but the tracked-path check failed (before_tree={before_tree}, after_tree={after_tree}): {e:#}"
        ))?
    {
        return Err(format!(
            "{display}: file applied but the path is ignored by Git (before_tree={before_tree}, after_tree={after_tree})"
        ));
    }

    let raw_diff = graph
        .diff_trees(&before_tree, &after_tree, &rel_bytes)
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
            // A FIFO, socket, or device is not a text file: reading one can
            // block forever (a FIFO with no writer) while the workspace lock is
            // held, and writing one is never a file edit.
            if !md.is_file() {
                return Err(format!(
                    "refusing to write over non-regular file {} (only regular files are supported)",
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
///
/// The file is opened with `O_NONBLOCK` and its type is checked on the open
/// descriptor, so a FIFO or device swapped in after validation can never block
/// the mutation while it holds the workspace lock.
fn read_optional(path: &Path) -> Result<Option<String>, String> {
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let metadata = file
        .metadata()
        .map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "refusing to read {}: not a regular file",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    (&file)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;
    use std::process::Command;
    use std::sync::mpsc;
    use std::time::Duration;

    fn git(cwd: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn repo() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        git(directory.path(), &["init", "-q", "-b", "main"]);
        git(directory.path(), &["config", "user.name", "Vcs Test"]);
        git(
            directory.path(),
            &["config", "user.email", "vcs@example.test"],
        );
        std::fs::write(directory.path().join("base"), "base\n").unwrap();
        git(directory.path(), &["add", "-A"]);
        git(directory.path(), &["commit", "-q", "-m", "base"]);
        directory
    }

    fn provenance() -> EditProvenance {
        EditProvenance {
            session_id: "test-session".into(),
            tool: "write".into(),
            selection_mode: None,
            path: None,
        }
    }

    fn apply(cwd: &Path, path: &Path, contents: &str) -> Result<Value, String> {
        mutate(cwd, path, &CancellationToken::new(), &provenance(), |_| {
            Ok(contents.to_string())
        })
    }

    #[test]
    fn mutate_rejects_a_fifo_target_without_blocking() {
        let directory = repo();
        let cwd = directory.path().to_path_buf();
        let fifo = cwd.join("pipe");
        assert!(
            Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );

        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(apply(&cwd, Path::new("pipe"), "written\n"));
        });

        // Reading a FIFO blocks until a writer appears; the mutation must refuse
        // the target instead of holding the workspace lock forever.
        let outcome = receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("mutate must not block on a FIFO target");
        let error = outcome.expect_err("a FIFO target must be refused");
        assert!(error.contains("regular file"), "{error}");
        assert!(!std::fs::metadata(&fifo).unwrap().is_file());
    }

    #[test]
    fn mutate_rejects_a_unix_socket_target() {
        let directory = repo();
        let cwd = directory.path();
        let socket = cwd.join("sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();

        let error = apply(cwd, Path::new("sock"), "written\n").unwrap_err();
        assert!(error.contains("regular file"), "{error}");
        drop(listener);
    }

    #[test]
    fn mutate_revalidates_the_target_after_acquiring_the_lock() {
        let directory = repo();
        let cwd = directory.path().to_path_buf();
        // The target is tracked and its directory is ignored, so the snapshot
        // keeps the path and the pre-write ignore check is skipped: only the
        // containment/symlink validation can stop the write.
        std::fs::write(cwd.join(".gitignore"), "d\n").unwrap();
        std::fs::create_dir_all(cwd.join("d")).unwrap();
        std::fs::write(cwd.join("d/escaped.txt"), "original\n").unwrap();
        git(&cwd, &["add", "-A"]);
        git(&cwd, &["add", "-f", "d/escaped.txt"]);
        git(&cwd, &["commit", "-q", "-m", "tracked target"]);
        let outside = tempfile::tempdir().unwrap();
        // The same relative path exists outside, so the snapshot still sees a
        // tracked file through the swapped symlink.
        std::fs::write(outside.path().join("escaped.txt"), "outside\n").unwrap();

        // Hold the workspace mutation lock so the racing mutation blocks after
        // its first (pre-lock) validation.
        let root = std::fs::canonicalize(&cwd).unwrap();
        let lock = MutationLock::acquire(&root).unwrap();

        let (sender, receiver) = mpsc::channel();
        let thread_cwd = cwd.clone();
        let handle = std::thread::spawn(move || {
            let _ = sender.send(apply(&thread_cwd, Path::new("d/escaped.txt"), "escaped\n"));
        });

        // Wait until the mutation has opened the change graph (which happens
        // before its pre-lock validation), then let that validation finish.
        let state = cwd.join(".git/juto/graph.json");
        for _ in 0..500 {
            if state.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(state.exists(), "the change graph must be initialized");
        std::thread::sleep(Duration::from_millis(300));

        // Swap the validated directory for a symlink pointing outside the
        // workspace, then release the lock.
        std::fs::remove_dir_all(cwd.join("d")).unwrap();
        symlink(outside.path(), cwd.join("d")).unwrap();
        drop(lock);

        let outcome = receiver.recv_timeout(Duration::from_secs(30)).unwrap();
        handle.join().unwrap();
        assert_eq!(
            std::fs::read_to_string(outside.path().join("escaped.txt")).unwrap(),
            "outside\n",
            "the write must not escape the workspace"
        );
        let error = outcome.expect_err("a swapped symlink must be refused");
        assert!(
            error.contains("refusing to edit through symlink"),
            "{error}"
        );
    }

    #[test]
    fn mutate_uses_raw_path_bytes_for_git_queries() {
        let directory = repo();
        let cwd = directory.path();

        // A backslash is an ordinary filename byte; the human-readable display
        // path must never be used to query Git.
        let receipt = apply(cwd, Path::new("a\\b.txt"), "backslash\n").unwrap();
        // `path` is the human-facing display form (backslashes normalized);
        // Git queries use the raw relative bytes.
        assert_eq!(receipt["path"], "a/b.txt");
        assert!(receipt["graph_node"].is_string(), "{receipt}");
        assert!(
            receipt["diff"].as_str().unwrap().contains("backslash"),
            "{receipt}"
        );
        assert_eq!(
            std::fs::read_to_string(cwd.join("a\\b.txt")).unwrap(),
            "backslash\n"
        );

        // A non-UTF-8 name must round-trip as raw bytes rather than U+FFFD.
        let raw = PathBuf::from(OsStr::from_bytes(b"raw-\xff.txt"));
        let receipt = apply(cwd, &raw, "raw\n").unwrap();
        assert!(receipt["graph_node"].is_string(), "{receipt}");
        assert_eq!(std::fs::read(cwd.join(&raw)).unwrap(), b"raw\n");
    }

    #[test]
    fn mutate_refuses_a_path_outside_the_sparse_spec() {
        let directory = repo();
        let cwd = directory.path();
        std::fs::create_dir_all(cwd.join("crates/foo")).unwrap();
        std::fs::write(cwd.join("crates/foo/a.txt"), "a\n").unwrap();
        std::fs::create_dir_all(cwd.join("crates/bar")).unwrap();
        std::fs::write(cwd.join("crates/bar/b.txt"), "b\n").unwrap();
        git(cwd, &["add", "-A"]);
        git(cwd, &["commit", "-q", "-m", "layout"]);

        let graph = ChangeGraph::open(cwd).unwrap();
        let root = graph.capture("main", provenance(), None).unwrap();

        let workspaces = tempfile::tempdir().unwrap();
        let left_dir = workspaces.path().join("left");
        graph
            .fork_sparse("left", &root.id, &left_dir, &["crates/foo".to_string()])
            .unwrap();

        // An edit inside the spec is recorded normally.
        let inside = apply(&left_dir, Path::new("crates/foo/a.txt"), "a edited\n").unwrap();
        assert_eq!(inside["changed"], true, "{inside}");

        // An edit outside the spec is refused before the file is touched: the
        // next capture would preserve the baseline's version, so the edit could
        // never be recorded.
        let error = apply(&left_dir, Path::new("crates/bar/b.txt"), "b edited\n").unwrap_err();
        assert!(
            error.contains("outside this workspace's sparse spec"),
            "{error}"
        );
        assert!(
            !left_dir.join("crates/bar/b.txt").exists(),
            "a refused edit must not create the file"
        );
    }
}
