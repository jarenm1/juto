//! Jujutsu-backed mutation path for file edits.
//!
//! [`mutate`] is the single write path shared by the `write`/`edit` tools and
//! the contextual-edit tools. It runs synchronously (callers wrap it in
//! `spawn_blocking`) and performs one transaction:
//!
//! 1. discover the workspace root with `jj root` from the caller's cwd;
//! 2. reject targets outside the canonical workspace root, through symlinks,
//!    or inside `.jj`/`.git` metadata;
//! 3. take a cross-process advisory lock on a file under `.jj`;
//! 4. snapshot the working copy and capture the working-copy commit/change id;
//! 5. reject conflicted or ignored paths;
//! 6. run the caller's transform, then atomically replace the file while
//!    preserving its permissions;
//! 7. snapshot again and return a path-scoped `jj diff --git` between the two
//!    snapshots.
//!
//! No new change is created: the working-copy change id is retained and no
//! `jj new`/`jj commit`/`jj rebase` is ever invoked. The path-scoped diff is
//! computed from the two snapshots, so unrelated preexisting edits are never
//! attributed to this operation.
//!
//! Failures are honest: a jj failure is a true error, and an error after the
//! file has been written explicitly reports that the file was applied and that
//! bookkeeping failed. Nothing is rolled back and success is never fabricated.

use std::{
    io::Write as _,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use fs2::FileExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

#[derive(serde::Serialize)]
pub(crate) struct EditProvenance {
    pub session_id: String,
    pub tool: String,
    pub selection_mode: Option<String>,
    pub path: Option<String>,
}

/// Maximum bytes of `jj diff` output returned inline; longer diffs are
/// truncated at a char boundary and flagged.
const DIFF_MAX_BYTES: usize = 64 * 1024;

/// Name of the per-workspace advisory lock file, created under `.jj`.
///
/// It lives in the workspace's own `.jj` directory (not the shared repository
/// directory referenced by `.jj/repo`), so it serializes mutations within one
/// working copy without ambiguously locking shared state.
const LOCK_FILE_NAME: &str = "juto-mutation.lock";

/// Monotonic counter for same-directory temporary file names.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Apply `transform` to `path` through a jj-backed, atomic, locked mutation.
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
///   "before_commit": "<hex>",
///   "after_commit": "<hex>",
///   "change_id": "<change id>",
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

    let root = jj_root(cwd)?;
    let target = validate_target(&root, cwd, path)?;
    let rel = target
        .strip_prefix(&root)
        .map_err(|_| format!("{} is outside {}", target.display(), root.display()))?;
    let display = rel.to_string_lossy().replace('\\', "/");
    let fileset = fileset_literal(&display);

    // The lock covers the entire transaction: read, transform, write, snapshot,
    // and diff.
    let _lock = MutationLock::acquire(&root)?;

    let before = jj_working_copy(&root)?;
    let before_text = read_optional(&target)?;

    if jj_path_conflicted(&root, &fileset)? {
        return Err(format!(
            "{display}: path has unresolved jujutsu conflicts; resolve them before editing"
        ));
    }

    // An existing file that jj does not track is ignored; refuse rather than
    // silently produce an untracked change.
    if before_text.is_some() && !jj_path_tracked(&root, &fileset)? {
        return Err(format!(
            "{display}: path is ignored by jujutsu; refusing to edit an untracked path"
        ));
    }

    let new_text = transform(before_text.as_deref())?;

    if before_text.as_deref() == Some(new_text.as_str()) {
        return Ok(json!({
            "path": display,
            "before_commit": before.commit_id,
            "after_commit": before.commit_id,
            "change_id": before.change_id,
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

    let after = jj_working_copy(&root).map_err(|e| {
        format!("{display}: file applied but jujutsu bookkeeping failed (before_commit={}, after_commit=unavailable): {e}", before.commit_id)
    })?;
    let raw_diff = jj_path_diff(&root, &before.commit_id, &after.commit_id, &fileset).map_err(
        |e| format!("{display}: file applied but jujutsu diff failed (before_commit={}, after_commit={}): {e}", before.commit_id, after.commit_id),
    )?;

    if raw_diff.trim().is_empty() {
        return Err(format!(
            "{display}: file applied but jujutsu reports no tracked change (path is ignored or untracked; before_commit={}, after_commit={})",
            before.commit_id, after.commit_id
        ));
    }

    let (diff, diff_truncated) = bound_text(raw_diff, DIFF_MAX_BYTES);
    let cancelled_after_apply = cancel.is_cancelled();

    Ok(json!({
        "path": display,
        "before_commit": before.commit_id,
        "after_commit": after.commit_id,
        "change_id": after.change_id,
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

/// Locate the canonical workspace root by running `jj root` in `cwd`.
fn jj_root(cwd: &Path) -> Result<PathBuf, String> {
    let out = run_jj(cwd, &["root", "--ignore-working-copy"])?;
    if !out.status.success() {
        return Err(format!(
            "not inside a jujutsu workspace ({}): {}",
            cwd.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let raw = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if raw.is_empty() {
        return Err("`jj root` returned an empty path".to_string());
    }
    std::fs::canonicalize(&raw).map_err(|e| format!("cannot canonicalize jujutsu root {raw}: {e}"))
}

/// Resolve `path` against `cwd` and require it to be a writable file inside the
/// canonical workspace `root`.
///
/// Rejects lexical traversal outside the root, `.jj`/`.git` metadata
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
            && (name == ".jj" || name == ".git")
        {
            return Err(format!(
                "refusing to edit jujutsu/git metadata path {}",
                lex.display()
            ));
        }
    }

    let rel = lex.strip_prefix(root).map_err(|_| {
        format!(
            "path {} is outside the jujutsu workspace {}",
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
                "path {} resolves outside the jujutsu workspace {}",
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
    pub(crate) fn acquire(root: &Path) -> Result<Self, String> {
        let dir = root.join(".jj");
        if !dir.is_dir() {
            return Err(format!(
                "{} is not a jujutsu metadata directory",
                dir.display()
            ));
        }
        let path = dir.join(LOCK_FILE_NAME);
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

// ---------------------------------------------------------------------------
// jj helpers
// ---------------------------------------------------------------------------

/// Working-copy identity captured from a snapshot.
struct WorkingCopy {
    commit_id: String,
    change_id: String,
}

/// Run `jj` in `dir` without a shell.
fn run_jj(dir: &Path, args: &[&str]) -> Result<std::process::Output, String> {
    Command::new("jj")
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|e| {
            format!(
                "failed to run `jj {}` in {}: {e}",
                args.join(" "),
                dir.display()
            )
        })
}

/// Format a jj command failure with its stderr.
fn jj_err(command: &str, out: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stderr = stderr.trim();
    if stderr.is_empty() {
        format!("`{command}` failed with status {}", out.status)
    } else {
        format!("`{command}` failed: {stderr}")
    }
}

/// Snapshot the working copy and read its commit id and change id.
fn jj_working_copy(root: &Path) -> Result<WorkingCopy, String> {
    const TEMPLATE: &str = "commit_id ++ \" \" ++ change_id";
    let out = run_jj(root, &["log", "--no-graph", "-r", "@", "-T", TEMPLATE])?;
    if !out.status.success() {
        return Err(jj_err(
            "jj log -r @ -T 'commit_id ++ \" \" ++ change_id'",
            &out,
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.split_whitespace();
    let commit_id = parts
        .next()
        .ok_or_else(|| "`jj log` returned no commit id".to_string())?
        .to_string();
    let change_id = parts
        .next()
        .ok_or_else(|| "`jj log` returned no change id".to_string())?
        .to_string();
    Ok(WorkingCopy {
        commit_id,
        change_id,
    })
}

/// Build a literal workspace-relative fileset for `rel`.
fn fileset_literal(rel: &str) -> String {
    let mut escaped = String::with_capacity(rel.len() + 2);
    for ch in rel.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            _ => escaped.push(ch),
        }
    }
    format!("root-file:\"{escaped}\"")
}

/// Whether `fileset` has unresolved conflicts in the working copy.
fn jj_path_conflicted(root: &Path, fileset: &str) -> Result<bool, String> {
    let out = run_jj(root, &["resolve", "--list", "--", fileset])?;
    if out.status.success() {
        return Ok(!String::from_utf8_lossy(&out.stdout).trim().is_empty());
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("No conflicts found") {
        return Ok(false);
    }
    Err(jj_err(&format!("jj resolve --list -- {fileset}"), &out))
}

/// Whether `fileset` is tracked by jj in the working copy.
fn jj_path_tracked(root: &Path, fileset: &str) -> Result<bool, String> {
    let out = run_jj(root, &["file", "list", "--", fileset])?;
    if !out.status.success() {
        return Err(jj_err(&format!("jj file list -- {fileset}"), &out));
    }
    Ok(!String::from_utf8_lossy(&out.stdout).trim().is_empty())
}

/// Path-scoped git-format diff between two snapshots.
fn jj_path_diff(root: &Path, from: &str, to: &str, fileset: &str) -> Result<String, String> {
    let out = run_jj(
        root,
        &["diff", "--git", "--from", from, "--to", to, "--", fileset],
    )?;
    if !out.status.success() {
        return Err(jj_err(
            &format!("jj diff --git --from {from} --to {to} -- {fileset}"),
            &out,
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
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
