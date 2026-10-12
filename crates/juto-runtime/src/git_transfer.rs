//! Git transfer: native Git-object snapshots, workspace forks, and safe export.
//!
//! The change graph (`change_graph`) owns the semantics of which native nodes
//! are marked and how unmarked nodes collapse. This module owns the Git side:
//! capturing immutable tree snapshots of a workspace, materializing an
//! independent fork directory from a snapshot, and turning an [`ExportPlan`] of
//! marked snapshots into real Git commits published with compare-and-swap
//! semantics.
//!
//! Live snapshots are trees and blobs only; no commit object is created until
//! [`GitRepository::export`] runs. Every Git invocation runs through
//! [`std::process::Command`] against an explicit `--git-dir` and `--work-tree`
//! with a sanitized environment, so the user's index, worktree, and `HEAD` are
//! never touched and no command ever checks out, stages, or force-pushes. The
//! only index this module writes is a private temporary one under
//! `<git-dir>/juto`. The methods are blocking, so callers run them off the GPUI
//! foreground thread (CLI or `spawn_blocking`).

use std::collections::{BTreeMap, HashSet};
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Environment variables that would silently redirect Git at a different
/// index, worktree, or object store. The repository is always addressed with an
/// explicit `--git-dir` and `--work-tree`, so these must not leak in from the
/// caller. Author and committer identity is removed as well so
/// [`GitRepository::signature`] reads configuration deterministically and
/// [`GitRepository::export`] sets its own.
const SANITIZED_ENV: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
    "GIT_CEILING_DIRECTORIES",
    "GIT_QUARANTINE_PATH",
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_DATE",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "GIT_COMMITTER_DATE",
];

/// Name of the native fork link marker written at the root of every fork
/// workspace. It records the shared Git directory and the fork's thread name so
/// [`GitRepository::open`] can rediscover the repository without a `.git`.
const WORKSPACE_MARKER: &str = ".juto-workspace.json";

/// Author/committer identity for a marked commit.
///
/// The timestamp is a Unix epoch second and is reused for both the author and
/// the committer so that re-exporting the same plan produces byte-identical
/// commit objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    pub name: String,
    pub email: String,
    pub timestamp: i64,
}

/// One marked point in the export plan: an immutable snapshot tree plus the
/// marked parents it descends from.
///
/// `parents` holds marked point IDs only; an empty list means the point
/// descends directly from [`ExportPlan::base_commit`]. Unmarked nodes between
/// marked points have already been collapsed by the change graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportPoint {
    pub id: String,
    pub tree: String,
    pub parents: Vec<String>,
    pub message: String,
    pub signature: Signature,
}

/// A complete, immutable export request.
///
/// `points` is in topological order: every parent referenced by a point must
/// appear earlier in the list (or be `base_commit`). `tip` names the marked
/// point that the feature branch should end up at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportPlan {
    pub base_commit: String,
    pub points: Vec<ExportPoint>,
    pub tip: String,
}

/// The outcome of a successful export.
///
/// `commits` maps every marked point ID to the Git commit object created for
/// it. `tip` is the Git commit the feature branch now points at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportResult {
    pub branch: String,
    pub tip: String,
    pub commits: BTreeMap<String, String>,
}

/// The outcome of a three-way tree merge.
///
/// `tree` is the merged tree, written to the object store even when there are
/// conflicts (the affected blobs then carry conflict markers). `conflicts`
/// lists the paths that conflicted and is empty for a clean merge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeOutcome {
    pub tree: String,
    pub conflicts: Vec<String>,
}

/// The on-disk contents of [`WORKSPACE_MARKER`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkspaceLink {
    git_dir: PathBuf,
    thread: String,
    /// The path prefixes a partial fork materialized. `None` means the fork
    /// materialized the whole tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sparse: Option<Vec<String>>,
}

/// The on-disk contents of a pending-fork registry entry.
///
/// One file per in-flight [`GitRepository::fork_workspace`] call lives under
/// `<git-dir>/juto/forks`. The file is held under an exclusive `flock` for the
/// whole materialization, so a crashed creator leaves an unlocked file that
/// [`GitRepository::cleanup_orphaned_forks`] can reclaim.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingFork {
    destination: PathBuf,
    thread: String,
}

/// A handle to the Git repository backing a workspace.
///
/// The repository is discovered with `git rev-parse` for ordinary worktrees, or
/// from a native fork link marker ([`WORKSPACE_MARKER`]) for independent fork
/// directories that share the same object store without a `.git` of their own.
#[derive(Debug, Clone)]
pub struct GitRepository {
    workspace: PathBuf,
    git_dir: PathBuf,
    thread: Option<String>,
    /// Path prefixes this workspace materialized, when it is a partial fork.
    sparse: Option<Vec<String>>,
}

impl GitRepository {
    /// Open the Git repository backing the workspace at `workspace`.
    ///
    /// A native fork link marker in `workspace` or one of its ancestors wins:
    /// the marker's parent is the workspace root and the marker records the
    /// shared Git directory and thread name. Otherwise the repository is
    /// discovered with `git rev-parse --show-toplevel --absolute-git-dir`.
    pub fn open(workspace: &Path) -> Result<Self> {
        if let Some((root, link)) = find_workspace_link(workspace)? {
            let workspace = root
                .canonicalize()
                .with_context(|| format!("workspace {} does not exist", root.display()))?;
            let git_dir = link.git_dir.canonicalize().with_context(|| {
                format!(
                    "workspace link {} points at a missing git directory {}",
                    workspace.join(WORKSPACE_MARKER).display(),
                    link.git_dir.display()
                )
            })?;
            return Ok(Self {
                workspace,
                git_dir,
                thread: Some(link.thread),
                sparse: link.sparse,
            });
        }
        let (workspace, git_dir) = discover(workspace)?;
        Ok(Self {
            workspace,
            git_dir,
            thread: None,
            sparse: None,
        })
    }

    /// The workspace root every command runs against.
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// The native fork thread name, when this workspace was created by
    /// [`GitRepository::fork_workspace`].
    pub fn thread(&self) -> Option<&str> {
        self.thread.as_deref()
    }

    /// The path prefixes this workspace materialized, when it is a partial fork.
    ///
    /// `None` means the workspace holds the whole tree. A partial fork is
    /// created by [`GitRepository::fork_workspace_sparse`]; capture preserves
    /// every baseline path outside these prefixes instead of reading its absence
    /// as a deletion.
    pub fn sparse(&self) -> Option<&[String]> {
        self.sparse.as_deref()
    }

    /// The resolved Git directory used for every plumbing command.
    pub fn git_dir(&self) -> &Path {
        &self.git_dir
    }

    /// The current `HEAD` commit, or `None` when the repository is unborn.
    pub fn head_commit(&self) -> Result<Option<String>> {
        match self.run_opt(&["rev-parse", "--verify", "--quiet", "HEAD^{commit}"])? {
            Some(id) if !id.is_empty() => Ok(Some(id)),
            _ => Ok(None),
        }
    }

    /// Resolve `object` to its tree object ID.
    ///
    /// Accepts a full commit or tree ID. An all-zero ID is the virtual root and
    /// resolves to the empty tree, which lets a brand-new repository export a
    /// parentless first commit.
    pub fn snapshot_tree(&self, object: &str) -> Result<String> {
        validate_object_id(object, "object")?;
        if object.bytes().all(|byte| byte == b'0') {
            return self.empty_tree();
        }
        let rev = format!("{object}^{{tree}}");
        self.run_trimmed(&["rev-parse", "--verify", "--end-of-options", rev.as_str()])
            .with_context(|| format!("cannot resolve the tree of {object}"))
    }

    /// Capture the workspace as an immutable tree on top of `baseline`.
    ///
    /// The capture uses a private temporary index under `<git-dir>/juto`, never
    /// the user's index: `baseline` is read into it, the worktree is enumerated
    /// and hashed from raw bytes, and the result is written with `git
    /// write-tree`. No clean filter, `text`/`autocrlf` normalization, or
    /// external filter is ever applied, so the snapshot is byte-exact. Normal
    /// Git ignores apply, but files already tracked by `baseline` are preserved
    /// even when they are now ignored. The workspace marker, the mutation lock,
    /// `.git`, `.jj`, and any atomic-write temp file left behind by a crash are
    /// always excluded. `baseline` may be an all-zero ID for an empty starting
    /// tree.
    pub fn capture_tree(&self, baseline: &str) -> Result<String> {
        let baseline_tree = if baseline.is_empty() {
            bail!("baseline tree is empty");
        } else if baseline.bytes().all(|byte| byte == b'0') {
            self.empty_tree()?
        } else {
            self.resolve_tree(baseline)?
        };
        let directory = self
            .git_dir
            .join("juto")
            .join(format!("capture-{}", Uuid::now_v7()));
        fs::create_dir_all(&directory).with_context(|| {
            format!(
                "failed to create the capture index directory {}",
                directory.display()
            )
        })?;
        let index = directory.join("index");
        let captured = self.capture_into(&baseline_tree, &index);
        let _ = fs::remove_dir_all(&directory);
        captured
    }

    /// Whether `path` exists in `tree`.
    ///
    /// `path` is the workspace-relative path as raw bytes (Git paths are byte
    /// strings, so a non-UTF-8 name is addressed exactly). It is matched
    /// literally, never as a glob.
    pub fn path_tracked(&self, tree: &str, path: &[u8]) -> Result<bool> {
        if path.is_empty() {
            bail!("path is empty");
        }
        let tree = self.resolve_tree(tree)?;
        let args: [&OsStr; 7] = [
            OsStr::new("--literal-pathspecs"),
            OsStr::new("ls-tree"),
            OsStr::new("-z"),
            OsStr::new("--full-tree"),
            OsStr::new(tree.as_str()),
            OsStr::new("--"),
            OsStr::from_bytes(path),
        ];
        let output = self
            .git_os(&args)
            .output()
            .context("failed to run `git ls-tree`")?;
        if !output.status.success() {
            bail!("`git ls-tree` failed: {}", describe_failure(&output));
        }
        Ok(!output.stdout.is_empty())
    }

    /// Whether `path` matches the repository's ignore rules.
    ///
    /// `path` is the workspace-relative path as raw bytes. The check uses
    /// `git check-ignore --no-index`, so it reports the ignore rules alone and
    /// never consults the index: a path that is already tracked still reports
    /// as ignored when a rule matches it. `check-ignore` rejects
    /// `--literal-pathspecs`, so the path is passed as an ordinary pathspec.
    /// The expected "not ignored" exit (1) is a plain `false`; any other
    /// failure is surfaced as an error.
    pub fn path_ignored(&self, path: &[u8]) -> Result<bool> {
        if path.is_empty() {
            bail!("path is empty");
        }
        let args: [&OsStr; 4] = [
            OsStr::new("check-ignore"),
            OsStr::new("-q"),
            OsStr::new("--no-index"),
            OsStr::from_bytes(path),
        ];
        Ok(self.run_opt_os(&args)?.is_some())
    }

    /// Render the patch that turns `before` into `after` for `path`.
    ///
    /// Both arguments are full commit or tree IDs; `path` is the
    /// workspace-relative path as raw bytes and is matched literally, so a file
    /// named `*.txt` is never attributed to another file. External diff drivers
    /// and textconv filters are disabled so the patch is deterministic.
    pub fn diff_trees(&self, before: &str, after: &str, path: &[u8]) -> Result<String> {
        let before = self.resolve_tree(before)?;
        let after = self.resolve_tree(after)?;
        let args: [&OsStr; 12] = [
            OsStr::new("--literal-pathspecs"),
            OsStr::new("diff-tree"),
            OsStr::new("-r"),
            OsStr::new("-p"),
            OsStr::new("--no-commit-id"),
            OsStr::new("--no-renames"),
            OsStr::new("--no-ext-diff"),
            OsStr::new("--no-textconv"),
            OsStr::new(before.as_str()),
            OsStr::new(after.as_str()),
            OsStr::new("--"),
            OsStr::from_bytes(path),
        ];
        self.run_os(&args)
    }

    /// Materialize `tree` into a brand-new independent workspace at
    /// `destination` and link it back to this repository.
    ///
    /// The tree is walked with `git ls-tree -r -z` and every blob is streamed
    /// out of a single `git cat-file --batch` child, preserving binary
    /// contents, executable bits, and symlinks. No checkout, index, or branch
    /// is created; the fork shares this repository's object store through the
    /// marker. `destination` must not exist: it is created atomically with
    /// `create_dir` (which fails if anything is already there), so a
    /// preexisting directory is never touched or removed. A gitlink (submodule)
    /// entry is skipped rather than faked: the workspace simply lacks the
    /// submodule directory, and the capture path preserves a baseline gitlink
    /// exactly, so the fork still captures a tree identical to the base.
    ///
    /// The fork is registered under `<git-dir>/juto/forks` before the
    /// destination is created and the registry entry is held under an exclusive
    /// lock for the whole materialization. A crash between `create_dir` and the
    /// marker write therefore leaves a reclaimable record instead of an
    /// orphaned directory that blocks every retry; see
    /// [`GitRepository::cleanup_orphaned_forks`].
    pub fn fork_workspace(&self, tree: &str, destination: &Path, thread: &str) -> Result<()> {
        self.fork_workspace_inner(tree, destination, thread, None)
    }

    /// Materialize only `paths` and their subtrees into a new partial fork.
    ///
    /// A partial fork is for a monorepo: an agent scoped to `crates/foo` pays
    /// for that subtree instead of the whole tree. Capture preserves every
    /// baseline path outside `paths` from the baseline, so the fork still
    /// captures a complete tree and the absence of the other paths never reads
    /// as a deletion. Each prefix is a repository-relative path with no leading
    /// or trailing `/`.
    pub fn fork_workspace_sparse(
        &self,
        tree: &str,
        destination: &Path,
        thread: &str,
        paths: &[String],
    ) -> Result<()> {
        ensure!(
            !paths.is_empty(),
            "a sparse fork needs at least one path prefix"
        );
        for path in paths {
            ensure!(!path.is_empty(), "sparse path prefix must not be empty");
            ensure!(
                !path.starts_with('/') && !path.ends_with('/'),
                "sparse path prefix must be a relative path without a leading or trailing '/': {path:?}"
            );
        }
        self.fork_workspace_inner(tree, destination, thread, Some(paths.to_vec()))
    }

    fn fork_workspace_inner(
        &self,
        tree: &str,
        destination: &Path,
        thread: &str,
        sparse: Option<Vec<String>>,
    ) -> Result<()> {
        if thread.is_empty() {
            bail!("fork thread name is empty");
        }
        let tree = self.resolve_tree(tree)?;
        let listing = self
            .git(&["ls-tree", "-r", "-z", tree.as_str()])
            .output()
            .context("failed to run `git ls-tree`")?;
        if !listing.status.success() {
            bail!("`git ls-tree` failed: {}", describe_failure(&listing));
        }
        if let Some(parent) = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let forks = self.forks_directory();
        fs::create_dir_all(&forks)
            .with_context(|| format!("failed to create {}", forks.display()))?;
        let pending_path = forks.join(pending_fork_name(destination));
        // Whether a crashed creator already registered this destination. Only a
        // pre-existing record may be reclaimed; an unregistered directory is
        // never touched.
        let pending_existed = pending_path.exists();
        let pending = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&pending_path)
            .with_context(|| format!("failed to open {}", pending_path.display()))?;
        // The exclusive lock is the liveness signal: a free lock means any
        // leftover directory belongs to a dead creator and may be reclaimed; a
        // held lock means another fork is still materializing here.
        if pending.try_lock_exclusive().is_err() {
            bail!(
                "fork destination {} is already being created",
                destination.display()
            );
        }
        write_pending(
            &pending,
            &PendingFork {
                destination: absolute_destination(destination),
                thread: thread.to_string(),
            },
        )?;

        // Atomic: fails with `AlreadyExists` instead of racing an `exists` check.
        match fs::create_dir(destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                // We hold the lock, so no live fork owns this path. A leftover
                // directory is reclaimed only when it is a genuine orphan: it
                // was registered by a crashed creator and carries no success
                // marker. Anything else is refused untouched.
                if pending_existed
                    && destination.is_dir()
                    && !destination.join(WORKSPACE_MARKER).exists()
                {
                    fs::remove_dir_all(destination).with_context(|| {
                        format!(
                            "failed to remove the orphaned fork {}",
                            destination.display()
                        )
                    })?;
                    fs::create_dir(destination).with_context(|| {
                        format!(
                            "failed to create the fork directory {} (it must not already exist)",
                            destination.display()
                        )
                    })?;
                } else {
                    let _ = fs::remove_file(&pending_path);
                    bail!(
                        "failed to create the fork directory {} (it must not already exist)",
                        destination.display()
                    );
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&pending_path);
                return Err(error).with_context(|| {
                    format!(
                        "failed to create the fork directory {} (it must not already exist)",
                        destination.display()
                    )
                });
            }
        }

        let materialized =
            self.materialize_tree(&listing.stdout, destination, thread, sparse.as_deref());
        match materialized {
            Ok(()) => {
                // The marker is the durable record; the pending file is done.
                let _ = fs::remove_file(&pending_path);
                Ok(())
            }
            Err(error) => {
                // Only ever removes the directory this call just created.
                let _ = fs::remove_dir_all(destination);
                let _ = fs::remove_file(&pending_path);
                Err(error)
            }
        }
    }

    /// The pending-fork registry directory, alongside the change graph's state.
    fn forks_directory(&self) -> PathBuf {
        self.git_dir.join("juto").join("forks")
    }

    /// Write every `git ls-tree -r -z` entry into `destination` and add the
    /// workspace marker.
    ///
    /// When `sparse` is set, only entries at or under one of its prefixes are
    /// written; the rest are left absent and preserved from the baseline by
    /// capture.
    fn materialize_tree(
        &self,
        listing: &[u8],
        destination: &Path,
        thread: &str,
        sparse: Option<&[String]>,
    ) -> Result<()> {
        let mut batch = BatchCatFile::spawn(self)?;
        for entry in listing.split(|byte| *byte == 0) {
            if entry.is_empty() {
                continue;
            }
            if let Some(sparse) = sparse {
                let path = entry_path(entry)?;
                if !sparse
                    .iter()
                    .any(|prefix| path_under(path, prefix.as_bytes()))
                {
                    continue;
                }
            }
            self.materialize_entry(entry, destination, &mut batch)?;
        }
        drop(batch);
        let link = WorkspaceLink {
            git_dir: self.git_dir.clone(),
            thread: thread.to_string(),
            sparse: sparse.map(<[String]>::to_vec),
        };
        let marker =
            serde_json::to_vec_pretty(&link).context("failed to encode the workspace link")?;
        fs::write(destination.join(WORKSPACE_MARKER), marker).with_context(|| {
            format!(
                "failed to write the workspace marker in {}",
                destination.display()
            )
        })?;
        Ok(())
    }

    /// Read the configured Git identity and stamp it with the current Unix time.
    pub fn signature(&self) -> Result<Signature> {
        let ident = self.run_trimmed(&["var", "GIT_AUTHOR_IDENT"])?;
        let (name, email) = parse_ident(&ident)?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_secs() as i64;
        Ok(Signature {
            name,
            email,
            timestamp,
        })
    }

    /// Pin `object` under `refs/juto/snapshots/<id>` so Git garbage collection
    /// cannot prune the snapshot before it is exported.
    ///
    /// `object` is a full commit or tree ID and is pinned exactly as given, so
    /// the graph can retain a baseline commit and its tree under separate ids.
    pub fn retain_snapshot(&self, id: &str, object: &str) -> Result<()> {
        if id.is_empty() {
            bail!("snapshot id is empty");
        }
        if id.contains('/') {
            bail!("snapshot id must be a single ref component: {id:?}");
        }
        let refname = format!("refs/juto/snapshots/{id}");
        self.validate_ref(&refname)?;
        let full = self.resolve_object(object)?;
        self.run_trimmed(&["update-ref", refname.as_str(), full.as_str()])?;
        Ok(())
    }

    /// Release a snapshot pinned by [`GitRepository::retain_snapshot`].
    ///
    /// Deletes `refs/juto/snapshots/<id>` so Git garbage collection may prune
    /// the object once nothing else references it. The id is validated exactly
    /// like [`GitRepository::retain_snapshot`]. Returns `true` when the ref
    /// existed and was removed, `false` when it was already absent.
    pub fn release_snapshot(&self, id: &str) -> Result<bool> {
        if id.is_empty() {
            bail!("snapshot id is empty");
        }
        if id.contains('/') {
            bail!("snapshot id must be a single ref component: {id:?}");
        }
        self.release_ref(&format!("refs/juto/snapshots/{id}"))
    }

    /// Delete `refname` if it exists, reporting whether it was removed.
    ///
    /// The name is validated with `git check-ref-format`, so only a well-formed
    /// ref can be deleted. Returns `true` when the ref existed and was removed,
    /// `false` when it was already absent. This is the generic primitive behind
    /// [`GitRepository::release_snapshot`] and the release of the
    /// `refs/juto/nodes/<id>` refs a pruned change-graph node leaves behind.
    pub fn release_ref(&self, refname: &str) -> Result<bool> {
        self.validate_ref(refname)?;
        if self
            .run_opt(&["rev-parse", "--verify", "--quiet", refname])?
            .is_none()
        {
            return Ok(false);
        }
        self.run_trimmed(&["update-ref", "-d", refname])?;
        Ok(true)
    }

    /// Pack every ref into `.git/packed-refs`, removing the loose files.
    ///
    /// The change graph pins two refs per node (`refs/juto/nodes/<id>` and
    /// `refs/juto/snapshots/<id>`), so a long-lived repository accumulates
    /// thousands of loose ref files that every `for-each-ref` — and therefore
    /// every publish and import — has to walk. Packing is transparent: refs
    /// resolve identically, and `update-ref`/`update-ref -d` keep working. It is
    /// what `git gc` does anyway; this lets the graph's own maintenance trigger
    /// it without a separate `git gc`.
    pub fn pack_refs(&self) -> Result<()> {
        self.run_trimmed(&["pack-refs", "--all"])?;
        Ok(())
    }

    /// Write `bytes` as a blob and return its object id.
    ///
    /// The blob is written with `git hash-object --no-filters`, so no clean
    /// filter, `text`/`autocrlf` normalization, or external filter can alter
    /// the bytes: the id is the hash of exactly `bytes`.
    pub fn write_blob(&self, bytes: &[u8]) -> Result<String> {
        self.hash_bytes(bytes)
    }

    /// Read the raw bytes of the blob `oid`.
    ///
    /// The object must be a blob; a commit or tree id is rejected by Git. The
    /// bytes are returned exactly as stored, with no filter applied.
    pub fn read_blob(&self, oid: &str) -> Result<Vec<u8>> {
        validate_object_id(oid, "blob")?;
        let output = self
            .git(&["cat-file", "blob", oid])
            .output()
            .context("failed to run `git cat-file blob`")?;
        if !output.status.success() {
            bail!(
                "`git cat-file blob {oid}` failed: {}",
                describe_failure(&output)
            );
        }
        Ok(output.stdout)
    }

    /// Create or update `refname` to point at `object` with compare-and-swap
    /// semantics.
    ///
    /// `expected` is `None` to require that the ref does not exist yet, or
    /// `Some(old)` to require that it currently points at `old`. The update is
    /// atomic: `git update-ref` is handed the expected old value, so a
    /// concurrent writer loses the race instead of clobbering the ref. Returns
    /// `true` when the ref changed and `false` when it already pointed at
    /// `object`. `object` may be any object type (blob, tree, or commit) and
    /// must already exist in the object store.
    pub fn update_ref(&self, refname: &str, object: &str, expected: Option<&str>) -> Result<bool> {
        self.validate_ref(refname)?;
        validate_object_id(object, "object")?;
        let current = self.run_opt(&[
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            refname,
        ])?;
        match expected {
            None => {
                if let Some(current) = &current {
                    bail!("ref {refname:?} already exists at {current}");
                }
            }
            Some(old) => {
                validate_object_id(old, "expected object")?;
                match &current {
                    Some(current) if current == old => {}
                    Some(current) => bail!("ref {refname:?} is at {current}, expected {old}"),
                    None => bail!("ref {refname:?} does not exist, expected {old}"),
                }
            }
        }
        if current.as_deref() == Some(object) {
            return Ok(false);
        }
        let zero = "0".repeat(object.len());
        let old = expected.unwrap_or(zero.as_str());
        self.run_trimmed(&["update-ref", refname, object, old])?;
        Ok(true)
    }

    /// List every ref under `prefix` as `(refname, object id)` pairs.
    ///
    /// The list is sorted by refname (Git's default) and includes refs of any
    /// object type. An empty `prefix` lists every ref in the repository.
    pub fn list_refs(&self, prefix: &str) -> Result<Vec<(String, String)>> {
        if prefix.starts_with('-') {
            bail!("ref prefix must not start with '-': {prefix:?}");
        }
        let mut args: Vec<&str> = vec!["for-each-ref", "--format=%(refname) %(objectname)"];
        if !prefix.is_empty() {
            args.push(prefix);
        }
        let output = self.run(&args)?;
        let mut refs = Vec::new();
        for line in output.lines() {
            if line.is_empty() {
                continue;
            }
            let (refname, object) = line
                .split_once(' ')
                .with_context(|| format!("malformed `git for-each-ref` line: {line:?}"))?;
            refs.push((refname.to_string(), object.to_string()));
        }
        Ok(refs)
    }

    /// Delete `refname` if it exists.
    ///
    /// Intended for staging cleanup: deleting an absent ref is a no-op, not an
    /// error.
    pub fn delete_ref(&self, refname: &str) -> Result<()> {
        self.validate_ref(refname)?;
        self.run_trimmed(&["update-ref", "-d", refname])?;
        Ok(())
    }

    /// Fetch `refspecs` from `remote` atomically.
    ///
    /// `--atomic` makes the whole fetch all-or-nothing and `--no-tags` keeps
    /// tag auto-following from widening the transfer. Every refspec is passed
    /// as an explicit argument (never an option). A remote that cannot do an
    /// atomic fetch fails the command rather than silently degrading.
    pub fn fetch(&self, remote: &str, refspecs: &[String]) -> Result<()> {
        if remote.is_empty() {
            bail!("remote is empty");
        }
        if remote.starts_with('-') {
            bail!("remote must not start with '-': {remote:?}");
        }
        if refspecs.is_empty() {
            bail!("no refspecs to fetch");
        }
        let mut args: Vec<&str> = vec!["fetch", "--atomic", "--no-tags", remote];
        for refspec in refspecs {
            if refspec.starts_with('-') {
                bail!("refspec must not start with '-': {refspec:?}");
            }
            args.push(refspec);
        }
        let output = self
            .git(&args)
            .output()
            .context("failed to run `git fetch`")?;
        if !output.status.success() {
            bail!("`git fetch {remote}` failed: {}", describe_failure(&output));
        }
        Ok(())
    }

    /// Push `refspecs` to `remote` atomically, never forcing.
    ///
    /// `--no-force` refuses any non-fast-forward update, `--atomic` makes the
    /// whole push all-or-nothing, and `--no-follow-tags` keeps tag
    /// auto-following from widening the transfer. Every refspec is passed as an
    /// explicit argument (never an option). A remote that cannot do an atomic
    /// push fails the command rather than silently degrading.
    pub fn push_refs(&self, remote: &str, refspecs: &[String]) -> Result<()> {
        if remote.is_empty() {
            bail!("remote is empty");
        }
        if remote.starts_with('-') {
            bail!("remote must not start with '-': {remote:?}");
        }
        if refspecs.is_empty() {
            bail!("no refspecs to push");
        }
        let mut args: Vec<&str> =
            vec!["push", "--no-force", "--atomic", "--no-follow-tags", remote];
        for refspec in refspecs {
            if refspec.starts_with('-') {
                bail!("refspec must not start with '-': {refspec:?}");
            }
            args.push(refspec);
        }
        let output = self
            .git(&args)
            .output()
            .context("failed to run `git push`")?;
        if !output.status.success() {
            bail!("`git push {remote}` failed: {}", describe_failure(&output));
        }
        Ok(())
    }

    /// Three-way merge `ours` and `theirs` with an explicit `base` tree,
    /// writing the merged tree.
    ///
    /// All three arguments are resolved to trees. The merge runs `git merge-tree
    /// --write-tree --merge-base=<base> -z --name-only --no-messages`, which
    /// needs Git >= 2.45 to accept tree operands; see
    /// [`GitRepository::supports_tree_merge`]. A clean merge returns the merged
    /// tree with no conflicts. A conflicting merge still returns the merged tree
    /// (with conflict markers in the affected blobs) plus the conflicting paths.
    /// Any other exit status is an error.
    pub fn merge_trees(&self, base: &str, ours: &str, theirs: &str) -> Result<MergeOutcome> {
        let base = self.resolve_tree(base)?;
        let ours = self.resolve_tree(ours)?;
        let theirs = self.resolve_tree(theirs)?;
        let merge_base = format!("--merge-base={base}");
        let output = self
            .git(&[
                "merge-tree",
                "--write-tree",
                merge_base.as_str(),
                "-z",
                "--name-only",
                "--no-messages",
                ours.as_str(),
                theirs.as_str(),
            ])
            .output()
            .context("failed to run `git merge-tree`")?;
        let code = output.status.code();
        if code != Some(0) && code != Some(1) {
            bail!("`git merge-tree` failed: {}", describe_failure(&output));
        }
        // With `-z` the output is `<tree-oid>\0` for a clean merge and
        // `<tree-oid>\0<path>\0...` for a conflicting one.
        let mut fields = output.stdout.split(|byte| *byte == 0);
        let tree = fields
            .next()
            .filter(|field| !field.is_empty())
            .context("`git merge-tree` produced no tree")?;
        let tree = String::from_utf8(tree.to_vec())
            .context("`git merge-tree` produced a non-UTF-8 tree id")?;
        let mut conflicts = Vec::new();
        for field in fields {
            if field.is_empty() {
                continue;
            }
            conflicts.push(String::from_utf8_lossy(field).into_owned());
        }
        Ok(MergeOutcome { tree, conflicts })
    }

    /// Apply the delta from the `current` tree to the `target` tree inside the
    /// existing workspace directory `destination`.
    ///
    /// The delta is computed with `git diff-tree -r -z --no-renames --raw`.
    /// Added, modified, and type-changed paths are materialized from the target
    /// tree (regular files, executable bits, and symlinks are preserved; a
    /// gitlink is refused). Deleted paths are removed and their now-empty parent
    /// directories are pruned up to `destination`. Paths outside the delta are
    /// never touched.
    pub fn restore_tree(&self, current: &str, target: &str, destination: &Path) -> Result<()> {
        let current = self.resolve_tree(current)?;
        let target = self.resolve_tree(target)?;
        if !destination.is_dir() {
            bail!(
                "restore destination {} is not a directory",
                destination.display()
            );
        }
        let output = self
            .git(&[
                "diff-tree",
                "-r",
                "-z",
                "--no-renames",
                "--raw",
                current.as_str(),
                target.as_str(),
            ])
            .output()
            .context("failed to run `git diff-tree`")?;
        if !output.status.success() {
            bail!("`git diff-tree` failed: {}", describe_failure(&output));
        }
        let mut batch = BatchCatFile::spawn(self)?;
        let mut fields = output.stdout.split(|byte| *byte == 0);
        while let Some(meta) = fields.next() {
            if meta.is_empty() {
                continue;
            }
            let path = fields
                .next()
                .context("malformed `git diff-tree` record: missing path")?;
            let change = parse_raw_change(meta)?;
            // A partial fork holds only its sparse prefixes. A delta path
            // outside them is left alone, matching capture, which preserves
            // those paths from the baseline: writing one here would diverge the
            // workspace from the tree the next capture records.
            if let Some(sparse) = &self.sparse
                && !sparse
                    .iter()
                    .any(|prefix| path_under(path, prefix.as_bytes()))
            {
                continue;
            }
            match change.status {
                b'A' | b'M' | b'T' => {
                    let relative = validate_tree_path(path)?;
                    let existing = destination.join(&relative);
                    if let Ok(metadata) = fs::symlink_metadata(&existing) {
                        if metadata.is_dir() {
                            bail!(
                                "cannot restore {}: a directory is in the way",
                                relative.display()
                            );
                        }
                        // Replace any existing entry so a type change (file
                        // <-> symlink) never writes through a symlink or fails
                        // on an existing one.
                        fs::remove_file(&existing)
                            .with_context(|| format!("failed to replace {}", existing.display()))?;
                    }
                    let entry = raw_entry(&change, path);
                    self.materialize_entry(&entry, destination, &mut batch)?;
                }
                b'D' => {
                    let relative = validate_tree_path(path)?;
                    remove_path(destination, &relative)?;
                }
                other => bail!(
                    "unsupported `git diff-tree` status {:?} for {}",
                    other as char,
                    String::from_utf8_lossy(path)
                ),
            }
        }
        Ok(())
    }

    /// Whether this Git supports `merge-tree --write-tree --merge-base`.
    ///
    /// The probe runs `git merge-tree -h` and looks for `--write-tree` and
    /// `merge-base` in the usage text (Git prints usage to stdout for `-h`; the
    /// option is rendered as `--[no-]merge-base`, so the bare `merge-base`
    /// substring is matched). If the usage text is unavailable, it falls back to
    /// parsing `git version` and requiring Git >= 2.45, the release that added
    /// `--merge-base` to `merge-tree`.
    pub fn supports_tree_merge(&self) -> Result<bool> {
        let output = self
            .git(&["merge-tree", "-h"])
            .output()
            .context("failed to run `git merge-tree -h`")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let usage = if stdout.trim().is_empty() {
            stderr
        } else {
            stdout
        };
        if !usage.trim().is_empty() {
            return Ok(usage.contains("--write-tree") && usage.contains("merge-base"));
        }
        let version = self.run_trimmed(&["version"])?;
        Ok(version_at_least(&version, 2, 45))
    }

    /// Reclaim fork directories left behind by a crashed
    /// [`GitRepository::fork_workspace`].
    ///
    /// Every in-flight fork holds an exclusive lock on its pending file under
    /// `<git-dir>/juto/forks`. A pending file whose lock is free belongs to a
    /// dead creator: its destination directory is removed (unless the success
    /// marker is present, meaning the fork completed and only the pending
    /// delete was lost) and the pending file is deleted. Returns the reclaimed
    /// destination paths. A missing registry directory is not an error.
    pub fn cleanup_orphaned_forks(&self) -> Result<Vec<PathBuf>> {
        let forks = self.forks_directory();
        let entries = match fs::read_dir(&forks) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", forks.display()));
            }
        };
        let mut reclaimed = Vec::new();
        for entry in entries {
            let path = entry
                .with_context(|| format!("failed to read {}", forks.display()))?
                .path();
            if !path.is_file() {
                continue;
            }
            let file = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(file) => file,
                Err(_) => continue,
            };
            // A held lock means the creator is still running: leave it alone.
            if file.try_lock_exclusive().is_err() {
                continue;
            }
            let raw = match fs::read(&path) {
                Ok(raw) => raw,
                Err(_) => continue,
            };
            let Ok(record) = serde_json::from_slice::<PendingFork>(&raw) else {
                continue;
            };
            if !record.destination.join(WORKSPACE_MARKER).exists() && record.destination.exists() {
                fs::remove_dir_all(&record.destination).with_context(|| {
                    format!(
                        "failed to remove the orphaned fork {}",
                        record.destination.display()
                    )
                })?;
                reclaimed.push(record.destination);
            }
            let _ = fs::remove_file(&path);
        }
        Ok(reclaimed)
    }

    /// Materialize `plan` as Git commits and move `branch` to the plan tip.
    ///
    /// The plan is validated before any ref moves: the base must resolve to a
    /// commit, point IDs must be unique, trees must resolve to trees, and every
    /// parent must name an earlier point (topological order). The branch update
    /// is a compare-and-swap that refuses non-fast-forward moves; an existing
    /// branch already at the target tip is a no-op.
    pub fn export(&self, plan: &ExportPlan, branch: &str) -> Result<ExportResult> {
        self.validate_branch(branch)?;
        if self
            .checked_out_branches()?
            .iter()
            .any(|checked| checked == branch)
        {
            bail!("refusing to update branch {branch:?}: it is checked out in a worktree");
        }

        validate_object_id(&plan.base_commit, "base commit")?;
        let base = if plan.base_commit.bytes().all(|byte| byte == b'0') {
            None
        } else {
            Some(self.resolve_commit(&plan.base_commit)?)
        };
        let mut commits: BTreeMap<String, String> = BTreeMap::new();
        let mut seen: HashSet<&str> = HashSet::new();

        for point in &plan.points {
            if point.id.is_empty() {
                bail!("export point has an empty id");
            }
            if !seen.insert(point.id.as_str()) {
                bail!("duplicate export point id {:?}", point.id);
            }
            if point.signature.name.is_empty() || point.signature.email.is_empty() {
                bail!(
                    "export point {:?} has an empty signature name or email",
                    point.id
                );
            }

            let tree = self.resolve_tree(&point.tree)?;
            let mut parents = Vec::with_capacity(point.parents.len().max(1));
            if point.parents.is_empty() {
                parents.extend(base.as_deref());
            } else {
                for parent in &point.parents {
                    match commits.get(parent) {
                        Some(commit) => parents.push(commit.as_str()),
                        None => bail!(
                            "export point {:?} references unknown or non-topological parent {:?}",
                            point.id,
                            parent
                        ),
                    }
                }
            }

            let commit = self.create_commit(&tree, &parents, &point.message, &point.signature)?;
            commits.insert(point.id.clone(), commit);
        }

        let tip = if plan.points.is_empty() {
            let tip = self.resolve_commit(&plan.tip)?;
            if Some(tip.as_str()) != base.as_deref() {
                bail!("empty export plan tip does not match a real base commit");
            }
            tip
        } else {
            match commits.get(&plan.tip) {
                Some(commit) => commit.clone(),
                None => bail!("export plan tip {:?} is not a marked point id", plan.tip),
            }
        };

        self.update_branch(branch, &tip)?;
        Ok(ExportResult {
            branch: branch.to_string(),
            tip,
            commits,
        })
    }

    /// Push `refs/heads/<branch>` to the same ref on `remote`, never forcing.
    ///
    /// The remote is passed as an explicit argument (never an option) and the
    /// refspec is fully qualified, so no `push.default` or `remote.*.push`
    /// configuration can widen the update. Relative remotes resolve against the
    /// workspace root, not the caller's process directory.
    pub fn push(&self, branch: &str, remote: &str) -> Result<()> {
        self.validate_branch(branch)?;
        if remote.is_empty() {
            bail!("remote is empty");
        }
        if remote.starts_with('-') {
            bail!("remote must not start with '-': {remote:?}");
        }
        let refspec = format!("refs/heads/{branch}:refs/heads/{branch}");
        let output = self
            .git(&[
                "push",
                "--no-force",
                "--no-follow-tags",
                remote,
                refspec.as_str(),
            ])
            .output()
            .context("failed to run `git push`")?;
        if !output.status.success() {
            bail!(
                "`git push {remote} {refspec}` failed: {}",
                describe_failure(&output)
            );
        }
        Ok(())
    }

    /// Stage the workspace into `index` on top of `baseline_tree` and write it.
    ///
    /// The snapshot is built from raw bytes: every path is enumerated from the
    /// private index (`--cached`) plus the worktree (`--others
    /// --exclude-standard`), regular files are hashed with `hash-object
    /// --no-filters` (so `.gitattributes` clean filters, `text`/`autocrlf`
    /// normalization, and external filters never touch the snapshot), symlinks
    /// are hashed from their link bytes, executable bits are preserved, and
    /// baseline gitlinks are left untouched. Missing paths are removed from the
    /// index (only a path the baseline tracks needs a removal record). The
    /// result is written with `write-tree`; no commit is created.
    fn capture_into(&self, baseline_tree: &str, index: &Path) -> Result<String> {
        self.run_with_index(index, &["read-tree", baseline_tree])?;
        let baseline = self.index_entries(index)?;
        let listing = self.run_with_index_bytes(
            index,
            &[
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--deduplicate",
                "-z",
            ],
        )?;

        // Classify every enumerated path before touching the index.
        let mut regular: Vec<(Vec<u8>, bool)> = Vec::new();
        let mut symlinks: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut removals: Vec<Vec<u8>> = Vec::new();
        for path in listing.split(|byte| *byte == 0) {
            if path.is_empty() || is_metadata_path(path) {
                continue;
            }
            if baseline.get(path).map(|(mode, _)| mode.as_str()) == Some("160000") {
                // A baseline gitlink is preserved exactly; submodules are never
                // materialized or faked.
                continue;
            }
            // A partial fork materialized only its sparse prefixes. A baseline
            // path outside them is simply absent from the worktree, which must
            // not read as a deletion: preserve it from the baseline.
            if let Some(sparse) = &self.sparse
                && baseline.contains_key(path)
                && !sparse
                    .iter()
                    .any(|prefix| path_under(path, prefix.as_bytes()))
            {
                continue;
            }
            let target = self.workspace.join(OsStr::from_bytes(path));
            let metadata = match fs::symlink_metadata(&target) {
                Ok(metadata) => metadata,
                // The path is gone from the worktree: it vanished between the
                // enumeration and this stat, or its parent is now a file
                // (ENOTDIR). Either way it is a removal candidate.
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                    ) =>
                {
                    removals.push(path.to_vec());
                    continue;
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("cannot stat {}", target.display()));
                }
            };
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                let link = fs::read_link(&target)
                    .with_context(|| format!("cannot read symlink {}", target.display()))?;
                symlinks.push((path.to_vec(), link.as_os_str().as_bytes().to_vec()));
            } else if file_type.is_file() {
                let executable = metadata.permissions().mode() & 0o111 != 0;
                regular.push((path.to_vec(), executable));
            } else {
                // A directory or special file cannot be represented as a blob.
                removals.push(path.to_vec());
            }
        }

        let regular_paths: Vec<Vec<u8>> = regular.iter().map(|(path, _)| path.clone()).collect();
        let regular_hashes = self.hash_files(&regular_paths)?;
        let mut symlink_hashes = Vec::with_capacity(symlinks.len());
        for (_, target) in &symlinks {
            symlink_hashes.push(self.hash_bytes(target)?);
        }

        let mut info: Vec<u8> = Vec::new();
        for ((path, executable), oid) in regular.iter().zip(&regular_hashes) {
            let mode = if *executable { "100755" } else { "100644" };
            push_index_entry(&mut info, mode, oid, path);
        }
        for ((path, _), oid) in symlinks.iter().zip(&symlink_hashes) {
            push_index_entry(&mut info, "120000", oid, path);
        }
        for path in &removals {
            // Only a path the baseline tracks needs a removal record. A path
            // that is not in the index (an untracked directory such as a nested
            // repository, or a file that vanished mid-capture) has no object ID
            // to remove and would make `git update-index` reject the whole
            // stream as malformed index info.
            let Some((_, oid)) = baseline.get(path.as_slice()) else {
                continue;
            };
            info.extend_from_slice(b"0 ");
            info.extend_from_slice(oid.as_bytes());
            info.push(b'\t');
            info.extend_from_slice(path);
            info.push(0);
        }

        self.update_index_info(index, &info)?;
        self.run_with_index(index, &["write-tree"])
    }

    /// Read `git ls-files --stage -z` from `index` into `path -> (mode, oid)`.
    fn index_entries(&self, index: &Path) -> Result<BTreeMap<Vec<u8>, (String, String)>> {
        let raw = self.run_with_index_bytes(index, &["ls-files", "--stage", "-z"])?;
        let mut entries = BTreeMap::new();
        for record in raw.split(|byte| *byte == 0) {
            if record.is_empty() {
                continue;
            }
            let tab = record
                .iter()
                .position(|byte| *byte == b'\t')
                .context("malformed `git ls-files` entry: missing tab")?;
            let (meta, path) = record.split_at(tab);
            let path = &path[1..];
            let meta = std::str::from_utf8(meta).context("`git ls-files` metadata is not UTF-8")?;
            let mut fields = meta.split_whitespace();
            let mode = fields
                .next()
                .context("`git ls-files` entry is missing a mode")?;
            let oid = fields
                .next()
                .context("`git ls-files` entry is missing an object id")?;
            let _stage = fields
                .next()
                .context("`git ls-files` entry is missing a stage")?;
            entries.insert(path.to_vec(), (mode.to_string(), oid.to_string()));
        }
        Ok(entries)
    }

    /// Hash regular files by path in batches, without any filters.
    fn hash_files(&self, paths: &[Vec<u8>]) -> Result<Vec<String>> {
        let mut hashes = Vec::with_capacity(paths.len());
        for chunk in paths.chunks(256) {
            let mut args: Vec<&OsStr> = Vec::with_capacity(chunk.len() + 4);
            args.push(OsStr::new("hash-object"));
            args.push(OsStr::new("--no-filters"));
            args.push(OsStr::new("-w"));
            args.push(OsStr::new("--"));
            for path in chunk {
                args.push(OsStr::from_bytes(path));
            }
            let output = self
                .git_os(&args)
                .output()
                .context("failed to run `git hash-object`")?;
            if !output.status.success() {
                bail!("`git hash-object` failed: {}", describe_failure(&output));
            }
            let text = String::from_utf8(output.stdout)
                .context("`git hash-object` produced non-UTF-8 output")?;
            for line in text.lines() {
                if !line.is_empty() {
                    hashes.push(line.trim().to_string());
                }
            }
        }
        Ok(hashes)
    }

    /// Hash raw bytes as a blob, without any filters.
    fn hash_bytes(&self, bytes: &[u8]) -> Result<String> {
        self.run_with_stdin(&["hash-object", "--no-filters", "-w", "--stdin"], bytes)
    }

    /// Apply `git update-index -z --index-info` records to the private index.
    fn update_index_info(&self, index: &Path, info: &[u8]) -> Result<()> {
        let mut command = self.git(&["update-index", "-z", "--index-info"]);
        command
            .env("GIT_INDEX_FILE", index)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .context("failed to spawn `git update-index`")?;
        child
            .stdin
            .take()
            .context("`git update-index` stdin was not piped")?
            .write_all(info)
            .context("failed to write to `git update-index` stdin")?;
        let output = child
            .wait_with_output()
            .context("failed to wait for `git update-index`")?;
        if !output.status.success() {
            bail!(
                "`git update-index --index-info` failed: {}",
                describe_failure(&output)
            );
        }
        Ok(())
    }

    /// Write one `git ls-tree -r -z` entry into `destination`, streaming blob
    /// content out of the shared `batch` child.
    fn materialize_entry(
        &self,
        entry: &[u8],
        destination: &Path,
        batch: &mut BatchCatFile,
    ) -> Result<()> {
        let tab = entry
            .iter()
            .position(|byte| *byte == b'\t')
            .context("malformed `git ls-tree` entry: missing tab")?;
        let (meta, path) = entry.split_at(tab);
        let path = &path[1..];
        let meta = std::str::from_utf8(meta).context("`git ls-tree` metadata is not UTF-8")?;
        let mut fields = meta.split_whitespace();
        let mode = fields
            .next()
            .context("`git ls-tree` entry is missing a mode")?;
        let kind = fields
            .next()
            .context("`git ls-tree` entry is missing a type")?;
        let oid = fields
            .next()
            .context("`git ls-tree` entry is missing an object id")?;
        if fields.next().is_some() {
            bail!("unexpected `git ls-tree` metadata: {meta:?}");
        }

        let relative = validate_tree_path(path)?;
        let target = destination.join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        match mode {
            "100644" | "100755" => {
                if kind != "blob" {
                    bail!(
                        "tree entry {} has mode {mode} but type {kind}",
                        relative.display()
                    );
                }
                // Stream straight into the file: a large blob is never buffered
                // whole in memory.
                let file = fs::File::create(&target)
                    .with_context(|| format!("failed to create {}", target.display()))?;
                let mut writer = std::io::BufWriter::new(file);
                batch.stream_blob(oid, &mut writer)?;
                writer
                    .flush()
                    .with_context(|| format!("failed to write {}", target.display()))?;
                drop(writer);
                let permissions = if mode == "100755" { 0o755 } else { 0o644 };
                fs::set_permissions(&target, fs::Permissions::from_mode(permissions))
                    .with_context(|| format!("failed to chmod {}", target.display()))?;
            }
            "120000" => {
                if kind != "blob" {
                    bail!(
                        "tree entry {} has mode {mode} but type {kind}",
                        relative.display()
                    );
                }
                // Symlink targets are small, so reading the blob is fine.
                let bytes = batch.read_blob(oid)?;
                std::os::unix::fs::symlink(OsStr::from_bytes(&bytes), &target)
                    .with_context(|| format!("failed to create symlink {}", target.display()))?;
            }
            // A submodule is a gitlink: a commit id with no tree of its own in
            // this repository. It is deliberately not materialized — the
            // workspace simply lacks the submodule directory — because the
            // capture path preserves a baseline gitlink exactly (it is skipped
            // before the worktree stat), so a fork that omits it still captures
            // a tree identical to the base. Materializing an empty directory
            // instead would let a write inside it add an untracked path that
            // conflicts with the gitlink in the index.
            "160000" => {}
            other => bail!(
                "unsupported tree entry mode {other} at {}",
                relative.display()
            ),
        }
        Ok(())
    }

    /// Build a Git command addressed at this repository with a sanitized
    /// environment and the workspace root as the working directory.
    fn base_command(&self) -> Command {
        let mut command = Command::new("git");
        command.arg("--git-dir").arg(&self.git_dir);
        command.arg("--work-tree").arg(&self.workspace);
        command.current_dir(&self.workspace);
        sanitize(&mut command);
        command
    }

    fn git(&self, args: &[&str]) -> Command {
        let mut command = self.base_command();
        command.args(args);
        command
    }

    /// Like [`GitRepository::git`], but for arguments that are raw path bytes.
    fn git_os(&self, args: &[&OsStr]) -> Command {
        let mut command = self.base_command();
        command.args(args);
        command
    }

    fn run(&self, args: &[&str]) -> Result<String> {
        let output = self
            .git(args)
            .output()
            .with_context(|| format!("failed to run `git {}`", args.join(" ")))?;
        if !output.status.success() {
            bail!(
                "`git {}` failed: {}",
                args.join(" "),
                describe_failure(&output)
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    fn run_trimmed(&self, args: &[&str]) -> Result<String> {
        Ok(self.run(args)?.trim().to_string())
    }

    /// Run a command whose non-zero exit is an expected "no" answer.
    fn run_opt(&self, args: &[&str]) -> Result<Option<String>> {
        let output = self
            .git(args)
            .output()
            .with_context(|| format!("failed to run `git {}`", args.join(" ")))?;
        if output.status.success() {
            Ok(Some(
                String::from_utf8_lossy(&output.stdout).trim().to_string(),
            ))
        } else if output.status.code() == Some(1) {
            Ok(None)
        } else {
            bail!(
                "`git {}` failed: {}",
                args.join(" "),
                describe_failure(&output)
            );
        }
    }

    /// Run a command whose arguments are raw path bytes.
    fn run_os(&self, args: &[&OsStr]) -> Result<String> {
        let output = self
            .git_os(args)
            .output()
            .with_context(|| format!("failed to run `git {}`", describe_args(args)))?;
        if !output.status.success() {
            bail!(
                "`git {}` failed: {}",
                describe_args(args),
                describe_failure(&output)
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Like [`GitRepository::run_opt`], but for arguments that are raw path bytes.
    fn run_opt_os(&self, args: &[&OsStr]) -> Result<Option<String>> {
        let output = self
            .git_os(args)
            .output()
            .with_context(|| format!("failed to run `git {}`", describe_args(args)))?;
        if output.status.success() {
            Ok(Some(
                String::from_utf8_lossy(&output.stdout).trim().to_string(),
            ))
        } else if output.status.code() == Some(1) {
            Ok(None)
        } else {
            bail!(
                "`git {}` failed: {}",
                describe_args(args),
                describe_failure(&output)
            );
        }
    }

    fn run_with_stdin(&self, args: &[&str], input: &[u8]) -> Result<String> {
        let mut command = self.git(args);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to spawn `git {}`", args.join(" ")))?;
        child
            .stdin
            .take()
            .context("`git` stdin was not piped")?
            .write_all(input)
            .context("failed to write to `git` stdin")?;
        let output = child
            .wait_with_output()
            .with_context(|| format!("failed to wait for `git {}`", args.join(" ")))?;
        if !output.status.success() {
            bail!(
                "`git {}` failed: {}",
                args.join(" "),
                describe_failure(&output)
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn run_with_index(&self, index: &Path, args: &[&str]) -> Result<String> {
        Ok(
            String::from_utf8_lossy(&self.run_with_index_bytes(index, args)?)
                .trim()
                .to_string(),
        )
    }

    fn run_with_index_bytes(&self, index: &Path, args: &[&str]) -> Result<Vec<u8>> {
        let output = self
            .git(args)
            .env("GIT_INDEX_FILE", index)
            .output()
            .with_context(|| format!("failed to run `git {}`", args.join(" ")))?;
        if !output.status.success() {
            bail!(
                "`git {}` failed: {}",
                args.join(" "),
                describe_failure(&output)
            );
        }
        Ok(output.stdout)
    }

    fn empty_tree(&self) -> Result<String> {
        self.run_with_stdin(&["hash-object", "-t", "tree", "-w", "--stdin"], b"")
    }

    fn resolve_commit(&self, commit: &str) -> Result<String> {
        validate_object_id(commit, "commit")?;
        let rev = format!("{commit}^{{commit}}");
        self.run_trimmed(&["rev-parse", "--verify", "--end-of-options", rev.as_str()])
            .with_context(|| format!("cannot resolve commit {commit:?}"))
    }

    fn resolve_tree(&self, tree: &str) -> Result<String> {
        validate_object_id(tree, "tree")?;
        let rev = format!("{tree}^{{tree}}");
        self.run_trimmed(&["rev-parse", "--verify", "--end-of-options", rev.as_str()])
            .with_context(|| format!("cannot resolve tree {tree:?}"))
    }

    /// Confirm `object` is an existing commit or tree and return it unchanged.
    fn resolve_object(&self, object: &str) -> Result<String> {
        validate_object_id(object, "object")?;
        let kind = self
            .run_trimmed(&["cat-file", "-t", object])
            .with_context(|| format!("cannot resolve object {object:?}"))?;
        if kind != "commit" && kind != "tree" {
            bail!("snapshot object must be a commit or tree, not {kind}: {object:?}");
        }
        Ok(object.to_string())
    }

    /// Create a commit object from an existing tree, preserving the tree's
    /// exact binary, executable, and symlink entries.
    fn create_commit(
        &self,
        tree: &str,
        parents: &[&str],
        message: &str,
        signature: &Signature,
    ) -> Result<String> {
        let date = format!("{} +0000", signature.timestamp);
        let mut command = self.git(&["commit-tree", "--no-gpg-sign", tree]);
        for parent in parents {
            command.arg("-p").arg(parent);
        }
        command
            .env("GIT_AUTHOR_NAME", &signature.name)
            .env("GIT_AUTHOR_EMAIL", &signature.email)
            .env("GIT_AUTHOR_DATE", &date)
            .env("GIT_COMMITTER_NAME", &signature.name)
            .env("GIT_COMMITTER_EMAIL", &signature.email)
            .env("GIT_COMMITTER_DATE", &date)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .context("failed to spawn `git commit-tree`")?;
        child
            .stdin
            .take()
            .context("`git commit-tree` stdin was not piped")?
            .write_all(message.as_bytes())
            .context("failed to write the commit message to `git commit-tree`")?;
        let output = child
            .wait_with_output()
            .context("failed to wait for `git commit-tree`")?;
        if !output.status.success() {
            bail!("`git commit-tree` failed: {}", describe_failure(&output));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Move `refs/heads/<branch>` to `new` with a compare-and-swap, refusing
    /// non-fast-forward moves and treating an identical tip as a no-op.
    ///
    /// A symbolic destination ref is refused outright and the update uses
    /// `--no-deref`, so an alias can never move the branch it points at.
    fn update_branch(&self, branch: &str, new: &str) -> Result<()> {
        let refname = format!("refs/heads/{branch}");
        if self
            .run_opt(&["symbolic-ref", "-q", refname.as_str()])?
            .is_some()
        {
            bail!("refusing to update {refname:?}: it is a symbolic ref");
        }
        let old = if self
            .run_opt(&["show-ref", "--verify", "--quiet", refname.as_str()])?
            .is_some()
        {
            Some(self.run_trimmed(&[
                "rev-parse",
                "--verify",
                "--end-of-options",
                refname.as_str(),
            ])?)
        } else {
            None
        };
        match old {
            Some(old) if old == new => Ok(()),
            Some(old) => {
                let output = self
                    .git(&["merge-base", "--is-ancestor", old.as_str(), new])
                    .output()
                    .context("failed to run `git merge-base --is-ancestor`")?;
                match output.status.code() {
                    Some(0) => {}
                    Some(1) => bail!(
                        "refusing to move branch {branch:?} from {old} to {new}: not a fast-forward"
                    ),
                    _ => bail!(
                        "`git merge-base --is-ancestor` failed: {}",
                        describe_failure(&output)
                    ),
                }
                self.run_trimmed(&[
                    "update-ref",
                    "--no-deref",
                    refname.as_str(),
                    new,
                    old.as_str(),
                ])?;
                Ok(())
            }
            None => {
                let zero = "0".repeat(new.len());
                self.run_trimmed(&[
                    "update-ref",
                    "--no-deref",
                    refname.as_str(),
                    new,
                    zero.as_str(),
                ])?;
                Ok(())
            }
        }
    }

    /// Branches currently checked out in this repository or any linked
    /// worktree. Updating one of these would desynchronize a live worktree.
    fn checked_out_branches(&self) -> Result<Vec<String>> {
        let mut branches = Vec::new();
        if let Some(head) = self.run_opt(&["symbolic-ref", "-q", "HEAD"])?
            && let Some(branch) = head.strip_prefix("refs/heads/")
        {
            branches.push(branch.to_string());
        }
        let list = self.run(&["worktree", "list", "--porcelain"])?;
        for line in list.lines() {
            if let Some(branch) = line.strip_prefix("branch refs/heads/") {
                branches.push(branch.trim().to_string());
            }
        }
        Ok(branches)
    }

    fn validate_branch(&self, branch: &str) -> Result<()> {
        if branch.is_empty() {
            bail!("branch name is empty");
        }
        if branch.starts_with('-') {
            bail!("branch name must not start with '-': {branch:?}");
        }
        if branch == "@" {
            bail!("branch name must not be the HEAD shorthand '@'");
        }
        let output = self
            .git(&["check-ref-format", "--branch", branch])
            .output()
            .context("failed to run `git check-ref-format`")?;
        if !output.status.success() {
            bail!(
                "invalid branch name {branch:?}: {}",
                describe_failure(&output)
            );
        }
        Ok(())
    }

    fn validate_ref(&self, refname: &str) -> Result<()> {
        let output = self
            .git(&["check-ref-format", refname])
            .output()
            .context("failed to run `git check-ref-format`")?;
        if !output.status.success() {
            bail!(
                "invalid ref name {refname:?}: {}",
                describe_failure(&output)
            );
        }
        Ok(())
    }
}

/// Remove every environment variable that could redirect Git.
fn sanitize(command: &mut Command) {
    for name in SANITIZED_ENV {
        command.env_remove(name);
    }
}

/// A single long-lived `git cat-file --batch` child used to stream every blob
/// in a tree listing.
///
/// One process serves the whole fork instead of one `git cat-file` per file,
/// and content is streamed straight into the destination file so a large blob
/// is never buffered whole in memory. The child is killed on drop.
struct BatchCatFile {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl BatchCatFile {
    /// Spawn `git cat-file --batch` against `repository`.
    fn spawn(repository: &GitRepository) -> Result<Self> {
        let mut child = repository
            .git(&["cat-file", "--batch"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn `git cat-file --batch`")?;
        let stdin = child
            .stdin
            .take()
            .context("`git cat-file --batch` stdin was not piped")?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("`git cat-file --batch` stdout was not piped")?,
        );
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    /// Request `oid` and return the blob's size, leaving the reader positioned
    /// at the first content byte.
    fn request(&mut self, oid: &str) -> Result<u64> {
        self.stdin
            .write_all(oid.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .context("failed to write to `git cat-file --batch` stdin")?;
        let mut header = String::new();
        let read = self
            .stdout
            .read_line(&mut header)
            .context("failed to read from `git cat-file --batch` stdout")?;
        if read == 0 {
            bail!("`git cat-file --batch` closed its output unexpectedly");
        }
        let header = header.trim_end_matches(['\n', '\r']);
        let mut fields = header.split(' ');
        let echoed = fields.next().unwrap_or_default();
        let kind = fields.next().unwrap_or_default();
        if kind == "missing" {
            bail!("object {oid} is missing from the object store");
        }
        if echoed != oid {
            bail!("`git cat-file --batch` returned {echoed:?} for {oid:?}");
        }
        if kind != "blob" {
            bail!("object {oid} is a {kind}, not a blob");
        }
        fields
            .next()
            .context("`git cat-file --batch` header is missing a size")?
            .parse::<u64>()
            .with_context(|| format!("invalid `git cat-file --batch` size for {oid}"))
    }

    /// Stream the blob `oid` into `sink` without buffering it whole.
    fn stream_blob(&mut self, oid: &str, sink: &mut impl Write) -> Result<()> {
        let size = self.request(oid)?;
        let mut limited = self.stdout.by_ref().take(size);
        let copied = std::io::copy(&mut limited, sink)
            .with_context(|| format!("failed to stream blob {oid}"))?;
        if copied != size {
            bail!("blob {oid} ended after {copied} of {size} bytes");
        }
        self.consume_trailer(oid)
    }

    /// Read the blob `oid` into memory (used for small symlink targets).
    fn read_blob(&mut self, oid: &str) -> Result<Vec<u8>> {
        let size = self.request(oid)?;
        let mut bytes = vec![0u8; size as usize];
        self.stdout
            .read_exact(&mut bytes)
            .with_context(|| format!("failed to read blob {oid}"))?;
        self.consume_trailer(oid)?;
        Ok(bytes)
    }

    /// Consume the trailing newline `--batch` writes after each object.
    fn consume_trailer(&mut self, oid: &str) -> Result<()> {
        let mut newline = [0u8; 1];
        self.stdout
            .read_exact(&mut newline)
            .with_context(|| format!("failed to read the trailer for blob {oid}"))?;
        if newline[0] != b'\n' {
            bail!("`git cat-file --batch` did not terminate blob {oid} with a newline");
        }
        Ok(())
    }
}

impl Drop for BatchCatFile {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The absolute form of `destination`, used for the pending record and hash.
fn absolute_destination(destination: &Path) -> PathBuf {
    std::path::absolute(destination).unwrap_or_else(|_| destination.to_path_buf())
}

/// Deterministic pending-file name for `destination`: a sanitized file name
/// plus a 64-bit FNV-1a hash of the absolute destination path.
fn pending_fork_name(destination: &Path) -> String {
    let name = destination
        .file_name()
        .map(|name| {
            name.to_string_lossy()
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                        c
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
        })
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "fork".to_string());
    let hash = fnv64(absolute_destination(destination).as_os_str().as_bytes());
    format!("fork-{name}-{hash:016x}")
}

/// FNV-1a 64-bit hash, used only to disambiguate pending-file names.
fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Truncate and rewrite a pending-fork registry file.
fn write_pending(file: &fs::File, record: &PendingFork) -> Result<()> {
    let bytes = serde_json::to_vec(record).context("failed to encode the pending fork record")?;
    let mut handle = file;
    handle
        .set_len(0)
        .context("failed to truncate the pending fork record")?;
    handle
        .write_all(&bytes)
        .context("failed to write the pending fork record")?;
    handle
        .flush()
        .context("failed to flush the pending fork record")?;
    Ok(())
}

/// Discover an ordinary Git worktree with `git rev-parse`.
fn discover(workspace: &Path) -> Result<(PathBuf, PathBuf)> {
    let mut command = Command::new("git");
    command
        .args(["rev-parse", "--show-toplevel", "--absolute-git-dir"])
        .current_dir(workspace);
    sanitize(&mut command);
    let output = command
        .output()
        .with_context(|| format!("failed to run `git rev-parse` in {}", workspace.display()))?;
    if !output.status.success() {
        bail!(
            "{} is not a Git repository: {}",
            workspace.display(),
            describe_failure(&output)
        );
    }
    let text =
        String::from_utf8(output.stdout).context("`git rev-parse` produced non-UTF-8 output")?;
    let mut lines = text.lines();
    let top = lines
        .next()
        .context("`git rev-parse --show-toplevel` produced no output")?;
    let git_dir = lines
        .next()
        .context("`git rev-parse --absolute-git-dir` produced no output")?;
    let workspace = PathBuf::from(top)
        .canonicalize()
        .with_context(|| format!("worktree root {top} does not exist"))?;
    let git_dir = PathBuf::from(git_dir)
        .canonicalize()
        .with_context(|| format!("git directory {git_dir} does not exist"))?;
    Ok((workspace, git_dir))
}

/// Find the nearest [`WORKSPACE_MARKER`] at `start` or one of its ancestors.
fn find_workspace_link(start: &Path) -> Result<Option<(PathBuf, WorkspaceLink)>> {
    let start = start
        .canonicalize()
        .with_context(|| format!("workspace {} does not exist", start.display()))?;
    let mut current: Option<&Path> = Some(start.as_path());
    while let Some(directory) = current {
        let marker = directory.join(WORKSPACE_MARKER);
        if marker.is_file() {
            let bytes = fs::read(&marker)
                .with_context(|| format!("failed to read {}", marker.display()))?;
            let link: WorkspaceLink = serde_json::from_slice(&bytes)
                .with_context(|| format!("failed to parse {}", marker.display()))?;
            if !link.git_dir.is_absolute() {
                bail!(
                    "workspace link {} has a relative git_dir: {}",
                    marker.display(),
                    link.git_dir.display()
                );
            }
            if link.thread.is_empty() {
                bail!("workspace link {} has an empty thread", marker.display());
            }
            return Ok(Some((directory.to_path_buf(), link)));
        }
        current = directory.parent();
    }
    Ok(None)
}

/// The path of a `git ls-tree -r -z` entry: everything after the tab.
fn entry_path(entry: &[u8]) -> Result<&[u8]> {
    let tab = entry
        .iter()
        .position(|byte| *byte == b'\t')
        .context("malformed `git ls-tree` entry: missing tab")?;
    Ok(&entry[tab + 1..])
}

/// Whether `path` is `prefix` or lies under it.
///
/// Used to decide whether a baseline path belongs to a partial fork's sparse
/// prefixes. The comparison is byte-exact on whole path components, so `src`
/// does not match `srcx`.
pub(crate) fn path_under(path: &[u8], prefix: &[u8]) -> bool {
    path == prefix
        || (path.len() > prefix.len() && path.starts_with(prefix) && path[prefix.len()] == b'/')
}

/// Whether any component of `path` is snapshot metadata that must never be
/// captured (the Git directory, the Jujutsu directory, the fork marker, the
/// mutation lock, and an atomic-write temp file left behind by a crash).
fn is_metadata_path(path: &[u8]) -> bool {
    path.split(|byte| *byte == b'/').any(|component| {
        component == b".git"
            || component == b".jj"
            || component == b".juto-workspace.json"
            || component == b".juto-mutation.lock"
            || is_atomic_write_temp(component)
    })
}

/// Whether `component` is exactly the temp name
/// `vcs::atomic_write_preserving` creates: `.<name>.juto-<pid>-<counter>.tmp`.
///
/// The match is deliberately tight: a leading dot, the `.juto-` marker, a
/// numeric pid and counter, and the `.tmp` suffix. A user file that merely
/// resembles the pattern is still captured.
fn is_atomic_write_temp(component: &[u8]) -> bool {
    let Some(rest) = component.strip_prefix(b".") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(b".tmp") else {
        return false;
    };
    // The marker is the last `.juto-` in the name, followed by `<pid>-<counter>`.
    let Some(marker) = rest.windows(6).rposition(|window| window == b".juto-") else {
        return false;
    };
    let mut fields = rest[marker + 6..].split(|byte| *byte == b'-');
    let pid = fields.next().unwrap_or_default();
    let counter = fields.next().unwrap_or_default();
    !pid.is_empty()
        && !counter.is_empty()
        && fields.next().is_none()
        && pid.iter().all(u8::is_ascii_digit)
        && counter.iter().all(u8::is_ascii_digit)
}

/// Append one `git update-index --index-info -z` record.
fn push_index_entry(info: &mut Vec<u8>, mode: &str, oid: &str, path: &[u8]) {
    info.extend_from_slice(mode.as_bytes());
    info.push(b' ');
    info.extend_from_slice(oid.as_bytes());
    info.push(b'\t');
    info.extend_from_slice(path);
    info.push(0);
}

/// Validate a `git ls-tree` path and turn it into a relative [`PathBuf`].
fn validate_tree_path(path: &[u8]) -> Result<PathBuf> {
    if path.is_empty() {
        bail!("tree entry has an empty path");
    }
    if path[0] == b'/' {
        bail!("tree entry has an absolute path");
    }
    let mut result = PathBuf::new();
    for component in path.split(|byte| *byte == b'/') {
        if component.is_empty() {
            bail!("tree entry path has an empty component");
        }
        if component == b"." || component == b".." {
            bail!("tree entry path escapes the destination");
        }
        if component == b".git" {
            bail!("tree entry path contains a .git component");
        }
        if component == WORKSPACE_MARKER.as_bytes() {
            bail!("tree entry path collides with the workspace marker");
        }
        result.push(OsStr::from_bytes(component));
    }
    Ok(result)
}

fn validate_object_id(id: &str, kind: &str) -> Result<()> {
    if !matches!(id.len(), 40 | 64) || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{kind} must be a full immutable object ID: {id:?}");
    }
    Ok(())
}

/// One parsed `git diff-tree --raw` record, minus its path.
struct RawChange {
    new_mode: String,
    new_oid: String,
    status: u8,
}

/// Parse the metadata field of a `git diff-tree --raw` record.
///
/// The field is `:<old-mode> <new-mode> <old-sha> <new-sha> <status>` (the
/// leading colon is optional here so the parser also accepts the colon-less
/// form). The path is a separate NUL-terminated field and is not parsed here.
fn parse_raw_change(meta: &[u8]) -> Result<RawChange> {
    let meta = meta.strip_prefix(b":").unwrap_or(meta);
    let meta = std::str::from_utf8(meta).context("`git diff-tree` metadata is not UTF-8")?;
    let mut fields = meta.split_whitespace();
    let _old_mode = fields
        .next()
        .context("`git diff-tree` record is missing an old mode")?;
    let new_mode = fields
        .next()
        .context("`git diff-tree` record is missing a new mode")?;
    let _old_oid = fields
        .next()
        .context("`git diff-tree` record is missing an old object id")?;
    let new_oid = fields
        .next()
        .context("`git diff-tree` record is missing a new object id")?;
    let status = fields
        .next()
        .context("`git diff-tree` record is missing a status")?;
    if fields.next().is_some() {
        bail!("unexpected `git diff-tree` metadata: {meta:?}");
    }
    let status = status.as_bytes();
    if status.len() != 1 {
        bail!("`git diff-tree` status must be one character: {status:?}");
    }
    Ok(RawChange {
        new_mode: new_mode.to_string(),
        new_oid: new_oid.to_string(),
        status: status[0],
    })
}

/// Build a `git ls-tree`-style entry from a raw change and its path, so the
/// existing [`GitRepository::materialize_entry`] can write it.
fn raw_entry(change: &RawChange, path: &[u8]) -> Vec<u8> {
    let kind = if change.new_mode == "160000" {
        "commit"
    } else {
        "blob"
    };
    let mut entry =
        Vec::with_capacity(change.new_mode.len() + change.new_oid.len() + path.len() + 4);
    entry.extend_from_slice(change.new_mode.as_bytes());
    entry.push(b' ');
    entry.extend_from_slice(kind.as_bytes());
    entry.push(b' ');
    entry.extend_from_slice(change.new_oid.as_bytes());
    entry.push(b'\t');
    entry.extend_from_slice(path);
    entry
}

/// Remove `relative` under `destination` and prune now-empty parent
/// directories, stopping at `destination`.
fn remove_path(destination: &Path, relative: &Path) -> Result<()> {
    let target = destination.join(relative);
    match fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.is_dir() => {
            bail!("cannot delete {}: it is a directory", target.display());
        }
        Ok(_) => {
            fs::remove_file(&target)
                .with_context(|| format!("failed to delete {}", target.display()))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("cannot stat {}", target.display()));
        }
    }
    let mut parent = target.parent();
    while let Some(directory) = parent {
        if directory == destination {
            break;
        }
        match fs::read_dir(directory) {
            Ok(mut entries) => {
                if entries.next().is_some() {
                    break;
                }
            }
            Err(_) => break,
        }
        if fs::remove_dir(directory).is_err() {
            break;
        }
        parent = directory.parent();
    }
    Ok(())
}

/// Whether `git version` output names a version at least `major.minor`.
fn version_at_least(version: &str, major: u32, minor: u32) -> bool {
    let Some(rest) = version.trim().strip_prefix("git version ") else {
        return false;
    };
    let mut fields = rest.split(|c: char| !c.is_ascii_digit());
    let Some(found_major) = fields.next().and_then(|field| field.parse::<u32>().ok()) else {
        return false;
    };
    let found_minor = fields
        .next()
        .and_then(|field| field.parse::<u32>().ok())
        .unwrap_or(0);
    (found_major, found_minor) >= (major, minor)
}

/// Parse `Name <email> timestamp tz` from `git var GIT_AUTHOR_IDENT`.
fn parse_ident(ident: &str) -> Result<(String, String)> {
    let open = ident.find('<').context("git identity is missing '<'")?;
    let close = ident[open..]
        .find('>')
        .map(|offset| offset + open)
        .context("git identity is missing '>'")?;
    let name = ident[..open].trim();
    let email = ident[open + 1..close].trim();
    if name.is_empty() || email.is_empty() {
        bail!("git identity has an empty name or email: {ident:?}");
    }
    Ok((name.to_string(), email.to_string()))
}

/// Render raw argument bytes for a command error message.
fn describe_args(args: &[&OsStr]) -> String {
    args.iter()
        .map(|arg| arg.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render a failed command's stderr, falling back to its exit status.
fn describe_failure(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim();
    if stderr.is_empty() {
        format!("exit status {}", output.status)
    } else {
        stderr.to_string()
    }
}
