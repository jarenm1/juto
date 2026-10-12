//! Behavioral integration tests for [`juto_runtime::git_transfer`].
//!
//! Every test drives the real `git` binary against temporary plain Git
//! repositories and a local bare remote. Nothing is mocked: snapshots are
//! inspected with Git plumbing, and branch/remote refs are read back from disk.
//! No Jujutsu binary is involved.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use juto_runtime::git_transfer::{ExportPlan, ExportPoint, GitRepository, Signature};

/// A fixed identity so exported commits are byte-for-byte reproducible.
fn test_signature() -> Signature {
    Signature {
        name: "Juto Test".into(),
        email: "juto@test.local".into(),
        timestamp: 1_700_000_000,
    }
}

fn run(program: &str, args: &[&str], cwd: &Path) -> String {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap_or_else(|error| panic!("failed to run `{program} {}`: {error}", args.join(" ")));
    assert!(
        output.status.success(),
        "`{program} {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn run_stdin(program: &str, args: &[&str], cwd: &Path, input: &[u8]) -> String {
    let mut child = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn `{program} {}`: {error}", args.join(" ")));
    child.stdin.take().unwrap().write_all(input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "`{program} {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Run `git` inside a worktree directory.
fn git(cwd: &Path, args: &[&str]) -> String {
    run("git", args, cwd)
}

/// Run `git` against an explicit git directory.
fn git_at(git_dir: &Path, args: &[&str]) -> String {
    let mut full = vec!["--git-dir"];
    full.push(git_dir.to_str().unwrap());
    full.extend_from_slice(args);
    run("git", &full, git_dir)
}

fn git_raw(git_dir: &Path, args: &[&str]) -> Vec<u8> {
    let mut full = vec!["--git-dir"];
    full.push(git_dir.to_str().unwrap());
    full.extend_from_slice(args);
    let output = Command::new("git").args(&full).output().unwrap();
    assert!(
        output.status.success(),
        "`git {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn git_status(git_dir: &Path, args: &[&str]) -> ExitStatus {
    let mut full = vec!["--git-dir"];
    full.push(git_dir.to_str().unwrap());
    full.extend_from_slice(args);
    Command::new("git").args(&full).output().unwrap().status
}

/// Initialize a plain Git repository with a deterministic identity.
fn init_repo(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q", "-b", "main"]);
    git(dir, &["config", "user.name", "Juto Test"]);
    git(dir, &["config", "user.email", "juto@test.local"]);
}

/// Stage everything and commit, returning the resulting commit ID.
fn commit(dir: &Path, message: &str) -> String {
    git(dir, &["add", "-A"]);
    git(dir, &["commit", "-q", "-m", message]);
    git(dir, &["rev-parse", "HEAD"])
}

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
}

fn set_executable(path: &Path) {
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn repo(temp: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
    let dir = temp.path().join(name);
    init_repo(&dir);
    dir
}

/// Parse `git ls-tree -r` into `path -> (mode, object id)`.
fn tree_entries(git_dir: &Path, tree: &str) -> BTreeMap<String, (String, String)> {
    git_at(git_dir, &["ls-tree", "-r", tree])
        .lines()
        .map(|line| {
            let (meta, path) = line.split_once('\t').expect("ls-tree line has a tab");
            let mut fields = meta.split_whitespace();
            let mode = fields.next().unwrap().to_string();
            let _kind = fields.next().unwrap();
            let oid = fields.next().unwrap().to_string();
            (path.to_string(), (mode, oid))
        })
        .collect()
}

/// The parent commit IDs of `commit`, in order.
fn parents_of(git_dir: &Path, commit: &str) -> Vec<String> {
    git_at(git_dir, &["rev-list", "--parents", "-n", "1", commit])
        .split_whitespace()
        .skip(1)
        .map(str::to_string)
        .collect()
}

#[test]
fn open_discovers_plain_git_repository() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    let repository = GitRepository::open(&dir).unwrap();
    assert_eq!(
        repository.workspace().to_path_buf(),
        dir.canonicalize().unwrap()
    );
    assert_eq!(
        repository.git_dir().to_path_buf(),
        dir.join(".git").canonicalize().unwrap()
    );
    assert_eq!(repository.thread(), None);

    // Opening from a nested directory still resolves the worktree root.
    let nested = dir.join("a/b");
    fs::create_dir_all(&nested).unwrap();
    let repository = GitRepository::open(&nested).unwrap();
    assert_eq!(
        repository.workspace().to_path_buf(),
        dir.canonicalize().unwrap()
    );
}

#[test]
fn head_commit_reports_unborn_and_born() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    let repository = GitRepository::open(&dir).unwrap();
    assert_eq!(repository.head_commit().unwrap(), None);

    write_file(&dir.join("a.txt"), b"a\n");
    let head = commit(&dir, "base");
    assert_eq!(repository.head_commit().unwrap(), Some(head));
}

#[test]
fn snapshot_tree_accepts_commit_tree_and_root() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    let head = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    assert_eq!(repository.snapshot_tree(&head).unwrap(), tree);
    assert_eq!(repository.snapshot_tree(&tree).unwrap(), tree);

    let zero = "0".repeat(40);
    let empty = git_at(&git_dir, &["hash-object", "-t", "tree", "--stdin"]);
    assert_eq!(repository.snapshot_tree(&zero).unwrap(), empty);

    // A blob is neither a commit nor a tree.
    let blob = git_at(&git_dir, &["rev-parse", "HEAD:a.txt"]);
    assert!(repository.snapshot_tree(&blob).is_err());
    assert!(repository.snapshot_tree("short").is_err());
}

#[test]
fn capture_tree_ignores_and_preserves_tracked_ignored() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    write_file(&dir.join("a.txt"), b"base\n");
    write_file(&dir.join("tracked.log"), b"tracked base\n");
    commit(&dir, "base");

    // Now ignore tracked.log and ignored.log, and commit the ignore rule.
    write_file(&dir.join(".gitignore"), b"tracked.log\nignored.log\n");
    commit(&dir, "ignore");
    let baseline = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    // Open before a stray marker appears so discovery uses `git rev-parse`.
    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    // Live edits: modify the tracked-but-ignored file, add an ignored file, and
    // add files that must never be captured.
    write_file(&dir.join("tracked.log"), b"tracked changed\n");
    write_file(&dir.join("ignored.log"), b"ignored\n");
    write_file(&dir.join("new.txt"), b"new\n");
    write_file(&dir.join(".jj/working"), b"jj\n");
    write_file(&dir.join(".juto-workspace.json"), b"{}\n");
    write_file(&dir.join(".juto-mutation.lock"), b"lock\n");

    let captured = repository.capture_tree(&baseline).unwrap();

    let entries = tree_entries(&git_dir, &captured);
    assert!(entries.contains_key("a.txt"));
    assert!(entries.contains_key("new.txt"));
    assert!(
        entries.contains_key("tracked.log"),
        "a file tracked by the baseline must survive being ignored"
    );
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["tracked.log"].1]),
        b"tracked changed\n"
    );
    assert!(!entries.contains_key("ignored.log"));
    assert!(!entries.contains_key(".jj/working"));
    assert!(!entries.contains_key(".juto-workspace.json"));
    assert!(!entries.contains_key(".juto-mutation.lock"));
}

#[test]
fn capture_tree_preserves_raw_bytes_under_attributes() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    // A `text` attribute would normalize line endings and a `filter` attribute
    // would rewrite content; neither may touch a native snapshot.
    write_file(
        &dir.join(".gitattributes"),
        b"*.txt text\n*.md filter=evil\n",
    );
    git(&dir, &["config", "filter.evil.clean", "tr a-z A-Z"]);
    write_file(&dir.join("a.txt"), b"line1\n");
    write_file(&dir.join("b.md"), b"mixed\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let baseline = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    // A CRLF-only edit must produce a different tree, and the filter must not run.
    write_file(&dir.join("a.txt"), b"line1\r\n");
    write_file(&dir.join("b.md"), b"mixed2\n");
    let captured = repository.capture_tree(&baseline).unwrap();
    assert_ne!(
        captured, baseline,
        "a CRLF-only edit must change the snapshot"
    );

    let entries = tree_entries(&git_dir, &captured);
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["a.txt"].1]),
        b"line1\r\n",
        "raw CRLF bytes must be preserved"
    );
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["b.md"].1]),
        b"mixed2\n",
        "the configured clean filter must not be invoked"
    );
}

#[test]
fn capture_tree_leaves_user_index_and_head_untouched() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"base\n");
    let head = commit(&dir, "base");
    let baseline = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    write_file(&dir.join("a.txt"), b"changed\n");
    write_file(&dir.join("b.txt"), b"new\n");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    let status_before = git(&dir, &["status", "--porcelain"]);
    let index_before = fs::read(git_dir.join("index")).unwrap();

    repository.capture_tree(&baseline).unwrap();

    let index_after = fs::read(git_dir.join("index")).unwrap();
    assert_eq!(
        index_after, index_before,
        "the user index must not be rewritten"
    );
    assert_eq!(
        git(&dir, &["status", "--porcelain"]),
        status_before,
        "the worktree status must not change"
    );
    assert_eq!(git_at(&git_dir, &["rev-parse", "HEAD"]), head);
}

#[test]
fn capture_tree_tolerates_untracked_nested_repository() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"base\n");
    let head = commit(&dir, "base");
    let baseline = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    // `git ls-files --others` enumerates an untracked nested repository as the
    // directory itself (`sub/`), which can never be a blob. It is not in the
    // baseline, so it needs no removal record either.
    let nested = dir.join("sub");
    fs::create_dir_all(&nested).unwrap();
    git(&nested, &["init", "-q", "-b", "main"]);

    let captured = repository.capture_tree(&baseline).unwrap();
    let entries = tree_entries(&git_dir, &captured);
    assert_eq!(
        entries.get("a.txt").map(|(mode, _)| mode.as_str()),
        Some("100644")
    );
    assert!(
        !entries
            .keys()
            .any(|path| path == "sub" || path.starts_with("sub/")),
        "an untracked nested repository must not be captured: {entries:?}"
    );
    assert_eq!(git_at(&git_dir, &["rev-parse", "HEAD"]), head);
}

#[test]
fn capture_tree_handles_file_replacing_tracked_directory() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a/b.txt"), b"nested\n");
    commit(&dir, "base");
    let baseline = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    // The worktree now has a FILE where the baseline tracked a directory, so
    // `symlink_metadata(workspace/a/b.txt)` fails with ENOTDIR.
    fs::remove_dir_all(dir.join("a")).unwrap();
    write_file(&dir.join("a"), b"now a file\n");

    let captured = repository.capture_tree(&baseline).unwrap();
    let entries = tree_entries(&git_dir, &captured);
    assert_eq!(
        entries.get("a").map(|(mode, _)| mode.as_str()),
        Some("100644")
    );
    assert!(
        !entries.contains_key("a/b.txt"),
        "the replaced directory must be removed from the snapshot: {entries:?}"
    );
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["a"].1]),
        b"now a file\n"
    );
}

#[test]
fn capture_tree_excludes_atomic_write_temp_files() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"base\n");
    commit(&dir, "base");
    let baseline = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    // Exactly the shape `vcs::atomic_write_preserving` leaves behind when it
    // crashes between `create_new` and `rename`.
    write_file(&dir.join(".a.txt.juto-4242-7.tmp"), b"partial\n");
    // Files that merely look similar are ordinary user files and must survive.
    write_file(&dir.join("keep.juto-1-2.tmp"), b"user\n");
    write_file(&dir.join(".keep.tmp"), b"user\n");
    write_file(&dir.join(".keep.juto-x.tmp"), b"user\n");

    let captured = repository.capture_tree(&baseline).unwrap();
    let entries = tree_entries(&git_dir, &captured);
    assert!(
        !entries.contains_key(".a.txt.juto-4242-7.tmp"),
        "an atomic-write temp file must never be captured: {entries:?}"
    );
    assert!(entries.contains_key("keep.juto-1-2.tmp"));
    assert!(entries.contains_key(".keep.tmp"));
    assert!(entries.contains_key(".keep.juto-x.tmp"));
}

#[test]
fn retain_snapshot_pins_object_ref_and_rejects_unsafe_ids() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let head = git_at(&git_dir, &["rev-parse", "HEAD"]);
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // The graph pins the baseline commit and its tree under separate ids.
    repository.retain_snapshot("base", &head).unwrap();
    assert_eq!(
        git_at(&git_dir, &["rev-parse", "refs/juto/snapshots/base"]),
        head
    );
    repository.retain_snapshot("base-tree", &tree).unwrap();
    assert_eq!(
        git_at(&git_dir, &["rev-parse", "refs/juto/snapshots/base-tree"]),
        tree
    );

    let uuid = "0192f0aa-0000-7000-8000-000000000000";
    repository.retain_snapshot(uuid, &tree).unwrap();
    assert_eq!(
        git_at(
            &git_dir,
            &["rev-parse", &format!("refs/juto/snapshots/{uuid}")]
        ),
        tree
    );

    // A blob is neither a commit nor a tree.
    let blob = git_at(&git_dir, &["rev-parse", "HEAD:a.txt"]);
    assert!(repository.retain_snapshot("blob", &blob).is_err());
    assert!(repository.retain_snapshot("../escape", &tree).is_err());
    assert!(repository.retain_snapshot("nested/id", &tree).is_err());
    assert!(repository.retain_snapshot("", &tree).is_err());
}

#[test]
fn path_tracked_and_diff_trees() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"base\n");
    commit(&dir, "base");
    let before = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    write_file(&dir.join("a.txt"), b"changed\n");
    write_file(&dir.join("b.txt"), b"new\n");
    commit(&dir, "second");
    let after = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    let repository = GitRepository::open(&dir).unwrap();
    assert!(repository.path_tracked(&after, b"a.txt").unwrap());
    assert!(repository.path_tracked(&after, b"b.txt").unwrap());
    assert!(!repository.path_tracked(&after, b"missing.txt").unwrap());
    assert!(!repository.path_tracked(&before, b"b.txt").unwrap());

    let patch = repository.diff_trees(&before, &after, b"a.txt").unwrap();
    assert!(patch.contains("changed"), "{patch}");
    assert!(patch.contains("a.txt"), "{patch}");
    assert!(
        !patch.contains("b.txt"),
        "diff must be limited to the path: {patch}"
    );
}

#[test]
fn pathspecs_are_literal() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    write_file(&dir.join("*.txt"), b"wild\n");
    commit(&dir, "base");
    let before = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    write_file(&dir.join("a.txt"), b"a2\n");
    write_file(&dir.join("*.txt"), b"wild2\n");
    commit(&dir, "second");
    let after = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    let repository = GitRepository::open(&dir).unwrap();

    // A wildcard path is matched literally, never as a glob.
    let wild = repository.diff_trees(&before, &after, b"*.txt").unwrap();
    assert!(wild.contains("*.txt"), "{wild}");
    assert!(
        !wild.contains("a.txt"),
        "a literal wildcard must not match a.txt: {wild}"
    );

    let exact = repository.diff_trees(&before, &after, b"a.txt").unwrap();
    assert!(exact.contains("a.txt"), "{exact}");
    assert!(!exact.contains("*.txt"), "{exact}");

    assert!(repository.path_tracked(&after, b"*.txt").unwrap());
    assert!(repository.path_tracked(&after, b"a.txt").unwrap());
    // `a*.txt` is not a file; a glob would have matched `a.txt`.
    assert!(!repository.path_tracked(&after, b"a*.txt").unwrap());
}

#[test]
fn path_ignored_reports_ignore_rules() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join(".gitignore"), b"*.log\n");
    write_file(&dir.join("a.log"), b"ignored\n");
    write_file(&dir.join("b.txt"), b"tracked\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    assert!(repository.path_ignored(b"a.log").unwrap());
    assert!(!repository.path_ignored(b"b.txt").unwrap());
    assert!(!repository.path_ignored(b"missing.txt").unwrap());
    assert!(repository.path_ignored(b"").is_err());

    // A tracked file that matches an ignore rule still reports as ignored.
    git(&dir, &["add", "-f", "a.log"]);
    git(&dir, &["commit", "-q", "-m", "force tracked"]);
    assert!(repository.path_ignored(b"a.log").unwrap());
}

#[test]
fn path_queries_accept_raw_path_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"base\n");
    commit(&dir, "base");
    let before = git(&dir, &["rev-parse", "HEAD^{tree}"]);

    // A backslash is an ordinary filename byte, not a directory separator, and
    // a non-UTF-8 name must round-trip as raw bytes.
    let backslash = dir.join("a\\b.txt");
    write_file(&backslash, b"backslash\n");
    let raw = dir.join(OsStr::from_bytes(b"raw-\xff.txt"));
    write_file(&raw, b"raw\n");

    let repository = GitRepository::open(&dir).unwrap();
    let after = repository.capture_tree(&before).unwrap();

    assert!(repository.path_tracked(&after, b"a\\b.txt").unwrap());
    assert!(
        !repository.path_tracked(&after, b"a/b.txt").unwrap(),
        "a backslash must not be rewritten into a separator"
    );
    assert!(repository.path_tracked(&after, b"raw-\xff.txt").unwrap());
    assert!(!repository.path_tracked(&after, b"raw-\xfe.txt").unwrap());

    let patch = repository.diff_trees(&before, &after, b"a\\b.txt").unwrap();
    assert!(patch.contains("backslash"), "{patch}");
    assert!(
        !patch.contains("raw-"),
        "diff must stay path-scoped: {patch}"
    );

    assert!(!repository.path_ignored(b"a\\b.txt").unwrap());
    assert!(repository.path_ignored(b"").is_err());
}

#[test]
fn fork_workspace_materializes_tree_and_links_workspace() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    write_file(&dir.join("keep.txt"), b"keep\n");
    write_file(&dir.join("bin.dat"), &[0x00, 0x01, 0xff, 0x00, 0x0a]);
    write_file(&dir.join("run.sh"), b"#!/bin/sh\necho hi\n");
    set_executable(&dir.join("run.sh"));
    symlink("keep.txt", dir.join("link")).unwrap();
    write_file(&dir.join("nested/deep.txt"), b"deep\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    let destination = temp.path().join("fork");
    repository
        .fork_workspace(&tree, &destination, "thread-1")
        .unwrap();

    assert_eq!(fs::read(destination.join("keep.txt")).unwrap(), b"keep\n");
    assert_eq!(
        fs::read(destination.join("bin.dat")).unwrap(),
        vec![0x00, 0x01, 0xff, 0x00, 0x0a]
    );
    assert_eq!(
        fs::metadata(destination.join("run.sh"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o755
    );
    assert_eq!(
        fs::read_link(destination.join("link")).unwrap(),
        PathBuf::from("keep.txt")
    );
    assert_eq!(
        fs::read(destination.join("nested/deep.txt")).unwrap(),
        b"deep\n"
    );

    // The fork has no `.git`; the marker links it back to the shared store.
    assert!(!destination.join(".git").exists());
    let marker = fs::read_to_string(destination.join(".juto-workspace.json")).unwrap();
    assert!(marker.contains("thread-1"), "{marker}");
    assert!(marker.contains(git_dir.to_str().unwrap()), "{marker}");

    let forked = GitRepository::open(&destination).unwrap();
    assert_eq!(
        forked.workspace().to_path_buf(),
        destination.canonicalize().unwrap()
    );
    assert_eq!(forked.thread(), Some("thread-1"));
    assert_eq!(forked.git_dir().to_path_buf(), git_dir);

    // The fork captures through the shared object store.
    write_file(&destination.join("keep.txt"), b"fork edit\n");
    let captured = forked.capture_tree(&tree).unwrap();
    assert_ne!(captured, tree);
    let entries = tree_entries(&git_dir, &captured);
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["keep.txt"].1]),
        b"fork edit\n"
    );
    assert!(!entries.contains_key(".juto-workspace.json"));

    // A fork destination must be brand new; an existing one is never touched.
    assert!(
        repository
            .fork_workspace(&tree, &destination, "again")
            .is_err()
    );
    assert_eq!(
        fs::read(destination.join("keep.txt")).unwrap(),
        b"fork edit\n",
        "a refused fork must leave the existing directory untouched"
    );
}

#[test]
fn fork_workspace_skips_gitlink_and_capture_preserves_it() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    let head = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let blob = run(
        "git",
        &[
            "--git-dir",
            git_dir.to_str().unwrap(),
            "rev-parse",
            &format!("{head}:a.txt"),
        ],
        &dir,
    );
    // A tree with a regular file and a gitlink (submodule) entry.
    let gitlink = run_stdin(
        "git",
        &["--git-dir", git_dir.to_str().unwrap(), "mktree"],
        &dir,
        format!("100644 blob {blob}\ta.txt\n160000 commit {head}\tsub\n").as_bytes(),
    );

    let destination = temp.path().join("fork");
    repository
        .fork_workspace(&gitlink, &destination, "thread-1")
        .unwrap();
    // The regular file is materialized; the gitlink is deliberately skipped.
    assert_eq!(fs::read(destination.join("a.txt")).unwrap(), b"a\n");
    assert!(
        !destination.join("sub").exists(),
        "a gitlink must not be materialized"
    );

    // Capturing the fork preserves the baseline gitlink, so the tree is
    // unchanged and the fork does not read as a submodule deletion.
    let fork = GitRepository::open(&destination).unwrap();
    assert_eq!(
        fork.capture_tree(&gitlink).unwrap(),
        gitlink,
        "capture must preserve a baseline gitlink"
    );
}

/// A repo with `top.txt`, `crates/foo/a.txt`, and `crates/bar/b.txt`.
fn sparse_repo(temp: &tempfile::TempDir) -> (std::path::PathBuf, String) {
    let dir = repo(temp, "repo");
    write_file(&dir.join("top.txt"), b"top\n");
    fs::create_dir_all(dir.join("crates/foo")).unwrap();
    write_file(&dir.join("crates/foo/a.txt"), b"a\n");
    fs::create_dir_all(dir.join("crates/bar")).unwrap();
    write_file(&dir.join("crates/bar/b.txt"), b"b\n");
    let head = commit(&dir, "base");
    (dir, head)
}

#[test]
fn sparse_fork_materializes_only_the_requested_subtree() {
    let temp = tempfile::tempdir().unwrap();
    let (dir, head) = sparse_repo(&temp);
    let repository = GitRepository::open(&dir).unwrap();
    let destination = temp.path().join("sparse");
    repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &["crates/foo".to_string()])
        .unwrap();

    assert_eq!(
        fs::read(destination.join("crates/foo/a.txt")).unwrap(),
        b"a\n"
    );
    assert!(
        !destination.join("crates/bar").exists(),
        "a path outside the spec must not be materialized"
    );
    assert!(
        !destination.join("top.txt").exists(),
        "a path outside the spec must not be materialized"
    );

    let fork = GitRepository::open(&destination).unwrap();
    assert_eq!(fork.sparse(), Some(["crates/foo".to_string()].as_slice()));
}

#[test]
fn sparse_fork_capture_preserves_paths_outside_the_spec() {
    let temp = tempfile::tempdir().unwrap();
    let (dir, head) = sparse_repo(&temp);
    let repository = GitRepository::open(&dir).unwrap();
    let destination = temp.path().join("sparse");
    repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &["crates/foo".to_string()])
        .unwrap();

    // Nothing changed, so the captured tree must equal the base tree: the paths
    // the fork never materialized must not read as deletions.
    let git_dir = repository.git_dir().to_path_buf();
    let base_tree = run(
        "git",
        &[
            "--git-dir",
            git_dir.to_str().unwrap(),
            "rev-parse",
            &format!("{head}^{{tree}}"),
        ],
        &dir,
    );
    let fork = GitRepository::open(&destination).unwrap();
    assert_eq!(
        fork.capture_tree(&head).unwrap(),
        base_tree,
        "a sparse fork must capture a tree identical to the base when nothing changed"
    );
}

#[test]
fn sparse_fork_capture_records_edits_inside_the_spec() {
    let temp = tempfile::tempdir().unwrap();
    let (dir, head) = sparse_repo(&temp);
    let repository = GitRepository::open(&dir).unwrap();
    let destination = temp.path().join("sparse");
    repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &["crates/foo".to_string()])
        .unwrap();

    write_file(&destination.join("crates/foo/a.txt"), b"edited\n");
    let fork = GitRepository::open(&destination).unwrap();
    let captured = fork.capture_tree(&head).unwrap();
    assert_ne!(captured, head, "the edit must change the tree");

    let git_dir = repository.git_dir().to_path_buf();
    let listing = run(
        "git",
        &[
            "--git-dir",
            git_dir.to_str().unwrap(),
            "ls-tree",
            "-r",
            "--name-only",
            &captured,
        ],
        &dir,
    );
    // The edit is captured and the un-materialized paths survive.
    assert!(listing.contains("crates/foo/a.txt"), "{listing}");
    assert!(listing.contains("crates/bar/b.txt"), "{listing}");
    assert!(listing.contains("top.txt"), "{listing}");
}

#[test]
fn sparse_fork_rejects_a_malformed_prefix() {
    let temp = tempfile::tempdir().unwrap();
    let (dir, head) = sparse_repo(&temp);
    let repository = GitRepository::open(&dir).unwrap();
    let destination = temp.path().join("sparse");

    let error = repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &[])
        .unwrap_err();
    assert!(
        error.to_string().contains("at least one path prefix"),
        "{error}"
    );

    let error = repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &["/abs".to_string()])
        .unwrap_err();
    assert!(error.to_string().contains("relative path"), "{error}");

    let error = repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &["trailing/".to_string()])
        .unwrap_err();
    assert!(error.to_string().contains("relative path"), "{error}");
}

#[test]
fn sparse_fork_restore_tree_leaves_paths_outside_the_spec_alone() {
    let temp = tempfile::tempdir().unwrap();
    let (dir, head) = sparse_repo(&temp);
    let repository = GitRepository::open(&dir).unwrap();
    let destination = temp.path().join("sparse");
    repository
        .fork_workspace_sparse(&head, &destination, "thread-1", &["crates/foo".to_string()])
        .unwrap();

    let git_dir = repository.git_dir().to_path_buf();
    let tree_of = |rev: &str| {
        run(
            "git",
            &[
                "--git-dir",
                git_dir.to_str().unwrap(),
                "rev-parse",
                &format!("{rev}^{{tree}}"),
            ],
            &dir,
        )
    };
    let base_tree = tree_of(&head);

    // A target that changes a path inside the spec and one outside it.
    write_file(&dir.join("crates/foo/a.txt"), b"inside\n");
    write_file(&dir.join("crates/bar/b.txt"), b"outside\n");
    let target_tree = tree_of(&commit(&dir, "target"));

    let fork = GitRepository::open(&destination).unwrap();
    fork.restore_tree(&base_tree, &target_tree, &destination)
        .unwrap();

    // The in-spec path is updated; the out-of-spec path is never written, so the
    // workspace cannot diverge from what the next capture records.
    assert_eq!(
        fs::read(destination.join("crates/foo/a.txt")).unwrap(),
        b"inside\n"
    );
    assert!(
        !destination.join("crates/bar/b.txt").exists(),
        "restore must not write outside the sparse spec"
    );
}

#[test]
fn export_creates_commits_only_on_export_and_pushes() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("base.txt"), b"base\n");
    let base = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let base_tree = repository.snapshot_tree(&base).unwrap();

    // A live edit is captured as an immutable tree, not a commit.
    write_file(&dir.join("marked.txt"), b"marked\n");
    let marked_tree = repository.capture_tree(&base_tree).unwrap();
    assert_eq!(
        git_at(&git_dir, &["rev-list", "--all", "--count"]),
        "1",
        "capturing a snapshot must not create a live commit"
    );
    assert_eq!(git_at(&git_dir, &["rev-parse", "HEAD"]), base);

    let plan = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "marked".into(),
            tree: marked_tree.clone(),
            parents: Vec::new(),
            message: "marked".into(),
            signature: test_signature(),
        }],
        tip: "marked".into(),
    };

    let result = repository.export(&plan, "feature").unwrap();
    assert_eq!(result.branch, "feature");
    assert_eq!(result.tip, result.commits["marked"]);
    assert_eq!(
        git_at(&git_dir, &["rev-parse", "refs/heads/feature"]),
        result.tip
    );
    assert_eq!(parents_of(&git_dir, &result.tip), vec![base.clone()]);
    assert_eq!(
        git_at(
            &git_dir,
            &["rev-parse", &format!("{}^{{tree}}", result.tip)]
        ),
        marked_tree
    );
    // HEAD never moved.
    assert_eq!(git_at(&git_dir, &["rev-parse", "HEAD"]), base);

    // Deterministic identity.
    let header = git_at(&git_dir, &["cat-file", "-p", &result.tip]);
    let ident = "Juto Test <juto@test.local> 1700000000 +0000";
    assert!(header.contains(&format!("author {ident}")), "{header}");
    assert!(header.contains(&format!("committer {ident}")), "{header}");

    // Push to a bare remote, then refuse a non-fast-forward.
    let remote = temp.path().join("remote.git");
    run(
        "git",
        &["init", "--bare", "-q", remote.to_str().unwrap()],
        temp.path(),
    );
    repository
        .push("feature", remote.to_str().unwrap())
        .unwrap();
    assert_eq!(
        git_at(&remote, &["rev-parse", "refs/heads/feature"]),
        result.tip
    );

    write_file(&dir.join("other.txt"), b"other\n");
    let other_tree = repository.capture_tree(&base_tree).unwrap();
    let divergent = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "other".into(),
            tree: other_tree,
            parents: Vec::new(),
            message: "other".into(),
            signature: test_signature(),
        }],
        tip: "other".into(),
    };
    let error = repository.export(&divergent, "feature").unwrap_err();
    assert!(error.to_string().contains("not a fast-forward"), "{error}");
    assert_eq!(
        git_at(&git_dir, &["rev-parse", "refs/heads/feature"]),
        result.tip
    );
}

#[test]
fn export_root_zero_base_yields_parentless_commit() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("first.txt"), b"first\n");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let zero = "0".repeat(40);
    let tree = repository.capture_tree(&zero).unwrap();

    let plan = ExportPlan {
        base_commit: zero,
        points: vec![ExportPoint {
            id: "first".into(),
            tree,
            parents: Vec::new(),
            message: "first".into(),
            signature: test_signature(),
        }],
        tip: "first".into(),
    };
    let result = repository.export(&plan, "feature").unwrap();
    assert!(
        parents_of(&git_dir, &result.tip).is_empty(),
        "the first commit of a root-zero export must be parentless"
    );
}

#[test]
fn export_refuses_symbolic_destination_ref() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    let base = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = repository.snapshot_tree(&base).unwrap();

    // `refs/heads/alias` is a symbolic alias for the checked-out branch.
    git_at(
        &git_dir,
        &["symbolic-ref", "refs/heads/alias", "refs/heads/main"],
    );

    let plan = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "a".into(),
            tree,
            parents: Vec::new(),
            message: "a".into(),
            signature: test_signature(),
        }],
        tip: "a".into(),
    };
    let error = repository.export(&plan, "alias").unwrap_err();
    assert!(error.to_string().contains("symbolic"), "{error}");
    assert_eq!(
        git_at(&git_dir, &["symbolic-ref", "refs/heads/alias"]),
        "refs/heads/main"
    );
}

#[test]
fn export_refuses_checked_out_branch() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    let base = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let head = git_at(&git_dir, &["symbolic-ref", "-q", "HEAD"]);
    let checked_out = head.strip_prefix("refs/heads/").unwrap();

    let plan = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "a".into(),
            tree: repository.snapshot_tree(&base).unwrap(),
            parents: Vec::new(),
            message: "a".into(),
            signature: test_signature(),
        }],
        tip: "a".into(),
    };
    let error = repository.export(&plan, checked_out).unwrap_err();
    assert!(error.to_string().contains("checked out"), "{error}");
}

#[test]
fn push_resolves_relative_remote_from_workspace() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    let base = commit(&dir, "base");

    let remote = temp.path().join("remote.git");
    run(
        "git",
        &["init", "--bare", "-q", remote.to_str().unwrap()],
        temp.path(),
    );

    let repository = GitRepository::open(&dir).unwrap();
    let plan = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "a".into(),
            tree: repository.snapshot_tree(&base).unwrap(),
            parents: Vec::new(),
            message: "a".into(),
            signature: test_signature(),
        }],
        tip: "a".into(),
    };
    let result = repository.export(&plan, "feature").unwrap();

    // `../remote.git` only exists relative to the workspace root, never
    // relative to the test process directory.
    repository.push("feature", "../remote.git").unwrap();
    assert_eq!(
        git_at(&remote, &["rev-parse", "refs/heads/feature"]),
        result.tip
    );
}

#[test]
fn signature_reads_git_identity() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    let repository = GitRepository::open(&dir).unwrap();
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let signature = repository.signature().unwrap();
    let after = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    assert_eq!(signature.name, "Juto Test");
    assert_eq!(signature.email, "juto@test.local");
    assert!(
        signature.timestamp >= before && signature.timestamp <= after,
        "timestamp {} not in [{before}, {after}]",
        signature.timestamp
    );
}

#[test]
fn export_rejects_malformed_plans_and_unsafe_branches() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("base.txt"), b"base\n");
    let base = commit(&dir, "base");
    write_file(&dir.join("one.txt"), b"one\n");
    commit(&dir, "one");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = repository.snapshot_tree(&base).unwrap();
    let blob = git_at(&git_dir, &["rev-parse", "HEAD:one.txt"]);

    let point = |id: &str, parents: Vec<String>| ExportPoint {
        id: id.into(),
        tree: tree.clone(),
        parents,
        message: id.into(),
        signature: test_signature(),
    };

    let unknown_parent = ExportPlan {
        base_commit: base.clone(),
        points: vec![point("a", vec!["ghost".into()])],
        tip: "a".into(),
    };
    assert!(repository.export(&unknown_parent, "feature").is_err());

    let non_topological = ExportPlan {
        base_commit: base.clone(),
        points: vec![point("a", vec!["b".into()]), point("b", Vec::new())],
        tip: "b".into(),
    };
    assert!(repository.export(&non_topological, "feature").is_err());

    let duplicate = ExportPlan {
        base_commit: base.clone(),
        points: vec![point("a", Vec::new()), point("a", Vec::new())],
        tip: "a".into(),
    };
    assert!(repository.export(&duplicate, "feature").is_err());

    let unknown_tip = ExportPlan {
        base_commit: base.clone(),
        points: vec![point("a", Vec::new())],
        tip: "ghost".into(),
    };
    assert!(repository.export(&unknown_tip, "feature").is_err());

    let blob_tree = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "a".into(),
            tree: blob,
            parents: Vec::new(),
            message: "a".into(),
            signature: test_signature(),
        }],
        tip: "a".into(),
    };
    assert!(repository.export(&blob_tree, "feature").is_err());

    let bad_base = ExportPlan {
        base_commit: "not-a-commit".into(),
        points: vec![point("a", Vec::new())],
        tip: "a".into(),
    };
    assert!(repository.export(&bad_base, "feature").is_err());

    let valid = ExportPlan {
        base_commit: base.clone(),
        points: vec![point("a", Vec::new())],
        tip: "a".into(),
    };
    assert!(repository.export(&valid, "-evil").is_err());
    assert!(repository.export(&valid, "bad..name").is_err());
    assert!(repository.export(&valid, "bad name").is_err());
    assert!(repository.export(&valid, "@").is_err());

    // No malformed plan ever created the branch.
    assert_ne!(
        git_status(&git_dir, &["rev-parse", "--verify", "refs/heads/feature"]).code(),
        Some(0)
    );
}

#[test]
fn export_leaves_head_index_and_worktree_untouched() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("base.txt"), b"base\n");
    let base = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = repository.snapshot_tree(&base).unwrap();

    let plan = ExportPlan {
        base_commit: base.clone(),
        points: vec![ExportPoint {
            id: "marked".into(),
            tree,
            parents: Vec::new(),
            message: "marked".into(),
            signature: test_signature(),
        }],
        tip: "marked".into(),
    };

    let head_before = git_at(&git_dir, &["rev-parse", "HEAD"]);
    let status_before = git(&dir, &["status", "--porcelain"]);
    let index_before = git(&dir, &["ls-files", "--stage"]);
    let index_file_before = fs::read(git_dir.join("index")).unwrap();
    let worktree_before = fs::read(dir.join("base.txt")).unwrap();

    repository.export(&plan, "feature").unwrap();

    let index_file_after = fs::read(git_dir.join("index")).unwrap();
    let worktree_after = fs::read(dir.join("base.txt")).unwrap();
    let head_after = git_at(&git_dir, &["rev-parse", "HEAD"]);
    let status_after = git(&dir, &["status", "--porcelain"]);
    let index_after = git(&dir, &["ls-files", "--stage"]);

    assert_eq!(head_after, head_before, "HEAD must not move");
    assert_eq!(
        status_after, status_before,
        "worktree status must not change"
    );
    assert_eq!(index_after, index_before, "index entries must not change");
    assert_eq!(
        index_file_after, index_file_before,
        "index file must not be rewritten"
    );
    assert_eq!(
        worktree_after, worktree_before,
        "worktree contents must not change"
    );
}

/// Mirror of the pending-file naming in `git_transfer` so a test can plant a
/// crashed creator's registry entry at the exact path the implementation uses.
fn pending_name(destination: &Path) -> String {
    let name: String = destination
        .file_name()
        .unwrap()
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let absolute = std::path::absolute(destination).unwrap();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in absolute.as_os_str().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fork-{name}-{hash:016x}")
}

/// Write a pending-fork registry entry the way a crashed creator would.
fn plant_pending(git_dir: &Path, destination: &Path, thread: &str) -> PathBuf {
    let forks = git_dir.join("juto/forks");
    fs::create_dir_all(&forks).unwrap();
    let pending = forks.join(pending_name(destination));
    fs::write(
        &pending,
        format!(
            "{{\"destination\":\"{}\",\"thread\":\"{thread}\"}}",
            destination.display()
        ),
    )
    .unwrap();
    pending
}

#[test]
fn fork_workspace_streams_regular_exec_and_symlink_entries() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");

    // A large blob exercises the streaming path: content must round-trip
    // byte-for-byte without ever being buffered whole.
    let large: Vec<u8> = (0..2 * 1024 * 1024u32).map(|i| (i % 251) as u8).collect();
    write_file(&dir.join("large.bin"), &large);
    write_file(&dir.join("plain.txt"), b"plain\n");
    write_file(&dir.join("script.sh"), b"#!/bin/sh\nexit 0\n");
    set_executable(&dir.join("script.sh"));
    symlink("plain.txt", dir.join("alias")).unwrap();
    for i in 0..64 {
        write_file(
            &dir.join(format!("many/f{i}.txt")),
            format!("file {i}\n").as_bytes(),
        );
    }
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    let destination = temp.path().join("fork");
    repository
        .fork_workspace(&tree, &destination, "thread-1")
        .unwrap();

    assert_eq!(fs::read(destination.join("large.bin")).unwrap(), large);
    assert_eq!(fs::read(destination.join("plain.txt")).unwrap(), b"plain\n");
    assert_eq!(
        fs::read(destination.join("script.sh")).unwrap(),
        b"#!/bin/sh\nexit 0\n"
    );
    assert_eq!(
        fs::metadata(destination.join("script.sh"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o755
    );
    assert_eq!(
        fs::read_link(destination.join("alias")).unwrap(),
        PathBuf::from("plain.txt")
    );
    for i in 0..64 {
        assert_eq!(
            fs::read(destination.join(format!("many/f{i}.txt"))).unwrap(),
            format!("file {i}\n").as_bytes()
        );
    }
}

#[test]
fn release_snapshot_removes_ref_and_reports_absence() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    repository.retain_snapshot("keep", &tree).unwrap();
    assert_eq!(
        git_at(&git_dir, &["rev-parse", "refs/juto/snapshots/keep"]),
        tree
    );

    assert!(repository.release_snapshot("keep").unwrap());
    assert_eq!(
        git_status(
            &git_dir,
            &[
                "rev-parse",
                "--verify",
                "--quiet",
                "refs/juto/snapshots/keep"
            ]
        )
        .code(),
        Some(1),
        "the snapshot ref must be gone"
    );
    // Releasing again reports that nothing was there.
    assert!(!repository.release_snapshot("keep").unwrap());

    // The id is validated exactly like `retain_snapshot`.
    assert!(repository.release_snapshot("").is_err());
    assert!(repository.release_snapshot("nested/id").is_err());
    assert!(repository.release_snapshot("../escape").is_err());
}

#[test]
fn cleanup_orphaned_forks_reclaims_crashed_fork_and_allows_retry() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // Simulate a crash between `create_dir` and the marker write: a partial
    // directory plus a registry entry, no marker.
    let destination = temp.path().join("fork");
    fs::create_dir_all(destination.join("partial")).unwrap();
    write_file(&destination.join("partial/left.txt"), b"left\n");
    let pending = plant_pending(&git_dir, &destination, "thread-1");

    let reclaimed = repository.cleanup_orphaned_forks().unwrap();
    assert_eq!(reclaimed, vec![destination.clone()]);
    assert!(
        !destination.exists(),
        "the orphaned directory must be removed"
    );
    assert!(!pending.exists(), "the pending record must be removed");

    // A fresh fork at the reclaimed destination now succeeds.
    repository
        .fork_workspace(&tree, &destination, "thread-1")
        .unwrap();
    assert_eq!(fs::read(destination.join("a.txt")).unwrap(), b"a\n");

    // A repository with no registry directory has nothing to reclaim.
    let fresh = repo(&temp, "fresh");
    let fresh_repository = GitRepository::open(&fresh).unwrap();
    assert!(
        fresh_repository
            .cleanup_orphaned_forks()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn fork_workspace_reclaims_registered_orphan() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // A crashed creator left a partial directory and its registry entry.
    let destination = temp.path().join("fork");
    fs::create_dir_all(destination.join("partial")).unwrap();
    write_file(&destination.join("partial/left.txt"), b"left\n");
    let pending = plant_pending(&git_dir, &destination, "thread-1");

    // The retry reclaims the orphan in place and materializes the tree.
    repository
        .fork_workspace(&tree, &destination, "thread-1")
        .unwrap();
    assert_eq!(fs::read(destination.join("a.txt")).unwrap(), b"a\n");
    assert!(
        !destination.join("partial").exists(),
        "the orphaned contents must be gone"
    );
    assert!(destination.join(".juto-workspace.json").exists());
    assert!(
        !pending.exists(),
        "the pending record must be cleared on success"
    );
}

#[test]
fn cleanup_orphaned_forks_preserves_completed_fork() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // A completed fork whose pending delete was lost: the marker is present.
    let destination = temp.path().join("fork");
    repository
        .fork_workspace(&tree, &destination, "thread-1")
        .unwrap();
    let pending = plant_pending(&git_dir, &destination, "thread-1");

    let reclaimed = repository.cleanup_orphaned_forks().unwrap();
    assert!(reclaimed.is_empty(), "a completed fork is not reclaimed");
    assert!(destination.exists(), "a completed fork must be preserved");
    assert!(
        !pending.exists(),
        "the stale pending record must still be cleared"
    );
}

#[test]
fn cleanup_orphaned_forks_skips_live_fork() {
    use fs2::FileExt;

    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    let destination = temp.path().join("fork");
    fs::create_dir_all(&destination).unwrap();
    let pending = plant_pending(&git_dir, &destination, "thread-1");

    // Hold the exclusive lock the way a live fork would.
    let live = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&pending)
        .unwrap();
    live.lock_exclusive().unwrap();

    let reclaimed = repository.cleanup_orphaned_forks().unwrap();
    assert!(reclaimed.is_empty(), "a live fork must not be reclaimed");
    assert!(
        destination.exists(),
        "a live fork's directory must be preserved"
    );
    assert!(
        pending.exists(),
        "a live fork's pending record must be preserved"
    );

    // Once the creator is gone, the same record is reclaimed.
    //
    // The reclaim is retried briefly. Releasing an `flock` and observing it as
    // free from a fresh descriptor is not instantaneous in this environment: the
    // descriptor is gone from every process (checked via /proc) and the lock is
    // still reported held for a moment, so a single immediate attempt is flaky.
    // In production this only defers an orphan's cleanup to the next fork.
    drop(live);
    let mut reclaimed = Vec::new();
    for _ in 0..100 {
        reclaimed = repository.cleanup_orphaned_forks().unwrap();
        if !reclaimed.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(reclaimed, vec![destination.clone()]);
    assert!(!destination.exists());
    assert!(!pending.exists());
}

#[test]
fn update_tree_entry_matches_a_full_capture() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    fs::create_dir_all(dir.join("src")).unwrap();
    write_file(&dir.join("src/a.txt"), b"a\n");
    write_file(&dir.join("src/b.txt"), b"b\n");
    write_file(&dir.join("top.txt"), b"top\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let head = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // A one-entry delta must equal what a full workspace scan would produce.
    write_file(&dir.join("src/a.txt"), b"edited\n");
    let blob = repository.write_blob(b"edited\n").unwrap();
    let delta = repository
        .update_tree_entry(&head, b"src/a.txt", "100644", &blob)
        .unwrap();
    assert_eq!(
        delta,
        repository.capture_tree(&head).unwrap(),
        "a one-entry delta must match a full capture"
    );

    // The exec bit is part of the entry. Base the delta on the previous result:
    // a full capture sees every accumulated change.
    let mut perms = fs::metadata(dir.join("src/b.txt")).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(dir.join("src/b.txt"), perms).unwrap();
    let blob = repository.write_blob(b"b\n").unwrap();
    let delta = repository
        .update_tree_entry(&delta, b"src/b.txt", "100755", &blob)
        .unwrap();
    assert_eq!(
        delta,
        repository.capture_tree(&head).unwrap(),
        "the exec bit must be recorded"
    );

    // Mode 0 removes the entry.
    let delta = repository
        .update_tree_entry(&delta, b"top.txt", "0", &"0".repeat(40))
        .unwrap();
    fs::remove_file(dir.join("top.txt")).unwrap();
    assert_eq!(
        delta,
        repository.capture_tree(&head).unwrap(),
        "mode 0 must remove the entry"
    );
}

#[test]
fn update_refs_applies_a_batch_and_enforces_compare_and_swap() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");
    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();

    let one = repository.write_blob(b"one\n").unwrap();
    let two = repository.write_blob(b"two\n").unwrap();

    repository
        .update_refs(&[
            ("refs/juto/nodes/n1".to_string(), one.clone(), None),
            ("refs/juto/nodes/n2".to_string(), two.clone(), None),
        ])
        .unwrap();
    assert_eq!(git_at(&git_dir, &["rev-parse", "refs/juto/nodes/n1"]), one);
    assert_eq!(git_at(&git_dir, &["rev-parse", "refs/juto/nodes/n2"]), two);

    // A create-only update on an existing ref fails, and leaves it alone.
    assert!(
        repository
            .update_refs(&[("refs/juto/nodes/n1".to_string(), two.clone(), None)])
            .is_err(),
        "create-only must refuse an existing ref"
    );
    assert_eq!(git_at(&git_dir, &["rev-parse", "refs/juto/nodes/n1"]), one);

    // A compare-and-swap with the right old value moves it.
    repository
        .update_refs(&[(
            "refs/juto/nodes/n1".to_string(),
            two.clone(),
            Some(one.clone()),
        )])
        .unwrap();
    assert_eq!(git_at(&git_dir, &["rev-parse", "refs/juto/nodes/n1"]), two);
}

#[test]
fn refs_transport_blobs_and_trees_over_a_bare_remote() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let tree = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // A local bare remote.
    let remote = temp.path().join("remote.git");
    run(
        "git",
        &["init", "-q", "--bare", remote.to_str().unwrap()],
        temp.path(),
    );

    // Write a blob and pin a tree under the wire-format refs.
    let blob = repository.write_blob(b"{\"node\":1}").unwrap();
    assert_eq!(repository.read_blob(&blob).unwrap(), b"{\"node\":1}");
    assert!(
        repository
            .update_ref("refs/juto/nodes/node-1", &blob, None)
            .unwrap()
    );
    assert!(
        repository
            .update_ref("refs/juto/snapshots/snap-1", &tree, None)
            .unwrap()
    );

    let refspecs = vec![
        "refs/juto/nodes/node-1:refs/juto/nodes/node-1".to_string(),
        "refs/juto/snapshots/snap-1:refs/juto/snapshots/snap-1".to_string(),
    ];
    repository
        .push_refs(remote.to_str().unwrap(), &refspecs)
        .unwrap();

    // The remote advertises both refs with the right object types.
    let remote_refs = git_at(
        &remote,
        &[
            "for-each-ref",
            "--format=%(refname) %(objectname) %(objecttype)",
        ],
    );
    assert!(
        remote_refs.contains(&format!("refs/juto/nodes/node-1 {blob} blob")),
        "the blob ref must transport as a blob: {remote_refs}"
    );
    assert!(
        remote_refs.contains(&format!("refs/juto/snapshots/snap-1 {tree} tree")),
        "the tree ref must transport as a tree: {remote_refs}"
    );

    // Fetch into a second repository and read the refs back.
    let dst = repo(&temp, "dst");
    let dst_repo = GitRepository::open(&dst).unwrap();
    dst_repo.fetch(remote.to_str().unwrap(), &refspecs).unwrap();
    let fetched = dst_repo.list_refs("refs/juto/").unwrap();
    assert!(fetched.contains(&("refs/juto/nodes/node-1".to_string(), blob.clone())));
    assert!(fetched.contains(&("refs/juto/snapshots/snap-1".to_string(), tree.clone())));
    assert_eq!(dst_repo.read_blob(&blob).unwrap(), b"{\"node\":1}");
    let dst_git_dir = dst_repo.git_dir().to_path_buf();
    assert_eq!(git_at(&dst_git_dir, &["cat-file", "-t", &tree]), "tree");
}

#[test]
fn update_ref_enforces_compare_and_swap() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    commit(&dir, "base");
    let repository = GitRepository::open(&dir).unwrap();

    let one = repository.write_blob(b"one").unwrap();
    let two = repository.write_blob(b"two").unwrap();

    // Create-only succeeds when absent and fails when present.
    assert!(
        repository
            .update_ref("refs/juto/nodes/n1", &one, None)
            .unwrap()
    );
    assert!(
        repository
            .update_ref("refs/juto/nodes/n1", &one, None)
            .is_err()
    );

    // Some(old) fails on a mismatch and succeeds on a match.
    assert!(
        repository
            .update_ref("refs/juto/nodes/n1", &two, Some(&one))
            .unwrap()
    );
    assert!(
        repository
            .update_ref("refs/juto/nodes/n1", &one, Some(&one))
            .is_err()
    );

    // An identical value is a no-op that reports no change.
    assert!(
        !repository
            .update_ref("refs/juto/nodes/n1", &two, Some(&two))
            .unwrap()
    );

    // Some(old) on a missing ref fails.
    assert!(
        repository
            .update_ref("refs/juto/nodes/n2", &one, Some(&one))
            .is_err()
    );

    // The ref resolves to the last written value.
    assert_eq!(
        git_at(repository.git_dir(), &["rev-parse", "refs/juto/nodes/n1"]),
        two
    );
}

#[test]
fn restore_tree_applies_delta_into_existing_directory() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("keep.txt"), b"keep\n");
    write_file(&dir.join("mod.txt"), b"old\n");
    write_file(&dir.join("del.txt"), b"delete me\n");
    write_file(&dir.join("script.sh"), b"#!/bin/sh\necho hi\n");
    write_file(&dir.join("unrelated.txt"), b"unrelated\n");
    symlink("keep.txt", dir.join("link")).unwrap();
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let base = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // Target: modify, delete, add, chmod +x, and add a symlink.
    write_file(&dir.join("mod.txt"), b"new\n");
    fs::remove_file(dir.join("del.txt")).unwrap();
    write_file(&dir.join("added.txt"), b"added\n");
    set_executable(&dir.join("script.sh"));
    symlink("mod.txt", dir.join("newlink")).unwrap();
    commit(&dir, "target");
    let target = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // Materialize the base tree into a fresh directory, then apply the delta.
    let work = temp.path().join("work");
    repository.fork_workspace(&base, &work, "thread-1").unwrap();
    assert_eq!(fs::read(work.join("mod.txt")).unwrap(), b"old\n");
    assert!(work.join("del.txt").exists());

    repository.restore_tree(&base, &target, &work).unwrap();

    assert_eq!(fs::read(work.join("mod.txt")).unwrap(), b"new\n");
    assert_eq!(fs::read(work.join("added.txt")).unwrap(), b"added\n");
    assert!(
        !work.join("del.txt").exists(),
        "a deleted path must be removed"
    );
    assert!(
        fs::metadata(work.join("script.sh"))
            .unwrap()
            .permissions()
            .mode()
            & 0o111
            != 0,
        "the executable bit must be applied"
    );
    assert_eq!(
        fs::read_link(work.join("newlink")).unwrap(),
        Path::new("mod.txt")
    );
    // Unrelated paths are untouched.
    assert_eq!(fs::read(work.join("keep.txt")).unwrap(), b"keep\n");
    assert_eq!(
        fs::read(work.join("unrelated.txt")).unwrap(),
        b"unrelated\n"
    );
    assert_eq!(
        fs::read_link(work.join("link")).unwrap(),
        Path::new("keep.txt")
    );
}

#[test]
fn merge_trees_merges_cleanly_and_reports_conflicts() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("f1.txt"), b"base\n");
    write_file(&dir.join("f2.txt"), b"base\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let base = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // Ours changes f1 only.
    write_file(&dir.join("f1.txt"), b"ours\n");
    commit(&dir, "ours");
    let ours = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    // Theirs (from base) changes f2 only: a disjoint, clean merge.
    git(&dir, &["checkout", "-q", "-b", "other", "HEAD~1"]);
    write_file(&dir.join("f2.txt"), b"theirs\n");
    commit(&dir, "theirs");
    let theirs = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    let clean = repository.merge_trees(&base, &ours, &theirs).unwrap();
    assert!(clean.conflicts.is_empty(), "a disjoint merge must be clean");
    let entries = tree_entries(&git_dir, &clean.tree);
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["f1.txt"].1]),
        b"ours\n"
    );
    assert_eq!(
        git_raw(&git_dir, &["cat-file", "blob", &entries["f2.txt"].1]),
        b"theirs\n"
    );

    // Now both sides change f1: a conflict that still yields a tree.
    git(&dir, &["checkout", "-q", "main"]);
    write_file(&dir.join("f1.txt"), b"ours2\n");
    commit(&dir, "ours2");
    let ours2 = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);
    git(&dir, &["checkout", "-q", "other"]);
    write_file(&dir.join("f1.txt"), b"theirs2\n");
    commit(&dir, "theirs2");
    let theirs2 = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    let conflicted = repository.merge_trees(&base, &ours2, &theirs2).unwrap();
    assert_eq!(conflicted.conflicts, vec!["f1.txt".to_string()]);
    assert!(
        tree_entries(&git_dir, &conflicted.tree).contains_key("f1.txt"),
        "a conflicting merge still returns a merged tree"
    );
}

#[test]
fn supports_tree_merge_reports_a_bool() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    let repository = GitRepository::open(&dir).unwrap();
    // The probe must not panic; this crate requires Git >= 2.45.
    assert!(repository.supports_tree_merge().unwrap());
}

#[test]
fn restore_tree_does_not_replace_through_a_symlinked_parent() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("nested/inner.txt"), b"base\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let base = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);
    write_file(&dir.join("nested/inner.txt"), b"target\n");
    commit(&dir, "target");
    let target = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    let work = temp.path().join("work");
    repository.fork_workspace(&base, &work, "thread-1").unwrap();
    let outside = temp.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    write_file(&outside.join("inner.txt"), b"outside\n");
    fs::remove_dir_all(work.join("nested")).unwrap();
    symlink(&outside, work.join("nested")).unwrap();

    let error = repository.restore_tree(&base, &target, &work).unwrap_err();
    assert!(error.to_string().contains("symlink"), "{error}");
    assert_eq!(
        fs::read(outside.join("inner.txt")).unwrap(),
        b"outside\n",
        "restore must not remove or replace a file through a symlinked parent"
    );
}

#[test]
fn restore_tree_does_not_delete_through_a_symlinked_parent() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("nested/gone.txt"), b"base\n");
    commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let base = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);
    fs::remove_file(dir.join("nested/gone.txt")).unwrap();
    commit(&dir, "target");
    let target = git_at(&git_dir, &["rev-parse", "HEAD^{tree}"]);

    let work = temp.path().join("work");
    repository.fork_workspace(&base, &work, "thread-1").unwrap();
    let outside = temp.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    write_file(&outside.join("gone.txt"), b"outside\n");
    fs::remove_dir_all(work.join("nested")).unwrap();
    symlink(&outside, work.join("nested")).unwrap();

    let error = repository.restore_tree(&base, &target, &work).unwrap_err();
    assert!(error.to_string().contains("symlink"), "{error}");
    assert_eq!(
        fs::read(outside.join("gone.txt")).unwrap(),
        b"outside\n",
        "restore must not delete through a symlinked parent"
    );
}
