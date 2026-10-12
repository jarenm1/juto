//! Behavioral integration tests for [`juto_runtime::git_transfer`].
//!
//! Every test drives the real `git` binary against temporary plain Git
//! repositories and a local bare remote. Nothing is mocked: snapshots are
//! inspected with Git plumbing, and branch/remote refs are read back from disk.
//! No Jujutsu binary is involved.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
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
    assert!(repository.path_tracked(&after, "a.txt").unwrap());
    assert!(repository.path_tracked(&after, "b.txt").unwrap());
    assert!(!repository.path_tracked(&after, "missing.txt").unwrap());
    assert!(!repository.path_tracked(&before, "b.txt").unwrap());

    let patch = repository.diff_trees(&before, &after, "a.txt").unwrap();
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
    let wild = repository.diff_trees(&before, &after, "*.txt").unwrap();
    assert!(wild.contains("*.txt"), "{wild}");
    assert!(
        !wild.contains("a.txt"),
        "a literal wildcard must not match a.txt: {wild}"
    );

    let exact = repository.diff_trees(&before, &after, "a.txt").unwrap();
    assert!(exact.contains("a.txt"), "{exact}");
    assert!(!exact.contains("*.txt"), "{exact}");

    assert!(repository.path_tracked(&after, "*.txt").unwrap());
    assert!(repository.path_tracked(&after, "a.txt").unwrap());
    // `a*.txt` is not a file; a glob would have matched `a.txt`.
    assert!(!repository.path_tracked(&after, "a*.txt").unwrap());
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
    assert!(repository.path_ignored("a.log").unwrap());
    assert!(!repository.path_ignored("b.txt").unwrap());
    assert!(!repository.path_ignored("missing.txt").unwrap());
    assert!(repository.path_ignored("").is_err());

    // A tracked file that matches an ignore rule still reports as ignored.
    git(&dir, &["add", "-f", "a.log"]);
    git(&dir, &["commit", "-q", "-m", "force tracked"]);
    assert!(repository.path_ignored("a.log").unwrap());
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
fn fork_workspace_refuses_gitlink() {
    let temp = tempfile::tempdir().unwrap();
    let dir = repo(&temp, "repo");
    write_file(&dir.join("a.txt"), b"a\n");
    let head = commit(&dir, "base");

    let repository = GitRepository::open(&dir).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let gitlink = run_stdin(
        "git",
        &["--git-dir", git_dir.to_str().unwrap(), "mktree"],
        &dir,
        format!("160000 commit {head}\tsub\n").as_bytes(),
    );

    let destination = temp.path().join("fork");
    let error = repository
        .fork_workspace(&gitlink, &destination, "thread-1")
        .unwrap_err();
    assert!(error.to_string().contains("submodule"), "{error}");
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
