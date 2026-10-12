//! Behavioral integration tests for Juto's DAG replication over Git transport.
//!
//! Every test drives the real `git` binary against temporary repositories and a
//! local bare remote. Nothing is mocked: the wire refs are inspected with Git
//! plumbing and the imported journal is read back through the public API.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use juto_runtime::change_graph::{ChangeGraph, ChangeNode, EditProvenance, ReconcileOutcome};
use juto_runtime::git_transfer::GitRepository;
use tempfile::TempDir;

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

fn git(cwd: &Path, args: &[&str]) -> String {
    run("git", args, cwd)
}

/// Run `git` against an explicit git directory.
fn git_at(git_dir: &Path, args: &[&str]) -> String {
    let mut full = vec!["--git-dir", git_dir.to_str().unwrap()];
    full.extend_from_slice(args);
    run("git", &full, git_dir)
}

fn init_repo(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    git(dir, &["init", "-q", "-b", "main"]);
    git(dir, &["config", "user.name", "Juto Test"]);
    git(dir, &["config", "user.email", "juto@test.local"]);
}

fn provenance(session: &str) -> EditProvenance {
    EditProvenance {
        session_id: session.into(),
        tool: "edit".into(),
        selection_mode: None,
        path: None,
    }
}

/// Two repositories cloned from one bare remote, so they share a base commit.
struct SharedRemote {
    _temp: TempDir,
    remote: PathBuf,
    a: PathBuf,
    b: PathBuf,
}

impl SharedRemote {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let remote = temp.path().join("remote.git");
        run(
            "git",
            &[
                "init",
                "--bare",
                "-q",
                "-b",
                "main",
                remote.to_str().unwrap(),
            ],
            temp.path(),
        );
        let a = temp.path().join("a");
        init_repo(&a);
        fs::write(a.join("base"), "base\n").unwrap();
        git(&a, &["add", "-A"]);
        git(&a, &["commit", "-q", "-m", "base"]);
        git(&a, &["remote", "add", "origin", remote.to_str().unwrap()]);
        git(&a, &["push", "-q", "origin", "main"]);
        let b = temp.path().join("b");
        run(
            "git",
            &["clone", "-q", remote.to_str().unwrap(), b.to_str().unwrap()],
            temp.path(),
        );
        git(&b, &["config", "user.name", "Juto Test"]);
        git(&b, &["config", "user.email", "juto@test.local"]);
        Self {
            _temp: temp,
            remote,
            a,
            b,
        }
    }

    fn remote(&self) -> &str {
        self.remote.to_str().unwrap()
    }
}

#[test]
fn publish_and_fetch_converge_nodes_heads_and_checkpoints() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    fs::write(fixture.a.join("one"), "one\n").unwrap();
    let first = a.capture("main", provenance("one"), None).unwrap();
    a.mark(&first.id, "c1", "feat: one", true).unwrap();
    fs::write(fixture.a.join("two"), "two\n").unwrap();
    let second = a.capture("main", provenance("two"), None).unwrap();

    let published = a.push_graph(fixture.remote()).unwrap();
    assert_eq!(published.nodes, 2);
    assert_eq!(published.heads, 1);
    assert_eq!(published.checkpoints, 1);
    assert!(published.snapshots >= 2);

    let imported = b.fetch_graph(fixture.remote()).unwrap();
    assert_eq!(imported.nodes_imported, 2);
    assert_eq!(imported.nodes_unchanged, 0);
    assert_eq!(imported.heads_imported, 1);
    assert_eq!(imported.checkpoints_imported, 1);
    assert!(imported.head_conflicts.is_empty());
    assert!(imported.checkpoint_conflicts.is_empty());

    let owner = a.machine_id().unwrap();
    let state_a = a.state().unwrap();
    let state_b = b.state().unwrap();
    assert_eq!(state_b.nodes, state_a.nodes);
    assert_eq!(state_b.heads[&format!("{owner}/main")], second.id);
    assert_eq!(state_b.checkpoints["c1"], first.id);

    // A second import imports nothing and does not rewrite the journal.
    let journal = GitRepository::open(&fixture.b)
        .unwrap()
        .git_dir()
        .join("juto/graph.json");
    let before = fs::read(&journal).unwrap();
    let again = b.fetch_graph(fixture.remote()).unwrap();
    assert_eq!(again.nodes_imported, 0);
    assert_eq!(again.nodes_unchanged, 2);
    assert_eq!(again.heads_imported, 0);
    assert_eq!(again.checkpoints_imported, 0);
    assert!(again.head_conflicts.is_empty());
    assert!(again.checkpoint_conflicts.is_empty());
    assert_eq!(
        fs::read(&journal).unwrap(),
        before,
        "an idempotent import must not save the journal"
    );
    assert_eq!(b.state().unwrap().nodes, state_a.nodes);
}

#[test]
fn a_node_with_different_content_is_a_hard_error() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    fs::write(fixture.a.join("one"), "one\n").unwrap();
    let node = a.capture("main", provenance("one"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();
    b.fetch_graph(fixture.remote()).unwrap();

    // Craft a staging ref for the same id carrying different content.
    let repository = GitRepository::open(&fixture.b).unwrap();
    let git_dir = repository.git_dir().to_path_buf();
    let oid = git_at(
        &git_dir,
        &["rev-parse", &format!("refs/juto/nodes/{}", node.id)],
    );
    let bytes = repository.read_blob(&oid).unwrap();
    let mut tampered: ChangeNode = serde_json::from_slice(&bytes).unwrap();
    tampered.provenance.tool = "tampered".into();
    let blob = repository
        .write_blob(&serde_json::to_vec(&tampered).unwrap())
        .unwrap();
    let staging = "refs/juto/incoming/manual";
    repository
        .update_ref(&format!("{staging}/nodes/{}", node.id), &blob, None)
        .unwrap();
    let tree = git_at(
        &git_dir,
        &["rev-parse", &format!("refs/juto/snapshots/{}", node.id)],
    );
    repository
        .update_ref(&format!("{staging}/snapshots/{}", node.id), &tree, None)
        .unwrap();

    let error = b.import_refs(staging).unwrap_err();
    assert!(
        error.to_string().contains("different content"),
        "unexpected error: {error}"
    );
    assert_eq!(b.state().unwrap().nodes, a.state().unwrap().nodes);
}

#[test]
fn a_missing_snapshot_ref_aborts_the_import() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    fs::write(fixture.a.join("one"), "one\n").unwrap();
    let node = a.capture("main", provenance("one"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    // Drop the tree pin on the remote: the node ref arrives without its tree.
    git_at(
        &fixture.remote,
        &[
            "update-ref",
            "-d",
            &format!("refs/juto/snapshots/{}", node.id),
        ],
    );

    let error = b.fetch_graph(fixture.remote()).unwrap_err();
    assert!(
        error.to_string().contains("snapshot"),
        "unexpected error: {error}"
    );
    assert!(b.state().unwrap().nodes.is_empty());
}

#[test]
fn concurrent_heads_on_the_same_thread_conflict_without_losing_history() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    // Both repositories act as the same owner identity (one agent identity
    // replicated to two machines), so their "main" heads collide.
    let owner = a.machine_id().unwrap();
    let b_state_dir = GitRepository::open(&fixture.b)
        .unwrap()
        .git_dir()
        .join("juto");
    fs::write(b_state_dir.join("machine-id"), format!("{owner}\n")).unwrap();
    assert_eq!(b.machine_id().unwrap(), owner);

    fs::write(fixture.a.join("a-file"), "a\n").unwrap();
    let head_a = a.capture("main", provenance("a"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    // B adopts A's published head, then advances "main" itself.
    b.fetch_graph(fixture.remote()).unwrap();
    fs::write(fixture.b.join("b-file"), "b\n").unwrap();
    let head_b = b.capture("main", provenance("b"), None).unwrap();

    let imported = b.fetch_graph(fixture.remote()).unwrap();
    assert_eq!(imported.head_conflicts.len(), 1);
    let conflict = &imported.head_conflicts[0];
    assert_eq!(conflict.owner, owner);
    assert_eq!(conflict.thread, "main");
    assert_eq!(conflict.local, head_b.id);
    assert_eq!(conflict.remote, head_a.id);

    let state = b.state().unwrap();
    assert_eq!(
        state.heads["main"], head_b.id,
        "the local head must be kept"
    );
    assert_eq!(state.heads[&format!("{owner}/main")], head_a.id);
}

#[test]
fn republishing_advances_the_owner_head_ref() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();

    fs::write(fixture.a.join("one"), "one\n").unwrap();
    let first = a.capture("main", provenance("one"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    let owner = a.machine_id().unwrap();
    let head_ref = format!("refs/juto/heads/{owner}/main");
    let head_blob = git_at(&fixture.remote, &["rev-parse", &head_ref]);
    assert_eq!(
        git_at(&fixture.remote, &["cat-file", "blob", &head_blob]),
        first.id
    );

    // Publishing again after the head advanced must move the owner's ref, not
    // fail on a create-only update.
    fs::write(fixture.a.join("two"), "two\n").unwrap();
    let second = a.capture("main", provenance("two"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();
    let head_blob = git_at(&fixture.remote, &["rev-parse", &head_ref]);
    assert_eq!(
        git_at(&fixture.remote, &["cat-file", "blob", &head_blob]),
        second.id
    );
}

#[test]
fn gc_releases_node_refs_for_pruned_nodes() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let root = a.capture("main", provenance("root"), None).unwrap();
    a.mark(&root.id, "base", "feat: base", true).unwrap();

    // A side thread records a point, then re-bases onto the checkpoint,
    // abandoning its earlier tip.
    let workspaces = tempfile::tempdir().unwrap();
    let side_dir = workspaces.path().join("side");
    a.fork("side", "base", &side_dir).unwrap();
    let side = ChangeGraph::open(&side_dir).unwrap();
    fs::write(side_dir.join("side"), "side\n").unwrap();
    let abandoned = side.capture("side", provenance("side"), None).unwrap();
    fs::write(side_dir.join("side"), "side rebased\n").unwrap();
    side.capture("side", provenance("side"), Some(vec![root.id.clone()]))
        .unwrap();

    // Publishing pins `refs/juto/nodes/<id>` for every node, including the one
    // about to be abandoned.
    a.push_graph(fixture.remote()).unwrap();
    let git_dir = GitRepository::open(&fixture.a)
        .unwrap()
        .git_dir()
        .to_path_buf();
    let abandoned_ref = format!("refs/juto/nodes/{}", abandoned.id);
    assert!(
        !git_at(
            &git_dir,
            &["rev-parse", "--verify", "--quiet", &abandoned_ref]
        )
        .is_empty(),
        "publishing must pin refs/juto/nodes/<id>"
    );

    let report = a.gc().unwrap();
    assert_eq!(report.pruned_nodes, 1);
    assert_eq!(report.released_refs, 1);
    assert_eq!(report.released_node_refs, 1);

    // The pruned node leaves no replication ref behind.
    let remaining = git_at(
        &git_dir,
        &["for-each-ref", "--format=%(refname)", "refs/juto/nodes/"],
    );
    assert!(
        !remaining.lines().any(|line| line == abandoned_ref),
        "gc must release the pruned node's refs/juto/nodes ref, got:\n{remaining}"
    );
    assert!(remaining.contains(&format!("refs/juto/nodes/{}", root.id)));
}

#[test]
fn a_failed_import_leaves_no_staging_refs() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    fs::write(fixture.a.join("one"), "one\n").unwrap();
    a.capture("main", provenance("one"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    // A repository with an unrelated base cannot import: the base guard aborts.
    let temp = tempfile::tempdir().unwrap();
    let c = temp.path().join("c");
    init_repo(&c);
    fs::write(c.join("other"), "other\n").unwrap();
    git(&c, &["add", "-A"]);
    git(&c, &["commit", "-q", "-m", "other"]);
    let graph_c = ChangeGraph::open(&c).unwrap();

    let error = graph_c.fetch_graph(fixture.remote()).unwrap_err();
    assert!(
        error.to_string().contains("does not match the local base"),
        "unexpected error: {error}"
    );

    // The staging namespace must be empty: a failed import may not leak refs.
    let git_dir = GitRepository::open(&c).unwrap().git_dir().to_path_buf();
    let staging = git_at(
        &git_dir,
        &["for-each-ref", "--format=%(refname)", "refs/juto/incoming/"],
    );
    assert!(
        staging.is_empty(),
        "a failed import must clean its staging refs, got:\n{staging}"
    );
}

#[test]
fn reconcile_joins_a_conflicting_imported_head() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    // Both machines advance "main" from the shared base, touching different
    // files, so the two heads merge cleanly.
    fs::write(fixture.a.join("a.txt"), "a\n").unwrap();
    let a_head = a.capture("main", provenance("a"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    fs::write(fixture.b.join("b.txt"), "b\n").unwrap();
    let b_head = b.capture("main", provenance("b"), None).unwrap();

    let imported = b.fetch_graph(fixture.remote()).unwrap();
    assert_eq!(imported.heads_imported, 1);
    let remote_head = format!("{}/main", a.machine_id().unwrap());
    let state = b.state().unwrap();
    assert_eq!(state.heads[&remote_head], a_head.id);
    assert_eq!(state.heads["main"], b_head.id);

    let outcomes = b.reconcile("main").unwrap();
    assert_eq!(outcomes.len(), 1);
    let ReconcileOutcome::Joined { remote, node, .. } = &outcomes[0] else {
        panic!("expected a clean join, got {:?}", outcomes[0]);
    };
    assert_eq!(remote, &remote_head);
    assert_eq!(node.parents.len(), 2);
    assert!(node.parents.contains(&a_head.id));
    assert!(node.parents.contains(&b_head.id));

    let state = b.state().unwrap();
    assert!(
        !state.heads.contains_key(&remote_head),
        "a clean reconcile must retire the imported head"
    );
    assert_eq!(state.heads["main"], node.id);
    // Both machines' work is present in the reconciled workspace.
    assert!(fixture.b.join("a.txt").exists());
    assert!(fixture.b.join("b.txt").exists());
}

#[test]
fn reconcile_reports_a_conflicting_merge_and_changes_nothing() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    // Both machines edit the same file differently: the merge must conflict.
    fs::write(fixture.a.join("shared"), "from a\n").unwrap();
    let a_head = a.capture("main", provenance("a"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    fs::write(fixture.b.join("shared"), "from b\n").unwrap();
    let b_head = b.capture("main", provenance("b"), None).unwrap();

    b.fetch_graph(fixture.remote()).unwrap();
    let remote_head = format!("{}/main", a.machine_id().unwrap());

    let outcomes = b.reconcile("main").unwrap();
    assert_eq!(outcomes.len(), 1);
    let ReconcileOutcome::Conflicted {
        remote, conflicts, ..
    } = &outcomes[0]
    else {
        panic!("expected a conflict, got {:?}", outcomes[0]);
    };
    assert_eq!(remote, &remote_head);
    assert_eq!(conflicts, &vec!["shared".to_string()]);

    // Nothing changed: both heads survive and the workspace is untouched.
    let state = b.state().unwrap();
    assert_eq!(state.heads[&remote_head], a_head.id);
    assert_eq!(state.heads["main"], b_head.id);
    assert_eq!(
        fs::read_to_string(fixture.b.join("shared")).unwrap(),
        "from b\n"
    );
}

#[test]
fn reconcile_adopts_a_remote_head_when_the_thread_has_no_local_head() {
    let fixture = SharedRemote::new();
    let a = ChangeGraph::open(&fixture.a).unwrap();
    let b = ChangeGraph::open(&fixture.b).unwrap();

    fs::write(fixture.a.join("a.txt"), "a\n").unwrap();
    let a_head = a.capture("main", provenance("a"), None).unwrap();
    a.push_graph(fixture.remote()).unwrap();

    // B has never captured on "main", so the import lands only under the
    // owner-qualified key.
    b.fetch_graph(fixture.remote()).unwrap();
    let remote_head = format!("{}/main", a.machine_id().unwrap());
    let state = b.state().unwrap();
    assert!(state.heads.contains_key(&remote_head));
    assert!(!state.heads.contains_key("main"));
    assert_eq!(
        b.pending_reconcile_threads().unwrap(),
        vec!["main".to_string()],
        "the imported head must be reported as a pending thread"
    );

    let outcomes = b.reconcile("main").unwrap();
    assert_eq!(outcomes.len(), 1);
    let ReconcileOutcome::Adopted { remote, node } = &outcomes[0] else {
        panic!("expected adoption, got {:?}", outcomes[0]);
    };
    assert_eq!(remote, &remote_head);
    assert_eq!(node.id, a_head.id);

    // The thread now owns the history and the owner-qualified key is retired.
    let state = b.state().unwrap();
    assert_eq!(state.heads["main"], a_head.id);
    assert!(!state.heads.contains_key(&remote_head));
    assert!(
        b.pending_reconcile_threads().unwrap().is_empty(),
        "a reconciled thread must no longer be pending"
    );
}

#[test]
fn reconcile_in_a_sparse_workspace_records_the_full_merged_tree() {
    let temp = tempfile::tempdir().unwrap();
    let remote = temp.path().join("remote.git");
    run(
        "git",
        &[
            "init",
            "--bare",
            "-q",
            "-b",
            "main",
            remote.to_str().unwrap(),
        ],
        temp.path(),
    );

    let a = temp.path().join("a");
    init_repo(&a);
    fs::create_dir_all(a.join("crates/foo")).unwrap();
    fs::write(a.join("crates/foo/a.txt"), "a\n").unwrap();
    fs::create_dir_all(a.join("crates/bar")).unwrap();
    fs::write(a.join("crates/bar/b.txt"), "b\n").unwrap();
    git(&a, &["add", "-A"]);
    git(&a, &["commit", "-q", "-m", "base"]);
    git(&a, &["remote", "add", "origin", remote.to_str().unwrap()]);
    git(&a, &["push", "-q", "origin", "main"]);

    let b = temp.path().join("b");
    run(
        "git",
        &["clone", "-q", remote.to_str().unwrap(), b.to_str().unwrap()],
        temp.path(),
    );
    git(&b, &["config", "user.name", "Juto Test"]);
    git(&b, &["config", "user.email", "juto@test.local"]);

    let graph_a = ChangeGraph::open(&a).unwrap();
    let root_a = graph_a.capture("main", provenance("root"), None).unwrap();
    let graph_b = ChangeGraph::open(&b).unwrap();
    let root_b = graph_b.capture("main", provenance("root"), None).unwrap();

    // Both machines fork a sparse workspace on the same thread.
    let sparse = vec!["crates/foo".to_string()];
    let a_left = temp.path().join("a-left");
    let b_left = temp.path().join("b-left");
    graph_a
        .fork_sparse("left", &root_a.id, &a_left, &sparse)
        .unwrap();
    graph_b
        .fork_sparse("left", &root_b.id, &b_left, &sparse)
        .unwrap();

    // A edits a.txt on "left" and publishes.
    let a_ws = ChangeGraph::open(&a_left).unwrap();
    fs::write(a_left.join("crates/foo/a.txt"), "a edited\n").unwrap();
    let a_head = a_ws.capture("left", provenance("a"), None).unwrap();
    a_ws.push_graph(remote.to_str().unwrap()).unwrap();

    // B adds c.txt on "left".
    let b_ws = ChangeGraph::open(&b_left).unwrap();
    fs::write(b_left.join("crates/foo/c.txt"), "c\n").unwrap();
    let b_head = b_ws.capture("left", provenance("b"), None).unwrap();

    // B imports A's head and reconciles from its sparse workspace.
    b_ws.fetch_graph(remote.to_str().unwrap()).unwrap();
    let remote_head = format!("{}/left", graph_a.machine_id().unwrap());
    assert!(b_ws.state().unwrap().heads.contains_key(&remote_head));

    let outcomes = b_ws.reconcile("left").unwrap();
    assert_eq!(outcomes.len(), 1);
    let ReconcileOutcome::Joined { node, .. } = &outcomes[0] else {
        panic!("expected a clean join, got {:?}", outcomes[0]);
    };
    assert_eq!(node.parents.len(), 2);
    assert!(node.parents.contains(&a_head.id));
    assert!(node.parents.contains(&b_head.id));

    // The recorded tree is the complete merge, including the path this sparse
    // workspace never materialized.
    let git_dir = GitRepository::open(&b).unwrap().git_dir().to_path_buf();
    let listing = git_at(&git_dir, &["ls-tree", "-r", "--name-only", &node.tree]);
    assert!(listing.contains("crates/foo/a.txt"), "{listing}");
    assert!(listing.contains("crates/foo/c.txt"), "{listing}");
    assert!(
        listing.contains("crates/bar/b.txt"),
        "reconcile must record the complete merged tree, got:\n{listing}"
    );
}
