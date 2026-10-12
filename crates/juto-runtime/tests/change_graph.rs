use std::{collections::BTreeSet, fs, path::Path, process::Command};

use juto_runtime::{
    change_graph::{ChangeGraph, EditProvenance},
    git_transfer::GitRepository,
};
use tempfile::TempDir;

fn command(cwd: &Path, program: &str, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{program} {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().into()
}

fn empty_fixture() -> TempDir {
    let directory = tempfile::tempdir().unwrap();
    command(directory.path(), "git", &["init"]);
    command(
        directory.path(),
        "git",
        &["config", "user.name", "Graph Test"],
    );
    command(
        directory.path(),
        "git",
        &["config", "user.email", "graph@example.test"],
    );
    directory
}

fn fixture() -> TempDir {
    let directory = empty_fixture();
    fs::write(directory.path().join("base"), "base\n").unwrap();
    command(directory.path(), "git", &["add", "base"]);
    command(directory.path(), "git", &["commit", "-m", "base"]);
    directory
}

fn provenance(session: &str) -> EditProvenance {
    EditProvenance {
        session_id: session.into(),
        tool: "edit".into(),
        selection_mode: Some("range".into()),
        path: Some("file".into()),
    }
}

fn commit_count(cwd: &Path) -> usize {
    command(
        cwd,
        "git",
        &["cat-file", "--batch-all-objects", "--batch-check"],
    )
    .lines()
    .filter(|line| line.split_whitespace().nth(1) == Some("commit"))
    .count()
}

#[test]
fn native_live_edits_are_not_commits_and_named_points_export_exact_trees() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    fs::write(cwd.join("human"), "pre-existing\n").unwrap();
    let before = graph.snapshot_tree().unwrap();
    fs::write(cwd.join("file"), "agent one\n").unwrap();
    let after = graph.snapshot_tree().unwrap();
    let first = graph
        .record_edit(&before, &after, provenance("one"))
        .unwrap()
        .unwrap();
    let state = graph.state().unwrap();
    assert_eq!(state.nodes[0].provenance.tool, "external");
    assert_eq!(first.parents, vec![state.nodes[0].id.clone()]);
    assert_eq!(first.thread, "one");
    graph
        .mark(&first.id, "verified-first", "feat: first", true)
        .unwrap();
    assert_eq!(
        commit_count(cwd),
        1,
        "live snapshots/checkpoints must not create Git commits"
    );
    assert!(
        graph
            .mark(&first.id, "verified-first", "feat: replacement", true)
            .is_err()
    );
    assert!(
        graph
            .record_edit(&after, &after, provenance("noop"))
            .unwrap()
            .is_none()
    );
    fs::write(cwd.join("file"), "agent two unmarked\n").unwrap();
    let next = graph.snapshot_tree().unwrap();
    let second = graph
        .record_edit(&after, &next, provenance("two"))
        .unwrap()
        .unwrap();
    assert_eq!(second.parents, vec![first.id.clone()]);
    drop(graph);
    let reopened = ChangeGraph::open(cwd).unwrap();
    assert_eq!(
        reopened.state().unwrap().nodes.last().unwrap().id,
        second.id
    );
    assert!(reopened.export(&second.id, "feature/graph").is_err());
    let result = reopened.export("verified-first", "feature/graph").unwrap();
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:file", result.tip)]),
        "agent one"
    );
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:human", result.tip)]),
        "pre-existing"
    );
    assert_eq!(
        fs::read_to_string(cwd.join("file")).unwrap(),
        "agent two unmarked\n"
    );
    assert_eq!(
        command(cwd, "git", &["rev-parse", &format!("{}^", result.tip)]),
        reopened.state().unwrap().base_commit
    );
    reopened
        .mark(&second.id, "second-added", "feat: second", false)
        .unwrap();
    assert!(
        reopened
            .mark(&first.id, "second-added", "feat: first", false)
            .is_err()
    );
    let advanced = reopened.export("second-added", "feature/graph").unwrap();
    assert_eq!(
        command(cwd, "git", &["rev-parse", &format!("{}^", advanced.tip)]),
        result.tip
    );
    assert_eq!(
        reopened
            .export("second-added", "feature/graph")
            .unwrap()
            .tip,
        advanced.tip
    );
}

#[test]
fn native_forks_isolate_agent_files_and_resolved_join_preserves_marked_dag() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph
        .mark(&root.id, "working-base", "feat: root", true)
        .unwrap();
    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    let right_dir = workspaces.path().join("right");
    graph.fork("left", "working-base", &left_dir).unwrap();
    graph.fork("right", &root.id, &right_dir).unwrap();
    let left_graph = ChangeGraph::open(&left_dir).unwrap();
    let right_graph = ChangeGraph::open(&right_dir).unwrap();
    assert_eq!(left_graph.thread(), Some("left"));
    assert_eq!(right_graph.thread(), Some("right"));
    fs::write(left_dir.join("left"), "left\n").unwrap();
    let left = left_graph
        .capture("left", provenance("left-agent"), None)
        .unwrap();
    left_graph
        .mark(&left.id, "left-ready", "feat: left", true)
        .unwrap();
    fs::write(right_dir.join("right"), "right\n").unwrap();
    let right_live = right_graph
        .capture("right", provenance("right-agent"), None)
        .unwrap();
    fs::write(right_dir.join("right"), "right final\n").unwrap();
    let right = right_graph
        .capture("right", provenance("right-agent"), None)
        .unwrap();
    right_graph
        .mark(&right.id, "right-ready", "feat: right", true)
        .unwrap();
    assert_eq!(left.parents, vec![root.id.clone()]);
    assert_eq!(right_live.parents, vec![root.id.clone()]);
    assert_eq!(right.parents, vec![right_live.id.clone()]);
    assert!(!right_dir.join("left").exists());
    assert!(!left_dir.join("right").exists());
    assert!(!cwd.join("left").exists());
    assert_eq!(commit_count(cwd), 1);
    fs::write(cwd.join("left"), "left\n").unwrap();
    fs::write(cwd.join("right"), "right final\n").unwrap();
    let merge = graph
        .capture(
            "integration",
            provenance("integrator"),
            Some(vec![left.id.clone(), right.id.clone()]),
        )
        .unwrap();
    graph
        .mark(&merge.id, "both-ready", "feat: joined", true)
        .unwrap();
    let result = graph.export("both-ready", "feature/join").unwrap();
    let parents = command(cwd, "git", &["show", "-s", "--format=%P", &result.tip]);
    let parent_set: BTreeSet<&str> = parents.split_whitespace().collect();
    assert_eq!(
        parent_set,
        BTreeSet::from([
            result.commits[&left.id].as_str(),
            result.commits[&right.id].as_str()
        ])
    );
    assert!(!result.commits.contains_key(&right_live.id));
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:left", result.tip)]),
        "left"
    );
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:right", result.tip)]),
        "right final"
    );
    let state = graph.state().unwrap();
    assert_eq!(state.heads["left"], left.id);
    assert_eq!(state.heads["right"], right.id);
    assert_eq!(state.heads["integration"], merge.id);
    assert!(
        left_graph
            .capture("right", provenance("wrong"), None)
            .is_err()
    );
    assert!(
        graph
            .fork("left", &root.id, &workspaces.path().join("duplicate"))
            .is_err()
    );
    assert!(
        graph
            .capture("bad", provenance("bad"), Some(vec!["missing".into()]))
            .is_err()
    );
    assert!(
        graph
            .capture(
                "bad",
                provenance("bad"),
                Some(vec![root.id.clone(), root.id])
            )
            .is_err()
    );
}

#[test]
fn native_thread_tool_edits_keep_fork_parent_not_another_threads_latest_head() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    fs::write(cwd.join("later-main"), "do not inherit\n").unwrap();
    let later = graph.capture("main", provenance("root"), None).unwrap();
    graph
        .mark(&later.id, "later", "feat: later main", true)
        .unwrap();
    let forks = tempfile::tempdir().unwrap();
    let child = forks.path().join("child");
    graph.fork("child", &root.id, &child).unwrap();
    let other = ChangeGraph::open(&child).unwrap();
    let before = other.snapshot_tree().unwrap();
    fs::write(child.join("child-edit"), "isolated\n").unwrap();
    let after = other.snapshot_tree().unwrap();
    let node = other
        .record_edit(&before, &after, provenance("child-session"))
        .unwrap()
        .unwrap();
    assert_eq!(node.thread, "child");
    assert_eq!(node.parents, vec![root.id]);
    graph
        .mark(&node.id, "child-ready", "feat: child", true)
        .unwrap();
    let exported = other.export("child-ready", "feature/child").unwrap();
    assert!(!exported.commits.contains_key(&later.id));
    assert_eq!(
        command(cwd, "git", &["ls-tree", &exported.tip, "later-main"]),
        ""
    );
    assert_eq!(
        command(
            cwd,
            "git",
            &["show", &format!("{}:child-edit", exported.tip)]
        ),
        "isolated"
    );
    assert_eq!(graph.state().unwrap().heads["main"], later.id);
}

#[test]
fn first_named_point_in_empty_repository_is_a_parentless_git_commit() {
    let directory = empty_fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    fs::write(cwd.join("first"), "initial contents\n").unwrap();
    let node = graph.capture("main", provenance("initial"), None).unwrap();
    assert_eq!(commit_count(cwd), 0);
    graph
        .mark(&node.id, "initial", "feat: initial commit", false)
        .unwrap();
    assert_eq!(commit_count(cwd), 0);
    let result = graph.export("initial", "feature/initial").unwrap();
    assert_eq!(
        command(cwd, "git", &["show", "-s", "--format=%P", &result.tip]),
        ""
    );
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:first", result.tip)]),
        "initial contents"
    );
}

#[test]
fn native_snapshot_refs_survive_git_gc_without_creating_commits() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    fs::write(cwd.join("transient"), "retained snapshot\n").unwrap();
    let point = graph.capture("main", provenance("one"), None).unwrap();
    fs::remove_file(cwd.join("transient")).unwrap();
    graph.capture("main", provenance("two"), None).unwrap();
    command(cwd, "git", &["gc", "--prune=now"]);
    assert_eq!(
        GitRepository::open(cwd)
            .unwrap()
            .snapshot_tree(&point.tree)
            .unwrap(),
        point.tree
    );
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:transient", point.tree)]),
        "retained snapshot"
    );
    assert_eq!(commit_count(cwd), 1);
}
