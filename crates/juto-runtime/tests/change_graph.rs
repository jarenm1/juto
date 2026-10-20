use std::os::unix::fs::PermissionsExt;
use std::{collections::BTreeSet, fs, path::Path, process::Command};

use juto_runtime::{
    change_graph::{ChangeGraph, EditProvenance, GcReport, JoinOutcome},
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

/// The native graph directory inside the repository's Git directory.
fn graph_dir(cwd: &Path) -> std::path::PathBuf {
    GitRepository::open(cwd).unwrap().git_dir().join("juto")
}

/// Every `refs/juto/snapshots/<id>` ref currently pinned in the repository.
fn snapshot_refs(cwd: &Path) -> BTreeSet<String> {
    command(
        cwd,
        "git",
        &["for-each-ref", "--format=%(refname)", "refs/juto/snapshots"],
    )
    .lines()
    .map(str::to_owned)
    .collect()
}

#[test]
fn record_edit_noop_does_not_regress_the_workspace_head() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    // Thread "main" advances the shared workspace head.
    fs::write(cwd.join("main-edit"), "main\n").unwrap();
    let main_tree = graph.snapshot_tree().unwrap();
    let main_node = graph
        .record_edit(&root.tree, &main_tree, provenance("main"))
        .unwrap()
        .unwrap();

    // Thread "other" records an edit on top of the shared workspace head.
    fs::write(cwd.join("other-edit"), "other\n").unwrap();
    let other_tree = graph.snapshot_tree().unwrap();
    let other_node = graph
        .record_edit(&main_tree, &other_tree, provenance("other"))
        .unwrap()
        .unwrap();

    let key = cwd.canonicalize().unwrap().to_string_lossy().into_owned();
    assert_eq!(graph.state().unwrap().workspace_heads[&key], other_node.id);

    // A no-op edit from "main" creates no node, so it must not move the shared
    // workspace head back to "main"'s older head.
    assert!(
        graph
            .record_edit(&other_tree, &other_tree, provenance("main"))
            .unwrap()
            .is_none()
    );
    let state = graph.state().unwrap();
    assert_eq!(
        state.workspace_heads[&key], other_node.id,
        "a no-op edit must not regress the shared workspace head"
    );
    assert_eq!(state.heads["main"], main_node.id);
    assert_eq!(state.heads["other"], other_node.id);
}

#[test]
fn fork_removes_materialized_directory_when_bookkeeping_fails() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    let refs_before = snapshot_refs(cwd);
    let state_dir = GitRepository::open(cwd).unwrap().git_dir().join("juto");
    let workspaces = tempfile::tempdir().unwrap();
    let destination = workspaces.path().join("child");

    // Make the native bookkeeping unwritable so `save` fails after the fork
    // directory has already been materialized.
    let original = fs::metadata(&state_dir).unwrap().permissions();
    let mut unwritable = original.clone();
    unwritable.set_mode(0o500);
    fs::set_permissions(&state_dir, unwritable).unwrap();

    let failed = graph.fork("child", &root.id, &destination);
    fs::set_permissions(&state_dir, original).unwrap();

    assert!(failed.is_err(), "fork must report the bookkeeping failure");
    assert!(
        !destination.exists(),
        "a failed fork must not leave a materialized directory behind"
    );
    assert_eq!(
        snapshot_refs(cwd),
        refs_before,
        "a failed fork must not leave its snapshot pinned"
    );
    // The same destination must be usable again.
    graph.fork("child", &root.id, &destination).unwrap();
    assert!(destination.join(".juto-workspace.json").is_file());
    assert_eq!(
        ChangeGraph::open(&destination).unwrap().thread(),
        Some("child")
    );
}

#[test]
fn fork_removes_materialized_directory_when_snapshot_pin_fails() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    let snapshots = GitRepository::open(cwd)
        .unwrap()
        .git_dir()
        .join("refs/juto/snapshots");
    let workspaces = tempfile::tempdir().unwrap();
    let destination = workspaces.path().join("child");

    // Pinning the fork's start tree writes a ref into this directory. Making
    // it unwritable fails the pin after the fork directory has been created.
    let original = fs::metadata(&snapshots).unwrap().permissions();
    let mut unwritable = original.clone();
    unwritable.set_mode(0o500);
    fs::set_permissions(&snapshots, unwritable).unwrap();

    let failed = graph.fork("child", &root.id, &destination);
    fs::set_permissions(&snapshots, original).unwrap();

    assert!(failed.is_err(), "fork must report the snapshot failure");
    assert!(
        !destination.exists(),
        "a fork whose snapshot pin failed must not leave a directory behind"
    );
    graph.fork("child", &root.id, &destination).unwrap();
}

#[test]
fn fork_pins_a_valid_snapshot_ref_for_any_thread_name() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    let workspaces = tempfile::tempdir().unwrap();
    // Names that are invalid as Git ref components, plus names that collide
    // if invalid characters were replaced instead of escaped (`a b`, `a_b`).
    let names = [
        "my work",
        "a b",
        "a_b",
        "a:b?c*d[e]f\\g~h^i",
        "dots..here",
        "at@{brace",
        "trailing.lock",
        "tab\tname",
    ];
    let before = snapshot_refs(cwd);
    let destinations: Vec<_> = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let destination = workspaces.path().join(index.to_string());
            graph
                .fork(name, &root.id, &destination)
                .unwrap_or_else(|error| panic!("fork {name:?} failed: {error:#}"));
            destination
        })
        .collect();
    let after = snapshot_refs(cwd);
    assert_eq!(
        after.len(),
        before.len() + names.len(),
        "every thread must pin its own snapshot"
    );
    for refname in after.difference(&before) {
        let status = Command::new("git")
            .args(["check-ref-format", refname])
            .status()
            .unwrap();
        assert!(status.success(), "{refname:?} is not a valid ref name");
    }

    // Removing a fork and collecting garbage releases exactly its snapshot.
    fs::remove_dir_all(&destinations[0]).unwrap();
    graph.gc().unwrap();
    let released = snapshot_refs(cwd);
    assert_eq!(
        after.difference(&released).count(),
        1,
        "gc must release exactly the removed fork's snapshot"
    );
}

#[test]
fn fork_rejects_thread_names_too_long_to_pin_without_materializing() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    let workspaces = tempfile::tempdir().unwrap();

    // The snapshot id is one filesystem path component (at most 255 bytes).
    let longest = "t".repeat(120);
    graph
        .fork(&longest, &root.id, &workspaces.path().join("longest"))
        .unwrap();

    let too_long = "t".repeat(121);
    let destination = workspaces.path().join("too-long");
    assert!(graph.fork(&too_long, &root.id, &destination).is_err());
    assert!(
        !destination.exists(),
        "a rejected fork must not materialize"
    );
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

#[test]
fn version_three_journal_roundtrips_edits_marks_and_forks() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    fs::write(cwd.join("file"), "one\n").unwrap();
    let after = graph.snapshot_tree().unwrap();
    let node = graph
        .record_edit(&root.tree, &after, provenance("main"))
        .unwrap()
        .unwrap();
    graph.mark(&node.id, "first", "feat: first", true).unwrap();
    let workspaces = tempfile::tempdir().unwrap();
    let child = workspaces.path().join("child");
    graph.fork("child", "first", &child).unwrap();

    let state_dir = graph_dir(cwd);
    assert!(state_dir.join("graph.json").is_file());
    assert!(state_dir.join("nodes.jsonl").is_file());
    assert!(
        !state_dir.join("change-graph.json").exists(),
        "a version 3 graph must not keep the legacy single-file state"
    );

    // The journal is one JSON object per line, in append (topological) order.
    let journal = fs::read_to_string(state_dir.join("nodes.jsonl")).unwrap();
    let lines: Vec<&str> = journal
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 2);
    assert!(
        lines
            .iter()
            .all(|line| line.starts_with('{') && line.ends_with('}'))
    );

    drop(graph);
    let reopened = ChangeGraph::open(cwd).unwrap();
    let state = reopened.state().unwrap();
    assert_eq!(state.version, 3);
    assert_eq!(state.nodes.len(), 2);
    assert_eq!(state.nodes[0].id, root.id);
    assert_eq!(state.nodes[1].id, node.id);
    assert_eq!(state.nodes[1].mark.as_ref().unwrap().name, "first");
    assert_eq!(state.heads["main"], node.id);
    assert_eq!(state.heads["child"], node.id);
    assert_eq!(state.checkpoints["first"], node.id);
    assert_eq!(ChangeGraph::open(&child).unwrap().thread(), Some("child"));
}

#[test]
fn version_two_state_migrates_to_the_journal_layout() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph.mark(&root.id, "base", "feat: base", true).unwrap();
    let mut legacy = graph.state().unwrap();
    drop(graph);

    // Downgrade the on-disk graph to the version 2 single-file layout.
    let state_dir = graph_dir(cwd);
    fs::remove_file(state_dir.join("graph.json")).unwrap();
    fs::remove_file(state_dir.join("nodes.jsonl")).unwrap();
    legacy.version = 2;
    fs::write(
        state_dir.join("change-graph.json"),
        serde_json::to_vec(&legacy).unwrap(),
    )
    .unwrap();

    let migrated = ChangeGraph::open(cwd).unwrap();
    let state = migrated.state().unwrap();
    assert_eq!(state.version, 3);
    assert_eq!(state.nodes.len(), 1);
    assert_eq!(state.nodes[0].id, root.id);
    assert_eq!(state.nodes[0].mark.as_ref().unwrap().name, "base");
    assert_eq!(state.heads["main"], root.id);
    assert_eq!(state.checkpoints["base"], root.id);
    assert!(state_dir.join("graph.json").is_file());
    assert!(state_dir.join("nodes.jsonl").is_file());
    assert!(
        !state_dir.join("change-graph.json").exists(),
        "migration must retire the legacy file once version 3 is durable"
    );

    // The migrated graph keeps working end to end.
    fs::write(cwd.join("after"), "after\n").unwrap();
    let next = migrated.capture("main", provenance("next"), None).unwrap();
    assert_eq!(next.parents, vec![root.id.clone()]);
    assert_eq!(migrated.state().unwrap().nodes.len(), 2);
}

#[test]
fn gc_prunes_abandoned_thread_nodes_and_releases_their_snapshot_refs() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph.mark(&root.id, "base", "feat: base", true).unwrap();

    // A side thread forks from the checkpoint and records a live point.
    let workspaces = tempfile::tempdir().unwrap();
    let side_dir = workspaces.path().join("side");
    graph.fork("side", "base", &side_dir).unwrap();
    let side = ChangeGraph::open(&side_dir).unwrap();
    fs::write(side_dir.join("side"), "side\n").unwrap();
    let abandoned = side.capture("side", provenance("side"), None).unwrap();

    // Re-basing the side thread onto the checkpoint abandons its earlier tip.
    fs::write(side_dir.join("side"), "side rebased\n").unwrap();
    let rebased = side
        .capture("side", provenance("side"), Some(vec![root.id.clone()]))
        .unwrap();
    assert_eq!(rebased.parents, vec![root.id.clone()]);

    let refs = snapshot_refs(cwd);
    assert!(refs.contains(&format!("refs/juto/snapshots/{}", abandoned.id)));
    assert!(refs.contains(&format!("refs/juto/snapshots/{}", root.id)));

    let report = graph.gc().unwrap();
    assert_eq!(
        report,
        GcReport {
            pruned_nodes: 1,
            retained_nodes: 2,
            released_refs: 1,
            released_node_refs: 0,
            dropped_workspace_heads: 0,
        }
    );

    let refs = snapshot_refs(cwd);
    assert!(!refs.contains(&format!("refs/juto/snapshots/{}", abandoned.id)));
    assert!(refs.contains(&format!("refs/juto/snapshots/{}", root.id)));
    assert!(refs.contains(&format!("refs/juto/snapshots/{}", rebased.id)));
    // The baseline anchors are not nodes and must never be released.
    assert!(refs.contains("refs/juto/snapshots/base"));
    assert!(refs.contains("refs/juto/snapshots/base-tree"));

    let state = graph.state().unwrap();
    assert_eq!(state.nodes.len(), 2);
    assert!(state.nodes.iter().all(|node| node.id != abandoned.id));
    assert_eq!(state.heads["main"], root.id);
    assert_eq!(state.heads["side"], rebased.id);
    assert_eq!(state.checkpoints["base"], root.id);

    // The surviving graph still exports and reloads.
    let reopened = ChangeGraph::open(cwd).unwrap();
    assert_eq!(reopened.state().unwrap().nodes.len(), 2);
    assert!(reopened.export("base", "feature/base").is_ok());
}

#[test]
fn gc_is_a_no_op_when_every_node_is_reachable() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph.mark(&root.id, "base", "feat: base", true).unwrap();
    fs::write(cwd.join("next"), "next\n").unwrap();
    let next = graph.capture("main", provenance("next"), None).unwrap();

    let state_dir = graph_dir(cwd);
    let before = fs::read(state_dir.join("nodes.jsonl")).unwrap();
    let report = graph.gc().unwrap();
    assert_eq!(
        report,
        GcReport {
            pruned_nodes: 0,
            retained_nodes: 2,
            released_refs: 0,
            released_node_refs: 0,
            dropped_workspace_heads: 0,
        }
    );
    assert_eq!(
        fs::read(state_dir.join("nodes.jsonl")).unwrap(),
        before,
        "a no-op sweep must not rewrite the journal"
    );
    assert_eq!(graph.state().unwrap().nodes.len(), 2);
    assert!(snapshot_refs(cwd).contains(&format!("refs/juto/snapshots/{}", next.id)));
}

#[test]
fn gc_drops_workspace_heads_whose_directory_is_gone() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let side_dir = workspaces.path().join("side");
    graph.fork("side", &root.id, &side_dir).unwrap();
    let side = ChangeGraph::open(&side_dir).unwrap();
    fs::write(side_dir.join("side"), "side\n").unwrap();
    side.capture("side", provenance("side"), None).unwrap();

    let key = side_dir
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert!(graph.state().unwrap().workspace_heads.contains_key(&key));

    // The subagent workspace is cleaned up: its directory is gone, so its head
    // can never advance again and must not pin the thread's nodes forever.
    fs::remove_dir_all(&side_dir).unwrap();

    let report = graph.gc().unwrap();
    assert_eq!(report.dropped_workspace_heads, 1);
    assert!(!graph.state().unwrap().workspace_heads.contains_key(&key));
}

#[test]
fn join_in_a_sparse_workspace_records_the_full_merged_tree() {
    let directory = fixture();
    let cwd = directory.path();
    fs::create_dir_all(cwd.join("crates/foo")).unwrap();
    fs::write(cwd.join("crates/foo/a.txt"), "a\n").unwrap();
    fs::create_dir_all(cwd.join("crates/bar")).unwrap();
    fs::write(cwd.join("crates/bar/b.txt"), "b\n").unwrap();
    command(cwd, "git", &["add", "-A"]);
    command(cwd, "git", &["commit", "-m", "layout"]);

    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    let right_dir = workspaces.path().join("right");
    let sparse = vec!["crates/foo".to_string()];
    graph
        .fork_sparse("left", &root.id, &left_dir, &sparse)
        .unwrap();
    graph
        .fork_sparse("right", &root.id, &right_dir, &sparse)
        .unwrap();

    let left = ChangeGraph::open(&left_dir).unwrap();
    fs::write(left_dir.join("crates/foo/a.txt"), "a edited\n").unwrap();
    let left_head = left.capture("left", provenance("left"), None).unwrap();

    let right = ChangeGraph::open(&right_dir).unwrap();
    fs::write(right_dir.join("crates/foo/c.txt"), "c\n").unwrap();
    let right_head = right.capture("right", provenance("right"), None).unwrap();

    // Join from the sparse left workspace. The merged tree contains
    // crates/bar/b.txt, which this workspace never materialized, so the node
    // must record the merge result rather than a snapshot of the workspace.
    let outcome = left.join("left", &left_head.id, &right_head.id).unwrap();
    let JoinOutcome::Joined { node, tree, .. } = outcome else {
        panic!("expected a clean join, got {outcome:?}");
    };
    assert_eq!(node.tree, tree);

    let listing = command(cwd, "git", &["ls-tree", "-r", "--name-only", &tree]);
    assert!(listing.contains("crates/foo/a.txt"), "{listing}");
    assert!(listing.contains("crates/foo/c.txt"), "{listing}");
    assert!(
        listing.contains("crates/bar/b.txt"),
        "the join must record the complete merged tree, got:\n{listing}"
    );
    assert!(listing.contains("base"), "{listing}");
}

#[test]
fn resolve_in_a_sparse_workspace_records_the_full_merged_tree() {
    let directory = fixture();
    let cwd = directory.path();
    fs::create_dir_all(cwd.join("crates/foo")).unwrap();
    fs::write(cwd.join("crates/foo/a.txt"), "a\n").unwrap();
    fs::create_dir_all(cwd.join("crates/bar")).unwrap();
    fs::write(cwd.join("crates/bar/b.txt"), "b\n").unwrap();
    command(cwd, "git", &["add", "-A"]);
    command(cwd, "git", &["commit", "-m", "layout"]);

    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    let right_dir = workspaces.path().join("right");
    let sparse = vec!["crates/foo".to_string()];
    graph
        .fork_sparse("left", &root.id, &left_dir, &sparse)
        .unwrap();
    // The right side is a full fork, so it can change a path the sparse left
    // workspace never materialized.
    graph.fork("right", &root.id, &right_dir).unwrap();

    let left = ChangeGraph::open(&left_dir).unwrap();
    fs::write(left_dir.join("crates/foo/a.txt"), "from left\n").unwrap();
    let left_head = left.capture("left", provenance("left"), None).unwrap();

    let right = ChangeGraph::open(&right_dir).unwrap();
    fs::write(right_dir.join("crates/foo/a.txt"), "from right\n").unwrap();
    fs::write(right_dir.join("crates/bar/b.txt"), "b from right\n").unwrap();
    let right_head = right.capture("right", provenance("right"), None).unwrap();

    // The agent resolves the conflict by hand, then records the merge. The
    // merged tree carries the right side's crates/bar/b.txt, which this
    // workspace never materialized, so the node must record the merge result
    // rather than a snapshot of the sparse workspace.
    fs::write(left_dir.join("crates/foo/a.txt"), "resolved\n").unwrap();
    let node = left
        .resolve("left", &left_head.id, &right_head.id, provenance("resolve"))
        .unwrap();

    let listing = command(cwd, "git", &["ls-tree", "-r", "--name-only", &node.tree]);
    assert!(
        listing.contains("crates/bar/b.txt"),
        "the resolve must record the complete merged tree, got:\n{listing}"
    );
    assert_eq!(
        command(
            cwd,
            "git",
            &["show", &format!("{}:crates/bar/b.txt", node.tree)]
        ),
        "b from right",
        "the resolve must carry the merge's version of an unmaterialized path"
    );
    assert_eq!(
        command(
            cwd,
            "git",
            &["show", &format!("{}:crates/foo/a.txt", node.tree)]
        ),
        "resolved"
    );
}

#[test]
fn resolve_carries_the_merge_and_materializes_it() {
    let directory = fixture();
    let cwd = directory.path();
    fs::write(cwd.join("f.txt"), "line1\nline2\nline3\n").unwrap();
    fs::write(cwd.join("g.txt"), "other\n").unwrap();
    command(cwd, "git", &["add", "-A"]);
    command(cwd, "git", &["commit", "-m", "base"]);

    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    let right_dir = workspaces.path().join("right");
    graph.fork("left", &root.id, &left_dir).unwrap();
    graph.fork("right", &root.id, &right_dir).unwrap();

    let left = ChangeGraph::open(&left_dir).unwrap();
    fs::write(left_dir.join("f.txt"), "from left\n").unwrap();
    let left_head = left.capture("left", provenance("left"), None).unwrap();

    let right = ChangeGraph::open(&right_dir).unwrap();
    fs::write(right_dir.join("f.txt"), "from right\n").unwrap();
    fs::write(right_dir.join("g.txt"), "g from right\n").unwrap();
    let right_head = right.capture("right", provenance("right"), None).unwrap();

    // The agent resolves the conflict on f.txt by hand. g.txt is not
    // conflicted, so it must come from the merge, not this stale workspace.
    fs::write(left_dir.join("f.txt"), "resolved\n").unwrap();
    let node = left
        .resolve("left", &left_head.id, &right_head.id, provenance("resolve"))
        .unwrap();

    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:g.txt", node.tree)]),
        "g from right"
    );
    assert_eq!(
        command(cwd, "git", &["show", &format!("{}:f.txt", node.tree)]),
        "resolved"
    );
    // The workspace matches the node, so a later capture cannot revert g.txt.
    assert_eq!(
        fs::read_to_string(left_dir.join("g.txt")).unwrap(),
        "g from right\n"
    );
    assert_eq!(
        fs::read_to_string(left_dir.join("f.txt")).unwrap(),
        "resolved\n"
    );
}

#[test]
fn dirty_paths_reports_uncaptured_edits() {
    let directory = fixture();
    let cwd = directory.path();
    fs::write(cwd.join("f.txt"), "base\n").unwrap();
    fs::write(cwd.join("g.txt"), "other\n").unwrap();
    command(cwd, "git", &["add", "-A"]);
    command(cwd, "git", &["commit", "-m", "base"]);

    let graph = ChangeGraph::open(cwd).unwrap();
    graph.capture("main", provenance("init"), None).unwrap();
    assert!(graph.dirty_paths().unwrap().is_empty());

    // A shell edit is not captured automatically, so it must be reported.
    fs::write(cwd.join("f.txt"), "changed\n").unwrap();
    fs::remove_file(cwd.join("g.txt")).unwrap();
    fs::write(cwd.join("h.txt"), "new\n").unwrap();
    assert_eq!(
        graph.dirty_paths().unwrap(),
        vec![
            "f.txt".to_string(),
            "g.txt".to_string(),
            "h.txt".to_string()
        ]
    );

    graph.capture("main", provenance("e1"), None).unwrap();
    assert!(graph.dirty_paths().unwrap().is_empty());
}

#[test]
fn rewind_in_a_sparse_workspace_records_the_target_tree() {
    let directory = fixture();
    let cwd = directory.path();
    fs::create_dir_all(cwd.join("crates/foo")).unwrap();
    fs::write(cwd.join("crates/foo/a.txt"), "a\n").unwrap();
    fs::create_dir_all(cwd.join("crates/bar")).unwrap();
    fs::write(cwd.join("crates/bar/b.txt"), "b\n").unwrap();
    command(cwd, "git", &["add", "-A"]);
    command(cwd, "git", &["commit", "-m", "layout"]);

    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    graph
        .fork_sparse("left", &root.id, &left_dir, &["crates/foo".to_string()])
        .unwrap();

    let left = ChangeGraph::open(&left_dir).unwrap();
    fs::write(left_dir.join("crates/foo/a.txt"), "a edited\n").unwrap();
    let edited = left.capture("left", provenance("left"), None).unwrap();
    assert_ne!(edited.tree, root.tree);

    // Rewinding to the fork point must record the fork point's complete tree,
    // including crates/bar/b.txt, which this workspace never materialized.
    let rewound = left.rewind("left", &root.id).unwrap();
    assert_eq!(rewound.tree, root.tree);
    assert_eq!(rewound.parents, vec![edited.id.clone()]);

    let listing = command(cwd, "git", &["ls-tree", "-r", "--name-only", &rewound.tree]);
    assert!(listing.contains("crates/bar/b.txt"), "{listing}");
    assert!(listing.contains("crates/foo/a.txt"), "{listing}");
}

#[test]
fn gc_packs_loose_refs() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    graph.capture("main", provenance("root"), None).unwrap();

    let git_dir = GitRepository::open(cwd).unwrap().git_dir().to_path_buf();
    let node = graph.state().unwrap().nodes.last().unwrap().clone();
    let loose = git_dir.join("refs/juto/snapshots").join(&node.id);
    assert!(
        loose.exists(),
        "a fresh capture must leave a loose snapshot ref"
    );

    graph.gc().unwrap();

    // The loose file is gone, packed-refs exists, and the ref still resolves.
    assert!(!loose.exists(), "gc must pack the loose ref away");
    assert!(
        git_dir.join("packed-refs").exists(),
        "gc must write packed-refs"
    );
    assert_eq!(
        command(
            cwd,
            "git",
            &["rev-parse", &format!("refs/juto/snapshots/{}", node.id)]
        ),
        node.tree
    );
}

#[test]
fn verify_reports_missing_and_mismatched_snapshot_refs() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    fs::write(cwd.join("next"), "next\n").unwrap();
    let next = graph.capture("main", provenance("next"), None).unwrap();

    let report = graph.verify().unwrap();
    assert_eq!(report.nodes, 2);
    assert!(report.is_ok(), "{report:?}");

    let git_dir = GitRepository::open(cwd).unwrap().git_dir().to_path_buf();
    let snapshot_ref = format!("refs/juto/snapshots/{}", next.id);

    // A missing snapshot ref is reported.
    command(
        cwd,
        "git",
        &[
            "--git-dir",
            git_dir.to_str().unwrap(),
            "update-ref",
            "-d",
            &snapshot_ref,
        ],
    );
    let report = graph.verify().unwrap();
    assert_eq!(report.missing_snapshot_refs, vec![next.id.clone()]);
    assert!(!report.is_ok());

    // A snapshot ref pointing at the wrong tree is reported.
    command(
        cwd,
        "git",
        &[
            "--git-dir",
            git_dir.to_str().unwrap(),
            "update-ref",
            &snapshot_ref,
            &root.tree,
        ],
    );
    let report = graph.verify().unwrap();
    assert_eq!(report.mismatched_snapshot_refs, vec![next.id.clone()]);
    assert!(!report.is_ok());
}

#[test]
fn fork_survives_a_stale_pending_fork_entry() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    // A crashed provision left a directory behind that cannot be reclaimed.
    let stale = graph_dir(cwd).join("forks").join("pending-crashed");
    fs::create_dir_all(stale.join("locked")).unwrap();
    fs::write(stale.join("locked").join("file"), "x").unwrap();
    let mut locked = fs::metadata(stale.join("locked")).unwrap().permissions();
    locked.set_mode(0o500);
    fs::set_permissions(stale.join("locked"), locked).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let child = workspaces.path().join("child");
    graph.fork("child", &root.id, &child).unwrap();
    assert!(child.join(".juto-workspace.json").is_file());
    assert_eq!(ChangeGraph::open(&child).unwrap().thread(), Some("child"));
    assert_eq!(graph.state().unwrap().heads["child"], root.id);

    // Restore permissions so the temporary directory can be cleaned up.
    let mut open = fs::metadata(stale.join("locked")).unwrap().permissions();
    open.set_mode(0o700);
    fs::set_permissions(stale.join("locked"), open).unwrap();
}

#[test]
fn thread_names_may_not_contain_a_slash() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    assert!(
        graph
            .capture("bad/thread", provenance("bad"), None)
            .is_err()
    );
    let workspaces = tempfile::tempdir().unwrap();
    let destination = workspaces.path().join("bad");
    assert!(graph.fork("bad/thread", &root.id, &destination).is_err());
    assert!(
        !destination.exists(),
        "a rejected fork must not materialize"
    );

    fs::write(cwd.join("edit"), "edit\n").unwrap();
    let after = graph.snapshot_tree().unwrap();
    assert!(
        graph
            .record_edit(&root.tree, &after, provenance("bad/thread"))
            .is_err()
    );
}

#[test]
fn rewind_materializes_the_target_tree_and_appends_a_forward_node() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph.mark(&root.id, "base", "feat: base", true).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let child_dir = workspaces.path().join("child");
    graph.fork("child", &root.id, &child_dir).unwrap();
    let child = ChangeGraph::open(&child_dir).unwrap();

    fs::write(child_dir.join("extra"), "extra\n").unwrap();
    child.capture("child", provenance("extra"), None).unwrap();
    fs::write(child_dir.join("base"), "changed\n").unwrap();
    let head = child.capture("child", provenance("change"), None).unwrap();
    assert_eq!(
        fs::read_to_string(child_dir.join("base")).unwrap(),
        "changed\n"
    );

    // Rewind by checkpoint name: the target tree restores one file and deletes
    // another, and the rewind itself is recorded as a forward node.
    let rewound = child.rewind("child", "base").unwrap();

    assert_eq!(rewound.tree, root.tree);
    assert_eq!(rewound.parents, vec![head.id.clone()]);
    assert_eq!(rewound.provenance.tool, "rewind");
    assert_eq!(
        fs::read_to_string(child_dir.join("base")).unwrap(),
        "base\n"
    );
    assert!(
        !child_dir.join("extra").exists(),
        "rewind must delete files the target tree does not contain"
    );
    let state = child.state().unwrap();
    assert_eq!(state.heads["child"], rewound.id);
    assert_eq!(state.nodes.last().unwrap().id, rewound.id);
    assert_eq!(state.checkpoints["base"], root.id);
}

#[test]
fn join_of_two_diverged_threads_produces_a_two_parent_node() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph.mark(&root.id, "base", "feat: base", true).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    let right_dir = workspaces.path().join("right");
    graph.fork("left", &root.id, &left_dir).unwrap();
    graph.fork("right", &root.id, &right_dir).unwrap();
    let left_graph = ChangeGraph::open(&left_dir).unwrap();
    let right_graph = ChangeGraph::open(&right_dir).unwrap();

    fs::write(left_dir.join("left"), "left\n").unwrap();
    let left = left_graph
        .capture("left", provenance("left"), None)
        .unwrap();
    fs::write(right_dir.join("right"), "right\n").unwrap();
    let right = right_graph
        .capture("right", provenance("right"), None)
        .unwrap();

    let outcome = graph.join("integration", &left.id, &right.id).unwrap();
    let JoinOutcome::Joined { node, base, tree } = outcome else {
        panic!("expected a clean join, got {outcome:?}");
    };
    assert_eq!(base, root.id);
    assert_eq!(node.parents, vec![left.id.clone(), right.id.clone()]);
    assert_eq!(node.tree, tree);
    assert_eq!(node.provenance.tool, "join");
    assert_eq!(fs::read_to_string(cwd.join("left")).unwrap(), "left\n");
    assert_eq!(fs::read_to_string(cwd.join("right")).unwrap(), "right\n");
    assert_eq!(fs::read_to_string(cwd.join("base")).unwrap(), "base\n");
    assert_eq!(graph.state().unwrap().heads["integration"], node.id);
}

#[test]
fn conflicting_join_reports_the_paths_and_creates_no_node() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let left_dir = workspaces.path().join("left");
    let right_dir = workspaces.path().join("right");
    graph.fork("left", &root.id, &left_dir).unwrap();
    graph.fork("right", &root.id, &right_dir).unwrap();
    let left_graph = ChangeGraph::open(&left_dir).unwrap();
    let right_graph = ChangeGraph::open(&right_dir).unwrap();

    fs::write(left_dir.join("shared"), "left\n").unwrap();
    let left = left_graph
        .capture("left", provenance("left"), None)
        .unwrap();
    fs::write(right_dir.join("shared"), "right\n").unwrap();
    let right = right_graph
        .capture("right", provenance("right"), None)
        .unwrap();

    let before = graph.state().unwrap().nodes.len();
    let outcome = graph.join("integration", &left.id, &right.id).unwrap();
    let JoinOutcome::Conflicted { base, conflicts } = outcome else {
        panic!("expected a conflicted join, got {outcome:?}");
    };
    assert_eq!(base, root.id);
    assert_eq!(conflicts, vec!["shared".to_string()]);
    let state = graph.state().unwrap();
    assert_eq!(
        state.nodes.len(),
        before,
        "a conflicted join must not append a node"
    );
    assert!(!state.heads.contains_key("integration"));
    assert!(
        !cwd.join("shared").exists(),
        "a conflicted join must not touch the workspace"
    );
}

#[test]
fn capture_thread_changes_reports_only_net_changes_on_the_fork() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph.mark(&root.id, "base", "feat: base", true).unwrap();

    let workspaces = tempfile::tempdir().unwrap();
    let side_dir = workspaces.path().join("side");
    graph.fork("side", "base", &side_dir).unwrap();
    let side = ChangeGraph::open(&side_dir).unwrap();

    // Reading the fork is not a change.
    assert!(
        side.capture_thread_changes(provenance("side"))
            .unwrap()
            .is_none()
    );

    // An edit no file tool recorded (as from a shell command) is captured.
    fs::write(side_dir.join("base"), "edited\n").unwrap();
    let edited = side
        .capture_thread_changes(provenance("side"))
        .unwrap()
        .expect("the edit is a change");
    assert_eq!(edited.thread, "side");
    assert_eq!(edited.parents, vec![root.id.clone()]);

    // Reverting to the fork's tree leaves no net change to keep.
    fs::write(side_dir.join("base"), "base\n").unwrap();
    assert!(
        side.capture_thread_changes(provenance("side"))
            .unwrap()
            .is_none()
    );
}

#[test]
fn drop_checkpoint_clears_the_name_and_the_mark() {
    let directory = fixture();
    let cwd = directory.path();
    let graph = ChangeGraph::open(cwd).unwrap();
    let root = graph.capture("main", provenance("root"), None).unwrap();
    graph
        .mark(&root.id, "subagent/a", "isolated work of subagent a", false)
        .unwrap();

    assert!(graph.drop_checkpoint("subagent/a").unwrap());
    let state = graph.state().unwrap();
    assert!(!state.checkpoints.contains_key("subagent/a"));
    let node = state.nodes.iter().find(|n| n.id == root.id).unwrap();
    assert!(node.mark.is_none());
    assert!(!graph.drop_checkpoint("subagent/a").unwrap());

    // The node is still a head of main, so gc keeps it and the name stays free.
    assert_eq!(graph.gc().unwrap().pruned_nodes, 0);
    graph
        .mark(&root.id, "subagent/a", "isolated work of subagent a", false)
        .unwrap();
}
