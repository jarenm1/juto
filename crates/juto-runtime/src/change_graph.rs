//! Juto's native live-change DAG. Live points contain immutable file trees, not commits.
//! Named checkpoints project into Git only when exported. Blocking methods run off GPUI.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::git_transfer::{ExportPlan, ExportPoint, ExportResult, GitRepository, Signature};
use crate::replication::{ImportReport, PublishReport};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditProvenance {
    pub session_id: String,
    pub tool: String,
    pub selection_mode: Option<String>,
    pub path: Option<String>,
}

impl EditProvenance {
    fn external() -> Self {
        Self {
            session_id: String::new(),
            tool: "external".into(),
            selection_mode: None,
            path: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mark {
    pub name: String,
    pub message: String,
    /// Records the caller's verification attestation, not a fabricated test result.
    pub verified: bool,
    pub signature: Signature,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeNode {
    pub id: String,
    pub thread: String,
    pub parents: Vec<String>,
    pub tree: String,
    pub provenance: EditProvenance,
    pub mark: Option<Mark>,
}

/// The result of [`ChangeGraph::join`].
///
/// A clean join materializes the merged tree into the workspace and appends a
/// node with both sides as parents. A conflicted join changes nothing: the
/// caller resolves the listed paths and retries.
///
/// `base` is the merge base: the id of the maximal common ancestor node, or the
/// graph's base tree id when the two sides share no node ancestor (both were
/// captured directly from the base).
///
/// The `Joined` variant carries the whole node by value because callers consume
/// it directly; boxing it would only add indirection to a cold path.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinOutcome {
    Joined {
        node: ChangeNode,
        base: String,
        tree: String,
    },
    Conflicted {
        base: String,
        conflicts: Vec<String>,
    },
}

/// What [`ChangeGraph::reconcile`] did with one imported remote head.
///
/// The `Joined` variant carries the whole node by value, matching
/// [`JoinOutcome`]; boxing it would only add indirection to a cold path.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconcileOutcome {
    /// The remote head merged cleanly into the local thread; the remote head was
    /// retired and the thread advanced to `node`.
    Joined {
        remote: String,
        node: ChangeNode,
        base: String,
        tree: String,
    },
    /// The thread had no local head, so the remote head was adopted as the
    /// thread's head and the owner-qualified key was retired. No merge was
    /// needed: there was no competing local history.
    Adopted { remote: String, node: ChangeNode },
    /// The remote head conflicts with the local thread. Nothing changed; both
    /// heads are still present and the conflicting paths are reported.
    Conflicted {
        remote: String,
        base: String,
        conflicts: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphState {
    pub version: u32,
    /// Existing Git history anchors exports; all zeros means a new repository.
    pub base_commit: String,
    pub base_tree: String,
    /// Append order is topological. Native node IDs and contents never change.
    pub nodes: Vec<ChangeNode>,
    pub heads: BTreeMap<String, String>,
    pub workspace_heads: BTreeMap<String, String>,
    pub checkpoints: BTreeMap<String, String>,
}

/// On-disk journal format. Version 3 splits the graph into a small, frequently
/// rewritten header (`graph.json`) and an append-only node journal
/// (`nodes.jsonl`), so recording an edit no longer rewrites every node.
const GRAPH_VERSION: u32 = 3;

/// The compact, frequently rewritten part of a version 3 graph.
///
/// It carries everything except the nodes, which live in `nodes.jsonl` in
/// topological (append) order.
#[derive(Serialize, Deserialize)]
struct GraphHeader {
    version: u32,
    base_commit: String,
    base_tree: String,
    heads: BTreeMap<String, String>,
    workspace_heads: BTreeMap<String, String>,
    checkpoints: BTreeMap<String, String>,
}

impl From<&GraphState> for GraphHeader {
    fn from(state: &GraphState) -> Self {
        Self {
            version: GRAPH_VERSION,
            base_commit: state.base_commit.clone(),
            base_tree: state.base_tree.clone(),
            heads: state.heads.clone(),
            workspace_heads: state.workspace_heads.clone(),
            checkpoints: state.checkpoints.clone(),
        }
    }
}

/// What a [`ChangeGraph::gc`] pass pruned and retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GcReport {
    /// Nodes swept out of the journal because no head, workspace head, or
    /// checkpoint reaches them.
    pub pruned_nodes: usize,
    /// Nodes kept because a head, workspace head, or checkpoint reaches them.
    pub retained_nodes: usize,
    /// Snapshot refs actually deleted by the sweep. A pruned node whose ref was
    /// already gone does not count.
    pub released_refs: usize,
    /// `refs/juto/nodes/<id>` refs actually deleted by the sweep, so a pruned
    /// node leaves no replication ref behind.
    pub released_node_refs: usize,
}

pub struct ChangeGraph {
    repository: GitRepository,
    directory: PathBuf,
}

impl ChangeGraph {
    pub fn open(workspace: &Path) -> Result<Self> {
        Self::initialize(workspace, None)
    }

    /// Initialize from existing Git HEAD or an explicit immutable Git commit ID.
    /// Live history belongs to Juto; Git branches/index/HEAD are not changed.
    pub fn initialize(workspace: &Path, base: Option<&str>) -> Result<Self> {
        let repository = GitRepository::open(workspace)?;
        let directory = repository.git_dir().join("juto");
        fs::create_dir_all(&directory)?;
        let graph = Self {
            repository,
            directory,
        };
        let _lock = graph.lock()?;
        if graph.graph_path().exists() || graph.legacy_path().exists() {
            let state = graph.load()?;
            if let Some(base) = base {
                ensure!(
                    state.base_commit == base,
                    "graph already initialized with a different base"
                );
            }
        } else {
            let base_commit = match base {
                Some(base) => base.to_owned(),
                None => graph
                    .repository
                    .head_commit()?
                    .unwrap_or_else(|| "0".repeat(40)),
            };
            let base_tree = graph.repository.snapshot_tree(&base_commit)?;
            graph.repository.retain_snapshot("base-tree", &base_tree)?;
            if !base_commit.bytes().all(|b| b == b'0') {
                graph.repository.retain_snapshot("base", &base_commit)?;
            }
            let state = GraphState {
                version: GRAPH_VERSION,
                base_commit,
                base_tree,
                nodes: Vec::new(),
                heads: BTreeMap::new(),
                workspace_heads: BTreeMap::new(),
                checkpoints: BTreeMap::new(),
            };
            graph.write_nodes(&state.nodes)?;
            graph.save_header(&state)?;
        }
        Ok(graph)
    }

    pub fn workspace(&self) -> &Path {
        self.repository.workspace()
    }
    pub fn thread(&self) -> Option<&str> {
        self.repository.thread()
    }

    pub fn state(&self) -> Result<GraphState> {
        let _lock = self.lock()?;
        self.load()
    }

    /// Caller holds the native workspace mutation lock while capturing.
    pub fn snapshot_tree(&self) -> Result<String> {
        let state = self.state()?;
        let baseline = state
            .workspace_heads
            .get(&self.workspace_key())
            .or_else(|| self.repository.thread().and_then(|t| state.heads.get(t)))
            .and_then(|id| state.nodes.iter().find(|n| &n.id == id))
            .map(|n| n.tree.as_str())
            .unwrap_or(&state.base_tree);
        self.repository.capture_tree(baseline)
    }

    pub fn path_tracked(&self, tree: &str, path: &[u8]) -> Result<bool> {
        self.repository.path_tracked(tree, path)
    }

    pub fn path_ignored(&self, path: &[u8]) -> Result<bool> {
        self.repository.path_ignored(path)
    }

    pub fn diff_trees(&self, before: &str, after: &str, path: &[u8]) -> Result<String> {
        self.repository.diff_trees(before, after, path)
    }

    /// Tool edits create native points in the owning session thread (or a fork's
    /// assigned thread). External pre-existing changes are never attributed to a tool.
    pub fn record_edit(
        &self,
        before_tree: &str,
        after_tree: &str,
        provenance: EditProvenance,
    ) -> Result<Option<ChangeNode>> {
        ensure!(
            self.repository.snapshot_tree(before_tree)? == before_tree,
            "before snapshot must be a tree ID"
        );
        ensure!(
            self.repository.snapshot_tree(after_tree)? == after_tree,
            "after snapshot must be a tree ID"
        );
        let thread = self
            .repository
            .thread()
            .unwrap_or(&provenance.session_id)
            .to_owned();
        validate_thread_name(&thread)?;
        let _lock = self.lock()?;
        let mut state = self.load()?;
        let parents = self.continuity_parents(&state, &thread, before_tree);
        let before_is_known = if parents.is_empty() {
            state.base_tree == before_tree
        } else {
            parents.iter().any(|id| {
                state
                    .nodes
                    .iter()
                    .any(|n| &n.id == id && n.tree == before_tree)
            })
        };
        // Only a node this call actually created may advance a head. Falling
        // back to `parents.first()` would move the shared workspace head back to
        // an older thread head in a shared working directory or a cross-thread
        // fork, dropping later lineage from the projection.
        let mut created: Option<String> = None;
        let parents = if !before_is_known {
            let external = self.append(
                &mut state,
                &thread,
                before_tree.into(),
                parents,
                EditProvenance::external(),
            )?;
            created = Some(external.id.clone());
            vec![external.id]
        } else {
            parents
        };
        let node = if before_tree == after_tree {
            None
        } else {
            let node = self.append(
                &mut state,
                &thread,
                after_tree.into(),
                parents.clone(),
                provenance,
            )?;
            created = Some(node.id.clone());
            Some(node)
        };
        if let Some(head) = created {
            state.heads.insert(thread, head.clone());
            state.workspace_heads.insert(self.workspace_key(), head);
        }
        self.save_header(&state)?;
        Ok(node)
    }

    /// Capture shell/manual edits. Explicit parents declare a resolved join;
    /// the caller must integrate file contents first. No merge success is invented.
    pub fn capture(
        &self,
        thread: &str,
        provenance: EditProvenance,
        parents: Option<Vec<String>>,
    ) -> Result<ChangeNode> {
        validate_thread_name(thread)?;
        if let Some(assigned) = self.repository.thread() {
            ensure!(
                thread == assigned,
                "workspace belongs to thread {assigned:?}"
            );
        }
        let _workspace_lock =
            crate::vcs::MutationLock::acquire(self.workspace()).map_err(anyhow::Error::msg)?;
        let tree = self.snapshot_tree()?;
        let _lock = self.lock()?;
        let mut state = self.load()?;
        let explicit = parents.is_some();
        let parents = parents.unwrap_or_else(|| self.continuity_parents(&state, thread, &tree));
        validate_parents(&state, &parents)?;
        if !explicit && let Some(id) = parents.first() {
            let previous = state
                .nodes
                .iter()
                .find(|n| &n.id == id)
                .context("missing thread head")?;
            if previous.tree == tree {
                return Ok(previous.clone());
            }
        }
        let node = self.append(&mut state, thread, tree, parents, provenance)?;
        state.heads.insert(thread.into(), node.id.clone());
        state
            .workspace_heads
            .insert(self.workspace_key(), node.id.clone());
        self.save_header(&state)?;
        Ok(node)
    }

    /// Fork an independent native workspace from an immutable live point.
    /// No Git branch/worktree/commit is created; its edits share only the object store.
    pub fn fork(&self, thread: &str, from: &str, destination: &Path) -> Result<GraphState> {
        self.fork_inner(thread, from, destination, None)
    }

    /// Fork a live point into a partial workspace holding only `paths`.
    ///
    /// The fork materializes just those subtrees, so an agent scoped to one
    /// crate in a monorepo does not pay for the whole tree. Capture preserves
    /// every baseline path outside `paths`, so the thread still records a
    /// complete tree.
    pub fn fork_sparse(
        &self,
        thread: &str,
        from: &str,
        destination: &Path,
        paths: &[String],
    ) -> Result<GraphState> {
        self.fork_inner(thread, from, destination, Some(paths))
    }

    fn fork_inner(
        &self,
        thread: &str,
        from: &str,
        destination: &Path,
        sparse: Option<&[String]>,
    ) -> Result<GraphState> {
        validate_thread_name(thread)?;
        let _lock = self.lock()?;
        let mut state = self.load()?;
        ensure!(!state.heads.contains_key(thread), "thread already exists");
        // Reclaim fork directories left behind by interrupted provisions. This
        // is best-effort housekeeping: a cleanup failure must never block
        // provisioning the fork the caller actually asked for.
        if let Err(error) = self.repository.cleanup_orphaned_forks() {
            eprintln!("juto: failed to reclaim orphaned fork directories: {error:#}");
        }
        let from = state
            .checkpoints
            .get(from)
            .map(String::as_str)
            .unwrap_or(from)
            .to_owned();
        let node = state
            .nodes
            .iter()
            .find(|n| n.id == from)
            .context("unknown fork point")?;
        match sparse {
            Some(paths) => {
                self.repository
                    .fork_workspace_sparse(&node.tree, destination, thread, paths)?
            }
            None => self
                .repository
                .fork_workspace(&node.tree, destination, thread)?,
        }
        let recorded = (|| -> Result<()> {
            let destination = destination.canonicalize()?;
            state.heads.insert(thread.into(), from.clone());
            state
                .workspace_heads
                .insert(destination.to_string_lossy().into_owned(), from.clone());
            self.save_header(&state)
        })();
        match recorded {
            Ok(()) => Ok(state),
            Err(error) => {
                // `fork_workspace` created `destination` atomically in this
                // call, so it is known-new and safe to remove. `save` is atomic,
                // so only remove it when the bookkeeping truly did not land;
                // otherwise the thread would point at a missing directory.
                let landed = self
                    .load()
                    .map(|state| state.heads.contains_key(thread))
                    .unwrap_or(false);
                if !landed {
                    let _ = fs::remove_dir_all(destination);
                }
                Err(error.context("workspace created but native thread bookkeeping failed"))
            }
        }
    }

    /// Freeze a named point, including exact tree, message, identity and timestamp.
    pub fn mark(&self, id: &str, name: &str, message: &str, verified: bool) -> Result<ChangeNode> {
        ensure!(!name.trim().is_empty(), "checkpoint name must not be empty");
        ensure!(
            !message.trim().is_empty(),
            "commit message must not be empty"
        );
        let _lock = self.lock()?;
        let mut state = self.load()?;
        if let Some(existing) = state.checkpoints.get(name) {
            ensure!(
                existing == id,
                "checkpoint name already belongs to another point"
            );
        }
        let node = state
            .nodes
            .iter_mut()
            .find(|n| n.id == id)
            .context("unknown change node")?;
        if let Some(mark) = &node.mark {
            ensure!(
                mark.name == name && mark.message == message && mark.verified == verified,
                "point already named; checkpoints are immutable"
            );
            return Ok(node.clone());
        }
        node.mark = Some(Mark {
            name: name.into(),
            message: message.into(),
            verified,
            signature: self.repository.signature()?,
        });
        let result = node.clone();
        state.checkpoints.insert(name.into(), id.into());
        // A mark edits an existing node in place, so the append-only journal
        // cannot express it: rewrite the journal atomically, then the header.
        self.write_nodes(&state.nodes)?;
        self.save_header(&state)?;
        Ok(result)
    }

    pub fn export(&self, checkpoint: &str, branch: &str) -> Result<ExportResult> {
        let state = self.state()?;
        let tip = state
            .checkpoints
            .get(checkpoint)
            .map(String::as_str)
            .unwrap_or(checkpoint);
        self.repository.export(&project(&state, tip)?, branch)
    }

    pub fn push(&self, branch: &str, remote: &str) -> Result<()> {
        self.repository.push(branch, remote)
    }

    /// This repository's stable replication identity.
    ///
    /// A UUIDv7 stored in `<git-dir>/juto/machine-id`, created on first use
    /// under the graph lock. It is local and never replicated; every workspace
    /// sharing this Git directory (including native forks) shares the identity.
    pub fn machine_id(&self) -> Result<String> {
        let _lock = self.lock()?;
        self.machine_id_locked()
    }

    /// Publish the whole graph to `remote` as `refs/juto/*` refs.
    pub fn push_graph(&self, remote: &str) -> Result<PublishReport> {
        crate::replication::publish(self, remote)
    }

    /// Fetch `refs/juto/*` from `remote` and union it into the local journal.
    pub fn fetch_graph(&self, remote: &str) -> Result<ImportReport> {
        crate::replication::fetch_graph(self, remote)
    }

    /// Import already-fetched staging refs. Performs no network access.
    pub fn import_refs(&self, staging: &str) -> Result<ImportReport> {
        crate::replication::import_refs(self, staging)
    }

    /// Join two threads' trees into a new node with both sides as parents.
    ///
    /// The merge base is the maximal common ancestor of `left` and `right`. A
    /// conflicted merge changes nothing; a clean merge materializes the merged
    /// tree into this workspace and appends a two-parent node.
    pub fn join(&self, thread: &str, left: &str, right: &str) -> Result<JoinOutcome> {
        validate_thread_name(thread)?;
        let state = self.state()?;
        let left_node = resolve_node(&state, left)?;
        let right_node = resolve_node(&state, right)?;
        let (base_id, base_tree) = match merge_base(&state, &left_node.id, &right_node.id)? {
            MergeBase::Node(id) => {
                let tree = state
                    .nodes
                    .iter()
                    .find(|n| n.id == id)
                    .context("missing merge base")?
                    .tree
                    .clone();
                (id, tree)
            }
            // Both sides were captured straight from the base, so the base tree
            // is the three-way merge base.
            MergeBase::Base => (state.base_tree.clone(), state.base_tree.clone()),
        };
        ensure!(
            self.repository.supports_tree_merge()?,
            "this Git does not support three-way tree merges (needs git >= 2.45)"
        );
        let merged = self
            .repository
            .merge_trees(&base_tree, &left_node.tree, &right_node.tree)?;
        if !merged.conflicts.is_empty() {
            return Ok(JoinOutcome::Conflicted {
                base: base_id,
                conflicts: merged.conflicts,
            });
        }
        let merged_tree = merged.tree;
        let left_id = left_node.id.clone();
        let right_id = right_node.id.clone();
        // The workspace materialization is serialized against other mutations,
        // but `capture` takes the same lock itself, so the guard is released
        // before the node is appended.
        {
            let _lock =
                crate::vcs::MutationLock::acquire(self.workspace()).map_err(anyhow::Error::msg)?;
            let current = self.snapshot_tree()?;
            self.repository
                .restore_tree(&current, &merged_tree, self.workspace())?;
        }
        let node = self.capture(
            thread,
            EditProvenance {
                session_id: String::new(),
                tool: "join".into(),
                selection_mode: None,
                path: None,
            },
            Some(vec![left_id, right_id]),
        )?;
        ensure!(
            node.tree == merged_tree,
            "joined node tree does not match the merged tree"
        );
        Ok(JoinOutcome::Joined {
            node,
            base: base_id,
            tree: merged_tree,
        })
    }

    /// Retire a thread head, reporting whether it existed.
    ///
    /// Only the head pointer is dropped; no node is deleted. Used to retire the
    /// `<owner>/<thread>` head an import creates once its history has been
    /// joined into the local thread.
    pub fn drop_head(&self, head: &str) -> Result<bool> {
        validate_head_name(head)?;
        let _lock = self.lock()?;
        let mut state = self.load_locked()?;
        if state.heads.remove(head).is_none() {
            return Ok(false);
        }
        self.save_locked(&state)?;
        Ok(true)
    }

    /// Join every imported `<owner>/<thread>` head back into `thread`.
    ///
    /// When two machines advance the same thread, [`ChangeGraph::fetch_graph`]
    /// keeps the local head under `<thread>` and the remote head under
    /// `<owner>/<thread>` rather than overwriting either. This joins each such
    /// remote head into the local thread: a clean merge advances the thread and
    /// retires the remote head, a conflicting merge changes nothing and is
    /// reported. Returns one outcome per remote head, in head-name order.
    pub fn reconcile(&self, thread: &str) -> Result<Vec<ReconcileOutcome>> {
        validate_thread_name(thread)?;
        let suffix = format!("/{thread}");
        let remotes: Vec<String> = self
            .state()?
            .heads
            .keys()
            .filter(|key| key.ends_with(&suffix))
            .cloned()
            .collect();
        let mut outcomes = Vec::new();
        for remote in remotes {
            // Re-read each round: a clean join advances the local head.
            let state = self.state()?;
            let Some(remote_head) = state.heads.get(&remote).cloned() else {
                continue;
            };
            let Some(local) = state.heads.get(thread).cloned() else {
                // No local history for this thread: adopt the remote head
                // rather than leaving it stranded under the owner-qualified key.
                let node = state
                    .nodes
                    .iter()
                    .find(|n| n.id == remote_head)
                    .cloned()
                    .with_context(|| {
                        format!("head {remote} names an unknown node {remote_head}")
                    })?;
                self.adopt_head(thread, &remote)?;
                outcomes.push(ReconcileOutcome::Adopted { remote, node });
                continue;
            };
            match self.join(thread, &local, &remote_head)? {
                JoinOutcome::Joined { node, base, tree } => {
                    self.drop_head(&remote)?;
                    outcomes.push(ReconcileOutcome::Joined {
                        remote,
                        node,
                        base,
                        tree,
                    });
                }
                JoinOutcome::Conflicted { base, conflicts } => {
                    outcomes.push(ReconcileOutcome::Conflicted {
                        remote,
                        base,
                        conflicts,
                    });
                }
            }
        }
        Ok(outcomes)
    }

    /// Move `remote`'s head onto `thread`, retiring the owner-qualified key.
    fn adopt_head(&self, thread: &str, remote: &str) -> Result<()> {
        let _lock = self.lock()?;
        let mut state = self.load_locked()?;
        let Some(head) = state.heads.remove(remote) else {
            return Ok(());
        };
        state.heads.insert(thread.to_string(), head);
        self.save_locked(&state)?;
        Ok(())
    }

    /// Revert `thread` to an earlier point by materializing its tree.
    ///
    /// The rewind is recorded as a *forward* node whose tree equals the target,
    /// so the audit trail survives and a head never moves backward silently.
    pub fn rewind(&self, thread: &str, to: &str) -> Result<ChangeNode> {
        validate_thread_name(thread)?;
        let state = self.state()?;
        let target = resolve_node(&state, to)?;
        let target_tree = target.tree.clone();
        let current_head = state.heads.get(thread).cloned();
        // As in `join`, the mutation lock covers the workspace materialization
        // only; `capture` re-acquires it for the append.
        {
            let _lock =
                crate::vcs::MutationLock::acquire(self.workspace()).map_err(anyhow::Error::msg)?;
            let current = self.snapshot_tree()?;
            self.repository
                .restore_tree(&current, &target_tree, self.workspace())?;
        }
        let node = self.capture(
            thread,
            EditProvenance {
                session_id: String::new(),
                tool: "rewind".into(),
                selection_mode: None,
                path: None,
            },
            current_head.map(|head| vec![head]),
        )?;
        ensure!(
            node.tree == target_tree,
            "rewound node tree does not match the target tree"
        );
        Ok(node)
    }

    /// The Git repository backing this graph. Used by `replication`.
    pub(crate) fn repository(&self) -> &GitRepository {
        &self.repository
    }

    /// Acquire the graph lock. Callers must not call [`ChangeGraph::machine_id`]
    /// or any other locking method while holding it.
    pub(crate) fn graph_lock(&self) -> Result<File> {
        self.lock()
    }

    /// Read the graph while the caller holds the graph lock.
    pub(crate) fn load_locked(&self) -> Result<GraphState> {
        self.load()
    }

    /// Rewrite the header while the caller holds the graph lock.
    pub(crate) fn save_locked(&self, state: &GraphState) -> Result<()> {
        self.save_header(state)
    }

    /// Rewrite the node journal while the caller holds the graph lock.
    pub(crate) fn write_nodes_locked(&self, nodes: &[ChangeNode]) -> Result<()> {
        self.write_nodes(nodes)
    }

    /// Read or create the replication identity while the caller holds the lock.
    pub(crate) fn machine_id_locked(&self) -> Result<String> {
        let path = self.directory.join("machine-id");
        if let Ok(existing) = fs::read_to_string(&path) {
            let trimmed = existing.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }
        let id = Uuid::now_v7().to_string();
        self.atomic_write(&path, |file| {
            file.write_all(id.as_bytes())?;
            file.write_all(b"\n")?;
            Ok(())
        })?;
        Ok(id)
    }

    /// Sweep nodes no live reference reaches and release their snapshot refs.
    ///
    /// The keep-set is every node reachable by walking parents from every value
    /// in `heads`, `workspace_heads`, and `checkpoints`. The state is reloaded
    /// under the graph lock, so a stale `graph.json` can never cause a live
    /// head or checkpoint to be pruned. Snapshot refs whose ids do not name a
    /// node (the `base` and `base-tree` anchors) are never touched.
    pub fn gc(&self) -> Result<GcReport> {
        let _lock = self.lock()?;
        let state = self.load()?;
        let by_id: BTreeMap<&str, &ChangeNode> =
            state.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
        let mut keep: BTreeSet<&str> = BTreeSet::new();
        let mut stack: Vec<&str> = state
            .heads
            .values()
            .chain(state.workspace_heads.values())
            .chain(state.checkpoints.values())
            .map(String::as_str)
            .collect();
        while let Some(id) = stack.pop() {
            if !keep.insert(id) {
                continue;
            }
            if let Some(node) = by_id.get(id) {
                stack.extend(node.parents.iter().map(String::as_str));
            }
        }
        let (retained, pruned): (Vec<ChangeNode>, Vec<ChangeNode>) = state
            .nodes
            .iter()
            .cloned()
            .partition(|node| keep.contains(node.id.as_str()));
        if pruned.is_empty() {
            return Ok(GcReport {
                pruned_nodes: 0,
                retained_nodes: retained.len(),
                released_refs: 0,
                released_node_refs: 0,
            });
        }
        let mut released_refs = 0;
        let mut released_node_refs = 0;
        for node in &pruned {
            if self.repository.release_snapshot(&node.id)? {
                released_refs += 1;
            }
            // A replicated node also pins its metadata under `refs/juto/nodes`;
            // releasing only the snapshot ref would leak that ref forever.
            if self
                .repository
                .release_ref(&crate::replication::node_ref(&node.id))?
            {
                released_node_refs += 1;
            }
        }
        self.write_nodes(&retained)?;
        self.save_header(&state)?;
        Ok(GcReport {
            pruned_nodes: pruned.len(),
            retained_nodes: retained.len(),
            released_refs,
            released_node_refs,
        })
    }

    fn continuity_parents(
        &self,
        state: &GraphState,
        thread: &str,
        before_tree: &str,
    ) -> Vec<String> {
        let mut parents = Vec::new();
        if let Some(head) = state.heads.get(thread) {
            parents.push(head.clone());
        }
        // A shared working directory creates real cross-thread dependencies.
        // Independent native fork directories have their own workspace head.
        if let Some(head) = state.workspace_heads.get(&self.workspace_key())
            && !parents.contains(head)
        {
            parents.push(head.clone());
        }
        if parents.is_empty()
            && let Some(node) = state.nodes.iter().rev().find(|n| n.tree == before_tree)
        {
            parents.push(node.id.clone());
        }
        parents
    }

    fn append(
        &self,
        state: &mut GraphState,
        thread: &str,
        tree: String,
        parents: Vec<String>,
        provenance: EditProvenance,
    ) -> Result<ChangeNode> {
        validate_parents(state, &parents)?;
        let node = ChangeNode {
            id: Uuid::now_v7().to_string(),
            thread: thread.into(),
            parents,
            tree,
            provenance,
            mark: None,
        };
        // Journal first: a node line without a snapshot ref is recoverable (gc
        // prunes the unreachable node), while a ref without a node line leaks
        // forever because gc only releases refs named by a node.
        self.append_node(&node)?;
        self.repository.retain_snapshot(&node.id, &node.tree)?;
        state.nodes.push(node.clone());
        Ok(node)
    }

    fn workspace_key(&self) -> String {
        self.workspace().to_string_lossy().into_owned()
    }
    /// The compact header rewritten by every mutation.
    fn graph_path(&self) -> PathBuf {
        self.directory.join("graph.json")
    }
    /// The append-only node journal, in topological (append) order.
    fn nodes_path(&self) -> PathBuf {
        self.directory.join("nodes.jsonl")
    }
    /// The version 2 single-file graph, read once and then migrated.
    fn legacy_path(&self) -> PathBuf {
        self.directory.join("change-graph.json")
    }
    fn lock(&self) -> Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.directory.join("change-graph.lock"))?;
        file.lock_exclusive()?;
        Ok(file)
    }

    /// Read the graph, migrating a version 2 file to the version 3 layout.
    ///
    /// Callers hold the graph lock: migration writes the new files and then
    /// deletes the legacy one, so a downgrade cannot resurrect a stale graph.
    fn load(&self) -> Result<GraphState> {
        let state = if self.graph_path().exists() {
            self.read_v3()?
        } else if self.legacy_path().exists() {
            self.migrate_legacy()?
        } else {
            bail!("native change graph is missing; refusing to replace it");
        };
        validate_state(&state)?;
        Ok(state)
    }

    fn read_v3(&self) -> Result<GraphState> {
        let header: GraphHeader = serde_json::from_reader(File::open(self.graph_path())?)
            .context("invalid native change graph header; refusing to replace it")?;
        ensure!(
            header.version == GRAPH_VERSION,
            "unsupported native change graph version {}",
            header.version
        );
        let journal = File::open(self.nodes_path())
            .context("native change graph is missing its node journal")?;
        let mut nodes = Vec::new();
        for line in BufReader::new(journal).lines() {
            let line = line.context("native change graph journal is unreadable")?;
            if line.trim().is_empty() {
                continue;
            }
            nodes.push(
                serde_json::from_str(&line)
                    .context("invalid native change graph node; refusing to replace it")?,
            );
        }
        Ok(GraphState {
            version: header.version,
            base_commit: header.base_commit,
            base_tree: header.base_tree,
            nodes,
            heads: header.heads,
            workspace_heads: header.workspace_heads,
            checkpoints: header.checkpoints,
        })
    }

    fn migrate_legacy(&self) -> Result<GraphState> {
        let legacy: GraphState = serde_json::from_reader(File::open(self.legacy_path())?)
            .context("invalid native change graph; refusing to replace it")?;
        ensure!(
            legacy.version == 2,
            "unsupported native change graph version {}",
            legacy.version
        );
        let state = GraphState {
            version: GRAPH_VERSION,
            ..legacy
        };
        self.write_nodes(&state.nodes)?;
        self.save_header(&state)?;
        // Both version 3 files are fsynced and renamed into place before the
        // legacy file is dropped, so the migration is crash-safe.
        fs::remove_file(self.legacy_path())?;
        File::open(&self.directory)?.sync_all()?;
        Ok(state)
    }

    /// Append one node line to the journal and flush it to disk.
    fn append_node(&self, node: &ChangeNode) -> Result<()> {
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.nodes_path())?;
        serde_json::to_writer(&mut file, node)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }

    /// Rewrite the whole journal atomically. Only `mark` and `gc` need this.
    fn write_nodes(&self, nodes: &[ChangeNode]) -> Result<()> {
        self.atomic_write(&self.nodes_path(), |file| {
            for node in nodes {
                serde_json::to_writer(&mut *file, node)?;
                file.write_all(b"\n")?;
            }
            Ok(())
        })
    }

    fn save_header(&self, state: &GraphState) -> Result<()> {
        let header = GraphHeader::from(state);
        self.atomic_write(&self.graph_path(), |file| {
            serde_json::to_writer(&mut *file, &header)?;
            file.write_all(b"\n")?;
            Ok(())
        })
    }

    /// Write `path` through a same-directory temporary file and rename it into
    /// place, fsyncing both the file and the directory.
    fn atomic_write(&self, path: &Path, write: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
        let temporary = self
            .directory
            .join(format!(".graph-{}.tmp", Uuid::now_v7()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            write(&mut file)?;
            file.sync_all()?;
            fs::rename(&temporary, path)?;
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }
}

pub(crate) fn validate_state(state: &GraphState) -> Result<()> {
    ensure!(
        state.version == GRAPH_VERSION,
        "unsupported native change graph version {}",
        state.version
    );
    let mut seen = BTreeSet::new();
    for node in &state.nodes {
        ensure!(
            node.parents.iter().all(|p| seen.contains(p)),
            "graph has missing or non-topological parents"
        );
        ensure!(seen.insert(node.id.clone()), "duplicate graph node ID");
    }
    ensure!(
        state
            .heads
            .values()
            .chain(state.workspace_heads.values())
            .all(|h| seen.contains(h)),
        "graph has missing head"
    );
    for (name, id) in &state.checkpoints {
        ensure!(
            state
                .nodes
                .iter()
                .any(|n| &n.id == id && n.mark.as_ref().is_some_and(|m| &m.name == name)),
            "invalid checkpoint mapping"
        );
    }
    Ok(())
}

/// Reject thread names that would make an owner-qualified head ref ambiguous.
///
/// Replication attaches the publishing owner as a `<owner>/<thread>` prefix, so
/// a local thread name containing `/` could collide with a qualified name.
fn validate_thread_name(thread: &str) -> Result<()> {
    ensure!(!thread.trim().is_empty(), "thread must not be empty");
    ensure!(
        !thread.contains('/'),
        "thread name must not contain '/': {thread:?}"
    );
    Ok(())
}

/// Validate a head key, which may be owner-qualified (`<owner>/<thread>`).
///
/// A local thread name never contains `/`, so a head key carries at most one
/// separator: the one an import inserts between the owning machine and the
/// thread it advanced.
fn validate_head_name(head: &str) -> Result<()> {
    ensure!(!head.trim().is_empty(), "head name must not be empty");
    ensure!(
        !head.starts_with('/') && !head.ends_with('/'),
        "head name must not start or end with '/': {head:?}"
    );
    ensure!(
        head.matches('/').count() <= 1,
        "head name must be a thread or an owner-qualified thread: {head:?}"
    );
    Ok(())
}

/// Resolve a node id or checkpoint name to its node.
fn resolve_node<'a>(state: &'a GraphState, spec: &str) -> Result<&'a ChangeNode> {
    let id = state
        .checkpoints
        .get(spec)
        .map(String::as_str)
        .unwrap_or(spec);
    state
        .nodes
        .iter()
        .find(|n| n.id == id)
        .with_context(|| format!("unknown change node {spec:?}"))
}

/// Every node reachable from `id` by walking parents, including `id` itself.
fn ancestor_closure(state: &GraphState, id: &str) -> BTreeSet<String> {
    let by_id: BTreeMap<&str, &ChangeNode> =
        state.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut seen = BTreeSet::new();
    let mut stack = vec![id];
    while let Some(current) = stack.pop() {
        if !seen.insert(current.to_string()) {
            continue;
        }
        if let Some(node) = by_id.get(current) {
            stack.extend(node.parents.iter().map(String::as_str));
        }
    }
    seen
}

/// The merge base of two nodes: the maximal common ancestors of both.
///
/// Exactly one maximal ancestor is the usual case. Several (a criss-cross
/// merge) resolve to the one with the greatest index in `state.nodes`, which is
/// the most recently appended and therefore the most conservative choice.
/// When the two nodes share no node ancestor, the graph base tree is the base.
enum MergeBase {
    Node(String),
    Base,
}

fn merge_base(state: &GraphState, left: &str, right: &str) -> Result<MergeBase> {
    let common: BTreeSet<String> = ancestor_closure(state, left)
        .intersection(&ancestor_closure(state, right))
        .cloned()
        .collect();
    if common.is_empty() {
        return Ok(MergeBase::Base);
    }
    let mut dominated: BTreeSet<String> = BTreeSet::new();
    for id in &common {
        for ancestor in ancestor_closure(state, id) {
            if &ancestor != id {
                dominated.insert(ancestor);
            }
        }
    }
    let mut maximal: Vec<&String> = common
        .iter()
        .filter(|id| !dominated.contains(*id))
        .collect();
    ensure!(!maximal.is_empty(), "cannot determine a merge base");
    let index = |id: &str| {
        state
            .nodes
            .iter()
            .position(|n| n.id == id)
            .unwrap_or(usize::MAX)
    };
    maximal.sort_by_key(|id| index(id));
    let base = maximal.last().expect("maximal is non-empty").to_string();
    if maximal.len() > 1 {
        eprintln!(
            "juto: join chose merge base {base} among {} common ancestors",
            maximal.len()
        );
    }
    Ok(MergeBase::Node(base))
}

fn validate_parents(state: &GraphState, parents: &[String]) -> Result<()> {
    let unique: BTreeSet<&str> = parents.iter().map(String::as_str).collect();
    ensure!(unique.len() == parents.len(), "duplicate parent");
    ensure!(
        parents
            .iter()
            .all(|p| state.nodes.iter().any(|n| &n.id == p)),
        "unknown parent change"
    );
    Ok(())
}

fn project(state: &GraphState, tip: &str) -> Result<ExportPlan> {
    let nodes: BTreeMap<&str, &ChangeNode> =
        state.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    ensure!(
        nodes
            .get(tip)
            .context("unknown change node")?
            .mark
            .is_some(),
        "export tip must be a named checkpoint"
    );
    let mut ancestors = BTreeSet::new();
    let mut stack = vec![tip];
    while let Some(id) = stack.pop() {
        if ancestors.insert(id) {
            stack.extend(
                nodes
                    .get(id)
                    .context("missing graph parent")?
                    .parents
                    .iter()
                    .map(String::as_str),
            );
        }
    }
    let mut frontier: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut marked_ancestors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut points = Vec::new();
    for node in state
        .nodes
        .iter()
        .filter(|n| ancestors.contains(n.id.as_str()))
    {
        let mut parents = BTreeSet::new();
        let mut all_marks = BTreeSet::new();
        for parent in &node.parents {
            parents.extend(
                frontier
                    .get(parent.as_str())
                    .context("non-topological graph")?
                    .iter()
                    .copied(),
            );
            all_marks.extend(
                marked_ancestors
                    .get(parent.as_str())
                    .context("non-topological graph")?
                    .iter()
                    .copied(),
            );
            if nodes[parent.as_str()].mark.is_some() {
                all_marks.insert(parent.as_str());
            }
        }
        let redundant: BTreeSet<&str> = parents
            .iter()
            .flat_map(|p| {
                marked_ancestors
                    .get(p)
                    .into_iter()
                    .flat_map(|a| a.iter().copied())
            })
            .collect();
        parents.retain(|p| !redundant.contains(p));
        if let Some(mark) = &node.mark {
            points.push(ExportPoint {
                id: node.id.clone(),
                tree: node.tree.clone(),
                parents: parents.iter().map(|p| (*p).into()).collect(),
                message: mark.message.clone(),
                signature: mark.signature.clone(),
            });
            frontier.insert(&node.id, BTreeSet::from([node.id.as_str()]));
        } else {
            frontier.insert(&node.id, parents);
        }
        marked_ancestors.insert(&node.id, all_marks);
    }
    Ok(ExportPlan {
        base_commit: state.base_commit.clone(),
        points,
        tip: tip.into(),
    })
}
