//! Juto's native live-change DAG. Live points contain immutable file trees, not commits.
//! Named checkpoints project into Git only when exported. Blocking methods run off GPUI.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::git_transfer::{ExportPlan, ExportPoint, ExportResult, GitRepository, Signature};

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mark {
    pub name: String,
    pub message: String,
    /// Records the caller's verification attestation, not a fabricated test result.
    pub verified: bool,
    pub signature: Signature,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangeNode {
    pub id: String,
    pub thread: String,
    pub parents: Vec<String>,
    pub tree: String,
    pub provenance: EditProvenance,
    pub mark: Option<Mark>,
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
        if graph.state_path().exists() {
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
            graph.save(&GraphState {
                version: 2,
                base_commit,
                base_tree,
                nodes: Vec::new(),
                heads: BTreeMap::new(),
                workspace_heads: BTreeMap::new(),
                checkpoints: BTreeMap::new(),
            })?;
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
        ensure!(!thread.trim().is_empty(), "edit thread must not be empty");
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
        self.save(&state)?;
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
        ensure!(!thread.trim().is_empty(), "thread must not be empty");
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
        self.save(&state)?;
        Ok(node)
    }

    /// Fork an independent native workspace from an immutable live point.
    /// No Git branch/worktree/commit is created; its edits share only the object store.
    pub fn fork(&self, thread: &str, from: &str, destination: &Path) -> Result<GraphState> {
        ensure!(!thread.trim().is_empty(), "thread must not be empty");
        let _lock = self.lock()?;
        let mut state = self.load()?;
        ensure!(!state.heads.contains_key(thread), "thread already exists");
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
        self.repository
            .fork_workspace(&node.tree, destination, thread)?;
        let recorded = (|| -> Result<()> {
            let destination = destination.canonicalize()?;
            state.heads.insert(thread.into(), from.clone());
            state
                .workspace_heads
                .insert(destination.to_string_lossy().into_owned(), from.clone());
            self.save(&state)
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
        self.save(&state)?;
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
        self.repository.retain_snapshot(&node.id, &node.tree)?;
        state.nodes.push(node.clone());
        Ok(node)
    }

    fn workspace_key(&self) -> String {
        self.workspace().to_string_lossy().into_owned()
    }
    fn state_path(&self) -> PathBuf {
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
    fn load(&self) -> Result<GraphState> {
        let state: GraphState = serde_json::from_reader(File::open(self.state_path())?)
            .context("invalid native change graph; refusing to replace it")?;
        ensure!(
            state.version == 2,
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
        Ok(state)
    }
    fn save(&self, state: &GraphState) -> Result<()> {
        let temporary = self
            .directory
            .join(format!(".graph-{}.tmp", Uuid::now_v7()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            serde_json::to_writer(&mut file, state)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::rename(&temporary, self.state_path())?;
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }
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
