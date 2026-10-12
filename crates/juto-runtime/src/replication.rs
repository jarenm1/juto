//! Cross-machine replication of Juto's live-change DAG over plain Git transport.
//!
//! The native graph (`change_graph`) is a local JSON journal. This module makes
//! it replicable without inventing a second transport: every piece of the DAG
//! is published as an ordinary Git ref, so a plain `git fetch`/`git push`
//! against any remote converges two machines.
//!
//! # Wire format
//!
//! ```text
//! refs/juto/nodes/<node-id>          -> blob(ChangeNode JSON)          create-only
//! refs/juto/snapshots/<id>           -> tree | commit  (EXISTS ALREADY)  create-only
//! refs/juto/heads/<owner>/<thread>   -> blob(head node id)              single writer per owner
//! refs/juto/checkpoints/<name>       -> blob(node id)                   single writer per name
//! ```
//!
//! The format is conflict-free by construction: node ids are UUIDv7 and node
//! content is never rewritten, so `refs/juto/nodes/*` and
//! `refs/juto/snapshots/*` are only ever created or re-written with an
//! identical value. Checkpoint refs are single-writer per name. Head refs are
//! the one mutable namespace — an owner advances its own head — and are pushed
//! with a force refspec because Git refuses to move a ref that points at a
//! non-commit object otherwise; the owner is the single writer, so that force
//! can only ever move the owner's own head. A rejected push therefore means a
//! real conflict, never a lost update.
//!
//! # Import
//!
//! [`fetch_graph`] mirrors the remote into a staging namespace
//! (`refs/juto/incoming/<slug>/*`) and then [`import_refs`] unions it into the
//! local journal under the graph lock. The union is by node id, so importing
//! twice is a no-op. A node id that already exists locally with different
//! content is corruption and aborts the whole import without changing anything.
//!
//! This module depends on `change_graph` and `git_transfer`; `git_transfer`
//! never depends on `change_graph`.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::change_graph::{ChangeGraph, ChangeNode, validate_state};
use crate::git_transfer::GitRepository;

/// `refs/juto/nodes/<node-id>` -> blob(ChangeNode JSON). Create-only.
const NODES_PREFIX: &str = "refs/juto/nodes";
/// `refs/juto/snapshots/<id>` -> tree | commit. Create-only; already used locally.
const SNAPSHOTS_PREFIX: &str = "refs/juto/snapshots";
/// `refs/juto/heads/<owner>/<thread>` -> blob(head node id). Single writer per owner.
const HEADS_PREFIX: &str = "refs/juto/heads";
/// `refs/juto/checkpoints/<name>` -> blob(node id). Single writer per name.
const CHECKPOINTS_PREFIX: &str = "refs/juto/checkpoints";
/// Fetched-but-not-yet-imported refs live under this prefix.
const INCOMING_PREFIX: &str = "refs/juto/incoming";

/// What an import changed, and every conflict it refused to resolve silently.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportReport {
    /// Nodes newly added to the local journal.
    pub nodes_imported: usize,
    /// Incoming nodes already present locally with identical content.
    pub nodes_unchanged: usize,
    /// Head refs adopted into the local graph (including conflict-preserved ones).
    pub heads_imported: usize,
    /// Heads that disagreed with an existing local head for the same thread.
    pub head_conflicts: Vec<HeadConflict>,
    /// Checkpoint names newly mapped locally.
    pub checkpoints_imported: usize,
    /// Checkpoint names that already mapped to a different node locally.
    pub checkpoint_conflicts: Vec<CheckpointConflict>,
}

/// A remote head that disagreed with the local head for the same thread.
///
/// The local head is kept; the remote head is preserved under
/// `<owner>/<thread>` so no history is lost.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadConflict {
    pub owner: String,
    pub thread: String,
    pub local: String,
    pub remote: String,
}

/// A remote checkpoint name that already maps to a different local node.
///
/// The local mapping is kept; the node itself is still imported.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointConflict {
    pub name: String,
    pub local: String,
    pub remote: String,
}

/// What a publish wrote and pushed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishReport {
    /// Node refs ensured.
    pub nodes: usize,
    /// Head refs ensured.
    pub heads: usize,
    /// Checkpoint refs ensured.
    pub checkpoints: usize,
    /// Snapshot refs present under `refs/juto/snapshots/*`.
    pub snapshots: usize,
}

/// Publish the whole graph to `remote` as `refs/juto/*` refs.
///
/// Every node, head, and checkpoint is written to a local ref first, then the
/// populated namespaces are pushed. A namespace with no refs is skipped: a
/// wildcard refspec that matches nothing is a hard error for `git push`. Only
/// the head namespace is pushed with a force refspec (see the module docs).
pub(crate) fn publish(graph: &ChangeGraph, remote: &str) -> Result<PublishReport> {
    ensure!(!remote.is_empty(), "remote is empty");
    let repository = graph.repository();
    let mut report = PublishReport::default();
    let refspecs = {
        let _lock = graph.graph_lock()?;
        let state = graph.load_locked()?;
        let owner = graph.machine_id_locked()?;
        // One listing per namespace decides create vs. compare-and-swap.
        let mut existing: BTreeMap<String, String> = BTreeMap::new();
        for prefix in [
            NODES_PREFIX,
            SNAPSHOTS_PREFIX,
            HEADS_PREFIX,
            CHECKPOINTS_PREFIX,
        ] {
            existing.extend(repository.list_refs(prefix)?);
        }
        for node in &state.nodes {
            let blob = repository.write_blob(&encode_node(node)?)?;
            let refname = node_ref(&node.id);
            ensure_ref(
                repository,
                &refname,
                &blob,
                existing.get(&refname).map(String::as_str),
            )?;
            repository.retain_snapshot(&node.id, &node.tree)?;
            report.nodes += 1;
        }
        for (thread, head) in &state.heads {
            let blob = repository.write_blob(head.as_bytes())?;
            let refname = head_ref(&owner, thread);
            ensure_ref(
                repository,
                &refname,
                &blob,
                existing.get(&refname).map(String::as_str),
            )?;
            report.heads += 1;
        }
        for (name, id) in &state.checkpoints {
            let blob = repository.write_blob(id.as_bytes())?;
            let refname = checkpoint_ref(name);
            ensure_ref(
                repository,
                &refname,
                &blob,
                existing.get(&refname).map(String::as_str),
            )?;
            report.checkpoints += 1;
        }
        // A wildcard refspec that matches nothing is a hard error for
        // `git push`, so only populated namespaces are pushed.
        let mut refspecs = Vec::new();
        for prefix in [
            NODES_PREFIX,
            SNAPSHOTS_PREFIX,
            HEADS_PREFIX,
            CHECKPOINTS_PREFIX,
        ] {
            let refs = repository.list_refs(prefix)?;
            if prefix == SNAPSHOTS_PREFIX {
                report.snapshots = refs.len();
            }
            if !refs.is_empty() {
                // Head refs are the one mutable namespace: an owner advances
                // its own head, and Git refuses to move a ref that points at a
                // non-commit object without a force. The owner is the single
                // writer, so forcing its own head is safe. Node, snapshot, and
                // checkpoint refs stay unforced, so a genuine collision is
                // rejected instead of silently overwritten.
                let force = if prefix == HEADS_PREFIX { "+" } else { "" };
                refspecs.push(format!("{force}{prefix}/*:{prefix}/*"));
            }
        }
        refspecs
    };
    if !refspecs.is_empty() {
        repository.push_refs(remote, &refspecs)?;
    }
    Ok(report)
}

/// Fetch `refs/juto/*` from `remote` into staging, then import it.
pub(crate) fn fetch_graph(graph: &ChangeGraph, remote: &str) -> Result<ImportReport> {
    ensure!(!remote.is_empty(), "remote is empty");
    let staging = staging_prefix(&staging_slug(remote));
    let refspecs = vec![
        format!("+{NODES_PREFIX}/*:{staging}/nodes/*"),
        format!("+{SNAPSHOTS_PREFIX}/*:{staging}/snapshots/*"),
        format!("+{HEADS_PREFIX}/*:{staging}/heads/*"),
        format!("+{CHECKPOINTS_PREFIX}/*:{staging}/checkpoints/*"),
    ];
    graph.repository().fetch(remote, &refspecs)?;
    import_refs(graph, &staging)
}

/// Union already-fetched staging refs into the local journal. No network access.
///
/// The whole import runs under the graph lock and commits the journal exactly
/// once, so any earlier failure leaves the local graph untouched. Staging refs
/// are deleted at the end; the objects stay pinned by the canonical refs.
pub(crate) fn import_refs(graph: &ChangeGraph, staging: &str) -> Result<ImportReport> {
    ensure!(!staging.is_empty(), "staging prefix is empty");
    let repository = graph.repository();
    let _lock = graph.graph_lock()?;
    let mut state = graph.load_locked()?;
    let owner = graph.machine_id_locked()?;

    let refs = repository.list_refs(staging)?;
    let mut node_refs: BTreeMap<String, String> = BTreeMap::new();
    let mut snapshot_refs: BTreeMap<String, String> = BTreeMap::new();
    let mut head_refs: Vec<(String, String, String)> = Vec::new();
    let mut checkpoint_refs: Vec<(String, String)> = Vec::new();
    for (refname, oid) in &refs {
        let Some(rest) = strip_staging(refname, staging) else {
            continue;
        };
        if let Some(id) = rest.strip_prefix("nodes/") {
            node_refs.insert(id.to_string(), oid.clone());
        } else if let Some(id) = rest.strip_prefix("snapshots/") {
            snapshot_refs.insert(id.to_string(), oid.clone());
        } else if let Some(rest) = rest.strip_prefix("heads/") {
            let (head_owner, thread) = rest
                .split_once('/')
                .with_context(|| format!("malformed replicated head ref {refname:?}"))?;
            head_refs.push((head_owner.to_string(), thread.to_string(), oid.clone()));
        } else if let Some(name) = rest.strip_prefix("checkpoints/") {
            checkpoint_refs.push((name.to_string(), oid.clone()));
        }
    }

    // Decode every incoming node; the ref name is authoritative for the id.
    let mut incoming: BTreeMap<String, ChangeNode> = BTreeMap::new();
    for (id, oid) in &node_refs {
        let bytes = repository.read_blob(oid)?;
        let node = decode_node(&bytes)?;
        ensure!(
            node.id == *id,
            "replicated node {id} carries a different id {}",
            node.id
        );
        incoming.insert(id.clone(), node);
    }

    // Union by id. Identical content is idempotent; different content is
    // corruption and aborts before anything is written.
    let mut report = ImportReport::default();
    let local: BTreeMap<&str, &ChangeNode> =
        state.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut new_nodes: Vec<ChangeNode> = Vec::new();
    for (id, node) in &incoming {
        match local.get(id.as_str()) {
            Some(existing) => {
                ensure!(*existing == node, "node {id} exists with different content");
                report.nodes_unchanged += 1;
            }
            None => new_nodes.push(node.clone()),
        }
    }

    // Partial-fetch guard: every incoming node's tree must be pinned in staging.
    for (id, node) in &incoming {
        let oid = snapshot_refs
            .get(id)
            .with_context(|| format!("replicated node {id} is missing its snapshot ref"))?;
        let tree = repository.snapshot_tree(oid)?;
        ensure!(
            tree == node.tree,
            "replicated node {id} snapshot does not match its tree"
        );
    }

    // Topological order: the local prefix is untouched, new nodes follow it.
    let order = order_new_nodes(&state.nodes, &new_nodes)?;
    let new_by_id: BTreeMap<String, ChangeNode> = new_nodes
        .into_iter()
        .map(|node| (node.id.clone(), node))
        .collect();
    let mut union = state.nodes.clone();
    for id in &order {
        union.push(new_by_id[id].clone());
    }
    report.nodes_imported = order.len();

    let mut heads_changed = false;
    {
        let union_ids: BTreeSet<&str> = union.iter().map(|n| n.id.as_str()).collect();
        for (head_owner, thread, oid) in &head_refs {
            let bytes = repository.read_blob(oid)?;
            let head_id = String::from_utf8(bytes)
                .context("replicated head is not UTF-8")?
                .trim()
                .to_string();
            ensure!(
                union_ids.contains(head_id.as_str()),
                "replicated head {head_owner}/{thread} names an unknown node {head_id}"
            );
            let local_key = if head_owner == &owner {
                thread.clone()
            } else {
                format!("{head_owner}/{thread}")
            };
            match state.heads.get(&local_key).cloned() {
                Some(existing) if existing == head_id => {}
                None => {
                    state.heads.insert(local_key, head_id);
                    report.heads_imported += 1;
                    heads_changed = true;
                }
                Some(existing) => {
                    // Never overwrite a local head: keep it and preserve the
                    // remote head under the owner-qualified key instead.
                    let mut key = format!("{head_owner}/{thread}");
                    let mut suffix = 1;
                    while state.heads.get(&key).is_some_and(|value| value != &head_id) {
                        key = format!("{head_owner}/{thread}@{suffix}");
                        suffix += 1;
                    }
                    if state.heads.get(&key) != Some(&head_id) {
                        state.heads.insert(key, head_id.clone());
                        heads_changed = true;
                    }
                    report.heads_imported += 1;
                    report.head_conflicts.push(HeadConflict {
                        owner: head_owner.clone(),
                        thread: thread.clone(),
                        local: existing,
                        remote: head_id,
                    });
                }
            }
        }
    }

    let mut checkpoints_changed = false;
    for (name, oid) in &checkpoint_refs {
        let bytes = repository.read_blob(oid)?;
        let id = String::from_utf8(bytes)
            .context("replicated checkpoint is not UTF-8")?
            .trim()
            .to_string();
        let node = union
            .iter()
            .find(|n| n.id == id)
            .with_context(|| format!("replicated checkpoint {name} names an unknown node {id}"))?;
        ensure!(
            node.mark.as_ref().is_some_and(|mark| mark.name == *name),
            "replicated checkpoint {name} does not match its node mark"
        );
        match state.checkpoints.get(name).cloned() {
            Some(existing) if existing == id => {}
            None => {
                state.checkpoints.insert(name.clone(), id);
                report.checkpoints_imported += 1;
                checkpoints_changed = true;
            }
            Some(existing) => report.checkpoint_conflicts.push(CheckpointConflict {
                name: name.clone(),
                local: existing,
                remote: id,
            }),
        }
    }

    // The base anchor is only adopted by an empty, unanchored graph.
    let mut base_adopted = false;
    if let Some(remote_base) = snapshot_refs.get("base") {
        if state.base_commit == *remote_base {
            // Already anchored to the same commit.
        } else if state.nodes.is_empty() && state.base_commit.bytes().all(|byte| byte == b'0') {
            let base_tree = match snapshot_refs.get("base-tree") {
                Some(tree_ref) => repository.snapshot_tree(tree_ref)?,
                None => repository.snapshot_tree(remote_base)?,
            };
            state.base_commit = remote_base.clone();
            state.base_tree = base_tree.clone();
            repository.retain_snapshot("base", remote_base)?;
            repository.retain_snapshot("base-tree", &base_tree)?;
            base_adopted = true;
        } else {
            bail!(
                "remote base {remote_base} does not match the local base {}",
                state.base_commit
            );
        }
    }

    // Pin every newly imported node so the import is durable and re-publishable.
    for id in &order {
        let node = &new_by_id[id];
        repository.retain_snapshot(&node.id, &node.tree)?;
        let blob = repository.write_blob(&encode_node(node)?)?;
        let refname = node_ref(&node.id);
        let current = current_ref(repository, &refname)?;
        ensure_ref(repository, &refname, &blob, current.as_deref())?;
    }

    // Single atomic commit point; skipped entirely when nothing changed.
    if !order.is_empty() || heads_changed || checkpoints_changed || base_adopted {
        state.nodes = union;
        validate_state(&state)?;
        graph.write_nodes_locked(&state.nodes)?;
        graph.save_locked(&state)?;
    }

    for (refname, _) in &refs {
        repository.delete_ref(refname)?;
    }
    Ok(report)
}

/// Order the new nodes topologically, after the untouched local prefix.
///
/// Local nodes are already topologically ordered and can only be parents of new
/// nodes (never the reverse), so emitting them first preserves the local prefix
/// exactly. New nodes are then ordered by Kahn's algorithm with the node id as
/// a deterministic tie-break.
fn order_new_nodes(local: &[ChangeNode], new: &[ChangeNode]) -> Result<Vec<String>> {
    let local_ids: BTreeSet<&str> = local.iter().map(|n| n.id.as_str()).collect();
    let new_ids: BTreeSet<&str> = new.iter().map(|n| n.id.as_str()).collect();
    let mut pending: BTreeMap<&str, usize> = BTreeMap::new();
    let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in new {
        let mut count = 0;
        for parent in &node.parents {
            if local_ids.contains(parent.as_str()) {
                continue;
            }
            ensure!(
                new_ids.contains(parent.as_str()),
                "replicated node {} references unknown parent {}",
                node.id,
                parent
            );
            count += 1;
            children
                .entry(parent.as_str())
                .or_default()
                .push(node.id.as_str());
        }
        pending.insert(node.id.as_str(), count);
    }
    let mut ready: BTreeSet<&str> = pending
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(id, _)| *id)
        .collect();
    let mut order = Vec::with_capacity(new.len());
    while let Some(id) = ready.iter().next().copied() {
        ready.remove(id);
        order.push(id.to_string());
        if let Some(kids) = children.get(id) {
            for kid in kids {
                let count = pending.get_mut(*kid).expect("known child");
                *count -= 1;
                if *count == 0 {
                    ready.insert(kid);
                }
            }
        }
    }
    ensure!(
        order.len() == new.len(),
        "replicated graph contains a cycle"
    );
    Ok(order)
}

/// Strip the staging prefix and its separating slash from a ref name.
fn strip_staging<'a>(refname: &'a str, staging: &str) -> Option<&'a str> {
    refname.strip_prefix(staging)?.strip_prefix('/')
}

/// Ensure `refname` holds `object`, tolerating an existing identical value.
///
/// `current` is the value observed for `refname` (`None` when absent). The
/// update is a compare-and-swap against it, so a concurrent writer is detected
/// instead of silently overwritten, and re-writing an identical value is a
/// no-op rather than an error.
fn ensure_ref(
    repository: &GitRepository,
    refname: &str,
    object: &str,
    current: Option<&str>,
) -> Result<()> {
    match current {
        Some(existing) if existing == object => Ok(()),
        Some(existing) => {
            repository.update_ref(refname, object, Some(existing))?;
            Ok(())
        }
        None => {
            repository.update_ref(refname, object, None)?;
            Ok(())
        }
    }
}

/// The current value of `refname`, or `None` when it does not exist.
fn current_ref(repository: &GitRepository, refname: &str) -> Result<Option<String>> {
    Ok(repository
        .list_refs(refname)?
        .into_iter()
        .find(|(name, _)| name == refname)
        .map(|(_, oid)| oid))
}

fn node_ref(id: &str) -> String {
    format!("{NODES_PREFIX}/{id}")
}

fn head_ref(owner: &str, thread: &str) -> String {
    format!("{HEADS_PREFIX}/{owner}/{thread}")
}

fn checkpoint_ref(name: &str) -> String {
    format!("{CHECKPOINTS_PREFIX}/{name}")
}

fn staging_prefix(slug: &str) -> String {
    format!("{INCOMING_PREFIX}/{slug}")
}

/// A ref-safe slug for a remote string: FNV-1a 64 rendered as 16 hex digits.
///
/// The slug only has to be stable and collision-resistant enough to keep two
/// different remotes' staging refs apart; it is never a security boundary.
fn staging_slug(remote: &str) -> String {
    format!("{:016x}", fnv1a64(remote.as_bytes()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn encode_node(node: &ChangeNode) -> Result<Vec<u8>> {
    serde_json::to_vec(node).context("failed to encode a change node")
}

fn decode_node(bytes: &[u8]) -> Result<ChangeNode> {
    serde_json::from_slice(bytes).context("invalid replicated change node")
}
