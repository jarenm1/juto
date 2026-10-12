use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use clap::Subcommand;
use juto_runtime::change_graph::{ChangeGraph, EditProvenance, JoinOutcome, ReconcileOutcome};

#[derive(Subcommand)]
pub enum ChangesCommand {
    /// Initialize Juto's native live-change DAG from Git HEAD (or an immutable base commit).
    Init {
        #[arg(long)]
        base: Option<String>,
    },
    /// Print live points, independent thread heads, and named checkpoints as JSON.
    Log,
    /// Capture shell/manual edits as a native point, not a Git commit.
    Capture {
        /// Native thread; defaults to the fork's assigned thread, or 'main'.
        #[arg(long)]
        thread: Option<String>,
        #[arg(long, default_value = "manual")]
        session: String,
        /// Explicit parents for a manually integrated join. Does not merge file contents.
        #[arg(long)]
        parent: Vec<String>,
    },
    /// Fork a live point into a new independent native workspace/thread.
    Fork {
        #[arg(long)]
        thread: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        directory: PathBuf,
        /// Materialize only these repository-relative subtrees (repeatable).
        /// Capture still records a complete tree.
        #[arg(long)]
        sparse: Vec<String>,
    },
    /// Name and freeze this exact native point as a future Git commit.
    Mark {
        node: String,
        #[arg(long)]
        name: String,
        /// Git commit message; defaults to the checkpoint name.
        #[arg(short, long)]
        message: Option<String>,
        /// Attest that this point was verified; does not run or invent verification.
        #[arg(long)]
        verified: bool,
    },
    /// Turn only named ancestors into Git commits on a feature branch.
    Export {
        /// Checkpoint name or marked native point ID.
        checkpoint: String,
        #[arg(long)]
        branch: String,
    },
    /// Push an exported Git feature branch without forcing remote history.
    Push {
        #[arg(long)]
        branch: String,
        #[arg(long, default_value = "origin")]
        remote: String,
    },
    /// Fetch an exported Git feature branch into a local branch.
    FetchBranch {
        #[arg(long)]
        branch: String,
        #[arg(long, default_value = "origin")]
        remote: String,
    },
    /// Publish the whole live-change DAG to a remote as `refs/juto/*`.
    PushGraph {
        #[arg(long, default_value = "origin")]
        remote: String,
    },
    /// Fetch `refs/juto/*` from a remote and union it into the local DAG.
    Fetch {
        #[arg(long, default_value = "origin")]
        remote: String,
    },
    /// Join two threads' trees into one node with two parents.
    Join {
        #[arg(long)]
        thread: String,
        #[arg(long)]
        left: String,
        #[arg(long)]
        right: String,
    },
    /// Record a resolved merge: capture the workspace as a two-parent node.
    ///
    /// Edit the conflicted files first; this records the result.
    Resolve {
        #[arg(long)]
        thread: String,
        #[arg(long)]
        left: String,
        #[arg(long)]
        right: String,
    },
    /// Revert a thread to an earlier point by materializing its tree delta.
    Rewind {
        #[arg(long)]
        thread: String,
        /// Checkpoint name or native point ID.
        #[arg(long)]
        to: String,
    },
    /// Join every imported remote head back into a thread after a fetch.
    Reconcile {
        #[arg(long)]
        thread: String,
    },
    /// Reclaim unreachable nodes, release their refs, and pack loose refs.
    Gc,
    /// Print a compact summary: counts, heads, and what needs reconciling.
    ///
    /// Unlike `log`, this stays small for a long session.
    Status,
    /// Check that every node's snapshot ref exists and resolves to its tree.
    ///
    /// Exits non-zero when a ref the graph depends on is missing or wrong.
    Verify,
}

pub fn run(cwd: &Path, command: &ChangesCommand) -> Result<()> {
    let graph = match command {
        ChangesCommand::Init { base } => ChangeGraph::initialize(cwd, base.as_deref())?,
        _ => ChangeGraph::open(cwd)?,
    };
    let value = match command {
        ChangesCommand::Init { .. } | ChangesCommand::Log => serde_json::to_value(graph.state()?)?,
        ChangesCommand::Capture {
            thread,
            session,
            parent,
        } => {
            ensure!(
                !session.trim().is_empty(),
                "session provenance must not be empty"
            );
            let thread = thread.as_deref().or(graph.thread()).unwrap_or("main");
            serde_json::to_value(graph.capture(
                thread,
                EditProvenance {
                    session_id: session.clone(),
                    tool: "capture".into(),
                    selection_mode: None,
                    path: None,
                },
                if parent.is_empty() {
                    None
                } else {
                    Some(parent.clone())
                },
            )?)?
        }
        ChangesCommand::Fork {
            thread,
            from,
            directory,
            sparse,
        } => serde_json::to_value(if sparse.is_empty() {
            graph.fork(thread, from, directory)?
        } else {
            graph.fork_sparse(thread, from, directory, sparse)?
        })?,
        ChangesCommand::Mark {
            node,
            name,
            message,
            verified,
        } => serde_json::to_value(graph.mark(
            node,
            name,
            message.as_deref().unwrap_or(name),
            *verified,
        )?)?,
        ChangesCommand::Export { checkpoint, branch } => {
            serde_json::to_value(graph.export(checkpoint, branch)?)?
        }
        ChangesCommand::Push { branch, remote } => {
            graph.push(branch, remote)?;
            serde_json::json!({"branch":branch, "remote":remote, "pushed":true})
        }
        ChangesCommand::FetchBranch { branch, remote } => {
            graph.fetch_branch(branch, remote)?;
            serde_json::json!({"branch":branch, "remote":remote, "fetched":true})
        }
        ChangesCommand::PushGraph { remote } => serde_json::to_value(graph.push_graph(remote)?)?,
        ChangesCommand::Fetch { remote } => serde_json::to_value(graph.fetch_graph(remote)?)?,
        ChangesCommand::Join {
            thread,
            left,
            right,
        } => match graph.join(thread, left, right)? {
            JoinOutcome::Joined { node, base, tree } => serde_json::json!({
                "status": "joined",
                "node": node,
                "base": base,
                "tree": tree,
            }),
            JoinOutcome::Conflicted { base, conflicts } => serde_json::json!({
                "status": "conflicted",
                "base": base,
                "conflicts": conflicts,
            }),
        },
        ChangesCommand::Resolve {
            thread,
            left,
            right,
        } => serde_json::to_value(graph.resolve(
            thread,
            left,
            right,
            EditProvenance {
                session_id: String::new(),
                tool: "resolve".into(),
                selection_mode: None,
                path: None,
            },
        )?)?,
        ChangesCommand::Rewind { thread, to } => serde_json::to_value(graph.rewind(thread, to)?)?,
        ChangesCommand::Reconcile { thread } => serde_json::Value::Array(
            graph
                .reconcile(thread)?
                .into_iter()
                .map(|outcome| match outcome {
                    ReconcileOutcome::Joined {
                        remote,
                        node,
                        base,
                        tree,
                    } => serde_json::json!({
                        "status": "joined",
                        "remote": remote,
                        "node": node,
                        "base": base,
                        "tree": tree,
                    }),
                    ReconcileOutcome::Adopted { remote, node } => serde_json::json!({
                        "status": "adopted",
                        "remote": remote,
                        "node": node,
                    }),
                    ReconcileOutcome::Conflicted {
                        remote,
                        base,
                        conflicts,
                    } => serde_json::json!({
                        "status": "conflicted",
                        "remote": remote,
                        "base": base,
                        "conflicts": conflicts,
                    }),
                })
                .collect(),
        ),
        ChangesCommand::Gc => serde_json::to_value(graph.gc()?)?,
        ChangesCommand::Status => {
            let state = graph.state()?;
            // A head key containing '/' is an imported remote head: the local
            // thread name never contains one.
            let remote_heads: Vec<&String> =
                state.heads.keys().filter(|key| key.contains('/')).collect();
            serde_json::json!({
                "thread": graph.thread(),
                "workspace": graph.workspace(),
                "sparse": graph.sparse(),
                "nodes": state.nodes.len(),
                "heads": state.heads,
                "remote_heads": remote_heads,
                "needs_reconcile": !remote_heads.is_empty(),
                "pending_threads": graph.pending_reconcile_threads()?,
                "checkpoints": state.checkpoints,
                "workspace_heads": state.workspace_heads.len(),
            })
        }
        ChangesCommand::Verify => {
            let report = graph.verify()?;
            let ok = report.is_ok();
            let mut value = serde_json::to_value(&report)?;
            if let Some(object) = value.as_object_mut() {
                object.insert("ok".to_string(), serde_json::Value::Bool(ok));
            }
            value
        }
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
    println!();
    // A verification command that always exits zero is not useful in a script.
    if matches!(command, ChangesCommand::Verify)
        && value.get("ok") == Some(&serde_json::Value::Bool(false))
    {
        anyhow::bail!("verification failed: a snapshot ref is missing or wrong");
    }
    Ok(())
}
