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
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &value)?;
    println!();
    Ok(())
}
