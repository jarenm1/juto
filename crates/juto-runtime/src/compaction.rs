//! Incremental, background-driven compaction.
//!
//! Summaries are nodes on the session path (`session::PathView::nodes`). A
//! summary of the oldest raw messages becomes a level-0 node; once more than
//! `MAX_SUMMARY_NODES` nodes are live, adjacent nodes merge into one node of the
//! next level. Background jobs run their model request off the run loop. A
//! finished job reaches the view only at the next run start, so every request in
//! a run sees the view that run began with.

use super::*;
use crate::session::{NewNode, NodePlan, Session};

/// Live summary nodes allowed before adjacent nodes merge.
pub(crate) const MAX_SUMMARY_NODES: usize = 4;

/// Share of the window at which a background summary starts. Below the sync
/// threshold, so a normal run gets its summary before the guard must compact.
const BACKGROUND_FRACTION: f64 = 0.8;

/// Input tokens per summary request, derived from the compactor's window.
fn input_budget_tokens(window: u64) -> u64 {
    (window * 3 / 4).saturating_sub(8192).max(256)
}

/// Splits `text` into pieces of at most `max_bytes`, preferring line breaks and
/// never splitting a character.
fn chunks(text: &str, max_bytes: usize) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while rest.len() > max_bytes {
        let mut cut = max_bytes;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        if let Some(newline) = rest[..cut].rfind('\n').filter(|at| *at > cut / 2) {
            cut = newline + 1;
        }
        let (piece, tail) = rest.split_at(cut);
        pieces.push(piece);
        rest = tail;
    }
    if !rest.is_empty() {
        pieces.push(rest);
    }
    pieces
}

impl Runtime {
    /// Summarizes `input` with the compactor. The transcript is chunked to the
    /// compactor's window; each chunk is folded into the previous chunk's summary.
    pub(crate) async fn summarize(
        &self,
        model: &Model,
        options: &StreamOptions,
        cancel: &CancellationToken,
        input: &[Message],
    ) -> Result<(String, Usage)> {
        let mut options = options.clone();
        options.max_tokens = Some(options.max_tokens.unwrap_or(4096).min(4096));
        let window = model.context_window.unwrap_or(128_000);
        let max_bytes = input_budget_tokens(window) as usize * 3;
        let transcript = serialize_conversation(input);
        let mut summary: Option<String> = None;
        let mut usage = Usage::default();
        for piece in chunks(&transcript, max_bytes) {
            let mut messages = Vec::new();
            if let Some(prior) = &summary {
                messages.push(Message::developer(format!(
                    "[Earlier conversation summary]\n{prior}"
                )));
            }
            messages.push(Message::user(piece.to_owned()));
            let context = Context {
                system_prompt: vec![COMPACTION_PROMPT.into()],
                messages,
                tools: Vec::new(),
            };
            let completed = self
                .complete_once(model, &options, &context, cancel)
                .await
                .context("compaction failed")?;
            usage.add(&completed.usage);
            let text = completed.text();
            if text.trim().is_empty() {
                bail!("compaction returned an empty summary");
            }
            summary = Some(text);
        }
        summary
            .map(|summary| (summary, usage))
            .context("compaction had nothing to summarize")
    }

    /// Summarizes the raw messages past `keep` into one level-0 node, then merges
    /// adjacent nodes until at most `MAX_SUMMARY_NODES` remain. Synchronous: the
    /// view is replaced before this returns.
    pub(crate) async fn compact_nodes(
        &self,
        session: &mut Session,
        model: &Model,
        options: &StreamOptions,
        cancel: &CancellationToken,
        keep: usize,
    ) -> Result<Usage> {
        let mut usage = Usage::default();
        if let Some(plan) = session.raw_summary_plan(keep)? {
            let (summary, spent) = self.summarize(model, options, cancel, &plan.input).await?;
            usage.add(&spent);
            session.append_node(plan.into_node(summary, None))?;
        }
        while session.path_view()?.nodes.len() > MAX_SUMMARY_NODES {
            let Some(plan) = session.merge_plan()? else {
                break;
            };
            let (summary, spent) = self.summarize(model, options, cancel, &plan.input).await?;
            usage.add(&spent);
            session.append_node(plan.into_node(summary, None))?;
        }
        Ok(usage)
    }

    /// Whether a background summary is pending for `session_id`.
    pub(crate) fn has_background_compaction(&self, session_id: &str) -> bool {
        self.inner.compactions.lock().contains_key(session_id)
    }

    /// Drops the pending background summary for `session_id`, if any. Called before
    /// a synchronous compaction replaces the view.
    pub(crate) fn abort_background_compaction(&self, session_id: &str) {
        let job = self.inner.compactions.lock().remove(session_id);
        if let Some(job) = job {
            job.abort();
        }
    }

    /// Starts `plan`'s summary on a spawned task. One job per session; its result
    /// waits in the job table until `apply_background_compaction`.
    fn start_background_compaction(&self, session: &Session, plan: NodePlan) -> Result<()> {
        let id = session.header().id.clone();
        let selector = session
            .current_model()
            .unwrap_or_else(|| self.config().model);
        let (model, options) = self.resolve_model(&selector)?;
        let runtime = self.clone();
        let job: tokio::task::JoinHandle<Result<NewNode>> = tokio::spawn(async move {
            let cancel = CancellationToken::new();
            let (summary, _usage) = runtime
                .summarize(&model, &options, &cancel, &plan.input)
                .await?;
            Ok(plan.into_node(summary, None))
        });
        self.inner.compactions.lock().insert(id, job);
        Ok(())
    }

    /// Schedules a background summary when the view has more nodes than the cap
    /// (a merge) or its estimated size passes the background threshold (a raw
    /// summary). Does nothing while a job is pending.
    pub(crate) fn schedule_background_compaction(
        &self,
        session: &Session,
        window_budget: u64,
        keep: usize,
    ) -> Result<()> {
        let id = session.header().id.clone();
        if self.has_background_compaction(&id) {
            return Ok(());
        }
        let nodes = session.path_view()?.nodes.len();
        let background = (window_budget as f64 * BACKGROUND_FRACTION) as u64;
        let over = estimate_request_tokens(&[], &[], &session.messages()?) > background;
        // A raw summary adds a node, so it is planned only below the cap; at the cap
        // a merge makes room first. Either way the applied view stays within the cap.
        let plan = if nodes > MAX_SUMMARY_NODES || (over && nodes == MAX_SUMMARY_NODES) {
            session.merge_plan()?
        } else if over {
            session.raw_summary_plan(keep)?
        } else {
            None
        };
        if let Some(plan) = plan {
            self.start_background_compaction(session, plan)?;
        }
        Ok(())
    }

    /// Appends a finished background summary to the view. Called at the run start
    /// only, which is the turn boundary. A failed job is reported and dropped; the
    /// next boundary plans again from the unchanged view.
    pub(crate) async fn apply_background_compaction(&self, session: &mut Session) {
        let id = session.header().id.clone();
        let job = {
            let mut jobs = self.inner.compactions.lock();
            let finished = jobs.get(&id).is_some_and(|job| job.is_finished());
            if finished { jobs.remove(&id) } else { None }
        };
        let Some(job) = job else {
            return;
        };
        match job.await {
            Ok(Ok(node)) => {
                if let Err(error) = session.append_node(node) {
                    eprintln!("juto: discarded background compaction for {id}: {error:#}");
                }
            }
            Ok(Err(error)) => {
                eprintln!("juto: background compaction failed for {id}: {error:#}");
            }
            Err(join) => {
                eprintln!("juto: background compaction task failed for {id}: {join}");
            }
        }
    }
}
