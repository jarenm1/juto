//! Run-scoped control channel: cancellation, pause/resume, and message queues.

use std::sync::Arc;

use juto_ai::Message;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

/// Shared state behind [`RunControl`]. Steering injects at the next loop
/// boundary (before a model call or after a tool batch settles); follow-ups
/// are only delivered when the model produced no tool calls and the run would
/// otherwise finish — matching oh-my-pi's `getSteeringMessages` /
/// `getFollowUpMessages` delivery rules.
#[derive(Debug)]
struct Shared {
    cancel: CancellationToken,
    paused: watch::Sender<bool>,
    _paused_rx: watch::Receiver<bool>,
    steering_tx: mpsc::UnboundedSender<Message>,
    follow_up_tx: mpsc::UnboundedSender<Message>,
    /// Taken by `Agent::run`; `None` afterwards, so queued sends from clones
    /// made before the run still land while the receiver is held.
    steering_rx: tokio::sync::Mutex<Option<mpsc::UnboundedReceiver<Message>>>,
    follow_up_rx: tokio::sync::Mutex<Option<mpsc::UnboundedReceiver<Message>>>,
    /// Background jobs started by this run that have not yet delivered their result.
    jobs: watch::Sender<usize>,
    /// Cancels this run's background jobs; fired when the run ends.
    jobs_cancel: CancellationToken,
}

/// External control handle for one `Agent::run`. Clone and hand clones to UI
/// actions; the agent drains the queues at safe boundaries only.
#[derive(Debug, Clone)]
pub struct RunControl {
    shared: Arc<Shared>,
}

impl Default for RunControl {
    fn default() -> Self {
        Self::new()
    }
}

impl RunControl {
    pub fn new() -> Self {
        let (paused, paused_rx) = watch::channel(false);
        let (steering_tx, steering_rx) = mpsc::unbounded_channel();
        let (follow_up_tx, follow_up_rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        Self {
            shared: Arc::new(Shared {
                jobs_cancel: cancel.child_token(),
                cancel,
                paused,
                _paused_rx: paused_rx,
                steering_tx,
                follow_up_tx,
                steering_rx: tokio::sync::Mutex::new(Some(steering_rx)),
                follow_up_rx: tokio::sync::Mutex::new(Some(follow_up_rx)),
                jobs: watch::channel(0).0,
            }),
        }
    }

    /// Cancellation token observed by the provider stream, tools, retry
    /// sleeps, and pause waits.
    pub fn token(&self) -> CancellationToken {
        self.shared.cancel.clone()
    }

    /// Abort the run. In-flight work unwinds; every committed tool call still
    /// receives a paired tool result before the agent ends.
    pub fn cancel(&self) {
        self.shared.cancel.cancel();
    }

    /// Queue a steering message for injection at the next boundary. Returns
    /// `false` when the run has already ended and cannot receive it.
    pub fn steer(&self, message: Message) -> bool {
        self.shared.steering_tx.send(message).is_ok()
    }

    /// Queue a message delivered only when the agent would otherwise stop.
    pub fn follow_up(&self, message: Message) {
        let _ = self.shared.follow_up_tx.send(message);
    }

    /// Open a background job whose result returns through [`Self::steer`]. The
    /// run stays open at its stop boundary until the guard is dropped, so the
    /// result reaches the model instead of arriving after the run has ended.
    pub fn begin_background_job(&self) -> BackgroundJob {
        self.shared.jobs.send_modify(|open| *open += 1);
        BackgroundJob {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Background jobs whose results have not been delivered yet.
    pub fn background_jobs(&self) -> usize {
        *self.shared.jobs.borrow()
    }

    /// Cancellation for this run's background jobs. It fires when the run is
    /// cancelled, and when the run ends with jobs still open.
    pub fn background_token(&self) -> CancellationToken {
        self.shared.jobs_cancel.clone()
    }

    /// Cancel every background job still open. Called when the run ends, so a
    /// queued or running job cannot deliver into a run that has finished.
    pub(crate) fn abandon_background_jobs(&self) {
        self.shared.jobs_cancel.cancel();
    }

    /// Wait until a background job settles or the run is cancelled; returns at
    /// once when none is open. The subscription is taken before the check, so
    /// a job settling in between is never missed.
    pub(crate) async fn wait_background_job(&self, cancel: &CancellationToken) {
        let mut rx = self.shared.jobs.subscribe();
        if *rx.borrow_and_update() == 0 {
            return;
        }
        tokio::select! {
            biased;
            () = cancel.cancelled() => {}
            _ = rx.changed() => {}
        }
    }

    /// Park the run at the next safe boundary (before a model call or a tool
    /// execution). Queued messages stay queued through the pause.
    pub fn pause(&self) {
        let _ = self.shared.paused.send_replace(true);
    }

    /// Release a parked or parking run.
    pub fn resume(&self) {
        let _ = self.shared.paused.send_replace(false);
    }

    /// Whether the gate is currently engaged.
    pub fn is_paused(&self) -> bool {
        *self.shared.paused.borrow()
    }

    /// Wait while the pause gate is engaged; returns early on cancellation so
    /// a cancelled run never stays parked.
    pub(crate) async fn wait_if_paused(&self) {
        if !self.is_paused() {
            return;
        }
        let mut rx = self.shared.paused.subscribe();
        while *rx.borrow_and_update() {
            tokio::select! {
                biased;
                () = self.shared.cancel.cancelled() => return,
                changed = rx.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            }
        }
    }

    /// Claim the run's queue receivers. Only the first call (the agent loop)
    /// gets them; later calls see empty receivers.
    pub(crate) async fn take_queues(
        &self,
    ) -> (
        mpsc::UnboundedReceiver<Message>,
        mpsc::UnboundedReceiver<Message>,
    ) {
        let steering = self.shared.steering_rx.lock().await.take();
        let follow_up = self.shared.follow_up_rx.lock().await.take();
        (
            steering.unwrap_or_else(|| mpsc::unbounded_channel().1),
            follow_up.unwrap_or_else(|| mpsc::unbounded_channel().1),
        )
    }
}

/// One background job of a run, open until dropped. Drop it only after the
/// job's result has been steered into the run, so the run cannot stop first.
#[derive(Debug)]
pub struct BackgroundJob {
    shared: Arc<Shared>,
}

impl Drop for BackgroundJob {
    fn drop(&mut self) {
        self.shared
            .jobs
            .send_modify(|open| *open = open.saturating_sub(1));
    }
}

/// Non-blocking drain of every message queued so far, in send order.
pub(crate) fn drain(rx: &mut mpsc::UnboundedReceiver<Message>) -> Vec<Message> {
    let mut drained = Vec::new();
    while let Ok(message) = rx.try_recv() {
        drained.push(message);
    }
    drained
}
