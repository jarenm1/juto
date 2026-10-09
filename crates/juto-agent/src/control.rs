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
        Self {
            shared: Arc::new(Shared {
                cancel: CancellationToken::new(),
                paused,
                _paused_rx: paused_rx,
                steering_tx,
                follow_up_tx,
                steering_rx: tokio::sync::Mutex::new(Some(steering_rx)),
                follow_up_rx: tokio::sync::Mutex::new(Some(follow_up_rx)),
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

    /// Queue a steering message for injection at the next boundary.
    pub fn steer(&self, message: Message) {
        let _ = self.shared.steering_tx.send(message);
    }

    /// Queue a message delivered only when the agent would otherwise stop.
    pub fn follow_up(&self, message: Message) {
        let _ = self.shared.follow_up_tx.send(message);
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

/// Non-blocking drain of every message queued so far, in send order.
pub(crate) fn drain(rx: &mut mpsc::UnboundedReceiver<Message>) -> Vec<Message> {
    let mut drained = Vec::new();
    while let Ok(message) = rx.try_recv() {
        drained.push(message);
    }
    drained
}
