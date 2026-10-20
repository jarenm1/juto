//! Shared admission for one run tree: three pools, one per scarce resource.
//!
//! - [`ProviderPool`] admits model requests. Every agent waits on the model, so
//!   in-flight requests are the shared bottleneck. The limit starts at the
//!   configured ceiling, halves when the provider throttles (HTTP 429), and grows
//!   back by one per stream the provider accepts. A request holds its lease until
//!   its stream ends or is dropped.
//! - [`Desks`] reserves memory for each agent turn through the resource broker. A
//!   desk is small, so agents share the same budget ledger as commands and are
//!   never capped by a fixed agent count; heavy work queues behind them instead.
//! - `review_wip` bounds focus threads, the only agents that hold a review slot.
//!
//! Ordinary agents therefore run without a count limit. Their model requests and
//! memory are admitted, and nothing else.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use async_trait::async_trait;
use futures::Stream;
use juto_agent::BlockingWait;
use juto_ai::{Context, Provider, ProviderError, ProviderEvent, ProviderStream, StreamOptions};
use juto_capacity::MachineCapacity;
use juto_catalog::Model;
use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::broker::{Broker, BrokerError, Lease as DeskLease, Priority, Request, Submission};
use crate::workspace::{DEFAULT_SLICE, is_systemd_run_available};

/// A desk is `budget / DESKS_PER_BUDGET` bytes. A full budget of desks is
/// `DESKS_PER_BUDGET` agents; past that, new agents queue for memory like any
/// other reservation. This is the only policy number in the desk rule.
pub const DESKS_PER_BUDGET: u64 = 512;

/// Adaptive in-flight limiter for model requests.
pub struct ProviderPool {
    ceiling: usize,
    state: Mutex<PoolState>,
    freed: Notify,
}

struct PoolState {
    limit: usize,
    in_flight: usize,
}

impl ProviderPool {
    pub fn new(ceiling: usize) -> Self {
        let ceiling = ceiling.max(1);
        Self {
            ceiling,
            state: Mutex::new(PoolState {
                limit: ceiling,
                in_flight: 0,
            }),
            freed: Notify::new(),
        }
    }

    /// Wait for a free slot. Cancellation while waiting returns
    /// [`ProviderError::Aborted`] and takes no slot.
    pub async fn acquire(
        self: &Arc<Self>,
        cancel: &CancellationToken,
    ) -> Result<Lease, ProviderError> {
        loop {
            {
                let mut state = self.state.lock();
                if state.in_flight < state.limit {
                    state.in_flight += 1;
                    return Ok(Lease { pool: self.clone() });
                }
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(ProviderError::Aborted),
                _ = self.freed.notified() => {}
            }
        }
    }

    /// Halve the limit after a throttle, never below one.
    fn throttled(&self) {
        let mut state = self.state.lock();
        state.limit = (state.limit / 2).max(1);
    }

    /// Grow the limit by one after an accepted stream, up to the ceiling.
    fn accepted(&self) {
        let mut state = self.state.lock();
        if state.limit < self.ceiling {
            state.limit += 1;
            self.freed.notify_one();
        }
    }

    fn observe(&self, error: &ProviderError) {
        if matches!(error, ProviderError::Http { status: 429, .. }) {
            self.throttled();
        }
    }

    #[cfg(test)]
    fn limit(&self) -> usize {
        self.state.lock().limit
    }

    #[cfg(test)]
    fn in_flight(&self) -> usize {
        self.state.lock().in_flight
    }
}

/// A held in-flight slot, released on drop.
pub struct Lease {
    pool: Arc<ProviderPool>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.state.lock().in_flight -= 1;
        self.pool.freed.notify_one();
    }
}

/// Errors reserving a desk.
#[derive(Debug, Error)]
pub enum DeskError {
    #[error(transparent)]
    Broker(#[from] BrokerError),
    #[error("cannot measure host capacity: {0}")]
    Capacity(String),
    #[error("desk task failed: {0}")]
    Task(String),
    #[error("systemd-run is unavailable: agent desks cannot be metered")]
    SystemdUnavailable,
    #[error("cancelled while waiting for a desk")]
    Cancelled,
}

/// Memory desks for agent turns, admitted through the shared broker ledger.
#[derive(Clone)]
pub enum Desks {
    /// Desks reserved in the broker ledger. `budget: None` measures the host's
    /// agent budget for each seat.
    Metered { broker: Broker, budget: Option<u64> },
    /// No memory accounting, for tests that exercise turn ordering only.
    #[cfg(test)]
    Unmetered,
}

impl Desks {
    pub fn new(broker: Broker) -> Self {
        Self::Metered {
            broker,
            budget: None,
        }
    }

    /// A desk pool with a fixed budget, bypassing host measurement.
    #[cfg(test)]
    pub fn with_budget(broker: Broker, budget: u64) -> Self {
        Self::Metered {
            broker,
            budget: Some(budget),
        }
    }

    /// Reserve a desk for one agent turn; the lease ends with the turn. Waits in
    /// the broker queue when the budget is full. A host without `systemd-run`
    /// fails closed: the broker cannot measure or enforce memory, so the turn is
    /// refused. `Ok(None)` comes only from the test-only unmetered pool.
    pub async fn seat(&self, label: &str) -> Result<Option<DeskLease>, DeskError> {
        let (broker, budget) = match self {
            Self::Metered { broker, budget } => (broker.clone(), *budget),
            #[cfg(test)]
            Self::Unmetered => return Ok(None),
        };
        let budget = match budget {
            Some(budget) => budget,
            None => {
                if !is_systemd_run_available() {
                    return Err(DeskError::SystemdUnavailable);
                }
                tokio::task::spawn_blocking(|| MachineCapacity::measure(DEFAULT_SLICE))
                    .await
                    .map_err(|error| DeskError::Task(error.to_string()))?
                    .map_err(|error| DeskError::Capacity(error.to_string()))?
                    .budget_bytes
            }
        };
        let request = Request {
            bytes: (budget / DESKS_PER_BUDGET).max(1),
            gpu: false,
            // An agent without a desk cannot make progress, so desks are admitted
            // ahead of background commands rather than starved behind them.
            priority: Priority::Attended,
            label: format!("agent desk: {label}"),
        };
        let submission = tokio::task::spawn_blocking(move || broker.submit(request, budget))
            .await
            .map_err(|error| DeskError::Task(error.to_string()))??;
        match submission {
            Submission::Admitted(lease) => Ok(Some(lease)),
            Submission::Queued(ticket) => Ok(Some(ticket.admitted().await?)),
        }
    }
}

tokio::task_local! {
    /// The desk of the turn whose future is being polled, if that turn holds one.
    static TURN_DESK: Arc<TurnDesk>;
}

/// The desk a turn holds for its whole run, kept in a slot so the turn can give
/// it back while it blocks on other agents and take it again afterward.
///
/// A turn that holds a desk while waiting for its children can starve them: once
/// the budget is full of parents, no child can be admitted and nothing finishes.
/// Blocking waits therefore run through [`release_while`].
pub struct TurnDesk {
    desks: Desks,
    label: String,
    lease: Mutex<Option<DeskLease>>,
}

impl TurnDesk {
    pub fn new(desks: Desks, label: String, lease: Option<DeskLease>) -> Arc<Self> {
        Arc::new(Self {
            desks,
            label,
            lease: Mutex::new(lease),
        })
    }

    /// Run `turn` with this desk as the calling turn's desk.
    pub fn hold<F: Future>(self: &Arc<Self>, turn: F) -> impl Future<Output = F::Output> {
        TURN_DESK.scope(self.clone(), turn)
    }
}

/// Run `blocked`, a wait on other agents or on the ledger, with the calling
/// turn's desk given back, then take the desk again with normal admission.
///
/// Outside a turn (the main thread) there is no desk and this only awaits
/// `blocked`. If `cancel` fires while the desk is queued again, the desk is not
/// reacquired and the error is [`DeskError::Cancelled`]; the cancelled turn ends.
pub async fn release_while<T>(
    cancel: &CancellationToken,
    blocked: impl Future<Output = T>,
) -> Result<T, DeskError> {
    let Ok(turn) = TURN_DESK.try_with(Arc::clone) else {
        return Ok(blocked.await);
    };
    drop(turn.lease.lock().take());
    let value = blocked.await;
    let lease = tokio::select! {
        biased;
        _ = cancel.cancelled() => return Err(DeskError::Cancelled),
        lease = turn.desks.seat(&turn.label) => lease?,
    };
    *turn.lease.lock() = lease;
    Ok(value)
}

/// Hook that [`release_while`] installs on every turn's run, so a turn gives its
/// desk back while it waits on background jobs or children.
#[derive(Debug)]
pub struct DeskWaits;

#[async_trait]
impl BlockingWait for DeskWaits {
    async fn release_while<'a>(
        &'a self,
        cancel: &'a CancellationToken,
        blocked: Pin<Box<dyn Future<Output = ()> + Send + 'a>>,
    ) {
        match release_while(cancel, blocked).await {
            Ok(()) | Err(DeskError::Cancelled) => {}
            // The turn could not take its desk back, so it must not keep running
            // without one: end it.
            Err(_) => cancel.cancel(),
        }
    }
}

/// Admission state shared by every runtime in one run tree.
#[derive(Clone)]
pub struct Admission {
    pub provider_pool: Arc<ProviderPool>,
    pub review_wip: Arc<Semaphore>,
    pub desks: Desks,
}

impl Admission {
    pub fn new(max_concurrency: usize, desks: Desks) -> Self {
        Self {
            provider_pool: Arc::new(ProviderPool::new(max_concurrency)),
            review_wip: Arc::new(Semaphore::new(max_concurrency.max(1))),
            desks,
        }
    }
}

#[cfg(test)]
impl Admission {
    /// Admission with a provider pool and review limit of `max_concurrency`,
    /// and no memory accounting.
    pub fn unmetered(max_concurrency: usize) -> Self {
        Self::new(max_concurrency, Desks::Unmetered)
    }
}

/// A [`Provider`] that admits every model request through a shared pool.
pub struct LimitedProvider {
    inner: Arc<dyn Provider>,
    pool: Arc<ProviderPool>,
}

impl LimitedProvider {
    pub fn new(inner: Arc<dyn Provider>, pool: Arc<ProviderPool>) -> Self {
        Self { inner, pool }
    }
}

#[async_trait]
impl Provider for LimitedProvider {
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let lease = self.pool.acquire(&cancel).await?;
        match self.inner.stream(model, context, options, cancel).await {
            Ok(stream) => {
                self.pool.accepted();
                Ok(Box::pin(LeasedStream {
                    stream,
                    lease: Some(lease),
                    pool: self.pool.clone(),
                }))
            }
            Err(error) => {
                self.pool.observe(&error);
                Err(error)
            }
        }
    }
}

/// A provider stream that releases its lease when it ends, fails, or is dropped.
struct LeasedStream {
    stream: ProviderStream,
    lease: Option<Lease>,
    pool: Arc<ProviderPool>,
}

impl Stream for LeasedStream {
    type Item = Result<ProviderEvent, ProviderError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let item = std::task::ready!(this.stream.as_mut().poll_next(cx));
        match &item {
            None => this.lease = None,
            Some(Err(error)) => {
                this.pool.observe(error);
                this.lease = None;
            }
            Some(Ok(_)) => {}
        }
        Poll::Ready(item)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::time::Duration;

    use super::*;

    /// Replies to each `stream` call in order: `Ok` returns a stream that never
    /// ends (so the lease stays held until dropped); `Err` fails the request.
    struct Scripted {
        replies: Mutex<VecDeque<Result<(), ProviderError>>>,
    }

    #[async_trait]
    impl Provider for Scripted {
        async fn stream(
            &self,
            _model: &Model,
            _context: &Context,
            _options: &StreamOptions,
            _cancel: CancellationToken,
        ) -> Result<ProviderStream, ProviderError> {
            let reply = self
                .replies
                .lock()
                .pop_front()
                .expect("scripted reply available");
            match reply {
                Ok(()) => {
                    let stream: ProviderStream = Box::pin(futures::stream::pending::<
                        Result<ProviderEvent, ProviderError>,
                    >());
                    Ok(stream)
                }
                Err(error) => Err(error),
            }
        }
    }

    fn throttle() -> ProviderError {
        ProviderError::Http {
            status: 429,
            message: "slow down".into(),
            retry_after: None,
        }
    }

    fn model() -> Model {
        serde_json::from_value(serde_json::json!({
            "id": "test",
            "name": "test",
            "api": "test",
            "provider": "test",
            "baseUrl": "",
        }))
        .expect("minimal test model")
    }

    fn provider(
        replies: Vec<Result<(), ProviderError>>,
        pool: &Arc<ProviderPool>,
    ) -> LimitedProvider {
        LimitedProvider::new(
            Arc::new(Scripted {
                replies: Mutex::new(replies.into()),
            }),
            pool.clone(),
        )
    }

    async fn open(
        provider: &LimitedProvider,
        cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        provider
            .stream(
                &model(),
                &Context::default(),
                &StreamOptions::default(),
                cancel,
            )
            .await
    }

    #[tokio::test]
    async fn throttling_halves_the_limit_and_acceptance_grows_it_back() {
        let pool = Arc::new(ProviderPool::new(4));
        let provider = provider(vec![Err(throttle()), Err(throttle()), Ok(())], &pool);

        assert!(open(&provider, CancellationToken::new()).await.is_err());
        assert_eq!(pool.limit(), 2);
        assert!(open(&provider, CancellationToken::new()).await.is_err());
        assert_eq!(pool.limit(), 1);
        let stream = open(&provider, CancellationToken::new()).await.unwrap();
        assert_eq!(pool.limit(), 2);
        assert_eq!(pool.in_flight(), 1);
        drop(stream);
        assert_eq!(pool.in_flight(), 0);
    }

    #[tokio::test]
    async fn a_request_waits_for_the_held_stream_to_end() {
        let pool = Arc::new(ProviderPool::new(1));
        let provider = provider(vec![Ok(()), Ok(())], &pool);
        let held = open(&provider, CancellationToken::new()).await.unwrap();

        let waiting = tokio::time::timeout(
            Duration::from_millis(50),
            open(&provider, CancellationToken::new()),
        )
        .await;
        assert!(waiting.is_err(), "second request must wait for the lease");

        drop(held);
        let _admitted = open(&provider, CancellationToken::new()).await.unwrap();
    }

    #[tokio::test]
    async fn cancelling_a_wait_for_a_lease_takes_no_slot() {
        let pool = Arc::new(ProviderPool::new(1));
        let provider = provider(vec![Ok(())], &pool);
        let _held = open(&provider, CancellationToken::new()).await.unwrap();

        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = open(&provider, cancel).await.err().expect("cancelled");
        assert!(matches!(error, ProviderError::Aborted));
        assert_eq!(pool.in_flight(), 1);
    }

    #[tokio::test]
    async fn a_full_budget_queues_desks_until_memory_is_released() {
        let dir = tempfile::tempdir().unwrap();
        let broker = Broker::new(dir.path().join("broker"))
            .with_pressure_file(dir.path().join("no-pressure"));
        let budget = 512;
        // Fill the whole budget with one reservation the desks must wait behind.
        let Submission::Admitted(filler) = broker
            .submit(
                Request {
                    bytes: budget,
                    gpu: false,
                    priority: Priority::Background,
                    label: "filler".into(),
                },
                budget,
            )
            .unwrap()
        else {
            panic!("an empty ledger admits the filler");
        };
        let desks = Desks::with_budget(broker, budget);

        let waiting = tokio::time::timeout(Duration::from_millis(50), desks.seat("waiting")).await;
        assert!(waiting.is_err(), "desk must queue behind a full budget");

        drop(filler);
        let desk = desks.seat("admitted").await.unwrap();
        assert!(desk.is_some());
    }

    #[tokio::test]
    async fn a_turn_waiting_on_a_queued_command_gives_its_desk_back() {
        let dir = tempfile::tempdir().unwrap();
        let broker = Broker::new(dir.path().join("broker"))
            .with_pressure_file(dir.path().join("no-pressure"));
        let budget = 512;
        let desks = Desks::with_budget(broker.clone(), budget);
        let parent = desks.seat("parent").await.unwrap();
        let turn = TurnDesk::new(desks, "parent".into(), parent);
        // The parent's desk fills the budget, so a full-budget command queues.
        let Submission::Queued(command) = broker
            .submit(
                Request {
                    bytes: budget,
                    gpu: false,
                    priority: Priority::Background,
                    label: "full-budget command".into(),
                },
                budget,
            )
            .unwrap()
        else {
            panic!("the parent's desk leaves no room for a full-budget command");
        };
        // The turn waits on the command. Its desk must be given back meanwhile, or
        // the command never admits and the wait never ends.
        let cancel = CancellationToken::new();
        let waited = tokio::time::timeout(
            Duration::from_secs(5),
            turn.hold(release_while(&cancel, async {
                drop(command.admitted().await);
            })),
        )
        .await;
        waited
            .expect("the queued command admits once the parent's desk is released")
            .unwrap();
        assert!(turn.lease.lock().is_some(), "the turn takes its desk back");
    }
}
