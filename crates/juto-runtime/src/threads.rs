//! Internal main threads.
//!
//! A main session holds no conversation of its own. Its journal records
//! routing over internal *threads*: each thread is an append-only session with
//! its own context, so a request only carries its topic's history and each
//! thread's prompt prefix stays stable for provider prompt caching.
//!
//! Every request is routed before it runs:
//! 1. With no threads yet, it starts one.
//! 2. Otherwise the judge ([`Runtime::judge`]) chooses among the active
//!    threads' labels, a new thread, or (when dormant threads exist) a
//!    history lookup. A low-confidence choice also becomes a lookup.
//! 3. A lookup asks the `@tiny` model for a search query and scores every
//!    thread's label and messages against it. A hit resumes that thread; a
//!    miss starts a new one.
//!
//! After a completed turn the `@tiny` model relabels the thread. Past
//! `maxActiveThreads`, the least recently routed thread goes dormant; it stays
//! reachable through lookup. Degraded routing (judge or query model
//! unavailable) is recorded on the route as a `note` and reported to the host;
//! it never fails the request.

use std::{
    collections::{BTreeSet, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
use juto_agent::{AgentEvent, RunControl, RunOutcome};
use juto_ai::{Choice, Context, Decision, Message, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    Runtime,
    session::{EntryKind, Session},
};

const CREATED: &str = "thread_created";
const ROUTED: &str = "thread_routed";
const LABELED: &str = "thread_labeled";
const STATUS: &str = "thread_status";

/// Judge option key for starting a new thread.
const NEW_KEY: &str = "new";
/// Judge option key for searching dormant history.
const LOOKUP_KEY: &str = "lookup";
/// A judge choice below this confidence is checked against history first.
const LOOKUP_BELOW_CONFIDENCE: f64 = 0.5;
/// Fraction of query terms a thread must match to be resumed by a lookup.
const MATCH_THRESHOLD: f64 = 0.5;

/// System prompt for the `@tiny` call that names a thread.
pub(crate) const LABEL_SYSTEM: &str = "You name conversation threads. Reply with one label of at most eight words naming the thread's current topic. No quotes, no trailing punctuation, nothing else.";
/// System prompt for the `@tiny` call that writes a history search query.
pub(crate) const QUERY_SYSTEM: &str = "You write search queries over earlier conversation threads. Given a user's request, reply with three to twelve keywords that would find the earlier thread it refers to. Reply with the keywords only.";

/// Whether the router may offer a thread to the judge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStatus {
    /// Offered to the judge by label.
    Active,
    /// Reachable only through history lookup or a direct target.
    Dormant,
}

/// One internal thread of a main session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadInfo {
    /// Thread id; equal to its session journal's header id.
    pub id: String,
    /// Condensed topic label the router judges by.
    pub label: String,
    pub status: ThreadStatus,
    /// Session journal holding this thread's messages.
    pub session_path: PathBuf,
    /// Creation time (ms since the epoch).
    pub created: u64,
    /// Last time a request was routed here (ms since the epoch).
    pub last_routed: u64,
}

/// How a request reached its thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    /// A new thread was started for it.
    New,
    /// It continued an active thread.
    Continue,
    /// A history lookup resumed a dormant thread.
    Resume,
    /// The host addressed the thread directly, e.g. to deliver a subagent
    /// report to the thread that spawned it.
    Direct,
}

/// A routing decision, as journaled in the main session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Route {
    pub thread: String,
    /// The thread's label when the request was routed.
    pub label: String,
    pub kind: RouteKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    /// History search query, when a lookup ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Why routing degraded (judge or query model unavailable), if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Thread lifecycle reported alongside a request's agent events.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThreadEvent {
    /// The request was routed; emitted before its agent events.
    ThreadRouted { route: Route },
    /// A thread was created, adopted, relabeled, resumed, or went dormant.
    ThreadUpdated { thread: ThreadInfo },
    /// Relabeling failed after the turn; the old label stays.
    ThreadLabelFailed { thread: String, error: String },
}

/// Event stream of a main-session request: routing plus the routed thread's
/// agent events. Serializes as the inner event, so JSON consumers see one
/// `type`-tagged object per line.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum MainEvent {
    Agent(AgentEvent),
    Thread(ThreadEvent),
}

/// One row of a main session's merged history, in time order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TimelineItem {
    /// A request was routed to a thread.
    Routed { route: Route },
    /// A message from a thread (`None`: from the main journal itself, which
    /// only sessions that predate threads contain).
    Message {
        thread: Option<String>,
        message: Message,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreatedRecord {
    thread: String,
    /// Journal file name inside the main session's thread directory.
    file: String,
    label: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelRecord {
    thread: String,
    label: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StatusRecord {
    thread: String,
    status: ThreadStatus,
}

/// A main session: its routing journal and the threads it records.
pub struct MainSession {
    journal: Session,
    threads: Vec<ThreadInfo>,
}

impl MainSession {
    /// Open an existing session as a main session. Any session file opens; one
    /// with messages but no threads (it predates threads) has its transcript
    /// adopted as a thread on the next request. Nothing is written here.
    pub fn open(path: &Path) -> Result<Self> {
        Self::load(Session::open(path)?)
    }

    fn load(journal: Session) -> Result<Self> {
        let dir = thread_dir(journal.path());
        let mut threads: Vec<ThreadInfo> = Vec::new();
        for entry in journal.active_entries()? {
            let EntryKind::Custom { custom_type, data } = &entry.kind else {
                continue;
            };
            match custom_type.as_str() {
                CREATED => {
                    let record: CreatedRecord = parse_record(custom_type, data)?;
                    threads.push(ThreadInfo {
                        id: record.thread,
                        label: record.label,
                        status: ThreadStatus::Active,
                        session_path: dir.join(record.file),
                        created: entry.timestamp,
                        last_routed: entry.timestamp,
                    });
                }
                ROUTED => {
                    let route: Route = parse_record(custom_type, data)?;
                    find_mut(&mut threads, &route.thread)?.last_routed = entry.timestamp;
                }
                LABELED => {
                    let record: LabelRecord = parse_record(custom_type, data)?;
                    find_mut(&mut threads, &record.thread)?.label = record.label;
                }
                STATUS => {
                    let record: StatusRecord = parse_record(custom_type, data)?;
                    find_mut(&mut threads, &record.thread)?.status = record.status;
                }
                _ => {}
            }
        }
        Ok(Self { journal, threads })
    }

    pub fn id(&self) -> &str {
        &self.journal.header().id
    }

    pub fn path(&self) -> &Path {
        self.journal.path()
    }

    /// Model the next request runs on; threads follow it.
    pub fn current_model(&self) -> Option<String> {
        self.journal.current_model()
    }

    /// Threads in creation order.
    pub fn threads(&self) -> &[ThreadInfo] {
        &self.threads
    }

    pub fn thread(&self, id: &str) -> Option<&ThreadInfo> {
        self.threads.iter().find(|thread| thread.id == id)
    }

    /// Routing records and every thread's messages, merged in time order.
    /// Messages a thread copied from the main journal (an adopted transcript)
    /// appear once.
    pub fn timeline(&self) -> Result<Vec<TimelineItem>> {
        let mut items: Vec<(u64, TimelineItem)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for entry in self.journal.active_entries()? {
            seen.insert(entry.id.clone());
            match &entry.kind {
                EntryKind::Message { message } => items.push((
                    entry.timestamp,
                    TimelineItem::Message {
                        thread: None,
                        message: message.clone(),
                    },
                )),
                EntryKind::Custom { custom_type, data } if custom_type == ROUTED => {
                    let route: Route = parse_record(custom_type, data)?;
                    items.push((entry.timestamp, TimelineItem::Routed { route }));
                }
                _ => {}
            }
        }
        for thread in &self.threads {
            let session = Session::open(&thread.session_path)
                .with_context(|| format!("opening thread {}", thread.id))?;
            for entry in session.active_entries()? {
                if !seen.insert(entry.id.clone()) {
                    continue;
                }
                if let EntryKind::Message { message } = &entry.kind {
                    items.push((
                        entry.timestamp,
                        TimelineItem::Message {
                            thread: Some(thread.id.clone()),
                            message: message.clone(),
                        },
                    ));
                }
            }
        }
        // Stable: on equal timestamps a routing record precedes the request
        // it routed, because main-journal rows are collected first.
        items.sort_by_key(|(timestamp, _)| *timestamp);
        Ok(items.into_iter().map(|(_, item)| item).collect())
    }

    /// A pre-thread session: messages in the main journal and no threads.
    fn is_legacy(&self) -> Result<bool> {
        Ok(self.threads.is_empty()
            && self
                .journal
                .active_entries()?
                .iter()
                .any(|entry| matches!(entry.kind, EntryKind::Message { .. })))
    }

    /// Copy a pre-thread transcript into a thread so routing can continue it.
    fn adopt_legacy(&mut self) -> Result<ThreadInfo> {
        let messages = self.journal.messages()?;
        let first_request = messages
            .iter()
            .find(|message| matches!(message, Message::User { .. }))
            .map(Message::text)
            .unwrap_or_default();
        let fork = self.journal.fork(&thread_dir(self.path()))?;
        let id = fork.header().id.clone();
        self.record_created(id, fork.path(), initial_label(&first_request))
    }

    fn create_thread(&mut self, label: String, model: &str) -> Result<ThreadInfo> {
        let parent = self.id().to_owned();
        let mut session = Session::create_child(
            &thread_dir(self.path()),
            &self.journal.header().cwd,
            &parent,
        )?;
        session.append(EntryKind::ModelChange {
            model: model.to_owned(),
        })?;
        let id = session.header().id.clone();
        self.record_created(id, session.path(), label)
    }

    fn record_created(&mut self, id: String, path: &Path, label: String) -> Result<ThreadInfo> {
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("thread journal has no file name")?
            .to_owned();
        let timestamp = self.append_record(
            CREATED,
            serde_json::to_value(CreatedRecord {
                thread: id.clone(),
                file,
                label: label.clone(),
            })?,
        )?;
        let info = ThreadInfo {
            id,
            label,
            status: ThreadStatus::Active,
            session_path: path.to_path_buf(),
            created: timestamp,
            last_routed: timestamp,
        };
        self.threads.push(info.clone());
        Ok(info)
    }

    /// Journal `route`. A dormant target becomes active first; the returned
    /// info reports that change.
    fn record_route(&mut self, route: &Route) -> Result<Option<ThreadInfo>> {
        let resumed = find(&self.threads, &route.thread)?.status == ThreadStatus::Dormant;
        if resumed {
            self.set_status(&route.thread, ThreadStatus::Active)?;
        }
        let timestamp = self.append_record(ROUTED, serde_json::to_value(route)?)?;
        let thread = find_mut(&mut self.threads, &route.thread)?;
        thread.last_routed = timestamp;
        Ok(resumed.then(|| thread.clone()))
    }

    fn set_label(&mut self, id: &str, label: &str) -> Result<ThreadInfo> {
        find(&self.threads, id)?;
        self.append_record(
            LABELED,
            serde_json::to_value(LabelRecord {
                thread: id.to_owned(),
                label: label.to_owned(),
            })?,
        )?;
        let thread = find_mut(&mut self.threads, id)?;
        thread.label = label.to_owned();
        Ok(thread.clone())
    }

    fn set_status(&mut self, id: &str, status: ThreadStatus) -> Result<ThreadInfo> {
        find(&self.threads, id)?;
        self.append_record(
            STATUS,
            serde_json::to_value(StatusRecord {
                thread: id.to_owned(),
                status,
            })?,
        )?;
        let thread = find_mut(&mut self.threads, id)?;
        thread.status = status;
        Ok(thread.clone())
    }

    /// Put the least recently routed active threads to sleep until at most
    /// `cap` remain active. `keep` is never retired.
    fn retire_beyond(&mut self, cap: usize, keep: &str) -> Result<Vec<ThreadInfo>> {
        let mut retired = Vec::new();
        loop {
            let active: Vec<&ThreadInfo> = self
                .threads
                .iter()
                .filter(|thread| thread.status == ThreadStatus::Active)
                .collect();
            if active.len() <= cap {
                return Ok(retired);
            }
            let Some(oldest) = active
                .iter()
                .filter(|thread| thread.id != keep)
                .min_by_key(|thread| thread.last_routed)
                .map(|thread| thread.id.clone())
            else {
                return Ok(retired);
            };
            retired.push(self.set_status(&oldest, ThreadStatus::Dormant)?);
        }
    }

    /// Append a routing record; returns its journal timestamp.
    fn append_record(&mut self, custom_type: &str, data: Value) -> Result<u64> {
        let id = self.journal.append(EntryKind::Custom {
            custom_type: custom_type.to_owned(),
            data,
        })?;
        self.journal
            .active_entries()?
            .last()
            .filter(|entry| entry.id == id)
            .map(|entry| entry.timestamp)
            .context("appended routing record is not the journal leaf")
    }
}

/// Where a request goes before its thread exists.
struct Plan {
    /// `None` starts a new thread.
    thread: Option<String>,
    kind: RouteKind,
    confidence: Option<f64>,
    rationale: Option<String>,
    query: Option<String>,
    note: Option<String>,
}

impl Plan {
    fn new_thread() -> Self {
        Self {
            thread: None,
            kind: RouteKind::New,
            confidence: None,
            rationale: None,
            query: None,
            note: None,
        }
    }

    /// The router could not decide: continue the most recent active thread,
    /// or start one if none is active. `note` records why.
    fn fallback(active: &[&ThreadInfo], note: String) -> Self {
        let mut plan = Self::new_thread();
        if let Some(thread) = active.first() {
            plan.thread = Some(thread.id.clone());
            plan.kind = RouteKind::Continue;
        }
        plan.note = Some(note);
        plan
    }

    fn from_decision(decision: &Decision, thread: Option<String>, kind: RouteKind) -> Self {
        Self {
            thread,
            kind,
            confidence: decision.confidence,
            rationale: decision.rationale.clone(),
            query: None,
            note: None,
        }
    }
}

impl Runtime {
    /// Create a main session on the configured model.
    pub fn new_main_session(&self) -> Result<MainSession> {
        MainSession::load(self.new_session()?)
    }

    /// Record `selector` as the main session's model; each thread switches to
    /// it on its next request.
    pub fn switch_main_model(&self, main: &mut MainSession, selector: &str) -> Result<()> {
        self.switch_model(&mut main.journal, selector)
    }

    /// Route `prompt` to one of `main`'s threads and run it there. `target`
    /// bypasses the router and runs in that thread (e.g. a subagent report for
    /// the thread that spawned it). `events` receives the routing decision
    /// before the thread's agent events, then any label or status changes.
    /// Routing, lookup, and labeling usage is added to the returned outcome.
    pub async fn send(
        &self,
        main: &mut MainSession,
        target: Option<&str>,
        prompt: impl Into<String>,
        events: mpsc::UnboundedSender<MainEvent>,
        control: RunControl,
    ) -> Result<RunOutcome> {
        let prompt = prompt.into();
        if main.journal.header().cwd != self.inner.cwd {
            bail!("session belongs to a different working directory");
        }
        let cancel = control.token();
        let emit = |event: ThreadEvent| {
            let _ = events.send(MainEvent::Thread(event));
        };
        if main.is_legacy()? {
            emit(ThreadEvent::ThreadUpdated {
                thread: main.adopt_legacy()?,
            });
        }
        let mut overhead = Usage::default();
        let plan = match target {
            Some(id) => {
                find(&main.threads, id)?;
                Plan {
                    thread: Some(id.to_owned()),
                    kind: RouteKind::Direct,
                    ..Plan::new_thread()
                }
            }
            None => {
                self.plan_route(main, &prompt, &cancel, &mut overhead)
                    .await?
            }
        };
        let model = main.current_model().unwrap_or_else(|| self.config().model);
        let thread_id = match plan.thread {
            Some(id) => id,
            None => {
                let thread = main.create_thread(initial_label(&prompt), &model)?;
                let id = thread.id.clone();
                emit(ThreadEvent::ThreadUpdated { thread });
                id
            }
        };
        let route = Route {
            label: find(&main.threads, &thread_id)?.label.clone(),
            thread: thread_id.clone(),
            kind: plan.kind,
            confidence: plan.confidence,
            rationale: plan.rationale,
            query: plan.query,
            note: plan.note,
        };
        if let Some(thread) = main.record_route(&route)? {
            emit(ThreadEvent::ThreadUpdated { thread });
        }
        emit(ThreadEvent::ThreadRouted {
            route: route.clone(),
        });
        for thread in main.retire_beyond(self.config().max_active_threads, &thread_id)? {
            emit(ThreadEvent::ThreadUpdated { thread });
        }

        let mut session = Session::open(&find(&main.threads, &thread_id)?.session_path)?;
        if session.current_model().as_deref() != Some(model.as_str()) {
            self.switch_model(&mut session, &model)?;
        }
        let (agent_events, mut receiver) = mpsc::unbounded_channel();
        let forward = events.clone();
        let forwarder = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                let _ = forward.send(MainEvent::Agent(event));
            }
        });
        let result = self.run(&mut session, prompt, agent_events, control).await;
        // The run dropped its sender; wait so every agent event precedes the
        // label update below.
        let _ = forwarder.await;
        let mut outcome = result?;
        outcome.usage.add(&overhead);

        if outcome.stop_reason == StopReason::Stop && !cancel.is_cancelled() {
            match self.relabel(&session, &route.label, &cancel).await {
                Ok((label, usage)) => {
                    outcome.usage.add(&usage);
                    if label != route.label {
                        emit(ThreadEvent::ThreadUpdated {
                            thread: main.set_label(&thread_id, &label)?,
                        });
                    }
                }
                Err(error) => emit(ThreadEvent::ThreadLabelFailed {
                    thread: thread_id,
                    error: format!("{error:#}"),
                }),
            }
        }
        Ok(outcome)
    }

    async fn plan_route(
        &self,
        main: &MainSession,
        prompt: &str,
        cancel: &CancellationToken,
        usage: &mut Usage,
    ) -> Result<Plan> {
        let mut active: Vec<&ThreadInfo> = main
            .threads
            .iter()
            .filter(|thread| thread.status == ThreadStatus::Active)
            .collect();
        active.sort_by_key(|thread| std::cmp::Reverse(thread.last_routed));
        let has_dormant = main
            .threads
            .iter()
            .any(|thread| thread.status == ThreadStatus::Dormant);
        if active.is_empty() && !has_dormant {
            return Ok(Plan::new_thread());
        }
        let options = route_options(&active, has_dormant);
        let judge = match self.judge().await {
            Ok(judge) => judge,
            Err(error) => {
                return Ok(Plan::fallback(
                    &active,
                    format!("router judge unavailable: {error:#}"),
                ));
            }
        };
        let context = route_context(prompt);
        let decision = tokio::select! {
            biased;
            _ = cancel.cancelled() => bail!("routing cancelled"),
            result = judge.choose(&context, &options) => result,
        };
        let decision = match decision {
            Ok(decision) => decision,
            Err(error) => {
                return Ok(Plan::fallback(
                    &active,
                    format!("router judge failed: {error}"),
                ));
            }
        };
        usage.add(&decision.usage);
        let chosen = thread_for_key(&active, &decision.key);
        if chosen.is_none() && decision.key != NEW_KEY && decision.key != LOOKUP_KEY {
            return Ok(Plan::fallback(
                &active,
                format!("router judge chose unknown option `{}`", decision.key),
            ));
        }
        let unsure = decision
            .confidence
            .is_some_and(|confidence| confidence < LOOKUP_BELOW_CONFIDENCE);
        if has_dormant && (decision.key == LOOKUP_KEY || unsure) {
            return self.lookup(main, prompt, &decision, cancel, usage).await;
        }
        Ok(match chosen {
            Some(thread) => {
                Plan::from_decision(&decision, Some(thread.id.clone()), RouteKind::Continue)
            }
            None => Plan::from_decision(&decision, None, RouteKind::New),
        })
    }

    /// Search every thread for the one `prompt` refers back to.
    async fn lookup(
        &self,
        main: &MainSession,
        prompt: &str,
        decision: &Decision,
        cancel: &CancellationToken,
        usage: &mut Usage,
    ) -> Result<Plan> {
        let mut note = None;
        let query = match self
            .tiny_text(QUERY_SYSTEM, excerpt(prompt, 4000), 64, cancel)
            .await
        {
            Ok((query, used)) if !query.trim().is_empty() => {
                usage.add(&used);
                query.trim().to_owned()
            }
            Ok(_) => {
                note = Some("query model returned no keywords; searched with the request".into());
                prompt.to_owned()
            }
            Err(error) => {
                if cancel.is_cancelled() {
                    bail!("routing cancelled");
                }
                note = Some(format!(
                    "query model failed ({error:#}); searched with the request"
                ));
                prompt.to_owned()
            }
        };
        let hit = best_match(&main.threads, &query)?;
        let (thread, kind) = match hit {
            Some(thread) if thread.status == ThreadStatus::Dormant => {
                (Some(thread.id.clone()), RouteKind::Resume)
            }
            Some(thread) => (Some(thread.id.clone()), RouteKind::Continue),
            None => (None, RouteKind::New),
        };
        let mut plan = Plan::from_decision(decision, thread, kind);
        plan.query = Some(query);
        plan.note = note;
        Ok(plan)
    }

    /// Name the thread after its latest exchange.
    async fn relabel(
        &self,
        session: &Session,
        current: &str,
        cancel: &CancellationToken,
    ) -> Result<(String, Usage)> {
        let messages = session.messages()?;
        let latest = |user: bool| {
            messages
                .iter()
                .rev()
                .find(|message| match message {
                    Message::User { .. } => user,
                    Message::Assistant(_) => !user,
                    _ => false,
                })
                .map(Message::text)
                .unwrap_or_default()
        };
        let input = format!(
            "Current label: {current}\n\nLatest request:\n{}\n\nLatest reply:\n{}",
            excerpt(&latest(true), 1500),
            excerpt(&latest(false), 1500),
        );
        let (text, usage) = self.tiny_text(LABEL_SYSTEM, input, 32, cancel).await?;
        let label = clean_label(&text).context("label model returned no label")?;
        Ok((label, usage))
    }

    /// One short tool-free completion on the `@tiny` model chain.
    async fn tiny_text(
        &self,
        system: &str,
        input: String,
        max_tokens: u64,
        cancel: &CancellationToken,
    ) -> Result<(String, Usage)> {
        let (model, mut options) = self.resolve_model("@tiny")?;
        options.max_tokens = Some(max_tokens);
        options.thinking = None;
        let context = Context {
            system_prompt: vec![system.to_owned()],
            messages: vec![Message::user(input)],
            tools: Vec::new(),
        };
        let message = self
            .complete_once(&model, &options, &context, cancel)
            .await?;
        Ok((message.text(), message.usage))
    }
}

/// Journal directory for a main session's threads, beside its file.
fn thread_dir(main_path: &Path) -> PathBuf {
    main_path.with_extension("").join("threads")
}

fn parse_record<T: serde::de::DeserializeOwned>(custom_type: &str, data: &Value) -> Result<T> {
    serde_json::from_value(data.clone())
        .with_context(|| format!("main session has a malformed {custom_type} record"))
}

fn find<'a>(threads: &'a [ThreadInfo], id: &str) -> Result<&'a ThreadInfo> {
    threads
        .iter()
        .find(|thread| thread.id == id)
        .with_context(|| format!("unknown thread {id}"))
}

fn find_mut<'a>(threads: &'a mut [ThreadInfo], id: &str) -> Result<&'a mut ThreadInfo> {
    threads
        .iter_mut()
        .find(|thread| thread.id == id)
        .with_context(|| format!("main session references unknown thread {id}"))
}

/// Judge option key for the active thread at `index` (most recent first).
fn thread_key(index: usize) -> String {
    format!("thread-{}", index + 1)
}

fn thread_for_key<'a>(active: &[&'a ThreadInfo], key: &str) -> Option<&'a ThreadInfo> {
    let index: usize = key.strip_prefix("thread-")?.parse().ok()?;
    active.get(index.checked_sub(1)?).copied()
}

fn route_options(active: &[&ThreadInfo], has_dormant: bool) -> Vec<Choice> {
    let mut options: Vec<Choice> = active
        .iter()
        .enumerate()
        .map(|(index, thread)| {
            let recent = if index == 0 { " (most recent)" } else { "" };
            Choice::new(
                thread_key(index),
                format!("Continue the thread \"{}\"{recent}", thread.label),
            )
        })
        .collect();
    options.push(Choice::new(
        NEW_KEY,
        "Start a new thread: the request opens a topic none of the listed threads cover",
    ));
    if has_dormant {
        options.push(Choice::new(
            LOOKUP_KEY,
            "Search earlier inactive threads: the request refers back to past work none of the listed threads cover",
        ));
    }
    options
}

fn route_context(prompt: &str) -> String {
    format!(
        "Route a user's request to the conversation thread that should handle it. Each thread keeps its own history. Short follow-ups such as \"yes\" or \"continue\" usually belong to the most recent thread.\n\nRequest:\n{}",
        excerpt(prompt, 4000)
    )
}

/// Label for a thread before its first relabel: the request's first line.
fn initial_label(prompt: &str) -> String {
    let line = prompt
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        "Untitled thread".into()
    } else {
        excerpt(&line, 60)
    }
}

/// First non-empty line of a model's label reply, unquoted and bounded.
fn clean_label(raw: &str) -> Option<String> {
    let line = raw.lines().map(str::trim).find(|line| !line.is_empty())?;
    let line = line
        .trim_start_matches(['#', '*', '-'])
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '*'))
        .trim_end_matches('.')
        .trim();
    let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
    (!line.is_empty()).then(|| excerpt(&line, 80))
}

/// At most `max` characters of `text`, with an ellipsis when cut.
fn excerpt(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

const STOPWORDS: &[&str] = &[
    "about", "after", "again", "also", "and", "any", "are", "back", "been", "but", "can", "could",
    "did", "does", "earlier", "for", "from", "get", "had", "has", "have", "how", "into", "its",
    "just", "let", "lets", "like", "make", "more", "not", "now", "our", "please", "should", "some",
    "that", "the", "their", "them", "then", "there", "these", "this", "those", "was", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "would", "you", "your",
];

/// Lowercased words of at least three characters, without stopwords.
fn search_terms(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().count() >= 3)
        .map(str::to_lowercase)
        .filter(|word| !STOPWORDS.contains(&word.as_str()))
        .collect()
}

/// Share of `query` terms found in a thread: a label hit counts fully, a
/// message hit half. In `[0, 1]`.
fn match_score(query: &BTreeSet<String>, label: &BTreeSet<String>, body: &BTreeSet<String>) -> f64 {
    if query.is_empty() {
        return 0.0;
    }
    let total: f64 = query
        .iter()
        .map(|term| {
            if label.contains(term) {
                1.0
            } else if body.contains(term) {
                0.5
            } else {
                0.0
            }
        })
        .sum();
    total / query.len() as f64
}

/// The thread scoring highest for `query` at or above [`MATCH_THRESHOLD`];
/// ties go to the most recently routed.
fn best_match<'a>(threads: &'a [ThreadInfo], query: &str) -> Result<Option<&'a ThreadInfo>> {
    let terms = search_terms(query);
    if terms.is_empty() {
        return Ok(None);
    }
    let mut best: Option<(f64, &ThreadInfo)> = None;
    for thread in threads {
        let session = Session::open(&thread.session_path)
            .with_context(|| format!("opening thread {}", thread.id))?;
        let body: String = session
            .messages()?
            .iter()
            .filter(|message| matches!(message, Message::User { .. } | Message::Assistant(_)))
            .map(Message::text)
            .collect::<Vec<_>>()
            .join("\n");
        let score = match_score(&terms, &search_terms(&thread.label), &search_terms(&body));
        let better = best.is_none_or(|(top, current)| {
            score > top || (score == top && thread.last_routed > current.last_routed)
        });
        if score >= MATCH_THRESHOLD && better {
            best = Some((score, thread));
        }
    }
    Ok(best.map(|(_, thread)| thread))
}

#[cfg(test)]
#[path = "threads_tests.rs"]
mod tests;
