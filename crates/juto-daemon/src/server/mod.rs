//! Workspace-scoped, user-authenticated runtime host. Connections never own runs.

mod services;

use crate::protocol::*;
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use juto_agent::{AgentEvent, RunControl};
use juto_runtime::{
    ApprovalRequest, ChannelApprovalHandler, Decision, EntryKind, Runtime, RuntimeConfig, Session,
    SubagentEvent, UnattendedOutcome,
};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::mpsc,
};
use tokio_util::sync::CancellationToken;

struct ActiveRun {
    session_id: String,
    info: RunInfo,
    control: RunControl,
}
struct Approval {
    session_id: String,
    view: PendingApproval,
    request: ApprovalRequest,
}
#[derive(Default)]
struct State {
    active: Option<ActiveRun>,
    approvals: HashMap<u64, Approval>,
    clients: Vec<mpsc::UnboundedSender<ServerEvent>>,
    sessions: HashMap<String, PathBuf>,
    last_session: Option<String>,
}

pub(crate) struct Daemon {
    runtime: Runtime,
    state: Mutex<State>,
    ids: AtomicU64,
    logins: Mutex<HashMap<u64, CancellationToken>>,
    desktop_socket: PathBuf,
    /// Signalled when a run starts or ends; the idle-wake loop waits on it.
    run_changed: tokio::sync::Notify,
    /// Operator notice for unattended runs blocked on a decision.
    notifier: Arc<dyn Notifier>,
}

/// Best-effort operator notice that an unattended run is blocked on a decision.
/// Carries the capability and decision id only, never arguments or output.
pub(crate) trait Notifier: Send + Sync {
    fn decision_blocked(&self, decision: &Decision);
}

/// Desktop notification through `notify-send`. A missing binary is ignored: the
/// decision is still durable and listed by the daemon.
pub(crate) struct SystemNotifier;

impl Notifier for SystemNotifier {
    fn decision_blocked(&self, decision: &Decision) {
        let title = format!("Juto needs a decision: {}", decision.capability);
        let body = format!("Decision {}. Approve or reject it in Juto.", decision.id);
        drop(tokio::task::spawn_blocking(move || {
            std::process::Command::new("notify-send")
                .arg(title)
                .arg(body)
                .status()
        }));
    }
}

/// The message that resumes an approved run: it names the exact action to retry.
fn continuation_prompt(decision: &Decision) -> String {
    format!(
        "The operator approved the blocked action `{}` ({}), decision {}. Continue the task. Retry that exact action once; it is now permitted.",
        decision.tool, decision.capability, decision.id
    )
}

fn failed(error: impl std::fmt::Display) -> RpcError {
    RpcError::new(ErrorCode::Failed, error.to_string())
}
fn invalid(error: impl std::fmt::Display) -> RpcError {
    RpcError::new(ErrorCode::Invalid, error.to_string())
}
fn publish(state: &mut State, event: ServerEvent) {
    state
        .clients
        .retain(|client| client.send(event.clone()).is_ok());
}
fn session_event(state: &mut State, session_id: &str, event: SessionEvent) {
    publish(
        state,
        ServerEvent::Session {
            session_id: session_id.to_owned(),
            event,
        },
    );
}
fn message_count(session: &Session) -> Result<u64> {
    Ok(session
        .active_entries()?
        .iter()
        .filter(|e| matches!(e.kind, EntryKind::Message { .. }))
        .count() as u64)
}
fn marker(session: &mut Session, kind: &str, run_id: &str) -> Result<()> {
    session.append(EntryKind::Custom {
        custom_type: kind.into(),
        data: serde_json::json!({"run_id": run_id}),
    })?;
    Ok(())
}

/// Repairs interrupted runs before accepting clients, including unpaired tool calls.
fn recover(session: &mut Session) -> Result<()> {
    let mut pending = HashMap::new();
    for entry in session.active_entries()? {
        if let EntryKind::Custom { custom_type, data } = &entry.kind
            && let Some(id) = data["run_id"].as_str()
        {
            match custom_type.as_str() {
                "daemon_run_started" => {
                    pending.insert(id.to_owned(), ());
                }
                "daemon_run_finished" | "run_interrupted" => {
                    pending.remove(id);
                }
                _ => {}
            }
        }
    }
    for id in pending.keys() {
        session.append_interrupted(id, "daemon restarted before the run completed")?;
    }
    Ok(())
}

/// What the idle-wake loop waits on next.
enum Wake {
    Busy,
    Started,
    Idle,
}

const WAKE_RETRY: std::time::Duration = std::time::Duration::from_secs(1);

/// Keeps child completions from waiting for the next user prompt. While a run is active
/// the loop waits only for the run boundary: the run's own forwarder must receive
/// arrivals, and an idle waiter here would take them.
async fn wake_loop(daemon: Arc<Daemon>) {
    loop {
        match daemon.wake_idle() {
            Ok(Wake::Busy) => daemon.run_changed.notified().await,
            Ok(Wake::Started) => {}
            Ok(Wake::Idle) => {
                let inbox = daemon.runtime.subagent_manager().main_inbox();
                tokio::select! {
                    _ = inbox.arrived() => {}
                    _ = daemon.run_changed.notified() => {}
                }
            }
            Err(error) => {
                eprintln!("juto-daemon: could not start a run for child messages: {error:#}");
                tokio::time::sleep(WAKE_RETRY).await;
            }
        }
    }
}

impl Daemon {
    fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::Relaxed)
    }
    fn emit(&self, event: ServerEvent) {
        publish(
            &mut self.state.lock().expect("daemon state poisoned"),
            event,
        );
    }
    fn register(&self, session: &Session) -> Result<(), RpcError> {
        if session.header().cwd != self.runtime.cwd() {
            return Err(invalid("session belongs to another workspace"));
        }
        self.state
            .lock()
            .unwrap()
            .sessions
            .insert(session.header().id.clone(), session.path().to_owned());
        Ok(())
    }
    fn open(&self, id: &str) -> Result<Session, RpcError> {
        let path = self
            .state
            .lock()
            .unwrap()
            .sessions
            .get(id)
            .cloned()
            .ok_or_else(|| RpcError::new(ErrorCode::NotFound, "unknown session"))?;
        Session::open(&path).map_err(failed)
    }
    fn info(&self, session: &Session, state: &State) -> SessionInfo {
        SessionInfo {
            id: session.header().id.clone(),
            path: session.path().to_owned(),
            cwd: session.header().cwd.clone(),
            created_ms: session.header().created,
            model: session.current_model(),
            run: state
                .active
                .as_ref()
                .filter(|r| r.session_id == session.header().id)
                .map(|r| r.info.clone()),
        }
    }
    fn view(&self, session: Session) -> Result<Reply, RpcError> {
        let mut state = self.state.lock().unwrap();
        state.last_session = Some(session.header().id.clone());
        let entries = session
            .active_entries()
            .map_err(failed)?
            .into_iter()
            .cloned()
            .collect();
        Ok(Reply::Session(SessionView {
            info: self.info(&session, &state),
            entries,
            committed: message_count(&session).map_err(failed)?,
            pending_approvals: state
                .approvals
                .values()
                .filter(|a| a.session_id == session.header().id && !a.request.is_abandoned())
                .map(|a| a.view.clone())
                .collect(),
        }))
    }
    fn start(
        self: &Arc<Self>,
        session_id: String,
        text: String,
        unattended: bool,
    ) -> Result<Reply, RpcError> {
        let mut state = self.state.lock().unwrap();
        if state.active.is_some() {
            return Err(RpcError::new(
                ErrorCode::Busy,
                "a foreground run is already active",
            ));
        }
        if !state.sessions.contains_key(&session_id) {
            return Err(RpcError::new(ErrorCode::NotFound, "unknown session"));
        }
        let run_id = self
            .start_locked(&mut state, session_id, text, unattended)
            .map_err(failed)?;
        Ok(Reply::Started { run_id })
    }

    /// Resolves a durable decision. Approval starts a run that retries the blocked
    /// action once; rejection records the refusal and starts nothing. The decision
    /// is persisted before the run starts, so a restart keeps it.
    fn resolve_decision(
        self: &Arc<Self>,
        decision_id: String,
        approve: bool,
    ) -> Result<Reply, RpcError> {
        let store = self.runtime.decision_store();
        let pending = store
            .get(&decision_id)
            .map_err(failed)?
            .ok_or_else(|| RpcError::new(ErrorCode::NotFound, "unknown decision"))?;
        let mut state = self.state.lock().unwrap();
        if !approve {
            let decision = store.resolve(&decision_id, false).map_err(invalid)?;
            session_event(
                &mut state,
                &decision.session_id,
                SessionEvent::DecisionResolved {
                    decision: decision.clone(),
                },
            );
            return Ok(Reply::Decision {
                decision,
                run_id: None,
            });
        }
        if state.active.is_some() {
            return Err(RpcError::new(
                ErrorCode::Busy,
                "a foreground run is already active",
            ));
        }
        if !state.sessions.contains_key(&pending.session_id) {
            return Err(RpcError::new(ErrorCode::NotFound, "unknown session"));
        }
        let decision = store.resolve(&decision_id, true).map_err(invalid)?;
        let run_id = self
            .start_locked(
                &mut state,
                decision.session_id.clone(),
                continuation_prompt(&decision),
                true,
            )
            .map_err(failed)?;
        session_event(
            &mut state,
            &decision.session_id,
            SessionEvent::DecisionResolved {
                decision: decision.clone(),
            },
        );
        Ok(Reply::Decision {
            decision,
            run_id: Some(run_id),
        })
    }

    /// Starts a run owned by the daemon: nothing here depends on the connection that asked.
    /// The caller holds the state lock and has checked that no run is active.
    fn start_locked(
        self: &Arc<Self>,
        state: &mut State,
        session_id: String,
        text: String,
        unattended: bool,
    ) -> Result<String> {
        let path = state
            .sessions
            .get(&session_id)
            .context("unknown session")?
            .clone();
        let mut session = Session::open(&path)?;
        let base = message_count(&session)?;
        let run_id = uuid::Uuid::now_v7().to_string();
        marker(&mut session, "daemon_run_started", &run_id)?;
        let control = RunControl::new();
        state.active = Some(ActiveRun {
            session_id: session_id.clone(),
            info: RunInfo {
                run_id: run_id.clone(),
                started_ms: juto_ai::timestamp_ms(),
            },
            control: control.clone(),
        });
        state.last_session = Some(session_id.clone());
        session_event(
            state,
            &session_id,
            SessionEvent::RunStarted {
                run_id: run_id.clone(),
            },
        );
        self.run_changed.notify_one();
        let daemon = self.clone();
        let reply_id = run_id.clone();
        tokio::spawn(async move {
            let (tx, mut rx) = mpsc::unbounded_channel();
            let pump_daemon = daemon.clone();
            let pump_session = session_id.clone();
            let pump = tokio::spawn(async move {
                let mut index = base;
                while let Some(event) = rx.recv().await {
                    let event = match event {
                        AgentEvent::MessageEnd { message } => {
                            let event = SessionEvent::MessageCommitted { index, message };
                            index += 1;
                            event
                        }
                        event => SessionEvent::Agent { event },
                    };
                    session_event(&mut pump_daemon.state.lock().unwrap(), &pump_session, event);
                }
            });
            let outcome = if unattended {
                daemon
                    .runtime
                    .run_unattended(&mut session, text, tx, control, None)
                    .await
                    .map(|outcome| match outcome {
                        UnattendedOutcome::Finished(run) => (Some(run.stop_reason), None),
                        UnattendedOutcome::Blocked(decision) => (None, Some(decision)),
                    })
            } else {
                daemon
                    .runtime
                    .run(&mut session, text, tx, control)
                    .await
                    .map(|run| (Some(run.stop_reason), None))
            };
            let (stop_reason, blocked, mut error) = match outcome {
                Ok((stop_reason, blocked)) => (stop_reason, blocked, None),
                Err(e) => (None, None, Some(format!("{e:#}"))),
            };
            let pump_result = pump.await;
            if let Err(e) = pump_result {
                error = Some(format!("event pump failed: {e}"));
            }
            if let Err(e) = marker(&mut session, "daemon_run_finished", &run_id) {
                error = Some(format!("could not persist run completion: {e:#}"));
            }
            if let Some(decision) = &blocked {
                daemon.notifier.decision_blocked(decision);
            }
            let mut state = daemon.state.lock().unwrap();
            if let Some(decision) = blocked {
                session_event(
                    &mut state,
                    &session_id,
                    SessionEvent::DecisionBlocked { decision },
                );
            }
            let approvals: Vec<_> = state
                .approvals
                .iter()
                .filter(|(_, a)| a.session_id == session_id)
                .map(|(id, _)| *id)
                .collect();
            for id in approvals {
                state.approvals.remove(&id);
                session_event(
                    &mut state,
                    &session_id,
                    SessionEvent::ApprovalResolved { approval_id: id },
                );
            }
            state.active = None;
            session_event(
                &mut state,
                &session_id,
                SessionEvent::RunFinished {
                    run_id,
                    stop_reason,
                    error,
                },
            );
            daemon.run_changed.notify_one();
        });
        Ok(reply_id)
    }

    /// Starts a run that delivers child messages which arrived while no run was active.
    /// The inbox is drained under the state lock, so a concurrent prompt cannot race it.
    fn wake_idle(self: &Arc<Self>) -> Result<Wake> {
        let mut state = self.state.lock().unwrap();
        if state.active.is_some() {
            return Ok(Wake::Busy);
        }
        let Some(session_id) = state.last_session.clone() else {
            return Ok(Wake::Idle);
        };
        let inbox = self.runtime.subagent_manager().main_inbox();
        let messages = inbox.drain();
        if messages.is_empty() {
            return Ok(Wake::Idle);
        }
        let prompt = juto_runtime::subagents::completion_prompt(&messages);
        if let Err(error) = self.start_locked(&mut state, session_id, prompt, false) {
            inbox.requeue_front(messages);
            return Err(error);
        }
        Ok(Wake::Started)
    }

    async fn request(self: &Arc<Self>, request: Request) -> Result<Reply, RpcError> {
        match request {
            Request::Hello {
                protocol_version, ..
            } if protocol_version == PROTOCOL_VERSION => Ok(Reply::Hello(HelloInfo {
                protocol_version: PROTOCOL_VERSION,
                daemon_version: env!("CARGO_PKG_VERSION").into(),
                pid: std::process::id(),
                cwd: self.runtime.cwd().to_owned(),
                data_dir: self.runtime.data_dir().to_owned(),
                desktop_socket: self.desktop_socket.clone(),
            })),
            Request::Hello { .. } => Err(RpcError::new(
                ErrorCode::ProtocolMismatch,
                "incompatible protocol version",
            )),
            Request::ListSessions => {
                let paths: Vec<_> = self
                    .state
                    .lock()
                    .unwrap()
                    .sessions
                    .values()
                    .cloned()
                    .collect();
                let mut sessions = Vec::new();
                for path in paths {
                    let session = Session::open(&path).map_err(failed)?;
                    sessions.push(self.info(&session, &self.state.lock().unwrap()));
                }
                sessions.sort_by_key(|s| s.created_ms);
                Ok(Reply::Sessions(sessions))
            }
            Request::CreateSession => {
                let session = self.runtime.new_session().map_err(failed)?;
                self.register(&session)?;
                self.view(session)
            }
            Request::OpenSession { session_id } => self.view(self.open(&session_id)?),
            Request::OpenSessionPath { path } => {
                let session = Session::open(&path).map_err(failed)?;
                self.register(&session)?;
                self.view(session)
            }
            Request::Prompt {
                session_id,
                text,
                unattended,
            } => self.start(session_id, text, unattended),
            Request::ListDecisions => self
                .runtime
                .decision_store()
                .list()
                .map(Reply::Decisions)
                .map_err(failed),
            Request::ResolveDecision {
                decision_id,
                approve,
            } => self.resolve_decision(decision_id, approve),
            Request::Steer { session_id, text } => {
                let state = self.state.lock().unwrap();
                let run = state
                    .active
                    .as_ref()
                    .filter(|r| r.session_id == session_id)
                    .ok_or_else(|| invalid("session has no active run"))?;
                if !run.control.steer(juto_ai::Message::user(text)) {
                    return Err(invalid("the run has already ended"));
                }
                Ok(Reply::Ok)
            }
            Request::Cancel { session_id } => {
                let state = self.state.lock().unwrap();
                let run = state.active.as_ref().filter(|r| r.session_id == session_id);
                if let Some(run) = run {
                    run.control.cancel();
                }
                Ok(Reply::Cancelled {
                    was_running: run.is_some(),
                })
            }
            Request::SetSessionModel {
                session_id,
                selector,
            } => {
                let mut session = self.open(&session_id)?;
                let state = self.state.lock().unwrap();
                if state
                    .active
                    .as_ref()
                    .is_some_and(|r| r.session_id == session_id)
                {
                    return Err(RpcError::new(
                        ErrorCode::Busy,
                        "cannot change model during a run",
                    ));
                }
                self.runtime
                    .switch_model(&mut session, &selector)
                    .map_err(invalid)?;
                Ok(Reply::Ok)
            }
            Request::ApprovalRespond { approval_id, allow } => {
                let mut state = self.state.lock().unwrap();
                let approval = state
                    .approvals
                    .remove(&approval_id)
                    .ok_or_else(|| RpcError::new(ErrorCode::NotFound, "unknown approval"))?;
                approval.request.respond(allow);
                session_event(
                    &mut state,
                    &approval.session_id,
                    SessionEvent::ApprovalResolved { approval_id },
                );
                Ok(Reply::Ok)
            }
            Request::GetConfig => services::get_config(self),
            Request::ListModels => services::list_models(self),
            Request::SetModelRole { role, selector } => {
                services::set_model_role(self, role, selector)
            }
            Request::ListSubagents => services::list_subagents(self),
            Request::SubagentTranscript { id } => services::subagent_transcript(self, id).await,
            Request::SubagentCommand { id, command } => {
                services::subagent_command(self, id, command).await
            }
            Request::CredentialStatus { providers } => {
                services::credential_status(self, providers).await
            }
            Request::SetApiKey { provider, key } => {
                services::set_api_key(self, provider, key).await
            }
            Request::RemoveCredential { provider } => {
                services::remove_credential(self, provider).await
            }
            Request::LoginStart { provider } => services::login_start(self, provider),
            Request::LoginCancel { login_id } => services::login_cancel(self, login_id),
            Request::DesktopOpen { argv } => self.desktop_open(argv).await,
            Request::DesktopDestroy { env_id } => {
                let supervisor = self.runtime.desktop_supervisor();
                tokio::task::spawn_blocking(move || supervisor.destroy_environment(&env_id))
                    .await
                    .map_err(failed)?
                    .map_err(failed)?;
                Ok(Reply::Ok)
            }
            Request::ResolveDesktopArtifact { artifact } => {
                let Ok(artifact) =
                    serde_json::from_value::<juto_runtime::tools::DesktopSessionArtifact>(artifact)
                else {
                    return Ok(Reply::Route(None));
                };
                let principal = format!("user:{}", unsafe { libc::geteuid() });
                match self
                    .runtime
                    .desktop_artifact_resolver()
                    .resolve(&principal, &artifact)
                {
                    Ok(route) => Ok(Reply::Route(Some(route))),
                    Err(juto_runtime::ResolveError::Unsupported(_)) => Ok(Reply::Route(None)),
                    Err(e) => Err(failed(e)),
                }
            }
        }
    }

    async fn desktop_open(&self, argv: Vec<String>) -> Result<Reply, RpcError> {
        if argv.is_empty() {
            return Err(invalid("desktop launch requires a command"));
        }
        let supervisor = self.runtime.desktop_supervisor();
        let cwd = self.runtime.cwd().to_owned();
        tokio::task::spawn_blocking(move || {
            use juto_desktop::supervisor::{
                CompositorMode, EnvironmentConfig, OutputConfig, RenderProfile, SystemLimits,
            };
            let env_id = format!("desk-{}", uuid::Uuid::now_v7());
            let config = EnvironmentConfig {
                env_id: env_id.clone(),
                owner_principal: format!("user:{}", unsafe { libc::geteuid() }),
                output: OutputConfig {
                    width: 1280,
                    height: 720,
                    scale: 1,
                    fps: 30,
                },
                render_profile: RenderProfile::Software,
                limits: SystemLimits::default(),
                workspace_dir: cwd,
                lease_ttl_secs: 6,
                compositor_mode: if juto_desktop::worker_available() {
                    CompositorMode::Worker
                } else {
                    CompositorMode::InProcess
                },
                owner_heartbeat_timeout_secs: 0,
            };
            let env = supervisor
                .create_environment(config, &env_id)
                .map_err(failed)?;
            if let Err(error) = env.launch("app", &argv, None, &[], &env_id) {
                let _ = supervisor.destroy_environment(&env_id);
                return Err(failed(error));
            }
            Ok(Reply::DesktopEnvironment(DesktopEnvironmentInfo {
                env_id,
                width: 1280,
                height: 720,
            }))
        })
        .await
        .map_err(failed)?
    }
}

async fn send_frame(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    frame: ServerFrame,
) -> Result<()> {
    let mut bytes = serde_json::to_vec(&frame)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    Ok(())
}

async fn connection(daemon: Arc<Daemon>, stream: UnixStream) -> Result<()> {
    if stream.peer_cred()?.uid() != unsafe { libc::geteuid() } {
        bail!("connection belongs to another user");
    }
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();
    let Some(line) = lines.next_line().await? else {
        return Ok(());
    };
    let frame: ClientFrame = serde_json::from_str(&line)?;
    if !matches!(&frame.request, Request::Hello { protocol_version, .. } if *protocol_version == PROTOCOL_VERSION)
    {
        send_frame(
            &mut writer,
            ServerFrame::Response {
                id: frame.id,
                result: Err(RpcError::new(
                    ErrorCode::ProtocolMismatch,
                    "first request must be a compatible Hello",
                )),
            },
        )
        .await?;
        return Ok(());
    }
    let (tx, mut events) = mpsc::unbounded_channel();
    daemon.state.lock().unwrap().clients.push(tx);
    send_frame(
        &mut writer,
        ServerFrame::Response {
            id: frame.id,
            result: daemon.request(frame.request).await,
        },
    )
    .await?;
    loop {
        tokio::select! {
            line = lines.next_line() => {
                let Some(line) = line? else { return Ok(()); };
                let frame: ClientFrame = serde_json::from_str(&line)?;
                let result = daemon.request(frame.request).await;
                send_frame(&mut writer, ServerFrame::Response { id: frame.id, result }).await?;
            }
            Some(event) = events.recv() => { send_frame(&mut writer, ServerFrame::Event { event }).await?; }
        }
    }
}

fn private_dir(path: &Path) -> Result<()> {
    match std::fs::create_dir(path) {
        Ok(()) => std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        bail!(
            "runtime directory must be user-owned and mode 0700: {}",
            path.display()
        );
    }
    Ok(())
}

struct SocketGuard {
    _lock: File,
    daemon: PathBuf,
    desktop: PathBuf,
}
impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.daemon);
        let _ = std::fs::remove_file(&self.desktop);
    }
}

/// Runs one foreground daemon for `workspace`. A filesystem lock excludes a second
/// daemon; the 0700 directory, 0600 sockets and peer UID checks exclude other users.
pub async fn serve(workspace: PathBuf) -> Result<()> {
    let workspace = workspace.canonicalize().context("invalid workspace")?;
    let dir = paths::workspace_runtime_dir(&workspace)?;
    private_dir(dir.parent().context("missing runtime parent")?)?;
    private_dir(&dir)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("daemon.lock"))?;
    lock.try_lock_exclusive()
        .context("a daemon is already running for this workspace")?;
    let daemon_socket = paths::daemon_socket(&workspace)?;
    let desktop_socket = paths::desktop_socket(&workspace)?;
    for path in [&daemon_socket, &desktop_socket] {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    let _guard = SocketGuard {
        _lock: lock,
        daemon: daemon_socket.clone(),
        desktop: desktop_socket.clone(),
    };
    let listener = UnixListener::bind(&daemon_socket)?;
    std::fs::set_permissions(&daemon_socket, std::fs::Permissions::from_mode(0o600))?;
    let desktop_listener = UnixListener::bind(&desktop_socket)?;
    std::fs::set_permissions(&desktop_socket, std::fs::Permissions::from_mode(0o600))?;
    let data_dir = juto_runtime::default_data_dir()?;
    let config = RuntimeConfig::load(
        &data_dir.join("config.yml"),
        &workspace.join(".juto/config.yml"),
    )?;
    let (approval_handler, mut approvals) = ChannelApprovalHandler::channel();
    let runtime =
        Runtime::from_config_with_approval(&workspace, &data_dir, config, Some(approval_handler))?;
    let daemon = Arc::new(Daemon {
        runtime,
        state: Mutex::new(State::default()),
        ids: AtomicU64::new(1),
        logins: Mutex::new(HashMap::new()),
        desktop_socket,
        run_changed: tokio::sync::Notify::new(),
        notifier: Arc::new(SystemNotifier),
    });
    for entry in std::fs::read_dir(data_dir.join("sessions"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "jsonl") {
            let mut session =
                Session::open(&path).with_context(|| format!("opening {}", path.display()))?;
            if session.header().cwd == workspace {
                recover(&mut session)?;
                daemon.register(&session).map_err(anyhow::Error::new)?;
            }
        }
    }
    tokio::spawn(wake_loop(daemon.clone()));
    let approval_daemon = daemon.clone();
    tokio::spawn(async move {
        while let Some(request) = approvals.recv().await {
            if request.is_abandoned() {
                continue;
            }
            let mut state = approval_daemon.state.lock().unwrap();
            let Some(run) = &state.active else {
                request.respond(false);
                continue;
            };
            let session_id = run.session_id.clone();
            let approval_id = approval_daemon.next_id();
            let view = PendingApproval {
                approval_id,
                call: request.call().clone(),
                tier: request.tier(),
            };
            state.approvals.insert(
                approval_id,
                Approval {
                    session_id: session_id.clone(),
                    view: view.clone(),
                    request,
                },
            );
            session_event(
                &mut state,
                &session_id,
                SessionEvent::ApprovalRequested { approval: view },
            );
        }
    });
    let manager = daemon.runtime.subagent_manager();
    let mut subevents = manager.subscribe_events();
    let subdaemon = daemon.clone();
    tokio::spawn(async move {
        loop {
            match subevents.recv().await {
                Ok(event) => subdaemon.emit(ServerEvent::Subagent { event }),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    for child in manager.list() {
                        subdaemon.emit(ServerEvent::Subagent {
                            event: SubagentEvent::StatusChanged {
                                id: child.id,
                                status: child.status,
                            },
                        });
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    loop {
        tokio::select! {
            accepted = listener.accept() => { let (stream, _) = accepted?; let daemon = daemon.clone(); tokio::spawn(async move { let _ = connection(daemon, stream).await; }); }
            accepted = desktop_listener.accept() => {
                let (stream, _) = accepted?;
                let supervisor = daemon.runtime.desktop_supervisor();
                let stream = stream.into_std()?;
                stream.set_nonblocking(false)?;
                tokio::task::spawn_blocking(move || { let _ = juto_desktop::control::serve(stream, supervisor); });
            }
        }
    }
}

#[cfg(test)]
mod tests;
