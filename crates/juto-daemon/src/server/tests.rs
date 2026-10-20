//! Daemon behavior through its request path: the Hello gate, run lifecycle
//! independent of clients, reattach snapshots, journal recovery, and idle wake.
//! The provider is scripted so a run can be held open while clients come and go.

use std::{path::Path, sync::atomic::AtomicBool, time::Duration};

use async_trait::async_trait;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, CredentialStore, Message, Provider, ProviderError,
    ProviderEvent, ProviderStream, StopReason, StreamOptions, Usage, timestamp_ms,
};
use juto_catalog::{Model, ModelCost};
use juto_runtime::jobs::{Job, JobBudget, JobContract, JobStatus};
use juto_runtime::subagents::{MainMessage, MainMessageKind};
use juto_runtime::tools::{DESKTOP_SESSION_ARTIFACT_VERSION, DesktopSessionArtifact};
use juto_runtime::{
    Decision, DecisionStatus, EntryKind, Route, RouteKind, SessionEntry, TaskSpec, ThreadEvent,
    ViewerRoute,
};
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    sync::{Notify, mpsc},
};
use tokio_util::sync::CancellationToken;

use super::*;

/// Answers every request with one text reply. While `hold` is set, a request waits
/// on `gate` first, so a test controls exactly when the run can finish.
struct ScriptedProvider {
    hold: AtomicBool,
    gate: Notify,
}

impl ScriptedProvider {
    fn new(hold: bool) -> Arc<Self> {
        Arc::new(Self {
            hold: AtomicBool::new(hold),
            gate: Notify::new(),
        })
    }

    fn release(&self) {
        self.hold.store(false, Ordering::SeqCst);
        self.gate.notify_one();
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        if self.hold.load(Ordering::SeqCst) {
            self.gate.notified().await;
        }
        let events: Vec<Result<ProviderEvent, ProviderError>> = vec![
            Ok(ProviderEvent::Start),
            Ok(ProviderEvent::TextDelta {
                index: 0,
                delta: "done".to_string(),
            }),
            Ok(ProviderEvent::BlockEnd { index: 0 }),
            Ok(ProviderEvent::Done {
                message: AssistantMessage {
                    content: vec![ContentBlock::Text {
                        text: "done".to_string(),
                    }],
                    api: "mock".to_string(),
                    provider: "mock".to_string(),
                    model: "mock/test".to_string(),
                    usage: Usage {
                        input: 1,
                        output: 1,
                        cache_read: 0,
                        cache_write: 0,
                        total_tokens: 2,
                        cost: 0.0,
                    },
                    stop_reason: StopReason::Stop,
                    timestamp: timestamp_ms(),
                    response_id: None,
                    error: None,
                },
            }),
        ];
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

fn test_config() -> RuntimeConfig {
    RuntimeConfig {
        model: "mock/test".to_string(),
        custom_models: vec![Model {
            id: "test".to_string(),
            name: "Mock Model".to_string(),
            api: "openai-completions".to_string(),
            provider: "mock".to_string(),
            base_url: "http://localhost".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: Some(8192),
            max_tokens: Some(1024),
            kind: "chat".to_string(),
            supports_tools: true,
            compat: Value::Null,
        }],
        ..RuntimeConfig::default()
    }
}

/// Records the decisions the daemon reports to the operator.
#[derive(Default)]
struct RecordingNotifier(Mutex<Vec<String>>);

impl Notifier for RecordingNotifier {
    fn decision_blocked(&self, decision: &Decision) {
        self.0.lock().unwrap().push(decision.id.clone());
    }
}

/// A daemon whose runtime lives under `root`, with no connections attached.
fn daemon(root: &Path, provider: Arc<dyn Provider>) -> Arc<Daemon> {
    daemon_with_notifier(root, provider, Arc::new(RecordingNotifier::default()))
}

/// Like [`daemon`], with operator notices delivered to `notifier`.
fn daemon_with_notifier(
    root: &Path,
    provider: Arc<dyn Provider>,
    notifier: Arc<RecordingNotifier>,
) -> Arc<Daemon> {
    let workspace = root.join("workspace");
    let data_dir = root.join("data");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();
    let credentials = Arc::new(CredentialStore::open(data_dir.join("credentials.json")).unwrap());
    // Admission reads memory pressure from this file. It is absent, so no stall is
    // reported; the host's real PSI would otherwise hold a child's desk seat past the
    // 10 s deadlines when the machine is under memory pressure.
    let runtime = Runtime::with_provider_and_pressure_file(
        &workspace,
        &data_dir,
        test_config(),
        credentials,
        provider,
        None,
        root.join("no-pressure"),
    )
    .unwrap();
    Daemon::new(runtime, &workspace, root.join("desktop.sock"), notifier).unwrap()
}

/// A git repo colocated with jj, with bookmark `feat/x` on its initial commit, so
/// `publish` resolves a real bookmark in the daemon's workspace.
fn publishable_workspace(workspace: &Path) {
    std::fs::create_dir_all(workspace).unwrap();
    run_in(workspace, "git", &["init", "-q"]);
    run_in(
        workspace,
        "git",
        &[
            "-c",
            "user.name=Daemon Test",
            "-c",
            "user.email=daemon@example.test",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "initial",
        ],
    );
    run_in(workspace, "jj", &["git", "init", "--colocate"]);
    run_in(
        workspace,
        "jj",
        &["config", "set", "--repo", "user.name", "Daemon Test"],
    );
    run_in(
        workspace,
        "jj",
        &[
            "config",
            "set",
            "--repo",
            "user.email",
            "daemon@example.test",
        ],
    );
    run_in(workspace, "jj", &["describe", "-m", "wip"]);
    run_in(workspace, "jj", &["metaedit", "--update-author"]);
    run_in(
        workspace,
        "jj",
        &["bookmark", "create", "feat/x", "-r", "@-"],
    );
}

fn run_in(dir: &Path, program: &str, args: &[&str]) {
    let out = std::process::Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run command");
    assert!(
        out.status.success(),
        "{program} {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Stands in for a connected client: its events arrive on the returned receiver.
fn attach(daemon: &Daemon) -> mpsc::UnboundedReceiver<ServerEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    daemon.state.lock().unwrap().clients.push(tx);
    rx
}

async fn create_session(daemon: &Arc<Daemon>) -> SessionView {
    match daemon
        .request(Request::CreateSession)
        .await
        .expect("create session")
    {
        Reply::Session(view) => view,
        _ => panic!("expected a session reply"),
    }
}

async fn open_session(daemon: &Arc<Daemon>, session_id: &str) -> SessionView {
    let request = Request::OpenSession {
        session_id: session_id.into(),
    };
    match daemon.request(request).await.expect("open session") {
        Reply::Session(view) => view,
        _ => panic!("expected a session reply"),
    }
}

async fn start_run(daemon: &Arc<Daemon>, session_id: &str, text: &str) -> String {
    let request = Request::Prompt {
        session_id: session_id.into(),
        text: text.into(),
        unattended: false,
    };
    match daemon.request(request).await.expect("start run") {
        Reply::Started { run_id } => run_id,
        _ => panic!("expected a started run"),
    }
}

/// Waits until `check` holds, polling the daemon state; fails after ten seconds.
async fn until(mut check: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !check() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("condition not reached within 10s");
}

/// Returns the first session event matching `wanted`; fails after ten seconds.
async fn next_session_event(
    rx: &mut mpsc::UnboundedReceiver<ServerEvent>,
    mut wanted: impl FnMut(&SessionEvent) -> bool,
) -> SessionEvent {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match rx.recv().await {
                Some(ServerEvent::Session { event, .. }) if wanted(&event) => return event,
                Some(_) => {}
                None => panic!("event stream closed"),
            }
        }
    })
    .await
    .expect("expected session event within 10s")
}

fn journal(path: &Path) -> Vec<SessionEntry> {
    let session = Session::open(path).unwrap();
    session
        .active_entries()
        .unwrap()
        .into_iter()
        .cloned()
        .collect()
}

/// Data of every custom journal marker named `custom_type`, in journal order.
fn custom_markers(entries: &[SessionEntry], custom_type: &str) -> Vec<Value> {
    entries
        .iter()
        .filter_map(|entry| match &entry.kind {
            EntryKind::Custom {
                custom_type: found,
                data,
            } if found == custom_type => Some(data.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn first_request_must_be_a_compatible_hello() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let (client, server) = UnixStream::pair().unwrap();
    let served = tokio::spawn(connection(daemon, server));
    let (read_half, mut write_half) = client.into_split();
    let mut lines = BufReader::new(read_half).lines();

    let request = ClientFrame {
        id: 1,
        request: Request::ListSessions,
    };
    write_half
        .write_all(format!("{}\n", serde_json::to_string(&request).unwrap()).as_bytes())
        .await
        .unwrap();
    let line = lines.next_line().await.unwrap().unwrap();
    match serde_json::from_str::<ServerFrame>(&line).unwrap() {
        ServerFrame::Response {
            id,
            result: Err(error),
        } => {
            assert_eq!(id, 1);
            assert!(matches!(error.code, ErrorCode::ProtocolMismatch));
        }
        _ => panic!("expected a protocol mismatch response"),
    }
    served.await.unwrap().unwrap();
}

#[tokio::test]
async fn compatible_hello_is_answered_and_the_connection_is_served() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let (client, server) = UnixStream::pair().unwrap();
    let served = tokio::spawn(connection(daemon, server));
    let (read_half, mut write_half) = client.into_split();
    let mut lines = BufReader::new(read_half).lines();

    let hello = ClientFrame {
        id: 1,
        request: Request::Hello {
            client: "test".into(),
            protocol_version: PROTOCOL_VERSION,
        },
    };
    write_half
        .write_all(format!("{}\n", serde_json::to_string(&hello).unwrap()).as_bytes())
        .await
        .unwrap();
    let line = lines.next_line().await.unwrap().unwrap();
    match serde_json::from_str::<ServerFrame>(&line).unwrap() {
        ServerFrame::Response {
            id,
            result: Ok(Reply::Hello(info)),
        } => {
            assert_eq!(id, 1);
            assert_eq!(info.protocol_version, PROTOCOL_VERSION);
        }
        _ => panic!("expected a hello reply"),
    }

    let list = ClientFrame {
        id: 2,
        request: Request::ListSessions,
    };
    write_half
        .write_all(format!("{}\n", serde_json::to_string(&list).unwrap()).as_bytes())
        .await
        .unwrap();
    let line = lines.next_line().await.unwrap().unwrap();
    assert!(matches!(
        serde_json::from_str::<ServerFrame>(&line).unwrap(),
        ServerFrame::Response {
            id: 2,
            result: Ok(Reply::Sessions(_))
        }
    ));

    write_half.shutdown().await.unwrap();
    served.await.unwrap().unwrap();
}

#[tokio::test]
async fn run_completes_and_is_journaled_after_every_client_disconnects() {
    let root = tempfile::tempdir().unwrap();
    let provider = ScriptedProvider::new(true);
    let daemon = daemon(root.path(), provider.clone());
    let view = create_session(&daemon).await;
    let rx = attach(&daemon);
    let run_id = start_run(&daemon, &view.info.id, "long task").await;

    drop(rx);
    assert!(
        daemon.state.lock().unwrap().active.is_some(),
        "run must still be active"
    );
    provider.release();
    until(|| daemon.state.lock().unwrap().active.is_none()).await;

    let entries = journal(&view.info.path);
    let finished = custom_markers(&entries, "daemon_run_finished");
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0]["run_id"], run_id.as_str());
}

#[tokio::test]
async fn reattach_shows_the_live_run_and_then_the_committed_transcript() {
    let root = tempfile::tempdir().unwrap();
    let provider = ScriptedProvider::new(true);
    let daemon = daemon(root.path(), provider.clone());
    let view = create_session(&daemon).await;
    let run_id = start_run(&daemon, &view.info.id, "long task").await;

    // A client that connects mid-run sees the run in progress.
    let live = open_session(&daemon, &view.info.id).await;
    assert_eq!(
        live.info.run.as_ref().map(|run| run.run_id.as_str()),
        Some(run_id.as_str())
    );

    // One foreground run at a time.
    let busy = daemon
        .request(Request::Prompt {
            session_id: view.info.id.clone(),
            text: "again".into(),
            unattended: false,
        })
        .await;
    assert!(matches!(
        busy,
        Err(RpcError {
            code: ErrorCode::Busy,
            ..
        })
    ));

    let mut rx = attach(&daemon);
    provider.release();
    let finished =
        next_session_event(&mut rx, |e| matches!(e, SessionEvent::RunFinished { .. })).await;
    let SessionEvent::RunFinished {
        run_id: finished_id,
        error,
        ..
    } = finished
    else {
        unreachable!("filtered to RunFinished above");
    };
    assert_eq!(finished_id, run_id);
    assert!(error.is_none(), "run failed: {error:?}");

    let done = open_session(&daemon, &view.info.id).await;
    assert!(done.info.run.is_none());
    assert_eq!(
        done.committed, 2,
        "user and assistant messages are committed"
    );
}

#[tokio::test]
async fn prompt_routes_to_a_thread_and_subscribed_clients_see_the_route() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let view = create_session(&daemon).await;
    let mut rx = attach(&daemon);
    let run_id = start_run(&daemon, &view.info.id, "plan the release").await;

    let route = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match rx.recv().await {
                Some(ServerEvent::Thread {
                    session_id,
                    event: ThreadEvent::ThreadRouted { route },
                }) => {
                    assert_eq!(session_id, view.info.id);
                    return route;
                }
                Some(_) => {}
                None => panic!("event stream closed"),
            }
        }
    })
    .await
    .expect("a routed thread event within 10s");
    let finished =
        next_session_event(&mut rx, |e| matches!(e, SessionEvent::RunFinished { .. })).await;
    let SessionEvent::RunFinished {
        run_id: finished_id,
        error,
        ..
    } = finished
    else {
        unreachable!("filtered to RunFinished above");
    };
    assert_eq!(finished_id, run_id);
    assert!(error.is_none(), "run failed: {error:?}");

    let timeline = MainSession::open(&view.info.path)
        .unwrap()
        .timeline()
        .unwrap();
    assert!(
        timeline.iter().any(|item| matches!(
            item,
            TimelineItem::Routed { route: journaled } if journaled.thread == route.thread
        )),
        "the route is journaled on the main timeline"
    );
    let thread_messages = timeline
        .iter()
        .filter(|item| {
            matches!(item, TimelineItem::Message { thread: Some(t), .. } if *t == route.thread)
        })
        .count();
    assert_eq!(
        thread_messages, 2,
        "the user and assistant messages are in the routed thread"
    );
    let done = open_session(&daemon, &view.info.id).await;
    assert_eq!(done.committed, 2);
}

#[test]
fn restart_marks_only_unpaired_runs_as_interrupted() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("sessions");
    let cwd = root.path().join("workspace");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&cwd).unwrap();
    let mut session = Session::create(&dir, &cwd).unwrap();
    marker(&mut session, "daemon_run_started", "run-a").unwrap();
    marker(&mut session, "daemon_run_started", "run-b").unwrap();
    marker(&mut session, "daemon_run_finished", "run-b").unwrap();
    let path = session.path().to_owned();
    drop(session);

    // The daemon restarts: the run that never finished is recorded as interrupted.
    let mut session = Session::open(&path).unwrap();
    recover(&mut session, &[]).unwrap();

    let interrupted = custom_markers(&journal(&path), "run_interrupted");
    assert_eq!(interrupted.len(), 1);
    assert_eq!(interrupted[0]["run_id"], "run-a");
}

#[tokio::test]
async fn child_completion_starts_a_run_while_no_client_is_prompting() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let view = create_session(&daemon).await;
    let mut rx = attach(&daemon);
    tokio::spawn(wake_loop(daemon.clone()));

    let child = daemon
        .runtime
        .subagent_manager()
        .spawn(TaskSpec::new("say hello"))
        .unwrap();
    next_session_event(&mut rx, |e| matches!(e, SessionEvent::RunStarted { .. })).await;
    let finished =
        next_session_event(&mut rx, |e| matches!(e, SessionEvent::RunFinished { .. })).await;
    let SessionEvent::RunFinished { error, .. } = finished else {
        unreachable!()
    };
    assert!(error.is_none(), "wake run failed: {error:?}");

    // The wake run is routed like any prompt, so the child's completion is in a
    // thread message, not in the main journal's raw text.
    let timeline = MainSession::open(&view.info.path)
        .unwrap()
        .timeline()
        .unwrap();
    let thread_messages: String = timeline
        .iter()
        .filter(|item| {
            matches!(
                item,
                TimelineItem::Message {
                    thread: Some(_),
                    ..
                }
            )
        })
        .map(|item| serde_json::to_string(item).unwrap())
        .collect();
    assert!(
        thread_messages.contains(&format!("[subagent {child} Completed]")),
        "the wake run must carry the child's completion"
    );
}

#[tokio::test]
async fn inbox_notices_are_announced_listed_and_dismissed() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let mut rx = attach(&daemon);
    tokio::spawn(inbox_loop(daemon.clone()));

    let inbox = daemon.runtime.subagent_manager().main_inbox();
    inbox.push(MainMessage {
        from: "child-1".to_string(),
        name: None,
        kind: MainMessageKind::Completed,
        body: "done".to_string(),
        digest_of: None,
    });
    assert!(matches!(rx.recv().await, Some(ServerEvent::Inbox)));

    let Reply::InboxMessages { messages } = daemon.request(Request::InboxMessages).await.unwrap()
    else {
        panic!("expected the inbox");
    };
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].body, "done");

    daemon
        .request(Request::InboxDismiss {
            message: messages[0].clone(),
        })
        .await
        .unwrap();
    assert!(matches!(rx.recv().await, Some(ServerEvent::Inbox)));
    let Reply::InboxMessages { messages } = daemon.request(Request::InboxMessages).await.unwrap()
    else {
        panic!("expected the inbox");
    };
    assert!(messages.is_empty(), "a dismissed notice must not be listed");
}

#[tokio::test]
async fn inbox_notices_survive_a_daemon_restart() {
    let root = tempfile::tempdir().unwrap();
    let first = daemon(root.path(), ScriptedProvider::new(false));
    first.runtime.subagent_manager().main_inbox().push_from(
        Some("session-a".to_string()),
        MainMessage {
            from: "child-1".to_string(),
            name: None,
            kind: MainMessageKind::Completed,
            body: "done before restart".to_string(),
            digest_of: None,
        },
    );
    drop(first);

    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let Reply::InboxMessages { messages } = daemon.request(Request::InboxMessages).await.unwrap()
    else {
        panic!("expected the inbox");
    };
    assert_eq!(messages.len(), 1, "the notice must survive the restart");
    assert_eq!(messages[0].body, "done before restart");
}

#[tokio::test]
async fn desktop_open_grants_the_session_viewer_and_the_artifact_resolves() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let Reply::DesktopEnvironment(env) = daemon
        // A short-lived app can exit before its scope is verified, which fails
        // verification (`verify_scope` polls `systemctl show`). Keep it running.
        .request(Request::DesktopOpen {
            argv: vec!["sleep".to_string(), "60".to_string()],
        })
        .await
        .unwrap()
    else {
        panic!("expected a desktop environment");
    };
    let artifact = DesktopSessionArtifact {
        version: DESKTOP_SESSION_ARTIFACT_VERSION,
        artifact_id: format!("dev-{}", env.env_id),
        title: "Developer compositor session".to_string(),
        host_id: env.host_id.clone(),
        environment_id: env.env_id.clone(),
        grant_id: env.grant_id.clone(),
    };
    let ViewerRoute::Local { env_id, .. } = daemon
        .runtime
        .desktop_artifact_resolver()
        .resolve(juto_runtime::SESSION_VIEWER_PRINCIPAL, &artifact)
        .unwrap()
    else {
        panic!("the granted artifact must resolve to a local route");
    };
    assert_eq!(env_id, env.env_id);
    daemon
        .request(Request::DesktopDestroy {
            env_id: env.env_id.clone(),
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn resolve_over_the_wire_accepts_the_artifact_desktop_open_granted() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let Reply::DesktopEnvironment(env) = daemon
        .request(Request::DesktopOpen {
            argv: vec!["sleep".to_string(), "60".to_string()],
        })
        .await
        .unwrap()
    else {
        panic!("expected a desktop environment");
    };
    let artifact = DesktopSessionArtifact {
        version: DESKTOP_SESSION_ARTIFACT_VERSION,
        artifact_id: format!("dev-{}", env.env_id),
        title: "Developer compositor session".to_string(),
        host_id: env.host_id.clone(),
        environment_id: env.env_id.clone(),
        grant_id: env.grant_id.clone(),
    };
    let Reply::Route(Some(ViewerRoute::Local { env_id, .. })) = daemon
        .request(Request::ResolveDesktopArtifact {
            artifact: serde_json::to_value(&artifact).unwrap(),
        })
        .await
        .unwrap()
    else {
        panic!("the artifact granted by DesktopOpen must resolve over the wire");
    };
    assert_eq!(env_id, env.env_id);
    daemon
        .request(Request::DesktopDestroy {
            env_id: env.env_id.clone(),
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn child_completion_wakes_the_session_that_spawned_it() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let spawning = create_session(&daemon).await;
    // A newer session becomes the one used last; the wake must not go there.
    let _newer = create_session(&daemon).await;
    let mut rx = attach(&daemon);
    tokio::spawn(wake_loop(daemon.clone()));

    let spec = TaskSpec {
        origin: Some(spawning.info.id.clone()),
        ..TaskSpec::new("say hello")
    };
    daemon.runtime.subagent_manager().spawn(spec).unwrap();

    let woken = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match rx.recv().await {
                Some(ServerEvent::Session {
                    session_id,
                    event: SessionEvent::RunStarted { .. },
                }) => return session_id,
                Some(_) => {}
                None => panic!("event stream closed"),
            }
        }
    })
    .await
    .expect("wake run started within 10s");
    assert_eq!(woken, spawning.info.id);
}

#[tokio::test]
async fn thread_events_reach_subscribed_clients() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let view = create_session(&daemon).await;
    let mut rx = attach(&daemon);
    let route = Route {
        thread: "thread-1".into(),
        label: "Protocol surface".into(),
        kind: RouteKind::New,
        confidence: None,
        rationale: None,
        query: None,
        note: None,
    };

    daemon.emit(ServerEvent::Thread {
        session_id: view.info.id.clone(),
        event: ThreadEvent::ThreadRouted {
            route: route.clone(),
        },
    });

    let delivered = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("thread event within 10s");
    match delivered {
        Some(ServerEvent::Thread {
            session_id,
            event: ThreadEvent::ThreadRouted { route: got },
        }) => {
            assert_eq!(session_id, view.info.id);
            assert_eq!(got, route);
        }
        other => panic!("expected a routed thread event, got {other:?}"),
    }
}

#[tokio::test]
async fn timeline_request_answers_for_a_session() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon(root.path(), ScriptedProvider::new(false));
    let view = create_session(&daemon).await;

    let Reply::Timeline { items } = daemon
        .request(Request::Timeline {
            session_id: view.info.id.clone(),
        })
        .await
        .expect("timeline")
    else {
        panic!("expected a timeline reply");
    };
    assert!(items.is_empty(), "a fresh session has no routed history");
}

fn assistant(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content,
        api: "mock".to_string(),
        provider: "mock".to_string(),
        model: "mock/test".to_string(),
        usage: Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 2,
            cost: 0.0,
        },
        stop_reason,
        timestamp: timestamp_ms(),
        response_id: None,
        error: None,
    }
}

/// Asks for `publish` until a tool result is in, then answers with text. An
/// unattended run reaches the publish gate on its first request.
struct PublishProvider;

#[async_trait]
impl Provider for PublishProvider {
    async fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let message = if matches!(context.messages.last(), Some(Message::ToolResult { .. })) {
            assistant(
                vec![ContentBlock::Text {
                    text: "done".to_string(),
                }],
                StopReason::Stop,
            )
        } else {
            assistant(
                vec![ContentBlock::ToolCall(juto_ai::ToolCall {
                    id: format!("publish-{}", context.messages.len()),
                    name: "publish".to_string(),
                    arguments: serde_json::json!({ "bookmark": "feat/x", "title": "Add x" }),
                    thought_signature: None,
                })],
                StopReason::ToolUse,
            )
        };
        let events: Vec<Result<ProviderEvent, ProviderError>> = vec![
            Ok(ProviderEvent::Start),
            Ok(ProviderEvent::Done { message }),
        ];
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// Number of `publish` calls that reached the tool, across the main journal and
/// every thread journal: an unattended run is routed to a thread, which journals
/// its own messages. A call the decision gate stopped is journaled as a
/// `Tool call rejected: blocked:` error and does not count.
fn publish_results(path: &Path) -> usize {
    let main = MainSession::open(path).unwrap();
    std::iter::once(path.to_path_buf())
        .chain(
            main.threads()
                .iter()
                .map(|thread| thread.session_path.clone()),
        )
        .map(|journal| {
            Session::open(&journal)
                .unwrap()
                .messages()
                .unwrap()
                .iter()
                .filter(|m| match m {
                    Message::ToolResult {
                        tool_name, content, ..
                    } => {
                        tool_name == "publish"
                            && !content.iter().any(|block| {
                                matches!(block, ContentBlock::Text { text }
                                    if text.starts_with("Tool call rejected: blocked:"))
                            })
                    }
                    _ => false,
                })
                .count()
        })
        .sum()
}

async fn start_unattended(daemon: &Arc<Daemon>, session_id: &str) {
    let request = Request::Prompt {
        session_id: session_id.into(),
        text: "publish the branch".into(),
        unattended: true,
    };
    daemon.request(request).await.expect("start unattended run");
}

async fn decisions(daemon: &Arc<Daemon>) -> Vec<Decision> {
    match daemon
        .request(Request::ListDecisions)
        .await
        .expect("list decisions")
    {
        Reply::Decisions(list) => list,
        _ => panic!("expected decisions"),
    }
}

async fn resolve(
    daemon: &Arc<Daemon>,
    decision_id: &str,
    approve: bool,
) -> Result<Reply, RpcError> {
    daemon
        .request(Request::ResolveDecision {
            decision_id: decision_id.into(),
            approve,
        })
        .await
}

#[tokio::test]
async fn unattended_publish_blocks_then_approval_runs_it_exactly_once() {
    let root = tempfile::tempdir().unwrap();
    publishable_workspace(&root.path().join("workspace"));
    let notifier = Arc::new(RecordingNotifier::default());
    let daemon = daemon_with_notifier(root.path(), Arc::new(PublishProvider), notifier.clone());
    let view = create_session(&daemon).await;
    let mut rx = attach(&daemon);

    start_unattended(&daemon, &view.info.id).await;
    next_session_event(&mut rx, |e| {
        matches!(e, SessionEvent::DecisionBlocked { .. })
    })
    .await;
    until(|| daemon.state.lock().unwrap().active.is_none()).await;

    let pending = decisions(&daemon).await;
    assert_eq!(pending.len(), 1);
    let decision = &pending[0];
    assert_eq!(decision.tool, "publish");
    assert_eq!(decision.status, DecisionStatus::Pending);
    assert_eq!(*notifier.0.lock().unwrap(), vec![decision.id.clone()]);
    assert_eq!(
        publish_results(&view.info.path),
        0,
        "a blocked action must not run"
    );

    let Reply::Decision { decision, run_id } = resolve(&daemon, &decision.id.clone(), true)
        .await
        .expect("approve")
    else {
        panic!("expected a decision reply");
    };
    assert_eq!(decision.status, DecisionStatus::Approved);
    assert!(run_id.is_some(), "approval starts the resumed run");
    until(|| daemon.state.lock().unwrap().active.is_none()).await;

    assert_eq!(
        publish_results(&view.info.path),
        1,
        "the approved action runs exactly once"
    );
    assert_eq!(
        decisions(&daemon).await[0].status,
        DecisionStatus::Consumed,
        "the grant is spent by the run it allowed"
    );
}

#[tokio::test]
async fn rejected_decision_never_runs_and_cannot_be_approved_later() {
    let root = tempfile::tempdir().unwrap();
    publishable_workspace(&root.path().join("workspace"));
    let daemon = daemon(root.path(), Arc::new(PublishProvider));
    let view = create_session(&daemon).await;
    let mut rx = attach(&daemon);

    start_unattended(&daemon, &view.info.id).await;
    next_session_event(&mut rx, |e| {
        matches!(e, SessionEvent::DecisionBlocked { .. })
    })
    .await;
    until(|| daemon.state.lock().unwrap().active.is_none()).await;
    let id = decisions(&daemon).await[0].id.clone();

    let Reply::Decision { decision, run_id } = resolve(&daemon, &id, false).await.expect("reject")
    else {
        panic!("expected a decision reply");
    };
    assert_eq!(decision.status, DecisionStatus::Rejected);
    assert!(run_id.is_none(), "rejection starts no run");
    assert!(daemon.state.lock().unwrap().active.is_none());
    assert_eq!(publish_results(&view.info.path), 0);
    assert!(matches!(
        next_session_event(&mut rx, |e| matches!(
            e,
            SessionEvent::DecisionResolved { .. }
        ))
        .await,
        SessionEvent::DecisionResolved { .. }
    ));

    let late = resolve(&daemon, &id, true).await;
    assert!(matches!(
        late,
        Err(RpcError {
            code: ErrorCode::Invalid,
            ..
        })
    ));
    assert_eq!(publish_results(&view.info.path), 0);
}

#[tokio::test]
async fn pending_decision_survives_a_daemon_restart_and_is_still_approvable() {
    let root = tempfile::tempdir().unwrap();
    publishable_workspace(&root.path().join("workspace"));
    let first = daemon(root.path(), Arc::new(PublishProvider));
    let view = create_session(&first).await;
    let mut rx = attach(&first);
    start_unattended(&first, &view.info.id).await;
    next_session_event(&mut rx, |e| {
        matches!(e, SessionEvent::DecisionBlocked { .. })
    })
    .await;
    until(|| first.state.lock().unwrap().active.is_none()).await;
    let id = decisions(&first).await[0].id.clone();
    drop(first);

    // A restarted daemon has no sessions loaded until a client opens one.
    let restarted = daemon(root.path(), Arc::new(PublishProvider));
    let pending = decisions(&restarted).await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, id);
    assert_eq!(pending[0].status, DecisionStatus::Pending);

    let opened = Request::OpenSessionPath {
        path: view.info.path.clone(),
    };
    restarted
        .request(opened)
        .await
        .expect("open session by path");
    let Reply::Decision { run_id, .. } = resolve(&restarted, &id, true)
        .await
        .expect("approve after restart")
    else {
        panic!("expected a decision reply");
    };
    assert!(run_id.is_some());
    until(|| restarted.state.lock().unwrap().active.is_none()).await;
    assert_eq!(publish_results(&view.info.path), 1);
}

fn init_repo(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&["config", "user.name", "Daemon Test"]);
    git(&["config", "user.email", "daemon@example.test"]);
    std::fs::write(dir.join("README.md"), "# Initial\n").unwrap();
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "initial"]);
}

/// Like [`daemon`], with the workspace a Git repository, as background jobs require.
fn daemon_in_repo(root: &Path, provider: Arc<dyn Provider>) -> Arc<Daemon> {
    init_repo(&root.join("workspace"));
    daemon(root, provider)
}

async fn start_job(
    daemon: &Arc<Daemon>,
    prompt: &str,
    contract: JobContract,
) -> Result<Job, RpcError> {
    let request = Request::StartJob {
        prompt: prompt.into(),
        budget: JobBudget {
            wall_secs: 60,
            max_attempts: 1,
            max_turns: None,
            max_spend: None,
        },
        contract,
    };
    match daemon.request(request).await? {
        Reply::Job(job) => Ok(job),
        _ => panic!("expected a job reply"),
    }
}

async fn job(daemon: &Arc<Daemon>, job_id: &str) -> Job {
    let request = Request::GetJob {
        job_id: job_id.into(),
    };
    match daemon.request(request).await.expect("get job") {
        Reply::Job(job) => job,
        _ => panic!("expected a job reply"),
    }
}

async fn drain(daemon: &Arc<Daemon>) -> Vec<Job> {
    match daemon
        .request(Request::DrainJobs)
        .await
        .expect("drain jobs")
    {
        Reply::Jobs(jobs) => jobs,
        _ => panic!("expected jobs"),
    }
}

/// Polls the job until `check` holds; fails after ten seconds.
async fn wait_for_job(daemon: &Arc<Daemon>, job_id: &str, check: impl Fn(&Job) -> bool) -> Job {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let current = job(daemon, job_id).await;
            if check(&current) {
                return current;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("job condition not reached within 10s")
}

#[tokio::test]
async fn a_job_runs_without_a_client_and_its_result_is_drained_once() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon_in_repo(root.path(), ScriptedProvider::new(false));
    let contract = JobContract {
        no_change: true,
        ..Default::default()
    };
    let started = start_job(&daemon, "summarize the repo", contract)
        .await
        .expect("start job");

    let finished = wait_for_job(&daemon, &started.id, |job| job.status.is_terminal()).await;
    assert_eq!(finished.status, JobStatus::Succeeded);
    assert_eq!(finished.result.as_deref(), Some("done"));

    let drained = drain(&daemon).await;
    let ids: Vec<&str> = drained.iter().map(|job| job.id.as_str()).collect();
    assert_eq!(ids, vec![started.id.as_str()]);
    assert!(
        drain(&daemon).await.is_empty(),
        "a finished job is drained once"
    );
}

#[tokio::test]
async fn a_blocked_job_resumes_when_its_decision_is_approved_through_resolve_decision() {
    let root = tempfile::tempdir().unwrap();
    publishable_workspace(&root.path().join("workspace"));
    let daemon = daemon(root.path(), Arc::new(PublishProvider));
    let contract = JobContract {
        no_change: true,
        ..Default::default()
    };
    let started = start_job(&daemon, "publish the branch", contract)
        .await
        .expect("start job");

    let blocked = wait_for_job(&daemon, &started.id, |job| job.status == JobStatus::Blocked).await;
    let decision_id = blocked
        .decision_id
        .clone()
        .expect("a blocked job names its decision");
    let listed = decisions(&daemon)
        .await
        .into_iter()
        .find(|decision| decision.id == decision_id)
        .expect("the job's decision is listed");
    assert_eq!(listed.job_id.as_deref(), Some(started.id.as_str()));
    assert_eq!(listed.status, DecisionStatus::Pending);

    let Reply::Decision { decision, run_id } =
        resolve(&daemon, &decision_id, true).await.expect("approve")
    else {
        panic!("expected a decision reply");
    };
    assert_eq!(decision.status, DecisionStatus::Approved);
    assert!(
        run_id.is_none(),
        "a job decision resumes the job, not a session run"
    );

    let finished = wait_for_job(&daemon, &started.id, |job| job.status.is_terminal()).await;
    assert_eq!(finished.status, JobStatus::Succeeded);
    let session = finished.session_path.expect("the job keeps its session");
    assert_eq!(
        publish_results(&session),
        1,
        "the approved action runs exactly once"
    );
}

#[tokio::test]
async fn a_job_without_a_completion_contract_is_refused_and_never_recorded() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon_in_repo(root.path(), ScriptedProvider::new(false));
    let error = start_job(&daemon, "just do it", JobContract::default())
        .await
        .expect_err("a job without a contract is refused");
    assert!(matches!(error.code, ErrorCode::Invalid));
    assert!(drain(&daemon).await.is_empty());
}

#[tokio::test]
async fn job_changes_are_published_to_attached_clients() {
    let root = tempfile::tempdir().unwrap();
    let daemon = daemon_in_repo(root.path(), ScriptedProvider::new(false));
    let mut rx = attach(&daemon);
    forward_job_events(&daemon);
    let contract = JobContract {
        no_change: true,
        ..Default::default()
    };
    let started = start_job(&daemon, "summarize the repo", contract)
        .await
        .expect("start job");

    let finished_id = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(event) = rx.recv().await {
            if let ServerEvent::Job { job } = event
                && job.status.is_terminal()
            {
                return job.id;
            }
        }
        panic!("client channel closed before the job finished");
    })
    .await
    .expect("a finished job event within 10s");
    assert_eq!(finished_id, started.id);
}
