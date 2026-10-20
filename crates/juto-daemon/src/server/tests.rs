//! Daemon behavior through its request path: the Hello gate, run lifecycle
//! independent of clients, reattach snapshots, journal recovery, and idle wake.
//! The provider is scripted so a run can be held open while clients come and go.

use std::{path::Path, sync::atomic::AtomicBool, time::Duration};

use async_trait::async_trait;
use juto_ai::{
    AssistantMessage, ContentBlock, Context, CredentialStore, Provider, ProviderError,
    ProviderEvent, ProviderStream, StopReason, StreamOptions, Usage, timestamp_ms,
};
use juto_catalog::{Model, ModelCost};
use juto_runtime::ViewerRoute;
use juto_runtime::subagents::{MainMessage, MainMessageKind};
use juto_runtime::tools::{DESKTOP_SESSION_ARTIFACT_VERSION, DesktopSessionArtifact};
use juto_runtime::{EntryKind, SessionEntry, TaskSpec};
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

/// A daemon whose runtime lives under `root`, with no connections attached.
fn daemon(root: &Path, provider: Arc<dyn Provider>) -> Arc<Daemon> {
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
    Daemon::new(runtime, &workspace, root.join("desktop.sock")).unwrap()
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
    recover(&mut session).unwrap();

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

    let journal_text = std::fs::read_to_string(&view.info.path).unwrap();
    assert!(
        journal_text.contains(&format!("[subagent {child} Completed]")),
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
