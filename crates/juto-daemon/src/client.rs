//! Client side of the daemon protocol. Connects to a workspace's daemon, starts
//! it as a systemd `--user` unit when it is not running, and reconnects after loss.

mod upgrade;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc, oneshot};

use crate::protocol::{
    ClientFrame, ErrorCode, HelloInfo, PROTOCOL_VERSION, Reply, Request, RpcError, ServerEvent,
    ServerFrame, paths,
};

const CLIENT_NAME: &str = concat!("juto-daemon-client/", env!("CARGO_PKG_VERSION"));
const DAEMON_BINARY: &str = "juto-daemon";
/// `Hello` uses id 0 on every connection; requests count up from 1.
const HELLO_ID: u64 = 0;
const EVENT_CAPACITY: usize = 256;
const RECONNECT_START: Duration = Duration::from_millis(200);
const RECONNECT_CAP: Duration = Duration::from_secs(2);
const DAEMON_START_TIMEOUT: Duration = Duration::from_secs(15);
const DAEMON_START_POLL: Duration = Duration::from_millis(100);

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error(transparent)]
    Rpc(RpcError),
    #[error("daemon connection lost")]
    Disconnected,
    #[error("{0}")]
    Start(String),
}

#[derive(Debug, Clone)]
pub enum Notification {
    /// A reconnect succeeded. Sent only after a loss, never for the first connection.
    Connected(HelloInfo),
    /// The connection was lost. Requests fail with `Disconnected` until `Connected`.
    Disconnected,
    Event(ServerEvent),
}

/// Handle to one workspace's daemon. Clones share the connection. Dropping the last
/// clone closes the connection and stops the background task.
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

impl Client {
    /// Connects to the daemon for `cwd`, starting its systemd `--user` unit if it is
    /// not running. Returns after the `Hello` handshake. Must run inside a Tokio runtime.
    /// The daemon binary must be found too: a running daemon is checked against it.
    pub async fn connect(cwd: &Path) -> Result<Client, ClientError> {
        let cwd = std::fs::canonicalize(cwd).map_err(|e| {
            ClientError::Start(format!("cannot open workspace {}: {e}", cwd.display()))
        })?;
        let socket =
            paths::daemon_socket(&cwd).map_err(|e| ClientError::Start(format!("{e:#}")))?;
        let unit = paths::unit_name(&cwd);
        let binary = daemon_binary()?;
        open(Target {
            cwd,
            socket,
            unit,
            binary,
        })
        .await
    }

    /// Sends `request` and waits for its reply. New independent work and startup
    /// config wait for a pending upgrade before sending anything. Already-sent
    /// requests are never replayed after a connection loss.
    pub async fn request(&self, request: Request) -> Result<Reply, ClientError> {
        if request.starts_work() || matches!(request, Request::GetConfig) {
            self.wait_for_upgrade().await?;
        }
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let line = encode_frame(&ClientFrame { id, request })?;
        let (reply, response) = oneshot::channel();
        {
            let link = lock(&self.inner.link);
            let link = link.as_ref().ok_or(ClientError::Disconnected)?;
            link.outbound
                .send(Outgoing { id, line, reply })
                .map_err(|_| ClientError::Disconnected)?;
        }
        match response.await {
            Ok(result) => result.map_err(ClientError::Rpc),
            Err(_) => Err(ClientError::Disconnected),
        }
    }

    async fn wait_for_upgrade(&self) -> Result<(), ClientError> {
        loop {
            let changed = self.inner.upgrade_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.inner.upgrading.load(Ordering::Acquire) {
                return Ok(());
            }
            if let Some(error) = lock(&self.inner.upgrade_error).as_ref() {
                return Err(ClientError::Start(error.clone()));
            }
            changed.await;
        }
    }

    /// Receives notifications from this point on. A receiver that falls behind
    /// `EVENT_CAPACITY` messages gets `RecvError::Lagged`.
    pub fn subscribe(&self) -> broadcast::Receiver<Notification> {
        self.inner.events.subscribe()
    }

    /// The hello of the current connection, or `None` while disconnected.
    pub fn hello(&self) -> Option<HelloInfo> {
        lock(&self.inner.link)
            .as_ref()
            .map(|link| link.hello.clone())
    }
}

struct Inner {
    /// `Some` while connected.
    link: Mutex<Option<Link>>,
    events: broadcast::Sender<Notification>,
    next_id: AtomicU64,
    upgrade_error: Mutex<Option<String>>,
    upgrading: AtomicBool,
    upgrade_changed: tokio::sync::Notify,
}

struct Link {
    hello: HelloInfo,
    outbound: mpsc::UnboundedSender<Outgoing>,
    outdated: bool,
}

/// A request handed to the connection task, which owns the pending-reply map.
struct Outgoing {
    id: u64,
    line: String,
    reply: oneshot::Sender<Result<Reply, RpcError>>,
}

/// Where a workspace's daemon lives.
struct Target {
    cwd: PathBuf,
    socket: PathBuf,
    unit: String,
    /// Startup executable; an initial build mismatch initiates replacement.
    binary: PathBuf,
}

/// One established connection: a line reader and a writer.
struct Wire {
    lines: Lines<BufReader<OwnedReadHalf>>,
    write: OwnedWriteHalf,
    outdated: bool,
}

async fn open(target: Target) -> Result<Client, ClientError> {
    let (wire, hello) = connect_or_start(&target).await?;
    let (outbound, rx) = mpsc::unbounded_channel();
    let (events, _) = broadcast::channel(EVENT_CAPACITY);
    let inner = Arc::new(Inner {
        link: Mutex::new(Some(Link {
            hello,
            outdated: wire.outdated,
            outbound,
        })),
        events: events.clone(),
        next_id: AtomicU64::new(1),
        upgrade_error: Mutex::new(None),
        upgrading: AtomicBool::new(wire.outdated),
        upgrade_changed: tokio::sync::Notify::new(),
    });
    tokio::spawn(supervise(Arc::downgrade(&inner), target, events, wire, rx));
    Ok(Client { inner })
}

/// Owns the connection for the life of the client: runs each connection to its end,
/// then reconnects. Holds only a `Weak` handle, so dropping the client ends it.
async fn supervise(
    inner: Weak<Inner>,
    target: Target,
    events: broadcast::Sender<Notification>,
    mut wire: Wire,
    mut outbound: mpsc::UnboundedReceiver<Outgoing>,
) {
    let mut retiring_build: Option<String> = None;
    loop {
        let Some(current) = inner.upgrade() else {
            return;
        };
        let outdated = wire
            .outdated
            .then(|| lock(&current.link).as_ref().map(|link| link.hello.clone()))
            .flatten();
        if retiring_build.is_none() {
            retiring_build = outdated.as_ref().map(|hello| hello.binary_id.clone());
        }
        drop(current);
        let connection = run_connection(wire, outbound, &events);
        tokio::pin!(connection);
        tokio::select! {
            _ = &mut connection => {}
            result = async {
                match outdated {
                    Some(hello) => upgrade::replace(&target, hello).await,
                    None => std::future::pending().await,
                }
            } => {
                if let Err(error) = result {
                    eprintln!("juto: automatic daemon upgrade failed: {error}");
                    if let Some(current) = inner.upgrade() {
                        *lock(&current.upgrade_error) = Some(error.to_string());
                        current.upgrade_changed.notify_waiters();
                    }
                }
                connection.await;
            }
        }
        if !set_link(&inner, None) {
            return;
        }
        let _ = events.send(Notification::Disconnected);
        let Some((mut next, hello)) = reconnect(&inner, &target).await else {
            return;
        };
        // A replacement initiated by another client must not be "upgraded" back
        // to this already-running client's older executable.
        next.outdated &= retiring_build.as_deref() == Some(hello.binary_id.as_str());
        if !next.outdated {
            retiring_build = None;
        }
        if let Some(current) = inner.upgrade() {
            *lock(&current.upgrade_error) = None;
        }
        let (tx, rx) = mpsc::unbounded_channel();
        if !set_link(
            &inner,
            Some(Link {
                hello: hello.clone(),
                outdated: next.outdated,
                outbound: tx,
            }),
        ) {
            return;
        }
        let _ = events.send(Notification::Connected(hello));
        wire = next;
        outbound = rx;
    }
}

/// Serves one connection until it ends. Every pending request is dropped on return,
/// so its waiter fails with `Disconnected`.
async fn run_connection(
    mut wire: Wire,
    mut outbound: mpsc::UnboundedReceiver<Outgoing>,
    events: &broadcast::Sender<Notification>,
) {
    let mut pending: HashMap<u64, oneshot::Sender<Result<Reply, RpcError>>> = HashMap::new();
    loop {
        tokio::select! {
            line = wire.lines.next_line() => {
                let Ok(Some(line)) = line else { return };
                let Ok(frame) = serde_json::from_str::<ServerFrame>(&line) else { return };
                match frame {
                    ServerFrame::Response { id, result } => {
                        if let Some(reply) = pending.remove(&id) {
                            let _ = reply.send(result);
                        }
                    }
                    ServerFrame::Event { event } => {
                        let _ = events.send(Notification::Event(event));
                    }
                }
            }
            outgoing = outbound.recv() => {
                let Some(Outgoing { id, line, reply }) = outgoing else { return };
                pending.insert(id, reply);
                if wire.write.write_all(line.as_bytes()).await.is_err() {
                    return;
                }
            }
        }
    }
}

/// Retries until a connection is established, or returns `None` once the client is
/// dropped. Restarts the unit when the socket is unreachable and the unit is not running.
async fn reconnect(inner: &Weak<Inner>, target: &Target) -> Option<(Wire, HelloInfo)> {
    let mut delay = RECONNECT_START;
    loop {
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(RECONNECT_CAP);
        if inner.strong_count() == 0 {
            return None;
        }
        if let Ok(attached) = handshake(target).await {
            return Some(attached);
        }
        if !unit_running(&target.unit).await {
            // A failed start is retried on the next round like any other failed attempt.
            let _ = start_unit(target).await;
        }
    }
}

/// Connects and handshakes, starting the daemon when it is not reachable, then waits
/// up to `DAEMON_START_TIMEOUT` for its socket. Incompatible protocols fail immediately.
async fn connect_or_start(target: &Target) -> Result<(Wire, HelloInfo), ClientError> {
    match handshake(target).await {
        Err(ClientError::Disconnected) => {}
        attached => return attached,
    }
    if !unit_running(&target.unit).await {
        start_unit(target).await?;
    }
    let deadline = Instant::now() + DAEMON_START_TIMEOUT;
    loop {
        tokio::time::sleep(DAEMON_START_POLL).await;
        match handshake(target).await {
            Err(ClientError::Disconnected) if Instant::now() >= deadline => {
                return Err(ClientError::Start(format!(
                    "juto daemon for {} did not open {} within {}s; read its log with `journalctl --user -u {}`",
                    target.cwd.display(),
                    target.socket.display(),
                    DAEMON_START_TIMEOUT.as_secs(),
                    target.unit
                )));
            }
            Err(ClientError::Disconnected) => {}
            attached => return attached,
        }
    }
}

/// Opens the socket and exchanges `Hello`. An outdated compatible build stays
/// connected while its existing work finishes; incompatible protocols fail.
async fn handshake(target: &Target) -> Result<(Wire, HelloInfo), ClientError> {
    let stream = UnixStream::connect(&target.socket)
        .await
        .map_err(|_| ClientError::Disconnected)?;
    let peer = stream
        .peer_cred()
        .map_err(|e| ClientError::Start(e.to_string()))?;
    if peer.uid() != unsafe { libc::geteuid() } {
        return Err(ClientError::Start("daemon belongs to another user".into()));
    }
    let (read, mut write) = stream.into_split();
    let mut lines = BufReader::new(read).lines();
    let hello = encode_frame(&ClientFrame {
        id: HELLO_ID,
        request: Request::Hello {
            client: CLIENT_NAME.to_string(),
            protocol_version: PROTOCOL_VERSION,
        },
    })?;
    write
        .write_all(hello.as_bytes())
        .await
        .map_err(|_| ClientError::Disconnected)?;
    let line = lines
        .next_line()
        .await
        .map_err(|_| ClientError::Disconnected)?
        .ok_or(ClientError::Disconnected)?;
    let info = match serde_json::from_str::<ServerFrame>(&line) {
        Ok(ServerFrame::Response {
            id: HELLO_ID,
            result: Ok(Reply::Hello(info)),
        }) => info,
        Ok(ServerFrame::Response {
            id: HELLO_ID,
            result: Err(err),
        }) if matches!(err.code, ErrorCode::ProtocolMismatch) => {
            return Err(ClientError::Rpc(RpcError::new(
                ErrorCode::ProtocolMismatch,
                format!(
                    "daemon rejected protocol {PROTOCOL_VERSION} ({}); automatic upgrade requires a compatible protocol",
                    err.message
                ),
            )));
        }
        Ok(ServerFrame::Response {
            id: HELLO_ID,
            result:
                Err(RpcError {
                    code: ErrorCode::Busy,
                    ..
                }),
        }) => return Err(ClientError::Disconnected),
        Ok(ServerFrame::Response {
            id: HELLO_ID,
            result: Err(err),
        }) => return Err(ClientError::Rpc(err)),
        Ok(_) => {
            return Err(ClientError::Rpc(RpcError::new(
                ErrorCode::Invalid,
                "daemon did not answer Hello first",
            )));
        }
        Err(err) => {
            return Err(ClientError::Rpc(RpcError::new(
                ErrorCode::Invalid,
                format!("malformed frame from daemon: {err}"),
            )));
        }
    };
    if peer.pid().map(|pid| pid as u32) != Some(info.pid) || info.cwd != target.cwd {
        return Err(ClientError::Start(
            "daemon identity does not match its socket and workspace".into(),
        ));
    }
    if info.protocol_version != PROTOCOL_VERSION {
        return Err(ClientError::Rpc(RpcError::new(
            ErrorCode::ProtocolMismatch,
            format!(
                "daemon speaks protocol {} but this client speaks {PROTOCOL_VERSION}; automatic upgrade requires a compatible protocol",
                info.protocol_version
            ),
        )));
    }
    // Execute the resolved launcher: packaged installs may put a shell wrapper
    // here, while Hello identifies the ELF image behind that wrapper.
    let output = Command::new(&target.binary)
        .arg("--build-id")
        .output()
        .await
        .map_err(|e| {
            ClientError::Start(format!(
                "cannot identify daemon binary {}: {e}",
                target.binary.display()
            ))
        })?;
    let expected = String::from_utf8_lossy(&output.stdout);
    let expected = expected.trim();
    if !output.status.success() || expected.is_empty() {
        return Err(ClientError::Start(format!(
            "cannot identify daemon binary {} with --build-id ({}): {}; reinstall matching Juto binaries",
            target.binary.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let outdated = info.binary_id != expected;
    Ok((
        Wire {
            lines,
            write,
            outdated,
        },
        info,
    ))
}

/// Starts the daemon as a transient unit. Only variable names go on the command line;
/// systemd takes their values from this process's environment.
async fn start_unit(target: &Target) -> Result<(), ClientError> {
    let output = Command::new("systemd-run")
        .args(["--user", "--collect", "--quiet"])
        .arg(format!("--unit={}", target.unit))
        .arg(format!("--working-directory={}", target.cwd.display()))
        .args(forwarded_env_names().map(|name| format!("--setenv={name}")))
        .arg(&target.binary)
        .arg("--workspace")
        .arg(&target.cwd)
        .output()
        .await
        .map_err(|e| ClientError::Start(format!("cannot run systemd-run: {e}")))?;
    if output.status.success() {
        return Ok(());
    }
    Err(ClientError::Start(format!(
        "systemd-run could not start {}: {}",
        target.unit,
        String::from_utf8_lossy(&output.stderr).trim()
    )))
}

/// Whether the unit exists and is not stopped. A unit in any of these states owns
/// its name, so starting it again would fail.
async fn unit_running(unit: &str) -> bool {
    match Command::new("systemctl")
        .args(["--user", "is-active", unit])
        .output()
        .await
    {
        Ok(output) => matches!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "active" | "activating" | "reloading" | "deactivating"
        ),
        Err(_) => false,
    }
}

/// The daemon binary: next to this executable, otherwise on `PATH`.
fn daemon_binary() -> Result<PathBuf, ClientError> {
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(DAEMON_BINARY)))
        .filter(|path| path.is_file());
    sibling
        .or_else(|| {
            std::env::var_os("PATH").and_then(|path_var| {
                std::env::split_paths(&path_var)
                    .map(|dir| dir.join(DAEMON_BINARY))
                    .find(|path| path.is_file())
            })
        })
        .ok_or_else(|| {
            ClientError::Start(format!(
                "{DAEMON_BINARY} is neither next to this executable nor on PATH"
            ))
        })
        .and_then(|path| {
            path.canonicalize()
                .map_err(|e| ClientError::Start(format!("cannot resolve daemon executable: {e}")))
        })
}

/// Names of every variable in this process's environment, except the ones systemd
/// sets for each service itself. The daemon inherits the rest, including provider
/// credentials.
fn forwarded_env_names() -> impl Iterator<Item = String> {
    std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| is_env_name(name) && !is_systemd_owned(name))
}

/// Variables systemd assigns to every service. Forwarding them would override those.
fn is_systemd_owned(name: &str) -> bool {
    matches!(name, "INVOCATION_ID" | "JOURNAL_STREAM" | "NOTIFY_SOCKET")
        || name.starts_with("LISTEN_")
}

/// systemd rejects the whole `systemd-run` invocation on an invalid `--setenv` name.
fn is_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn encode_frame(frame: &ClientFrame) -> Result<String, ClientError> {
    let mut line = serde_json::to_string(frame).map_err(|e| {
        ClientError::Rpc(RpcError::new(
            ErrorCode::Invalid,
            format!("cannot encode request: {e}"),
        ))
    })?;
    line.push('\n');
    Ok(line)
}

fn set_link(inner: &Weak<Inner>, link: Option<Link>) -> bool {
    match inner.upgrade() {
        Some(inner) => {
            let outdated = link.as_ref().map(|link| link.outdated);
            *lock(&inner.link) = link;
            if let Some(outdated) = outdated {
                inner.upgrading.store(outdated, Ordering::Release);
            }
            inner.upgrade_changed.notify_waiters();
            true
        }
        None => false,
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use tokio::net::UnixListener;

    use super::*;

    /// A fake daemon answers `Hello`, reads one request, and closes the connection
    /// without answering. The pending request must fail with `Disconnected`.
    #[tokio::test]
    async fn pending_request_fails_with_disconnected_when_connection_drops() {
        let (dir, listener, target) = scratch("pending");

        let server_dir = dir.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut lines = BufReader::new(read).lines();

            let hello = next_client_frame(&mut lines).await;
            assert!(matches!(hello.request, Request::Hello { .. }));
            send_server_frame(
                &mut write,
                &ServerFrame::Response {
                    id: hello.id,
                    result: Ok(Reply::Hello(HelloInfo {
                        protocol_version: PROTOCOL_VERSION,
                        daemon_version: "test".to_string(),
                        pid: std::process::id(),
                        cwd: server_dir.clone(),
                        data_dir: server_dir.join("data"),
                        desktop_socket: server_dir.join("desktop.sock"),
                        binary_id: "test-build".into(),
                        graceful_upgrade: false,
                    })),
                },
            )
            .await;

            let request = next_client_frame(&mut lines).await;
            assert!(matches!(request.request, Request::ListSessions));
            // Returning drops both halves: the request is pending and gets no answer.
        });

        let client = open(target).await.expect("handshake with the fake daemon");
        let result = client.request(Request::ListSessions).await;
        assert!(
            matches!(result, Err(ClientError::Disconnected)),
            "{result:?}"
        );
        drop(client);
        server.await.unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A daemon from the same build as the client's binary is attached to.
    #[tokio::test]
    async fn a_daemon_from_the_same_build_is_reused() {
        let (dir, listener, target) = scratch("same");
        let hello = hello_info(&dir, "test-build".into());
        let server = tokio::spawn(serve_hello(listener, Ok(Reply::Hello(hello))));

        let client = open(target).await.expect("the same build is accepted");
        drop(client);
        server.await.unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn incompatible_protocol_is_not_opened_for_upgrade() {
        let (dir, listener, target) = scratch("incompatible");
        let rejection = RpcError::new(ErrorCode::ProtocolMismatch, "incompatible protocol");
        let server = tokio::spawn(serve_hello(listener, Err(rejection)));
        let result = open(target).await;
        assert!(matches!(
            result,
            Err(ClientError::Rpc(RpcError {
                code: ErrorCode::ProtocolMismatch,
                ..
            }))
        ));
        server.await.unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Scratch workspace: a bound socket, a stand-in daemon binary, and the target.
    fn scratch(name: &str) -> (PathBuf, UnixListener, Target) {
        let dir = std::env::temp_dir().join(format!(
            "juto-client-{name}-{}",
            uuid::Uuid::now_v7().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let socket = dir.join("daemon.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let binary = dir.join(DAEMON_BINARY);
        std::fs::write(&binary, b"#!/bin/sh\nprintf '%s\\n' test-build\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let target = Target {
            cwd: dir.clone(),
            socket,
            unit: "juto-daemon-test".to_string(),
            binary,
        };
        (dir, listener, target)
    }

    fn hello_info(dir: &Path, binary_id: String) -> HelloInfo {
        HelloInfo {
            protocol_version: PROTOCOL_VERSION,
            daemon_version: "test".to_string(),
            pid: std::process::id(),
            cwd: dir.to_path_buf(),
            data_dir: dir.join("data"),
            desktop_socket: dir.join("desktop.sock"),
            binary_id,
            graceful_upgrade: false,
        }
    }

    /// Serves one connection: answers `Hello` with `result`, then holds the connection
    /// until the client drops it.
    async fn serve_hello(listener: UnixListener, result: Result<Reply, RpcError>) {
        let (stream, _) = listener.accept().await.unwrap();
        let (read, mut write) = stream.into_split();
        let mut lines = BufReader::new(read).lines();
        let hello = next_client_frame(&mut lines).await;
        assert!(matches!(hello.request, Request::Hello { .. }));
        send_server_frame(
            &mut write,
            &ServerFrame::Response {
                id: hello.id,
                result,
            },
        )
        .await;
        let _ = lines.next_line().await;
    }

    async fn next_client_frame(lines: &mut Lines<BufReader<OwnedReadHalf>>) -> ClientFrame {
        let line = lines
            .next_line()
            .await
            .unwrap()
            .expect("client sent a frame");
        serde_json::from_str(&line).unwrap()
    }

    async fn send_server_frame(write: &mut OwnedWriteHalf, frame: &ServerFrame) {
        let line = serde_json::to_string(frame).unwrap() + "\n";
        write.write_all(line.as_bytes()).await.unwrap();
    }
}
