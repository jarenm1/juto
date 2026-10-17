//! Compositor worker process.
//!
//! The compositor runs in its own process inside the environment slice, so its
//! CPU and memory are charged to the environment cgroup and a compositor failure
//! cannot take the supervisor down with it (`docs/agent-compositor.md` sections 2
//! and 3).
//!
//! The worker owns the sealed frame pool: it maps the pool writable, applies the
//! seals, and hands the descriptor to the supervisor, which maps it read-only.
//! Slot reservations travel back over the control socket so the producer honours
//! a reader in another process.

use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::compositor::{CompositorCommand, CompositorConfig, SnapshotInfo};
use crate::error::DesktopError;
use crate::protocol::{DesktopEvent, InputEvent};
use crate::scm;
use crate::shm::{POOL_SLOT_COUNT, ShmFramePool, ShmReader};

/// Maximum accepted control frame size.
const MAX_FRAME: usize = 8 * 1024 * 1024;

/// Requests the supervisor sends to the worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "camelCase")]
pub enum WorkerRequest {
    Input(InputEvent),
    Configure {
        width: u32,
        height: u32,
        scale: u32,
        fps: u32,
        generation: u64,
    },
    Snapshot,
    AcquireSlot {
        slot: u32,
    },
    ReleaseSlot {
        slot: u32,
    },
    Shutdown,
}

/// Responses the worker sends back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "response", rename_all = "camelCase")]
pub enum WorkerResponse {
    /// Sent once, with the pool descriptor as ancillary data.
    Ready {
        pool_id: String,
        width: u32,
        height: u32,
        stride: u32,
        format: String,
        slot_count: u32,
        slot_size: u32,
        pool_bytes: u64,
    },
    Snapshot {
        frame_id: u64,
        generation: u64,
        timestamp_us: u64,
        width: u32,
        height: u32,
        slot: u32,
        ticks: u64,
        frames_composed: u64,
        frames_dropped: u64,
        compose_micros: u64,
    },
    Event(DesktopEvent),
    Error {
        message: String,
    },
}

/// Worker process arguments.
#[derive(Debug, Clone)]
pub struct WorkerArgs {
    pub env_id: String,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    pub fps: u32,
    pub render_profile: crate::supervisor::RenderProfile,
    pub wayland_socket: PathBuf,
    pub control_socket: PathBuf,
}

/// Runs the compositor worker until the supervisor closes the connection.
pub fn run_worker(args: WorkerArgs) -> Result<(), DesktopError> {
    let stream = UnixStream::connect(&args.control_socket).map_err(|e| DesktopError::Io {
        path: args.control_socket.clone(),
        source: e,
    })?;
    let writer = Arc::new(Mutex::new(stream.try_clone().map_err(|e| {
        DesktopError::Io {
            path: args.control_socket.clone(),
            source: e,
        }
    })?));

    // The worker owns the producer mapping and seals the pool before anyone else
    // can map it.
    let pool = Arc::new(ShmFramePool::create_unsealed(
        format!("{}-primary", args.env_id),
        args.width,
        args.height,
    )?);
    pool.seal()?;

    let (cmd_tx, cmd_rx) = smithay::reexports::calloop::channel::channel::<CompositorCommand>();
    let (event_tx, event_rx) = std_mpsc::channel::<DesktopEvent>();
    let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), String>>(1);

    let config = CompositorConfig {
        env_id: args.env_id.clone(),
        width: args.width,
        height: args.height,
        scale: args.scale,
        fps: args.fps,
        render_profile: args.render_profile,
        socket_path: args.wayland_socket.clone(),
    };
    let pool_for_compositor = pool.clone();
    let compositor = std::thread::Builder::new()
        .name(format!("juto-compositor-{}", args.env_id))
        .spawn(move || {
            if let Err(err) = crate::compositor::run_compositor(
                config,
                pool_for_compositor,
                event_tx,
                cmd_rx,
                &ready_tx,
            ) {
                tracing::error!(error = %err, "compositor worker exited with error");
            }
        })
        .map_err(|e| DesktopError::Compositor(format!("worker thread: {e}")))?;

    match ready_rx.recv_timeout(Duration::from_secs(15)) {
        Ok(Ok(())) => {}
        Ok(Err(message)) => return Err(DesktopError::Compositor(message)),
        Err(_) => {
            return Err(DesktopError::Compositor(
                "compositor worker did not become ready".to_string(),
            ));
        }
    }

    // Announce readiness only once the Wayland socket is listening, so the
    // supervisor may launch clients as soon as it returns.
    let ready = WorkerResponse::Ready {
        pool_id: pool.pool_id().to_string(),
        width: args.width,
        height: args.height,
        stride: args.width * 4,
        format: "argb8888".to_string(),
        slot_count: POOL_SLOT_COUNT as u32,
        slot_size: pool.slot_size() as u32,
        pool_bytes: pool.size() as u64,
    };
    send_response(&writer, &ready, Some(pool.raw_fd()))?;

    // Forward compositor events to the supervisor.
    {
        let writer = writer.clone();
        std::thread::Builder::new()
            .name(format!("juto-worker-events-{}", args.env_id))
            .spawn(move || {
                while let Ok(event) = event_rx.recv() {
                    if send_response(&writer, &WorkerResponse::Event(event), None).is_err() {
                        return;
                    }
                }
            })
            .ok();
    }

    let mut stream = stream;
    while let Some(frame) = read_frame(&mut stream)? {
        let request: WorkerRequest = match serde_json::from_slice(&frame) {
            Ok(request) => request,
            Err(err) => {
                let _ = send_response(
                    &writer,
                    &WorkerResponse::Error {
                        message: format!("invalid request: {err}"),
                    },
                    None,
                );
                continue;
            }
        };
        match request {
            WorkerRequest::Input(event) => {
                let _ = cmd_tx.send(CompositorCommand::Input(event));
            }
            WorkerRequest::Configure {
                width,
                height,
                scale,
                fps,
                generation,
            } => {
                let (reply, rx) = std_mpsc::sync_channel(1);
                let _ = cmd_tx.send(CompositorCommand::Configure {
                    width,
                    height,
                    scale,
                    fps,
                    generation,
                    reply,
                });
                if let Ok(Err(message)) = rx.recv_timeout(Duration::from_secs(10)) {
                    let _ = send_response(&writer, &WorkerResponse::Error { message }, None);
                }
            }
            WorkerRequest::Snapshot => {
                let (reply, rx) = std_mpsc::sync_channel(1);
                let _ = cmd_tx.send(CompositorCommand::Snapshot { reply });
                // Answer off this loop: a composition can wait on a reader to
                // release its slot, and that release is read by this loop.
                let writer = writer.clone();
                std::thread::spawn(move || {
                    let response = match rx.recv_timeout(Duration::from_secs(10)) {
                        Ok(Ok(info)) => snapshot_response(&info),
                        Ok(Err(message)) => WorkerResponse::Error { message },
                        Err(_) => WorkerResponse::Error {
                            message: "snapshot timed out".to_string(),
                        },
                    };
                    let _ = send_response(&writer, &response, None);
                });
            }
            WorkerRequest::AcquireSlot { slot } => {
                let _ = cmd_tx.send(CompositorCommand::AcquireSlot {
                    slot: slot as usize,
                });
            }
            WorkerRequest::ReleaseSlot { slot } => {
                let _ = cmd_tx.send(CompositorCommand::ReleaseSlot {
                    slot: slot as usize,
                });
            }
            WorkerRequest::Shutdown => break,
        }
    }

    let _ = cmd_tx.send(CompositorCommand::Shutdown);
    let _ = compositor.join();
    Ok(())
}

fn snapshot_response(info: &SnapshotInfo) -> WorkerResponse {
    WorkerResponse::Snapshot {
        frame_id: info.frame_id,
        generation: info.generation,
        timestamp_us: info.timestamp_us,
        width: info.width,
        height: info.height,
        slot: info.slot as u32,
        ticks: info.ticks,
        frames_composed: info.frames_composed,
        frames_dropped: info.frames_dropped,
        compose_micros: info.compose_micros,
    }
}

fn send_response(
    writer: &Arc<Mutex<UnixStream>>,
    response: &WorkerResponse,
    fd: Option<std::os::fd::RawFd>,
) -> Result<(), DesktopError> {
    let bytes = serde_json::to_vec(response)
        .map_err(|e| DesktopError::Protocol(format!("encode worker response: {e}")))?;
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(&bytes);
    let stream = writer.lock();
    match fd {
        Some(fd) => scm::send_with_fd(&stream, &frame, fd)
            .map_err(|e| DesktopError::Protocol(format!("write worker response: {e}"))),
        None => {
            let mut stream = &*stream;
            stream
                .write_all(&frame)
                .and_then(|_| stream.flush())
                .map_err(|e| DesktopError::Protocol(format!("write worker response: {e}")))
        }
    }
}

fn read_frame(stream: &mut UnixStream) -> Result<Option<Vec<u8>>, DesktopError> {
    let mut length = [0u8; 4];
    match stream.read_exact(&mut length) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => {
            return Err(DesktopError::Protocol(format!(
                "read worker request length: {err}"
            )));
        }
    }
    let size = u32::from_be_bytes(length) as usize;
    if size > MAX_FRAME {
        return Err(DesktopError::Protocol(format!(
            "worker request of {size} bytes exceeds {MAX_FRAME}"
        )));
    }
    let mut payload = vec![0u8; size];
    stream
        .read_exact(&mut payload)
        .map_err(|e| DesktopError::Protocol(format!("read worker request: {e}")))?;
    Ok(Some(payload))
}

/// Retires a freshly spawned worker (and its systemd scope, when used) if the
/// startup handshake fails before [`WorkerCompositor`] takes the child.
struct WorkerGuard {
    child: Option<Child>,
    unit: Option<String>,
}

impl WorkerGuard {
    /// Transfers ownership of the child to the caller once the handshake has
    /// succeeded.
    fn disarm(&mut self) -> (Child, Option<String>) {
        (
            self.child.take().expect("worker guard armed"),
            self.unit.take(),
        )
    }
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        if let Some(unit) = self.unit.take() {
            let _ = Command::new("systemctl")
                .args(["--user", "stop", &unit])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Supervisor-side handle for a compositor worker process.
pub struct WorkerCompositor {
    socket: Arc<Mutex<UnixStream>>,
    reader: Arc<ShmReader>,
    child: Mutex<Option<Child>>,
    event_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    unit: Option<String>,
    /// Reply slot for the one outstanding request/response exchange.
    pending: Arc<Mutex<Option<std_mpsc::SyncSender<WorkerResponse>>>>,
}

impl WorkerCompositor {
    /// Spawns the worker, optionally inside a systemd scope, and waits for the
    /// pool descriptor.
    pub fn spawn(
        config: CompositorConfig,
        events: std_mpsc::Sender<DesktopEvent>,
        scope: Option<(&str, &str)>,
    ) -> Result<Self, DesktopError> {
        let control_socket = config
            .socket_path
            .parent()
            .ok_or_else(|| DesktopError::Compositor("socket path has no parent".to_string()))?
            .join("compositor.sock");
        let _ = std::fs::remove_file(&control_socket);
        let listener = std::os::unix::net::UnixListener::bind(&control_socket).map_err(|e| {
            DesktopError::Compositor(format!("cannot bind the worker control socket: {e}"))
        })?;

        let exe = resolve_worker_executable()?;
        let worker_args = [
            "worker".to_string(),
            "--env".to_string(),
            config.env_id.clone(),
            "--width".to_string(),
            config.width.to_string(),
            "--height".to_string(),
            config.height.to_string(),
            "--scale".to_string(),
            config.scale.to_string(),
            "--fps".to_string(),
            config.fps.to_string(),
            "--render-profile".to_string(),
            config.render_profile.as_str().to_string(),
            "--wayland-socket".to_string(),
            config.socket_path.display().to_string(),
            "--control-socket".to_string(),
            control_socket.display().to_string(),
        ];

        let (mut command, unit) = match scope {
            Some((slice, unit)) => {
                let mut command = Command::new("systemd-run");
                command.args(["--user", "--scope", "--quiet", "--collect"]);
                command.arg(format!("--unit={unit}"));
                command.arg(format!("--slice={slice}"));
                command.arg("--");
                command.arg(&exe);
                command.args(&worker_args);
                (command, Some(unit.to_string()))
            }
            None => {
                let mut command = Command::new(&exe);
                command.args(&worker_args);
                (command, None)
            }
        };
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            // Worker tracing writes to stderr; inherit it so warnings reach the daemon's log.
            .stderr(Stdio::inherit());

        let child = command.spawn().map_err(|e| {
            DesktopError::Compositor(format!("cannot spawn the compositor worker: {e}"))
        })?;

        // Every error after the spawn must retire the worker and its scope;
        // the guard owns that until the live compositor takes the child.
        let mut worker_guard = WorkerGuard {
            child: Some(child),
            unit,
        };
        let spawned = accept_ready(&listener);
        let (stream, handshake, fd) = match spawned {
            Ok(parts) => parts,
            Err(err) => {
                let _ = std::fs::remove_file(&control_socket);
                return Err(err);
            }
        };
        let _ = std::fs::remove_file(&control_socket);

        let ready = match handshake {
            WorkerResponse::Ready {
                pool_id,
                width,
                height,
                stride: _,
                format: _,
                slot_count: _,
                slot_size: _,
                pool_bytes,
            } => (pool_id, width, height, pool_bytes),
            WorkerResponse::Error { message } => {
                return Err(DesktopError::Compositor(message));
            }
            other => {
                return Err(DesktopError::Compositor(format!(
                    "unexpected worker handshake: {other:?}"
                )));
            }
        };
        let fd = fd.ok_or_else(|| {
            DesktopError::Compositor("worker handshake carried no pool descriptor".to_string())
        })?;

        let socket = Arc::new(Mutex::new(stream));
        let reservation: Box<dyn Fn(usize, bool) + Send + Sync> = {
            let socket = socket.clone();
            Box::new(move |slot, acquire| {
                let request = if acquire {
                    WorkerRequest::AcquireSlot { slot: slot as u32 }
                } else {
                    WorkerRequest::ReleaseSlot { slot: slot as u32 }
                };
                let _ = send_request(&socket, &request);
            })
        };
        let reader = Arc::new(ShmReader::attach_named(
            fd.as_raw_fd(),
            ready.3 as usize,
            ready.0,
            Some(reservation),
        )?);
        // `ShmReader` owns the descriptor from here on.
        std::mem::forget(fd);

        let pending: Arc<Mutex<Option<std_mpsc::SyncSender<WorkerResponse>>>> =
            Arc::new(Mutex::new(None));
        let event_thread = {
            let socket = socket.clone();
            let pending = pending.clone();
            std::thread::Builder::new()
                .name(format!("juto-worker-events-{}", config.env_id))
                .spawn(move || {
                    let stream = socket.lock().try_clone();
                    let Ok(stream) = stream else {
                        return;
                    };
                    loop {
                        match read_response(&stream) {
                            Ok(WorkerResponse::Event(event)) => {
                                if events.send(event).is_err() {
                                    return;
                                }
                            }
                            Ok(response) => {
                                // A reply to the outstanding request, or an
                                // unsolicited error.
                                let reply = pending.lock().take();
                                match reply {
                                    Some(reply) => {
                                        let _ = reply.send(response);
                                    }
                                    None => {
                                        if let WorkerResponse::Error { message } = response {
                                            tracing::warn!(
                                                error = %message,
                                                "compositor worker error"
                                            );
                                        }
                                    }
                                }
                            }
                            Err(_) => return,
                        }
                    }
                })
                .ok()
        };

        let (child, unit) = worker_guard.disarm();
        Ok(Self {
            socket,
            reader,
            child: Mutex::new(Some(child)),
            event_thread: Mutex::new(event_thread),
            unit,
            pending,
        })
    }

    /// The read-only view of the worker's frame pool.
    pub fn reader(&self) -> Arc<ShmReader> {
        self.reader.clone()
    }

    /// The scope unit the worker runs in, if any.
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Sends a fire-and-forget request.
    pub fn send(&self, request: WorkerRequest) -> Result<(), DesktopError> {
        send_request(&self.socket, &request)
    }

    /// Requests a fresh composition and returns its metadata.
    pub fn snapshot(&self, timeout: Duration) -> Result<SnapshotInfo, DesktopError> {
        let (reply, rx) = std_mpsc::sync_channel(1);
        *self.pending.lock() = Some(reply);
        if let Err(err) = self.send(WorkerRequest::Snapshot) {
            *self.pending.lock() = None;
            return Err(err);
        }
        let response = match rx.recv_timeout(timeout) {
            Ok(response) => response,
            Err(_) => {
                *self.pending.lock() = None;
                return Err(DesktopError::ObservationTimeout {
                    timeout_ms: timeout.as_millis() as u64,
                });
            }
        };
        match response {
            WorkerResponse::Snapshot {
                frame_id,
                generation,
                timestamp_us,
                width,
                height,
                slot,
                ticks,
                frames_composed,
                frames_dropped,
                compose_micros,
            } => Ok(SnapshotInfo {
                frame_id,
                generation,
                timestamp_us,
                width,
                height,
                slot: slot as usize,
                ticks,
                frames_composed,
                frames_dropped,
                compose_micros,
            }),
            WorkerResponse::Error { message } => Err(DesktopError::Compositor(message)),
            other => Err(DesktopError::Compositor(format!(
                "unexpected snapshot response: {other:?}"
            ))),
        }
    }

    /// Stops the worker and joins its threads.
    pub fn shutdown(&self) {
        let _ = self.send(WorkerRequest::Shutdown);
        if let Some(handle) = self.event_thread.lock().take() {
            let _ = handle.join();
        }
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for WorkerCompositor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Whether a compositor worker binary can be located.
pub fn worker_available() -> bool {
    resolve_worker_executable().is_ok()
}

/// Locates the `juto-desktop` binary that hosts the worker subcommand.
///
/// Order: an explicit `JUTO_DESKTOP_WORKER`, the current executable when it is
/// the daemon, a sibling `juto-desktop` next to the current executable (the
/// build tree and an installed profile), then `PATH`.
fn resolve_worker_executable() -> Result<PathBuf, DesktopError> {
    if let Some(path) = std::env::var_os("JUTO_DESKTOP_WORKER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(DesktopError::Compositor(format!(
            "JUTO_DESKTOP_WORKER points at `{}`, which is not a file",
            path.display()
        )));
    }
    let current = std::env::current_exe()
        .map_err(|e| DesktopError::Compositor(format!("current executable: {e}")))?;
    if current
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.contains("juto-desktop"))
        .unwrap_or(false)
    {
        return Ok(current);
    }
    if let Some(dir) = current.parent() {
        let sibling = dir.join("juto-desktop");
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("juto-desktop");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(DesktopError::Compositor(
        "cannot locate the `juto-desktop` worker binary; set JUTO_DESKTOP_WORKER".to_string(),
    ))
}

fn accept_ready(
    listener: &std::os::unix::net::UnixListener,
) -> Result<(UnixStream, WorkerResponse, Option<std::os::fd::OwnedFd>), DesktopError> {
    // The handshake is bounded: environment creation holds the supervisor's
    // admission lock, so a worker that never connects must not block every
    // other environment operation indefinitely.
    listener
        .set_nonblocking(true)
        .map_err(|e| DesktopError::Compositor(format!("worker listener: {e}")))?;
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let (stream, _) = loop {
        match listener.accept() {
            Ok(connection) => break connection,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return Err(DesktopError::Compositor(
                        "worker did not connect within 20s".to_string(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => {
                return Err(DesktopError::Compositor(format!(
                    "worker did not connect: {err}"
                )));
            }
        }
    };
    stream
        .set_nonblocking(false)
        .map_err(|e| DesktopError::Compositor(format!("worker socket: {e}")))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .map_err(|e| DesktopError::Compositor(format!("worker socket: {e}")))?;
    let (payload, fd) = read_response_with_fd(&stream)?;
    // The 20s bound applies to the handshake only. The socket stays open for the
    // life of the environment and is read by the event thread, which must block
    // rather than end after a quiet period.
    stream
        .set_read_timeout(None)
        .map_err(|e| DesktopError::Compositor(format!("worker socket: {e}")))?;
    let response: WorkerResponse = serde_json::from_slice(&payload)
        .map_err(|e| DesktopError::Compositor(format!("worker handshake: {e}")))?;
    Ok((stream, response, fd))
}

fn send_request(
    socket: &Arc<Mutex<UnixStream>>,
    request: &WorkerRequest,
) -> Result<(), DesktopError> {
    let stream = socket.lock();
    send_request_locked(&stream, request)
}

fn send_request_locked(stream: &UnixStream, request: &WorkerRequest) -> Result<(), DesktopError> {
    let bytes = serde_json::to_vec(request)
        .map_err(|e| DesktopError::Protocol(format!("encode worker request: {e}")))?;
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(&bytes);
    let mut stream = stream;
    stream
        .write_all(&frame)
        .and_then(|_| stream.flush())
        .map_err(|e| DesktopError::Protocol(format!("write worker request: {e}")))
}

fn read_response(stream: &UnixStream) -> Result<WorkerResponse, DesktopError> {
    let (payload, _) = read_response_with_fd(stream)?;
    serde_json::from_slice(&payload)
        .map_err(|e| DesktopError::Protocol(format!("decode worker response: {e}")))
}

fn read_response_with_fd(
    stream: &UnixStream,
) -> Result<(Vec<u8>, Option<std::os::fd::OwnedFd>), DesktopError> {
    let mut length = [0u8; 4];
    let mut read = 0;
    let mut fd = None;
    while read < length.len() {
        let (n, received) = scm::recv_with_fd(stream, &mut length[read..])
            .map_err(|e| DesktopError::Protocol(format!("read worker length: {e}")))?;
        if n == 0 {
            return Err(DesktopError::Protocol(
                "compositor worker closed the connection".to_string(),
            ));
        }
        if received.is_some() {
            fd = received;
        }
        read += n;
    }
    let size = u32::from_be_bytes(length) as usize;
    if size > MAX_FRAME {
        return Err(DesktopError::Protocol(format!(
            "worker response of {size} bytes exceeds {MAX_FRAME}"
        )));
    }
    let mut payload = vec![0u8; size];
    let mut read = 0;
    while read < size {
        let (n, received) = scm::recv_with_fd(stream, &mut payload[read..])
            .map_err(|e| DesktopError::Protocol(format!("read worker payload: {e}")))?;
        if n == 0 {
            return Err(DesktopError::Protocol(
                "compositor worker closed the connection".to_string(),
            ));
        }
        if received.is_some() {
            fd = received;
        }
        read += n;
    }
    Ok((payload, fd))
}
