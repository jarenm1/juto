//! `juto-desktop` daemon: hosts the desktop supervisor and serves the
//! length-prefixed JSON control protocol over a private Unix socket.
//!
//! Usage:
//!   juto-desktop --socket <path>
//!
//! The socket defaults to `$XDG_RUNTIME_DIR/juto-desktop/control.sock`.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use juto_desktop::protocol::{
    DesktopCommand, DesktopEvent, DesktopRequest, DesktopResponse, PROTOCOL_VERSION,
};
use juto_desktop::supervisor::DesktopSupervisor;

/// Maximum accepted control frame size (spec section 4).
const MAX_FRAME: usize = 64 * 1024;

fn main() {
    tracing_subscriber_init();

    // `juto-desktop worker ...` runs the compositor for one environment inside
    // that environment's slice.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("worker") {
        if let Err(err) = run_worker_command(&args[1..]) {
            eprintln!("juto-desktop worker: {err}");
            std::process::exit(1);
        }
        return;
    }

    let socket_path = parse_socket_path();
    if let Some(parent) = socket_path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        eprintln!("juto-desktop: cannot create {}: {err}", parent.display());
        std::process::exit(1);
    }
    let _ = std::fs::remove_file(&socket_path);

    let listener = match UnixListener::bind(&socket_path) {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("juto-desktop: cannot bind {}: {err}", socket_path.display());
            std::process::exit(1);
        }
    };
    eprintln!("juto-desktop: listening on {}", socket_path.display());

    let supervisor = Arc::new(DesktopSupervisor::new());
    let report = supervisor.reconcile();
    if !report.stopped_units.is_empty() || !report.removed_directories.is_empty() {
        eprintln!(
            "juto-desktop: reconciled {} orphan scope(s) and {} stale director(ies)",
            report.stopped_units.len(),
            report.removed_directories.len()
        );
    }
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

    {
        let supervisor = supervisor.clone();
        let shutdown = shutdown.clone();
        let socket_path = socket_path.clone();
        ctrlc_handler(move || {
            if shutdown.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            eprintln!("juto-desktop: shutting down");
            supervisor.destroy_all();
            let _ = std::fs::remove_file(&socket_path);
            std::process::exit(0);
        });
    }

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let supervisor = supervisor.clone();
                std::thread::Builder::new()
                    .name("juto-desktop-conn".to_string())
                    .spawn(move || {
                        if let Err(err) = serve(stream, supervisor) {
                            eprintln!("juto-desktop: connection error: {err}");
                        }
                    })
                    .ok();
            }
            Err(err) => eprintln!("juto-desktop: accept failed: {err}"),
        }
    }
}

fn run_worker_command(args: &[String]) -> Result<(), juto_desktop::DesktopError> {
    let mut env_id = None;
    let mut width = 1280u32;
    let mut height = 720u32;
    let mut scale = 1u32;
    let mut fps = 30u32;
    let mut render_profile = juto_desktop::RenderProfile::Software;
    let mut wayland_socket = None;
    let mut control_socket = None;
    let mut index = 0;
    while index < args.len() {
        let value = args.get(index + 1).cloned().unwrap_or_default();
        match args[index].as_str() {
            "--env" => env_id = Some(value),
            "--width" => width = value.parse().unwrap_or(width),
            "--height" => height = value.parse().unwrap_or(height),
            "--scale" => scale = value.parse().unwrap_or(scale),
            "--fps" => fps = value.parse().unwrap_or(fps),
            "--render-profile" => {
                render_profile = value.parse().unwrap_or(render_profile);
            }
            "--wayland-socket" => wayland_socket = Some(PathBuf::from(value)),
            "--control-socket" => control_socket = Some(PathBuf::from(value)),
            _ => {}
        }
        index += 2;
    }
    let (Some(env_id), Some(wayland_socket), Some(control_socket)) =
        (env_id, wayland_socket, control_socket)
    else {
        return Err(juto_desktop::DesktopError::Protocol(
            "worker requires --env, --wayland-socket, and --control-socket".to_string(),
        ));
    };
    juto_desktop::run_worker(juto_desktop::WorkerArgs {
        env_id,
        width,
        height,
        scale,
        fps,
        render_profile,
        wayland_socket,
        control_socket,
    })
}

fn tracing_subscriber_init() {
    // Keep the daemon quiet by default; RUST_LOG enables tracing output.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();
}

fn parse_socket_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--socket" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        } else if let Some(path) = arg.strip_prefix("--socket=") {
            return PathBuf::from(path);
        }
    }
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("juto-desktop").join("control.sock")
}

fn serve(mut stream: UnixStream, supervisor: Arc<DesktopSupervisor>) -> std::io::Result<()> {
    // Local attachment is authenticated by peer credentials: only the daemon's
    // own user may drive its environments (spec section 3).
    match juto_desktop::scm::peer_credentials(&stream) {
        Ok(creds) if creds.uid == unsafe { libc::geteuid() } => {}
        Ok(creds) => {
            let _ = write_response(
                &mut stream,
                &DesktopResponse::Error {
                    code: "Unauthorized".to_string(),
                    message: format!("peer uid {} is not the daemon user", creds.uid),
                },
            );
            return Ok(());
        }
        Err(err) => {
            eprintln!("juto-desktop: cannot read peer credentials: {err}");
            return Ok(());
        }
    }

    let mut _viewer: Option<ViewerGuard> = None;
    loop {
        let Some(frame) = read_frame(&mut stream)? else {
            return Ok(());
        };
        let request: DesktopRequest = match serde_json::from_slice(&frame) {
            Ok(request) => request,
            Err(err) => {
                write_response(
                    &mut stream,
                    &DesktopResponse::Error {
                        code: "Protocol".to_string(),
                        message: format!("invalid command: {err}"),
                    },
                )?;
                continue;
            }
        };
        if request.version != PROTOCOL_VERSION {
            write_response(
                &mut stream,
                &DesktopResponse::Error {
                    code: "Protocol".to_string(),
                    message: format!(
                        "unsupported protocol version {} (this daemon speaks {PROTOCOL_VERSION})",
                        request.version
                    ),
                },
            )?;
            continue;
        }
        let command = request.command;

        if let DesktopCommand::Subscribe { env_id } = command {
            write_response(
                &mut stream,
                &DesktopResponse::Subscribed {
                    env_id: env_id.clone(),
                },
            )?;
            return stream_events(stream, supervisor, env_id);
        }

        let attach_env = match &command {
            DesktopCommand::Attach { env_id, .. } => Some(env_id.clone()),
            _ => None,
        };
        let (response, fd) = handle(command, &supervisor);
        if let (Some(env_id), DesktopResponse::Attached { viewer_id, .. }) = (attach_env, &response)
        {
            _viewer = Some(ViewerGuard {
                supervisor: supervisor.clone(),
                env_id,
                viewer_id: viewer_id.clone(),
            });
        }
        write_response_with_fd(&mut stream, &response, fd)?;
    }
}

/// Releases an attached viewer's slot when its control connection ends.
struct ViewerGuard {
    supervisor: Arc<DesktopSupervisor>,
    env_id: String,
    viewer_id: String,
}

impl Drop for ViewerGuard {
    fn drop(&mut self) {
        if let Ok(env) = self.supervisor.get_environment(&self.env_id) {
            env.release_viewer(&self.viewer_id);
        }
    }
}

fn stream_events(
    mut stream: UnixStream,
    supervisor: Arc<DesktopSupervisor>,
    env_id: String,
) -> std::io::Result<()> {
    let env = match supervisor.get_environment(&env_id) {
        Ok(env) => env,
        Err(err) => {
            return write_response(
                &mut stream,
                &DesktopResponse::Error {
                    code: "UnknownEnvironment".to_string(),
                    message: err.to_string(),
                },
            );
        }
    };
    let mut receiver = env.subscribe();
    loop {
        match receiver.blocking_recv() {
            Ok(event) => {
                let bytes = serde_json::to_vec(&event).unwrap_or_default();
                write_frame(&mut stream, &bytes)?;
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                // The consumer fell behind the 256-event ring: tell it to
                // resynchronize with `Describe` rather than losing exits.
                let event = DesktopEvent::ResyncRequired {
                    env_id: env_id.clone(),
                };
                let bytes = serde_json::to_vec(&event).unwrap_or_default();
                write_frame(&mut stream, &bytes)?;
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

fn handle(
    command: DesktopCommand,
    supervisor: &DesktopSupervisor,
) -> (DesktopResponse, Option<std::os::fd::RawFd>) {
    let response = match command {
        DesktopCommand::Create { spec, request_id } => {
            match supervisor.create_environment(spec, &request_id) {
                Ok(env) => DesktopResponse::Created {
                    env_id: env.env_id().to_string(),
                    effective_output: env.output(),
                    effective_limits: env.limits(),
                },
                Err(err) => error_response(&err),
            }
        }
        DesktopCommand::Launch {
            env_id,
            app_id,
            argv,
            cwd,
            env,
            request_id,
            ..
        } => match supervisor.get_environment(&env_id) {
            Ok(env_handle) => {
                let overrides: Vec<(String, String)> = env.into_iter().collect();
                match env_handle.launch(&app_id, &argv, Some(&cwd), &overrides, &request_id) {
                    Ok(handle) => DesktopResponse::Launched {
                        app_id: handle.app_id,
                        pid: handle.pid,
                    },
                    Err(err) => error_response(&err),
                }
            }
            Err(err) => error_response(&err),
        },
        DesktopCommand::Terminate { env_id, app_id } => match supervisor.get_environment(&env_id) {
            Ok(env) => match env.terminate(&app_id) {
                Ok(()) => DesktopResponse::Terminated { app_id },
                Err(err) => error_response(&err),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Describe { env_id, .. } => match supervisor.get_environment(&env_id) {
            Ok(env) => DesktopResponse::Description(Box::new(env.get_status())),
            // A destroyed environment's terminal state stays queryable for a
            // bounded retention period (spec section 4).
            Err(err) => match supervisor.retired_environment(&env_id) {
                Some(status) => DesktopResponse::Description(Box::new(status)),
                None => error_response(&err),
            },
        },
        DesktopCommand::ConfigureOutput {
            env_id,
            expected_generation,
            mode,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => match env.configure_output(expected_generation, mode) {
                Ok(generation) => DesktopResponse::OutputConfigured {
                    generation,
                    effective_output: env.output(),
                },
                Err(err) => error_response(&err),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Observe {
            env_id, timeout_ms, ..
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => match env.observe(Duration::from_millis(timeout_ms.max(1))) {
                Ok(info) => DesktopResponse::Observation {
                    frame_id: info.frame_id,
                    generation: info.generation,
                    timestamp_us: info.timestamp_us,
                    width: info.width,
                    height: info.height,
                    shm_pool_id: env.frame_pool().pool_id().to_string(),
                    shm_slot: info.slot as u32,
                    ticks: info.ticks,
                    frames_composed: info.frames_composed,
                    frames_dropped: info.frames_dropped,
                    compose_micros: info.compose_micros,
                },
                Err(err) => error_response(&err),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Grant {
            env_id,
            principal,
            roles,
            audience,
            expiry_seconds,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => DesktopResponse::GrantIssued(env.issue_grant(
                principal,
                roles,
                audience,
                expiry_seconds,
            )),
            Err(err) => error_response(&err),
        },
        DesktopCommand::Revoke { env_id, grant_id } => match supervisor.get_environment(&env_id) {
            Ok(env) => {
                if env.revoke_grant(&grant_id) {
                    DesktopResponse::Revoked
                } else {
                    DesktopResponse::Error {
                        code: "Unauthorized".to_string(),
                        message: format!("unknown grant `{grant_id}`"),
                    }
                }
            }
            Err(err) => error_response(&err),
        },
        DesktopCommand::Control {
            env_id,
            action,
            is_human,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => {
                match env.set_control(&env.get_status().owner_principal, &action, is_human) {
                    Ok(lease) => DesktopResponse::ControlChanged { lease },
                    Err(err) => error_response(&err),
                }
            }
            Err(err) => error_response(&err),
        },
        DesktopCommand::Input {
            env_id,
            epoch,
            generation,
            events,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => {
                let caller = env.get_status().owner_principal;
                match env.dispatch_input(&caller, epoch, generation, events) {
                    Ok(watermark) => DesktopResponse::InputAccepted {
                        applied_watermark: watermark,
                    },
                    Err(err) => error_response(&err),
                }
            }
            Err(err) => error_response(&err),
        },
        DesktopCommand::AutomationQuiesced {
            env_id,
            old_epoch,
            barrier_id,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => {
                env.acknowledge_automation_quiescence(old_epoch, barrier_id);
                DesktopResponse::ControlChanged { lease: None }
            }
            Err(err) => error_response(&err),
        },
        DesktopCommand::Destroy { env_id } => match supervisor.destroy_environment(&env_id) {
            Ok(()) => DesktopResponse::Destroyed,
            Err(err) => error_response(&err),
        },
        DesktopCommand::Heartbeat {
            env_id,
            target,
            epoch,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => match env.heartbeat(&target, epoch) {
                Ok(expires_at) => DesktopResponse::HeartbeatExtended { expires_at },
                Err(err) => error_response(&err),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Retain {
            env_id,
            human_owner,
            deadline_seconds,
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => match env.retain(&human_owner, deadline_seconds) {
                Ok(retention) => DesktopResponse::Retained {
                    human_owner: retention.human_owner,
                    expires_at: retention.expires_at,
                },
                Err(err) => error_response(&err),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Attach { env_id, .. } => match supervisor.get_environment(&env_id) {
            Ok(env) => {
                let viewer_id = format!("local-{}", uuid::Uuid::now_v7());
                if let Err(err) = env.admit_viewer(&viewer_id) {
                    return (error_response(&err), None);
                }
                let pool = env.frame_pool();
                let output = env.output();
                let fd = unsafe { libc::dup(pool.raw_fd()) };
                if fd < 0 {
                    return (
                        DesktopResponse::Error {
                            code: "Shm".to_string(),
                            message: "cannot duplicate the frame pool descriptor".to_string(),
                        },
                        None,
                    );
                }
                return (
                    DesktopResponse::Attached {
                        viewer_id,
                        negotiated_transport: "shm".to_string(),
                        pool_id: pool.pool_id().to_string(),
                        width: output.width,
                        height: output.height,
                        stride: output.width * 4,
                        format: "argb8888".to_string(),
                        slot_count: juto_desktop::shm::POOL_SLOT_COUNT as u32,
                        slot_size: pool.slot_size() as u32,
                        pool_bytes: pool.size() as u64,
                    },
                    Some(fd),
                );
            }
            Err(err) => error_response(&err),
        },
        DesktopCommand::GeometryReady {
            env_id,
            viewer_id,
            generation,
            ..
        } => match supervisor.get_environment(&env_id) {
            Ok(env) => match env.acknowledge_geometry(&viewer_id, generation) {
                Ok(()) => DesktopResponse::GeometryAcknowledged {
                    viewer_id,
                    generation,
                },
                Err(err) => error_response(&err),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Subscribe { .. } => DesktopResponse::Error {
            code: "Protocol".to_string(),
            message: "Subscribe is handled before dispatch".to_string(),
        },
    };
    (response, None)
}

fn error_response(err: &juto_desktop::DesktopError) -> DesktopResponse {
    DesktopResponse::Error {
        code: error_code(err),
        message: err.to_string(),
    }
}

fn error_code(err: &juto_desktop::DesktopError) -> String {
    use juto_desktop::DesktopError::*;
    match err {
        EnvironmentNotFound(_) => "UnknownEnvironment",
        Unauthorized { .. } => "Unauthorized",
        InvalidOutputMode { .. } => "InvalidMode",
        LimitExceeded(_) => "LimitExceeded",
        ExecutableUnavailable { .. } => "ExecutableUnavailable",
        ControlBusy { .. } => "ControlBusy",
        AutomationBusy { .. } => "AutomationBusy",
        StaleGeneration { .. } => "StaleGeneration",
        StaleLease { .. } => "StaleLease",
        ObservationTimeout { .. } => "ObservationTimeout",
        TransportUnavailable(_) => "TransportUnavailable",
        EnvironmentFailed(_, _) => "EnvironmentFailed",
        IsolationUnavailable(_) => "IsolationUnavailable",
        RendererUnavailable(_) => "RendererUnavailable",
        GpuBusy => "GpuBusy",
        TextUnavailable(_) => "TextUnavailable",
        Io { .. } => "Io",
        Protocol(_) => "Protocol",
        Shm(_) => "Shm",
        Compositor(_) => "Compositor",
    }
    .to_string()
}

fn read_frame(stream: &mut UnixStream) -> std::io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match stream.read_exact(&mut len) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let size = u32::from_be_bytes(len) as usize;
    if size > MAX_FRAME {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!("control frame of {size} bytes exceeds {MAX_FRAME}"),
        ));
    }
    let mut buf = vec![0u8; size];
    stream.read_exact(&mut buf)?;
    Ok(Some(buf))
}

fn write_frame(stream: &mut UnixStream, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

fn write_response(stream: &mut UnixStream, response: &DesktopResponse) -> std::io::Result<()> {
    write_response_with_fd(stream, response, None)
}

fn write_response_with_fd(
    stream: &mut UnixStream,
    response: &DesktopResponse,
    fd: Option<std::os::fd::RawFd>,
) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(response).unwrap_or_default();
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(&bytes);
    match fd {
        Some(fd) => juto_desktop::scm::send_with_fd(stream, &frame, fd),
        None => {
            stream.write_all(&frame)?;
            stream.flush()
        }
    }
}

/// Minimal SIGINT/SIGTERM handling: the signal handler only writes one byte to
/// a self-pipe (async-signal-safe); a dedicated thread performs the shutdown.
fn ctrlc_handler<F: Fn() + Send + 'static>(handler: F) {
    use std::os::fd::{FromRawFd, OwnedFd};

    let mut fds = [0 as libc::c_int; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        eprintln!("juto-desktop: cannot create signal pipe; Ctrl-C will not clean up");
        return;
    }
    let read_fd = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write_fd = fds[1];

    unsafe extern "C" fn trampoline(sig: libc::c_int) {
        let byte = sig as u8;
        let fd = SIGNAL_PIPE_WRITE.load(std::sync::atomic::Ordering::Relaxed);
        if fd >= 0 {
            unsafe {
                libc::write(fd, &byte as *const u8 as *const libc::c_void, 1);
            }
        }
    }
    unsafe {
        SIGNAL_PIPE_WRITE.store(write_fd, std::sync::atomic::Ordering::Relaxed);
        libc::signal(libc::SIGINT, trampoline as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, trampoline as *const () as libc::sighandler_t);
    }

    std::thread::Builder::new()
        .name("juto-desktop-signal".to_string())
        .spawn(move || {
            use std::io::Read;
            let mut file = std::fs::File::from(read_fd);
            let mut buf = [0u8; 1];
            if file.read_exact(&mut buf).is_ok() {
                handler();
            }
        })
        .ok();
}

/// Write end of the signal self-pipe, set once before the handler is installed.
static SIGNAL_PIPE_WRITE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
