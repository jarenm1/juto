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

use juto_desktop::protocol::{DesktopCommand, DesktopEvent, DesktopResponse};
use juto_desktop::supervisor::DesktopSupervisor;

/// Maximum accepted control frame size (spec section 4).
const MAX_FRAME: usize = 64 * 1024;

fn main() {
    tracing_subscriber_init();

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

    loop {
        let Some(frame) = read_frame(&mut stream)? else {
            return Ok(());
        };
        let command: DesktopCommand = match serde_json::from_slice(&frame) {
            Ok(command) => command,
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

        if let DesktopCommand::Subscribe { env_id } = command {
            write_response(
                &mut stream,
                &DesktopResponse::Subscribed {
                    env_id: env_id.clone(),
                },
            )?;
            return stream_events(stream, supervisor, env_id);
        }

        let (response, fd) = handle(command, &supervisor);
        write_response_with_fd(&mut stream, &response, fd)?;
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
                let event = DesktopEvent::Failed {
                    env_id: env_id.clone(),
                    reason: "ResyncRequired".to_string(),
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
        DesktopCommand::Create { spec, .. } => match supervisor.create_environment(spec) {
            Ok(env) => DesktopResponse::Created {
                env_id: env.env_id().to_string(),
                effective_output: env.output(),
                effective_limits: env.limits(),
            },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Launch {
            env_id,
            app_id,
            argv,
            cwd,
            env,
            ..
        } => match supervisor.get_environment(&env_id) {
            Ok(env_handle) => {
                let overrides: Vec<(String, String)> = env.into_iter().collect();
                match env_handle.launch(&app_id, &argv, Some(&cwd), &overrides) {
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
            Err(err) => error_response(&err),
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
        DesktopCommand::Heartbeat { env_id, .. } => match supervisor.get_environment(&env_id) {
            Ok(_) => DesktopResponse::HeartbeatExtended { expires_at: 0 },
            Err(err) => error_response(&err),
        },
        DesktopCommand::Attach { env_id, .. } => match supervisor.get_environment(&env_id) {
            Ok(env) => {
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
                        viewer_id: format!("local-{}", uuid::Uuid::now_v7()),
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
        DesktopCommand::GeometryReady { env_id, .. } => DesktopResponse::Error {
            code: "TransportUnavailable".to_string(),
            message: format!("geometry acknowledgement for `{env_id}` is not implemented yet"),
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
