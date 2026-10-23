//! Length-prefixed JSON control protocol for the desktop supervisor.
//!
//! [`serve`] drives one control connection against a [`DesktopSupervisor`]. The
//! `juto-desktop` binary and the juto daemon both serve this protocol.

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::Duration;

use crate::protocol::{
    DesktopCommand, DesktopEvent, DesktopRequest, DesktopResponse, PROTOCOL_VERSION,
};
use crate::supervisor::DesktopSupervisor;

/// Maximum accepted control frame size (spec section 4).
const MAX_FRAME: usize = 64 * 1024;

/// Serves one control connection until the peer disconnects.
pub fn serve(mut stream: UnixStream, supervisor: Arc<DesktopSupervisor>) -> std::io::Result<()> {
    // Local attachment is authenticated by peer credentials: only the daemon's
    // own user may drive its environments (spec section 3). The peer principal
    // is the kernel-authenticated identity used wherever a command otherwise
    // accepts a self-asserted one (for example `Retain`'s human owner).
    let peer_principal = match crate::scm::peer_credentials(&stream) {
        Ok(creds) if creds.uid == unsafe { libc::geteuid() } => {
            format!("user:{}", creds.uid)
        }
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
    };

    let mut viewer: Option<ViewerGuard> = None;
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
        // `GeometryReady` unlocks input, so it is bound to the viewer this
        // connection authenticated as at `Attach`; a connection cannot answer
        // for a viewer it does not own.
        if let DesktopCommand::GeometryReady {
            env_id, viewer_id, ..
        } = &command
        {
            let owns_viewer = viewer
                .as_ref()
                .is_some_and(|guard| guard.env_id == *env_id && guard.viewer_id == *viewer_id);
            if !owns_viewer {
                write_response(
                    &mut stream,
                    &DesktopResponse::Error {
                        code: "Unauthorized".to_string(),
                        message: format!("this connection does not own viewer `{viewer_id}`"),
                    },
                )?;
                continue;
            }
        }
        let (response, fd) = handle(command, &supervisor, &peer_principal);
        if let (Some(env_id), DesktopResponse::Attached { viewer_id, .. }) = (attach_env, &response)
        {
            // `Destroy` closes viewers: watch the environment's events and shut
            // this connection down when it stops, so the viewer cannot keep
            // reading a retired pool (spec section 4).
            if let Ok(env) = supervisor.get_environment(&env_id)
                && let Ok(watcher_stream) = stream.try_clone()
            {
                let mut events = env.subscribe();
                let _ = std::thread::Builder::new()
                    .name(format!("juto-desktop-viewer-{viewer_id}"))
                    .spawn(move || {
                        while let Ok(event) = events.blocking_recv() {
                            if matches!(
                                event,
                                DesktopEvent::Stopped { .. } | DesktopEvent::Failed { .. }
                            ) {
                                let _ = watcher_stream.shutdown(std::net::Shutdown::Both);
                                return;
                            }
                        }
                    });
            }
            viewer = Some(ViewerGuard {
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
    peer_principal: &str,
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
                let caller = if is_human {
                    crate::protocol::HUMAN_VIEWER_PRINCIPAL.to_string()
                } else {
                    env.get_status().owner_principal
                };
                match env.set_control(&caller, &action, is_human) {
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
                // Input over this socket is the attached human viewer's. Its lease holder
                // is the viewer principal, so the caller must match that principal.
                let caller = crate::protocol::HUMAN_VIEWER_PRINCIPAL.to_string();
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
            deadline_seconds,
            ..
        } => match supervisor.get_environment(&env_id) {
            // Retention transfers lifetime responsibility to an authenticated
            // human session: the recorded owner is the kernel-authenticated
            // peer, not a self-asserted name in the request (spec section 4).
            Ok(env) => match env.retain(peer_principal, deadline_seconds) {
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
                    env.release_viewer(&viewer_id);
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
                        slot_count: crate::shm::POOL_SLOT_COUNT as u32,
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

fn error_response(err: &crate::DesktopError) -> DesktopResponse {
    DesktopResponse::Error {
        code: error_code(err),
        message: err.to_string(),
    }
}

fn error_code(err: &crate::DesktopError) -> String {
    use crate::DesktopError::*;
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
        ControlLeaseExpired { .. } => "ControlLeaseExpired",
        ObservationTimeout { .. } => "ObservationTimeout",
        TransportUnavailable(_) => "TransportUnavailable",
        EnvironmentFailed(_, _) => "EnvironmentFailed",
        IsolationUnavailable(_) => "IsolationUnavailable",
        RendererUnavailable(_) => "RendererUnavailable",
        GpuBusy => "GpuBusy",
        TextUnavailable(_) => "TextUnavailable",
        Io { .. } => "Io",
        Protocol(_) => "Protocol",
        Rejected { code, .. } => code.as_str(),
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
        Some(fd) => crate::scm::send_with_fd(stream, &frame, fd),
        None => {
            stream.write_all(&frame)?;
            stream.flush()
        }
    }
}
