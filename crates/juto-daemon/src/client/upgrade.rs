//! Keep the old connection usable while waiting for a compatible upgrade.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use super::*;

const IDLE_POLL: Duration = Duration::from_millis(500);
const RPC_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) async fn replace(target: &Target, previous: HelloInfo) -> Result<(), ClientError> {
    let (mut wire, hello) = handshake(target).await?;
    if !wire.outdated || hello.pid != previous.pid || hello.binary_id != previous.binary_id {
        return Ok(());
    }
    if hello.graceful_upgrade {
        match call(
            &mut wire,
            Request::BeginUpgrade {
                binary: target.binary.clone(),
            },
        )
        .await?
        {
            Reply::Ok => {}
            _ => {
                return Err(ClientError::Start(
                    "daemon did not accept the upgrade".into(),
                ));
            }
        }
        // The daemon owns the upgrade now, even if this client closes. It execs
        // the replacement only after runs, children, jobs and approvals settle.
        loop {
            match wire.lines.next_line().await {
                Ok(Some(_)) => {}
                _ => return Ok(()),
            }
        }
    }

    // Legacy daemons have no admission fence or graceful signal handler. Never
    // signal an active one. A pidfd pins its identity across the idle checks;
    // another process can never receive the signal through PID reuse.
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, hello.pid, 0) };
    if raw < 0 {
        return Err(ClientError::Start(format!(
            "cannot watch old daemon: {}",
            std::io::Error::last_os_error()
        )));
    }
    let process = unsafe { OwnedFd::from_raw_fd(raw as _) };
    let mut was_idle = false;
    loop {
        let idle = legacy_idle(&mut wire, hello.pid).await?;
        if idle && was_idle {
            let result = unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    process.as_raw_fd(),
                    libc::SIGTERM,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            };
            if result < 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
                return Err(ClientError::Start(format!(
                    "cannot retire idle daemon: {}",
                    std::io::Error::last_os_error()
                )));
            }
            return Ok(());
        }
        was_idle = idle;
        tokio::time::sleep(IDLE_POLL).await;
    }
}

async fn legacy_idle(wire: &mut Wire, pid: u32) -> Result<bool, ClientError> {
    match call(wire, Request::ListSubagents).await? {
        Reply::Subagents(children) if children.iter().all(|child| child.status.is_terminal()) => {}
        Reply::Subagents(_) => return Ok(false),
        _ => {
            return Err(ClientError::Start(
                "daemon did not report its subagents".into(),
            ));
        }
    }
    match call(wire, Request::ListJobs).await? {
        Reply::Jobs(jobs)
            if !jobs.iter().any(|job| {
                job.owner_pid == Some(pid)
                    && matches!(
                        job.status,
                        juto_runtime::jobs::JobStatus::Queued
                            | juto_runtime::jobs::JobStatus::Running
                    )
            }) => {}
        Reply::Jobs(_) => return Ok(false),
        _ => return Err(ClientError::Start("daemon did not report its jobs".into())),
    }
    // Check foreground work last: a child's completion may have woken its main.
    match call(wire, Request::ListSessions).await? {
        Reply::Sessions(sessions) => Ok(sessions.iter().all(|session| session.run.is_none())),
        _ => Err(ClientError::Start(
            "daemon did not report its sessions".into(),
        )),
    }
}

async fn call(wire: &mut Wire, request: Request) -> Result<Reply, ClientError> {
    tokio::time::timeout(RPC_TIMEOUT, async {
        let line = encode_frame(&ClientFrame { id: 1, request })?;
        wire.write
            .write_all(line.as_bytes())
            .await
            .map_err(|_| ClientError::Disconnected)?;
        loop {
            let line = wire
                .lines
                .next_line()
                .await
                .map_err(|_| ClientError::Disconnected)?
                .ok_or(ClientError::Disconnected)?;
            match serde_json::from_str::<ServerFrame>(&line) {
                Ok(ServerFrame::Response { id: 1, result }) => {
                    return result.map_err(ClientError::Rpc);
                }
                Ok(ServerFrame::Event { .. }) => {}
                _ => {
                    return Err(ClientError::Start(
                        "invalid upgrade response from daemon".into(),
                    ));
                }
            }
        }
    })
    .await
    .map_err(|_| ClientError::Start("daemon upgrade status request timed out".into()))?
}
