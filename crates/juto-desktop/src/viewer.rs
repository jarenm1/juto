//! Same-host viewer client.
//!
//! Attaches to a running `juto-desktop` daemon over its private control socket,
//! receives the environment's sealed shared-memory pool as an `SCM_RIGHTS`
//! descriptor, and reads composed frames without a video codec
//! (`docs/agent-compositor.md` section 7).
//!
//! The client never receives a writable mapping: the pool is sealed with
//! `F_SEAL_FUTURE_WRITE`, so a viewer cannot alter desktop content.

use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::error::DesktopError;
use crate::protocol::{
    DesktopCommand, DesktopEvent, DesktopRequest, DesktopResponse, Grant, GrantRole, InputBatch,
    InputEvent,
};
use crate::scm;
use crate::shm::{FrameWait, ShmReader, ShmSlotHeader};
use crate::supervisor::EnvironmentStatus;

/// Negotiated attachment metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachInfo {
    pub viewer_id: String,
    pub transport: String,
    pub pool_id: String,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: String,
    pub slot_count: u32,
    pub slot_size: u32,
    pub pool_bytes: u64,
}

/// A local viewer attached to one environment.
pub struct ViewerClient {
    stream: UnixStream,
    reader: ShmReader,
    info: AttachInfo,
    env_id: String,
    epoch: Option<u64>,
    lease_renewed_at: Option<Instant>,
    seq: u64,
}

/// How often a viewer holding control renews its lease. Well under the daemon's
/// lease TTL (6s in the Juto runtime), so one missed renewal does not drop control.
const LEASE_RENEW_INTERVAL: Duration = Duration::from_secs(2);

impl ViewerClient {
    /// Attaches to `env_id` on the daemon listening at `socket`.
    pub fn attach(socket: &Path, env_id: &str) -> Result<Self, DesktopError> {
        let stream = UnixStream::connect(socket).map_err(|e| DesktopError::Io {
            path: socket.to_path_buf(),
            source: e,
        })?;
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| DesktopError::Io {
                path: socket.to_path_buf(),
                source: e,
            })?;

        let command = DesktopCommand::Attach {
            env_id: env_id.to_string(),
            transport: "shm".to_string(),
            capabilities: vec!["shm".to_string()],
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode attach: {e}")))?;
        write_frame(&stream, &payload)?;

        let (frame, fd) = read_frame_with_fd(&stream)?;
        let response: DesktopResponse = serde_json::from_slice(&frame)
            .map_err(|e| DesktopError::Protocol(format!("decode attach response: {e}")))?;

        let info = match response {
            DesktopResponse::Attached {
                viewer_id,
                negotiated_transport,
                pool_id,
                width,
                height,
                stride,
                format,
                slot_count,
                slot_size,
                pool_bytes,
            } => AttachInfo {
                viewer_id,
                transport: negotiated_transport,
                pool_id,
                width,
                height,
                stride,
                format,
                slot_count,
                slot_size,
                pool_bytes,
            },
            DesktopResponse::Error { code, message } => {
                return Err(DesktopError::Rejected { code, message });
            }
            other => {
                return Err(DesktopError::Protocol(format!(
                    "unexpected attach response: {other:?}"
                )));
            }
        };

        let fd = fd.ok_or_else(|| {
            DesktopError::Protocol("attach response carried no descriptor".to_string())
        })?;
        let reader = ShmReader::attach(fd.as_raw_fd(), info.pool_bytes as usize)?;
        // `ShmReader` owns the descriptor from here on.
        std::mem::forget(fd);

        Ok(Self {
            stream,
            reader,
            info,
            env_id: env_id.to_string(),
            epoch: None,
            lease_renewed_at: None,
            seq: 0,
        })
    }

    /// Negotiated attachment metadata.
    pub fn info(&self) -> &AttachInfo {
        &self.info
    }

    /// The most recently written frame, if the pool has been written.
    pub fn latest_frame(&self) -> Result<Option<(ShmSlotHeader, Vec<u8>)>, DesktopError> {
        let Some(frame) = self.reader.latest_frame()? else {
            return Ok(None);
        };
        // The slot stays reserved until its pixels are copied out, so the
        // compositor cannot overwrite them mid-read.
        self.reader.acquire_slot(frame.slot);
        let copied = self
            .reader
            .read_slot(frame.slot, self.info.slot_size as usize)
            .map(|(header, pixels)| (header, pixels.to_vec()));
        self.reader.release_slot(frame.slot);
        Ok(Some(copied?))
    }

    /// Requests a fresh composition and returns the resulting frame.
    pub fn observe(
        &mut self,
        generation: u64,
        timeout: Duration,
    ) -> Result<(ShmSlotHeader, Vec<u8>), DesktopError> {
        let command = DesktopCommand::Observe {
            env_id: self.env_id.clone(),
            generation,
            after_frame: None,
            timeout_ms: timeout.as_millis() as u64,
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode observe: {e}")))?;
        write_frame(&self.stream, &payload)?;
        let response = self.read_response()?;
        match response {
            DesktopResponse::Observation { .. } => {}
            DesktopResponse::Error { code, message } => {
                return Err(DesktopError::Protocol(format!("{code}: {message}")));
            }
            other => {
                return Err(DesktopError::Protocol(format!(
                    "unexpected observe response: {other:?}"
                )));
            }
        }
        self.latest_frame()?
            .ok_or_else(|| DesktopError::Compositor("no frame after observe".to_string()))
    }

    /// Reports that this viewer presented a frame of `generation`, unlocking
    /// input gated by a resize. The daemon only accepts the acknowledgement
    /// over the connection that owns the viewer.
    pub fn geometry_ready(&mut self, generation: u64) -> Result<(), DesktopError> {
        let command = DesktopCommand::GeometryReady {
            env_id: self.env_id.clone(),
            viewer_id: self.info.viewer_id.clone(),
            generation,
            track_or_pool_id: self.info.pool_id.clone(),
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode geometry-ready: {e}")))?;
        write_frame(&self.stream, &payload)?;
        match self.read_response()? {
            DesktopResponse::GeometryAcknowledged { .. } => Ok(()),
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Rejected { code, message })
            }
            other => Err(DesktopError::Protocol(format!(
                "unexpected geometry-ready response: {other:?}"
            ))),
        }
    }

    /// Acquires the human control lease for this viewer.
    pub fn take_control(&mut self) -> Result<u64, DesktopError> {
        let command = DesktopCommand::Control {
            env_id: self.env_id.clone(),
            action: "acquire".to_string(),
            is_human: true,
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode control: {e}")))?;
        write_frame(&self.stream, &payload)?;
        match self.read_response()? {
            DesktopResponse::ControlChanged { lease: Some(lease) } => {
                self.epoch = Some(lease.epoch);
                self.lease_renewed_at = Some(Instant::now());
                Ok(lease.epoch)
            }
            DesktopResponse::ControlChanged { lease: None } => Err(DesktopError::ControlBusy {
                holder: "none".to_string(),
                epoch: 0,
            }),
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Rejected { code, message })
            }
            other => Err(DesktopError::Protocol(format!(
                "unexpected control response: {other:?}"
            ))),
        }
    }

    /// Releases the human control lease held by this viewer.
    pub fn release_control(&mut self) -> Result<(), DesktopError> {
        let command = DesktopCommand::Control {
            env_id: self.env_id.clone(),
            action: "release".to_string(),
            is_human: true,
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode control: {e}")))?;
        write_frame(&self.stream, &payload)?;
        match self.read_response()? {
            DesktopResponse::ControlChanged { .. } => {
                self.epoch = None;
                self.lease_renewed_at = None;
                Ok(())
            }
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Rejected { code, message })
            }
            other => Err(DesktopError::Protocol(format!(
                "unexpected control response: {other:?}"
            ))),
        }
    }

    /// Renews the environment's owner lifetime. The viewer is the human owner
    /// while it is open, so the environment is not retired as abandoned.
    pub fn heartbeat_owner(&mut self) -> Result<(), DesktopError> {
        let command = DesktopCommand::Heartbeat {
            env_id: self.env_id.clone(),
            target: "owner".to_string(),
            epoch: None,
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode heartbeat: {e}")))?;
        write_frame(&self.stream, &payload)?;
        match self.read_response()? {
            DesktopResponse::HeartbeatExtended { .. } => Ok(()),
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Rejected { code, message })
            }
            other => Err(DesktopError::Protocol(format!(
                "unexpected heartbeat response: {other:?}"
            ))),
        }
    }

    /// Renews the human control lease this viewer holds. Acquiring while another
    /// holder has control would take it over, so only the current epoch renews.
    pub fn heartbeat_lease(&mut self) -> Result<(), DesktopError> {
        let epoch = self.epoch.ok_or_else(|| DesktopError::Unauthorized {
            caller: self.info.viewer_id.clone(),
            role: "Control".to_string(),
            environment: self.env_id.clone(),
        })?;
        let env_id = self.env_id.clone();
        let command = DesktopCommand::Heartbeat {
            env_id,
            target: "lease".to_string(),
            epoch: Some(epoch),
        };
        match self.call(command)? {
            DesktopResponse::HeartbeatExtended { .. } => Ok(()),
            other => Err(unexpected("heartbeat", other)),
        }
    }

    /// Renews the held control lease when it is due. The daemon ends a lease that
    /// is not renewed within its TTL, so a holder must call this at least every
    /// `LEASE_RENEW_INTERVAL`. `send_input` calls it too. A failed renewal means the
    /// daemon no longer holds the lease for this viewer; the viewer drops its epoch
    /// and reports `ControlLeaseExpired`.
    pub fn keep_control_alive(&mut self) -> Result<(), DesktopError> {
        if self.epoch.is_none() {
            return Ok(());
        }
        if self
            .lease_renewed_at
            .is_some_and(|at| at.elapsed() < LEASE_RENEW_INTERVAL)
        {
            return Ok(());
        }
        match self.heartbeat_lease() {
            Ok(()) => {
                self.lease_renewed_at = Some(Instant::now());
                Ok(())
            }
            Err(_) => {
                self.epoch = None;
                self.lease_renewed_at = None;
                Err(DesktopError::ControlLeaseExpired {
                    environment: self.env_id.clone(),
                })
            }
        }
    }

    /// Keeps the environment awake for `deadline_seconds` on behalf of
    /// `human_owner`, so the idle policy does not freeze a desktop being watched.
    /// Returns the retention expiry (unix seconds). Unlike main's in-process
    /// `AwakeHold`, the hold is a deadline: it lapses unless renewed.
    pub fn retain(
        &mut self,
        human_owner: &str,
        deadline_seconds: u64,
    ) -> Result<u64, DesktopError> {
        let command = DesktopCommand::Retain {
            env_id: self.env_id.clone(),
            human_owner: human_owner.to_string(),
            deadline_seconds,
        };
        match self.call(command)? {
            DesktopResponse::Retained { expires_at, .. } => Ok(expires_at),
            other => Err(unexpected("retain", other)),
        }
    }

    /// The control epoch this viewer holds, or `None` without control.
    pub fn control_epoch(&self) -> Option<u64> {
        self.epoch
    }

    /// The environment's current status: control lease, readiness, and grants.
    pub fn describe(&mut self) -> Result<EnvironmentStatus, DesktopError> {
        let env_id = self.env_id.clone();
        let command = DesktopCommand::Describe {
            env_id,
            after_event_seq: None,
        };
        match self.call(command)? {
            DesktopResponse::Description(status) => Ok(*status),
            other => Err(unexpected("describe", other)),
        }
    }

    /// Issues a capability grant on this environment.
    pub fn grant(
        &mut self,
        principal: &str,
        roles: Vec<GrantRole>,
        audience: &str,
        expiry_seconds: u64,
    ) -> Result<Grant, DesktopError> {
        let env_id = self.env_id.clone();
        let command = DesktopCommand::Grant {
            env_id,
            principal: principal.to_string(),
            roles,
            audience: audience.to_string(),
            expiry_seconds,
        };
        match self.call(command)? {
            DesktopResponse::GrantIssued(grant) => Ok(grant),
            other => Err(unexpected("grant", other)),
        }
    }

    /// Blocks until a frame newer than publish sequence `after` is written, or
    /// until `timeout` passes.
    pub fn wait_for_frame(&self, after: u32, timeout: Duration) -> Result<FrameWait, DesktopError> {
        self.reader.wait_for_frame(after, timeout)
    }

    /// Sends one command and reads its response, mapping daemon errors to `Err`.
    fn call(&mut self, command: DesktopCommand) -> Result<DesktopResponse, DesktopError> {
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode request: {e}")))?;
        write_frame(&self.stream, &payload)?;
        match self.read_response()? {
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Rejected { code, message })
            }
            response => Ok(response),
        }
    }

    /// Sends input for the held lease.
    pub fn send_input(
        &mut self,
        generation: u64,
        events: Vec<InputEvent>,
    ) -> Result<(), DesktopError> {
        self.keep_control_alive()?;
        let epoch = self.epoch.ok_or_else(|| DesktopError::Unauthorized {
            caller: self.info.viewer_id.clone(),
            role: "Control".to_string(),
            environment: self.env_id.clone(),
        })?;
        self.seq = self.seq.wrapping_add(1);
        let command = DesktopCommand::Input {
            env_id: self.env_id.clone(),
            epoch,
            generation,
            events: InputBatch {
                seq: self.seq,
                client_timestamp_us: 0,
                events,
            },
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode input: {e}")))?;
        write_frame(&self.stream, &payload)?;
        match self.read_response()? {
            DesktopResponse::InputAccepted { .. } => Ok(()),
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Rejected { code, message })
            }
            other => Err(DesktopError::Protocol(format!(
                "unexpected input response: {other:?}"
            ))),
        }
    }

    fn read_response(&mut self) -> Result<DesktopResponse, DesktopError> {
        let frame = read_frame(&self.stream)?;
        serde_json::from_slice(&frame)
            .map_err(|e| DesktopError::Protocol(format!("decode response: {e}")))
    }
}

fn write_frame(stream: &UnixStream, payload: &[u8]) -> Result<(), DesktopError> {
    let mut stream = stream;
    stream
        .write_all(&(payload.len() as u32).to_be_bytes())
        .map_err(|e| DesktopError::Protocol(format!("write length: {e}")))?;
    stream
        .write_all(payload)
        .map_err(|e| DesktopError::Protocol(format!("write payload: {e}")))?;
    stream
        .flush()
        .map_err(|e| DesktopError::Protocol(format!("flush: {e}")))?;
    Ok(())
}

/// Largest control response the viewer accepts.
const CONTROL_FRAME_LIMIT: usize = 64 * 1024;

/// Largest event the viewer accepts. Cursor bitmaps travel as JSON, so the
/// control limit is too small for them.
const EVENT_FRAME_LIMIT: usize = 16 * 1024 * 1024;

fn read_frame(stream: &UnixStream) -> Result<Vec<u8>, DesktopError> {
    read_frame_limited(stream, CONTROL_FRAME_LIMIT)
}

fn read_frame_limited(stream: &UnixStream, limit: usize) -> Result<Vec<u8>, DesktopError> {
    let mut stream = stream;
    let mut length = [0u8; 4];
    stream
        .read_exact(&mut length)
        .map_err(|e| DesktopError::Protocol(format!("read length: {e}")))?;
    let size = u32::from_be_bytes(length) as usize;
    if size > limit {
        return Err(DesktopError::Protocol(format!(
            "frame of {size} bytes exceeds {limit} bytes"
        )));
    }
    let mut payload = vec![0u8; size];
    stream
        .read_exact(&mut payload)
        .map_err(|e| DesktopError::Protocol(format!("read payload: {e}")))?;
    Ok(payload)
}

fn unexpected(what: &str, response: DesktopResponse) -> DesktopError {
    DesktopError::Protocol(format!("unexpected {what} response: {response:?}"))
}

/// A dedicated connection that streams one environment's events. It carries no
/// commands, so it never affects the environment's awake state.
pub struct EventStream {
    stream: UnixStream,
}

impl EventStream {
    /// Subscribes to `env_id` events on the daemon listening at `socket`.
    pub fn subscribe(socket: &Path, env_id: &str) -> Result<Self, DesktopError> {
        let stream = UnixStream::connect(socket).map_err(|e| DesktopError::Io {
            path: socket.to_path_buf(),
            source: e,
        })?;
        let command = DesktopCommand::Subscribe {
            env_id: env_id.to_string(),
        };
        let payload = serde_json::to_vec(&DesktopRequest::new(command))
            .map_err(|e| DesktopError::Protocol(format!("encode subscribe: {e}")))?;
        write_frame(&stream, &payload)?;
        let frame = read_frame(&stream)?;
        let response: DesktopResponse = serde_json::from_slice(&frame)
            .map_err(|e| DesktopError::Protocol(format!("decode subscribe response: {e}")))?;
        match response {
            DesktopResponse::Subscribed { .. } => Ok(Self { stream }),
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Protocol(format!("{code}: {message}")))
            }
            other => Err(unexpected("subscribe", other)),
        }
    }

    /// Blocks for the next event. `ResyncRequired` marks events the daemon dropped.
    pub fn next_event(&self) -> Result<DesktopEvent, DesktopError> {
        let frame = read_frame_limited(&self.stream, EVENT_FRAME_LIMIT)?;
        serde_json::from_slice(&frame)
            .map_err(|e| DesktopError::Protocol(format!("decode event: {e}")))
    }

    /// Ends the stream, unblocking a pending `next_event` on another thread.
    pub fn close(&self) -> Result<(), DesktopError> {
        self.stream
            .shutdown(Shutdown::Both)
            .map_err(|e| DesktopError::Protocol(format!("shutdown event stream: {e}")))
    }
}

fn read_frame_with_fd(
    stream: &UnixStream,
) -> Result<(Vec<u8>, Option<std::os::fd::OwnedFd>), DesktopError> {
    let mut length = [0u8; 4];
    let mut read = 0;
    let mut fd = None;
    // Ancillary data is delivered with the first byte of the message, so the
    // descriptor may arrive while reading the length prefix.
    while read < length.len() {
        let (n, received) = scm::recv_with_fd(stream, &mut length[read..])
            .map_err(|e| DesktopError::Protocol(format!("read length: {e}")))?;
        if n == 0 {
            return Err(DesktopError::Protocol(
                "daemon closed the connection".to_string(),
            ));
        }
        if received.is_some() {
            fd = received;
        }
        read += n;
    }
    let size = u32::from_be_bytes(length) as usize;
    if size > 64 * 1024 {
        return Err(DesktopError::Protocol(format!(
            "control frame of {size} bytes exceeds 64 KiB"
        )));
    }
    let mut payload = vec![0u8; size];
    let mut read = 0;
    while read < size {
        let (n, received) = scm::recv_with_fd(stream, &mut payload[read..])
            .map_err(|e| DesktopError::Protocol(format!("read payload: {e}")))?;
        if n == 0 {
            return Err(DesktopError::Protocol(
                "daemon closed the connection".to_string(),
            ));
        }
        if received.is_some() {
            fd = received;
        }
        read += n;
    }
    Ok((payload, fd))
}
