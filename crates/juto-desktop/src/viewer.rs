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
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use crate::error::DesktopError;
use crate::protocol::{DesktopCommand, DesktopRequest, DesktopResponse, InputBatch, InputEvent};
use crate::scm;
use crate::shm::{ShmReader, ShmSlotHeader};

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
    seq: u64,
}

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
                return Err(DesktopError::Protocol(format!("{code}: {message}")));
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
        let (header, pixels) = self
            .reader
            .read_slot(frame.slot, self.info.slot_size as usize)?;
        Ok(Some((header, pixels.to_vec())))
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
                Err(DesktopError::Protocol(format!("{code}: {message}")))
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
                Ok(lease.epoch)
            }
            DesktopResponse::ControlChanged { lease: None } => Err(DesktopError::ControlBusy {
                holder: "none".to_string(),
                epoch: 0,
            }),
            DesktopResponse::Error { code, message } => {
                Err(DesktopError::Protocol(format!("{code}: {message}")))
            }
            other => Err(DesktopError::Protocol(format!(
                "unexpected control response: {other:?}"
            ))),
        }
    }

    /// Sends input for the held lease.
    pub fn send_input(
        &mut self,
        generation: u64,
        events: Vec<InputEvent>,
    ) -> Result<(), DesktopError> {
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
                Err(DesktopError::Protocol(format!("{code}: {message}")))
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

fn read_frame(stream: &UnixStream) -> Result<Vec<u8>, DesktopError> {
    let mut stream = stream;
    let mut length = [0u8; 4];
    stream
        .read_exact(&mut length)
        .map_err(|e| DesktopError::Protocol(format!("read length: {e}")))?;
    let size = u32::from_be_bytes(length) as usize;
    if size > 64 * 1024 {
        return Err(DesktopError::Protocol(format!(
            "control frame of {size} bytes exceeds 64 KiB"
        )));
    }
    let mut payload = vec![0u8; size];
    stream
        .read_exact(&mut payload)
        .map_err(|e| DesktopError::Protocol(format!("read payload: {e}")))?;
    Ok(payload)
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
