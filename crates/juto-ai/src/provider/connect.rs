//! Connect-protocol streaming envelope (shared with upstream Cursor/Devin
//! transports): 1 flag byte + 4-byte big-endian payload length + payload.

use super::ProviderError;
pub(super) const CONNECT_COMPRESSED_FLAG: u8 = 0x01;
/// Payload is the end-of-stream JSON trailer.
pub(super) const CONNECT_END_STREAM_FLAG: u8 = 0x02;

const CONNECT_HEADER_BYTES: usize = 5;

/// Wrap one Connect-protocol message.
pub(super) fn frame_connect_message(data: &[u8], flags: u8) -> Vec<u8> {
    let mut frame = Vec::with_capacity(CONNECT_HEADER_BYTES + data.len());
    frame.push(flags);
    frame.extend_from_slice(&(data.len() as u32).to_be_bytes());
    frame.extend_from_slice(data);
    frame
}

/// One decoded Connect envelope.
pub(super) struct ConnectFrame {
    pub flags: u8,
    pub payload: Vec<u8>,
}

/// Incremental Connect frame decoder with a hard payload cap, mirroring
/// upstream `ConnectFrameDecoder`: a frame spanning several chunks is joined
/// once its announced length has fully arrived.
pub(super) struct ConnectFrameDecoder {
    buf: Vec<u8>,
    max_payload: usize,
}

impl ConnectFrameDecoder {
    pub fn new(max_payload: usize) -> Self {
        Self {
            buf: Vec::new(),
            max_payload,
        }
    }

    /// Buffer `chunk` and return every frame now complete, in wire order.
    /// An oversize length prefix errors after the frames that precede it.
    pub fn decode(&mut self, chunk: &[u8]) -> Result<Vec<ConnectFrame>, ProviderError> {
        self.buf.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while self.buf.len() >= CONNECT_HEADER_BYTES {
            let length = u32::from_be_bytes(self.buf[1..5].try_into().expect("4 bytes")) as usize;
            if length > self.max_payload {
                return Err(ProviderError::Protocol(format!(
                    "Devin Connect frame length {length} exceeds {}-byte cap",
                    self.max_payload
                )));
            }
            if self.buf.len() < CONNECT_HEADER_BYTES + length {
                break;
            }
            let frame: Vec<u8> = self.buf.drain(..CONNECT_HEADER_BYTES + length).collect();
            frames.push(ConnectFrame {
                flags: frame[0],
                payload: frame[CONNECT_HEADER_BYTES..].to_vec(),
            });
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_frames_split_across_chunks() {
        let one = frame_connect_message(b"alpha", 0);
        let two = frame_connect_message(b"beta", CONNECT_END_STREAM_FLAG);
        let mut wire = one.clone();
        wire.extend_from_slice(&two);
        let mut decoder = ConnectFrameDecoder::new(1024);
        let mut frames = Vec::new();
        for chunk in wire.chunks(3) {
            frames.extend(decoder.decode(chunk).unwrap());
        }
        frames.extend(decoder.decode(&[]).unwrap());
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].flags, 0);
        assert_eq!(frames[0].payload, b"alpha");
        assert_eq!(
            frames[1].flags & CONNECT_END_STREAM_FLAG,
            CONNECT_END_STREAM_FLAG
        );
        assert_eq!(frames[1].payload, b"beta");
    }

    #[test]
    fn rejects_oversize_length_prefix() {
        let mut decoder = ConnectFrameDecoder::new(8);
        let frame = frame_connect_message(b"0123456789", 0);
        assert!(decoder.decode(&frame).is_err());
    }
}
