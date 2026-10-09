//! Incremental wire decoders for provider streams.
//!
//! Two formats are needed by the transports: Server-Sent Events (OpenAI chat
//! completions, OpenAI Responses/Codex, Anthropic, Gemini `alt=sse`) and the
//! newline-delimited JSON used by Ollama `/api/chat`. Both decoders are fed raw
//! response bytes and tolerate arbitrary chunk fragmentation — including UTF-8
//! multi-byte sequences split across reads — because lines are split on the
//! byte level (`0x0A` can never occur inside a multi-byte UTF-8 sequence) and
//! each complete line is decoded with strict UTF-8 validation. CRLF, LF, and
//! bare-CR line endings are accepted per the SSE specification.
//!
//! Behavioral reference: oh-my-pi `packages/utils/src/sse.ts` (SSE framing)
//! and `readJsonl` (NDJSON); see licenses/OMP-MIT.txt.

use crate::types::ProviderError;

/// One complete SSE event: every accumulated `data:` line joined by `\n` (the
/// spec requires each `data:` line to start a new line), plus the optional
/// `event:` name and `id:`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
}

/// Incremental Server-Sent Events decoder.
///
/// Feed raw byte chunks via [`SseDecoder::feed`]; complete events pop out of
/// the returned vector in wire order. Call [`SseDecoder::finish`] at end of
/// stream to dispatch the final data line and event even without a trailing
/// newline. The transport still validates JSON and requires its explicit
/// protocol-terminal event; an incomplete payload cannot become success.
pub struct SseDecoder {
    /// Raw undispatched bytes (a partial line tail).
    pending: Vec<u8>,
    /// `data:` lines accumulated for the event currently being read.
    data_lines: Vec<String>,
    /// `event:` name for the current event.
    event_name: Option<String>,
    /// `id:` for the current event (last wins).
    event_id: Option<String>,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SseDecoder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            data_lines: Vec::new(),
            event_name: None,
            event_id: None,
        }
    }

    /// Feed a chunk of response body bytes; returns all complete events.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ProviderError> {
        self.pending.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut offset = 0;
        while let Some((line_bytes, consumed)) = take_line(&self.pending[offset..]) {
            let line = decode_line(line_bytes)?;
            offset += consumed;
            self.handle_line(&line, &mut events);
        }
        self.pending.drain(..offset);
        Ok(events)
    }

    /// Signal end of stream and dispatch the final unterminated data line.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, ProviderError> {
        let mut events = Vec::new();
        if !self.pending.is_empty() {
            let bytes = std::mem::take(&mut self.pending);
            let line = decode_line(&bytes)?;
            self.handle_line(line.trim_end_matches('\r'), &mut events);
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn handle_line(&mut self, line: &str, events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(events);
            return;
        }
        // Per spec a leading colon is a comment/heartbeat line.
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, rest)) => {
                // A single optional leading space after the colon is stripped.
                let value = rest.strip_prefix(' ').unwrap_or(rest);
                (field, value)
            }
            None => (line, ""),
        };
        match field {
            "data" => self.data_lines.push(value.to_string()),
            "event" => self.event_name = Some(value.to_string()),
            "id" => self.event_id = Some(value.to_string()),
            // `retry:` and unknown fields are ignored.
            _ => {}
        }
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if self.data_lines.is_empty() && self.event_name.is_none() && self.event_id.is_none() {
            return;
        }
        // An event with no data lines carries no payload; still surface it so
        // transports can observe control frames, matching SSE semantics.
        let data = self.data_lines.join("\n");
        events.push(SseEvent {
            event: self.event_name.take(),
            data,
            id: self.event_id.take(),
        });
        self.data_lines.clear();
    }
}

/// Incremental NDJSON decoder: one complete JSON value per line.
///
/// Same fragmentation guarantees as [`SseDecoder`]. Blank lines are skipped.
/// A trailing non-empty line at EOF is emitted by [`NdjsonDecoder::finish`]
/// (Ollama closes the connection after its final chunk without always sending
/// a trailing newline).
#[derive(Default)]
pub struct NdjsonDecoder {
    pending: Vec<u8>,
}

impl NdjsonDecoder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<String>, ProviderError> {
        self.pending.extend_from_slice(chunk);
        let mut lines = Vec::new();
        let mut offset = 0;
        while let Some((line_bytes, consumed)) = take_line(&self.pending[offset..]) {
            let line = decode_line(line_bytes)?;
            offset += consumed;
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }
        self.pending.drain(..offset);
        Ok(lines)
    }

    pub fn finish(&mut self) -> Result<Vec<String>, ProviderError> {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let line = String::from_utf8(pending)
            .map_err(|_| ProviderError::Protocol("invalid UTF-8 in NDJSON stream tail".into()))?;
        let trimmed = line.trim();
        Ok(if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed.to_string()]
        })
    }
}

/// Remove and return the next complete line from `buf`, accepting LF, CRLF, or
/// a bare CR terminator. Returns `(line_without_terminator, bytes_consumed)`.
/// `None` when no terminator is present yet.
fn take_line(buf: &[u8]) -> Option<(&[u8], usize)> {
    for (i, &b) in buf.iter().enumerate() {
        match b {
            b'\n' => {
                let line = if i > 0 && buf[i - 1] == b'\r' {
                    &buf[..i - 1]
                } else {
                    &buf[..i]
                };
                return Some((line, i + 1));
            }
            b'\r' => {
                // CRLF is one terminator; a CR at the buffer tail may be the
                // first half of a CRLF split across chunks — wait for the next
                // byte before deciding. A bare CR terminates the line.
                if i + 1 >= buf.len() {
                    return None;
                }
                if buf[i + 1] == b'\n' {
                    return Some((&buf[..i], i + 2));
                }
                return Some((&buf[..i], i + 1));
            }
            _ => {}
        }
    }
    None
}

fn decode_line(bytes: &[u8]) -> Result<String, ProviderError> {
    std::str::from_utf8(bytes)
        .map(|s| s.to_string())
        .map_err(|_| ProviderError::Protocol("invalid UTF-8 sequence in stream".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(dec: &mut SseDecoder, chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(dec.feed(chunk).expect("feed"));
        }
        out.extend(dec.finish().expect("finish"));
        out
    }

    #[test]
    fn split_utf8_across_chunks() {
        // "é" is 0xC3 0xA9; split between bytes inside a data line.
        let mut dec = SseDecoder::new();
        let chunks: &[&[u8]] = &[b"data: {\"a\":\"", &[0xC3][..], &[0xA9][..], b"\"}\n\n"];
        let events = feed_all(&mut dec, chunks);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"a\":\"é\"}");
    }

    #[test]
    fn crlf_and_bare_cr_and_lf() {
        // CRLF and bare CR both terminate lines; three `data:` lines before
        // the blank separator accumulate into one event.
        let mut dec = SseDecoder::new();
        let events = feed_all(
            &mut dec,
            &[b"data: one\r\ndata: two\rdata: three\n\n" as &[u8]],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "one\ntwo\nthree");
    }

    #[test]
    fn multi_line_data_joined() {
        let mut dec = SseDecoder::new();
        let events = feed_all(
            &mut dec,
            &[b"data: {\ndata: \"a\":1}\ndata:\n\ndata: x\n\n"],
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "{\n\"a\":1}\n");
        assert_eq!(events[1].data, "x");
    }

    #[test]
    fn comment_and_colonless_and_fields() {
        let mut dec = SseDecoder::new();
        let events = feed_all(
            &mut dec,
            &[b":keep-alive\r\nretry: 3000\nevent: ping\ndata: {}\nid: 7\n\n"],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("ping"));
        assert_eq!(events[0].id.as_deref(), Some("7"));
        assert_eq!(events[0].data, "{}");
    }

    #[test]
    fn colonless_data_field() {
        // "data" alone means field `data` with empty value.
        let mut dec = SseDecoder::new();
        let events = feed_all(&mut dec, &[b"data\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn invalid_utf8_is_protocol_error() {
        let mut dec = SseDecoder::new();
        let err = dec.feed(b"data: \xFF\xFE\n\n").unwrap_err();
        assert!(matches!(err, ProviderError::Protocol(_)));
    }

    #[test]
    fn ndjson_fragmented() {
        let mut dec = NdjsonDecoder::new();
        let mut lines = dec.feed(b"{\"done\":false}\r\n{\"done\":tru").unwrap();
        assert_eq!(lines.len(), 1);
        lines.extend(dec.feed(b"e}\n").unwrap());
        lines.extend(dec.finish().unwrap());
        assert_eq!(lines, vec!["{\"done\":false}", "{\"done\":true}"]);
    }

    #[test]
    fn ndjson_tail_without_newline() {
        let mut dec = NdjsonDecoder::new();
        dec.feed(b"{\"a\":1}").unwrap();
        let tail = dec.finish().unwrap();
        assert_eq!(tail, vec!["{\"a\":1}"]);
    }
}
