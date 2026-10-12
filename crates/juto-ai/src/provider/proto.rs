//! Minimal proto3 codec for the Devin (Codeium Cascade) wire protocol.
//!
//! Only the fields the transport consumes are modeled — the same pruning
//! upstream applies in `packages/catalog/src/discovery/devin-proto.ts`
//! ("retains only fields consumed by the client"). Unknown fields are skipped
//! on decode by wire type, so adding upstream fields later is additive.

/// Protobuf writer. `put_*` helpers follow proto3 implicit-presence rules:
/// zero values and empty fields are not emitted.
#[derive(Default)]
pub(super) struct ProtoWriter {
    buf: Vec<u8>,
}

impl ProtoWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    fn key(&mut self, field: u32, wire: u8) {
        self.varint(((field as u64) << 3) | wire as u64);
    }

    fn varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                self.buf.push(byte);
                return;
            }
            self.buf.push(byte | 0x80);
        }
    }

    fn len_delimited(&mut self, field: u32, bytes: &[u8]) {
        self.key(field, 2);
        self.varint(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }

    /// `string` / `bytes` field, omitted when empty.
    pub fn string(&mut self, field: u32, value: &str) {
        if !value.is_empty() {
            self.len_delimited(field, value.as_bytes());
        }
    }

    /// Unsigned varint field (`uint32`/`uint64`/`enum`), omitted when zero.
    pub fn varint_field(&mut self, field: u32, value: u64) {
        if value != 0 {
            self.key(field, 0);
            self.varint(value);
        }
    }

    /// `bool` field, omitted when false.
    pub fn bool(&mut self, field: u32, value: bool) {
        if value {
            self.key(field, 0);
            self.varint(1);
        }
    }

    /// `double` field, omitted when zero.
    pub fn double(&mut self, field: u32, value: f64) {
        if value != 0.0 {
            self.key(field, 1);
            self.buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Embedded message field; emitted only when `Some`.
    pub fn message(&mut self, field: u32, value: Option<&ProtoWriter>) {
        if let Some(value) = value {
            self.len_delimited(field, &value.buf);
        }
    }

    /// Repeated embedded message field.
    pub fn repeated_message<'a>(
        &mut self,
        field: u32,
        values: impl IntoIterator<Item = &'a ProtoWriter>,
    ) {
        for value in values {
            self.len_delimited(field, &value.buf);
        }
    }

    /// Repeated `string` field.
    pub fn repeated_string<'a>(
        &mut self,
        field: u32,
        values: impl IntoIterator<Item = &'a String>,
    ) {
        for value in values {
            self.len_delimited(field, value.as_bytes());
        }
    }
}

/// Field values decoded from a message, keyed by field number. Scalar getters
/// return the last occurrence (proto3 semantics); `nested`/`strings` collect
/// every occurrence for repeated fields.
#[derive(Default)]
pub(super) struct ProtoMessage {
    fields: Vec<(u32, FieldValue)>,
}

#[derive(Clone)]
enum FieldValue {
    Varint(u64),
    /// `fixed64`/`double` fields parse but are never read; skip data.
    Fixed64,
    Bytes(Vec<u8>),
    /// `fixed32`/`float` fields parse but are never read; skip data.
    Fixed32,
}

impl ProtoMessage {
    fn get(&self, field: u32) -> Option<&FieldValue> {
        self.fields
            .iter()
            .rev()
            .find(|(no, _)| *no == field)
            .map(|(_, value)| value)
    }

    fn all(&self, field: u32) -> impl Iterator<Item = &FieldValue> {
        self.fields
            .iter()
            .filter(move |(no, _)| *no == field)
            .map(|(_, value)| value)
    }

    /// `string` field; absent or non-UTF-8 decodes as empty.
    pub fn string(&self, field: u32) -> String {
        match self.get(field) {
            Some(FieldValue::Bytes(bytes)) => String::from_utf8_lossy(bytes).into_owned(),
            _ => String::new(),
        }
    }

    /// Varint scalar (`uint32`/`uint64`/`enum`/`int32`/`int64`/`bool`).
    /// `int32`/`int64` sign handling is irrelevant for values the transport
    /// reads (token counts, stop reasons, credit costs stay non-negative).
    pub fn varint(&self, field: u32) -> u64 {
        match self.get(field) {
            Some(FieldValue::Varint(value)) => *value,
            _ => 0,
        }
    }

    /// Nested message field.
    pub fn message(&self, field: u32) -> Option<ProtoMessage> {
        match self.get(field) {
            Some(FieldValue::Bytes(bytes)) => Self::parse(bytes).ok(),
            _ => None,
        }
    }

    /// All occurrences of a repeated embedded message field; undecodable
    /// elements are skipped rather than failing the frame.
    pub fn repeated_message(&self, field: u32) -> Vec<ProtoMessage> {
        self.all(field)
            .filter_map(|value| match value {
                FieldValue::Bytes(bytes) => Self::parse(bytes).ok(),
                _ => None,
            })
            .collect()
    }

    /// Parse a complete message. Fails on truncated wire data or unknown wire
    /// types so callers can fall back (e.g. gzip) rather than accept garbage.
    pub fn parse(data: &[u8]) -> Result<Self, ProtoError> {
        let mut message = Self::default();
        let mut cursor = Cursor { data, pos: 0 };
        while cursor.pos < data.len() {
            let key = cursor.varint()?;
            let field = (key >> 3) as u32;
            let value = match key & 7 {
                0 => FieldValue::Varint(cursor.varint()?),
                1 => {
                    let _ = cursor.fixed64()?;
                    FieldValue::Fixed64
                }
                2 => {
                    let len = cursor.varint()? as usize;
                    FieldValue::Bytes(cursor.take(len)?.to_vec())
                }
                5 => {
                    let _ = cursor.fixed32()?;
                    FieldValue::Fixed32
                }
                wire => return Err(ProtoError::WireType(wire)),
            };
            message.fields.push((field, value));
        }
        Ok(message)
    }
}

#[derive(Debug)]
pub(super) enum ProtoError {
    /// Truncated input.
    Eof,
    /// Varint exceeded 10 bytes.
    Overflow,
    /// Reserved or unsupported wire type (groups are not used by the API).
    WireType(u64),
}

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eof => write!(f, "truncated protobuf"),
            Self::Overflow => write!(f, "varint overflow"),
            Self::WireType(wire) => write!(f, "unsupported wire type {wire}"),
        }
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn varint(&mut self) -> Result<u64, ProtoError> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *self.data.get(self.pos).ok_or(ProtoError::Eof)?;
            self.pos += 1;
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(ProtoError::Overflow)
    }

    fn fixed64(&mut self) -> Result<u64, ProtoError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    fn fixed32(&mut self) -> Result<u32, ProtoError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ProtoError> {
        let end = self.pos.checked_add(len).ok_or(ProtoError::Eof)?;
        if end > self.data.len() {
            return Err(ProtoError::Eof);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_scalars_and_strings() {
        let mut w = ProtoWriter::new();
        w.string(1, "hello");
        w.string(2, "");
        w.varint_field(3, 300);
        w.varint_field(4, 0);
        w.bool(5, true);
        w.double(6, 0.4);
        w.repeated_string(7, &["a".to_string(), "b".to_string()]);
        let msg = ProtoMessage::parse(&w.finish()).unwrap();
        assert_eq!(msg.string(1), "hello");
        assert_eq!(msg.string(2), "");
        assert_eq!(msg.varint(3), 300);
        assert_eq!(msg.varint(4), 0);
        assert!(msg.varint(5) == 1);
        // `double` (6) and repeated strings (7) have no reader — writers are
        // still exercised above; field presence is covered by the other gets.
    }

    #[test]
    fn roundtrip_nested_and_skips_unknown() {
        let mut inner = ProtoWriter::new();
        inner.string(1, "nested");
        inner.varint_field(21, 7);
        let mut w = ProtoWriter::new();
        w.message(3, Some(&inner));
        w.repeated_message(10, vec![&inner]);
        // Synthesize an unknown fixed32 field (wire 5) before parsing.
        let mut buf = w.finish();
        buf.extend_from_slice(&[0xdd, 0x06, 1, 2, 3, 4]); // field 111, fixed32
        let msg = ProtoMessage::parse(&buf).unwrap();
        let nested = msg.message(3).unwrap();
        assert_eq!(nested.string(1), "nested");
        assert_eq!(nested.varint(21), 7);
        assert_eq!(msg.repeated_message(10).len(), 1);
    }

    #[test]
    fn rejects_truncated_and_bad_wire() {
        assert!(matches!(
            ProtoMessage::parse(&[0x0a, 5, b'a']),
            Err(ProtoError::Eof)
        ));
        assert!(matches!(
            ProtoMessage::parse(&[0x0b]),
            Err(ProtoError::WireType(3))
        ));
        assert!(matches!(
            ProtoMessage::parse(&[0x80; 10]),
            Err(ProtoError::Overflow | ProtoError::Eof)
        ));
    }
}
