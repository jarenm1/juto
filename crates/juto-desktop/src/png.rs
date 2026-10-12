//! Minimal PNG encoder for composed desktop snapshots.
//!
//! Snapshots are lossless RGB(A) images, per `docs/agent-compositor.md`
//! section 7: 4:2:0 video blurs small text, so observation does not go through
//! the streaming codec path.

use crate::error::DesktopError;

const SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Encodes 8-bit BGRA pixels (the compositor's `A8R8G8B8` memory order) as a
/// lossless RGBA PNG.
pub fn encode_bgra(width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>, DesktopError> {
    if width == 0 || height == 0 {
        return Err(DesktopError::Protocol(
            "cannot encode a zero-sized image".to_string(),
        ));
    }
    let expected = width as usize * height as usize * 4;
    if pixels.len() < expected {
        return Err(DesktopError::Protocol(format!(
            "pixel buffer of {} bytes is smaller than the {expected} bytes required for {width}x{height}",
            pixels.len()
        )));
    }

    // Raw scanlines: one filter byte (0 = None) followed by RGBA pixels.
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(0);
        let row = &pixels[y * stride..y * stride + stride];
        for px in row.chunks_exact(4) {
            // BGRA -> RGBA
            raw.push(px[2]);
            raw.push(px[1]);
            raw.push(px[0]);
            raw.push(px[3]);
        }
    }

    let mut out = Vec::with_capacity(raw.len() / 2 + 128);
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: truecolor with alpha
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter method: adaptive
    ihdr.push(0); // interlace: none
    write_chunk(&mut out, b"IHDR", &ihdr);

    let compressed = deflate(&raw)?;
    write_chunk(&mut out, b"IDAT", &compressed);
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

fn deflate(data: &[u8]) -> Result<Vec<u8>, DesktopError> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(data)
        .map_err(|e| DesktopError::Protocol(format!("png deflate failed: {e}")))?;
    encoder
        .finish()
        .map_err(|e| DesktopError::Protocol(format!("png deflate failed: {e}")))
}

/// CRC-32 (IEEE 802.3), as required by PNG chunk framing.
struct Crc32 {
    value: u32,
}

impl Crc32 {
    fn new() -> Self {
        Self { value: 0xFFFF_FFFF }
    }

    fn update(&mut self, data: &[u8]) {
        for byte in data {
            let idx = ((self.value ^ u32::from(*byte)) & 0xFF) as usize;
            self.value = CRC_TABLE[idx] ^ (self.value >> 8);
        }
    }

    fn finish(self) -> u32 {
        self.value ^ 0xFFFF_FFFF
    }
}

static CRC_TABLE: [u32; 256] = build_crc_table();

const fn build_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_a_valid_png_header_and_chunks() {
        let pixels = vec![0u8; 4 * 4 * 4];
        let png = encode_bgra(4, 4, &pixels).expect("encode");
        assert_eq!(&png[..8], &SIGNATURE);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 4);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 4);
        assert!(png.ends_with(&[0xAE, 0x42, 0x60, 0x82]), "IEND crc");
    }

    #[test]
    fn rejects_short_buffers() {
        assert!(encode_bgra(4, 4, &[0u8; 8]).is_err());
        assert!(encode_bgra(0, 4, &[]).is_err());
    }

    #[test]
    fn crc32_matches_known_vector() {
        let mut crc = Crc32::new();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xCBF4_3926);
    }
}
