//! Rasterizes one page of archive text into a grayscale PNG.

use anyhow::{Result, ensure};

use super::{
    font::{self, Font, Glyph},
    text::NEWLINE_GLYPH,
};

/// Frame width in pixels. Every shape fits its grid to this width.
pub const FRAME_WIDTH: usize = 1568;
/// Shortest frame height; short pages are padded to it.
const MIN_HEIGHT: usize = 64;
const INK: u8 = 0;
const PAPER: u8 = 255;

/// Cell grid for one image family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub cell_width: usize,
    pub cell_height: usize,
    pub frame_width: usize,
}

impl Shape {
    /// Anthropic family: omp's `11on16` cell.
    pub const ANTHROPIC: Self = Self {
        cell_width: 11,
        cell_height: 16,
        frame_width: FRAME_WIDTH,
    };
    /// Every other family: omp's `8on22` cell.
    pub const DEFAULT: Self = Self {
        cell_width: 8,
        cell_height: 22,
        frame_width: FRAME_WIDTH,
    };

    pub fn for_api(api: &str) -> Self {
        if api.starts_with("anthropic") {
            Self::ANTHROPIC
        } else {
            Self::DEFAULT
        }
    }

    pub fn for_model(api: &str, id: &str) -> Self {
        let mut shape = if id.contains("claude") {
            Self::ANTHROPIC
        } else {
            Self::for_api(api)
        };
        if api.starts_with("google") {
            shape.frame_width = 2048;
        } else if id.contains("opus-4-7")
            || id.contains("opus-4-8")
            || id.contains("mythos")
            || id.contains("fable")
        {
            shape.frame_width = 1932;
        }
        shape
    }

    pub fn cols(self) -> usize {
        self.frame_width / self.cell_width
    }

    pub fn rows(self) -> usize {
        self.frame_width / self.cell_height
    }

    /// Characters one full page holds.
    pub fn capacity(self) -> usize {
        self.cols() * self.rows()
    }
}

/// Renders `cells` (row-major, at most `shape.capacity()`) into a PNG.
/// The height follows the rows in use, with a floor of `MIN_HEIGHT`.
pub fn render_page(cells: &[char], shape: Shape) -> Result<Vec<u8>> {
    ensure!(
        cells.len() <= shape.capacity(),
        "page of {} cells exceeds capacity {}",
        cells.len(),
        shape.capacity()
    );
    let cols = shape.cols();
    let used_rows = cells.len().div_ceil(cols).max(1);
    let height = (used_rows * shape.cell_height).max(MIN_HEIGHT);
    let mut canvas = Canvas {
        pixels: vec![PAPER; shape.frame_width * height],
        width: shape.frame_width,
        height,
    };
    let font = font::bundled();
    for (index, &ch) in cells.iter().enumerate() {
        let x = (index % cols) * shape.cell_width;
        let y = (index / cols) * shape.cell_height;
        if ch == NEWLINE_GLYPH {
            canvas.fill(x, y, shape.cell_width, shape.cell_height);
        } else if let Some(glyph) = font.glyph(ch).or_else(|| font.glyph('?')) {
            canvas.draw(font, glyph, x, y);
        }
    }
    encode_png(&canvas.pixels, canvas.width, canvas.height)
}

struct Canvas {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

impl Canvas {
    fn set(&mut self, x: isize, y: isize) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.pixels[y as usize * self.width + x as usize] = INK;
        }
    }

    fn fill(&mut self, x: usize, y: usize, width: usize, height: usize) {
        for py in y..(y + height).min(self.height) {
            for px in x..(x + width).min(self.width) {
                self.pixels[py * self.width + px] = INK;
            }
        }
    }

    /// Places a glyph so its top sits on the cell top, using the bundled font's ascent.
    fn draw(&mut self, font: &Font, glyph: &Glyph, cell_x: usize, cell_y: usize) {
        let top = font.ascent - (glyph.height + glyph.y_offset);
        for (row, bits) in glyph.rows.iter().enumerate() {
            for col in 0..glyph.width {
                if bits & (0x80 >> col) != 0 {
                    self.set(
                        cell_x as isize + glyph.x_offset as isize + col as isize,
                        cell_y as isize + top as isize + row as isize,
                    );
                }
            }
        }
    }
}

fn encode_png(pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(pixels)?;
        writer.finish()?;
    }
    Ok(out)
}
