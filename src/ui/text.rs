//! Text: 5x7 bitmap font -> triangles. Temporary until the glyph atlas (Phase 2).

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// Upper bound for text vertices (the shared buffer is exactly this size).
pub const MAX_VERTS: usize = 8192;

const FONT: [(&str, [u8; 7]); 22] = [
    ("0", [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
    ("1", [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    ("2", [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
    ("3", [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110]),
    ("4", [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
    ("5", [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
    ("6", [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
    ("7", [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
    ("8", [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
    ("9", [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100]),
    ("F", [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    ("P", [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
    ("S", [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110]),
    (".", [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100]),
    // letters for the flip status: FLIP OK / FLIP LOST DWM
    ("L", [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
    ("I", [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111]),
    ("O", [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    ("K", [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    ("D", [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    ("W", [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010]),
    ("M", [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
    ("T", [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
];

fn glyph(ch: char) -> Option<[u8; 7]> {
    let s = ch.to_string();
    FONT.iter().find(|(k, _)| *k == s).map(|(_, v)| *v)
}

#[inline]
fn ndc(px: f32, py: f32, w: f32, h: f32) -> (f32, f32) {
    (px / w * 2.0 - 1.0, 1.0 - py / h * 2.0)
}

/// Triangulate text into NDC vertices for a w x h viewport, color (r,g,b).
pub fn build_text(
    text: &str,
    x0: f32,
    y0: f32,
    scale: f32,
    w: f32,
    h: f32,
    color: (f32, f32, f32),
) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(text.len() * 6 * 20);
    let (r, g, b) = color;
    let mut cx = x0;

    for ch in text.chars() {
        if ch == ' ' {
            cx += 6.0 * scale;
            continue;
        }
        let Some(rows) = glyph(ch) else {
            cx += 6.0 * scale;
            continue;
        };
        for (ry, row) in rows.iter().enumerate() {
            for rx in 0..5 {
                if row & (1 << (4 - rx)) == 0 {
                    continue;
                }
                let px = cx + rx as f32 * scale;
                let py = y0 + ry as f32 * scale;
                let (x1, y1) = ndc(px, py, w, h);
                let (x2, y2) = ndc(px + scale, py, w, h);
                let (x3, y3) = ndc(px + scale, py + scale, w, h);
                let (x4, y4) = ndc(px, py + scale, w, h);
                out.push(Vertex { x: x1, y: y1, r, g, b });
                out.push(Vertex { x: x2, y: y2, r, g, b });
                out.push(Vertex { x: x3, y: y3, r, g, b });
                out.push(Vertex { x: x1, y: y1, r, g, b });
                out.push(Vertex { x: x3, y: y3, r, g, b });
                out.push(Vertex { x: x4, y: y4, r, g, b });
            }
        }
        cx += 6.0 * scale;
    }
    out
}
