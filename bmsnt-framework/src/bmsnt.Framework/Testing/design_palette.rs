//! Theme: palette and design-space layout constants.
//!
//! The UI is designed at 1920x1080 and uniformly scaled to the real
//! viewport (the osu-framework approach: everything is laid out in a fixed
//! design space, then scaled as one unit).

/// layout is designed at 1920x1080 and scaled to the real viewport
pub const DESIGN_W: f32 = 1920.0;
pub const DESIGN_H: f32 = 1080.0;

pub const TOP_BAR_H: f32 = 52.0;
pub const SIDEBAR_W: f32 = 240.0;
pub const ACTIONS_W: f32 = 220.0;
pub const ROW_H: f32 = 38.0;

/// palette (hex values from the spec)
pub type Color = (f32, f32, f32);

pub const COL_TOP_BAR: Color = (
    0x38 as f32 / 255.0,
    0x38 as f32 / 255.0,
    0x38 as f32 / 255.0,
);
pub const COL_SIDEBAR: Color = (
    0x24 as f32 / 255.0,
    0x24 as f32 / 255.0,
    0x24 as f32 / 255.0,
);
pub const COL_ACTIONS: Color = (
    0x1f as f32 / 255.0,
    0x1f as f32 / 255.0,
    0x1f as f32 / 255.0,
);
/// gradient endpoints around #161616: lighter at the top, darker at the bottom
pub const COL_GRAD_TOP: Color = (
    0x1e as f32 / 255.0,
    0x1e as f32 / 255.0,
    0x1e as f32 / 255.0,
);
pub const COL_GRAD_BOTTOM: Color = (
    0x0e as f32 / 255.0,
    0x0e as f32 / 255.0,
    0x0e as f32 / 255.0,
);

pub const COL_ROW: Color = (
    0x2a as f32 / 255.0,
    0x2a as f32 / 255.0,
    0x2a as f32 / 255.0,
);
pub const COL_ROW_HOVER: Color = (
    0x32 as f32 / 255.0,
    0x32 as f32 / 255.0,
    0x32 as f32 / 255.0,
);
pub const COL_ROW_SEL: Color = (
    0x3a as f32 / 255.0,
    0x3a as f32 / 255.0,
    0x3a as f32 / 255.0,
);
pub const COL_ACCENT: Color = (0.15, 0.75, 0.60);
pub const COL_TEXT: Color = (0.92, 0.92, 0.92);
pub const COL_TEXT_DIM: Color = (0.55, 0.55, 0.55);
pub const COL_GAME_BG: Color = (0.035, 0.035, 0.042);
pub const COL_BORDER: Color = (0.13, 0.13, 0.13);
