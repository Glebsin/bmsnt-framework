//! Input folder test: on-screen keyboard replicating the
//! ReadableKeyCombinationProvider test scene (osu.Framework.Input):
//! pressed keys highlight, and the readable key combination string is shown
//! below ("pressed:").

use crate::graphics::primitives::Rect;
use crate::graphics::sprites::SpriteBatcher;
use crate::input::InputCore;
use crate::testing::test_scene::TestCase;
use crate::text::Vertex;

use sdl3_sys::everything::*;

/// One physical keycap: scancode + display label + width in key units.
struct Cap {
    sc: SDL_Scancode,
    label: &'static str,
    /// width in units (1 = standard key); wide keys span more
    units: f32,
}

const fn k(sc: SDL_Scancode, label: &'static str) -> Cap {
    Cap { sc, label, units: 1.0 }
}

const fn kw(sc: SDL_Scancode, label: &'static str, units: f32) -> Cap {
    Cap { sc, label, units }
}

/// Keyboard rows, top to bottom (standard ANSI layout; every row sums to
/// 15.0 units so the whole keyboard block is a rectangle, like the osu
/// test scene).
const ROWS: [&[Cap]; 5] = [
    &[
        k(SDL_Scancode::GRAVE, "`"),
        k(SDL_Scancode::_1, "1"),
        k(SDL_Scancode::_2, "2"),
        k(SDL_Scancode::_3, "3"),
        k(SDL_Scancode::_4, "4"),
        k(SDL_Scancode::_5, "5"),
        k(SDL_Scancode::_6, "6"),
        k(SDL_Scancode::_7, "7"),
        k(SDL_Scancode::_8, "8"),
        k(SDL_Scancode::_9, "9"),
        k(SDL_Scancode::_0, "0"),
        k(SDL_Scancode::MINUS, "MINUS"),
        k(SDL_Scancode::EQUALS, "="),
        kw(SDL_Scancode::BACKSPACE, "BACKSP", 2.0),
    ],
    &[
        kw(SDL_Scancode::TAB, "TAB", 1.5),
        k(SDL_Scancode::Q, "Q"),
        k(SDL_Scancode::W, "W"),
        k(SDL_Scancode::E, "E"),
        k(SDL_Scancode::R, "R"),
        k(SDL_Scancode::T, "T"),
        k(SDL_Scancode::Y, "Y"),
        k(SDL_Scancode::U, "U"),
        k(SDL_Scancode::I, "I"),
        k(SDL_Scancode::O, "O"),
        k(SDL_Scancode::P, "P"),
        k(SDL_Scancode::LEFTBRACKET, "["),
        k(SDL_Scancode::RIGHTBRACKET, "]"),
        kw(SDL_Scancode::BACKSLASH, "\\", 1.5),
    ],
    &[
        kw(SDL_Scancode::CAPSLOCK, "CAPS", 1.75),
        k(SDL_Scancode::A, "A"),
        k(SDL_Scancode::S, "S"),
        k(SDL_Scancode::D, "D"),
        k(SDL_Scancode::F, "F"),
        k(SDL_Scancode::G, "G"),
        k(SDL_Scancode::H, "H"),
        k(SDL_Scancode::J, "J"),
        k(SDL_Scancode::K, "K"),
        k(SDL_Scancode::L, "L"),
        k(SDL_Scancode::SEMICOLON, ";"),
        k(SDL_Scancode::APOSTROPHE, "'"),
        kw(SDL_Scancode::RETURN, "ENTER", 2.25),
    ],
    &[
        kw(SDL_Scancode::LSHIFT, "LSHIFT", 2.25),
        k(SDL_Scancode::Z, "Z"),
        k(SDL_Scancode::X, "X"),
        k(SDL_Scancode::C, "C"),
        k(SDL_Scancode::V, "V"),
        k(SDL_Scancode::B, "B"),
        k(SDL_Scancode::N, "N"),
        k(SDL_Scancode::M, "M"),
        k(SDL_Scancode::COMMA, ","),
        k(SDL_Scancode::PERIOD, "."),
        k(SDL_Scancode::SLASH, "/"),
        kw(SDL_Scancode::RSHIFT, "RSHIFT", 2.75),
    ],
    &[
        kw(SDL_Scancode::LCTRL, "LCTRL", 1.25),
        kw(SDL_Scancode::LGUI, "LWIN", 1.25),
        kw(SDL_Scancode::LALT, "LALT", 1.25),
        kw(SDL_Scancode::SPACE, "SPACE", 6.25),
        kw(SDL_Scancode::RALT, "RALT", 1.25),
        kw(SDL_Scancode::RGUI, "RWIN", 1.25),
        kw(SDL_Scancode::APPLICATION, "MENU", 1.25),
        kw(SDL_Scancode::RCTRL, "RCTRL", 1.25),
    ],
];

/// Generic modifier display keys (bottom row): lit when either side is held.
const GENERIC: [(&str, SDL_Scancode, SDL_Scancode); 3] = [
    ("CTRL", SDL_Scancode::LCTRL, SDL_Scancode::RCTRL),
    ("WIN", SDL_Scancode::LGUI, SDL_Scancode::RGUI),
    ("ALT", SDL_Scancode::LALT, SDL_Scancode::RALT),
];

/// keycap size (design px) and pitch; the full block is 15 units wide
const KEY_W: f32 = 74.0;
const KEY_H: f32 = 56.0;
const PITCH: f32 = 80.0;
const ROW_PITCH: f32 = 62.0;
/// label font scale inside a keycap
const LABEL_SCALE: f32 = 2.0;

pub struct KeyCombinationScreen;

impl KeyCombinationScreen {
    pub fn new() -> Self {
        Self
    }

    /// Total keyboard width for a row: sum of unit widths.
    fn row_width(row: &[Cap]) -> f32 {
        row.iter().map(|c| c.units).sum::<f32>() * PITCH - (PITCH - KEY_W)
    }

    /// Readable key name, following ReadableKeyCombinationProvider's table.
    /// NOTE: match arms MUST be qualified constant paths — a bare identifier
    /// parses as a binding pattern and matches every scancode (the
    /// "everything is ESC" bug).
    fn readable_name(sc: SDL_Scancode) -> String {
        let fixed = match sc {
            SDL_Scancode::ESCAPE => "ESC",
            SDL_Scancode::BACKSPACE => "BACKSP",
            SDL_Scancode::CAPSLOCK => "CAPS",
            SDL_Scancode::LCTRL => "LCTRL",
            SDL_Scancode::RCTRL => "RCTRL",
            SDL_Scancode::LSHIFT => "LSHIFT",
            SDL_Scancode::RSHIFT => "RSHIFT",
            SDL_Scancode::LALT => "LALT",
            SDL_Scancode::RALT => "RALT",
            SDL_Scancode::LGUI => "LWIN",
            SDL_Scancode::RGUI => "RWIN",
            SDL_Scancode::APPLICATION => "MENU",
            SDL_Scancode::RETURN => "ENTER",
            SDL_Scancode::MINUS => "MINUS",
            SDL_Scancode::EQUALS => "=",
            SDL_Scancode::LEFTBRACKET => "[",
            SDL_Scancode::RIGHTBRACKET => "]",
            SDL_Scancode::BACKSLASH => "\\",
            SDL_Scancode::SEMICOLON => ";",
            SDL_Scancode::APOSTROPHE => "'",
            SDL_Scancode::COMMA => ",",
            SDL_Scancode::PERIOD => ".",
            SDL_Scancode::SLASH => "/",
            SDL_Scancode::GRAVE => "`",
            SDL_Scancode::KP_0 => "NUMPAD0",
            SDL_Scancode::KP_1 => "NUMPAD1",
            SDL_Scancode::KP_2 => "NUMPAD2",
            SDL_Scancode::KP_3 => "NUMPAD3",
            SDL_Scancode::KP_4 => "NUMPAD4",
            SDL_Scancode::KP_5 => "NUMPAD5",
            SDL_Scancode::KP_6 => "NUMPAD6",
            SDL_Scancode::KP_7 => "NUMPAD7",
            SDL_Scancode::KP_8 => "NUMPAD8",
            SDL_Scancode::KP_9 => "NUMPAD9",
            SDL_Scancode::KP_DIVIDE => "NUMPADDIVIDE",
            SDL_Scancode::KP_MULTIPLY => "NUMPADMULTIPLY",
            SDL_Scancode::KP_MINUS => "NUMPADMINUS",
            SDL_Scancode::KP_PLUS => "NUMPADPLUS",
            SDL_Scancode::KP_ENTER => "NUMPADENTER",
            SDL_Scancode::KP_DECIMAL => "NUMPADDECIMAL",
            _ => "",
        };
        if !fixed.is_empty() {
            return fixed.to_string();
        }
        // fallback: the system keymap name (the provider re-reads it on
        // KeymapChanged; we read it live, which is equivalent here)
        unsafe {
            let p = SDL_GetScancodeName(sc);
            if p.is_null() {
                return format!("KEY {}", sc.0);
            }
            std::ffi::CStr::from_ptr(p)
                .to_string_lossy()
                .to_ascii_uppercase()
        }
    }

    /// The readable combination string for the held keys, in press order,
    /// joined with '-' (GetReadableString analog). Keys with no readable
    /// name are skipped, like the provider's null filter.
    fn readable_string(held: &[(u64, SDL_Scancode)]) -> String {
        held.iter()
            .map(|&(_, sc)| Self::readable_name(sc))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }
}

impl TestCase for KeyCombinationScreen {
    fn name(&self) -> &'static str {
        "KEY COMBINATION"
    }

    fn folder(&self) -> &'static str {
        "INPUT"
    }

    // active immediately on selection; no RUN/STOP actions, no game view —
    // the keyboard covers the whole area right of the panels (as in the
    // osu-framework test scene)
    fn auto_start(&self) -> bool {
        true
    }

    fn full_area(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        sprites: &mut SpriteBatcher,
        view: Rect,
        s: f32,
        res: (f32, f32),
        _t: f32,
        input: &InputCore,
        text: &mut Vec<Vertex>,
    ) -> anyhow::Result<()> {
        let fb = (view.x * s, view.y * s, view.w * s, view.h * s);
        sprites.begin_section(fb.0, fb.1, fb.2, fb.3);

        let held = input.held_keys();
        let is_down = |sc: SDL_Scancode| held.iter().any(|&(_, h)| h == sc);

        let mut y = view.y + 48.0;
        let cx = view.x + view.w / 2.0;

        for row in ROWS {
            let rw = Self::row_width(row);
            let mut x = view.x + (view.w - rw) / 2.0;
            for cap in row {
                let w = KEY_W + (cap.units - 1.0) * PITCH;
                let down = is_down(cap.sc);
                sprites.solid(
                    x,
                    y,
                    w,
                    KEY_H,
                    if down { (0.92, 0.92, 0.92) } else { (0.30, 0.30, 0.30) },
                    res,
                );
                // keycap label: dark on the pressed (bright) cap, dim otherwise
                let lw = cap.label.chars().count() as f32 * 6.0 * LABEL_SCALE;
                text.extend(crate::text::build_text(
                    cap.label,
                    x + (w - lw) / 2.0,
                    y + (KEY_H - 7.0 * LABEL_SCALE) / 2.0,
                    LABEL_SCALE,
                    res.0,
                    res.1,
                    if down { (0.10, 0.10, 0.10) } else { (0.75, 0.75, 0.75) },
                ));
                x += cap.units * PITCH;
            }
            y += ROW_PITCH;
        }

        // generic modifier row: lit when either specific side is held
        let gw = 15.0 * PITCH - (PITCH - KEY_W);
        let mut x = view.x + (view.w - gw) / 2.0;
        for &(label, l, r) in &GENERIC {
            let down = is_down(l) || is_down(r);
            let w = 5.0 * PITCH - (PITCH - KEY_W);
            sprites.solid(
                x,
                y,
                w,
                KEY_H,
                if down { (0.92, 0.92, 0.92) } else { (0.30, 0.30, 0.30) },
                res,
            );
            let lw = label.len() as f32 * 6.0 * LABEL_SCALE;
            text.extend(crate::text::build_text(
                label,
                x + (w - lw) / 2.0,
                y + (KEY_H - 7.0 * LABEL_SCALE) / 2.0,
                LABEL_SCALE,
                res.0,
                res.1,
                if down { (0.10, 0.10, 0.10) } else { (0.75, 0.75, 0.75) },
            ));
            x += 5.0 * PITCH;
        }
        y += ROW_PITCH + 30.0;

        // "pressed:" + the readable combination string, one line (as in the
        // osu test scene)
        let combo = Self::readable_string(&held);
        let line = if combo.is_empty() {
            "PRESSED:".to_string()
        } else {
            format!("PRESSED: {}", combo)
        };
        // fit long combinations into the view
        let scale = ((view.w * 0.92) / (line.chars().count() as f32 * 6.0))
            .min(3.0)
            .max(1.0);
        let lw = line.chars().count() as f32 * 6.0 * scale;
        text.extend(crate::text::build_text(
            &line,
            cx - lw / 2.0,
            y,
            scale,
            res.0,
            res.1,
            (0.92, 0.92, 0.92),
        ));
        Ok(())
    }
}
