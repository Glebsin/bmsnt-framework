//! Input folder test: held keys shown in the center of the game view as
//! "O + A + B" (press order); the label grows out of the center whenever
//! the combination changes.

use crate::graphics::primitives::Rect;
use crate::graphics::sprites::SpriteBatcher;
use crate::input::InputCore;
use crate::testing::test_scene::TestCase;
use crate::text::Vertex;

pub struct KeysScreen {
    /// label composed on the previous frame (change detection for the pop)
    shown: String,
    /// t (app seconds) when the label last changed; drives the pop anim
    changed_at: f32,
}

/// text scale of a settled label (design px per font cell)
const KEYS_TEXT_SCALE: f32 = 4.5;
/// pop animation duration (seconds)
const KEYS_POP_SECS: f32 = 0.22;

impl KeysScreen {
    pub fn new() -> Self {
        Self {
            shown: String::new(),
            changed_at: -10.0,
        }
    }

    /// SDL scancode -> display name ("O", "SPACE", ...).
    fn key_name(sc: sdl3_sys::everything::SDL_Scancode) -> String {
        unsafe {
            let p = sdl3_sys::everything::SDL_GetScancodeName(sc);
            if p.is_null() {
                return format!("KEY {}", sc.0);
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl TestCase for KeysScreen {
    fn name(&self) -> &'static str {
        "Keys"
    }

    fn folder(&self) -> &'static str {
        "INPUT"
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        _sprites: &mut SpriteBatcher,
        view: Rect,
        _s: f32,
        res: (f32, f32),
        t: f32,
        input: &InputCore,
        text: &mut Vec<Vertex>,
    ) -> anyhow::Result<()> {
        // held keys in press order -> "O + A + B"
        let held = input.held_keys();
        let label = if held.is_empty() {
            String::new()
        } else {
            held.iter()
                .map(|&(_, sc)| Self::key_name(sc))
                .collect::<Vec<_>>()
                .join(" + ")
        };

        if label != self.shown {
            self.shown = label.clone();
            self.changed_at = t;
        }

        let cx = view.x + view.w / 2.0;
        let cy = view.y + view.h / 2.0;

        if label.is_empty() {
            // idle hint, static and dim
            const HINT: &str = "PRESS ANY KEY";
            let w = HINT.len() as f32 * 6.0 * 2.0;
            text.extend(crate::text::build_text(
                HINT,
                cx - w / 2.0,
                cy - 7.0,
                2.0,
                res.0,
                res.1,
                (0.55, 0.55, 0.55),
            ));
            return Ok(());
        }

        // pop: the label grows out of the center when the combo changes
        let p = ((t - self.changed_at) / KEYS_POP_SECS).clamp(0.0, 1.0);
        let k = 1.0 - (1.0 - p) * (1.0 - p) * (1.0 - p); // ease-out cubic
                                                         // fit long combos into the view
        let fit = (view.w * 0.9) / (label.chars().count() as f32 * 6.0);
        let scale = fit.min(KEYS_TEXT_SCALE) * (0.25 + 0.75 * k);
        let w = label.chars().count() as f32 * 6.0 * scale;
        text.extend(crate::text::build_text(
            &label,
            cx - w / 2.0,
            cy - 3.5 * scale,
            scale,
            res.0,
            res.1,
            (0.92, 0.92, 0.92),
        ));
        Ok(())
    }
}
