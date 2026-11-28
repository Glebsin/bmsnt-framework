//! Test cases ("screens"): each owns the drawing of its scene inside the
//! scissored game view. Adding a test = implementing `TestCase` and
//! registering it in `Game`.

use crate::graphics::sprites::SpriteBatcher;
use crate::input::InputCore;
use crate::text::Vertex;

use crate::graphics::primitives::Rect;

/// One test case. The game view is scissored by the caller; everything drawn
/// here renders only inside it.
pub trait TestCase {
    /// display name in the sidebar list
    fn name(&self) -> &'static str;

    /// sidebar folder label; rows are grouped in first-appearance order
    fn folder(&self) -> &'static str {
        "GRAPHIC TESTS"
    }

    /// Draw one frame of the scene. `view` — game-view rect in design space,
    /// `s` — design->framebuffer scale, `res` — framebuffer size,
    /// `t` — seconds since app start, `input` — per-frame input state.
    /// Sprite instances go to `sprites`; text (NDC vertices, `ui::text`)
    /// goes to `text`. Errors abort the frame before submit.
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        sprites: &mut SpriteBatcher,
        view: Rect,
        s: f32,
        res: (f32, f32),
        t: f32,
        input: &InputCore,
        text: &mut Vec<Vertex>,
    ) -> anyhow::Result<()>;
}

/// stress scene: moving rounded sprites (target 1% low < 1 ms)
pub struct SpriteBatchScreen {
    /// precomputed instance data (all static fields) for the current view;
    /// per frame only x/y of each sprite are recomputed FROM base_pos (never
    /// stored back — the grid base must stay fixed)
    base: Option<(Vec<f32>, Vec<(f32, f32)>, f32, f32)>,
}

/// 1200 moving rounded sprites (40x30 grid + wobble), covering the FULL game area
const STRESS_SPRITES: usize = 1200;
/// 20 f32 per instance (see SpriteBatcher::sprite)
const STRIDE: usize = 20;

impl SpriteBatchScreen {
    pub fn new() -> Self {
        Self { base: None }
    }

    /// Rebuild the static parts of the instance array for a new game view:
    /// grid base positions, sprite size, atlas uv, colors, params.
    fn rebuild_base(&mut self, view: Rect, size: f32, res: (f32, f32)) {
        // atlas rect of the radius-0 rounded sprite (same as batcher's
        // ROUNDED_SPRITES[0])
        const ATLAS: u32 = 512;
        let (ax0, ay0, ax1, ay1) = (16.0f32, 0.0, 80.0, 64.0);
        let uv = [
            ax0 / ATLAS as f32,
            ay0 / ATLAS as f32,
            ax1 / ATLAS as f32,
            ay1 / ATLAS as f32,
        ];

        let cols = 40usize;
        let rows = (STRESS_SPRITES + cols - 1) / cols;
        let mut data = Vec::with_capacity(STRESS_SPRITES * STRIDE);
        let mut base_pos = Vec::with_capacity(STRESS_SPRITES);
        for i in 0..STRESS_SPRITES {
            let fi = i as f32;
            let col = (i % cols) as f32;
            let row = (i / cols) as f32;
            let bx = view.x + 8.0 + (view.w - 24.0) * (col + 0.5) / cols as f32;
            let by = view.y + 8.0 + (view.h - 24.0) * (row + 0.5) / rows as f32;
            base_pos.push((bx, by));
            let color = (
                0.5 + 0.5 * (fi * 0.01).sin(),
                0.5 + 0.5 * (fi * 0.013 + 2.0).sin(),
                0.5 + 0.5 * (fi * 0.017 + 4.0).sin(),
            );
            data.extend_from_slice(&[
                bx, by, size, size, uv[0], uv[1], uv[2], uv[3], color.0, color.1, color.2, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, res.0, res.1, 0.0,
            ]);
        }
        self.base = Some((data, base_pos, view.w, view.h));
    }
}

impl TestCase for SpriteBatchScreen {
    fn name(&self) -> &'static str {
        "Sprites"
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &mut self,
        sprites: &mut SpriteBatcher,
        view: Rect,
        s: f32,
        res: (f32, f32),
        t: f32,
        _input: &InputCore,
        _text: &mut Vec<Vertex>,
    ) -> anyhow::Result<()> {
        let fb = (view.x * s, view.y * s, view.w * s, view.h * s);
        sprites.begin_section(fb.0, fb.1, fb.2, fb.3);

        // pixel snapping: round screen-space coords to whole framebuffer
        // pixels, otherwise fractional origins rasterize 14x14 as 14x15/15x14
        // (cubes "breathe" and stop being square frame to frame)
        let snap = |v: f32| (v * s).round() / s;
        let size = snap(14.0);

        let needs_rebuild = match &self.base {
            Some((_, _, vw, vh)) => *vw != view.w || *vh != view.h,
            None => true,
        };
        if needs_rebuild {
            self.rebuild_base(view, size, res);
        }
        let (data, base_pos) = {
            let b = self.base.as_mut().unwrap();
            (&mut b.0, &b.1)
        };

        // per frame: only the position of each sprite (two sin/cos + snap),
        // always computed from the fixed grid base — never accumulated
        for i in 0..STRESS_SPRITES {
            let fi = i as f32;
            let o = i * STRIDE;
            let x = base_pos[i].0 + 10.0 * (t * 2.0 + fi * 0.13).sin();
            let y = base_pos[i].1 + 10.0 * (t * 1.7 + fi * 0.17).cos();
            data[o] = snap(x);
            data[o + 1] = snap(y);
        }
        // one batched section push instead of 1200 individual ones
        sprites.push_cached(data, [fb.0, fb.1, fb.2, fb.3])?;
        Ok(())
    }
}

/// Input folder test: held keys shown in the center of the game view as
/// "O + A + B" (press order); the label grows out of the center whenever
/// the combination changes.
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
