//! Test scene infrastructure (the `osu.Framework.Testing.TestScene` analog):
//! the `TestCase` trait every test scene implements.

use crate::graphics::primitives::Rect;
use crate::graphics::sprites::SpriteBatcher;
use crate::input::InputCore;
use crate::text::Vertex;

/// One test case. The game view is scissored by the caller; everything drawn
/// here renders only inside it.
pub trait TestCase {
    /// display name in the sidebar list
    fn name(&self) -> &'static str;

    /// sidebar folder label; rows are grouped in first-appearance order
    fn folder(&self) -> &'static str {
        "GRAPHIC TESTS"
    }

    /// The scene is active immediately on selection (no RUN/STOP actions).
    fn auto_start(&self) -> bool {
        false
    }

    /// The scene draws over the whole area right of the panels (no 16:9
    /// game view, no caption).
    fn full_area(&self) -> bool {
        false
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
