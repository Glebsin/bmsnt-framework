//! Test browser infrastructure (the `osu.Framework.Testing` analog):
//! - `test_browser` — the browser app: window, frame loop, screenshots
//!   (TestBrowser + TestBrowserTestRunner analogs);
//! - `test_scene` — the `TestCase` trait (TestScene analog);
//! - `scenes` — concrete test scenes;
//! - `ui` — browser UI drawables and layout (Drawables analog);
//! - `design_palette` — shared design constants.
//! The binary is a thin shell calling [`run`].

#[path = "design_palette.rs"]
pub mod design_palette;
#[path = "scenes/mod.rs"]
pub mod scenes;
#[path = "test_browser.rs"]
pub mod test_browser;
#[path = "test_scene.rs"]
pub mod test_scene;
#[path = "ui/mod.rs"]
pub mod ui;

/// Run the test browser to completion. `Err` = fatal (logged by main).
pub unsafe fn run() -> anyhow::Result<()> {
    test_browser::run()
}
