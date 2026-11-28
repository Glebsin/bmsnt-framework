//! Test browser infrastructure:
//! the browser app flow, test scenes, browser UI and the design palette.
//! The binary is a thin shell calling [`run`].

#[path = "design_palette.rs"]
pub mod design_palette;
#[path = "test_browser.rs"]
pub mod test_browser;
#[path = "test_browser_ui.rs"]
pub mod test_browser_ui;
#[path = "test_scenes.rs"]
pub mod test_scenes;

/// Run the test browser to completion. `Err` = fatal (logged by main).
pub unsafe fn run() -> anyhow::Result<()> {
    test_browser::run()
}
