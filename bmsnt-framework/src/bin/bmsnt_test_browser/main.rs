//! TestApp: test browser for engine features (Phase 2 proving ground).
//!
//! Structure (osu-framework style):
//! - `game`   — GameHost: window, gfx stack, frame loop (input -> update ->
//!   draw -> present), swapchain recreation / device-lost recovery;
//! - `screens` — test cases ("screens"), each owning its scene drawing inside
//!   the scissored 16:9 game view;
//! - `ui`     — immediate-mode drawables: hit-test rects, layout, panels and
//!   text drawn by the sprite batcher;
//! - `theme`  — palette + design-space layout constants (1920x1080, uniformly
//!   scaled).

// no console: a visible console forces DWM composition instead of flip
#![windows_subsystem = "windows"]

mod game;
mod screens;
mod theme;
mod ui;

use anyhow::Result;

fn main() -> Result<()> {
    bmsnt_framework::init_log("bmsnt_test_browser.log");

    // panics abort silently (windows subsystem has no console) — log them
    std::panic::set_hook(Box::new(|info| {
        bmsnt_framework::elog!("PANIC: {info}");
    }));

    // high priority: fewer scheduling jitters; failure is logged, not fatal
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS,
        };
        if SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) == 0 {
            bmsnt_framework::elog!("SetPriorityClass(HIGH_PRIORITY_CLASS) failed");
        }
    }
    // log fatal errors: GUI-subsystem has no console, an unlogged error
    // means a silent death with an empty-looking log
    unsafe {
        if let Err(e) = game::run() {
            bmsnt_framework::elog!("FATAL: {e:?}");
            std::process::exit(1);
        }
    }
    bmsnt_framework::elog!("exit: clean");
    Ok(())
}
