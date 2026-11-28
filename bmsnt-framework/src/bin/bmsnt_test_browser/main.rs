//! TestApp: test browser for engine features (Phase 2 proving ground).
//! Thin shell: the whole browser lives in the framework library
//! (`bmsnt_framework::testing`, osu-framework `Testing/` analogue).

// no console: a visible console forces DWM composition instead of flip
#![windows_subsystem = "windows"]

use anyhow::Result;

fn main() -> Result<()> {
    bmsnt_framework::logging::init_log("bmsnt_test_browser.log");

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
        if let Err(e) = bmsnt_framework::testing::run() {
            bmsnt_framework::elog!("FATAL: {e:?}");
            std::process::exit(1);
        }
    }
    bmsnt_framework::elog!("exit: clean");
    Ok(())
}
