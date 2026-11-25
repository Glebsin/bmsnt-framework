//! SDL3 window wrapper: creation, fullscreen mode.

use anyhow::{anyhow, Result};
use std::ffi::{c_int, c_void, CStr, CString};
use std::ptr;

use sdl3_sys::everything::*;

pub struct Window {
    handle: *mut SDL_Window,
}

impl Window {
    /// Creates a hidden window, sets an exclusive fullscreen display mode
    /// before showing (no windowed -> fullscreen transition).
    /// Important: NO SDL_WINDOW_RESIZABLE — otherwise Windows drops the
    /// swapchain off exclusive flip into DWM composition (6.6k -> 2.8k fps).
    pub unsafe fn new(title: &str, w: c_int, h: c_int) -> Result<Self> {
        // SDL_INIT_GAMEPAD needs the sdl-joystick feature of sdl3-sys
        // (SDL is built with no-default-subsystems, see Cargo.toml)
        if !SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD) {
            return Err(sdl_err("SDL_Init"));
        }
        let title = CString::new(title)?;
        let handle = SDL_CreateWindow(
            title.as_ptr(),
            w,
            h,
            SDL_WINDOW_FULLSCREEN | SDL_WINDOW_HIDDEN,
        );
        if handle.is_null() {
            return Err(sdl_err("SDL_CreateWindow"));
        }
        let display = SDL_GetPrimaryDisplay();
        let mode = SDL_GetDesktopDisplayMode(display);
        if mode.is_null() {
            return Err(sdl_err("SDL_GetDesktopDisplayMode"));
        }
        if !SDL_SetWindowFullscreenMode(handle, mode) {
            return Err(sdl_err("SDL_SetWindowFullscreenMode(exclusive)"));
        }
        SDL_SyncWindow(handle);
        SDL_ShowWindow(handle);

        let flags = SDL_GetWindowFlags(handle);
        let exclusive = !SDL_GetWindowFullscreenMode(handle).is_null();
        crate::elog!(
            "window flags: {:#x} fullscreen={} exclusive={exclusive}",
            u64::from(flags),
            flags & SDL_WINDOW_FULLSCREEN != 0
        );
        crate::elog!(
            "display: {}x{} @ {:.0} Hz",
            (*mode).w,
            (*mode).h,
            (*mode).refresh_rate
        );
        Ok(Self { handle })
    }

    pub fn hwnd(&self) -> *mut c_void {
        unsafe {
            let props = SDL_GetWindowProperties(self.handle);
            let hwnd = SDL_GetPointerProperty(
                props,
                SDL_PROP_WINDOW_WIN32_HWND_POINTER,
                ptr::null_mut::<c_void>(),
            );
            if hwnd.is_null() {
                panic!("SDL_GetPointerProperty(HWND) = null");
            }
            hwnd
        }
    }

    pub fn destroy(&self) {
        unsafe {
            SDL_DestroyWindow(self.handle);
            SDL_Quit();
        }
    }
}

fn sdl_err(ctx: &str) -> anyhow::Error {
    let raw = SDL_GetError();
    let msg = if raw.is_null() {
        "(no error)".to_string()
    } else {
        unsafe { CStr::from_ptr(raw).to_string_lossy().into_owned() }
    };
    anyhow!("[{ctx}] {msg}")
}
