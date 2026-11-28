//! NVML: lock the GPU clock to the driver max.
//!
//! nvml.dll ships with the NVIDIA driver (System32). The lock requires admin
//! rights. Cleanup: on normal destruction the clock lock is reset and NVML is
//! shut down. NOTE: the release profile uses `panic = "abort"`, so no Drop
//! runs after a panic/abort, forced termination, or a hard crash — the clock
//! lock may persist after such an exit and must be cleared externally
//! (fresh admin run of this tool, NVIDIA app, or a reboot). A startup reset
//! is attempted before applying a new lock to recover from previous crashes.

use anyhow::{anyhow, Result};
use std::ffi::c_char;

use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

const NVML_CLOCK_GRAPHICS: i32 = 0;

type FnInit = unsafe extern "system" fn() -> i32;
type FnShutdown = unsafe extern "system" fn() -> i32;
type FnGetHandle = unsafe extern "system" fn(index: u32, device: *mut usize) -> i32;
type FnGetMaxClock =
    unsafe extern "system" fn(device: usize, clock_type: i32, mhz: *mut u32) -> i32;
type FnSetLockedClocks =
    unsafe extern "system" fn(device: usize, min_mhz: u32, max_mhz: u32) -> i32;
type FnResetLockedClocks = unsafe extern "system" fn(device: usize) -> i32;
type FnGetName = unsafe extern "system" fn(device: usize, name: *mut c_char, len: i32) -> i32;

pub struct Nvml {
    _lib: HMODULE,
    handle: usize,
    pub gpu_name: String,
    pub locked_mhz: u32,
    fn_shutdown: FnShutdown,
    fn_set_locked: FnSetLockedClocks,
    fn_reset_locked: Option<FnResetLockedClocks>,
}

/// Resolve one NVML symbol: check availability first, convert to the typed
/// function pointer only afterwards (never transmute a null pointer).
///
/// # Safety
///
/// `lib` must be a valid module handle; `T` must match the real signature of
/// the named NVML export (extern "system").
unsafe fn sym<T>(lib: HMODULE, name: &[u8]) -> Result<T> {
    let addr = match unsafe { GetProcAddress(lib, name.as_ptr()) } {
        Some(f) => f as usize,
        None => 0,
    };
    if addr == 0 {
        return Err(anyhow!(
            "{} missing",
            String::from_utf8_lossy(&name[..name.len() - 1])
        ));
    }
    Ok(unsafe { std::mem::transmute_copy(&addr) })
}

impl Nvml {
    /// Init NVML and lock the GPU clock to the driver max.
    ///
    /// # Safety
    ///
    /// Loads nvml.dll and calls its exported C functions; the caller must
    /// run on a thread that may load system libraries.
    pub unsafe fn init_and_lock_max() -> Result<Self> {
        let lib = unsafe { LoadLibraryA(c"nvml.dll".as_ptr().cast()) };
        let lib = if lib.is_null() {
            unsafe {
                LoadLibraryA(
                    c"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"
                        .as_ptr()
                        .cast(),
                )
            }
        } else {
            lib
        };
        if lib.is_null() {
            return Err(anyhow!("nvml.dll not found"));
        }

        // resolve and validate every symbol before nvmlInit: at this point
        // there is no NVML session to clean up on failure
        let init: FnInit = unsafe { sym(lib, b"nvmlInit_v2\0")? };
        let shutdown: FnShutdown = unsafe { sym(lib, b"nvmlShutdown\0")? };
        let get_handle: FnGetHandle = unsafe { sym(lib, b"nvmlDeviceGetHandleByIndex_v2\0")? };
        let get_max: FnGetMaxClock = unsafe { sym(lib, b"nvmlDeviceGetMaxClockInfo\0")? };
        let set_locked: FnSetLockedClocks = unsafe { sym(lib, b"nvmlDeviceSetGpuLockedClocks\0")? };
        // reset is optional on old NVML versions
        let reset_locked: Option<FnResetLockedClocks> =
            unsafe { sym::<FnResetLockedClocks>(lib, b"nvmlDeviceResetGpuLockedClocks\0") }.ok();
        let get_name: FnGetName = unsafe { sym(lib, b"nvmlDeviceGetName\0")? };

        let r = unsafe { init() };
        if r != 0 {
            return Err(anyhow!("nvmlInit_v2 = {r}"));
        }

        // from here every error path must call nvmlShutdown
        let inner = unsafe {
            Self::setup(
                lib,
                shutdown,
                get_handle,
                get_max,
                set_locked,
                reset_locked,
                get_name,
            )
        };
        match inner {
            Ok(nvml) => Ok(nvml),
            Err(e) => {
                let rs = unsafe { shutdown() };
                if rs != 0 {
                    crate::elog!("nvmlShutdown after init failure = {rs}");
                }
                Err(e)
            }
        }
    }

    /// Post-init setup: device handle, name, startup clock reset, new lock.
    /// Errors here leave the shutdown responsibility with the caller.
    unsafe fn setup(
        lib: HMODULE,
        fn_shutdown: FnShutdown,
        get_handle: FnGetHandle,
        get_max: FnGetMaxClock,
        set_locked: FnSetLockedClocks,
        reset_locked: Option<FnResetLockedClocks>,
        get_name: FnGetName,
    ) -> Result<Self> {
        // existing limitation: NVML device index 0 is used without verifying
        // it matches the physical device chosen by Vulkan (single-GPU target)
        let mut handle = 0usize;
        let r = unsafe { get_handle(0, &mut handle) };
        if r != 0 {
            return Err(anyhow!("nvmlDeviceGetHandleByIndex_v2 = {r}"));
        }

        let mut name_buf = [0 as c_char; 96];
        unsafe { get_name(handle, name_buf.as_mut_ptr(), 96) };
        let gpu_name = {
            let end = name_buf.iter().position(|&c| c == 0).unwrap_or(96);
            String::from_utf8_lossy(
                &name_buf[..end]
                    .iter()
                    .map(|&c| c as u8)
                    .collect::<Vec<u8>>(),
            )
            .into_owned()
        };

        let mut max_mhz = 0u32;
        let r = unsafe { get_max(handle, NVML_CLOCK_GRAPHICS, &mut max_mhz) };
        if r != 0 || max_mhz == 0 {
            return Err(anyhow!("nvmlDeviceGetMaxClockInfo = {r}"));
        }

        // startup recovery: clear a clock lock left behind by a previous
        // crashed run before applying a new one
        match reset_locked {
            Some(reset) => {
                let r = unsafe { reset(handle) };
                if r == 0 {
                    crate::elog!("NVML: pre-existing clock lock cleared");
                } else {
                    crate::elog!("NVML: clock reset failed = {r}");
                }
            }
            None => {
                crate::elog!("NVML: reset function unavailable — a lock left by a crashed run cannot be cleared here");
            }
        }

        let r = unsafe { set_locked(handle, max_mhz, max_mhz) };
        if r != 0 {
            return Err(anyhow!(
                "nvmlDeviceSetGpuLockedClocks({max_mhz}) = {r} (admin rights required)"
            ));
        }

        Ok(Self {
            _lib: lib,
            handle,
            gpu_name,
            locked_mhz: max_mhz,
            fn_shutdown,
            fn_set_locked: set_locked,
            fn_reset_locked: reset_locked,
        })
    }

    /// Change the lock (for a future "ultimate mode" setting).
    ///
    /// # Safety
    ///
    /// Calls the NVML C API; the NVML session from init_and_lock_max must
    /// still be alive.
    #[allow(dead_code)]
    pub unsafe fn set_clock(&self, mhz: u32) -> Result<()> {
        let r = unsafe { (self.fn_set_locked)(self.handle, mhz, mhz) };
        if r != 0 {
            return Err(anyhow!("nvmlDeviceSetGpuLockedClocks({mhz}) = {r}"));
        }
        Ok(())
    }

    /// Reset the clock lock, then shut NVML down. Runs on normal Drop only —
    /// an abort/crash bypasses this entirely (see module docs).
    ///
    /// # Safety
    ///
    /// Calls the NVML C API; the NVML session from init_and_lock_max must
    /// still be alive.
    pub unsafe fn reset(&self) {
        match self.fn_reset_locked {
            Some(reset) => {
                let r = unsafe { reset(self.handle) };
                if r == 0 {
                    crate::elog!("NVML: clock lock reset");
                } else {
                    crate::elog!("NVML: clock reset failed = {r}");
                }
            }
            None => {
                crate::elog!("NVML: clock lock could not be reset (reset function unavailable)");
            }
        }
        let r = unsafe { (self.fn_shutdown)() };
        if r != 0 {
            crate::elog!("nvmlShutdown = {r}");
        }
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe { self.reset() };
    }
}
