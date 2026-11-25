//! NVML: лок максимальной доступной частоты GPU.
//!
//! nvml.dll идёт с драйвером NVIDIA (System32). Лок требует прав администратора
//! (манифест requireAdministrator). Сброс при выходе.

use anyhow::{anyhow, Result};
use std::ffi::c_char;
use std::ptr;

use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

const NVML_CLOCK_GRAPHICS: i32 = 0;

type FnInit = unsafe extern "system" fn() -> i32;
type FnGetHandle = unsafe extern "system" fn(index: u32, device: *mut usize) -> i32;
type FnGetMaxClock = unsafe extern "system" fn(device: usize, clock_type: i32, mhz: *mut u32) -> i32;
type FnSetLockedClocks = unsafe extern "system" fn(device: usize, min_mhz: u32, max_mhz: u32) -> i32;
type FnResetLockedClocks = unsafe extern "system" fn(device: usize) -> i32;
type FnGetName = unsafe extern "system" fn(device: usize, name: *mut c_char, len: i32) -> i32;

pub struct Nvml {
    _lib: HMODULE,
    handle: usize,
    pub gpu_name: String,
    pub locked_mhz: u32,
    fn_set_locked: FnSetLockedClocks,
    fn_reset_locked: Option<FnResetLockedClocks>,
}

impl Nvml {
    /// Инициализация NVML + лок частоты GPU на максимум, доступный драйверу.
    pub unsafe fn init_and_lock_max() -> Result<Self> {
        let lib = LoadLibraryA(b"nvml.dll\0".as_ptr());
        let lib = if lib.is_null() {
            LoadLibraryA(b"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll\0".as_ptr())
        } else {
            lib
        };
        if lib.is_null() {
            return Err(anyhow!("nvml.dll не найдена"));
        }

        let get = |name: &[u8]| -> usize {
            match GetProcAddress(lib, name.as_ptr()) {
                Some(f) => f as usize,
                None => 0,
            }
        };

        let init: FnInit = unsafe { std::mem::transmute(get(b"nvmlInit_v2\0")) };
        if get(b"nvmlInit_v2\0") == 0 {
            return Err(anyhow!("nvmlInit_v2 нет"));
        }
        let get_handle: FnGetHandle = unsafe { std::mem::transmute(get(b"nvmlDeviceGetHandleByIndex_v2\0")) };
        if get(b"nvmlDeviceGetHandleByIndex_v2\0") == 0 {
            return Err(anyhow!("nvmlDeviceGetHandleByIndex_v2 нет"));
        }
        let get_max: FnGetMaxClock = unsafe { std::mem::transmute(get(b"nvmlDeviceGetMaxClockInfo\0")) };
        if get(b"nvmlDeviceGetMaxClockInfo\0") == 0 {
            return Err(anyhow!("nvmlDeviceGetMaxClockInfo нет"));
        }
        if get(b"nvmlDeviceSetGpuLockedClocks\0") == 0 {
            return Err(anyhow!("nvmlDeviceSetGpuLockedClocks нет"));
        }
        let set_locked: FnSetLockedClocks = unsafe { std::mem::transmute(get(b"nvmlDeviceSetGpuLockedClocks\0")) };
        let reset_locked: Option<FnResetLockedClocks> = if get(b"nvmlDeviceResetGpuLockedClocks\0") != 0 {
            Some(unsafe { std::mem::transmute(get(b"nvmlDeviceResetGpuLockedClocks\0")) })
        } else {
            None
        };
        if get(b"nvmlDeviceGetName\0") == 0 {
            return Err(anyhow!("nvmlDeviceGetName нет"));
        }
        let get_name: FnGetName = unsafe { std::mem::transmute(get(b"nvmlDeviceGetName\0")) };

        let r = init();
        if r != 0 {
            return Err(anyhow!("nvmlInit_v2 = {r}"));
        }

        let mut handle = 0usize;
        let r = get_handle(0, &mut handle);
        if r != 0 {
            return Err(anyhow!("nvmlDeviceGetHandleByIndex_v2 = {r}"));
        }

        let mut name_buf = [0 as c_char; 96];
        let _ = get_name(handle, name_buf.as_mut_ptr(), 96);
        let gpu_name = {
            let end = name_buf.iter().position(|&c| c == 0).unwrap_or(96);
            String::from_utf8_lossy(
                &name_buf[..end].iter().map(|&c| c as u8).collect::<Vec<u8>>(),
            )
            .into_owned()
        };

        let mut max_mhz = 0u32;
        let r = get_max(handle, NVML_CLOCK_GRAPHICS, &mut max_mhz);
        if r != 0 || max_mhz == 0 {
            return Err(anyhow!("nvmlDeviceGetMaxClockInfo = {r}"));
        }

        let r = set_locked(handle, max_mhz, max_mhz);
        if r != 0 {
            return Err(anyhow!(
                "nvmlDeviceSetGpuLockedClocks({max_mhz}) = {r} (нужны права администратора)"
            ));
        }

        Ok(Self {
            _lib: lib,
            handle,
            gpu_name,
            locked_mhz: max_mhz,
            fn_set_locked: set_locked,
            fn_reset_locked: reset_locked,
        })
    }

    /// Смена лока (для будущего «ультимативного режима» из настроек).
    #[allow(dead_code)]
    pub unsafe fn set_clock(&self, mhz: u32) -> Result<()> {
        let r = (self.fn_set_locked)(self.handle, mhz, mhz);
        if r != 0 {
            return Err(anyhow!("nvmlDeviceSetGpuLockedClocks({mhz}) = {r}"));
        }
        Ok(())
    }

    /// Сброс лока — вызывать при выходе.
    pub unsafe fn reset(&self) {
        if let Some(reset) = self.fn_reset_locked {
            let _ = reset(self.handle);
        }
        // reset-функции может не быть в старых NVML — тогда лок снимается
        // самим драйвером после завершения процесса
        let _ = ptr::null_mut::<c_char>();
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe { self.reset() };
    }
}
