//! Vulkan context: instance, surface, device, queue. Owns the loader.

use anyhow::{anyhow, Result};
use std::ffi::{c_void, CStr};
use std::ptr;

use ash::vk;
use ash::{Entry, Instance};

pub struct GfxContext {    /// keeps the Vulkan loader alive; must not be dropped early
    #[allow(dead_code)]
    entry: Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    pub pdevice: vk::PhysicalDevice,
    device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    surface_instance: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
}

/// Debug utils callback: validation messages go to the log file.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> vk::Bool32 {
    if let Some(d) = data.as_ref() {
        let msg = if d.p_message.is_null() {
            "(no message)".to_string()
        } else {
            CStr::from_ptr(d.p_message).to_string_lossy().into_owned()
        };
        crate::elog!("[VVL {:?}] {}", severity, msg);
    }
    vk::FALSE
}

impl GfxContext {
    /// `hwnd` — Win32 window handle.
    pub unsafe fn new(hwnd: *mut c_void) -> Result<Self> {
        let entry = Entry::load().map_err(|e| anyhow!("vulkan loader: {e:?}"))?;

        // TESTAPP_VALIDATE=1 (or vk5k) enables the Khronos validation layer
        // + debug utils; violations are logged via elog
        let validate = std::env::var("VALIDATE").is_ok();
        let mut layers: Vec<*const i8> = Vec::new();
        let mut inst_exts: Vec<*const i8> = vec![
            c"VK_KHR_surface".as_ptr(),
            c"VK_KHR_win32_surface".as_ptr(),
            c"VK_KHR_get_physical_device_properties2".as_ptr(),
        ];
        if validate {
            let has_layer = entry
                .enumerate_instance_layer_properties()
                .map_err(|e| anyhow!("layer enum: {e:?}"))?
                .iter()
                .any(|p| {
                    let name = unsafe { CStr::from_ptr(p.layer_name.as_ptr()) };
                    name == c"VK_LAYER_KHRONOS_validation"
                });
            if has_layer {
                layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
                inst_exts.push(c"VK_EXT_debug_utils".as_ptr());
                crate::elog!("validation layer enabled");
            } else {
                crate::elog!("validation layer not installed");
            }
        }
        let app_name = c"black-vk".as_ptr();
        let app = vk::ApplicationInfo {
            p_application_name: app_name,
            api_version: vk::API_VERSION_1_4,
            ..Default::default()
        };
        let inst_info = vk::InstanceCreateInfo {
            p_application_info: &app,
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: inst_exts.len() as u32,
            pp_enabled_extension_names: inst_exts.as_ptr(),
            ..Default::default()
        };
        let instance = entry
            .create_instance(&inst_info, None)
            .map_err(|e| anyhow!("create_instance: {e:?}"))?;

        // debug messenger (only when validating)
        if validate {
            let dbg = ash::ext::debug_utils::Instance::load(&entry, &instance);
            let name = c"vk5k".as_ptr();
            let info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(debug_callback),
                p_user_data: name as *const c_void as *mut _ ,
                ..Default::default()
            };
            let messenger = unsafe { dbg.create_debug_utils_messenger(&info, None) }
                .map_err(|e| anyhow!("debug messenger: {e:?}"))?;
            // the raw handle is intentionally never destroyed: the callback
            // must stay alive until process shutdown (one instance per run);
            // the loader struct is kept alive the same way
            let _ = messenger;
            std::mem::forget(dbg);
            let _ = instance.handle();
        }
        crate::elog!("instance created");

        // --- surface ---
        let hinstance =
            windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(ptr::null());
        if hinstance.is_null() {
            return Err(anyhow!("GetModuleHandleW = null"));
        }
        let win32_surface = ash::khr::win32_surface::Instance::load(&entry, &instance);
        let surface_info = vk::Win32SurfaceCreateInfoKHR {
            hinstance: hinstance as isize,
            hwnd: hwnd as isize,
            ..Default::default()
        };
        let surface = {
            let fp = win32_surface.fp();
            let mut out = vk::SurfaceKHR::null();
            let r = (fp.create_win32_surface_khr)(
                instance.handle(),
                &surface_info,
                ptr::null(),
                &mut out,
            );
            if r != vk::Result::SUCCESS {
                return Err(anyhow!("create_win32_surface: {r:?}"));
            }
            out
        };
        crate::elog!("surface created");

        let surface_instance = ash::khr::surface::Instance::load(&entry, &instance);

        // --- device: first one with graphics+present ---
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| anyhow!("enumerate_physical_devices: {e:?}"))?;
        if devices.is_empty() {
            return Err(anyhow!("no physical devices"));
        }
        let mut chosen: Option<(vk::PhysicalDevice, u32)> = None;
        for pd in &devices {
            let props = instance.get_physical_device_properties(*pd);
            let families = instance
                .get_physical_device_queue_family_properties(*pd)
                .into_iter()
                .enumerate();
            for (idx, qf) in families {
                let graphics = qf.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let present = {
                    let mut ok = 0u32;
                    let r = (surface_instance
                        .fp()
                        .get_physical_device_surface_support_khr)(*pd, idx as u32, surface, &mut ok);
                    r == vk::Result::SUCCESS && ok == 1
                };
                if graphics && present {
                    crate::elog!(
                        "device: {} (queue family {idx})",
                        CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy()
                    );
                    chosen = Some((*pd, idx as u32));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let (pdevice, qfi) = chosen.ok_or_else(|| anyhow!("no device with graphics+present queue"))?;

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: qfi,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };
        let dev_exts = [c"VK_KHR_swapchain".as_ptr()];
        // dynamicRendering is core in 1.3 but must be enabled EXPLICITLY via
        // Vulkan13Features (validation checks it, the driver may not)
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features {
            dynamic_rendering: vk::TRUE,
            ..Default::default()
        };
        let dev_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: dev_exts.len() as u32,
            pp_enabled_extension_names: dev_exts.as_ptr(),
            p_next: &mut vulkan13_features as *mut _ as *const c_void,
            ..Default::default()
        };
        let device = instance
            .create_device(pdevice, &dev_info, None)
            .map_err(|e| anyhow!("create_device: {e:?}"))?;
        let queue = device.get_device_queue(qfi, 0);
        let swapchain_loader = ash::khr::swapchain::Device::load(&instance, &device);

        Ok(Self {
            entry,
            instance,
            surface,
            pdevice,
            device,
            queue_family: qfi,
            queue,
            surface_instance,
            swapchain_loader,
        })
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn surface_instance(&self) -> &ash::khr::surface::Instance {
        &self.surface_instance
    }

    pub fn swapchain_loader(&self) -> &ash::khr::swapchain::Device {
        &self.swapchain_loader
    }

    /// Recreate the VkDevice (after an unrecoverable device lost, e.g. an
    /// NVIDIA fullscreen-switch transition). Instance/surface are kept.
    /// All device-level objects (swapchain, renderer resources, fences) must
    /// be destroyed before and rebuilt after this call.
    pub unsafe fn recreate_device(&mut self) -> Result<()> {
        self.device.destroy_device(None);
        let device = Self::create_device_for(&self.instance, self.pdevice, self.queue_family)?;
        self.queue = device.get_device_queue(self.queue_family, 0);
        self.swapchain_loader = ash::khr::swapchain::Device::load(&self.instance, &device);
        self.device = device;
        crate::elog!("device recreated");
        Ok(())
    }

    /// Full teardown: device -> surface -> instance. Call after everything
    /// else (renderer/swapchain/frames) is destroyed and idle.
    pub unsafe fn destroy(self) {
        self.device.destroy_device(None);
        (self.surface_instance.fp().destroy_surface_khr)(
            self.instance.handle(),
            self.surface,
            ptr::null(),
        );
        self.instance.destroy_instance(None);
    }

    /// Device creation shared by `new` and `recreate_device`.
    unsafe fn create_device_for(
        instance: &Instance,
        pdevice: vk::PhysicalDevice,
        qfi: u32,
    ) -> Result<ash::Device> {        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: qfi,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };
        let dev_exts = [c"VK_KHR_swapchain".as_ptr()];
        // dynamicRendering is core in 1.3 but must be enabled EXPLICITLY via
        // Vulkan13Features (validation checks it, the driver may not)
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features {
            dynamic_rendering: vk::TRUE,
            ..Default::default()
        };
        let dev_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: dev_exts.len() as u32,
            pp_enabled_extension_names: dev_exts.as_ptr(),
            p_next: &mut vulkan13_features as *mut _ as *const c_void,
            ..Default::default()
        };
        instance
            .create_device(pdevice, &dev_info, None)
            .map_err(|e| anyhow!("create_device: {e:?}"))
    }
}
