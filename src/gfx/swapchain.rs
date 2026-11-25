//! Swapchain: deep queue (min+5, clamped to max), IMMEDIATE, per-image
//! done semaphores. Recreated on OUT_OF_DATE.

use anyhow::{anyhow, Result};
use std::ptr;

use ash::vk;

use super::device::GfxContext;

/// vkAcquireNextImageKHR result.
pub enum Acquire {
    Image(u32),
    /// VK_NOT_READY — the Windows+IMMEDIATE driver sometimes returns it even
    /// with an infinite timeout; just retry.
    NotReady,
    /// VK_ERROR_OUT_OF_DATE_KHR — recreate the swapchain.
    OutOfDate,
}

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    /// swapchain function-table pointer; the table lives in GfxContext,
    /// which outlives the swapchain
    fp: *const ash::khr::swapchain::DeviceFn,
    pub images: Vec<vk::Image>,
    /// done semaphores: one per SWAPCHAIN IMAGE, not per frame in flight.
    /// A fence only guarantees the end of queue_submit; present waits on the
    /// semaphore inside the presentation engine. The image comes back through
    /// acquire only after its presentation finishes, so its semaphore must not
    /// be re-signaled earlier than acquire hands the image out again.
    done_sems: Vec<vk::Semaphore>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
}

impl Swapchain {
    pub unsafe fn new(ctx: &GfxContext) -> Result<Self> {
        let device = ctx.device();
        let sfp = ctx.surface_instance().fp();

        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        if (sfp.get_physical_device_surface_capabilities_khr)(
            ctx.pdevice,
            ctx.surface(),
            &mut caps,
        ) != vk::Result::SUCCESS
        {
            return Err(anyhow!("surface capabilities"));
        }

        let formats = query_vec(|n, p| {
            (sfp.get_physical_device_surface_formats_khr)(ctx.pdevice, ctx.surface(), n, p)
        });
        let present_modes = query_vec(|n, p| {
            (sfp.get_physical_device_surface_present_modes_khr)(ctx.pdevice, ctx.surface(), n, p)
        });
        let formats = if formats.is_empty() {
            vec![vk::SurfaceFormatKHR::default()]
        } else {
            formats
        };
        let format = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .copied()
            .unwrap_or(formats[0]);

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let mut image_count = caps.min_image_count + 5;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: 1920,
                height: 1080,
            }
        };

        let sc_info = vk::SwapchainCreateInfoKHR {
            surface: ctx.surface(),
            min_image_count: image_count,
            image_format: format.format,
            image_color_space: format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let loader = ctx.swapchain_loader();
        let fp = loader.fp() as *const ash::khr::swapchain::DeviceFn;
        let handle = {
            let fp = loader.fp();
            let mut out = vk::SwapchainKHR::null();
            let r = (fp.create_swapchain_khr)(device.handle(), &sc_info, ptr::null(), &mut out);
            if r != vk::Result::SUCCESS {
                return Err(anyhow!("create_swapchain: {r:?}"));
            }
            out
        };
        let images = query_vec(|n, p| {
            (loader.fp().get_swapchain_images_khr)(device.handle(), handle, n, p)
        });
        if images.is_empty() {
            return Err(anyhow!("swapchain has no images"));
        }
        let done_sems = (0..images.len())
            .map(|_| {
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .expect("done sem")
            })
            .collect();

        Ok(Self {
            handle,
            fp,
            images,
            done_sems,
            format: format.format,
            extent,
            present_mode,
        })
    }

    /// vkAcquireNextImageKHR with an infinite timeout and the frame semaphore.
    pub unsafe fn acquire(&self, device: &ash::Device, img_sem: vk::Semaphore) -> Result<Acquire> {
        let mut idx = 0u32;
        let r = ((*self.fp).acquire_next_image_khr)(
            device.handle(),
            self.handle,
            u64::MAX,
            img_sem,
            vk::Fence::null(),
            &mut idx,
        );
        Ok(match r {
            vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => Acquire::Image(idx),
            vk::Result::NOT_READY => Acquire::NotReady,
            vk::Result::ERROR_OUT_OF_DATE_KHR => Acquire::OutOfDate,
            other => return Err(anyhow!("acquire: {other:?}")),
        })
    }

    /// vkQueuePresentKHR; OUT_OF_DATE is returned as Err(vk::Result) for recreation.
    pub unsafe fn present(
        &self,
        queue: vk::Queue,
        img_idx: u32,
        done_sem: vk::Semaphore,
    ) -> Result<(), vk::Result> {
        let present = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &done_sem,
            swapchain_count: 1,
            p_swapchains: &self.handle,
            p_image_indices: &img_idx,
            ..Default::default()
        };
        let r = ((*self.fp).queue_present_khr)(queue, &present);
        match r {
            vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => Ok(()),
            other => Err(other),
        }
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn done_sem(&self, img_idx: usize) -> vk::Semaphore {
        self.done_sems[img_idx]
    }

    #[allow(dead_code)]
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        for s in &self.done_sems {
            device.destroy_semaphore(*s, None);
        }
        ((*self.fp).destroy_swapchain_khr)(device.handle(), self.handle, ptr::null());
    }
}

/// Two-step array query (count first, then data).
pub unsafe fn query_vec<T>(
    mut call: impl FnMut(*mut u32, *mut T) -> vk::Result,
) -> Vec<T> {
    let mut n = 0u32;
    let _ = call(&mut n, ptr::null_mut());
    let mut v: Vec<T> = Vec::with_capacity(n as usize);
    let r = call(&mut n, v.as_mut_ptr());
    if r != vk::Result::SUCCESS {
        return Vec::new();
    }
    v.set_len(n as usize);
    v
}
