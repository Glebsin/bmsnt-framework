//! Frames in flight: fences + image semaphores + soft signaled tracking.
//!
//! Soft tracking exists because Vulkan cannot re-signal a fence: after
//! device_wait_idle (swapchain recreation) all work is actually done, but a
//! fence left in the reset state would hang the next wait forever.

use anyhow::{Context, Result};

use ash::vk;

/// frames in flight (and ring buffer slots)
pub const FRAMES_IN_FLIGHT: usize = 8;

/// Configurable frame count for controlled 2/4/8 comparisons (BMSNT_FIF env,
/// clamped to 2..=8). Default stays 8 (IMMEDIATE benchmark scenario). Read
/// once; the same value must size every slot-dependent buffer.
pub fn frames_in_flight() -> usize {
    static VAL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("BMSNT_FIF")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.clamp(2, 8))
            .unwrap_or(FRAMES_IN_FLIGHT)
    })
}

pub struct FramesInFlight {
    fences: Vec<vk::Fence>,
    img_sems: Vec<vk::Semaphore>,
    /// software copy of fence state (true = signaled or "work complete")
    #[allow(dead_code)]
    signaled: Vec<bool>,
    index: usize,
}

impl FramesInFlight {
    pub unsafe fn new(device: &ash::Device, count: usize) -> Self {
        let mut fences = Vec::with_capacity(count);
        let mut img_sems = Vec::with_capacity(count);
        for _ in 0..count {
            fences.push(
                device
                    .create_fence(
                        &vk::FenceCreateInfo {
                            flags: vk::FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                        None,
                    )
                    .expect("fence"),
            );
            img_sems.push(
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .expect("img sem"),
            );
        }
        Self {
            fences,
            img_sems,
            signaled: vec![true; count],
            index: 0,
        }
    }

    pub fn slot(&self) -> usize {
        self.index
    }

    pub fn fence(&self, slot: usize) -> vk::Fence {
        self.fences[slot]
    }

    #[allow(dead_code)]
    pub fn all_fences(&self) -> &[vk::Fence] {
        &self.fences
    }

    pub fn image_semaphore(&self, slot: usize) -> vk::Semaphore {
        self.img_sems[slot]
    }

    pub fn image_semaphore_ref(&self, slot: usize) -> &vk::Semaphore {
        &self.img_sems[slot]
    }

    /// Wait for the slot fence if still in flight, bounded by `timeout_ns`.
    /// Returns Ok(true) when the slot is idle, Ok(false) on timeout (the
    /// device may be wedged — the caller decides to rebuild or retry).
    pub unsafe fn try_ensure_idle(
        &mut self,
        device: &ash::Device,
        slot: usize,
        timeout_ns: u64,
    ) -> Result<bool> {
        if self.signaled[slot] {
            return Ok(true);
        }
        match device.wait_for_fences(&[self.fences[slot]], true, timeout_ns) {
            Ok(()) => {
                self.signaled[slot] = true;
                Ok(true)
            }
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Wait for the slot fence if still in flight. The fence itself is NOT
    /// reset here — reset happens right before submit (`reset_for_submit`).
    pub unsafe fn ensure_idle(&mut self, device: &ash::Device, slot: usize) -> Result<()> {
        if !self.signaled[slot] {
            device
                .wait_for_fences(&[self.fences[slot]], true, u64::MAX)
                .context("wait fence")?;
            self.signaled[slot] = true;
        }
        Ok(())
    }

    /// Wait for all in-flight frames. Not needed with the per-frame ring
    /// vertex buffer; kept for future "rewrite everything" cases.
    #[allow(dead_code)]
    pub unsafe fn wait_all(&mut self, device: &ash::Device) -> Result<()> {
        let pending: Vec<vk::Fence> = self
            .fences
            .iter()
            .copied()
            .zip(self.signaled.iter())
            .filter(|(_, s)| !**s)
            .map(|(f, _)| f)
            .collect();
        if !pending.is_empty() {
            device
                .wait_for_fences(&pending, true, u64::MAX)
                .context("wait all fences")?;
        }
        for s in self.signaled.iter_mut() {
            *s = true;
        }
        Ok(())
    }

    /// Reset the fence right before queue_submit.
    pub unsafe fn reset_for_submit(&mut self, device: &ash::Device, slot: usize) -> Result<()> {
        device
            .reset_fences(&[self.fences[slot]])
            .context("reset fence")?;
        self.signaled[slot] = false;
        Ok(())
    }

    /// After device_wait_idle: everything is complete, all fences are
    /// "logically signaled".
    pub fn on_idle(&mut self) {
        for s in self.signaled.iter_mut() {
            *s = true;
        }
    }

    pub fn advance(&mut self) {
        self.index = (self.index + 1) % self.fences.len();
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        for f in &self.fences {
            device.destroy_fence(*f, None);
        }
        for s in &self.img_sems {
            device.destroy_semaphore(*s, None);
        }
    }
}
