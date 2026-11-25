//! Кадры в полёте: fences + image-семафоры + софт-трекинг сигнальности.
//!
//! Софт-трекинг нужен потому, что у Vulkan нет «пересигналить fence»:
//! после device_wait_idle (пересоздание свопчейна) фактически вся работа
//! завершена, но fence в reset-состоянии повесил бы следующий wait навсегда.

use anyhow::{Context, Result};

use ash::vk;

/// кадров в полёте (и слотов кольцевого буфера)
pub const FRAMES_IN_FLIGHT: usize = 8;

pub struct FramesInFlight {
    fences: Vec<vk::Fence>,
    img_sems: Vec<vk::Semaphore>,
    /// программная копия состояния fence (true = signaled или «работа завершена»)
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

    /// Ждёт fence слота, если тот ещё в полёте. Сам fence НЕ сбрасывается —
    /// сброс происходит непосредственно перед submit (`reset_for_submit`).
    pub unsafe fn ensure_idle(&mut self, device: &ash::Device, slot: usize) -> Result<()> {
        if !self.signaled[slot] {
            device
                .wait_for_fences(&[self.fences[slot]], true, u64::MAX)
                .context("wait fence")?;
            self.signaled[slot] = true;
        }
        Ok(())
    }

    /// Ждёт все кадры в полёте. С кольцевым per-frame буфером для записи вершин
    /// не нужен; оставлен для будущих случаев «перезаписать всё».
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

    /// Сброс fence непосредственно перед queue_submit.
    pub unsafe fn reset_for_submit(&mut self, device: &ash::Device, slot: usize) -> Result<()> {
        device
            .reset_fences(&[self.fences[slot]])
            .context("reset fence")?;
        self.signaled[slot] = false;
        Ok(())
    }

    /// После device_wait_idle: всё завершено, все fence «логически signaled».
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
