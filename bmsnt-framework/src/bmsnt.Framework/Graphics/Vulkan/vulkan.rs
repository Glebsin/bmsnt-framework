//! Vulkan backend: device, swapchain, frames-in-flight sync and the
//! renderer (osu-framework `Graphics/OpenGL` analogue, Vulkan instead).

#[path = "vulkan_device.rs"]
pub mod device;
#[path = "frame_sync.rs"]
pub mod frame_sync;
#[path = "vulkan_renderer.rs"]
pub mod renderer;
#[path = "vulkan_swapchain.rs"]
pub mod swapchain;
