//! bmsnt-framework engine library: raw Vulkan 1.4 renderer, SDL3 platform,
//! input core. All engine code lives in `bmsnt.Framework/` (
//! style layout); binaries are thin shells on top of this crate.

#[path = "bmsnt.Framework/framework.rs"]
pub mod bmsnt_framework;

/// Re-export the engine root at the crate root: lib-internal code uses
/// `crate::graphics`, `crate::text`, etc.; external users use
/// `bmsnt_framework::graphics`, ...
pub use bmsnt_framework::*;
