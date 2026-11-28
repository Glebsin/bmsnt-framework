//! bmsnt-framework engine library root.
//!
//! Layout (component style): `Graphics` (Vulkan backend, sprites,
//! primitives), `Input`, `Logging`, `Platform` (SDL3 window, NVIDIA driver),
//! `Statistics`, `Text`, `Testing` (the test browser lives inside the
//! framework). The PascalCase folder name
//! needs an explicit `#[path]` — Rust module identifiers cannot contain
//! a dot.

#[path = "Graphics/graphics.rs"]
pub mod graphics;
#[path = "Input/input_core.rs"]
pub mod input;
#[path = "Logging/logging.rs"]
pub mod logging;
#[path = "Platform/platform.rs"]
pub mod platform;
#[path = "Statistics/frame_statistics.rs"]
pub mod statistics;
#[path = "Testing/testing.rs"]
pub mod testing;
#[path = "Text/bitmap_font.rs"]
pub mod text;
