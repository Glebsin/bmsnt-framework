//! Per-frame input results and edge events (`osu.Framework.Input.Events` analog).
//!
//! The framework dispatches one `UiEvent` per input change; this engine is
//! frame-based, so edges are flattened into per-frame queries and this module
//! keeps only the poll result handed back to the caller.

/// Per-frame poll result, returned by [`super::InputCore::poll`].
#[derive(Default)]
pub struct FrameEvents {
    pub quit: bool,
}
