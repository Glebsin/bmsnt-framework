//! Input: device states, per-frame events and action mapping, mirroring the
//! layout of `osu.Framework.Input`:
//!
//! - `states` — `osu.Framework.Input.States`: `InputState` aggregating
//!   `KeyboardState` / `MouseState` / `JoystickState`, built on per-button
//!   `ButtonState` (the `ButtonStates` analog);
//! - `events` — `osu.Framework.Input.Events` analog: per-frame poll results;
//! - `actions` — `FrameworkActionContainer` analog: named actions bound to
//!   device buttons with per-frame computed states;
//! - `input_manager` — `InputManager` / `UserInputManager` analog: drains the
//!   SDL3 queue into the state, keeps the gamepad open, updates the actions.
//!
//! Public API is unchanged from the previous single-file `input_core.rs`.

mod actions;
mod events;
mod input_manager;
mod states;

pub use actions::{ActionBindings, ActionDef, ActionId, ActionState, Binding};
pub use events::FrameEvents;
pub use input_manager::InputCore;
pub use states::{ButtonState, InputState, JoystickState, KeyboardState, MouseState};
