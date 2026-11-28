//! Input states: snapshots of raw device levels.
//!
//! Mirrors `osu.Framework.Input.States`: an [`InputState`] aggregates the
//! per-device states (keyboard, mouse, joystick). States hold *levels*;
//! frame edges are derived in `super::events`.

use sdl3_sys::everything::*;

/// number of SDL3 scancode slots
pub(crate) const SCANCODE_COUNT: usize = SDL_Scancode::COUNT.0 as usize;
/// mouse button slots; SDL indexes 1..=5 (left, middle, right, x1, x2)
pub(crate) const MOUSE_BUTTONS: usize = 6;
/// SDL_GamepadButton::COUNT is not const-friendly; all standard buttons +
/// paddles + misc fit under 24
pub(crate) const PAD_BUTTONS: usize = 24;

/// State of one binary input (key/button): level + this-frame edge.
///
/// Rust analog of `osu.Framework.Input.States.ButtonStates`: the level plus
/// the pressed/released edges of the current frame.
#[derive(Clone, Copy, Default)]
pub struct ButtonState {
    pub down: bool,
    /// pressed this frame (true for exactly one frame)
    pub pressed: bool,
    /// released this frame
    pub released: bool,
    /// timestamp of the latest edge event (ns, SDL_GetTicksNS basis)
    pub ts_ns: u64,
}

impl ButtonState {
    /// Edges live for exactly one frame; called at the start of every poll.
    pub(crate) fn clear_edges(&mut self) {
        self.pressed = false;
        self.released = false;
    }
}

/// Flat keyboard storage, indexed by scancode. `osu.Framework.Input.States.KeyboardState` analog.
pub struct KeyboardState {
    pub(crate) keys: Vec<ButtonState>,
}

impl KeyboardState {
    pub(crate) fn new() -> Self {
        Self {
            keys: vec![ButtonState::default(); SCANCODE_COUNT],
        }
    }
    pub(crate) fn get(&mut self, sc: SDL_Scancode) -> &mut ButtonState {
        &mut self.keys[(sc.0 as usize) % SCANCODE_COUNT]
    }
    pub(crate) fn clear_edges(&mut self) {
        for b in self.keys.iter_mut() {
            b.clear_edges();
        }
    }
    /// Scancodes currently held, ordered by press time (oldest first).
    /// Timestamps are the SDL event ns of the latest edge per key.
    pub fn held_keys(&self) -> Vec<(u64, SDL_Scancode)> {
        let mut v: Vec<(u64, SDL_Scancode)> = self
            .keys
            .iter()
            .enumerate()
            .filter(|(_, b)| b.down)
            .map(|(i, b)| (b.ts_ns, SDL_Scancode(i as i32)))
            .collect();
        v.sort_by_key(|(ts, _)| *ts);
        v
    }
    pub fn key_down(&self, sc: SDL_Scancode) -> bool {
        self.keys[(sc.0 as usize) % SCANCODE_COUNT].down
    }
    pub fn key_pressed(&self, sc: SDL_Scancode) -> bool {
        self.keys[(sc.0 as usize) % SCANCODE_COUNT].pressed
    }
}

/// Mouse position, per-frame delta, wheel and buttons.
/// `osu.Framework.Input.States.MouseState` analog.
pub struct MouseState {
    pub x: f32,
    pub y: f32,
    /// delta accumulated over the frame (sum of xrel/yrel events)
    pub dx: f32,
    pub dy: f32,
    pub wheel_x: f32,
    pub wheel_y: f32,
    pub(crate) buttons: Vec<ButtonState>,
}

impl MouseState {
    pub(crate) fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            dx: 0.0,
            dy: 0.0,
            wheel_x: 0.0,
            wheel_y: 0.0,
            buttons: vec![ButtonState::default(); MOUSE_BUTTONS],
        }
    }
    pub(crate) fn clear_frame(&mut self) {
        for b in self.buttons.iter_mut() {
            b.clear_edges();
        }
        self.dx = 0.0;
        self.dy = 0.0;
        self.wheel_x = 0.0;
        self.wheel_y = 0.0;
    }
    pub(crate) fn button(&mut self, button: u8) -> &mut ButtonState {
        &mut self.buttons[(button as usize) % MOUSE_BUTTONS]
    }
    pub fn mouse_down(&self, button: u8) -> bool {
        self.buttons[(button as usize) % MOUSE_BUTTONS].down
    }
    pub fn mouse_pressed(&self, button: u8) -> bool {
        self.buttons[(button as usize) % MOUSE_BUTTONS].pressed
    }
    pub fn pos(&self) -> (f32, f32) {
        (self.x, self.y)
    }
    /// Mouse delta over the frame (sum of events; for aim/cursor).
    pub fn delta(&self) -> (f32, f32) {
        (self.dx, self.dy)
    }
    pub fn wheel(&self) -> (f32, f32) {
        (self.wheel_x, self.wheel_y)
    }
}

/// Gamepad button levels + per-frame edges. `osu.Framework.Input.States.JoystickState` analog.
#[derive(Clone, Copy, Default)]
pub struct JoystickState {
    pub(crate) buttons: [ButtonState; PAD_BUTTONS],
}

impl JoystickState {
    pub(crate) fn clear_edges(&mut self) {
        for b in self.buttons.iter_mut() {
            b.clear_edges();
        }
    }
    pub(crate) fn button(&mut self, i: usize) -> &mut ButtonState {
        &mut self.buttons[i % PAD_BUTTONS]
    }
    pub fn button_down(&self, i: usize) -> bool {
        self.buttons[i % PAD_BUTTONS].down
    }
}

/// An object that stores all input states (`osu.Framework.Input.States.InputState` analog).
pub struct InputState {
    pub keyboard: KeyboardState,
    pub mouse: MouseState,
    pub joystick: JoystickState,
}

impl InputState {
    pub(crate) fn new() -> Self {
        Self {
            keyboard: KeyboardState::new(),
            mouse: MouseState::new(),
            joystick: JoystickState::default(),
        }
    }
    /// Frame edges live for exactly one frame; call at the start of every poll.
    pub(crate) fn clear_frame(&mut self) {
        self.keyboard.clear_edges();
        self.mouse.clear_frame();
        self.joystick.clear_edges();
    }
}
