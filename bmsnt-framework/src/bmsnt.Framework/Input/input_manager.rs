//! InputManager: drains the SDL3 event queue into the [`InputState`] and
//! updates the action bindings (`osu.Framework.Input.InputManager` /
//! `UserInputManager` analog).
//!
//! Low-latency principles:
//! - events are drained EVERY frame, before update/render; a key pressed
//!   mid-frame is handled at the start of the next frame;
//! - keyboard and mouse are event-driven (SDL_EVENT_KEY_DOWN / MOUSE_BUTTON_DOWN,
//!   repeats ignored); gamepad buttons are events too (gbutton);
//!   gamepad axes/triggers are polled once per frame (axes have no events);
//! - SDL3 gives every event a nanosecond timestamp вЂ” edges keep it, so the
//!   engine can measure input-to-display latency;
//! - no allocations or HashMap in the poll loop: state lives in flat arrays
//!   indexed by scancode / button index.

use sdl3_sys::everything::*;

use super::actions::{ActionBindings, ActionId, Binding};
use super::events::FrameEvents;
use super::states::{InputState, PAD_BUTTONS};

/// Open gamepad.
struct Gamepad {
    handle: *mut SDL_Gamepad,
    instance_id: SDL_JoystickID,
}

/// Input manager. One instance per app; call `poll()` once at the start of each frame.
pub struct InputCore {
    /// raw device states (keyboard / mouse / joystick)
    pub state: InputState,
    /// named actions computed from `state` every frame
    pub bindings: ActionBindings,
    gamepad: Option<Gamepad>,
    /// latest edge event for the debug overlay: (action, timestamp)
    last_edge: Option<(ActionId, u64)>,
    /// events consumed in the last poll (diagnostics)
    events_last_poll: u32,
}

impl InputCore {
    pub fn new() -> Self {
        Self {
            state: InputState::new(),
            bindings: ActionBindings::new(),
            gamepad: None,
            last_edge: None,
            events_last_poll: 0,
        }
    }

    /// Register an action. ActionId = registration order.
    pub fn map(&mut self, name: &'static str, bindings: &[Binding]) -> ActionId {
        self.bindings.map(name, bindings)
    }

    /// Drain the SDL3 event queue. Call exactly once per frame, before update
    /// and render.
    pub fn poll(&mut self) -> FrameEvents {
        let mut ev = FrameEvents::default();

        // edges live for exactly one frame
        self.last_edge = None;
        self.state.clear_frame();

        let mut event: SDL_Event = unsafe { std::mem::zeroed() };
        let mut eaten = 0u32;
        unsafe {
            while SDL_PollEvent(&mut event) {
                eaten += 1;
                self.handle_event(&mut ev, &event);
            }
        }
        self.events_last_poll = eaten;

        // sync gamepad button levels with the driver (SDL_GetGamepadButton):
        // events may be missed on desync; axes/triggers would be polled here too
        if self.gamepad.is_some() {
            for i in 0..PAD_BUTTONS {
                let btn = SDL_GamepadButton(i as ::core::ffi::c_int);
                let driver_level =
                    unsafe { SDL_GetGamepadButton(self.gamepad.as_ref().unwrap().handle, btn) };
                let b = self.state.joystick.button(i);
                if driver_level {
                    if !b.down {
                        // pressed without an event (or the event arrives next frame)
                        b.down = true;
                        b.pressed = true;
                        b.ts_ns = unsafe { SDL_GetTicksNS() };
                    }
                } else if b.down {
                    b.down = false;
                    b.released = true;
                    b.ts_ns = unsafe { SDL_GetTicksNS() };
                }
            }
        }

        self.last_edge = self.bindings.update(&self.state);
        ev
    }

    /// Route one SDL event into the input state. Reads SDL_Event union fields.
    unsafe fn handle_event(&mut self, ev: &mut FrameEvents, event: &SDL_Event) {
        match event.r#type {
            t if t == SDL_EventType::QUIT.0 => {
                ev.quit = true;
            }
            // keyboard: event-driven, repeats ignored
            t if t == SDL_EventType::KEY_DOWN.0 => {
                let k = &event.key;
                if !k.repeat {
                    let b = self.state.keyboard.get(k.scancode);
                    b.down = true;
                    b.pressed = true;
                    b.ts_ns = k.timestamp;
                }
            }
            t if t == SDL_EventType::KEY_UP.0 => {
                let k = &event.key;
                let b = self.state.keyboard.get(k.scancode);
                b.down = false;
                b.released = true;
                b.ts_ns = k.timestamp;
            }
            // mouse: motion deltas accumulate over the frame
            t if t == SDL_EventType::MOUSE_MOTION.0 => {
                let m = &event.motion;
                let mouse = &mut self.state.mouse;
                mouse.x = m.x;
                mouse.y = m.y;
                mouse.dx += m.xrel;
                mouse.dy += m.yrel;
            }
            t if t == SDL_EventType::MOUSE_BUTTON_DOWN.0 => {
                let m = &event.button;
                let b = self.state.mouse.button(m.button);
                b.down = true;
                b.pressed = true;
                b.ts_ns = m.timestamp;
            }
            t if t == SDL_EventType::MOUSE_BUTTON_UP.0 => {
                let m = &event.button;
                let b = self.state.mouse.button(m.button);
                b.down = false;
                b.released = true;
                b.ts_ns = m.timestamp;
            }
            t if t == SDL_EventType::MOUSE_WHEEL.0 => {
                let w = &event.wheel;
                self.state.mouse.wheel_x += w.x;
                self.state.mouse.wheel_y += w.y;
            }
            // gamepad hot-plug
            t if t == SDL_EventType::GAMEPAD_ADDED.0 => {
                self.on_gamepad_added(event.gdevice.which);
            }
            t if t == SDL_EventType::GAMEPAD_REMOVED.0 => {
                self.on_gamepad_removed(event.gdevice.which);
            }
            // gamepad buttons: events, like the keyboard
            t if t == SDL_EventType::GAMEPAD_BUTTON_DOWN.0 => {
                let g = &event.gbutton;
                let b = self.state.joystick.button(g.button as usize);
                b.down = true;
                b.pressed = true;
                b.ts_ns = g.timestamp;
            }
            t if t == SDL_EventType::GAMEPAD_BUTTON_UP.0 => {
                let g = &event.gbutton;
                let b = self.state.joystick.button(g.button as usize);
                b.down = false;
                b.released = true;
                b.ts_ns = g.timestamp;
            }
            _ => {}
        }
    }

    fn on_gamepad_added(&mut self, id: SDL_JoystickID) {
        if self.gamepad.is_some() || unsafe { !SDL_IsGamepad(id) } {
            return;
        }
        let handle = unsafe { SDL_OpenGamepad(id) };
        if !handle.is_null() {
            crate::elog!("gamepad connected (id {})", id.0);
            self.gamepad = Some(Gamepad {
                handle,
                instance_id: id,
            });
        }
    }

    fn on_gamepad_removed(&mut self, id: SDL_JoystickID) {
        if let Some(pad) = &self.gamepad {
            if pad.instance_id == id {
                crate::elog!("gamepad disconnected");
                unsafe { SDL_CloseGamepad(pad.handle) };
                self.gamepad = None;
            }
        }
    }

    // --- direct device queries (pass through to the state) ---

    pub fn key_down(&self, sc: SDL_Scancode) -> bool {
        self.state.keyboard.key_down(sc)
    }
    pub fn key_pressed(&self, sc: SDL_Scancode) -> bool {
        self.state.keyboard.key_pressed(sc)
    }
    /// Scancodes currently held, ordered by press time (oldest first).
    pub fn held_keys(&self) -> Vec<(u64, SDL_Scancode)> {
        self.state.keyboard.held_keys()
    }
    pub fn mouse_pos(&self) -> (f32, f32) {
        self.state.mouse.pos()
    }
    /// Mouse delta over the frame (sum of events; for aim/cursor).
    pub fn mouse_delta(&self) -> (f32, f32) {
        self.state.mouse.delta()
    }
    pub fn wheel(&self) -> (f32, f32) {
        self.state.mouse.wheel()
    }
    pub fn mouse_down(&self, button: u8) -> bool {
        self.state.mouse.mouse_down(button)
    }
    pub fn mouse_pressed(&self, button: u8) -> bool {
        self.state.mouse.mouse_pressed(button)
    }
    pub fn is_gamepad_connected(&self) -> bool {
        self.gamepad.is_some()
    }

    // --- action queries (pass through to the bindings) ---

    /// Action pressed THIS frame (edge).
    pub fn pressed(&self, a: ActionId) -> bool {
        self.bindings.pressed(a)
    }
    /// Action released THIS frame (edge).
    pub fn released(&self, a: ActionId) -> bool {
        self.bindings.released(a)
    }
    /// Action held down.
    pub fn down(&self, a: ActionId) -> bool {
        self.bindings.down(a)
    }
    /// Timestamp of the action's latest edge event (ns; 0 вЂ” no edge).
    pub fn edge_ts_ns(&self, a: ActionId) -> u64 {
        self.bindings.edge_ts_ns(a)
    }
    pub fn action_name(&self, a: ActionId) -> &'static str {
        self.bindings.action_name(a)
    }

    /// Latest edge event this frame: (action, timestamp). For the overlay.
    pub fn last_edge(&self) -> Option<(ActionId, u64)> {
        self.last_edge
    }
    pub fn events_last_poll(&self) -> u32 {
        self.events_last_poll
    }

    /// Release SDL handles before SDL_Quit. Idempotent: Drop calls it too,
    /// so a late drop after SDL_Quit is a no-op.
    pub fn close(&mut self) {
        if let Some(pad) = self.gamepad.take() {
            unsafe { SDL_CloseGamepad(pad.handle) };
        }
    }
}

impl Drop for InputCore {
    fn drop(&mut self) {
        self.close();
    }
}
