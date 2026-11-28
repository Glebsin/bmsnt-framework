//! Input core: action mapping, edge events,
//! per-event timestamps in SDL monotonic ns.
//!
//! Low-latency principles:
//! - events are drained EVERY frame, before update/render; a key pressed
//!   mid-frame is handled at the start of the next frame;
//! - keyboard and mouse are event-driven (SDL_EVENT_KEY_DOWN / MOUSE_BUTTON_DOWN,
//!   repeats ignored); gamepad buttons are events too (gbutton);
//!   gamepad axes/triggers are polled once per frame (axes have no events);
//! - SDL3 gives every event a nanosecond timestamp — edges keep it, so the
//!   engine can measure input-to-display latency;
//! - no allocations or HashMap in the poll loop: state lives in flat arrays
//!   indexed by scancode / button index.

use sdl3_sys::everything::*;

/// number of SDL3 scancode slots
const SCANCODE_COUNT: usize = SDL_Scancode::COUNT.0 as usize;
/// mouse button slots; SDL indexes 1..=5 (left, middle, right, x1, x2)
const MOUSE_BUTTONS: usize = 6;
/// SDL_GamepadButton::COUNT is not const-friendly; all standard buttons +
/// paddles + misc fit under 24
const PAD_BUTTONS: usize = 24;

/// Stable action id: index into `InputCore::actions`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ActionId(pub usize);

/// How an action is bound.
#[derive(Clone, Copy)]
pub enum Binding {
    /// physical key (scancode — layout independent)
    Key(SDL_Scancode),
    /// mouse button: 1=left, 2=middle, 3=right, 4=x1, 5=x2
    Mouse(u8),
    /// gamepad button in the standard SDL scheme (SOUTH/EAST/...)
    Pad(SDL_GamepadButton),
}

/// State of one binary input (key/button): level + this-frame edge.
#[derive(Clone, Copy, Default)]
struct Binary {
    down: bool,
    /// pressed this frame (true for exactly one frame)
    pressed: bool,
    /// released this frame
    released: bool,
    /// timestamp of the latest edge event (ns, SDL_GetTicksNS basis)
    ts_ns: u64,
}

impl Binary {
    fn clear_edges(&mut self) {
        self.pressed = false;
        self.released = false;
    }
}

/// Gamepad state: button levels + per-frame edges.
#[derive(Clone, Copy, Default)]
struct PadState {
    level: [bool; PAD_BUTTONS],
    pressed: [bool; PAD_BUTTONS],
    released: [bool; PAD_BUTTONS],
    ts_ns: [u64; PAD_BUTTONS],
}

/// Flat keyboard storage.
struct Keyboard {
    keys: Vec<Binary>,
}
impl Keyboard {
    fn new() -> Self {
        Self {
            keys: vec![Binary::default(); SCANCODE_COUNT],
        }
    }
    fn get(&mut self, sc: SDL_Scancode) -> &mut Binary {
        &mut self.keys[(sc.0 as usize) % SCANCODE_COUNT]
    }
}

struct Mouse {
    x: f32,
    y: f32,
    /// delta accumulated over the frame (sum of xrel/yrel events)
    dx: f32,
    dy: f32,
    wheel_x: f32,
    wheel_y: f32,
    buttons: Vec<Binary>,
}

/// Action: name + bindings.
struct ActionDef {
    name: &'static str,
    bindings: Vec<Binding>,
}

/// Per-frame action state.
#[derive(Clone, Copy, Default)]
pub struct ActionState {
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
    /// timestamp of the freshest edge event among bindings (ns)
    pub ts_ns: u64,
}

/// Open gamepad.
struct Gamepad {
    handle: *mut SDL_Gamepad,
    instance_id: SDL_JoystickID,
}

/// Per-frame poll result.
#[derive(Default)]
pub struct FrameEvents {
    pub quit: bool,
}

/// Input core. One instance per app; call `poll()` once at the start of each frame.
pub struct InputCore {
    keyboard: Keyboard,
    mouse: Mouse,
    pad: Option<PadState>,
    gamepad: Option<Gamepad>,
    actions: Vec<ActionDef>,
    action_states: Vec<ActionState>,
    /// latest edge event for the debug overlay: (action, timestamp)
    last_edge: Option<(ActionId, u64)>,
    /// events consumed in the last poll (diagnostics)
    events_last_poll: u32,
}

impl InputCore {
    pub fn new() -> Self {
        Self {
            keyboard: Keyboard::new(),
            mouse: Mouse {
                x: 0.0,
                y: 0.0,
                dx: 0.0,
                dy: 0.0,
                wheel_x: 0.0,
                wheel_y: 0.0,
                buttons: vec![Binary::default(); MOUSE_BUTTONS],
            },
            pad: None,
            gamepad: None,
            actions: Vec::new(),
            action_states: Vec::new(),
            last_edge: None,
            events_last_poll: 0,
        }
    }

    /// Register an action. ActionId = registration order.
    pub fn map(&mut self, name: &'static str, bindings: &[Binding]) -> ActionId {
        let id = ActionId(self.actions.len());
        self.actions.push(ActionDef {
            name,
            bindings: bindings.to_vec(),
        });
        self.action_states.push(ActionState::default());
        id
    }

    /// Drain the SDL3 event queue. Call exactly once per frame, before update
    /// and render.
    pub fn poll(&mut self) -> FrameEvents {
        let mut ev = FrameEvents::default();

        // edges live for exactly one frame
        self.last_edge = None;
        for b in self.keyboard.keys.iter_mut() {
            b.clear_edges();
        }
        for b in self.mouse.buttons.iter_mut() {
            b.clear_edges();
        }
        self.mouse.dx = 0.0;
        self.mouse.dy = 0.0;
        self.mouse.wheel_x = 0.0;
        self.mouse.wheel_y = 0.0;
        if let Some(pad) = &mut self.pad {
            pad.pressed = [false; PAD_BUTTONS];
            pad.released = [false; PAD_BUTTONS];
        }

        let mut event: SDL_Event = unsafe { std::mem::zeroed() };
        let mut eaten = 0u32;
        unsafe {
            while SDL_PollEvent(&mut event) {
                eaten += 1;
                match event.r#type {
                    t if t == SDL_EventType::QUIT.0 => {
                        ev.quit = true;
                    }
                    // keyboard: event-driven, repeats ignored
                    t if t == SDL_EventType::KEY_DOWN.0 => {
                        let k = &event.key;
                        if !k.repeat {
                            let b = self.keyboard.get(k.scancode);
                            b.down = true;
                            b.pressed = true;
                            b.ts_ns = k.timestamp;
                        }
                    }
                    t if t == SDL_EventType::KEY_UP.0 => {
                        let k = &event.key;
                        let b = self.keyboard.get(k.scancode);
                        b.down = false;
                        b.released = true;
                        b.ts_ns = k.timestamp;
                    }
                    // mouse: motion deltas accumulate over the frame
                    t if t == SDL_EventType::MOUSE_MOTION.0 => {
                        let m = &event.motion;
                        self.mouse.x = m.x;
                        self.mouse.y = m.y;
                        self.mouse.dx += m.xrel;
                        self.mouse.dy += m.yrel;
                    }
                    t if t == SDL_EventType::MOUSE_BUTTON_DOWN.0 => {
                        let m = &event.button;
                        let b = &mut self.mouse.buttons[(m.button as usize) % MOUSE_BUTTONS];
                        b.down = true;
                        b.pressed = true;
                        b.ts_ns = m.timestamp;
                    }
                    t if t == SDL_EventType::MOUSE_BUTTON_UP.0 => {
                        let m = &event.button;
                        let b = &mut self.mouse.buttons[(m.button as usize) % MOUSE_BUTTONS];
                        b.down = false;
                        b.released = true;
                        b.ts_ns = m.timestamp;
                    }
                    t if t == SDL_EventType::MOUSE_WHEEL.0 => {
                        let w = &event.wheel;
                        self.mouse.wheel_x += w.x;
                        self.mouse.wheel_y += w.y;
                    }
                    // gamepad hot-plug
                    t if t == SDL_EventType::GAMEPAD_ADDED.0 => {
                        let id = event.gdevice.which;
                        if self.gamepad.is_none() && SDL_IsGamepad(id) {
                            let handle = SDL_OpenGamepad(id);
                            if !handle.is_null() {
                                crate::elog!("gamepad connected (id {})", id.0);
                                self.pad = Some(PadState::default());
                                self.gamepad = Some(Gamepad {
                                    handle,
                                    instance_id: id,
                                });
                            }
                        }
                    }
                    t if t == SDL_EventType::GAMEPAD_REMOVED.0 => {
                        if let Some(pad) = &self.gamepad {
                            if pad.instance_id == event.gdevice.which {
                                crate::elog!("gamepad disconnected");
                                SDL_CloseGamepad(pad.handle);
                                self.gamepad = None;
                                self.pad = None;
                            }
                        }
                    }
                    // gamepad buttons: events, like the keyboard
                    t if t == SDL_EventType::GAMEPAD_BUTTON_DOWN.0 => {
                        let g = &event.gbutton;
                        if let Some(pad) = &mut self.pad {
                            let i = (g.button as usize) % PAD_BUTTONS;
                            pad.level[i] = true;
                            pad.pressed[i] = true;
                            pad.ts_ns[i] = g.timestamp;
                        }
                    }
                    t if t == SDL_EventType::GAMEPAD_BUTTON_UP.0 => {
                        let g = &event.gbutton;
                        if let Some(pad) = &mut self.pad {
                            let i = (g.button as usize) % PAD_BUTTONS;
                            pad.level[i] = false;
                            pad.released[i] = true;
                            pad.ts_ns[i] = g.timestamp;
                        }
                    }
                    _ => {}
                }
            }
        }
        self.events_last_poll = eaten;

        // sync gamepad button levels with the driver (SDL_GetGamepadButton):
        // events may be missed on desync; axes/triggers would be polled here too
        if let (Some(pad), Some(gamepad)) = (&mut self.pad, &self.gamepad) {
            for i in 0..PAD_BUTTONS {
                let btn = SDL_GamepadButton(i as ::core::ffi::c_int);
                if unsafe { SDL_GetGamepadButton(gamepad.handle, btn) } {
                    if !pad.level[i] {
                        // pressed without an event (or the event arrives next frame)
                        pad.level[i] = true;
                        pad.pressed[i] = true;
                        pad.ts_ns[i] = unsafe { SDL_GetTicksNS() };
                    }
                } else if pad.level[i] {
                    pad.level[i] = false;
                    pad.released[i] = true;
                    pad.ts_ns[i] = unsafe { SDL_GetTicksNS() };
                }
            }
        }

        self.compute_actions();
        ev
    }

    /// Compute action states from binding levels. Once per frame.
    fn compute_actions(&mut self) {
        // defer last_edge update to avoid a borrow conflict
        let mut edge_seen: Option<(ActionId, u64)> = None;

        for (i, def) in self.actions.iter().enumerate() {
            let mut st = ActionState::default();
            for b in &def.bindings {
                let bin: &Binary = match b {
                    Binding::Key(sc) => &self.keyboard.keys[(sc.0 as usize) % SCANCODE_COUNT],
                    Binding::Mouse(btn) => &self.mouse.buttons[(*btn as usize) % MOUSE_BUTTONS],
                    Binding::Pad(btn) => {
                        if let Some(pad) = &self.pad {
                            let idx = ((*btn).0 as usize) % PAD_BUTTONS;
                            if pad.pressed[idx] {
                                st.pressed = true;
                                st.ts_ns = st.ts_ns.max(pad.ts_ns[idx]);
                            }
                            if pad.released[idx] {
                                st.released = true;
                                st.ts_ns = st.ts_ns.max(pad.ts_ns[idx]);
                            }
                            if pad.level[idx] {
                                st.down = true;
                            }
                        }
                        continue;
                    }
                };
                if bin.pressed {
                    st.pressed = true;
                    st.ts_ns = st.ts_ns.max(bin.ts_ns);
                }
                if bin.released {
                    st.released = true;
                    st.ts_ns = st.ts_ns.max(bin.ts_ns);
                }
                if bin.down {
                    st.down = true;
                }
            }
            self.action_states[i] = st;
            if st.pressed || st.released {
                edge_seen = Some((ActionId(i), st.ts_ns));
            }
        }

        if let Some(e) = edge_seen {
            self.last_edge = Some(e);
        }
    }

    // --- action queries ---

    /// Action pressed THIS frame (edge).
    pub fn pressed(&self, a: ActionId) -> bool {
        self.action_states[a.0].pressed
    }
    /// Action released THIS frame (edge).
    pub fn released(&self, a: ActionId) -> bool {
        self.action_states[a.0].released
    }
    /// Action held down.
    pub fn down(&self, a: ActionId) -> bool {
        self.action_states[a.0].down
    }
    /// Timestamp of the action's latest edge event (ns; 0 — no edge).
    pub fn edge_ts_ns(&self, a: ActionId) -> u64 {
        self.action_states[a.0].ts_ns
    }
    pub fn action_name(&self, a: ActionId) -> &'static str {
        self.actions[a.0].name
    }

    // --- direct device queries ---

    pub fn key_down(&self, sc: SDL_Scancode) -> bool {
        self.keyboard.keys[(sc.0 as usize) % SCANCODE_COUNT].down
    }
    pub fn key_pressed(&self, sc: SDL_Scancode) -> bool {
        self.keyboard.keys[(sc.0 as usize) % SCANCODE_COUNT].pressed
    }
    /// Scancodes currently held, ordered by press time (oldest first).
    /// Timestamps are the SDL event ns of the latest edge per key.
    pub fn held_keys(&self) -> Vec<(u64, SDL_Scancode)> {
        let mut v: Vec<(u64, SDL_Scancode)> = self
            .keyboard
            .keys
            .iter()
            .enumerate()
            .filter(|(_, b)| b.down)
            .map(|(i, b)| (b.ts_ns, SDL_Scancode(i as i32)))
            .collect();
        v.sort_by_key(|(ts, _)| *ts);
        v
    }
    pub fn mouse_pos(&self) -> (f32, f32) {
        (self.mouse.x, self.mouse.y)
    }
    /// Mouse delta over the frame (sum of events; for aim/cursor).
    pub fn mouse_delta(&self) -> (f32, f32) {
        (self.mouse.dx, self.mouse.dy)
    }
    pub fn wheel(&self) -> (f32, f32) {
        (self.mouse.wheel_x, self.mouse.wheel_y)
    }
    pub fn mouse_down(&self, button: u8) -> bool {
        self.mouse.buttons[(button as usize) % MOUSE_BUTTONS].down
    }
    pub fn mouse_pressed(&self, button: u8) -> bool {
        self.mouse.buttons[(button as usize) % MOUSE_BUTTONS].pressed
    }
    pub fn is_gamepad_connected(&self) -> bool {
        self.gamepad.is_some()
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
