//! Action mapping: named actions bound to device buttons, with per-frame
//! computed states (`osu.Framework.Input.FrameworkActionContainer` analog).

use super::states::{InputState, MOUSE_BUTTONS, PAD_BUTTONS, SCANCODE_COUNT};
use sdl3_sys::everything::*;

/// Stable action id: index into [`ActionBindings::actions`].
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

/// Action: name + bindings.
pub struct ActionDef {
    pub name: &'static str,
    pub bindings: Vec<Binding>,
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

/// Registered actions and their per-frame states. The mapping layer between
/// raw device states and gameplay queries (`GetPressedActions` in osu).
pub struct ActionBindings {
    pub(crate) actions: Vec<ActionDef>,
    pub(crate) states: Vec<ActionState>,
}

impl ActionBindings {
    pub(crate) fn new() -> Self {
        Self {
            actions: Vec::new(),
            states: Vec::new(),
        }
    }

    /// Register an action. ActionId = registration order.
    pub(crate) fn map(&mut self, name: &'static str, bindings: &[Binding]) -> ActionId {
        let id = ActionId(self.actions.len());
        self.actions.push(ActionDef {
            name,
            bindings: bindings.to_vec(),
        });
        self.states.push(ActionState::default());
        id
    }

    /// Compute action states from binding levels. Once per frame.
    pub(crate) fn update(&mut self, state: &InputState) -> Option<(ActionId, u64)> {
        let mut last_edge: Option<(ActionId, u64)> = None;

        for (i, def) in self.actions.iter().enumerate() {
            let mut st = ActionState::default();
            for b in &def.bindings {
                let bin: &super::states::ButtonState = match b {
                    Binding::Key(sc) => &state.keyboard.keys[(*sc).0 as usize % SCANCODE_COUNT],
                    Binding::Mouse(btn) => &state.mouse.buttons[*btn as usize % MOUSE_BUTTONS],
                    Binding::Pad(btn) => {
                        let idx = ((*btn).0 as usize) % PAD_BUTTONS;
                        &state.joystick.buttons[idx]
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
            self.states[i] = st;
            if st.pressed || st.released {
                last_edge = Some((ActionId(i), st.ts_ns));
            }
        }

        last_edge
    }

    // --- action queries ---

    /// Action pressed THIS frame (edge).
    pub fn pressed(&self, a: ActionId) -> bool {
        self.states[a.0].pressed
    }
    /// Action released THIS frame (edge).
    pub fn released(&self, a: ActionId) -> bool {
        self.states[a.0].released
    }
    /// Action held down.
    pub fn down(&self, a: ActionId) -> bool {
        self.states[a.0].down
    }
    /// Timestamp of the action's latest edge event (ns; 0 — no edge).
    pub fn edge_ts_ns(&self, a: ActionId) -> u64 {
        self.states[a.0].ts_ns
    }
    pub fn action_name(&self, a: ActionId) -> &'static str {
        self.actions[a.0].name
    }
}
