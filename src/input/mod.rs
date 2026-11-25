//! Ядро инпута (Фаза 2.1) — в духе osu!lazer: action-mapping, edge-события,
//! таймстампы каждого события в наносекундах SDL-монотонных тиков.
//!
//! Принципы минимальной задержки:
//! - события опрашиваются КАЖДЫЙ кадр, до update/рендера (никакой очереди
//!   «раз в тик»); кнопка, пришедшая в середине кадра, обрабатывается в его начале;
//! - клавиатура и мышь — event-driven (SDL_EVENT_KEY_DOWN / MOUSE_BUTTON_DOWN,
//!   повторы игнорируются), геймпад-кнопки — тоже события (gbutton);
//!   оси/триггеры геймпада — поллинг состояния раз в кадр (у осей нет событий);
//! - SDL3 даёт timestamp каждого события в наносекундах — для edge-состояний
//!   он сохраняется, движок может мерить input-to-display латентность;
//! - никаких аллокаций и HashMap в опросе: состояния в плоских массивах
//!   по scancode / индексу кнопки.

use sdl3_sys::everything::*;

/// Число scancode-слотов SDL3.
const SCANCODE_COUNT: usize = SDL_Scancode::COUNT.0 as usize;
/// Слоты кнопок мыши: SDL индексирует 1..=5 (left, middle, right, x1, x2).
const MOUSE_BUTTONS: usize = 6;
/// SDL_GamepadButton::COUNT не const-friendly — берём запас сверху
/// (все стандартные кнопки + paddles + misc < 24).
const PAD_BUTTONS: usize = 24;

/// Стабильный идентификатор действия. Индекс в `InputCore::actions`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ActionId(pub usize);

/// Чем биндится действие.
#[derive(Clone, Copy)]
pub enum Binding {
    /// Физическая клавиша (scancode — независимо от раскладки).
    Key(SDL_Scancode),
    /// Кнопка мыши: 1=left, 2=middle, 3=right, 4=x1, 5=x2.
    Mouse(u8),
    /// Кнопка геймпада по стандартной схеме SDL (SOUTH/EAST/...).
    Pad(SDL_GamepadButton),
}

/// Состояние одного бинара (клавиша/кнопка): уровень + edge этого кадра.
#[derive(Clone, Copy, Default)]
struct Binary {
    down: bool,
    /// edge «нажато в этом кадре» (true ровно один кадр)
    pressed: bool,
    /// edge «отпущено в этом кадре»
    released: bool,
    /// timestamp события последнего edge (ns, базис SDL_GetTicksNS)
    ts_ns: u64,
}

impl Binary {
    fn clear_edges(&mut self) {
        self.pressed = false;
        self.released = false;
    }
}

/// Состояние геймпада: уровни + edge-события кнопок за кадр.
#[derive(Clone, Copy, Default)]
struct PadState {
    level: [bool; PAD_BUTTONS],
    pressed: [bool; PAD_BUTTONS],
    released: [bool; PAD_BUTTONS],
    ts_ns: [u64; PAD_BUTTONS],
}

/// Плоское хранилище клавиатуры.
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
    /// накопленная дельта между кадрами (сумма всех xrel/yrel за кадр)
    dx: f32,
    dy: f32,
    wheel_x: f32,
    wheel_y: f32,
    buttons: Vec<Binary>,
}

/// Действие: имя + биндинги.
struct ActionDef {
    name: &'static str,
    bindings: Vec<Binding>,
}

/// Состояние действия за кадр.
#[derive(Clone, Copy, Default)]
pub struct ActionState {
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
    /// timestamp самого свежего edge-события среди биндингов (ns)
    pub ts_ns: u64,
}

/// Открытый геймпад.
struct Gamepad {
    handle: *mut SDL_Gamepad,
    instance_id: SDL_JoystickID,
}

/// Результат опроса событий за кадр.
#[derive(Default)]
pub struct FrameEvents {
    pub quit: bool,
}

/// Ядро инпута. Один экземпляр на приложение; `poll()` — в начале каждого кадра.
pub struct InputCore {
    keyboard: Keyboard,
    mouse: Mouse,
    pad: Option<PadState>,
    gamepad: Option<Gamepad>,
    actions: Vec<ActionDef>,
    action_states: Vec<ActionState>,
    /// последнее edge-событие для отладочного оверлея: (действие, ts)
    last_edge: Option<(ActionId, u64)>,
    /// сколько событий съедено за последний кадр (диагностика)
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

    /// Регистрирует действие. ActionId = порядковый номер регистрации.
    pub fn map(&mut self, name: &'static str, bindings: &[Binding]) -> ActionId {
        let id = ActionId(self.actions.len());
        self.actions.push(ActionDef {
            name,
            bindings: bindings.to_vec(),
        });
        self.action_states.push(ActionState::default());
        id
    }

    /// Опрос событий SDL3: вызывать строго один раз в начале кадра,
    /// до update и рендера. Полностью осушает очередь событий.
    pub fn poll(&mut self) -> FrameEvents {
        let mut ev = FrameEvents::default();

        // edges живут ровно один кадр
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
                    // клавиатура: event-driven, повторы игнорируем
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
                    // мышь: движение суммируем в дельту кадра
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
                    // геймпад: подключение/отключение
                    t if t == SDL_EventType::GAMEPAD_ADDED.0 => {
                        let id = event.gdevice.which;
                        if self.gamepad.is_none() && SDL_IsGamepad(id) {
                            let handle = SDL_OpenGamepad(id);
                            if !handle.is_null() {
                                crate::elog!("gamepad подключён (id {})", id.0);
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
                                crate::elog!("gamepad отключён");
                                SDL_CloseGamepad(pad.handle);
                                self.gamepad = None;
                                self.pad = None;
                            }
                        }
                    }
                    // геймпад-кнопки: события, как клавиатура
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

        // синхронизация уровня геймпада с драйвером (GetGamepadButton):
        // события могли потеряться при рассинхроне, оси/триггеры поллим здесь же
        if let (Some(pad), Some(gamepad)) = (&mut self.pad, &self.gamepad) {
            for i in 0..PAD_BUTTONS {
                let btn = SDL_GamepadButton((i as ::core::ffi::c_int));
                if unsafe { SDL_GetGamepadButton(gamepad.handle, btn) } {
                    if !pad.level[i] {
                        // нажали без события (или событие приедет в след. кадре)
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

    /// Вычисление состояний действий из уровней биндингов. Один раз за кадр.
    fn compute_actions(&mut self) {
        // отложенные апдейты last_edge, чтобы не конфликтовать по borrow
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

    // --- запросы действий ---

    /// Действие нажато в ЭТОМ кадре (edge).
    pub fn pressed(&self, a: ActionId) -> bool {
        self.action_states[a.0].pressed
    }
    /// Действие отпущено в ЭТОМ кадре (edge).
    pub fn released(&self, a: ActionId) -> bool {
        self.action_states[a.0].released
    }
    /// Действие удерживается.
    pub fn down(&self, a: ActionId) -> bool {
        self.action_states[a.0].down
    }
    /// Timestamp последнего edge-события действия (ns; 0 — edge не было).
    pub fn edge_ts_ns(&self, a: ActionId) -> u64 {
        self.action_states[a.0].ts_ns
    }
    pub fn action_name(&self, a: ActionId) -> &'static str {
        self.actions[a.0].name
    }

    // --- запросы устройств напрямую ---

    pub fn key_down(&self, sc: SDL_Scancode) -> bool {
        self.keyboard.keys[(sc.0 as usize) % SCANCODE_COUNT].down
    }
    pub fn key_pressed(&self, sc: SDL_Scancode) -> bool {
        self.keyboard.keys[(sc.0 as usize) % SCANCODE_COUNT].pressed
    }
    pub fn mouse_pos(&self) -> (f32, f32) {
        (self.mouse.x, self.mouse.y)
    }
    /// Дельта мыши за кадр (сумма событий за кадр; для aim/cursor).
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

    /// Последнее edge-событие за кадр: (действие, timestamp). Для оверлея.
    pub fn last_edge(&self) -> Option<(ActionId, u64)> {
        self.last_edge
    }
    pub fn events_last_poll(&self) -> u32 {
        self.events_last_poll
    }
}

impl Drop for InputCore {
    fn drop(&mut self) {
        if let Some(pad) = &self.gamepad {
            unsafe { SDL_CloseGamepad(pad.handle) };
        }
    }
}
