//! TestApp game host (osu-framework style): owns the window, the gfx stack
//! and the frame loop (input -> update -> draw -> present). Tests live in
//! `screens`, immediate-mode UI in `ui`, palette/layout in `theme`.
//!
//! Exclusive fullscreen. Top bar, collapsible test list, actions panel and a
//! 16:9 game view where the running test scene renders (scissored).

use anyhow::{anyhow, Context, Result};
use std::time::Instant;

use ash::vk;

use vk5k::core::time::{FrameStats, WARMUP_MS};
use vk5k::gfx::device::GfxContext;
use vk5k::gfx::frames::FramesInFlight;
use vk5k::gfx::renderer::Renderer;
use vk5k::gfx::sprite::SpriteBatcher;
use vk5k::gfx::swapchain::{Acquire, Swapchain};
use vk5k::input::{Binding, InputCore};
use vk5k::platform::window::Window;
use vk5k::ui::text::MAX_VERTS;

use crate::screens::{SpriteBatchScreen, TestCase};
use crate::theme::{DESIGN_H, DESIGN_W};
use crate::ui::{self, Layout, UiState};

const FRAMES_IN_FLIGHT: usize = 8;
const STATS_WINDOW_MS: u64 = 250;

/// fixed-dt update stage (120 Hz)
const FIXED_DT: f64 = 1.0 / 120.0;

/// Frame signature: everything the idle UI's geometry depends on. When it
/// does not change and no scene runs, the sprite/text data and command
/// buffers are left untouched (zero re-record cost).
type UiSig = (
    u32,
    u32, // mouse x, y
    i64,
    u64, // fps text
    bool, // flip status
    bool, // ui.expanded
    Option<usize>, // ui.selected_test
    bool, // ui.running
    u32,
    u32, // framebuffer size
);

/// Run the test browser to completion. `Err` = fatal (logged by main).
pub unsafe fn run() -> Result<()> {
    // lock the GPU clock to the driver max, same as vk5d (needs admin;
    // failure is logged and non-fatal; reset on exit via Drop)
    let nvml = match vk5k::core::nvml::Nvml::init_and_lock_max() {
        Ok(n) => {
            vk5k::elog!("GPU: {} | clock locked at {} MHz", n.gpu_name, n.locked_mhz);
            Some(n)
        }
        Err(e) => {
            vk5k::elog!("NVML clock lock unavailable: {e}");
            None
        }
    };
    let result = Game::new()?.run();
    drop(nvml); // reset clocks on exit
    result
}

struct Game {
    ctx: GfxContext,
    swapchain: Swapchain,
    renderer: Renderer,
    sprites: SpriteBatcher,
    frames: FramesInFlight,
    input: InputCore,
    act_quit: vk5k::input::ActionId,
    act_shot: vk5k::input::ActionId,
    ui: UiState,
    screens: Vec<Box<dyn TestCase>>,
    stats: FrameStats,
    /// seconds since app start; drives the test scenes
    t_start: Instant,
    /// previous frame signature for idle-change detection
    last_sig: Option<UiSig>,
    /// TESTAPP_NOSPRITES read once at init (not per frame)
    no_sprites: bool,
    /// static display names of the tests (no per-frame allocation)
    screen_names: Vec<&'static str>,
    /// cached UI layer: instances after ui::draw and text verts (reused while
    /// a scene runs — the UI does not change frame to frame)
    ui_snap: Vec<f32>,
    ui_verts: Vec<vk5k::ui::text::Vertex>,
    /// last field: must be dropped AFTER ctx (the surface holds the hwnd)
    window: Window,
}

impl Game {
    /// Boot: window -> gfx context -> swapchain/renderer/sprites -> input/UI.
    unsafe fn new() -> Result<Self> {
        let window = Window::new("vk-testapp", 1920, 1080)?;
        let ctx = GfxContext::new(window.hwnd())?;
        let swapchain = Swapchain::new(&ctx)?;
        let mut renderer = Renderer::new(&ctx)?;
        renderer.create_swapchain_objects(&ctx, &swapchain)?;
        let sprites = SpriteBatcher::new(&ctx)?;
        let frames = FramesInFlight::new(ctx.device(), FRAMES_IN_FLIGHT);

        let mut input = InputCore::new();
        let act_quit = input.map(
            "quit",
            &[
                Binding::Key(sdl3_sys::everything::SDL_Scancode::ESCAPE),
                Binding::Pad(sdl3_sys::everything::SDL_GamepadButton::START),
            ],
        );
        let act_shot = input.map(
            "screenshot",
            &[Binding::Key(sdl3_sys::everything::SDL_Scancode::F12)],
        );

        let screens: Vec<Box<dyn TestCase>> = vec![Box::new(SpriteBatchScreen::new())];
        let screen_names: Vec<&'static str> = screens.iter().map(|t| t.name()).collect();

        vk5k::elog!(
            "testapp swapchain: {}x{} images={} present_mode={:?}",
            swapchain.extent.width,
            swapchain.extent.height,
            swapchain.image_count(),
            swapchain.present_mode
        );

        Ok(Self {
            // window declared last in the struct so it is dropped after ctx
            ctx,
            swapchain,
            renderer,
            sprites,
            frames,
            input,
            act_quit,
            act_shot,
            ui: UiState::from_env(),
            screens,
            stats: FrameStats::new(sdl3_sys::everything::SDL_GetTicks()),
            t_start: Instant::now(),
            last_sig: None,
            no_sprites: std::env::var("TESTAPP_NOSPRITES").is_ok(),
            screen_names,
            ui_snap: Vec::new(),
            ui_verts: Vec::new(),
            window,
        })
    }

    unsafe fn run(mut self) -> Result<()> {
        let auto_exit_ms: Option<u64> = std::env::var("BLACK_SCREEN_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|s| s * 1000);
        let start_ticks = sdl3_sys::everything::SDL_GetTicks();

        let mut running = true;
        let mut shown_fps: i64 = -1;
        let mut fps_ms: f64 = 0.0;
        let mut flip_lost = false;
        let mut last = Instant::now();
        let mut update_acc = 0.0f64;

        'main: while running {
            if !self.poll_input()? {
                break;
            }

            // update stage: fixed 120 Hz step, free-running render
            let now_inst = Instant::now();
            update_acc += (now_inst - last).as_secs_f64();
            last = now_inst;
            while update_acc >= FIXED_DT {
                if self.input.pressed(self.act_shot) {
                    self.renderer.request_screenshot();
                }
                update_acc -= FIXED_DT;
            }

            let mut need_recreate = false;

            let slot = self.frames.slot();
            let t0 = Instant::now();
            // bounded fence wait: an unsignaled fence means the device is
            // wedged — rebuild instead of hanging the window
            let mut fence_ok = false;
            for _ in 0..4 {
                match self.frames.try_ensure_idle(self.ctx.device(), slot, 250_000_000) {
                    Ok(true) => {
                        fence_ok = true;
                        break;
                    }
                    Ok(false) => {}
                    Err(e) => return Err(e),
                }
            }
            self.stats.t_wait += t0.elapsed().as_nanos();
            if !fence_ok {
                vk5k::elog!("frame fence did not signal in 1 s — full rebuild");
                self.full_rebuild()?;
                continue 'main;
            }

            // --- acquire ---
            let img_idx = self.acquire(slot, &mut need_recreate)?;
            if need_recreate {
                self.recreate()?;
                continue;
            }

            // --- draw: UI + the running test scene (skipped when idle and
            // nothing changed: sprite/text data and CBs stay valid) ---
            let (w, h) = self.framebuffer_size();
            let s = (w / DESIGN_W).min(h / DESIGN_H);
            let (mx, my) = (self.input.mouse_pos().0 / s, self.input.mouse_pos().1 / s);

            // one layout per frame: clicks and drawing share it
            let layout = Layout::new();
            self.clicks(&layout, mx, my);

            // active scene (None = idle): computed AFTER clicks, a click may
            // clear the selection and stop the scene
            let scene_idx = self
                .ui
                .selected_test
                .filter(|&i| i < self.screens.len() && self.ui.running);
            let scene_running = scene_idx.is_some();

            let sig: UiSig = (
                self.input.mouse_pos().0.to_bits(),
                self.input.mouse_pos().1.to_bits(),
                shown_fps,
                fps_ms.to_bits(),
                flip_lost,
                self.ui.expanded,
                self.ui.selected_test,
                self.ui.running,
                w.to_bits(),
                h.to_bits(),
            );
            let ui_changed = self.last_sig != Some(sig) || self.renderer.shot_pending();
            self.last_sig = Some(sig);

            if ui_changed {
                let status = format!(
                    "FPS {shown_fps}  {fps_ms:.2} MS  {}",
                    if flip_lost { "FLIP LOST" } else { "FLIP OK" }
                );
                let t_ui0 = Instant::now();
                self.sprites.begin();
                let verts = ui::draw(
                    &mut self.sprites,
                    &layout,
                    &self.ui,
                    &self.screen_names,
                    mx,
                    my,
                    (w, h),
                    &status,
                    scene_running,
                );
                self.stats.t_ui += t_ui0.elapsed().as_nanos();

                if verts.len() > MAX_VERTS {
                    return Err(anyhow!("ui frame too long: {} verts", verts.len()));
                }

                // cache the static UI layer (instances pushed so far share
                // the full-frame scissor section created by ui::draw)
                self.ui_verts.clear();
                self.ui_verts.extend_from_slice(&verts);
                self.ui_snap.clear();
                self.ui_snap.extend_from_slice(self.sprites.instances());

                let t0 = Instant::now();
                self.renderer.upload_verts(&self.ui_verts);
                // static UI: identical data in every ring slot, so old CBs
                // stay valid between updates
                for s_i in 0..FRAMES_IN_FLIGHT {
                    self.sprites.upload_slot(self.ctx.device(), s_i);
                }
                for i in 0..self.swapchain.image_count() {
                    self.renderer.mark_dirty(i);
                }
                self.stats.t_rec += t0.elapsed().as_nanos();
            }

            if scene_running {
                if !ui_changed {
                    // restore the cached UI layer (one section, full frame)
                    self.sprites.begin();
                    self.sprites
                        .push_cached(&self.ui_snap, [0.0, 0.0, w, h]);
                }
                if let Some(i) = scene_idx {
                    let screen = self.screens[i].as_mut();
                    let t = self.t_start.elapsed().as_secs_f32();
                    let t_sc0 = Instant::now();
                    screen.draw(&mut self.sprites, layout.game, s, (w, h), t);
                    self.stats.t_scene += t_sc0.elapsed().as_nanos();
                }
                // per-frame scene data: current slot only, every image's CB
                // is re-recorded with it
                let t0 = Instant::now();
                self.sprites.upload_slot(self.ctx.device(), slot);
                for i in 0..self.swapchain.image_count() {
                    self.renderer.mark_dirty(i);
                }
                self.stats.t_rec += t0.elapsed().as_nanos();
            }

            // --- record (only images whose CB is stale) ---
            let img_idx_u = img_idx as usize;
            let sprites_ref = if self.no_sprites { None } else { Some(&self.sprites) };
            if self.renderer.is_dirty(img_idx_u) {
                let t0 = Instant::now();
                self.renderer
                    .record_frame(self.ctx.device(), img_idx_u, slot, sprites_ref)?;
                self.stats.t_rec += t0.elapsed().as_nanos();
            }
            let done_sem = self.swapchain.done_sem(img_idx_u);

            self.submit(slot, done_sem, img_idx_u, &mut need_recreate)?;
            if need_recreate {
                self.recreate()?;
                continue;
            }

            // screenshot requested this frame: fence already guarantees the copy
            // finished only after submit; wait for it and write the file
            if self.renderer.shot_ready() {
                self.ctx.device().device_wait_idle().context("wait shot")?;
                let path = screenshot_path()?;
                let path_str = path.to_string_lossy().into_owned();
                self.renderer.write_png(&path_str)?;
                vk5k::elog!("screenshot saved: {}", path.display());
            }

            self.frames.advance();
            self.stats.end_frame();

            // --- stats every STATS_WINDOW_MS (after warmup) ---
            let now = sdl3_sys::everything::SDL_GetTicks();
            if now - self.stats.window_start >= STATS_WINDOW_MS {
                if now - start_ticks >= WARMUP_MS {
                    let window_ms = now - self.stats.window_start;
                    let (fps, ms) = self.stats.report(window_ms);
                    self.stats.window_start = now;
                    shown_fps = fps;
                    fps_ms = ms;
                    flip_lost = self.stats.flip_lost;
                } else {
                    self.stats.reset_window();
                    self.stats.window_start = now;
                }
            }

            if let Some(ms) = auto_exit_ms {
                if now - start_ticks >= ms {
                    running = false;
                }
            }
        }

        // --- teardown ---
        // tolerant: the device may have been lost during the session
        let _ = self.ctx.device().device_wait_idle();
        self.frames.on_idle();
        self.renderer.destroy(self.ctx.device());
        self.sprites.destroy(self.ctx.device());
        self.frames.destroy(self.ctx.device());
        self.swapchain.destroy(self.ctx.device());
        self.ctx.destroy();
        self.window.destroy();
        Ok(())
    }

    /// Drain the event queue once per frame. Returns false when the app
    /// should quit.
    fn poll_input(&mut self) -> Result<bool> {
        let t_ev0 = Instant::now();
        let ev = self.input.poll();
        self.stats.t_events += t_ev0.elapsed().as_nanos();
        Ok(!(ev.quit || self.input.pressed(self.act_quit)))
    }

    /// Acquire the next swapchain image. Sets `need_recreate` on OUT_OF_DATE
    /// and on device-lost (fullscreen switch); returns u32::MAX in that case.
    /// NOT_READY retries are bounded: if no image arrives in ~100 ms, recreate
    /// the swapchain instead of spinning forever (the window stops pumping).
    unsafe fn acquire(&mut self, slot: usize, need_recreate: &mut bool) -> Result<u32> {
        let img_sem = self.frames.image_semaphore(slot);
        let mut img_idx = u32::MAX;
        let deadline = Instant::now() + std::time::Duration::from_millis(100);
        loop {
            self.stats.acq_calls += 1;
            let t0 = Instant::now();
            match self.swapchain.acquire(self.ctx.device(), img_sem)? {
                Acquire::Image(idx) => {
                    self.stats.t_acq += t0.elapsed().as_nanos();
                    img_idx = idx;
                    break;
                }
                Acquire::NotReady => {
                    self.stats.acq_retries += 1;
                    if Instant::now() > deadline {
                        vk5k::elog!(
                            "acquire: no image for 100 ms ({} retries) — recreating",
                            self.stats.acq_retries
                        );
                        *need_recreate = true;
                        break;
                    }
                }
                Acquire::OutOfDate => {
                    self.stats.t_acq += t0.elapsed().as_nanos();
                    *need_recreate = true;
                    break;
                }
                Acquire::DeviceLost => {
                    self.stats.t_acq += t0.elapsed().as_nanos();
                    vk5k::elog!("acquire: device lost (fullscreen switch) — recreating");
                    *need_recreate = true;
                    break;
                }
            }
        }
        Ok(img_idx)
    }

    /// Click handling: buttons, test rows; empty space clears the selection
    /// and stops the scene.
    fn clicks(&mut self, layout: &Layout, mx: f32, my: f32) {
        if !self.input.mouse_pressed(1) {
            return;
        }
        let mut handled = false;
        if layout.shot_btn.hit(mx, my) {
            self.renderer.request_screenshot();
            handled = true;
        } else if layout.header.hit(mx, my) {
            self.ui.expanded = !self.ui.expanded;
            handled = true;
        } else if self.ui.expanded {
            if let Some(i) = (0..self.screens.len()).find(|&i| layout.row(i).hit(mx, my)) {
                self.ui.selected_test = Some(i);
                handled = true;
            }
        }
        if self.ui.selected_test.is_some() {
            if layout.run_btn.hit(mx, my) {
                self.ui.running = true;
                handled = true;
            } else if layout.stop_btn.hit(mx, my) {
                self.ui.running = false;
                handled = true;
            }
        }
        // clicks inside the game view never clear the selection
        if !handled && !layout.game.hit(mx, my) {
            self.ui.selected_test = None;
            self.ui.running = false;
        }
    }

    /// Submit the frame; device-lost on submit is recovered like a fullscreen
    /// switch (the driver reports lost transiently) unless vkDeviceWaitIdle
    /// also fails — then the device is really gone.
    unsafe fn submit(
        &mut self,
        slot: usize,
        done_sem: vk::Semaphore,
        img_idx_u: usize,
        need_recreate: &mut bool,
    ) -> Result<()> {
        self.frames.reset_for_submit(self.ctx.device(), slot)?;
        let t0 = Instant::now();
        let submit = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: self.frames.image_semaphore_ref(slot),
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: self.renderer.command_buffer(img_idx_u),
            signal_semaphore_count: 1,
            p_signal_semaphores: &done_sem,
            ..Default::default()
        };
        let result = self
            .ctx
            .device()
            .queue_submit(self.ctx.queue, &[submit], self.frames.fence(slot));
        if let Err(vk::Result::ERROR_DEVICE_LOST) = result {
            // NVIDIA reports transient device lost during exclusive fullscreen
            // switches. Try the cheap path (swapchain recreation) first;
            // if vkDeviceWaitIdle also reports lost, recreate() escalates to
            // a full device-level rebuild.
            vk5k::elog!("submit: device lost (fullscreen switch) — recreating");
            self.frames.on_idle(); // mark fences "logically signaled" before rebuild
            *need_recreate = true;
            return Ok(());
        }
        result.context("queue submit")?;
        self.stats.t_submit += t0.elapsed().as_nanos();

        // --- present ---
        let t0 = Instant::now();
        let img_idx = img_idx_u as u32;
        let present_result = self.swapchain.present(self.ctx.queue, img_idx, done_sem);
        self.stats.t_present += t0.elapsed().as_nanos();
        if let Err(vk::Result::ERROR_OUT_OF_DATE_KHR) = present_result {
            *need_recreate = true;
        } else if let Err(vk::Result::ERROR_DEVICE_LOST) = present_result {
            vk5k::elog!("present: device lost (fullscreen switch) — recreating");
            *need_recreate = true;
        } else if let Err(e) = present_result {
            return Err(anyhow!("present: {e:?}"));
        }
        Ok(())
    }

    fn framebuffer_size(&self) -> (f32, f32) {
        let ext = self.swapchain.extent;
        (ext.width as f32, ext.height as f32)
    }

    /// Recreate the swapchain: idle -> destroy -> create -> rebuild views/CBs.
    /// If the device itself is lost (wait_idle fails), escalate to a full
    /// device-level rebuild.
    unsafe fn recreate(&mut self) -> Result<()> {
        if self.ctx.device().device_wait_idle().is_err() {
            return self.full_rebuild();
        }
        // after a fullscreen switch the device may report lost — do not fail here
        let _ = self.ctx.device().device_wait_idle();
        self.frames.on_idle();
        self.renderer.destroy_swapchain_objects(self.ctx.device());
        self.swapchain.destroy(self.ctx.device());
        self.swapchain = Swapchain::new(&self.ctx)?;
        self.renderer.create_swapchain_objects(&self.ctx, &self.swapchain)?;
        vk5k::elog!(
            "swapchain recreated: {}x{}",
            self.swapchain.extent.width,
            self.swapchain.extent.height
        );
        Ok(())
    }

    /// Full rebuild after a REAL device loss: destroy every device-level
    /// object, recreate the VkDevice and rebuild the whole gfx stack.
    /// Retries briefly — the lost state during a fullscreen switch can take
    /// a moment to clear.
    unsafe fn full_rebuild(&mut self) -> Result<()> {
        vk5k::elog!("device lost — full rebuild");
        // best-effort destruction on a lost device
        self.renderer.destroy(self.ctx.device());
        self.sprites.destroy(self.ctx.device());
        self.frames.destroy(self.ctx.device());
        self.swapchain.destroy(self.ctx.device());
        for attempt in 1..=10u32 {
            match self.rebuild_device_stack() {
                Ok(()) => return Ok(()),
                Err(e) if attempt < 10 => {
                    vk5k::elog!("rebuild attempt {attempt} failed: {e:#}; retrying");
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("retry loop returns explicitly")
    }

    unsafe fn rebuild_device_stack(&mut self) -> Result<()> {
        self.ctx.recreate_device()?;
        self.frames = FramesInFlight::new(self.ctx.device(), FRAMES_IN_FLIGHT);
        self.sprites = SpriteBatcher::new(&self.ctx)?;
        self.swapchain = Swapchain::new(&self.ctx)?;
        self.renderer = Renderer::new(&self.ctx)?;
        self.renderer.create_swapchain_objects(&self.ctx, &self.swapchain)?;
        Ok(())
    }
}

/// Screenshots go to `<exe dir>/Screenshots/testapp_<n>.png`; `n` is the
/// first free number so files never overwrite each other.
fn screenshot_path() -> Result<std::path::PathBuf> {
    let exe = std::env::current_exe().context("current_exe")?;
    let dir = exe
        .parent()
        .ok_or_else(|| anyhow!("no exe dir"))?
        .join("Screenshots");
    std::fs::create_dir_all(&dir).context("create Screenshots dir")?;
    let mut n = 1u32;
    loop {
        let p = dir.join(format!("testapp_{n:04}.png"));
        if !p.exists() {
            return Ok(p);
        }
        n += 1;
    }
}
