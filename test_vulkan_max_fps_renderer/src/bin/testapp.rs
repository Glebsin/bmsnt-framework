//! TestApp: test browser for engine features (Phase 2 proving ground).
//!
//! Exclusive fullscreen. Layout (design 1920x1080, uniformly scaled):
//! - top bar (#383838): title, SCREENSHOT button, fps status;
//! - left panel (#242424): collapsible test case list;
//! - actions panel (#1f1f1f): real actions of the selected test (RUN/STOP);
//! - game view: 16:9 rect on the right; test scenes render only inside it
//!   (scissored sections in the sprite batcher).
//!
//! The "SpriteBatch" test = 1200 moving rounded sprites (target 1% low < 1 ms).

// no console: a visible console forces DWM composition instead of flip
#![windows_subsystem = "windows"]

use anyhow::{anyhow, Context, Result};
use std::time::Instant;

use ash::vk;

use vk5k::core::time::{FrameStats, WARMUP_MS};
use vk5k::gfx::device::GfxContext;
use vk5k::gfx::frames::FramesInFlight;
use vk5k::gfx::renderer::Renderer;
use vk5k::gfx::sprite::SpriteBatcher;
use vk5k::gfx::swapchain::Swapchain;
use vk5k::input::{Binding, InputCore};
use vk5k::platform::window::Window;
use vk5k::ui::text::{build_text, Vertex, MAX_VERTS};

const STATS_WINDOW_MS: u64 = 250;
const FRAMES_IN_FLIGHT: usize = 8;

// --- palette (hex values from the spec) ---
const COL_TOP_BAR: (f32, f32, f32) = (0x38 as f32 / 255.0, 0x38 as f32 / 255.0, 0x38 as f32 / 255.0);
const COL_SIDEBAR: (f32, f32, f32) = (0x24 as f32 / 255.0, 0x24 as f32 / 255.0, 0x24 as f32 / 255.0);
const COL_ACTIONS: (f32, f32, f32) = (0x1f as f32 / 255.0, 0x1f as f32 / 255.0, 0x1f as f32 / 255.0);
/// gradient endpoints around #161616: lighter at the top, darker at the bottom
const COL_GRAD_TOP: (f32, f32, f32) = (0x1e as f32 / 255.0, 0x1e as f32 / 255.0, 0x1e as f32 / 255.0);
const COL_GRAD_BOTTOM: (f32, f32, f32) = (0x0e as f32 / 255.0, 0x0e as f32 / 255.0, 0x0e as f32 / 255.0);

const COL_ROW: (f32, f32, f32) = (0x2a as f32 / 255.0, 0x2a as f32 / 255.0, 0x2a as f32 / 255.0);
const COL_ROW_HOVER: (f32, f32, f32) = (0x32 as f32 / 255.0, 0x32 as f32 / 255.0, 0x32 as f32 / 255.0);
const COL_ROW_SEL: (f32, f32, f32) = (0x3a as f32 / 255.0, 0x3a as f32 / 255.0, 0x3a as f32 / 255.0);
const COL_ACCENT: (f32, f32, f32) = (0.15, 0.75, 0.60);
const COL_TEXT: (f32, f32, f32) = (0.92, 0.92, 0.92);
const COL_TEXT_DIM: (f32, f32, f32) = (0.55, 0.55, 0.55);
const COL_GAME_BG: (f32, f32, f32) = (0.035, 0.035, 0.042);
const COL_BORDER: (f32, f32, f32) = (0.13, 0.13, 0.13);

/// test cases (real ones only); SpriteBatch is the batcher stress scene
const TESTS: [&str; 1] = ["SpriteBatch"];

/// stress scene: moving rounded sprites
const STRESS_SPRITES: usize = 1200;

/// layout is designed at 1920x1080 and scaled to the real viewport
const DESIGN_W: f32 = 1920.0;
const DESIGN_H: f32 = 1080.0;
const TOP_BAR_H: f32 = 52.0;
const SIDEBAR_W: f32 = 240.0;
const ACTIONS_W: f32 = 220.0;
const ROW_H: f32 = 38.0;

struct UiState {
    expanded: bool,
    selected_test: Option<usize>,
    running: bool,
}

struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
impl Rect {
    fn hit(&self, mx: f32, my: f32) -> bool {
        mx >= self.x && mx <= self.x + self.w && my >= self.y && my <= self.y + self.h
    }
}

fn main() -> Result<()> {
    vk5k::init_log("testapp.log");

    // high priority: fewer scheduling jitters
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS,
        };
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
    // log fatal errors: GUI-subsystem has no console, an unlogged error
    // means a silent death with an empty-looking log
    unsafe {
        if let Err(e) = run() {
            vk5k::elog!("FATAL: {e:?}");
            std::process::exit(1);
        }
    }
    Ok(())
}

unsafe fn run() -> Result<()> {
    let auto_exit_ms: Option<u64> = std::env::var("BLACK_SCREEN_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(|s| s * 1000);

    let window = Window::new("vk-testapp", 1920, 1080)?;
    let start_ticks = sdl3_sys::everything::SDL_GetTicks();

    let ctx = GfxContext::new(window.hwnd())?;
    let mut swapchain = Swapchain::new(&ctx)?;
    let mut renderer = Renderer::new(&ctx)?;
    renderer.create_swapchain_objects(&ctx, &swapchain)?;
    let mut sprites = SpriteBatcher::new(&ctx)?;
    let mut frames = FramesInFlight::new(ctx.device(), FRAMES_IN_FLIGHT);

    vk5k::elog!(
        "testapp swapchain: {}x{} images={} present_mode={:?}",
        swapchain.extent.width,
        swapchain.extent.height,
        swapchain.image_count(),
        swapchain.present_mode
    );

    let mut stats = FrameStats::new(start_ticks);
    let mut running = true;
    let t_start = Instant::now();

    // --- input core ---
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

    let mut ui = UiState {
        expanded: true,
        // TESTAPP_TEST=0 preselects SpriteBatch (for automated runs)
        selected_test: std::env::var("TESTAPP_TEST")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&i| i < TESTS.len()),
        running: std::env::var("TESTAPP_TEST").is_ok(), // automated runs auto-start
    };

    let mut shown_fps: i64 = -1;

    let mut fps_ms: f64 = 0.0;
    let mut flip_lost = false;

    // --- fixed-dt update stage (120 Hz) ---
    const FIXED_DT: f64 = 1.0 / 120.0;
    let mut last = Instant::now();
    let mut update_acc = 0.0f64;

    while running {
        // --- input: drain the event queue once per frame, before update ---
        let t_ev0 = Instant::now();
        let ev = input.poll();
        stats.t_events += t_ev0.elapsed().as_nanos();
        if ev.quit || input.pressed(act_quit) {
            break;
        }

        // update stage: fixed 120 Hz step, free-running render
        let now_inst = Instant::now();
        update_acc += (now_inst - last).as_secs_f64();
        last = now_inst;
        while update_acc >= FIXED_DT {
            if input.pressed(act_shot) {
                renderer.request_screenshot();
            }
            update_acc -= FIXED_DT;
        }

        let mut need_recreate = false;

        let slot = frames.slot();
        let t0 = Instant::now();
        frames.ensure_idle(ctx.device(), slot)?;
        stats.t_wait += t0.elapsed().as_nanos();

        // --- acquire ---
        let img_idx = loop {
            stats.acq_calls += 1;
            let t0 = Instant::now();
            match swapchain.acquire(ctx.device(), frames.image_semaphore(slot))? {
                vk5k::gfx::swapchain::Acquire::Image(idx) => {
                    stats.t_acq += t0.elapsed().as_nanos();
                    break idx;
                }
                vk5k::gfx::swapchain::Acquire::NotReady => {
                    stats.acq_retries += 1;
                }
                vk5k::gfx::swapchain::Acquire::OutOfDate => {
                    stats.t_acq += t0.elapsed().as_nanos();
                    need_recreate = true;
                    break u32::MAX;
                }
                vk5k::gfx::swapchain::Acquire::DeviceLost => {
                    stats.t_acq += t0.elapsed().as_nanos();
                    vk5k::elog!("acquire: device lost (fullscreen switch) — recreating");
                    need_recreate = true;
                    break u32::MAX;
                }
            }
        };

        if need_recreate {
            recreate(&ctx, &mut swapchain, &mut renderer, &mut frames)?;
            continue;
        }

        // --- layout ---
        let w = swapchain.extent.width as f32;
        let h = swapchain.extent.height as f32;
        let s = (w / DESIGN_W).min(h / DESIGN_H);
        let res = (w, h);

        // game view: 16:9 rect fitted into the space right of the panels
        let area_x = SIDEBAR_W + ACTIONS_W;
        let area_y = TOP_BAR_H;
        let area_w = DESIGN_W - area_x - 20.0;
        let area_h = DESIGN_H - area_y - 20.0;
        let mut gw = area_w;
        let mut gh = gw * 9.0 / 16.0;
        if gh > area_h {
            gh = area_h;
            gw = gh * 16.0 / 9.0;
        }
        let gx = area_x + (area_w - gw) / 2.0;
        let gy = area_y + (area_h - gh) / 2.0;

        let mx = input.mouse_pos().0 / s;
        let my = input.mouse_pos().1 / s;

        // hit-test rects
        let header_rect = Rect {
            x: 8.0,
            y: TOP_BAR_H + 10.0,
            w: SIDEBAR_W - 16.0,
            h: ROW_H,
        };
        let test_row = |i: usize| Rect {
            x: 16.0,
            y: TOP_BAR_H + 10.0 + ROW_H + 4.0 + i as f32 * ROW_H,
            w: SIDEBAR_W - 32.0,
            h: ROW_H,
        };
        let shot_btn = Rect {
            x: 180.0,
            y: 11.0,
            w: 130.0,
            h: 30.0,
        };
        // action buttons of the selected test
        let run_btn = Rect {
            x: SIDEBAR_W + 16.0,
            y: TOP_BAR_H + 76.0,
            w: ACTIONS_W - 32.0,
            h: ROW_H,
        };
        let stop_btn = Rect {
            x: SIDEBAR_W + 16.0,
            y: TOP_BAR_H + 76.0 + ROW_H + 6.0,
            w: ACTIONS_W - 32.0,
            h: ROW_H,
        };

        // --- clicks: empty space clears the selection and stops the scene ---
        if input.mouse_pressed(1) {
            let mut handled = false;
            if shot_btn.hit(mx, my) {
                renderer.request_screenshot();
                handled = true;
            } else if header_rect.hit(mx, my) {
                ui.expanded = !ui.expanded;
                handled = true;
            } else if ui.expanded {
                if let Some(i) = (0..TESTS.len()).find(|&i| test_row(i).hit(mx, my)) {
                    ui.selected_test = Some(i);
                    handled = true;
                }
            }
            if ui.selected_test.is_some() {
                if run_btn.hit(mx, my) {
                    ui.running = true;
                    handled = true;
                } else if stop_btn.hit(mx, my) {
                    ui.running = false;
                    handled = true;
                }
            }
            if !handled {
                ui.selected_test = None;
                ui.running = false;
            }
        }

        // --- sprites: background, panels, rows ---
        sprites.begin();
        sprites.begin_section(0.0, 0.0, w, h);
        sprites.gradient(0.0, 0.0, DESIGN_W, DESIGN_H, COL_GRAD_TOP, COL_GRAD_BOTTOM, res);

        // top bar + screenshot button
        sprites.solid(0.0, 0.0, DESIGN_W, TOP_BAR_H, COL_TOP_BAR, res);
        let btn_hover = shot_btn.hit(mx, my);
        sprites.solid(
            shot_btn.x,
            shot_btn.y,
            shot_btn.w,
            shot_btn.h,
            if btn_hover { COL_ROW_HOVER } else { COL_ROW },
            res,
        );

        // left panel (test list) + separator
        sprites.solid(0.0, TOP_BAR_H, SIDEBAR_W, DESIGN_H - TOP_BAR_H, COL_SIDEBAR, res);
        sprites.solid(SIDEBAR_W, TOP_BAR_H, 1.0, DESIGN_H - TOP_BAR_H, COL_BORDER, res);

        // collapsible header item
        let header_hover = header_rect.hit(mx, my);
        sprites.solid(
            header_rect.x,
            header_rect.y,
            header_rect.w,
            header_rect.h,
            if header_hover { COL_ROW_HOVER } else { COL_ROW_SEL },
            res,
        );

        // expanded list of test cases
        if ui.expanded {
            for (i, _) in TESTS.iter().enumerate() {
                let r = test_row(i);
                let sel = ui.selected_test == Some(i);
                let hovered = r.hit(mx, my);
                if sel || hovered {
                    sprites.solid(
                        r.x,
                        r.y,
                        r.w,
                        r.h,
                        if sel { COL_ROW_SEL } else { COL_ROW_HOVER },
                        res,
                    );
                }
                if sel {
                    sprites.solid(r.x, r.y, 3.0, r.h, COL_ACCENT, res);
                }
            }
        }

        // actions panel + separator
        sprites.solid(SIDEBAR_W, TOP_BAR_H, ACTIONS_W, DESIGN_H - TOP_BAR_H, COL_ACTIONS, res);
        sprites.solid(SIDEBAR_W + ACTIONS_W, TOP_BAR_H, 1.0, DESIGN_H - TOP_BAR_H, COL_BORDER, res);

        // action buttons (real actions of the selected test)
        if ui.selected_test.is_some() {
            let run_hover = run_btn.hit(mx, my);
            let stop_hover = stop_btn.hit(mx, my);
            sprites.solid(
                run_btn.x,
                run_btn.y,
                run_btn.w,
                run_btn.h,
                if ui.running { COL_ACCENT } else if run_hover { COL_ROW_HOVER } else { COL_ROW },
                res,
            );
            sprites.solid(
                stop_btn.x,
                stop_btn.y,
                stop_btn.w,
                stop_btn.h,
                if !ui.running { COL_ROW_SEL } else if stop_hover { COL_ROW_HOVER } else { COL_ROW },
                res,
            );
        }

        // game view backdrop + border (the scene renders inside, scissored)
        sprites.solid(gx - 1.0, gy - 1.0, gw + 2.0, gh + 2.0, COL_BORDER, res);
        sprites.solid(gx, gy, gw, gh, COL_GAME_BG, res);

        // stress scene inside the 16:9 game view, clipped by scissor;
        // sprites cover the FULL game area (40x30 grid + wobble)
        if ui.selected_test == Some(0) && ui.running {
            let fb = (gx * s, gy * s, gw * s, gh * s);
            sprites.begin_section(fb.0, fb.1, fb.2, fb.3);
            let t = t_start.elapsed().as_secs_f32();
            let cols = 40usize;
            let rows = (STRESS_SPRITES + cols - 1) / cols;
            for i in 0..STRESS_SPRITES {
                let fi = i as f32;
                let col = (i % cols) as f32;
                let row = (i / cols) as f32;
                let bx = gx + 8.0 + (gw - 24.0) * (col + 0.5) / cols as f32;
                let by = gy + 8.0 + (gh - 24.0) * (row + 0.5) / rows as f32;
                let x = bx + 10.0 * (t * 2.0 + fi * 0.13).sin();
                let y = by + 10.0 * (t * 1.7 + fi * 0.17).cos();
                let col = (
                    0.5 + 0.5 * (fi * 0.01).sin(),
                    0.5 + 0.5 * (fi * 0.013 + 2.0).sin(),
                    0.5 + 0.5 * (fi * 0.017 + 4.0).sin(),
                );
                sprites.rounded(x, y, 14.0, 14.0, 0, col, res);
            }
        }

        // --- text ---
        let mut verts: Vec<Vertex> = Vec::new();

        // top bar
        verts.extend(build_text("TEST BROWSER", 20.0, 16.0, 2.0, w, h, COL_TEXT));
        verts.extend(build_text(
            "SCREENSHOT",
            shot_btn.x + (shot_btn.w - "SCREENSHOT".len() as f32 * 12.0) / 2.0,
            shot_btn.y + 8.0,
            2.0,
            w,
            h,
            if btn_hover { COL_TEXT } else { COL_TEXT_DIM },
        ));
        let status = format!(
            "FPS {shown_fps}  {fps_ms:.2} MS  {}",
            if flip_lost { "FLIP LOST" } else { "FLIP OK" }
        );
        let status_w = status.len() as f32 * 12.0;
        verts.extend(build_text(
            &status,
            DESIGN_W - status_w - 24.0,
            16.0,
            2.0,
            w,
            h,
            COL_TEXT_DIM,
        ));

        // sidebar: header + test list
        verts.extend(build_text(
            "TEST CASES",
            header_rect.x + 12.0,
            header_rect.y + 11.0,
            2.0,
            w,
            h,
            COL_TEXT,
        ));
        verts.extend(build_text(
            if ui.expanded { "-" } else { "+" },
            header_rect.x + header_rect.w - 24.0,
            header_rect.y + 11.0,
            2.0,
            w,
            h,
            COL_TEXT_DIM,
        ));
        if ui.expanded {
            for (i, name) in TESTS.iter().enumerate() {
                let r = test_row(i);
                let sel = ui.selected_test == Some(i);
                verts.extend(build_text(
                    name,
                    r.x + 14.0,
                    r.y + 11.0,
                    2.0,
                    w,
                    h,
                    if sel { COL_TEXT } else { COL_TEXT_DIM },
                ));
            }
        }

        // actions panel
        verts.extend(build_text(
            "ACTIONS",
            SIDEBAR_W + 16.0,
            TOP_BAR_H + 18.0,
            2.0,
            w,
            h,
            COL_TEXT_DIM,
        ));
        if let Some(test) = ui.selected_test {
            verts.extend(build_text(
                TESTS[test],
                SIDEBAR_W + 16.0,
                TOP_BAR_H + 44.0,
                2.0,
                w,
                h,
                COL_TEXT,
            ));
            verts.extend(build_text(
                "RUN",
                run_btn.x + (run_btn.w - 3.0 * 12.0) / 2.0,
                run_btn.y + 11.0,
                2.0,
                w,
                h,
                if ui.running { COL_TEXT } else { COL_TEXT_DIM },
            ));
            verts.extend(build_text(
                "STOP",
                stop_btn.x + (stop_btn.w - 4.0 * 12.0) / 2.0,
                stop_btn.y + 11.0,
                2.0,
                w,
                h,
                if ui.running { COL_TEXT } else { COL_TEXT_DIM },
            ));
        } else {
            verts.extend(build_text(
                "SELECT A TEST",
                SIDEBAR_W + 16.0,
                TOP_BAR_H + 44.0,
                2.0,
                w,
                h,
                COL_TEXT_DIM,
            ));
        }

        // game view caption
        let cap = format!(
            "GAME VIEW 16:9  {}x{}  {}",
            gw as u32,
            gh as u32,
            if ui.selected_test == Some(0) && ui.running {
                "RUNNING"
            } else {
                "IDLE"
            }
        );
        verts.extend(build_text(&cap, gx + 8.0, gy + 8.0, 2.0, w, h, COL_TEXT_DIM));

        if verts.len() > MAX_VERTS {
            return Err(anyhow!("ui frame too long: {} verts", verts.len()));
        }

        // --- upload + record (immediate mode: every frame) ---
        let img_idx_u = img_idx as usize;
        for i in 0..swapchain.image_count() {
            renderer.mark_dirty(i);
        }
        let t0 = Instant::now();
        sprites.upload_slot(ctx.device(), slot);
        renderer.upload_verts(&verts);
        stats.t_rec += t0.elapsed().as_nanos();
        let no_sprites = std::env::var("TESTAPP_NOSPRITES").is_ok();
        if renderer.is_dirty(img_idx_u) && !no_sprites {
            let t0 = Instant::now();
            renderer.record_frame(ctx.device(), img_idx_u, slot, Some(&sprites))?;
            stats.t_rec += t0.elapsed().as_nanos();
        } else if renderer.is_dirty(img_idx_u) {
            let t0 = Instant::now();
            renderer.record_frame(ctx.device(), img_idx_u, slot, None)?;
            stats.t_rec += t0.elapsed().as_nanos();
        }
        let done_sem = swapchain.done_sem(img_idx_u);

        frames.reset_for_submit(ctx.device(), slot)?;
        let t0 = Instant::now();
        let submit = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: frames.image_semaphore_ref(slot),
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: 1,
            p_command_buffers: renderer.command_buffer(img_idx_u),
            signal_semaphore_count: 1,
            p_signal_semaphores: &done_sem,
            ..Default::default()
        };
        ctx.device()
            .queue_submit(ctx.queue, &[submit], frames.fence(slot))
            .context("queue submit")?;
        stats.t_submit += t0.elapsed().as_nanos();

        // --- present ---
        let t0 = Instant::now();
        let present_result = swapchain.present(ctx.queue, img_idx, done_sem);
        stats.t_present += t0.elapsed().as_nanos();
        if let Err(vk::Result::ERROR_OUT_OF_DATE_KHR) = present_result {
            need_recreate = true;
        } else if let Err(vk::Result::ERROR_DEVICE_LOST) = present_result {
            vk5k::elog!("present: device lost (fullscreen switch) — recreating");
            need_recreate = true;
        } else if let Err(e) = present_result {
            return Err(anyhow!("present: {e:?}"));
        }

        if need_recreate {
            recreate(&ctx, &mut swapchain, &mut renderer, &mut frames)?;
            continue;
        }

        // screenshot requested this frame: fence already guarantees the copy
        // finished only after submit; wait for it and write the file
        if renderer.shot_ready() {
            ctx.device().device_wait_idle().context("wait shot")?;
            let path = screenshot_path()?;
            let path_str = path.to_string_lossy().into_owned();
            renderer.write_png(&path_str)?;
            vk5k::elog!("screenshot saved: {}", path.display());
        }

        frames.advance();
        stats.end_frame();

        // --- stats every STATS_WINDOW_MS (after warmup) ---
        let now = sdl3_sys::everything::SDL_GetTicks();
        if now - stats.window_start >= STATS_WINDOW_MS {
            if now - start_ticks >= WARMUP_MS {
                let window_ms = now - stats.window_start;
                let (fps, ms) = stats.report(window_ms);
                stats.window_start = now;
                shown_fps = fps;
                fps_ms = ms;
                flip_lost = stats.flip_lost;
            } else {
                stats.reset_window();
                stats.window_start = now;
            }
        }

        if let Some(ms) = auto_exit_ms {
            if now - start_ticks >= ms {
                running = false;
            }
        }
    }

    // --- teardown ---
    ctx.device().device_wait_idle().context("device_wait_idle")?;
    frames.on_idle();
    renderer.destroy(ctx.device());
    sprites.destroy(ctx.device());
    frames.destroy(ctx.device());
    swapchain.destroy(ctx.device());
    ctx.destroy();
    window.destroy();
    Ok(())
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

/// Recreate the swapchain: idle -> destroy -> create -> rebuild views/CBs.
unsafe fn recreate(
    ctx: &GfxContext,
    swapchain: &mut Swapchain,
    renderer: &mut Renderer,
    frames: &mut FramesInFlight,
) -> Result<()> {
    // after a fullscreen switch the device may report lost — do not fail here
    let _ = ctx.device().device_wait_idle();
    frames.on_idle();
    renderer.destroy_swapchain_objects(ctx.device());
    swapchain.destroy(ctx.device());
    *swapchain = Swapchain::new(ctx)?;
    renderer.create_swapchain_objects(ctx, swapchain)?;
    vk5k::elog!(
        "swapchain recreated: {}x{}",
        swapchain.extent.width,
        swapchain.extent.height
    );
    Ok(())
}
