//! Black screen + FPS counter. Raw Vulkan 1.4 via pixelforge-ash.
//!
//! Swapchain: minImageCount = min+5 (deep queue), PRESENT_MODE_IMMEDIATE,
//! per-image done semaphores, pre-recorded command buffers.
//! Window: SDL3, exclusive fullscreen from the first frame; swapchain is
//! recreated on OUT_OF_DATE.
//! GUI-subsystem: no console, all logging goes to vk5k.log next to the exe
//! (a visible console window breaks exclusive flip on Windows).
//! ESC — quit. BLACK_SCREEN_SECS=N — auto-exit after N seconds.

// no console: a visible console forces DWM composition instead of flip
#![windows_subsystem = "windows"]

mod core;
mod gfx;
mod input;
mod platform;
mod ui;

use anyhow::{anyhow, Context, Result};
use std::time::Instant;

use ash::vk;

use crate::core::nvml::Nvml;
use crate::core::time::{FrameStats, WARMUP_MS};
use crate::gfx::device::GfxContext;
use crate::gfx::frames::FramesInFlight;
use crate::gfx::renderer::Renderer;
use crate::gfx::swapchain::Swapchain;
use crate::input::{Binding, InputCore};
use crate::platform::window::Window;
use crate::ui::text::{build_text, MAX_VERTS};

/// fps/stats averaging window
const STATS_WINDOW_MS: u64 = 250;
const FRAMES_IN_FLIGHT: usize = 8;

static LOG_SINK: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {{
        use std::io::Write;
        if let Ok(mut f) = $crate::LOG_SINK.lock() {
            if let Some(file) = f.as_mut() {
                let _ = writeln!(file, $($arg)*);
            }
        }
    }};
}

fn main() -> Result<()> {
    // GUI-subsystem: no stdout — log goes to vk5k.log next to the exe.
    let log_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("vk5k.log")));
    if let Some(path) = &log_path {
        if let Ok(file) = std::fs::File::create(path) {
            *crate::LOG_SINK.lock().unwrap() = Some(file);
        }
    }

    // GPU clock lock needs admin: relaunch self via UAC (runas) when not
    // elevated. UAC declined -> continue without the lock.
    unsafe {
        if windows_sys::Win32::UI::Shell::IsUserAnAdmin() == 0 {
            crate::elog!("not elevated — relaunching via UAC (runas)");
            if let Ok(exe) = std::env::current_exe() {
                let exe_w: Vec<u16> = exe
                    .as_os_str()
                    .to_str()
                    .unwrap_or("")
                    .encode_utf16()
                    .chain(Some(0))
                    .collect();
                let verb: Vec<u16> = "runas\0".encode_utf16().collect();
                let r = windows_sys::Win32::UI::Shell::ShellExecuteW(
                    std::ptr::null_mut(),
                    verb.as_ptr(),
                    exe_w.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1, // SW_SHOWNORMAL
                );
                if r as isize > 32 {
                    std::process::exit(0);
                }
                crate::elog!("UAC declined — running without the clock lock");
            }
        }
    }

    // high priority: fewer scheduling jitters in p95/max
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS,
        };
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
    unsafe { run() }
}

unsafe fn run() -> Result<()> {
    let auto_exit_ms: Option<u64> = std::env::var("BLACK_SCREEN_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(|s| s * 1000);

    let window = Window::new("black-vk", 1920, 1080)?;
    let start_ticks = sdl3_sys::everything::SDL_GetTicks();

    // lock GPU clock to the driver max (NVML, needs admin); reset on exit
    let _nvml = match Nvml::init_and_lock_max() {
        Ok(n) => {
            crate::elog!("GPU: {} | clock locked at {} MHz", n.gpu_name, n.locked_mhz);
            Some(n)
        }
        Err(e) => {
            crate::elog!("NVML: {e} — clock not locked");
            None
        }
    };

    let ctx = GfxContext::new(window.hwnd())?;
    let mut swapchain = Swapchain::new(&ctx)?;
    let mut renderer = Renderer::new(&ctx)?;
    renderer.create_swapchain_objects(ctx.device(), &swapchain)?;
    let mut frames = FramesInFlight::new(ctx.device(), FRAMES_IN_FLIGHT);

    crate::elog!(
        "swapchain: {}x{} images={} present_mode={:?}",
        swapchain.extent.width,
        swapchain.extent.height,
        swapchain.image_count(),
        swapchain.present_mode
    );

    let mut stats = FrameStats::new(start_ticks);
    let mut text = String::from("FPS --");
    let mut status_text = String::from("FLIP ...");
    let mut status_color = (0.7f32, 0.7f32, 0.7f32);
    let mut shown_fps: i64 = -1;
    let mut uploaded = false;
    let mut running = true;

    // --- input core: action mapping, edge events, ns timestamps ---
    let mut input = InputCore::new();
    let act_quit = input.map(
        "quit",
        &[Binding::Key(sdl3_sys::everything::SDL_Scancode::ESCAPE), Binding::Pad(sdl3_sys::everything::SDL_GamepadButton::START)],
    );
    let act_jump = input.map(
        "jump",
        &[
            Binding::Key(sdl3_sys::everything::SDL_Scancode::SPACE),
            Binding::Mouse(1),
            Binding::Pad(sdl3_sys::everything::SDL_GamepadButton::SOUTH),
        ],
    );
    let mut inp_overlay = String::new();

    // --- fixed-dt update stage (120 Hz): game logic goes here ---
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
            // demo edge handling: last input -> overlay with latency
            // (event timestamp vs now)
            if let Some((id, ts)) = input.last_edge() {
                let now_ns = unsafe { sdl3_sys::everything::SDL_GetTicksNS() };
                let age_ms = now_ns.saturating_sub(ts) as f64 / 1e6;
                inp_overlay = format!(
                    "INP {} +{age_ms:.2}ms  ev={} pad={} jump={}",
                    input.action_name(id),
                    input.events_last_poll(),
                    input.is_gamepad_connected() as u8,
                    input.down(act_jump) as u8
                );
                uploaded = false;
                break; // frame already contains fresh input
            }
            // TODO(Phase 2): simulation/UI step
            update_acc -= FIXED_DT;
        }

        let mut need_recreate = false;

        // no need to wait all frames before overwriting a ring slot:
        // a slot is not reused until its fence signals
        let slot = frames.slot();
        let t0 = Instant::now();
        frames.ensure_idle(ctx.device(), slot)?;
        stats.t_wait += t0.elapsed().as_nanos();

        // --- acquire (retry on NOT_READY, recreate on OUT_OF_DATE) ---
        let img_idx = loop {
            stats.acq_calls += 1;
            let t0 = Instant::now();
            match swapchain.acquire(ctx.device(), frames.image_semaphore(slot))? {
                gfx::swapchain::Acquire::Image(idx) => {
                    stats.t_acq += t0.elapsed().as_nanos();
                    break idx;
                }
                gfx::swapchain::Acquire::NotReady => {
                    stats.acq_retries += 1;
                }
                gfx::swapchain::Acquire::OutOfDate => {
                    stats.t_acq += t0.elapsed().as_nanos();
                    need_recreate = true;
                    break u32::MAX;
                }
            }
        };

        if need_recreate {
            recreate(&ctx, &mut swapchain, &mut renderer, &mut frames)?;
            uploaded = false;
            continue;
        }

        // --- content (counter text) ---
        if !uploaded {
            uploaded = true;
            for i in 0..swapchain.image_count() {
                renderer.mark_dirty(i);
            }
            let mut verts = build_text(
                &text,
                16.0,
                16.0,
                2.0,
                swapchain.extent.width as f32,
                swapchain.extent.height as f32,
                (1.0, 1.0, 1.0),
            );
            // second line: exclusive fullscreen status, colored
            verts.extend(build_text(
                &status_text,
                16.0,
                40.0,
                2.0,
                swapchain.extent.width as f32,
                swapchain.extent.height as f32,
                status_color,
            ));
            // third line: last input edge with latency (Phase 2.1 demo)
            if !inp_overlay.is_empty() {
                verts.extend(build_text(
                    &inp_overlay,
                    16.0,
                    64.0,
                    2.0,
                    swapchain.extent.width as f32,
                    swapchain.extent.height as f32,
                    (0.5, 0.8, 1.0),
                ));
            }
            if verts.len() > MAX_VERTS {
                return Err(anyhow!("text too long: {} verts", verts.len()));
            }
            renderer.upload_verts(&verts);
        }

        // --- record (dirty only) + submit ---
        let img_idx_u = img_idx as usize;
        if renderer.is_dirty(img_idx_u) {
            let t0 = Instant::now();
            renderer.record(ctx.device(), img_idx_u, slot)?;
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
        } else if let Err(e) = present_result {
            return Err(anyhow!("present: {e:?}"));
        }

        if need_recreate {
            recreate(&ctx, &mut swapchain, &mut renderer, &mut frames)?;
            uploaded = false;
            continue;
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
                if fps != shown_fps {
                    shown_fps = fps;
                    // composition marker visible on screen, not only in the log
                    text = if stats.flip_lost {
                        format!("FPS {fps} {ms:.2} [DWM]")
                    } else {
                        format!("FPS {fps}  {ms:.2}")
                    };
                    if stats.flip_lost {
                        status_text = String::from("FLIP LOST DWM");
                        status_color = (1.0, 0.25, 0.2);
                    } else {
                        status_text = String::from("FLIP OK");
                        status_color = (0.1, 1.0, 0.2);
                    }
                    uploaded = false;
                }
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
    frames.destroy(ctx.device());
    swapchain.destroy(ctx.device());
    ctx.destroy();
    window.destroy();
    Ok(())
}

/// Recreate the swapchain: idle -> destroy -> create -> rebuild views/CBs
/// for the new extent.
unsafe fn recreate(
    ctx: &GfxContext,
    swapchain: &mut Swapchain,
    renderer: &mut Renderer,
    frames: &mut FramesInFlight,
) -> Result<()> {
    ctx.device().device_wait_idle().context("wait idle (recreate)")?;
    frames.on_idle();
    renderer.destroy_swapchain_objects(ctx.device());
    swapchain.destroy(ctx.device());
    *swapchain = Swapchain::new(ctx)?;
    renderer.create_swapchain_objects(ctx.device(), swapchain)?;
    crate::elog!(
        "swapchain recreated: {}x{}",
        swapchain.extent.width,
        swapchain.extent.height
    );
    Ok(())
}
