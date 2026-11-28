//! Frame timing and statistics.

use std::time::Instant;

/// first N ms are ignored: presentation-path init and GPU clock transitions
/// would skew the stats
pub const WARMUP_MS: u64 = 2000;

#[derive(Default)]
pub struct FrameStats {
    pub t_wait: u128,
    pub t_acq: u128,
    pub t_rec: u128,
    pub t_submit: u128,
    pub t_present: u128,
    pub t_events: u128,
    /// ui::draw (text + panels geometry)
    pub t_ui: u128,
    /// test scene draw 
    pub t_scene: u128,
    pub acq_calls: u32,
    pub acq_retries: u32,
    /// present went through DWM composition instead of exclusive flip
    pub flip_lost: bool,
    /// hysteresis: consecutive windows below / above the flip threshold —
    /// present time fluctuates around 80 us under load, so the state may
    /// only change after several consistent windows
    flip_below_streak: u32,
    flip_above_streak: u32,
    frame_us: Vec<u32>,
    t_frame: Option<Instant>,
    pub window_start: u64,
    frames_in_window: u64,
}

impl FrameStats {
    pub fn new(now_ms: u64) -> Self {
        Self {
            t_frame: Some(Instant::now()),
            window_start: now_ms,
            ..Default::default()
        }
    }

    /// end of frame: measure the full turn
    pub fn end_frame(&mut self) {
        if let Some(t) = self.t_frame.take() {
            self.frame_us
                .push(t.elapsed().as_micros().min(u32::MAX as u128) as u32);
        }
        self.t_frame = Some(Instant::now());
        self.frames_in_window += 1;
    }

    /// Print stats for a `window_ms` window; returns (fps, ms/frame).
    pub fn report(&mut self, window_ms: u64) -> (i64, f64) {
        let frames = self.frames_in_window;
        let n = frames.max(1) as f64;
        let fps = (frames as f64 * 1000.0 / window_ms.max(1) as f64) as i64;
        let mut sorted = self.frame_us.clone();
        sorted.sort_unstable();
        let pct = |p: f64| -> u32 {
            if sorted.is_empty() {
                0
            } else {
                sorted[((sorted.len() as f64 * p).min((sorted.len() - 1) as f64)) as usize]
            }
        };
        // 1%/0.1% lows: time of the worst 1%/0.1% of frames, in ms
        let low = |frac: f64| -> f64 {
            if sorted.is_empty() {
                0.0
            } else {
                let k = ((sorted.len() as f64 * frac).ceil() as usize).max(1);
                sorted[sorted.len() - k] as f64 / 1000.0
            }
        };

        // presentation-path detector: on exclusive flip present costs
        // ~110-130 us CPU; consistently < 80 us means frames go through DWM
        // composition (flip lost: overlay/occlusion/HDR/driver settings).
        // Composition ALSO caps fps at ~3k on this machine — a low present
        // time with high fps means the flip path simply got faster (e.g.
        // NVML clock lock), not that flip was lost. So LOST requires BOTH:
        // low present time AND the composition fps ceiling. Hysteresis: 3
        // consecutive windows are required to change the state.
        let present_us = self.t_present as f64 / n / 1000.0;
        const FLIP_LOST_AFTER: u32 = 3;
        const FLIP_OK_AFTER: u32 = 3;
        const COMPOSITION_FPS_CEILING: i64 = 4500;
        if self.t_present > 0 && present_us < 80.0 && fps > 1000 && fps < COMPOSITION_FPS_CEILING {
            self.flip_below_streak += 1;
            self.flip_above_streak = 0;
            if self.flip_below_streak >= FLIP_LOST_AFTER {
                self.flip_lost = true;
            }
        } else if self.t_present > 0 {
            self.flip_above_streak += 1;
            self.flip_below_streak = 0;
            if self.flip_above_streak >= FLIP_OK_AFTER {
                self.flip_lost = false;
            }
        }
        let flip_note = if self.flip_lost {
            "  << !! FLIP LOST: present via DWM composition >>"
        } else {
            ""
        };

        crate::elog!(
            "fps: {fps} | cpu avg us/frame: wait {:.1} acq {:.1} rec {:.1} ui {:.1} scene {:.1} submit {:.1} present {:.1} events {:.1} | frame us avg {:.0} p95 {} max {} | 1% low {:.2} ms 0.1% low {:.2} ms | acq calls {} retries {}{flip_note}",
            self.t_wait as f64 / n / 1000.0,
            self.t_acq as f64 / n / 1000.0,
            self.t_rec as f64 / n / 1000.0,
            self.t_ui as f64 / n / 1000.0,
            self.t_scene as f64 / n / 1000.0,
            self.t_submit as f64 / n / 1000.0,
            self.t_present as f64 / n / 1000.0,
            self.t_events as f64 / n / 1000.0,
            sorted.iter().map(|&x| x as u64).sum::<u64>() as f64 / sorted.len().max(1) as f64,
            pct(0.95),
            sorted.last().copied().unwrap_or(0),
            low(0.01),
            low(0.001),
            self.acq_calls,
            self.acq_retries,
        );

        let ms = if fps > 0 { 1000.0 / fps as f64 } else { 0.0 };
        self.reset_window();
        (fps, ms)
    }

    pub fn reset_window(&mut self) {
        self.t_wait = 0;
        self.t_acq = 0;
        self.t_rec = 0;
        self.t_submit = 0;
        self.t_present = 0;
        self.t_events = 0;
        self.t_ui = 0;
        self.t_scene = 0;
        self.acq_calls = 0;
        self.acq_retries = 0;
        self.frame_us.clear();
        self.frames_in_window = 0;
    }
}
