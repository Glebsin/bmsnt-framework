//! Immediate-mode UI: hit-test rects ("drawables"), app state and a one-shot
//! `draw` of all panels, buttons and text — the osu-framework drawable layer
//! rendered by the sprite batcher. Layout lives in design space (`theme`).

use bmsnt_framework::gfx::sprite::SpriteBatcher;
use bmsnt_framework::ui::text::{build_text, Vertex};

use crate::theme::*;

/// Axis-aligned rect in design space; the primitive of every UI element.
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
    pub fn hit(&self, mx: f32, my: f32) -> bool {
        mx >= self.x && mx <= self.x + self.w && my >= self.y && my <= self.y + self.h
    }
}

/// Mutable UI state (selection, running test, list expanded).
pub struct UiState {
    pub expanded: bool,
    pub selected_test: Option<usize>,
    pub running: bool,
}

impl UiState {
    /// `BMSNT_TEST=<i>` preselects test `i` and auto-starts it (for
    /// automated runs).
    pub fn from_env() -> Self {
        Self {
            expanded: true,
            selected_test: std::env::var("BMSNT_TEST")
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
            running: std::env::var("BMSNT_TEST").is_ok(),
        }
    }
}

/// One-frame layout of all UI rects (pure design space, no scaling here).
pub struct Layout {
    pub header: Rect,
    pub shot_btn: Rect,
    pub run_btn: Rect,
    pub stop_btn: Rect,
    /// the 16:9 game view fitted right of the panels
    pub game: Rect,
}

impl Layout {
    pub fn new() -> Self {
        // game area: right of the panels, flush to the right and bottom
        // edges; the 16:9 view is maximized inside it and centered
        let area_x = SIDEBAR_W + ACTIONS_W;
        let area_y = TOP_BAR_H;
        let area_w = DESIGN_W - area_x;
        let area_h = DESIGN_H - area_y;
        let mut gw = area_w;
        let mut gh = gw * 9.0 / 16.0;
        if gh > area_h {
            gh = area_h;
            gw = gh * 16.0 / 9.0;
        }
        Self {
            header: Rect::new(8.0, TOP_BAR_H + 10.0, SIDEBAR_W - 16.0, ROW_H),
            shot_btn: Rect::new(180.0, 11.0, 130.0, 30.0),
            run_btn: Rect::new(SIDEBAR_W + 16.0, TOP_BAR_H + 76.0, ACTIONS_W - 32.0, ROW_H),
            stop_btn: Rect::new(
                SIDEBAR_W + 16.0,
                TOP_BAR_H + 76.0 + ROW_H + 6.0,
                ACTIONS_W - 32.0,
                ROW_H,
            ),
            game: Rect::new(
                area_x + (area_w - gw) / 2.0,
                area_y + (area_h - gh) / 2.0,
                gw,
                gh,
            ),
        }
    }

    /// Rect of test row `i` in the (collapsible) sidebar list.
    pub fn row(&self, i: usize) -> Rect {
        Rect::new(
            16.0,
            TOP_BAR_H + 10.0 + ROW_H + 4.0 + i as f32 * ROW_H,
            SIDEBAR_W - 32.0,
            ROW_H,
        )
    }
}

/// Draw the whole UI: background, panels, buttons and text. `mx`/`my` are
/// mouse coords in design space, `res` the framebuffer size, `status` the
/// top-bar status line ("FPS .. FLIP .."), `scene_running` whether a test
/// scene is active (drives the game-view caption and RUN/STOP colors),
/// `names` — display names of the tests (sidebar rows).
pub fn draw(
    sprites: &mut SpriteBatcher,
    layout: &Layout,
    ui: &UiState,
    names: &[&str],
    mx: f32,
    my: f32,
    res: (f32, f32),
    status: &str,
    scene_running: bool,
) -> Vec<Vertex> {
    let mut verts = Vec::new();

    // --- sprites: background, panels, rows ---
    sprites.begin_section(0.0, 0.0, res.0, res.1);
    sprites.gradient(0.0, 0.0, DESIGN_W, DESIGN_H, COL_GRAD_TOP, COL_GRAD_BOTTOM, res);

    // top bar + screenshot button
    sprites.solid(0.0, 0.0, DESIGN_W, TOP_BAR_H, COL_TOP_BAR, res);
    let btn_hover = layout.shot_btn.hit(mx, my);
    sprites.solid(
        layout.shot_btn.x,
        layout.shot_btn.y,
        layout.shot_btn.w,
        layout.shot_btn.h,
        if btn_hover { COL_ROW_HOVER } else { COL_ROW },
        res,
    );

    // left panel (test list) + separator
    sprites.solid(0.0, TOP_BAR_H, SIDEBAR_W, DESIGN_H - TOP_BAR_H, COL_SIDEBAR, res);
    sprites.solid(SIDEBAR_W, TOP_BAR_H, 1.0, DESIGN_H - TOP_BAR_H, COL_BORDER, res);

    // collapsible header item
    let header_hover = layout.header.hit(mx, my);
    sprites.solid(
        layout.header.x,
        layout.header.y,
        layout.header.w,
        layout.header.h,
        if header_hover { COL_ROW_HOVER } else { COL_ROW_SEL },
        res,
    );

    // expanded list of test cases
    if ui.expanded {
        for i in 0..names.len() {
            let r = layout.row(i);
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

    // actions panel (no separator on the right: the game view starts flush)
    sprites.solid(SIDEBAR_W, TOP_BAR_H, ACTIONS_W, DESIGN_H - TOP_BAR_H, COL_ACTIONS, res);

    // action buttons (real actions of the selected test)
    if ui.selected_test.is_some() {
        let run_hover = layout.run_btn.hit(mx, my);
        let stop_hover = layout.stop_btn.hit(mx, my);
        sprites.solid(
            layout.run_btn.x,
            layout.run_btn.y,
            layout.run_btn.w,
            layout.run_btn.h,
            if ui.running {
                COL_ACCENT
            } else if run_hover {
                COL_ROW_HOVER
            } else {
                COL_ROW
            },
            res,
        );
        sprites.solid(
            layout.stop_btn.x,
            layout.stop_btn.y,
            layout.stop_btn.w,
            layout.stop_btn.h,
            if !ui.running {
                COL_ROW_SEL
            } else if stop_hover {
                COL_ROW_HOVER
            } else {
                COL_ROW
            },
            res,
        );
    }

    // game view backdrop (no border frame; the scene renders inside, scissored)
    let g = &layout.game;
    sprites.solid(g.x, g.y, g.w, g.h, COL_GAME_BG, res);

    // --- text ---
    verts.extend(build_text("TEST BROWSER", 20.0, 16.0, 2.0, res.0, res.1, COL_TEXT));
    verts.extend(build_text(
        "SCREENSHOT",
        layout.shot_btn.x + (layout.shot_btn.w - "SCREENSHOT".len() as f32 * 12.0) / 2.0,
        layout.shot_btn.y + 8.0,
        2.0,
        res.0,
        res.1,
        if btn_hover { COL_TEXT } else { COL_TEXT_DIM },
    ));
    let status_w = status.len() as f32 * 12.0;
    verts.extend(build_text(
        status,
        DESIGN_W - status_w - 24.0,
        16.0,
        2.0,
        res.0,
        res.1,
        COL_TEXT_DIM,
    ));

    // sidebar: header + test list
    verts.extend(build_text(
        "TEST CASES",
        layout.header.x + 12.0,
        layout.header.y + 11.0,
        2.0,
        res.0,
        res.1,
        COL_TEXT,
    ));
    verts.extend(build_text(
        if ui.expanded { "-" } else { "+" },
        layout.header.x + layout.header.w - 24.0,
        layout.header.y + 11.0,
        2.0,
        res.0,
        res.1,
        COL_TEXT_DIM,
    ));
    if ui.expanded {
        for (i, name) in names.iter().enumerate() {
            let r = layout.row(i);
            let sel = ui.selected_test == Some(i);
            verts.extend(build_text(
                name,
                r.x + 14.0,
                r.y + 11.0,
                2.0,
                res.0,
                res.1,
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
        res.0,
        res.1,
        COL_TEXT_DIM,
    ));
    if let Some(test) = ui.selected_test.filter(|&i| i < names.len()) {
        verts.extend(build_text(
            names[test],
            SIDEBAR_W + 16.0,
            TOP_BAR_H + 44.0,
            2.0,
            res.0,
            res.1,
            COL_TEXT,
        ));
        verts.extend(build_text(
            "RUN",
            layout.run_btn.x + (layout.run_btn.w - 3.0 * 12.0) / 2.0,
            layout.run_btn.y + 11.0,
            2.0,
            res.0,
            res.1,
            if ui.running { COL_TEXT } else { COL_TEXT_DIM },
        ));
        verts.extend(build_text(
            "STOP",
            layout.stop_btn.x + (layout.stop_btn.w - 4.0 * 12.0) / 2.0,
            layout.stop_btn.y + 11.0,
            2.0,
            res.0,
            res.1,
            if ui.running { COL_TEXT } else { COL_TEXT_DIM },
        ));
    } else {
        verts.extend(build_text(
            "SELECT A TEST",
            SIDEBAR_W + 16.0,
            TOP_BAR_H + 44.0,
            2.0,
            res.0,
            res.1,
            COL_TEXT_DIM,
        ));
    }

    // game view caption
    let cap = format!(
        "GAME VIEW 16:9  {}x{}  {}",
        layout.game.w as u32,
        layout.game.h as u32,
        if scene_running { "RUNNING" } else { "IDLE" }
    );
    verts.extend(build_text(&cap, g.x + 8.0, g.y + 8.0, 2.0, res.0, res.1, COL_TEXT_DIM));

    verts
}
