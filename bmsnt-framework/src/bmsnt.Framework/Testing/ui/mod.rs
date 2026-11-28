//! Browser UI drawables (the `osu.Framework.Testing.Drawables` analog):
//! immediate-mode hit-test rects, app state and a one-shot `draw` of all
//! panels, buttons and text, rendered by the sprite batcher. Layout lives
//! in design space (`theme`).

mod layout;

pub use layout::{list_rows, Layout, ListRow};

use crate::graphics::sprites::SpriteBatcher;
use crate::text::{build_text, Vertex};

use crate::testing::design_palette::*;

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

/// Draw the whole UI: background, panels, buttons and text. `mx`/`my` are
/// mouse coords in design space, `res` the framebuffer size, `status` the
/// top-bar status line ("FPS .. FLIP .."), `scene_running` whether a test
/// scene is active (drives the game-view caption and RUN/STOP colors),
/// `items` — (display name, folder) of the tests (sidebar rows grouped by
/// folder).
pub fn draw(
    sprites: &mut SpriteBatcher,
    layout: &Layout,
    ui: &UiState,
    items: &[(&'static str, &'static str)],
    mx: f32,
    my: f32,
    res: (f32, f32),
    status: &str,
    scene_running: bool,
) -> Vec<Vertex> {
    let mut verts = Vec::new();
    let rows = list_rows(items);

    // --- sprites: background, panels, rows ---
    sprites.begin_section(0.0, 0.0, res.0, res.1);
    sprites.gradient(
        0.0,
        0.0,
        DESIGN_W,
        DESIGN_H,
        COL_GRAD_TOP,
        COL_GRAD_BOTTOM,
        res,
    );

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
    sprites.solid(
        0.0,
        TOP_BAR_H,
        SIDEBAR_W,
        DESIGN_H - TOP_BAR_H,
        COL_SIDEBAR,
        res,
    );
    sprites.solid(
        SIDEBAR_W,
        TOP_BAR_H,
        1.0,
        DESIGN_H - TOP_BAR_H,
        COL_BORDER,
        res,
    );

    // collapsible header item
    let header_hover = layout.header.hit(mx, my);
    sprites.solid(
        layout.header.x,
        layout.header.y,
        layout.header.w,
        layout.header.h,
        if header_hover {
            COL_ROW_HOVER
        } else {
            COL_ROW_SEL
        },
        res,
    );

    // expanded list: folder headers + test rows
    if ui.expanded {
        for row in &rows {
            if let Some(_folder) = row.folder {
                // folder header: thin accent line on top of the row
                sprites.solid(row.rect.x, row.rect.y, row.rect.w, 1.0, COL_BORDER, res);
            } else {
                let r = &row.rect;
                let sel = ui.selected_test == row.test;
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
    }

    // actions panel (no separator on the right: the game view starts flush)
    sprites.solid(
        SIDEBAR_W,
        TOP_BAR_H,
        ACTIONS_W,
        DESIGN_H - TOP_BAR_H,
        COL_ACTIONS,
        res,
    );

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
    verts.extend(build_text(
        "TEST BROWSER",
        20.0,
        16.0,
        2.0,
        res.0,
        res.1,
        COL_TEXT,
    ));
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
        "TESTS",
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
        for row in &rows {
            let r = &row.rect;
            if let Some(folder) = row.folder {
                verts.extend(build_text(
                    folder,
                    r.x + 14.0,
                    r.y + 11.0,
                    2.0,
                    res.0,
                    res.1,
                    COL_ACCENT,
                ));
            } else {
                let i = row.test.unwrap_or(0);
                let sel = ui.selected_test == row.test;
                verts.extend(build_text(
                    items[i].0,
                    r.x + 14.0,
                    r.y + 11.0,
                    2.0,
                    res.0,
                    res.1,
                    if sel { COL_TEXT } else { COL_TEXT_DIM },
                ));
            }
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
    if let Some(test) = ui.selected_test.filter(|&i| i < items.len()) {
        verts.extend(build_text(
            items[test].0,
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
    verts.extend(build_text(
        &cap,
        g.x + 8.0,
        g.y + 8.0,
        2.0,
        res.0,
        res.1,
        COL_TEXT_DIM,
    ));

    verts
}
