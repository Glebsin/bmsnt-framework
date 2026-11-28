//! One-frame layout of the browser UI: rects of panels, buttons and the game
//! view, all in design space (the geometry part of `osu.Framework.Testing
//! .Drawables`).

use crate::graphics::primitives::Rect;

use crate::testing::design_palette::*;

/// One-frame layout of all UI rects (pure design space, no scaling here).
pub struct Layout {
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
}

/// Folders of `items` in first-appearance order.
pub fn folders(items: &[(&'static str, &'static str)]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for &(_, folder) in items {
        if !out.contains(&folder) {
            out.push(folder);
        }
    }
    out
}

/// One display row of the sidebar: a folder group header or a test row.
pub struct ListRow {
    pub rect: Rect,
    /// Some(label) — folder group header row (a collapse toggle); None — test row
    pub folder: Option<&'static str>,
    /// test index for test rows
    pub test: Option<usize>,
}

/// Sidebar display rows: one group header per folder (first-appearance
/// order), and — only for expanded folders — the test rows of that folder.
/// `expanded` — the set of expanded folder labels.
pub fn list_rows(
    items: &[(&'static str, &'static str)],
    expanded: &[&'static str],
) -> Vec<ListRow> {
    let mut rows = Vec::new();
    let mut seen: Vec<&'static str> = Vec::new();
    let mut y = TOP_BAR_H + 10.0;
    for (i, &(_, folder)) in items.iter().enumerate() {
        if !seen.contains(&folder) {
            seen.push(folder);
            rows.push(ListRow {
                rect: Rect::new(8.0, y, SIDEBAR_W - 16.0, ROW_H),
                folder: Some(folder),
                test: None,
            });
            y += ROW_H;
        }
        if !expanded.contains(&folder) {
            continue;
        }
        rows.push(ListRow {
            rect: Rect::new(16.0, y, SIDEBAR_W - 32.0, ROW_H),
            folder: None,
            test: Some(i),
        });
        y += ROW_H;
    }
    rows
}
