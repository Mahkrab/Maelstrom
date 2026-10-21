//! The small bitmap panel is drawn after the scene, keeping diagnostics readable
//! without adding an application framework or a second event scheduler.

use font8x8::{BASIC_FONTS, UnicodeFonts};

pub(crate) fn panel(lines: &[String], width: u32, height: u32, scale: f32) -> Vec<f32> {
    let mut out = Vec::new();
    let cell = scale;
    let columns = (((width as f32 - 20.0) / (8.0 * cell)) as usize).max(1);
    let wrapped: Vec<String> = lines
        .iter()
        .flat_map(|line| {
            let chars: Vec<_> = line.chars().collect();
            if chars.is_empty() {
                vec![String::new()]
            } else {
                chars
                    .chunks(columns)
                    .map(|chunk| chunk.iter().collect())
                    .collect()
            }
        })
        .collect();
    let lines = &wrapped;
    let line_height = 10.0 * cell;
    let panel_width =
        (lines.iter().map(|s| s.chars().count()).max().unwrap_or(0) as f32 * 8.0 * cell + 20.0)
            .min(width as f32);
    rect(
        &mut out,
        [
            0.0,
            0.0,
            panel_width,
            (lines.len() as f32 * line_height + 16.0).min(height as f32),
        ],
        [0.003, 0.006, 0.009],
        width,
        height,
    );
    for (row, line) in lines.iter().enumerate() {
        let y = 8.0 + row as f32 * line_height;
        if y + 8.0 * cell > height as f32 {
            break;
        }
        for (column, c) in line.chars().enumerate() {
            let x = 10.0 + column as f32 * 8.0 * cell;
            if x + 8.0 * cell > width as f32 {
                break;
            }
            let glyph = BASIC_FONTS
                .get(c)
                .or_else(|| BASIC_FONTS.get('?'))
                .unwrap_or([0; 8]);
            for (gy, bits) in glyph.into_iter().enumerate() {
                for gx in 0..8 {
                    if bits & (1 << gx) != 0 {
                        rect(
                            &mut out,
                            [x + gx as f32 * cell, y + gy as f32 * cell, cell, cell],
                            [0.75, 0.85, 0.9],
                            width,
                            height,
                        );
                    }
                }
            }
        }
    }
    out
}
fn rect(out: &mut Vec<f32>, box_px: [f32; 4], colour: [f32; 3], width: u32, height: u32) {
    let [x, y, w, h] = box_px;
    let w = w.min(width as f32 - x).max(0.0);
    let h = h.min(height as f32 - y).max(0.0);
    for (px, py) in [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y),
        (x + w, y + h),
        (x, y + h),
    ] {
        out.extend([
            2.0 * px / width as f32 - 1.0,
            1.0 - 2.0 * py / height as f32,
            colour[0],
            colour[1],
            colour[2],
        ]);
    }
}
