//! Spatial text-layout reconstruction: turn recognized text spans with pixel
//! boxes into monospace text whose spacing approximates the source image —
//! the `pdftotext -layout` idea applied to OCR boxes. Sidebar items stay
//! left, content stays right, button rows stay on one line.
//!
//! The output is meant for a fenced code block (Markdown collapses runs of
//! spaces in ordinary paragraphs, a code fence preserves them).

/// One recognized snippet with its box in source-image pixels (any consistent
/// unit works — only ratios are used).
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub text: String,
    pub l: f32,
    pub t: f32,
    pub r: f32,
    pub b: f32,
}

/// Vertical-gap multiple (of the median span height) that inserts a blank
/// line between rows.
const BLANK_GAP: f32 = 1.6;
/// Hard cap on the starting column, so one stray box at the far right of a
/// 4K screenshot can't produce kilometer lines.
const MAX_COL: usize = 200;

/// Render spans as layout-preserving monospace text.
pub fn grid(spans: &[Span]) -> String {
    if spans.is_empty() {
        return String::new();
    }
    // Estimate the source's character width as the median of per-span
    // width-per-character — proportional fonts make this approximate, which
    // is fine: we want relative indentation, not pixel fidelity.
    let mut widths: Vec<f32> = spans
        .iter()
        .filter(|s| !s.text.is_empty())
        .map(|s| (s.r - s.l).max(1.0) / s.text.chars().count() as f32)
        .collect();
    if widths.is_empty() {
        return String::new();
    }
    widths.sort_by(|a, b| a.total_cmp(b));
    let char_w = widths[widths.len() / 2].max(1.0);
    let mut heights: Vec<f32> = spans.iter().map(|s| (s.b - s.t).max(1.0)).collect();
    heights.sort_by(|a, b| a.total_cmp(b));
    let line_h = heights[heights.len() / 2];

    // Group into rows by vertical-center proximity (same rule as the
    // detector's reading order), left-to-right within a row.
    let mut sorted: Vec<&Span> = spans.iter().collect();
    sorted.sort_by(|a, b| ((a.t + a.b) / 2.0).total_cmp(&((b.t + b.b) / 2.0)));
    let mut rows: Vec<Vec<&Span>> = Vec::new();
    for s in sorted {
        let cy = (s.t + s.b) / 2.0;
        match rows.last_mut() {
            Some(row) if (cy - (row[0].t + row[0].b) / 2.0).abs() < (row[0].b - row[0].t) / 2.0 => {
                row.push(s)
            }
            _ => rows.push(vec![s]),
        }
    }

    let mut out = String::new();
    let mut prev_bottom: Option<f32> = None;
    for row in &mut rows {
        row.sort_by(|a, b| a.l.total_cmp(&b.l));
        // A vertical gap well beyond one line height becomes a blank line —
        // dialog sections keep their visual separation.
        if let Some(pb) = prev_bottom {
            if row[0].t - pb > BLANK_GAP * line_h {
                out.push('\n');
            }
        }
        prev_bottom = Some(row.iter().fold(f32::MIN, |m, s| m.max(s.b)));
        let mut line = String::new();
        for s in row.iter() {
            let col = ((s.l / char_w).round() as usize).min(MAX_COL);
            let cur = line.chars().count();
            // At least one space between neighbors whose estimate collides.
            let pad = col.max(cur + usize::from(cur > 0)) - cur;
            line.extend(std::iter::repeat_n(' ', pad));
            line.push_str(&s.text);
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, l: f32, t: f32, r: f32, b: f32) -> Span {
        Span { text: text.into(), l, t, r, b }
    }

    #[test]
    fn side_by_side_spans_share_a_line_with_spacing() {
        // ~10 px/char. "Menu" at x=0, "Content" at x=300 → column 30.
        let spans = [
            span("Menu", 0.0, 0.0, 40.0, 12.0),
            span("Content", 300.0, 1.0, 370.0, 13.0),
            span("Below", 0.0, 30.0, 50.0, 42.0),
        ];
        let g = grid(&spans);
        let lines: Vec<&str> = g.lines().collect();
        assert_eq!(lines.len(), 2, "{g:?}");
        assert!(lines[0].starts_with("Menu"), "{g:?}");
        let col = lines[0].find("Content").unwrap();
        assert!((28..=32).contains(&col), "Content at ~col 30: {col} in {g:?}");
        assert_eq!(lines[1].trim(), "Below");
    }

    #[test]
    fn big_vertical_gap_becomes_blank_line() {
        let spans = [
            span("Header", 0.0, 0.0, 60.0, 12.0),
            span("Footer", 0.0, 100.0, 60.0, 112.0),
        ];
        let g = grid(&spans);
        assert_eq!(g, "Header\n\nFooter", "{g:?}");
    }

    #[test]
    fn colliding_estimates_keep_one_space_apart() {
        // Two snippets whose column estimates overlap must not concatenate.
        let spans = [
            span("LongLeftText", 0.0, 0.0, 120.0, 12.0),
            span("Right", 122.0, 0.0, 172.0, 12.0),
        ];
        let g = grid(&spans);
        assert!(g.contains("LongLeftText Right"), "{g:?}");
    }

    #[test]
    fn indentation_is_relative_to_char_width() {
        // 5 px/char here; x=50 → column 10.
        let spans = [
            span("aaaaaaaaaa", 0.0, 0.0, 50.0, 10.0),
            span("indented", 50.0, 20.0, 90.0, 30.0),
        ];
        let g = grid(&spans);
        let line2 = g.lines().nth(1).unwrap();
        assert_eq!(line2.find("indented").unwrap(), 10, "{g:?}");
    }

    #[test]
    fn empty_and_degenerate_input() {
        assert_eq!(grid(&[]), "");
        assert_eq!(grid(&[span("", 0.0, 0.0, 10.0, 10.0)]), "");
    }

    #[test]
    fn far_right_box_is_capped() {
        let spans = [
            span("x", 0.0, 0.0, 8.0, 10.0),
            span("far", 100_000.0, 0.0, 100_024.0, 10.0),
        ];
        let g = grid(&spans);
        assert!(g.lines().next().unwrap().chars().count() <= MAX_COL + 3, "{g:?}");
    }
}
