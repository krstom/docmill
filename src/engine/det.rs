//! DBNet-style text detection pre/post-processing for the PP-OCRv5 mobile
//! detection model, in pure Rust (no OpenCV).
//!
//! Mirrors PaddleOCR's inference config for this model (`inference.yml`):
//! resize long side to 960 with /32-aligned dims, **BGR** channel order,
//! ImageNet mean/std normalization, then `DBPostProcess` with thresh 0.3,
//! box_thresh 0.6, unclip_ratio 1.5. The one simplification: instead of
//! contour tracing + polygon unclipping we run connected-component labeling
//! on the binarized probability map and unclip each component's axis-aligned
//! bounding box (offset = area·ratio / perimeter, the same Vatti formula DB
//! uses). Document screenshots and embedded figures have axis-aligned text,
//! where the two are equivalent; heavily rotated scans are not this engine's
//! job (the VLM engine handles those better anyway).

use image::{imageops, imageops::FilterType, RgbImage};

/// DB binarization threshold (config: `thresh`).
pub const DET_THRESH: f32 = 0.3;
/// Minimum mean probability for a kept box (config: `box_thresh`).
pub const DET_BOX_THRESH: f32 = 0.6;
/// Box expansion ratio (config: `unclip_ratio`).
pub const DET_UNCLIP: f32 = 1.5;
/// Target long side for the detection input (config: `resize_long`).
pub const DET_SIDE: u32 = 960;

/// The detection model's prepared input plus the factors mapping the
/// probability map back to source-image pixels.
pub struct DetInput {
    /// `3 * h * w`, CHW, BGR, ImageNet-normalized.
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    /// Source px per det-map px, horizontal / vertical.
    pub sx: f32,
    pub sy: f32,
}

/// Resize (long side → [`DET_SIDE`], dims /32-aligned), normalize, CHW/BGR.
pub fn det_preprocess(img: &RgbImage) -> DetInput {
    let (ow, oh) = img.dimensions();
    let scale = DET_SIDE as f32 / ow.max(oh).max(1) as f32;
    // /32-aligned like DetResizeForTest, min one block.
    let w = (((ow as f32 * scale / 32.0).round() as u32).max(1) * 32).max(32);
    let h = (((oh as f32 * scale / 32.0).round() as u32).max(1) * 32).max(32);
    let resized = imageops::resize(img, w, h, FilterType::Triangle);
    // PaddleOCR feeds the BGR-decoded image straight into NormalizeImage, so
    // the mean/std triples apply in BGR channel order.
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let n = (w * h) as usize;
    let mut data = vec![0f32; 3 * n];
    for (i, px) in resized.pixels().enumerate() {
        let bgr = [px[2], px[1], px[0]];
        for c in 0..3 {
            data[c * n + i] = (bgr[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    DetInput {
        data,
        w: w as usize,
        h: h as usize,
        sx: ow as f32 / w as f32,
        sy: oh as f32 / h as f32,
    }
}

/// A detected text region in det-map pixels, `(l, t, r, b)` half-open.
pub type DetBox = (u32, u32, u32, u32);

/// DB post-process: binarize the probability map, label connected components,
/// score each by its mean probability, drop weak/tiny ones, unclip the rest.
pub fn det_boxes(prob: &[f32], w: usize, h: usize) -> Vec<DetBox> {
    let mut label = vec![false; w * h];
    for (i, &p) in prob.iter().enumerate().take(w * h) {
        label[i] = p > DET_THRESH;
    }
    let mut seen = vec![false; w * h];
    let mut boxes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !label[start] || seen[start] {
            continue;
        }
        // Flood-fill one component, tracking bbox and probability mass.
        let (mut l, mut t, mut r, mut b) = (w, h, 0usize, 0usize);
        let (mut sum, mut count) = (0f32, 0usize);
        stack.push(start);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            l = l.min(x);
            r = r.max(x + 1);
            t = t.min(y);
            b = b.max(y + 1);
            sum += prob[i];
            count += 1;
            if x > 0 && label[i - 1] && !seen[i - 1] {
                seen[i - 1] = true;
                stack.push(i - 1);
            }
            if x + 1 < w && label[i + 1] && !seen[i + 1] {
                seen[i + 1] = true;
                stack.push(i + 1);
            }
            if y > 0 && label[i - w] && !seen[i - w] {
                seen[i - w] = true;
                stack.push(i - w);
            }
            if y + 1 < h && label[i + w] && !seen[i + w] {
                seen[i + w] = true;
                stack.push(i + w);
            }
        }
        // Noise and low-confidence blobs out; then undo DB's training-time
        // shrink by expanding the box.
        if count < 10 || (r - l) < 3 || (b - t) < 3 || (sum / count as f32) < DET_BOX_THRESH {
            continue;
        }
        boxes.push(unclip((l as u32, t as u32, r as u32, b as u32), w as u32, h as u32));
    }
    boxes
}

/// Expand a box by DB's unclip offset `area·ratio / perimeter`, clamped.
fn unclip(bx: DetBox, w: u32, h: u32) -> DetBox {
    let (l, t, r, b) = bx;
    let (bw, bh) = ((r - l) as f32, (b - t) as f32);
    let d = (bw * bh * DET_UNCLIP / (2.0 * (bw + bh))).round() as u32;
    (
        l.saturating_sub(d),
        t.saturating_sub(d),
        (r + d).min(w),
        (b + d).min(h),
    )
}

/// Sort boxes into reading order: group into rows by vertical-center
/// proximity (half the running row height), left-to-right within a row.
pub fn order_boxes(boxes: &mut [DetBox]) {
    boxes.sort_by_key(|&(_, t, _, b)| (t + b) / 2);
    let mut rows: Vec<Vec<DetBox>> = Vec::new();
    for &bx in boxes.iter() {
        let cy = (bx.1 + bx.3) as f32 / 2.0;
        match rows.last_mut() {
            Some(row) => {
                let (rt, rb) = (row[0].1 as f32, row[0].3 as f32);
                if (cy - (rt + rb) / 2.0).abs() < (rb - rt) / 2.0 {
                    row.push(bx);
                } else {
                    rows.push(vec![bx]);
                }
            }
            None => rows.push(vec![bx]),
        }
    }
    let mut i = 0;
    for row in &mut rows {
        row.sort_by_key(|&(l, ..)| l);
        for &bx in row.iter() {
            boxes[i] = bx;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prob map with two strong blobs on one row and one on the next.
    fn map_3blobs(w: usize, h: usize) -> Vec<f32> {
        let mut p = vec![0f32; w * h];
        let mut blob = |x0: usize, y0: usize, x1: usize, y1: usize| {
            for y in y0..y1 {
                for x in x0..x1 {
                    p[y * w + x] = 0.9;
                }
            }
        };
        blob(50, 10, 90, 20); // row 1, right
        blob(5, 10, 40, 20); // row 1, left
        blob(5, 40, 60, 52); // row 2
        p
    }

    #[test]
    fn finds_scores_and_unclips_components() {
        let (w, h) = (100, 60);
        let mut boxes = det_boxes(&map_3blobs(w, h), w, h);
        assert_eq!(boxes.len(), 3);
        order_boxes(&mut boxes);
        // Reading order: row 1 left → right, then row 2.
        assert!(boxes[0].0 < boxes[1].0 && boxes[0].1 == boxes[1].1);
        assert!(boxes[2].1 > boxes[0].1);
        // Unclip grew the left blob (5,10,40,20) outward.
        let (l, t, r, b) = boxes[0];
        assert!(l < 5 && t < 10 && r > 40 && b > 20, "unclipped: {boxes:?}");
    }

    #[test]
    fn weak_and_tiny_blobs_are_dropped() {
        let (w, h) = (64, 64);
        let mut p = vec![0f32; w * h];
        // Strong but tiny (2x2 = 4 px < 10).
        for y in 2..4 {
            for x in 2..4 {
                p[y * w + x] = 0.95;
            }
        }
        // Big but weak (mean 0.4 < box_thresh 0.6).
        for y in 20..30 {
            for x in 10..50 {
                p[y * w + x] = 0.4;
            }
        }
        assert!(det_boxes(&p, w, h).is_empty());
    }

    #[test]
    fn preprocess_shapes_and_scale_factors() {
        let img = RgbImage::from_pixel(1920, 1080, image::Rgb([255, 255, 255]));
        let inp = det_preprocess(&img);
        assert_eq!(inp.w, 960);
        assert_eq!(inp.h, 544); // 1080*0.5 = 540 → /32-rounded = 544
        assert_eq!(inp.data.len(), 3 * inp.w * inp.h);
        assert!((inp.sx - 2.0).abs() < 0.01);
        // White image normalizes to (1 - mean)/std per channel, B first.
        let n = inp.w * inp.h;
        assert!((inp.data[0] - (1.0 - 0.485) / 0.229).abs() < 1e-3);
        assert!((inp.data[n] - (1.0 - 0.456) / 0.224).abs() < 1e-3);
    }
}
