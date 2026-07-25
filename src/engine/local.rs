//! Local PP-OCR recognition, built entirely on docling-pdf's *public*
//! `ocr_prep` API plus our own onnxruntime sessions — the upstream repo is a
//! read-only dependency here.
//!
//! Two model generations, auto-selected by what's on disk:
//!
//! - **PP-OCRv5 det+rec** (preferred): a DBNet text-detection pass finds
//!   every text box (see `det.rs`), each box is cropped and recognized by
//!   the v5 CTC recognizer with its 18k-char multilingual dictionary. This
//!   is what makes GUI screenshots readable — the v3 setup below misses most
//!   of their text for want of a detector. Files looked up in the models
//!   dir: `ppocrv5_mobile_det.onnx`, `ppocrv5_mobile_rec.onnx`,
//!   `ppocrv5_dict.txt` (export with paddle2onnx from the official PaddleX
//!   models; see README). `--img-ocr-lang` is ignored — the v5 dictionary is
//!   multilingual.
//! - **PP-OCRv3 recognition-only** (fallback): whole-image line segmentation
//!   by ink projection (`prep_page_lines`, docling-pdf's wasm path) + the
//!   en/ch v3 recognizer docling.rs already ships. Adequate for clean
//!   single-column text; blind on busy screenshots.

use std::path::{Path, PathBuf};

use docling_pdf::ocr_prep::{
    batch_input, decode_row, dict_chars, normalize_polarity, prep_line, prep_page_lines,
    width_batches, PrepLine, REC_HEIGHT,
};
use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use super::det::{det_boxes, det_preprocess, order_boxes};
use super::{OcrEngine, OcrFailure, OcrText};
use crate::layout::{self, Span};

/// Which model files this engine resolved to.
#[derive(Debug, Clone)]
enum Models {
    V5 { det: PathBuf, rec: PathBuf, dict: PathBuf },
    V3 { rec: PathBuf, dict: PathBuf },
}

struct Ready {
    /// Detection session — `Some` only in v5 mode.
    det: Option<Session>,
    rec: Session,
    chars: Vec<String>,
}

enum State {
    Unloaded,
    /// Load failed once — remembered so the error surfaces once, not per image
    /// (the runner also kills the engine on the fatal error, this is belt and
    /// suspenders against a fresh runner reusing the engine).
    Missing,
    Ready(Box<Ready>),
}

pub struct LocalPpocr {
    lang: String,
    models: Models,
    state: State,
}

impl LocalPpocr {
    /// `models_dir` is where the model files live — typically the docling.rs
    /// checkout's `models/` directory. The v5 triple wins when present;
    /// explicit env paths win over everything:
    /// `DOCMILL_{DET_ONNX,REC_ONNX,DICT}` select a v5 set,
    /// `DOCLING_OCR_REC_ONNX`/`DOCLING_OCR_DICT` the v3 pair (same contract
    /// as docling-pdf itself). `lang` (`en`/`ch`) applies to v3 only.
    pub fn new(lang: &str, models_dir: &Path) -> Self {
        Self {
            lang: lang.to_string(),
            models: resolve_models(lang, models_dir),
            state: State::Unloaded,
        }
    }

    fn load(&mut self) -> Result<(), String> {
        // Single-threaded intra-op for the same reason docling-pdf pins it:
        // multi-threaded float-reduction order flips CTC argmax on
        // low-confidence characters, making output run-to-run nondeterministic.
        let session = |path: &Path| -> Result<Session, String> {
            Session::builder()
                .map_err(|e| format!("ort builder: {e}"))?
                .with_intra_threads(1)
                .map_err(|e| format!("ort intra_threads: {e}"))?
                .commit_from_file(path)
                .map_err(|e| format!("load {}: {e}", path.display()))
        };
        let (det, rec_path, dict_path) = match &self.models {
            Models::V5 { det, rec, dict } => (Some(session(det)?), rec.clone(), dict.clone()),
            Models::V3 { rec, dict } => (None, rec.clone(), dict.clone()),
        };
        let rec = session(&rec_path)?;
        let dict = std::fs::read_to_string(&dict_path)
            .map_err(|e| format!("read dict {}: {e}", dict_path.display()))?;
        self.state = State::Ready(Box::new(Ready {
            det,
            rec,
            chars: dict_chars(&dict),
        }));
        Ok(())
    }
}

/// Pick v5 (env triple, else on-disk triple) or fall back to the v3 pair.
fn resolve_models(lang: &str, models_dir: &Path) -> Models {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty()).map(PathBuf::from);
    if let (Some(det), Some(rec), Some(dict)) = (
        env("DOCMILL_DET_ONNX"),
        env("DOCMILL_REC_ONNX"),
        env("DOCMILL_DICT"),
    ) {
        return Models::V5 { det, rec, dict };
    }
    let v5 = (
        models_dir.join("ppocrv5_mobile_det.onnx"),
        models_dir.join("ppocrv5_mobile_rec.onnx"),
        models_dir.join("ppocrv5_dict.txt"),
    );
    if v5.0.exists() && v5.1.exists() && v5.2.exists() {
        return Models::V5 { det: v5.0, rec: v5.1, dict: v5.2 };
    }
    let (rec, dict) = resolve_rec_pair(lang, models_dir);
    Models::V3 { rec, dict }
}

/// Resolve the v3 model/dictionary pair for `lang` under `models_dir`, with
/// the same degradation docling-pdf applies: a missing English pair falls
/// back to the multilingual `ch_` pair (weaker Latin word spacing) with a
/// warning.
fn resolve_rec_pair(lang: &str, models_dir: &Path) -> (PathBuf, PathBuf) {
    const CH: (&str, &str) = ("ocr_rec.onnx", "ppocr_keys_v1.txt");
    const EN: (&str, &str) = ("ocr_rec_en.onnx", "en_dict.txt");
    let want_ch = lang.eq_ignore_ascii_case("ch");
    let pick = if want_ch { CH } else { EN };
    let (mut onnx, mut dict) = (models_dir.join(pick.0), models_dir.join(pick.1));
    if !want_ch && (!onnx.exists() || !dict.exists()) {
        let (ch_onnx, ch_dict) = (models_dir.join(CH.0), models_dir.join(CH.1));
        if ch_onnx.exists() && ch_dict.exists() {
            eprintln!(
                "docmill: English OCR model not found ({}); falling back to the multilingual \
                 ch_ model — expect weak Latin word spacing",
                onnx.display()
            );
            (onnx, dict) = (ch_onnx, ch_dict);
        }
    }
    (
        std::env::var("DOCLING_OCR_REC_ONNX").map(PathBuf::from).unwrap_or(onnx),
        std::env::var("DOCLING_OCR_DICT").map(PathBuf::from).unwrap_or(dict),
    )
}

/// Swap an RGB image to BGR in place — the PP-OCRv5 models consume the
/// cv2-decoded (BGR) channel order; the v3 path in docling.rs feeds RGB and
/// is left exactly as validated there.
fn to_bgr(mut img: RgbImage) -> RgbImage {
    for px in img.pixels_mut() {
        px.0.swap(0, 2);
    }
    img
}

/// Recognize prepared line crops with deterministic same-width batching —
/// shared by both model generations.
fn recognize(rec: &mut Session, chars: &[String], lines: &[PrepLine]) -> Result<Vec<String>, String> {
    let mut texts = vec![String::new(); lines.len()];
    for (w, chunk) in width_batches(lines) {
        let data = batch_input(w, &chunk, lines);
        let input = Tensor::from_array(([chunk.len(), 3, REC_HEIGHT as usize, w], data))
            .map_err(|e| format!("input tensor: {e}"))?;
        let outputs = rec
            .run(ort::inputs!["x" => input])
            .map_err(|e| format!("rec inference: {e}"))?;
        let (shape, probs) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract rec: {e}"))?;
        let t_len = shape[1] as usize;
        let nc = shape[2] as usize;
        for (i, &ix) in chunk.iter().enumerate() {
            texts[ix] = decode_row(chars, &probs[i * t_len * nc..(i + 1) * t_len * nc], nc);
        }
    }
    Ok(texts)
}

impl OcrEngine for LocalPpocr {
    fn id(&self) -> &'static str {
        "ppocr"
    }

    fn cache_salt(&self) -> String {
        // Model identity — a different generation, file set, or (v3) language
        // must invalidate cached text.
        match &self.models {
            // "v5g" (not "v5"): the payload gained the grid rendering, so
            // pre-grid cache entries must miss rather than serve gridless.
            Models::V5 { det, rec, dict } => {
                format!("v5g|{}|{}|{}", det.display(), rec.display(), dict.display())
            }
            Models::V3 { rec, dict } => {
                format!("{}|{}|{}", self.lang, rec.display(), dict.display())
            }
        }
    }

    fn ocr(&mut self, bytes: &[u8], _mimetype: &str) -> Result<OcrText, OcrFailure> {
        if matches!(self.state, State::Unloaded) {
            if let Err(e) = self.load() {
                self.state = State::Missing;
                return Err(OcrFailure::Engine(e));
            }
        }
        let State::Ready(ready) = &mut self.state else {
            return Err(OcrFailure::Engine("recognition model unavailable".into()));
        };
        // Per-image decode failure (WMF/EMF, corrupt stream): this engine
        // can't read it, but a remote engine might — don't penalize.
        let img = image::load_from_memory(bytes)
            .map_err(|e| OcrFailure::Image(format!("decode image: {e}")))?
            .to_rgb8();
        let engine_err = OcrFailure::Engine;

        if let Some(det) = ready.det.as_mut() {
            // v5: detect text boxes on the full image, recognize each crop.
            let inp = det_preprocess(&img);
            let input = Tensor::from_array(([1usize, 3, inp.h, inp.w], inp.data))
                .map_err(|e| engine_err(format!("det tensor: {e}")))?;
            let outputs = det
                .run(ort::inputs!["x" => input])
                .map_err(|e| engine_err(format!("det inference: {e}")))?;
            let (_, prob) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| engine_err(format!("extract det: {e}")))?;
            let mut boxes = det_boxes(prob, inp.w, inp.h);
            order_boxes(&mut boxes);
            // Crop each detected box (source pixels), keeping its geometry so
            // recognized texts can be laid back out spatially.
            let mut rects = Vec::new();
            let mut lines = Vec::new();
            for &(l, t, r, b) in &boxes {
                let (iw, ih) = img.dimensions();
                let x0 = (l as f32 * inp.sx) as u32;
                let y0 = (t as f32 * inp.sy) as u32;
                let x1 = ((r as f32 * inp.sx) as u32).min(iw);
                let y1 = ((b as f32 * inp.sy) as u32).min(ih);
                if x1 <= x0 + 2 || y1 <= y0 + 2 {
                    continue;
                }
                let crop = to_bgr(image::imageops::crop_imm(&img, x0, y0, x1 - x0, y1 - y0).to_image());
                if let Some(pl) = prep_line(&crop) {
                    rects.push((x0 as f32, y0 as f32, x1 as f32, y1 as f32));
                    lines.push(pl);
                }
            }
            let texts = recognize(&mut ready.rec, &ready.chars, &lines).map_err(engine_err)?;
            let spans: Vec<Span> = rects
                .iter()
                .zip(&texts)
                .filter(|(_, t)| !t.trim().is_empty())
                .map(|(&(l, t, r, b), text)| Span {
                    text: text.trim().to_string(),
                    l,
                    t,
                    r,
                    b,
                })
                .collect();
            let plain = spans.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join("\n");
            let grid = (!spans.is_empty()).then(|| layout::grid(&spans));
            Ok(OcrText { text: plain, grid })
        } else {
            // v3: no detector — whole-image ink-projection line segmentation,
            // no geometry to lay out.
            let img = normalize_polarity(img);
            let lines = prep_page_lines(&img);
            let texts = recognize(&mut ready.rec, &ready.chars, &lines).map_err(engine_err)?;
            Ok(OcrText::plain(
                texts
                    .iter()
                    .map(|t| t.trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        }
    }
}
