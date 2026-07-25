//! Remote OCR through a PaddleOCR HTTP server.
//!
//! Two serving generations are in the wild and their wire shapes differ:
//!
//! - **PaddleHub serving** (`hub serving start -m ocr_system`, default port
//!   8866): request `{"images": ["<base64>"]}`, response
//!   `{"results": [[{"text": "...", "confidence": ...}, …]]}`.
//! - **PaddleX / PaddleOCR 3.x serving**: request `{"file": "<base64>",
//!   "fileType": 1}`, response `{"result": {"ocrResults": [{"prunedResult":
//!   {"rec_texts": ["…"]}}]}}`.
//!
//! The engine sends the PaddleHub shape first and retries once with the
//! PaddleX shape on a 4xx (wrong-schema rejections come back as 400/422);
//! parsing is tolerant — known shapes first, then a recursive scan for
//! `text`/`rec_texts` fields — so minor server-version drift degrades to
//! "still extracts the text" rather than an error.

use std::time::Duration;

use super::{OcrEngine, OcrFailure, OcrText};
use crate::layout::{self, Span};

pub struct PaddleServer {
    endpoint: String,
    agent: ureq::Agent,
}

impl PaddleServer {
    pub fn new(endpoint: String, timeout_secs: u64) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            .http_status_as_error(false)
            .build()
            .into();
        Self { endpoint, agent }
    }

    /// One POST; returns (status, body-text).
    fn post(&self, payload: &str) -> Result<(u16, String), String> {
        let mut resp = self
            .agent
            .post(&self.endpoint)
            .header("content-type", "application/json")
            .send(payload.as_bytes())
            .map_err(|e| format!("{}: {e}", self.endpoint))?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("{}: read response: {e}", self.endpoint))?;
        Ok((status, text))
    }
}

/// Extract recognized lines — and, when the response carries box geometry, a
/// layout-preserving grid — from any known PaddleOCR serving response shape.
pub fn parse_paddle_response(body: &str) -> Result<OcrText, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("malformed JSON response: {e}"))?;
    // PaddleHub: results[0] = [{text, confidence, text_region}, …]. An empty
    // item list is a textless image — a valid empty result, not a shape miss.
    if let Some(items) = v["results"][0].as_array() {
        let lines: Vec<&str> = items.iter().filter_map(|i| i["text"].as_str()).collect();
        let spans: Vec<Span> = items
            .iter()
            .filter_map(|i| {
                let text = i["text"].as_str()?.trim();
                if text.is_empty() {
                    return None;
                }
                Some(with_bounds(text, poly_bounds(i["text_region"].as_array()?)?))
            })
            .collect();
        return Ok(OcrText {
            text: lines.join("\n"),
            grid: (spans.len() == lines.iter().filter(|l| !l.trim().is_empty()).count()
                && !spans.is_empty())
            .then(|| layout::grid(&spans)),
        });
    }
    // PaddleX serving: result.ocrResults[*].prunedResult { rec_texts,
    // rec_polys | rec_boxes }.
    if let Some(results) = v["result"]["ocrResults"].as_array() {
        let mut lines: Vec<String> = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        let mut geometry_complete = true;
        for r in results {
            let pruned = &r["prunedResult"];
            let texts: Vec<&str> = pruned["rec_texts"]
                .as_array()
                .map(|a| a.iter().filter_map(|t| t.as_str()).collect())
                .unwrap_or_default();
            for (i, text) in texts.iter().enumerate() {
                lines.push(text.to_string());
                if text.trim().is_empty() {
                    continue;
                }
                // rec_polys: [[[x,y]×4]…]; rec_boxes: [[x0,y0,x1,y1]…]
                let bounds = pruned["rec_polys"][i]
                    .as_array()
                    .and_then(|p| poly_bounds(p))
                    .or_else(|| {
                        let b = pruned["rec_boxes"][i].as_array()?;
                        Some((
                            b.get(0)?.as_f64()? as f32,
                            b.get(1)?.as_f64()? as f32,
                            b.get(2)?.as_f64()? as f32,
                            b.get(3)?.as_f64()? as f32,
                        ))
                    });
                match bounds {
                    Some(bx) => spans.push(with_bounds(text.trim(), bx)),
                    None => geometry_complete = false,
                }
            }
        }
        return Ok(OcrText {
            text: lines.join("\n"),
            grid: (geometry_complete && !spans.is_empty()).then(|| layout::grid(&spans)),
        });
    }
    // Last resort: harvest every string under a "text" key or inside a
    // "rec_texts" array, anywhere in the tree. No geometry.
    let mut lines = Vec::new();
    scavenge(&v, &mut lines);
    if lines.is_empty() {
        return Err(format!(
            "unrecognized response shape: {}",
            body.chars().take(200).collect::<String>()
        ));
    }
    Ok(OcrText::plain(lines.join("\n")))
}

/// Axis-aligned bounds of a `[[x,y]…]` polygon.
fn poly_bounds(points: &[serde_json::Value]) -> Option<(f32, f32, f32, f32)> {
    let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in points {
        let x = p.get(0)?.as_f64()? as f32;
        let y = p.get(1)?.as_f64()? as f32;
        l = l.min(x);
        t = t.min(y);
        r = r.max(x);
        b = b.max(y);
    }
    (r > l && b > t).then_some((l, t, r, b))
}

fn with_bounds(text: &str, (l, t, r, b): (f32, f32, f32, f32)) -> Span {
    Span {
        text: text.to_string(),
        l,
        t,
        r,
        b,
    }
}

fn scavenge(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                if k == "text" {
                    if let Some(s) = val.as_str() {
                        out.push(s.to_string());
                        continue;
                    }
                }
                if k == "rec_texts" {
                    if let Some(arr) = val.as_array() {
                        out.extend(arr.iter().filter_map(|t| t.as_str()).map(String::from));
                        continue;
                    }
                }
                scavenge(val, out);
            }
        }
        serde_json::Value::Array(arr) => {
            for val in arr {
                scavenge(val, out);
            }
        }
        _ => {}
    }
}

impl OcrEngine for PaddleServer {
    fn id(&self) -> &'static str {
        "paddle"
    }

    fn cache_salt(&self) -> String {
        // "grid|": the payload gained the layout rendering, so pre-grid cache
        // entries must miss rather than serve gridless.
        format!("grid|{}", self.endpoint)
    }

    fn ocr(&mut self, bytes: &[u8], _mimetype: &str) -> Result<OcrText, OcrFailure> {
        let b64 = docling_core::base64::encode(bytes);
        let hub = serde_json::json!({ "images": [b64] }).to_string();
        let paddlex = serde_json::json!({ "file": b64, "fileType": 1 }).to_string();
        let mut delay = Duration::from_secs(2);
        let mut last_err = String::new();
        for attempt in 0..4 {
            if attempt > 0 {
                std::thread::sleep(delay);
                delay *= 2;
            }
            match self.post(&hub) {
                Ok((200, body)) => {
                    return parse_paddle_response(&body).map_err(OcrFailure::Transient)
                }
                // Schema rejection → speak PaddleX once within this attempt.
                Ok((s, _)) if (400..500).contains(&s) && s != 408 && s != 429 => {
                    match self.post(&paddlex) {
                        Ok((200, body)) => {
                            return parse_paddle_response(&body).map_err(OcrFailure::Transient)
                        }
                        Ok((s2, body)) => {
                            // Both shapes rejected: configuration problem.
                            return Err(OcrFailure::Engine(format!(
                                "{}: HTTP {s} (hub shape) / HTTP {s2} (paddlex shape): {}",
                                self.endpoint,
                                body.chars().take(200).collect::<String>()
                            )));
                        }
                        Err(e) => last_err = e,
                    }
                }
                Ok((s, body)) => {
                    last_err = format!(
                        "{}: HTTP {s} (attempt {}): {}",
                        self.endpoint,
                        attempt + 1,
                        body.replace(['\n', '\r'], " ").chars().take(300).collect::<String>()
                    );
                }
                Err(e) => last_err = format!("{e} (attempt {})", attempt + 1),
            }
        }
        Err(OcrFailure::Transient(format!("giving up after 4 attempts: {last_err}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_paddlehub_shape() {
        let body = r#"{"msg":"","results":[[
            {"confidence":0.99,"text":"INVOICE","text_region":[[1,1],[2,1],[2,2],[1,2]]},
            {"confidence":0.98,"text":"Total: $5","text_region":[[1,3],[2,3],[2,4],[1,4]]}
        ]],"status":"000"}"#;
        let out = parse_paddle_response(body).unwrap();
        assert_eq!(out.text, "INVOICE\nTotal: $5");
        assert!(out.grid.is_some(), "text_region present → grid rendered");
    }

    #[test]
    fn parses_paddlehub_empty_result() {
        // A textless image: results[0] exists but is empty — valid empty text.
        let body = r#"{"results":[[]],"status":"000"}"#;
        assert_eq!(parse_paddle_response(body).unwrap(), OcrText::plain(String::new()));
    }

    #[test]
    fn parses_paddlex_shape() {
        let body = r#"{"logId":"x","result":{"ocrResults":[
            {"prunedResult":{"rec_texts":["Line A","Line B"],"rec_scores":[0.9,0.8]}}
        ]}}"#;
        let out = parse_paddle_response(body).unwrap();
        assert_eq!(out.text, "Line A\nLine B");
        // No rec_polys/rec_boxes in this response → no geometry, no grid.
        assert_eq!(out.grid, None);
    }

    #[test]
    fn paddlex_boxes_produce_a_grid() {
        let body = r#"{"result":{"ocrResults":[
            {"prunedResult":{"rec_texts":["Left","Right"],
                             "rec_boxes":[[0,0,40,12],[300,0,360,12]]}}
        ]}}"#;
        let out = parse_paddle_response(body).unwrap();
        assert_eq!(out.text, "Left\nRight");
        let grid = out.grid.expect("rec_boxes → grid");
        // Same row, spatially separated.
        assert_eq!(grid.lines().count(), 1, "{grid:?}");
        assert!(grid.starts_with("Left"), "{grid:?}");
        assert!(grid.find("Right").unwrap() > 10, "{grid:?}");
    }

    #[test]
    fn scavenges_unknown_but_texty_shapes() {
        let body = r#"{"data":{"blocks":[{"text":"found me"},{"text":"me too"}]}}"#;
        assert_eq!(
            parse_paddle_response(body).unwrap(),
            OcrText::plain("found me\nme too".into())
        );
    }

    #[test]
    fn rejects_shapes_with_no_text_at_all() {
        assert!(parse_paddle_response(r#"{"status":"ok"}"#).is_err());
        assert!(parse_paddle_response("not json").is_err());
    }
}
