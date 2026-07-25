//! Remote OCR through an OpenAI-compatible vision endpoint (vLLM, Ollama,
//! LM Studio, or a hosted service running e.g. glm-ocr / Qwen-VL) — the same
//! request loop docling.rs's own VLM pipeline uses (`docling/src/vlm.rs`),
//! with an OCR prompt instead of a DocLang-eliciting one: blocking `ureq`,
//! temperature 0, ×4 retry with exponential backoff on transport errors and
//! 408/429/5xx, and a JSON-merge escape hatch for server-specific knobs
//! (`DOCMILL_EXTRA_BODY`).

use std::time::Duration;

use super::{OcrEngine, OcrFailure, OcrText};

/// The transcription instruction sent with every image.
pub const DEFAULT_OCR_PROMPT: &str = "Transcribe all text visible in this image exactly as \
     written, preserving line breaks. Output only the transcribed text, nothing else. If the \
     image contains no text, output nothing.";

/// Completion budget: an image crop's text is far smaller than a dense PDF
/// page, but tables/screenshot dumps run long — 4096 leaves headroom.
const MAX_TOKENS: usize = 4096;

pub struct OpenAiVlm {
    endpoint: String,
    model: String,
    prompt: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl OpenAiVlm {
    pub fn new(
        endpoint: String,
        model: String,
        prompt: Option<String>,
        api_key: Option<String>,
        timeout_secs: u64,
    ) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            // Keep non-2xx as inspectable responses for the retry decision.
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            endpoint,
            model,
            prompt: prompt.unwrap_or_else(|| DEFAULT_OCR_PROMPT.to_string()),
            api_key,
            agent,
        }
    }

    fn url(&self) -> String {
        let base = self.endpoint.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }

    fn body(&self, image: &[u8], mimetype: &str) -> Result<String, OcrFailure> {
        let data_uri = format!("data:{mimetype};base64,{}", docling_core::base64::encode(image));
        let mut body = serde_json::json!({
            "model": self.model,
            // Transcription is a deterministic task; sampling noise only hurts.
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": self.prompt },
                    { "type": "image_url", "image_url": { "url": data_uri } },
                ],
            }],
        });
        if let Ok(extra) = std::env::var("DOCMILL_EXTRA_BODY") {
            match serde_json::from_str::<serde_json::Value>(&extra) {
                Ok(serde_json::Value::Object(map)) => {
                    for (k, v) in map {
                        body[k] = v;
                    }
                }
                _ => {
                    return Err(OcrFailure::Engine(
                        "DOCMILL_EXTRA_BODY is not a JSON object; fix or unset it".into(),
                    ))
                }
            }
        }
        Ok(body.to_string())
    }
}

/// Pull `choices[0].message.content` out of a chat-completions response.
pub fn parse_openai_response(body: &str) -> Result<String, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("malformed JSON response: {e}"))?;
    parsed["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "no choices[0].message.content in response".to_string())
}

impl OcrEngine for OpenAiVlm {
    fn id(&self) -> &'static str {
        "vlm"
    }

    fn cache_salt(&self) -> String {
        format!("{}|{}|{}", self.endpoint, self.model, self.prompt)
    }

    fn ocr(&mut self, bytes: &[u8], mimetype: &str) -> Result<OcrText, OcrFailure> {
        let payload = self.body(bytes, mimetype)?;
        let url = self.url();
        let mut delay = Duration::from_secs(2);
        let mut last_err = String::new();
        for attempt in 0..4 {
            if attempt > 0 {
                std::thread::sleep(delay);
                delay *= 2;
            }
            let mut req = self.agent.post(&url).header("content-type", "application/json");
            if let Some(key) = &self.api_key {
                req = req.header("authorization", &format!("Bearer {key}"));
            }
            match req.send(payload.as_bytes()) {
                Ok(mut resp) => {
                    let status = resp.status().as_u16();
                    let text = resp
                        .body_mut()
                        .read_to_string()
                        .map_err(|e| OcrFailure::Transient(format!("{url}: read response: {e}")))?;
                    if status == 408 || status == 429 || status >= 500 {
                        // Keep a body snippet — quota errors vs plain rate
                        // limits live there.
                        last_err = format!(
                            "{url}: HTTP {status} (attempt {}): {}",
                            attempt + 1,
                            text.replace(['\n', '\r'], " ").chars().take(300).collect::<String>()
                        );
                        continue;
                    }
                    if status != 200 {
                        // 401/403/404/422: configuration, not weather — the
                        // runner disables the engine rather than hammering.
                        return Err(OcrFailure::Engine(format!(
                            "{url}: HTTP {status}: {}",
                            text.chars().take(300).collect::<String>()
                        )));
                    }
                    // No geometry from a chat endpoint; VLMs tend to keep the
                    // source's line structure in the text itself.
                    return parse_openai_response(&text)
                        .map(|t| OcrText::plain(t.trim().to_string()))
                        .map_err(|e| OcrFailure::Transient(format!("{url}: {e}")));
                }
                Err(e) => last_err = format!("{url}: {e} (attempt {})", attempt + 1),
            }
        }
        Err(OcrFailure::Transient(format!("giving up after 4 attempts: {last_err}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_shape() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"Hello\nWorld"}}]}"#;
        assert_eq!(parse_openai_response(body).unwrap(), "Hello\nWorld");
    }

    #[test]
    fn rejects_missing_content_and_garbage() {
        assert!(parse_openai_response(r#"{"choices":[]}"#).is_err());
        assert!(parse_openai_response("not json").is_err());
        assert!(parse_openai_response(r#"{"error":{"message":"boom"}}"#).is_err());
    }

    #[test]
    fn url_appends_chat_completions_once() {
        let e = |ep: &str| OpenAiVlm::new(ep.into(), "m".into(), None, None, 5).url();
        assert_eq!(e("http://h:1/v1"), "http://h:1/v1/chat/completions");
        assert_eq!(e("http://h:1/v1/"), "http://h:1/v1/chat/completions");
        assert_eq!(e("http://h:1/v1/chat/completions"), "http://h:1/v1/chat/completions");
    }
}
