//! Remote OCR through an OpenAI-compatible vision endpoint (vLLM, Ollama,
//! LM Studio, or a hosted service running e.g. glm-ocr / Qwen-VL) — the same
//! request loop docling.rs's own VLM pipeline uses (`docling/src/vlm.rs`),
//! with an OCR prompt instead of a DocLang-eliciting one: blocking `ureq`,
//! temperature 0, configurable retries with exponential backoff on transport
//! errors and 408/429/5xx (local timeouts fall through immediately), and a
//! frozen JSON-merge escape hatch for server-specific knobs (`DOCMILL_EXTRA_BODY`).

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
    max_retries: u32,
    extra_body: serde_json::Map<String, serde_json::Value>,
}

impl OpenAiVlm {
    pub fn new(
        endpoint: String,
        model: String,
        prompt: Option<String>,
        api_key: Option<String>,
        timeout_secs: u64,
    ) -> Self {
        let agent = super::remote::agent(timeout_secs);
        Self {
            endpoint,
            model,
            prompt: prompt.unwrap_or_else(|| DEFAULT_OCR_PROMPT.to_string()),
            api_key,
            agent,
            max_retries: 3,
            extra_body: Default::default(),
        }
    }

    pub fn request_options(
        mut self,
        max_retries: u32,
        extra_body: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.max_retries = max_retries;
        self.extra_body = extra_body;
        self
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
        let data_uri = format!(
            "data:{mimetype};base64,{}",
            docling_core::base64::encode(image)
        );
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
        for (key, value) in &self.extra_body {
            body[key] = value.clone();
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
        crate::cache::canonical_json(serde_json::json!({
            "endpoint": self.url(),
            "body": serde_json::from_str::<serde_json::Value>(&self.body(&[], "image/png").expect("serializable request")).expect("JSON request"),
        }))
    }

    fn ocr(&mut self, bytes: &[u8], mimetype: &str) -> Result<OcrText, OcrFailure> {
        let payload = self.body(bytes, mimetype)?;
        let url = self.url();
        let text = super::remote::retry(&url, self.max_retries, || {
            let mut req = self
                .agent
                .post(&url)
                .header("content-type", "application/json");
            if let Some(key) = &self.api_key {
                req = req.header("authorization", &format!("Bearer {key}"));
            }
            let mut response = req.send(payload.as_bytes())?;
            let status = response.status().as_u16();
            Ok((status, response.body_mut().read_to_string()?))
        })?;
        parse_openai_response(&text)
            .map(|text| OcrText::plain(text.trim().to_string()))
            .map_err(OcrFailure::Transient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "serve")]
    #[test]
    fn http_request_uses_frozen_options_and_accepts_the_response() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", server.server_addr());
        let handle = std::thread::spawn(move || {
            let mut request = server
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap()
                .unwrap();
            assert_eq!(request.url(), "/v1/chat/completions");
            let mut body = String::new();
            request.as_reader().read_to_string(&mut body).unwrap();
            let body: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(body["temperature"], 0.25);
            assert_eq!(body["model"], "custom-model");
            assert_eq!(
                body["messages"][0]["content"][1]["image_url"]["url"],
                "data:image/png;base64,aW1hZ2U="
            );
            request
                .respond(tiny_http::Response::from_string(
                    r#"{"choices":[{"message":{"content":"OCR result"}}]}"#,
                ))
                .unwrap();
        });
        let extra = serde_json::from_str(r#"{"temperature":0.25,"model":"custom-model"}"#).unwrap();
        let mut engine =
            OpenAiVlm::new(url, "original".into(), None, None, 2).request_options(0, extra);
        assert_eq!(
            engine.ocr(b"image", "image/png").unwrap().text,
            "OCR result"
        );
        handle.join().unwrap();
    }

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
        assert_eq!(
            e("http://h:1/v1/chat/completions"),
            "http://h:1/v1/chat/completions"
        );
    }
    #[test]
    fn request_options_invalidate_cache_but_retry_policy_and_credentials_do_not() {
        let extra = serde_json::from_str(r#"{"temperature":0.2,"options":{"b":2,"a":1}}"#).unwrap();
        let a = OpenAiVlm::new("http://h/v1".into(), "m".into(), None, None, 120)
            .request_options(3, extra);
        let reordered =
            serde_json::from_str(r#"{"options":{"a":1,"b":2},"temperature":0.2}"#).unwrap();
        let b = OpenAiVlm::new(
            "http://h/v1/".into(),
            "m".into(),
            None,
            Some("secret".into()),
            1,
        )
        .request_options(0, reordered);
        assert_eq!(a.cache_salt(), b.cache_salt());
        assert!(!a.cache_salt().contains("secret"));
        let c = OpenAiVlm::new("http://h/v1".into(), "m".into(), None, None, 120);
        assert_ne!(a.cache_salt(), c.cache_salt());
        let request: serde_json::Value =
            serde_json::from_str(&a.body(b"bytes", "image/png").unwrap()).unwrap();
        assert_eq!(request["temperature"], 0.2);
        assert_eq!(request["options"]["a"], 1);
    }
}
