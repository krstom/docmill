//! Flag/env/default resolution for the OCR side of the CLI, and construction
//! of the engine chain. Layering follows the docling.rs house pattern
//! (`VlmOptions::resolve`): explicit CLI value > `DOCMILL_*` env
//! var > default. Misconfiguration that can never work (a remote engine with
//! no endpoint) errors up front; everything that might recover (missing
//! model file, downed server) degrades at OCR time instead.

use std::path::PathBuf;

use crate::cache::OcrCache;
use crate::engine::paddle::PaddleServer;
use crate::engine::vlm::OpenAiVlm;
use crate::engine::{OcrEngine, OcrRunner};
use crate::postprocess::OutputMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Local,
    Vlm,
    Paddle,
}

impl EngineKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "local" | "ppocr" => Some(Self::Local),
            "vlm" => Some(Self::Vlm),
            "paddle" => Some(Self::Paddle),
            _ => None,
        }
    }
}

/// Raw values from the arg parser — all optional, resolution fills the gaps.
#[derive(Debug, Default)]
pub struct CliOverrides {
    pub engine: Option<String>,
    pub mode: Option<String>,
    pub lang: Option<String>,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub prompt: Option<String>,
    pub min_px: Option<String>,
    pub cache_dir: Option<String>,
    pub no_cache: bool,
    pub timeout: Option<String>,
    pub models_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ImgOcrConfig {
    /// Ordered fallback chain.
    pub engines: Vec<EngineKind>,
    pub mode: OutputMode,
    /// Local-engine recognition language (`en`/`ch`).
    pub lang: String,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub prompt: Option<String>,
    pub min_pixels: u32,
    pub cache_dir: Option<PathBuf>,
    pub cache_enabled: bool,
    pub timeout_secs: u64,
    /// Where the local engine's `ocr_rec*.onnx` + dictionaries live.
    pub models_dir: PathBuf,
}

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// Default `.models/` location, mirroring docling-pdf's asset resolver: the
/// current directory when it has one, else next to the (symlink-resolved)
/// executable or its parent — which is what makes the installed tree
/// (`$PREFIX/{bin/docmill, .models/}` with a symlink on PATH) work
/// from any working directory.
fn default_models_dir() -> PathBuf {
    let cwd = PathBuf::from(".models");
    if cwd.exists() {
        return cwd;
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.canonicalize().ok())
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
    {
        for base in [Some(dir.as_path()), dir.parent()].into_iter().flatten() {
            let p = base.join(".models");
            if p.exists() {
                return p;
            }
        }
    }
    cwd
}

impl ImgOcrConfig {
    pub fn resolve(cli: CliOverrides) -> Result<Self, String> {
        let engine_spec = cli
            .engine
            .or_else(|| env("DOCMILL_ENGINE"))
            .unwrap_or_else(|| "local".into());
        let mut engines = Vec::new();
        for part in engine_spec.split(',') {
            let kind = EngineKind::parse(part)
                .ok_or_else(|| format!("--img-ocr-engine: {part:?} is not local|vlm|paddle"))?;
            if !engines.contains(&kind) {
                engines.push(kind);
            }
        }
        if engines.is_empty() {
            return Err("--img-ocr-engine: empty engine list".into());
        }

        let mode = match cli
            .mode
            .or_else(|| env("DOCMILL_MODE"))
            .as_deref()
            .unwrap_or("fence")
        {
            "fence" => OutputMode::Fence,
            "markers" => OutputMode::Markers,
            "text" => OutputMode::Text,
            "quote" => OutputMode::Quote,
            "placeholder" => OutputMode::Placeholder,
            other => {
                return Err(format!(
                    "--img-ocr-mode: {other:?} is not fence|markers|text|quote|placeholder"
                ))
            }
        };

        let lang = cli
            .lang
            .or_else(|| env("DOCMILL_LANG"))
            .unwrap_or_else(|| "en".into());
        if !matches!(lang.as_str(), "en" | "ch") {
            return Err(format!("--img-ocr-lang: {lang:?} is not en|ch"));
        }

        let endpoint = cli.endpoint.or_else(|| env("DOCMILL_ENDPOINT"));
        let model = cli.model.or_else(|| env("DOCMILL_MODEL"));
        let api_key = cli.api_key.or_else(|| env("DOCMILL_API_KEY"));
        let prompt = cli.prompt.or_else(|| env("DOCMILL_PROMPT"));

        // Remote engines can never work without their endpoint — fail now,
        // not after a full document conversion.
        if engines.contains(&EngineKind::Vlm) {
            if endpoint.is_none() {
                return Err("vlm engine needs --img-ocr-endpoint or DOCMILL_ENDPOINT".into());
            }
            if model.is_none() {
                return Err("vlm engine needs --img-ocr-model or DOCMILL_MODEL".into());
            }
        }
        if engines.contains(&EngineKind::Paddle) && endpoint.is_none() {
            return Err("paddle engine needs --img-ocr-endpoint or DOCMILL_ENDPOINT".into());
        }

        let min_pixels = match cli.min_px.or_else(|| env("DOCMILL_MIN_PX")) {
            Some(v) => v
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("--img-ocr-min-px: {v:?} is not a number"))?,
            None => 2500, // ~50×50 — skips icons/bullets/logos
        };

        let timeout_secs = match cli.timeout.or_else(|| env("DOCMILL_TIMEOUT")) {
            Some(v) => v
                .trim()
                .parse::<u64>()
                .map_err(|_| format!("--img-ocr-timeout: {v:?} is not a number of seconds"))?,
            None => 120,
        };

        let cache_dir = cli
            .cache_dir
            .map(PathBuf::from)
            .or_else(|| env("DOCMILL_CACHE_DIR").map(PathBuf::from))
            .or_else(OcrCache::default_dir);
        let cache_enabled = !cli.no_cache;

        let models_dir = cli
            .models_dir
            .map(PathBuf::from)
            .or_else(|| env("DOCMILL_MODELS_DIR").map(PathBuf::from))
            .unwrap_or_else(default_models_dir);

        Ok(Self {
            engines,
            mode,
            lang,
            endpoint,
            model,
            api_key,
            prompt,
            min_pixels,
            cache_dir,
            cache_enabled,
            timeout_secs,
            models_dir,
        })
    }

    /// True when the first (preferred) engine is remote — the standalone-image
    /// fast path then skips the local ML pipeline entirely.
    pub fn remote_first(&self) -> bool {
        !matches!(self.engines.first(), Some(EngineKind::Local) | None)
    }

    pub fn build_runner(&self) -> Result<OcrRunner, String> {
        let mut chain: Vec<Box<dyn OcrEngine>> = Vec::new();
        for kind in &self.engines {
            match kind {
                EngineKind::Local => {
                    #[cfg(feature = "local-ocr")]
                    chain.push(Box::new(crate::engine::local::LocalPpocr::new(
                        &self.lang,
                        &self.models_dir,
                    )));
                    #[cfg(not(feature = "local-ocr"))]
                    eprintln!(
                        "docmill: built without the local-ocr feature; skipping the local engine"
                    );
                }
                EngineKind::Vlm => chain.push(Box::new(OpenAiVlm::new(
                    self.endpoint.clone().expect("validated in resolve"),
                    self.model.clone().expect("validated in resolve"),
                    self.prompt.clone(),
                    self.api_key.clone(),
                    self.timeout_secs,
                ))),
                EngineKind::Paddle => chain.push(Box::new(PaddleServer::new(
                    self.endpoint.clone().expect("validated in resolve"),
                    self.timeout_secs,
                ))),
            }
        }
        if chain.is_empty() {
            return Err("no usable OCR engine in the configured chain".into());
        }
        let cache = match (&self.cache_dir, self.cache_enabled) {
            (Some(dir), true) => OcrCache::new(dir.clone()),
            _ => OcrCache::disabled(),
        };
        Ok(OcrRunner::new(chain, cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: resolution reads DOCMILL_* env vars; tests pass explicit
    // CLI values for everything they assert on, so a developer's environment
    // can't flip outcomes (except the deliberately-env-focused ones, which
    // set/remove their own var — see `endpoint_env_fallback`).

    #[test]
    fn defaults_are_local_markers_en() {
        let cfg = ImgOcrConfig::resolve(CliOverrides {
            engine: Some("local".into()),
            mode: Some("markers".into()),
            lang: Some("en".into()),
            min_px: Some("2500".into()),
            timeout: Some("120".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(cfg.engines, vec![EngineKind::Local]);
        assert_eq!(cfg.mode, OutputMode::Markers);
        assert!(!cfg.remote_first());
        assert_eq!(cfg.min_pixels, 2500);
    }

    #[test]
    fn engine_chain_parses_in_order_and_dedups() {
        let cfg = ImgOcrConfig::resolve(CliOverrides {
            engine: Some("vlm,local,vlm".into()),
            endpoint: Some("http://h/v1".into()),
            model: Some("m".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(cfg.engines, vec![EngineKind::Vlm, EngineKind::Local]);
        assert!(cfg.remote_first());
    }

    #[test]
    fn vlm_without_endpoint_or_model_errors_up_front() {
        let err = ImgOcrConfig::resolve(CliOverrides {
            engine: Some("vlm".into()),
            model: Some("m".into()),
            endpoint: None,
            ..Default::default()
        });
        // Only assert when the environment doesn't provide the endpoint.
        if std::env::var("DOCMILL_ENDPOINT").is_err() {
            assert!(err.is_err());
        }
        let err = ImgOcrConfig::resolve(CliOverrides {
            engine: Some("paddle".into()),
            endpoint: Some("http://h:8866/predict/ocr_system".into()),
            ..Default::default()
        });
        assert!(err.is_ok(), "paddle needs only the endpoint");
    }

    #[test]
    fn bad_values_error() {
        assert!(ImgOcrConfig::resolve(CliOverrides {
            engine: Some("tesseract".into()),
            ..Default::default()
        })
        .is_err());
        assert!(ImgOcrConfig::resolve(CliOverrides {
            mode: Some("fancy".into()),
            ..Default::default()
        })
        .is_err());
        assert!(ImgOcrConfig::resolve(CliOverrides {
            lang: Some("de".into()),
            ..Default::default()
        })
        .is_err());
        assert!(ImgOcrConfig::resolve(CliOverrides {
            min_px: Some("lots".into()),
            ..Default::default()
        })
        .is_err());
    }
}
