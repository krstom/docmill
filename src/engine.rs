//! The OCR engine abstraction and the runner that drives a fallback chain
//! through the disk cache.
//!
//! Engines are synchronous (`&mut self` lets the local engine own its ort
//! session lazily) and classify their failures so the runner can tell "this
//! engine is broken, stop asking" from "this one image is beyond it":
//!
//! - [`OcrFailure::Engine`] — the engine itself is unusable (model missing,
//!   endpoint misconfigured): warn once, never call it again this run.
//! - [`OcrFailure::Image`] — only this image failed (undecodable WMF, oversize
//!   payload rejected): fall through to the next engine, don't penalize.
//! - [`OcrFailure::Transient`] — a request failed after its own retries:
//!   warn, and give up on the engine after [`DEAD_AFTER`] consecutive
//!   failures so a downed server doesn't stall every remaining picture.

#[cfg(feature = "local-ocr")]
pub mod det;
#[cfg(feature = "local-ocr")]
pub mod local;
pub mod paddle;
pub mod vlm;

use crate::cache::{now_unix, CachedOcr, OcrCache};

/// How an [`OcrEngine`] call failed — see the module docs for runner behavior.
#[derive(Debug)]
pub enum OcrFailure {
    Engine(String),
    Image(String),
    Transient(String),
}

impl OcrFailure {
    fn message(&self) -> &str {
        match self {
            OcrFailure::Engine(m) | OcrFailure::Image(m) | OcrFailure::Transient(m) => m,
        }
    }
}

/// What an engine recognized: the flat reading-order text, plus — when the
/// engine has box geometry (local v5 detection, paddle regions) — a
/// layout-preserving monospace rendering (see `layout.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrText {
    pub text: String,
    pub grid: Option<String>,
}

impl OcrText {
    pub fn plain(text: String) -> Self {
        Self { text, grid: None }
    }
}

/// One OCR backend. `cache_salt` must fold in every parameter that changes
/// the engine's output (model paths, endpoint, prompt), so cache entries
/// can't outlive a config change. `Send` so a serve worker thread can own
/// the runner (every real engine is: ort sessions and ureq agents are Send).
pub trait OcrEngine: Send {
    fn id(&self) -> &'static str;
    fn cache_salt(&self) -> String;
    fn ocr(&mut self, bytes: &[u8], mimetype: &str) -> Result<OcrText, OcrFailure>;
}

/// A successful OCR outcome. `text` may be empty — a legitimate "no text in
/// this image" answer, cached like any other (see `cache.rs`).
pub struct OcrOutcome {
    pub text: String,
    pub grid: Option<String>,
    pub engine: String,
    pub from_cache: bool,
}

/// Consecutive [`OcrFailure::Transient`] failures before an engine is
/// declared dead for the rest of the run.
const DEAD_AFTER: u32 = 3;

struct Slot {
    engine: Box<dyn OcrEngine>,
    dead: bool,
    consecutive_failures: u32,
}

/// Drives the ordered engine chain through the cache: per image, the first
/// cache hit (in chain order) wins; on a full miss, the first engine that
/// returns `Ok` — including an empty string — wins and is cached.
pub struct OcrRunner {
    slots: Vec<Slot>,
    cache: OcrCache,
}

impl OcrRunner {
    pub fn new(engines: Vec<Box<dyn OcrEngine>>, cache: OcrCache) -> Self {
        Self {
            slots: engines
                .into_iter()
                .map(|engine| Slot {
                    engine,
                    dead: false,
                    consecutive_failures: 0,
                })
                .collect(),
            cache,
        }
    }

    /// OCR one image. `None` means every engine failed (or was dead) — the
    /// caller leaves the picture untouched.
    pub fn run(&mut self, bytes: &[u8], mimetype: &str) -> Option<OcrOutcome> {
        // Cache pass first, in chain order: a hit for the *first* configured
        // engine must win even if a later engine also has one, so the chain
        // order stays meaningful across runs.
        for slot in &self.slots {
            let key = OcrCache::key(slot.engine.id(), &slot.engine.cache_salt(), bytes);
            if let Some(hit) = self.cache.get(&key) {
                return Some(OcrOutcome {
                    text: hit.text,
                    grid: hit.grid,
                    engine: hit.engine,
                    from_cache: true,
                });
            }
        }
        for slot in &mut self.slots {
            if slot.dead {
                continue;
            }
            let id = slot.engine.id();
            match slot.engine.ocr(bytes, mimetype) {
                Ok(out) => {
                    slot.consecutive_failures = 0;
                    let key = OcrCache::key(id, &slot.engine.cache_salt(), bytes);
                    self.cache.put(
                        &key,
                        &CachedOcr {
                            text: out.text.clone(),
                            grid: out.grid.clone(),
                            engine: id.to_string(),
                            created: now_unix(),
                        },
                    );
                    return Some(OcrOutcome {
                        text: out.text,
                        grid: out.grid,
                        engine: id.to_string(),
                        from_cache: false,
                    });
                }
                Err(OcrFailure::Engine(msg)) => {
                    eprintln!("docmill: {id}: {msg} — disabling this engine for the run");
                    slot.dead = true;
                }
                Err(fail @ OcrFailure::Image(_)) => {
                    eprintln!("docmill: {id}: {} — trying next engine", fail.message());
                }
                Err(fail @ OcrFailure::Transient(_)) => {
                    slot.consecutive_failures += 1;
                    if slot.consecutive_failures >= DEAD_AFTER {
                        eprintln!(
                            "docmill: {id}: {} — {DEAD_AFTER} consecutive failures, disabling \
                             this engine for the run",
                            fail.message()
                        );
                        slot.dead = true;
                    } else {
                        eprintln!("docmill: {id}: {}", fail.message());
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Scripted engine for runner/post-processor tests: pops one response per
    /// call. Shared with `postprocess` tests via `pub(crate)`. `grid`, when
    /// set, is attached to every `Ok` response (a geometry-capable engine).
    /// Send-safe internals (Mutex/atomics) because [`OcrEngine`] is `Send`.
    pub(crate) struct MockEngine {
        pub id: &'static str,
        pub salt: String,
        pub grid: Option<String>,
        pub responses: std::sync::Mutex<Vec<Result<String, OcrFailure>>>,
        pub calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockEngine {
        pub(crate) fn new(
            id: &'static str,
            responses: Vec<Result<String, OcrFailure>>,
        ) -> (Self, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
            let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            (
                Self {
                    id,
                    salt: String::new(),
                    grid: None,
                    responses: std::sync::Mutex::new(responses),
                    calls: calls.clone(),
                },
                calls,
            )
        }
    }

    impl OcrEngine for MockEngine {
        fn id(&self) -> &'static str {
            self.id
        }
        fn cache_salt(&self) -> String {
            self.salt.clone()
        }
        fn ocr(&mut self, _bytes: &[u8], _mimetype: &str) -> Result<OcrText, OcrFailure> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut r = self.responses.lock().unwrap();
            let text = if r.is_empty() {
                Ok("default".into())
            } else {
                r.remove(0)
            }?;
            Ok(OcrText {
                text,
                grid: self.grid.clone(),
            })
        }
    }

    fn runner_with(engines: Vec<Box<dyn OcrEngine>>, dir: &std::path::Path) -> OcrRunner {
        OcrRunner::new(engines, OcrCache::new(dir.to_path_buf()))
    }

    #[test]
    fn first_ok_wins_and_is_cached() {
        let dir = tempfile::tempdir().unwrap();
        let (a, a_calls) = MockEngine::new("a", vec![Ok("text-a".into())]);
        let (b, b_calls) = MockEngine::new("b", vec![Ok("text-b".into())]);
        let mut runner = runner_with(vec![Box::new(a), Box::new(b)], dir.path());
        let out = runner.run(b"img", "image/png").unwrap();
        assert_eq!(out.text, "text-a");
        assert!(!out.from_cache);
        assert_eq!(b_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        // Second run: served from cache, no engine calls.
        let out2 = runner.run(b"img", "image/png").unwrap();
        assert!(out2.from_cache);
        assert_eq!(out2.text, "text-a");
        assert_eq!(a_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn engine_failure_falls_through_the_chain() {
        let dir = tempfile::tempdir().unwrap();
        let (a, _) = MockEngine::new("a", vec![Err(OcrFailure::Image("undecodable".into()))]);
        let (b, _) = MockEngine::new("b", vec![Ok("from-b".into())]);
        let mut runner = runner_with(vec![Box::new(a), Box::new(b)], dir.path());
        let out = runner.run(b"wmf", "image/x-wmf").unwrap();
        assert_eq!((out.text.as_str(), out.engine.as_str()), ("from-b", "b"));
    }

    #[test]
    fn fatal_engine_error_disables_engine() {
        let dir = tempfile::tempdir().unwrap();
        let (a, a_calls) = MockEngine::new("a", vec![Err(OcrFailure::Engine("no model".into()))]);
        let (b, _) = MockEngine::new("b", vec![Ok("b1".into()), Ok("b2".into())]);
        let mut runner = runner_with(vec![Box::new(a), Box::new(b)], dir.path());
        assert_eq!(runner.run(b"one", "image/png").unwrap().text, "b1");
        assert_eq!(runner.run(b"two", "image/png").unwrap().text, "b2");
        // The dead engine was only ever called once.
        assert_eq!(a_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn transient_failures_kill_engine_after_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let responses = (0..5)
            .map(|_| Err(OcrFailure::Transient("timeout".into())))
            .collect();
        let (a, a_calls) = MockEngine::new("a", responses);
        let mut runner = runner_with(vec![Box::new(a)], dir.path());
        for i in 0..5 {
            assert!(runner.run(format!("img{i}").as_bytes(), "image/png").is_none());
        }
        assert_eq!(a_calls.load(std::sync::atomic::Ordering::SeqCst), DEAD_AFTER as usize);
    }

    #[test]
    fn empty_result_is_valid_and_negatively_cached() {
        let dir = tempfile::tempdir().unwrap();
        let (a, a_calls) = MockEngine::new("a", vec![Ok(String::new())]);
        let (b, b_calls) = MockEngine::new("b", vec![]);
        let mut runner = runner_with(vec![Box::new(a), Box::new(b)], dir.path());
        let out = runner.run(b"logo", "image/png").unwrap();
        assert_eq!(out.text, "");
        // Empty does NOT fall through to the next engine…
        assert_eq!(b_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        // …and the second run hits the cache.
        let out2 = runner.run(b"logo", "image/png").unwrap();
        assert!(out2.from_cache);
        assert_eq!(a_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
