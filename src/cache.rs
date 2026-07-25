//! On-disk OCR result cache, keyed by content hash.
//!
//! Key = sha256 over (engine id, engine cache-salt, image bytes) — the salt
//! folds in every parameter that changes an engine's output (model paths,
//! endpoint, prompt), so switching models never serves stale text. Layout is
//! `<dir>/<hh>/<hash>.json` (two-hex-char shard dirs keep any one directory
//! small); each entry is a tiny JSON object `{"text","engine","created"}`.
//!
//! Every IO failure here warns and degrades — a broken cache must never fail
//! or slow a conversion beyond re-running OCR. Writes go through a `.tmp` +
//! rename so a crashed run can't leave a torn entry that later parses as an
//! empty result.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

/// One cached OCR result. `text` may legitimately be empty: an image with no
/// text in it is a *result*, and caching it is what stops a remote engine
/// from being asked about the same logo on every run (negative caching).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedOcr {
    pub text: String,
    /// Layout-preserving rendering, when the engine had box geometry.
    pub grid: Option<String>,
    /// Engine id that produced the text (`ppocr` / `vlm` / `paddle`).
    pub engine: String,
    /// Unix seconds at write time — provenance for debugging, never compared.
    pub created: u64,
}

/// Disk cache handle. A disabled cache (`OcrCache::disabled()`) is a full
/// no-op, so callers never branch on enablement.
pub struct OcrCache {
    dir: Option<PathBuf>,
}

impl OcrCache {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir: Some(dir) }
    }

    pub fn disabled() -> Self {
        Self { dir: None }
    }

    /// The default cache location: `$XDG_CACHE_HOME`/`$HOME/.cache` on
    /// Unix, `%LOCALAPPDATA%`/`%USERPROFILE%\.cache` on Windows, +
    /// `docmill` (hand-rolled — the workspace carries no `dirs`
    /// crate and this is the only path that would need it).
    pub fn default_dir() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
            .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(base.join("docmill"))
    }

    /// Cache key for one (engine, params, image) triple. NUL separators keep
    /// the fields from running into each other ("ab"+"c" vs "a"+"bc").
    pub fn key(engine_id: &str, salt: &str, image: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(engine_id.as_bytes());
        h.update([0u8]);
        h.update(salt.as_bytes());
        h.update([0u8]);
        h.update(image);
        let digest = h.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write as _;
            let _ = write!(hex, "{b:02x}");
        }
        hex
    }

    fn entry_path(&self, key: &str) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        Some(dir.join(&key[..2]).join(format!("{key}.json")))
    }

    pub fn get(&self, key: &str) -> Option<CachedOcr> {
        let path = self.entry_path(key)?;
        let raw = std::fs::read_to_string(&path).ok()?;
        let v: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => {
                // A torn/corrupt entry is treated as a miss; the rewrite after
                // re-OCR replaces it.
                return None;
            }
        };
        Some(CachedOcr {
            text: v.get("text")?.as_str()?.to_string(),
            grid: v.get("grid").and_then(|g| g.as_str()).map(str::to_string),
            engine: v.get("engine")?.as_str()?.to_string(),
            created: v.get("created").and_then(|c| c.as_u64()).unwrap_or(0),
        })
    }

    /// Best-effort write; all failures warn once per message kind and degrade.
    pub fn put(&self, key: &str, entry: &CachedOcr) {
        let Some(path) = self.entry_path(key) else {
            return;
        };
        if let Err(e) = self.put_inner(&path, entry) {
            eprintln!("docmill: cache write {}: {e} (continuing uncached)", path.display());
        }
    }

    fn put_inner(&self, path: &Path, entry: &CachedOcr) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut body = serde_json::json!({
            "text": entry.text,
            "engine": entry.engine,
            "created": entry.created,
        });
        if let Some(grid) = &entry.grid {
            body["grid"] = serde_json::Value::String(grid.clone());
        }
        let body = body.to_string();
        // Atomic-enough replace: temp file in the same directory + rename, so
        // readers only ever see complete entries. Windows' rename refuses to
        // overwrite, hence the remove-and-retry fallback.
        let tmp = path.with_extension("json.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
        }
        std::fs::rename(&tmp, path).or_else(|_| {
            let _ = std::fs::remove_file(path);
            std::fs::rename(&tmp, path)
        })
    }
}

/// Unix seconds now — the `created` stamp for fresh entries.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_and_input_sensitive() {
        let k = OcrCache::key("ppocr", "en", b"imagebytes");
        assert_eq!(k, OcrCache::key("ppocr", "en", b"imagebytes"));
        assert_eq!(k.len(), 64);
        // Every component participates, and the separator prevents field bleed.
        assert_ne!(k, OcrCache::key("vlm", "en", b"imagebytes"));
        assert_ne!(k, OcrCache::key("ppocr", "ch", b"imagebytes"));
        assert_ne!(k, OcrCache::key("ppocr", "en", b"imagebyteS"));
        assert_ne!(OcrCache::key("a", "bc", b"d"), OcrCache::key("ab", "c", b"d"));
    }

    #[test]
    fn roundtrip_and_shard_layout() {
        let dir = tempfile::tempdir().unwrap();
        let cache = OcrCache::new(dir.path().to_path_buf());
        let key = OcrCache::key("vlm", "salt", b"png");
        assert!(cache.get(&key).is_none());
        let entry = CachedOcr {
            text: "hello\nworld".into(),
            grid: Some("hello   world".into()),
            engine: "vlm".into(),
            created: 1234,
        };
        cache.put(&key, &entry);
        assert_eq!(cache.get(&key), Some(entry));
        // Sharded path: <dir>/<hh>/<hash>.json
        let shard = dir.path().join(&key[..2]);
        assert!(shard.join(format!("{key}.json")).exists());
    }

    #[test]
    fn empty_text_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let cache = OcrCache::new(dir.path().to_path_buf());
        let key = OcrCache::key("ppocr", "", b"logo");
        cache.put(
            &key,
            &CachedOcr {
                text: String::new(),
                grid: None,
                engine: "ppocr".into(),
                created: 1,
            },
        );
        let hit = cache.get(&key).expect("negative result is cached");
        assert_eq!(hit.text, "");
        assert_eq!(hit.grid, None, "absent grid field reads back as None");
    }

    #[test]
    fn corrupt_entry_is_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = OcrCache::new(dir.path().to_path_buf());
        let key = OcrCache::key("ppocr", "s", b"x");
        let shard = dir.path().join(&key[..2]);
        std::fs::create_dir_all(&shard).unwrap();
        std::fs::write(shard.join(format!("{key}.json")), b"{not json").unwrap();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        let cache = OcrCache::disabled();
        let key = OcrCache::key("ppocr", "s", b"x");
        cache.put(
            &key,
            &CachedOcr {
                text: "t".into(),
                grid: None,
                engine: "ppocr".into(),
                created: 1,
            },
        );
        assert!(cache.get(&key).is_none());
    }
}
