# docmill

Convert documents with [docling.rs](https://github.com/docling-project/docling.rs)
and additionally **OCR the pictures embedded in them** — DOCX drawings, PDF
figure regions, standalone image files — where plain `docling-rs` emits only an
`<!-- image -->` placeholder. Document parsing and format support come from
docling.rs; docmill post-processes the extracted picture nodes with the chosen
OCR engine.

Standalone project: it builds against a sibling `../docling.rs` checkout via
path dependencies and never modifies it. With OCR disabled
(`--img-ocr-mode placeholder`) the output is byte-identical to `docling-rs`.

> docmill is an independent project built on top of docling.rs. It is **not
> affiliated with or endorsed by** the [Docling project](https://github.com/docling-project)
> (LF AI & Data) or IBM.

## ⚠️ Project status & disclaimer

This is a **personal project in an early phase of development**, created to
extend [docling.rs](https://github.com/docling-project/docling.rs) with
functionality it does not currently provide (picture OCR). Please read the
following before using it:

- **Experimental / not thoroughly tested.** Expect bugs, rough edges, and
  breaking changes between versions without notice. Do not rely on it for
  production or critical workloads.
- **Provided "as is"**, without warranty of any kind, express or implied —
  see the [MIT License](LICENSE). **Use at your own risk**; the author is not
  liable for any damage or data loss resulting from its use.
- **No support.** The author does not offer support of any kind. Issues and
  pull requests may go unanswered, and there is no commitment to maintenance,
  fixes, or a release schedule.
- **Temporary by design.** If similar functionality appears in future
  versions of docling.rs, this project will be archived or removed in its
  favor. Do not build long-term dependencies on it.

```
docmill [conversion flags] [--img-ocr-* flags] <input-file>
docmill --input GLOB|DIR --output DIR [--jobs N] [conversion/img-ocr flags]
```

`docmill --help` prints the full flag reference; `--version` the
version.

## Example

```console
$ docmill --img-ocr-engine local --img-ocr-models-dir ../docling.rs/.models invoice.docx
Text before the picture.

<!-- ocr:begin engine=ppocr -->

INVOICE 2024-117 TOTAL 1499 USD
DUE DATE 01 AUGUST 2026

<!-- ocr:end -->

Text after the picture.
```

## OCR engines (`--img-ocr-engine`, ordered fallback chain)

| Engine | What it is | Needs |
|---|---|---|
| `local` | PP-OCR via onnxruntime, in process. Auto-selects **PP-OCRv5 det+rec** (a DBNet detection pass + the 18k-char multilingual recognizer — reads GUI screenshots) when `ppocrv5_mobile_det.onnx` + `ppocrv5_mobile_rec.onnx` + `ppocrv5_dict.txt` are present in `--img-ocr-models-dir`; otherwise falls back to PP-OCRv3 recognition-only (whole-image line segmentation — clean single-column text only) | v5: export the official PaddleX models with `paddle2onnx --model_dir ~/.paddlex/official_models/PP-OCRv5_mobile_det --model_filename inference.json --params_filename inference.pdiparams --save_file ppocrv5_mobile_det.onnx --opset_version 14` (same for `_rec`; dict from the rec model's `inference.yml` PostProcess character list). v3: `ocr_rec_en.onnx` + `en_dict.txt` via docling.rs's `scripts/install/download_dependencies.sh` |
| `vlm` | Any OpenAI-compatible vision endpoint (vLLM, Ollama, LM Studio, hosted glm-ocr / Qwen-VL, …) | `--img-ocr-endpoint`, `--img-ocr-model`, optionally `--img-ocr-api-key` |
| `paddle` | A PaddleOCR HTTP server (PaddleHub serving or PaddleX serving; both wire shapes handled) | `--img-ocr-endpoint` |

Chains compose: `--img-ocr-engine local,vlm` tries the local model first and
falls back to the endpoint per image. A broken engine (missing model, bad
endpoint) warns once and is skipped; a downed server is dropped after 3
consecutive transient failures in CLI runs. The service waits 30 seconds
before trying it again. A picture no engine could read keeps its placeholder —
OCR problems never fail the conversion.

Remote calls retry transport errors and HTTP 408/429/5xx up to three times
with 2/4/8-second backoff. A local request timeout immediately falls through
to the next engine. `--img-ocr-max-retries 0` disables retries. Paddle's
alternate request schema is attempted only after HTTP 400 or 422, never
after an authentication or missing-endpoint response.

## Output modes (`--img-ocr-mode`)

- `fence` (default) — a fenced code block; engines with box geometry (local
  v5, paddle) render the text with the source image's approximate spacing —
  sidebar items stay left, content stays right, button rows stay on one line
  (the `pdftotext -layout` idea applied to OCR boxes). Geometry-less engines
  (vlm) fence their flat text.
- `markers` — `<!-- ocr:begin engine=… -->` / text / `<!-- ocr:end -->`
  in place of the image placeholder
- `text` — just the recognized text
- `quote` — the text as a Markdown blockquote box (`> …` per line)
- `placeholder` — no OCR; byte-identical to `docling-rs`

With `--images embedded|referenced` the picture itself still renders and the
OCR text is appended after it. A picture whose OCR comes back empty (a logo,
a decorative border) is dropped from the output entirely — no placeholder —
unless the image itself renders (`--images embedded|referenced`) or every
engine *failed* (then the placeholder stays, since text may exist). Captions
and caption links are preserved. OCR follows the selected exporter:
Markdown and chunks omit hidden content; DocLang includes furniture and
rich table cells. JSON processes the authoritative item tree when present
(including its content layers and rich cells), preserving hierarchy,
comments and provenance. Flat-node JSON follows upstream's layer rules.
Only the representation selected for export is transformed.

## Cache

Results land in `~/.cache/docmill` (override:
`--img-ocr-cache-dir` / `DOCMILL_CACHE_DIR`; disable:
`--no-img-ocr-cache`), keyed by SHA-256 over the cache schema, docmill version,
engine, output-affecting settings, MIME type and image bytes. Local model,
dictionary and `.onnx.data` contents participate in the identity, so replacing
a model at the same path invalidates its results. VLM identity uses the
effective request body, including `DOCMILL_EXTRA_BODY`, with canonical JSON
key ordering. Credentials, timeouts, retries and presentation modes do not
affect identity. This release intentionally misses older cache entries;
existing files can remain on disk.

Local fingerprints are computed at the first picture lookup and reused by
that OCR runner. Restart the service after replacing model files.

Each engine gets a cache lookup followed by a live attempt, in configured
order. A cached fallback therefore cannot mask a recovered preferred engine.
Empty results are cached too. A run reports what happened:

```
docmill: 7 picture(s), 5 ocr'd (3 cached), 1 skipped (small), 0 skipped (structured table), 1 empty, 0 failed
```

## All flags / env vars

Every `--img-ocr-*` flag falls back to a `DOCMILL_*` env var
(flag > env > default):

| Flag | Env | Default |
|---|---|---|
| `--img-ocr-engine local\|vlm\|paddle[,…]` | `DOCMILL_ENGINE` | `local` |
| `--img-ocr-mode fence\|markers\|text\|quote\|placeholder` | `DOCMILL_MODE` | `fence` |
| `--img-ocr-lang en\|ch` | `DOCMILL_LANG` | `en` |
| `--img-ocr-endpoint URL` | `DOCMILL_ENDPOINT` | — (required for vlm/paddle) |
| `--img-ocr-model NAME` | `DOCMILL_MODEL` | — (required for vlm) |
| `--img-ocr-api-key KEY` | `DOCMILL_API_KEY` | — |
| `--img-ocr-prompt TEXT` | `DOCMILL_PROMPT` | built-in transcription prompt |
| `--img-ocr-min-px N` | `DOCMILL_MIN_PX` | `2500` (skip icons < ~50×50) |
| `--img-ocr-cache-dir DIR` | `DOCMILL_CACHE_DIR` | `~/.cache/docmill` |
| `--no-img-ocr-cache` | — | cache on |
| `--img-ocr-timeout SECS` | `DOCMILL_TIMEOUT` | `120` |
| `--img-ocr-max-retries N` | `DOCMILL_MAX_RETRIES` | `3` (retries after the initial attempt) |
| `--img-ocr-models-dir DIR` | `DOCMILL_MODELS_DIR` | `./.models` |

`DOCMILL_EXTRA_BODY` is validated once at startup and merges a JSON object
into every VLM request (server-specific knobs the OpenAI shape doesn't cover).
The local engine also
honors explicit model paths: `DOCMILL_DET_ONNX` +
`DOCMILL_REC_ONNX` + `DOCMILL_DICT` (all three → a v5
det+rec set), or docling.rs's `DOCLING_OCR_REC_ONNX` / `DOCLING_OCR_DICT`
for the v3 pair. `--img-ocr-lang` applies to v3 only — the v5 dictionary is
multilingual.

Conversion flags carried over from `docling-rs` (same semantics, buffered
path): `--to md|json|dclx|chunks`, `-o/--output`, `--input GLOB|DIR`, `--jobs N`,
`--strict`, `--pages A-B`,
`--images placeholder|embedded|referenced`, `--fetch-images`,
`--no-table-former`, `--no-ocr`, `--force-full-page-ocr` (OCR every PDF page
even when it has a text layer), `--no-text-panels` (keep every detected
picture as a picture instead of demoting text panels to paragraphs),
`--ocr-lang TAG` (English/Chinese page OCR, including aliases such as `en-US`,
`iso:eng` and `zh-Hant` — independent of picture OCR), `--asr-model`,
`--asr-lang CODE|auto`, `--video-frames`, and `--enrich-*`.

New page conversion controls apply consistently to single files, batches and
HTTP requests:

| CLI flag | Purpose |
|---|---|
| `--skip-ocr` | Skip page text recognition while retaining layout, tables and picture crops. Picture OCR still runs. |
| `--ocr-mode MODE` | `default`, `full_page`, `layout_regions`, or `pdf_aware_layout_regions` |
| `--ocr-scale N` | Finite, positive OCR pixels per PDF point |
| `--heading-hierarchy` | Infer PDF heading levels |
| `--encoding NAME` | Decode text inputs with an explicit encoding, e.g. `windows-1251` |
| `--page-break-placeholder TEXT` | Insert text between Markdown pages |

Batch mode recursively converts a directory or the files matched by a quoted
glob, keeps their relative directory structure under `--output DIR`, and uses
`.md`, `.json`, `.dclx`, or `.chunks.json` extensions according to `--to`.
`--jobs N` runs independent conversion and picture-OCR workers while sharing
one warm docling PDF/image pipeline. Failed files are reported and skipped;
the remaining files continue, and the command exits non-zero if any failed.
For a positional single input, `-o/--output` retains its existing meaning of
an exact output filename.

Tracking docling.rs: this release targets **v1.67.0**, commit
`8f665b094c1ac2a5ff6d6d6f0a84e1939476c23b`. Cargo versions, CI and the
installer are pinned together. It adopts upstream's item-tree JSON fidelity,
page OCR controls and model asset updates, plus remote retry behavior from
docling-mcp. See [upgrade validation](docs/UPGRADE_1_67.md) for coverage,
measurements and limits.

## Format specifics

- **DOCX**: embedded raster pictures carry their original bytes — OCR'd
  directly. Shape-only drawings (SmartArt) have no bytes and keep the
  placeholder. WMF/EMF can't be decoded by the local engine and fall through
  to the next engine in the chain.
- **PDF** (`pdf` feature, default): the ML pipeline crops every detected
  figure region; those crops are OCR'd. `--no-ocr` (text-layer-only mode)
  produces no crops, so there is nothing to picture-OCR. When docling already
  recovered a non-empty structured table from a table screenshot, docmill
  skips picture OCR if at least 80% of that table lies inside the picture on
  the same page. The picture/placeholder remains, avoiding duplicate table
  text while preserving the original figure.
- **RTF**: converted by docling.rs. Embedded PNG/JPEG pictures retain their
  bytes and therefore pass through docmill's selected picture-OCR engine.
- **XLSB**: converted by docling.rs's cells-first Calamine path. Its reader does
  not expose drawings, charts, comments, or embedded picture bytes, so those
  items cannot be picture-OCRed.
- **Standalone images** (PNG/JPEG/TIFF/…): with a local-first chain the image
  runs through docling's full ML pipeline (layout + text OCR + tables) and
  detected figure sub-regions are OCR'd like PDF figures. With a
  **remote-first** chain the ML pipeline is bypassed — the file goes straight
  to the endpoint, so no local models are needed at all.
- The local engine in v5 det+rec mode handles GUI screenshots and mixed
  layouts well (~5–6 s/image on CPU, measured on a par with a PaddleOCR
  server's quality). The v3 fallback reads single-column text top-to-bottom
  only. For heavily rotated scans or complex figures (newspaper clippings,
  charts), the `vlm` engine remains the strongest option.

## Installing

One command builds from source and installs a self-contained tree under
`/usr/local/docmill` (binary + `.models` + pdfium) with a
`docmill` symlink in `/usr/local/bin` — same install shape as
docling.rs. The binary resolves its assets relative to its own
(symlink-resolved) location, so it works from any directory with no
environment setup. Uses sudo only for the copy steps when the prefix isn't
writable.

```console
$ scripts/install/install.sh
$ docmill your.docx > out.md
```

Options via env vars: `DOCMILL_PREFIX` (default
`/usr/local/docmill`), `DOCMILL_BIN_DIR` (default
`/usr/local/bin`), `DOCMILL_SUDO=0`, `DOCLING_RS_DIR` (the docling.rs
checkout; cloned beside this one when missing). The installer verifies the
pinned commit before downloading assets; it leaves mismatched existing
checkouts untouched. `DOCLING_RS_DIR` must resolve to Cargo's sibling path.

Runtime models alone (into the current directory, idempotent):

```console
$ scripts/install/download_dependencies.sh          # everything
$ scripts/install/download_dependencies.sh --no-pdf # picture-OCR models only
```

Flags: `--force` re-fetch, `--no-pdf` (skip pdfium/layout/page detector/TableFormer),
`--no-tableformer`, `--no-v5` (skip the PP-OCRv5 conversion, which needs
python3 + pip for a one-time paddle2onnx run), `--ort` (vendor the ONNX
Runtime shared build for old-glibc hosts — see below; install.sh then links
it dynamically with an rpath into the prefix automatically).

The PDF assets now include an optional PP-OCRv6 page detector
(`.models/ocr_det.onnx`) and TableFormer FP16 encoder. The page detector and
English recognizer/dictionary use the upstream release mirror with source
fallbacks. Local picture OCR continues to prefer the PP-OCRv5 det+rec pair.

docling.rs v1 resolves runtime assets from `.models/` only. The download,
install, and package scripts automatically rename a legacy `models/` directory
when `.models/` is absent. If both exist, they preserve both, warn, and use
`.models/` without merging or overwriting either tree.

## Web service (`docmill serve`)

A small local HTTP conversion service — synchronous like the converter
itself (tiny_http, no async runtime), with one conversion worker owning a
warm OCR engine chain and PDF/image pipeline, plus the same disk cache:

```console
$ docmill serve --addr 127.0.0.1:8877 [--img-ocr-* flags]
```

- `POST /convert` — the file as `multipart/form-data` (field `file`, e.g.
  `curl -F file=@doc.docx`) or as the raw body with `?filename=doc.docx`.
  Optional query/form fields: `to=md|json`, `mode=fence|markers|text|quote|placeholder`,
  `images=placeholder|embedded`, `pages=A-B`, `strict=1`, `no_ocr=1`,
  `skip_ocr=1`, `no_table_former=1`, `force_full_page_ocr=1`, `no_text_panels=1`,
  `ocr_lang=TAG`, `ocr_mode=MODE`, `ocr_scale=N`, `heading_hierarchy=1`,
  `encoding=NAME`, `page_break_placeholder=TEXT`.
  Returns Markdown (`text/markdown`) or docling JSON. `images=referenced`
  returns HTTP 400 because the service does not serve artifact files.
- `GET /health` — liveness (never blocks behind a running conversion).
- `GET /` — a minimal HTML upload form for manual testing.

Four conversions can wait while one runs; further requests receive HTTP 503
before docmill buffers their bodies. `/health` bypasses conversion work.
The engine chain is fixed at startup. Request settings start from the startup
defaults each time; mutable pipeline options reset and construction options
rebuild the pipeline when changed. A conversion panic returns HTTP 500 and
discards worker state so the next request can recreate it.

Uploads are capped at 200 MiB. Keep nginx request buffering and body limits
enabled: tiny_http may drain unread request bodies when replying, so the
application queue alone does not isolate slow or oversized direct uploads.
Bind to localhost and put nginx in front for TLS/auth/limits:

```nginx
location /docmill/ {
    proxy_pass http://127.0.0.1:8877/;
    client_max_body_size 200m;
    # OCR-heavy conversions can run for minutes on CPU:
    proxy_read_timeout 600s;
    proxy_request_buffering on;
}
```

## Linux packages

```console
$ scripts/package/build_packages.sh [--models] [--with-models] [--tar] [--rpm] [--deb]
```

Builds into `dist/`: a portable tarball plus `.rpm`/`.deb` when
`rpmbuild`/`dpkg-deb` are installed (no format flags = everything the host
can build). All formats ship the same self-contained tree rooted at
`/opt/docmill` (`bin/`, `lib/` with bundled onnxruntime when built
against a vendored copy, the model download script, licenses) plus a
`/usr/bin/docmill` symlink in rpm/deb. The binary carries an
`$ORIGIN/../lib` rpath, so the tarball also runs extracted anywhere — no
environment needed.

The main packages are small (~15–20 MB, no models) and print a post-install
hint. Models ship separately with `--models`: a companion
**`docmill-models`** package (`.models/` + pdfium, ~500 MB) installing
into the same `/opt/docmill` tree — the main rpm/deb `Suggests` it,
it `Enhances` the main one, and either can be installed/upgraded without
the other. Tarball users extract both archives over the same root.
Alternatively `--with-models` builds one fat all-in-one package.

## Platform support

Like docling.rs itself: **Linux** and **macOS** use
`scripts/install/download_dependencies.sh` (macOS gets its `libpdfium.dylib`
from the official pdfium-binaries builds; ort's static onnxruntime download
works natively there, so the glibc workaround below is Linux-only).
**Windows** uses `scripts\install\download_dependencies.bat` (needs the
curl.exe/tar.exe that ship with Windows 10+), then a plain
`cargo build --release` from the repo root — the binary resolves `.models\`
and `.pdfium\lib` next to the CWD or the executable. `install.sh` targets
Unix prefixes; on Windows run from the repo root or copy the binary next to
its `.models\` directory.

## Building manually

```console
$ git clone --branch v1.67.0 https://github.com/docling-project/docling.rs ../docling.rs
$ git -C ../docling.rs rev-parse HEAD # must be 8f665b094c1ac2a5ff6d6d6f0a84e1939476c23b
$ cargo build --locked --release
```

Features: `default = ["pdf", "asr", "fetch-images", "vlm", "local-ocr", "serve"]`.
RTF and XLSB are provided by docling.rs and remain available in
`--no-default-features` builds. A
remote-only build with no onnxruntime link at all (converts every declarative
format, OCRs via vlm/paddle):

```console
$ cargo build --release --no-default-features --features fetch-images
```

**Old-glibc hosts** (RHEL/Oracle Linux 9, glibc < 2.38): the static
onnxruntime that `ort` auto-downloads fails to link (`__isoc23_strtol`
undefined). Link the official ONNX Runtime shared build instead:

```console
$ curl -fsSLO https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
$ tar xzf onnxruntime-linux-x64-1.22.0.tgz -C ~/opt
$ export ORT_LIB_LOCATION=~/opt/onnxruntime-linux-x64-1.22.0/lib \
         ORT_PREFER_DYNAMIC_LINK=1
$ cargo build --release
$ LD_LIBRARY_PATH=$ORT_LIB_LOCATION target/release/docmill …
```

## Tests

Run `cargo test --locked`, `cargo test --locked --no-default-features`, and
`cargo test --locked --no-default-features --features serve`. Tests cover
format detection, item-tree/node OCR, cache identity, mock HTTP retry/schema
behavior, service admission/recovery and CLI parity with the pinned upstream
converter. They use committed upstream fixtures and local loopback HTTP;
no model downloads or remote OCR services are needed at test runtime.
On old-glibc hosts use the no-default-feature commands or the
`ORT_LIB_LOCATION` setup above. CI runs the portable and service suites on
Linux, macOS and Windows and checks default-feature compilation.

## License

MIT — see [LICENSE](LICENSE). The project builds on docling.rs (MIT) and
downloads PaddleOCR models (Apache-2.0), docling layout/table models (MIT),
pdfium (BSD-3-Clause), and ONNX Runtime (MIT) at install time; see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full inventory.
