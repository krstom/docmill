# Upstream adoption and validation

Validated on 2026-09-23 against docling.rs v1.67.0
(`8f665b094c1ac2a5ff6d6d6f0a84e1939476c23b`) and docling-mcp
`a8a41e6014ba3a148261702e760421086e9c80e3` (v3.2.0 plus its timeout/retry
fix). The sibling repositories were left unchanged.

## Adopted behavior

- Exact Cargo versions, pinned CI checkout and installer revision checks.
  The lockfile also adopts upstream's rustls 0.23.45 security update.
- JSON picture OCR now edits the authoritative item tree. Other outputs
  retain the node representation, including DocLang rich cells. Captions,
  hyperlinks, layers, comments, hierarchy and provenance survive replacement.
  Only rendered tables can suppress duplicate picture OCR; the threshold
  remains 80% of table area on the same page, using comparable coordinates.
- Shared conversion settings across single files, batches and HTTP, including
  OCR mode/scale, page OCR skipping, language aliases, heading hierarchy,
  explicit encoding and Markdown page separators.
- Content-based model identities, canonical effective VLM requests, MIME-aware
  cache keys and engine priority across cached/live results. Model hashes are
  lazy and computed once per runner; existing services must restart after
  model replacement.
- Three configurable remote retries, immediate fallback on local timeout,
  Paddle schema negotiation restricted to 400/422, and service recovery after
  a 30-second cooldown following three consecutive transient failures.
- One service conversion worker, four queued requests, HTTP 503 on overflow,
  a warm PDF pipeline, reset per-request options, and HTTP 500 plus discarded
  worker state after a conversion panic. Embedded HTTP images are supported;
  referenced artifacts are rejected with HTTP 400.
- Page detector and optional FP16 TableFormer assets, with mirror fallbacks
  for the detector and English recognition pair. Local picture OCR keeps v5.

MCP tooling, cloud/RAG integration, whole-document caching, GPU work and
additional exporters remain outside this upgrade.

## Verification

Linux x86-64, Rust 1.93.1:

| Command | Result |
|---|---|
| `cargo test --locked` | 97 tests passed |
| `cargo test --locked --no-default-features` | 79 tests passed |
| `cargo test --locked --no-default-features --features serve` | 89 tests passed |
| `cargo check --locked --all-targets` | Passed |
| `cargo build --locked --release` | Passed |
| Installer/package shell syntax | Passed |
| Download script with mocked curl | Mirror fallback, FP16 destination, atomic completion and idempotence passed |

Placeholder Markdown and JSON are compared byte-for-byte with the pinned
`DocumentConverter` on committed DOCX, PPTX, HTML, RTF and digital PDF
fixtures. CLI tests cover single/batch encoding and invalid OCR controls.
OCR tests cover real office/HTML trees, valid JSON references, rich cells,
wrapper preservation, empty/failed results, image retention and table overlap.

Remote tests use loopback HTTP for VLM payloads and Paddle negotiation, plus
controlled retry clocks/sleep callbacks. Runner tests cover fallback cache
priority and recovery without waiting through the cooldown. HTTP tests cover
queue overflow, independent health dispatch, option reset, encoding, image
mode validation and panic recovery. Tests need no downloaded models or remote
OCR services at runtime.

CI has portable/service tests and default-feature compilation on Linux,
macOS and Windows. Only Linux was executed locally; the Windows download
script and macOS installer were reviewed but not run.

## Measurements

Release binaries are compared with the saved v1.4.2 binary on an Intel Xeon
E5-2699 v4 host (8 logical CPUs visible, 30 GiB RAM). Measurements use
`/usr/bin/time` for elapsed time and peak RSS. Each CLI invocation starts a
fresh process. “Cold” means an empty picture-result cache, not a flushed OS
file cache; “cached” is the next invocation with that cache. These are single
observations for regression checking, not statistically controlled benchmarks.

The inputs are committed upstream fixtures: `docx_rich_cells.docx`,
`powerpoint_with_image.pptx`, `multi_page.pdf`, `ocr_test_raster.pdf`, and an
HTML data-URI wrapper around `llama_vs_mistral_example.png`. Runs use local
PP-OCRv5, `--img-ocr-min-px 0`, Markdown output and explicit paths to the
existing models. Digital PDF uses `--no-ocr`; scanned PDF uses
`--no-table-former`. HTML uses `--fetch-images`.

| Input | v1.4.2 cold / cached (s) | v1.67.0 cold / cached (s) | v1.4.2 peak RSS cold / cached (MiB) | v1.67.0 peak RSS cold / cached (MiB) | Live picture OCR calls cold / cached |
|---|---:|---:|---:|---:|---:|
| HTML screenshot | 5.75 / 0.02 | 5.82 / 0.21 | 256.4 / 17.8 | 260.2 / 19.6 | 1 / 0 |
| DOCX rich cells | 0.01 / 0.01 | 0.01 / 0.01 | 17.1 / 16.8 | 19.5 / 19.6 | 0 / 0 |
| PPTX image | 1.42 / <0.01 | 1.60 / 0.17 | 210.2 / 15.2 | 212.5 / 17.3 | 1 / 0 |
| Digital PDF | 0.07 / 0.04 | 0.04 / 0.04 | 23.2 / 23.2 | 23.8 / 24.1 | 0 / 0 |
| Scanned PDF | 4.35 / 4.37 | 2.58 / 2.62 | 370.0 / 401.9 | 494.1 / 492.9 | 0 / 0 |

Picture-call counts match both versions. The scanned PDF uses page OCR,
which is separate from these picture counters. The DOCX picture is inside a
rich cell omitted from Markdown; the JSON regression test exercises it.
All runs succeeded without failed picture OCR. Each cached output matched
its cold output byte-for-byte. HTML, PPTX and scanned PDF also matched v1.4.2
byte-for-byte; DOCX and digital PDF inherit changed upstream rendering and
pass parity against v1.67.0.

For actual warm model reuse, `--bench-warm 3 --no-table-former` on the scanned
PDF averaged **2.367465 s/document before** and **1.660099 s/document after**
(one initial warmup excluded). The benchmark process's peak RSS increased
from **486.6 MiB to 659.5 MiB**. This fixture was about 30% faster with warm
models, at about 36% higher peak memory; it is not a general speedup claim.

The new content fingerprints cost roughly 0.17–0.20 seconds on a fresh local
picture-OCR process, visible in the cached HTML/PPTX runs. Services and batch
workers amortize that cost across images. Deferring fingerprints until the
first picture lookup removed this overhead from documents with no visible
pictures. Replacing content hashes with size/mtime keys would lose the
same-path model replacement guarantee.

To reproduce with installed assets, run each command with a new cache
directory, then repeat it with the same directory:

```sh
/usr/bin/time -f '%e seconds; %M KiB' target/release/docmill \
  --img-ocr-engine local --img-ocr-models-dir .models \
  --img-ocr-cache-dir /tmp/docmill-benchmark-cache --img-ocr-min-px 0 \
  --no-stream --no-table-former \
  ../docling.rs/tests/data/scanned/sources/ocr_test_raster.pdf > /tmp/docmill-benchmark.md
target/release/docmill --bench-warm 3 --no-table-former \
  --img-ocr-mode placeholder ../docling.rs/tests/data/scanned/sources/ocr_test_raster.pdf
```

The local run used explicit `DOCLING_LAYOUT_ONNX`, `DOCLING_OCR_REC_ONNX`,
`DOCLING_OCR_DICT` and `PDFIUM_DYNAMIC_LIB_PATH` values to read existing assets
without migrating or modifying the sibling checkout. Raw timings, output
hashes, logs and the harness are in the local ignored
`target/upgrade-validation/` directory.

## Practical limits

The page detector and TableFormer assets were unavailable on this host, so
real inference measurements use the existing FP32 layout model and English
recognizer with TableFormer disabled. Download paths were tested with a mock;
FP16/TableFormer and detector accuracy/performance were not measured.

The service queue bounds admitted conversion work, not all network resources.
tiny_http can drain unread upload bodies when responding. Keep the documented
nginx request buffering and body limits enabled to isolate slow or oversized
direct clients. Panic recovery handles Rust unwinding; process aborts and
allocation failures cannot be recovered this way.
