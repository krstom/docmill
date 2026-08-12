# Third-party notices

docmill is MIT-licensed (see [LICENSE](LICENSE)). It builds on, links
against, and — via `scripts/install/download_dependencies.sh` — downloads the
following third-party components. Nothing here is redistributed inside this
repository; the scripts fetch each component from its upstream source at
install time, subject to that component's own license.

## Libraries (linked/compiled)

| Component | License | Source |
|---|---|---|
| [docling.rs](https://github.com/docling-project/docling.rs) (`docling`, `docling-core`, `docling-pdf`) | MIT | path dependency on a sibling checkout |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (via the [`ort`](https://github.com/pykeio/ort) crate, MIT/Apache-2.0) | MIT | statically linked by default, or the official shared build fetched with `--ort` |
| [pdfium](https://pdfium.googlesource.com/pdfium/) (`libpdfium.so`) | BSD-3-Clause (with Apache-2.0-licensed parts) | prebuilt binary from the docling.rs models release |
| Rust crate dependencies (`calamine`, `zip`, `image`, `ureq`, `serde_json`, `sha2`, …) | MIT/Apache-2.0 or compatible permissive licenses (see each crate) | crates.io |

## Models (downloaded at install time)

| Model | License | Source |
|---|---|---|
| PP-OCRv5 mobile detection + recognition, and the PP-OCRv5/`en`/`ch` dictionaries | Apache-2.0 ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)) | official [PaddlePaddle](https://huggingface.co/PaddlePaddle) Hugging Face repos; converted locally to ONNX with [paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX) (Apache-2.0) |
| PP-OCRv3 English recognition ONNX export | Apache-2.0 | [RapidOCR](https://huggingface.co/SWHL/RapidOCR) Hugging Face repo |
| PP-OCRv3 multilingual (`ch`) recognition | Apache-2.0 (PaddleOCR) | docling.rs models release |
| Layout (RT-DETR "heron") and TableFormer ONNX exports | MIT ([docling / IBM](https://github.com/docling-project)) | docling.rs models release |

## Optional runtime services (not installed by this project)

- The `vlm` engine talks to any OpenAI-compatible endpoint you configure; the
  server and model you point it at carry their own licenses.
- The `paddle` engine talks to a PaddleOCR HTTP server you run yourself
  (PaddleOCR itself is Apache-2.0).
