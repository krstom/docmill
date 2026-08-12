#!/usr/bin/env sh
# Fetch docmill's runtime dependencies into the current directory —
# the same layout docling.rs uses (.models/ + .pdfium/lib), plus the
# PP-OCRv5 detection/recognition pair that powers the local picture-OCR
# engine's screenshot-grade quality.
#
# Run from the project root (or the directory you'll run the binary from):
#   scripts/install/download_dependencies.sh
#
# Downloads:
#   .pdfium/lib/libpdfium.so                    page rendering (PDF input)
#   .models/layout_heron.onnx                    RT-DETR layout (PDF input)
#   .models/tableformer/{encoder,decoder,bbox}.onnx table structure (PDF input)
#   .models/ocr_rec.onnx + ppocr_keys_v1.txt     PP-OCRv3 ch pair (page OCR +
#   .models/ocr_rec_en.onnx + en_dict.txt          v3 picture-OCR fallback)
#   .models/ppocrv5_mobile_{det,rec}.onnx        PP-OCRv5 det+rec — converted
#   .models/ppocrv5_dict.txt                       locally with paddle2onnx from
#                                                 the official PaddlePaddle
#                                                 Hugging Face repos (needs
#                                                 python3 + pip; skipped with a
#                                                 warning when unavailable —
#                                                 the v3 fallback still works)
#   vendor/onnxruntime/…                        (--ort only) official ONNX
#                                                 Runtime shared build, for
#                                                 hosts whose glibc predates
#                                                 the static binaries ort
#                                                 downloads (see README)
#
# Idempotent: skips files already on disk. Pass --force to re-fetch everything.
set -eu

BASE_URL="${DOCLING_RS_MODELS_URL:-https://github.com/docling-project/docling.rs/releases/download/models-v1}"
PADDLE_HF="${DOCMILL_PADDLE_URL:-https://huggingface.co/PaddlePaddle}"
ORT_VERSION="${DOCMILL_ORT_VERSION:-1.22.0}"

FORCE=false
WITH_PDF=true
WITH_TABLEFORMER=true
WITH_V5=true
WITH_ORT=false

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    --no-pdf) WITH_PDF=false ;;
    --no-tableformer) WITH_TABLEFORMER=false ;;
    --no-v5) WITH_V5=false ;;
    --ort) WITH_ORT=true ;;
    *)
      echo "usage: download_dependencies.sh [--force] [--no-pdf] [--no-tableformer] [--no-v5] [--ort]" >&2
      exit 2
      ;;
  esac
done

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }

# docling.rs v1 dropped the legacy `models/` fallback. Preserve existing user
# data when both directories exist; otherwise make the one-time rename before
# downloading so already-fetched assets are reused.
if [ -d models ] && [ ! -e .models ]; then
  echo "migrating legacy models/ to .models/"
  mv models .models
elif [ -d models ] && [ -e .models ]; then
  echo "warning: both models/ and .models/ exist; preserving both and using .models/" >&2
fi
mkdir -p .models

# Same transfer guards as docling.rs: bounded connect, stall abort, retries.
CURL_TIMEOUTS="--connect-timeout 30 --speed-limit 1024 --speed-time 60 --retry 3 --retry-delay 2"

fetch() { # <url> <dest>
  if [ "$FORCE" = false ] && [ -f "$2" ]; then
    echo "  = $2 (already present)"
    return 0
  fi
  echo "  > $2"
  # shellcheck disable=SC2086 # CURL_TIMEOUTS is a flag list, splitting intended
  curl -fsSL $CURL_TIMEOUTS -o "$2.download" "$1"
  mv "$2.download" "$2"
}

fetch_optional() { # <url> <dest> — ignore a missing/failed asset
  if [ "$FORCE" = false ] && [ -f "$2" ]; then
    return 0
  fi
  # shellcheck disable=SC2086
  if curl -fsSL $CURL_TIMEOUTS -o "$2.download" "$1" 2>/dev/null; then
    mv "$2.download" "$2"
    echo "  > $2"
  else
    rm -f "$2.download"
  fi
}

# --- PDF pipeline assets (docling.rs models release) -------------------------
# pdfium is the one platform-specific binary: Linux uses the docling.rs
# release's libpdfium.so; macOS fetches the official pdfium-binaries build
# (same source docling.rs's Windows script uses).
fetch_pdfium() {
  case "$(uname -s)" in
    Darwin)
      [ "$FORCE" = false ] && [ -f .pdfium/lib/libpdfium.dylib ] && {
        echo "  = .pdfium/lib/libpdfium.dylib (already present)"; return 0; }
      case "$(uname -m)" in
        arm64) tgz=pdfium-mac-arm64.tgz ;;
        *) tgz=pdfium-mac-x64.tgz ;;
      esac
      echo "  > .pdfium/lib/libpdfium.dylib (pdfium-binaries $tgz)"
      # shellcheck disable=SC2086
      curl -fsSL $CURL_TIMEOUTS -o .pdfium/pdfium.tgz \
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/$tgz"
      tar xzf .pdfium/pdfium.tgz -C .pdfium lib/libpdfium.dylib
      rm -f .pdfium/pdfium.tgz
      ;;
    *)
      fetch "$BASE_URL/libpdfium.so" .pdfium/lib/libpdfium.so
      ;;
  esac
}

if [ "$WITH_PDF" = true ]; then
  echo "fetching PDF pipeline assets from $BASE_URL"
  mkdir -p .pdfium/lib
  fetch_pdfium
  fetch "$BASE_URL/layout_heron.onnx" .models/layout_heron.onnx
  fetch_optional "$BASE_URL/layout_heron_int8.onnx" .models/layout_heron_int8.onnx
  if [ "$WITH_TABLEFORMER" = true ]; then
    mkdir -p .models/tableformer
    fetch "$BASE_URL/encoder.onnx" .models/tableformer/encoder.onnx
    fetch_optional "$BASE_URL/encoder.onnx.data" .models/tableformer/encoder.onnx.data
    fetch "$BASE_URL/decoder.onnx" .models/tableformer/decoder.onnx
    fetch_optional "$BASE_URL/decoder.onnx.data" .models/tableformer/decoder.onnx.data
    fetch_optional "$BASE_URL/decoder_kv.onnx" .models/tableformer/decoder_kv.onnx
    fetch_optional "$BASE_URL/decoder_kv.onnx.data" .models/tableformer/decoder_kv.onnx.data
    fetch "$BASE_URL/bbox.onnx" .models/tableformer/bbox.onnx
    fetch_optional "$BASE_URL/bbox.onnx.data" .models/tableformer/bbox.onnx.data
  fi
fi

# --- PP-OCRv3 pairs (page OCR + v3 picture-OCR fallback) ---------------------
echo "fetching PP-OCRv3 recognition pairs"
fetch "$BASE_URL/ocr_rec.onnx" .models/ocr_rec.onnx
fetch "$BASE_URL/ppocr_keys_v1.txt" .models/ppocr_keys_v1.txt
fetch "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx" .models/ocr_rec_en.onnx
fetch "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt" .models/en_dict.txt

# --- PP-OCRv5 det+rec (the local picture-OCR engine's preferred models) ------
# No public host serves these as ONNX, so we fetch the official PaddlePaddle
# inference models from Hugging Face and convert with paddle2onnx — a pure
# model-format conversion, done once. The result is 22 MB of ONNX; the
# paddle/python toolchain is only needed here, never at runtime.
if [ "$WITH_V5" = true ]; then
  if [ "$FORCE" = false ] && [ -f .models/ppocrv5_mobile_det.onnx ] \
      && [ -f .models/ppocrv5_mobile_rec.onnx ] && [ -f .models/ppocrv5_dict.txt ]; then
    echo "  = PP-OCRv5 det+rec (already present)"
  elif ! command -v python3 >/dev/null 2>&1; then
    echo "warning: python3 not found — skipping PP-OCRv5 conversion (the v3 fallback still works;" >&2
    echo "         picture OCR on screenshots will be much weaker without v5)" >&2
  else
    P2O="$(command -v paddle2onnx || true)"
    [ -z "$P2O" ] && [ -x "$HOME/.local/bin/paddle2onnx" ] && P2O="$HOME/.local/bin/paddle2onnx"
    if [ -z "$P2O" ]; then
      echo "  installing paddle2onnx (pip --user; one-time, conversion only)"
      python3 -m pip install --user --quiet paddle2onnx || true
      [ -x "$HOME/.local/bin/paddle2onnx" ] && P2O="$HOME/.local/bin/paddle2onnx"
    fi
    if [ -z "$P2O" ]; then
      echo "warning: paddle2onnx unavailable — skipping PP-OCRv5 conversion (v3 fallback still works)" >&2
    else
      echo "fetching + converting PP-OCRv5 mobile det/rec from $PADDLE_HF"
      TMP="$(mktemp -d)"
      trap 'rm -rf "$TMP"' EXIT
      ok=true
      for model in PP-OCRv5_mobile_det PP-OCRv5_mobile_rec; do
        mkdir -p "$TMP/$model"
        for f in inference.json inference.pdiparams inference.yml; do
          # shellcheck disable=SC2086
          curl -fsSL $CURL_TIMEOUTS -o "$TMP/$model/$f" "$PADDLE_HF/$model/resolve/main/$f" || ok=false
        done
      done
      if [ "$ok" = true ] \
        && "$P2O" --model_dir "$TMP/PP-OCRv5_mobile_det" --model_filename inference.json \
             --params_filename inference.pdiparams --save_file .models/ppocrv5_mobile_det.onnx \
             --opset_version 14 >/dev/null 2>&1 \
        && "$P2O" --model_dir "$TMP/PP-OCRv5_mobile_rec" --model_filename inference.json \
             --params_filename inference.pdiparams --save_file .models/ppocrv5_mobile_rec.onnx \
             --opset_version 14 >/dev/null 2>&1; then
        echo "  > .models/ppocrv5_mobile_det.onnx"
        echo "  > .models/ppocrv5_mobile_rec.onnx"
        fetch "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt" .models/ppocrv5_dict.txt
      else
        rm -f .models/ppocrv5_mobile_det.onnx .models/ppocrv5_mobile_rec.onnx
        echo "warning: PP-OCRv5 fetch/conversion failed — the v3 fallback still works" >&2
      fi
    fi
  fi
fi

# --- Optional: ONNX Runtime shared build (old-glibc hosts) -------------------
if [ "$WITH_ORT" = true ]; then
  if [ "$FORCE" = false ] && [ -d "vendor/onnxruntime/lib" ]; then
    echo "  = vendor/onnxruntime (already present)"
  else
    echo "fetching ONNX Runtime $ORT_VERSION shared build (dynamic-link escape hatch)"
    mkdir -p vendor
    # shellcheck disable=SC2086
    curl -fsSL $CURL_TIMEOUTS -o vendor/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"
    rm -rf vendor/onnxruntime
    tar xzf vendor/ort.tgz -C vendor
    mv "vendor/onnxruntime-linux-x64-$ORT_VERSION" vendor/onnxruntime
    rm -f vendor/ort.tgz
    echo "  > vendor/onnxruntime/lib"
  fi
fi

echo "done."
