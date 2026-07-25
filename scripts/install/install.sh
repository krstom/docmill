#!/usr/bin/env bash
# Build docmill from source and install a self-contained tree under
# /usr/local/docmill (by default) with a `docmill` command on
# PATH — the same install shape as docling.rs's install.sh.
#
# From a checkout (with ../docling.rs beside it — path dependencies):
#   scripts/install/install.sh
#
# What it does:
#   1. Checks for a Rust toolchain and a sibling ../docling.rs checkout
#      (cloned automatically when missing).
#   2. Fetches runtime models via scripts/install/download_dependencies.sh
#      into the source tree (idempotent).
#   3. Builds the CLI in release mode. On hosts whose glibc is too old for
#      ort's static onnxruntime download (link errors about __isoc23_*),
#      run download_dependencies.sh --ort first: a vendored ONNX Runtime is
#      then linked dynamically, installed into the prefix, and found at
#      runtime via an rpath — no environment needed.
#   4. Installs to $DOCMILL_PREFIX (default /usr/local/docmill):
#        bin/docmill    the CLI
#        models/…, .pdfium/…    runtime assets
#        lib/…                  vendored onnxruntime (dynamic builds only)
#      and symlinks it as /usr/local/bin/docmill. The binary resolves
#      models and pdfium relative to its own (symlink-resolved) location, so
#      it works from any working directory with no environment setup.
#
# Options (env vars):
#   DOCMILL_PREFIX=/opt/docmill   install tree
#   DOCMILL_BIN_DIR=/usr/local/bin       where the symlink goes
#   DOCMILL_SUDO=0                       never invoke sudo (fail instead)
#   DOCLING_RS_DIR=/path/to/docling.rs          sibling checkout location
set -euo pipefail

PREFIX="${DOCMILL_PREFIX:-/usr/local/docmill}"
BIN_DIR="${DOCMILL_BIN_DIR:-/usr/local/bin}"
DOCLING_RS_REPO="https://github.com/docling-project/docling.rs"

say() { printf '\033[1m[docmill]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[docmill]\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"

# Run from the project root (the script lives in scripts/install/).
SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SRC_DIR"
[ -f Cargo.toml ] || die "cannot locate the docmill checkout (expected at $SRC_DIR)"

# Privilege helper: only used for the install/copy steps, never for the build.
# Writability is judged on the nearest *existing* ancestor, so a not-yet-created
# prefix under $HOME doesn't spuriously demand sudo.
nearest_existing() {
  d="$1"
  while [ ! -e "$d" ]; do d="$(dirname "$d")"; done
  printf '%s' "$d"
}
SUDO=""
if [ ! -w "$(nearest_existing "$PREFIX")" ] || [ ! -w "$(nearest_existing "$BIN_DIR")" ]; then
  if [ "${DOCMILL_SUDO:-1}" = "0" ]; then
    die "$PREFIX or $BIN_DIR is not writable and DOCMILL_SUDO=0"
  fi
  command -v sudo >/dev/null 2>&1 || die "$PREFIX is not writable and sudo is unavailable"
  SUDO="sudo"
fi

# --- 1. Toolchain + the docling.rs sibling checkout ---------------------------
if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
command -v cargo >/dev/null 2>&1 || die "cargo not found — install Rust (https://rustup.rs) first"
say "using $(cargo --version)"

DOCLING_RS="${DOCLING_RS_DIR:-$SRC_DIR/../docling.rs}"
if [ ! -f "$DOCLING_RS/crates/docling/Cargo.toml" ]; then
  say "cloning docling.rs beside the checkout (path dependency)"
  command -v git >/dev/null 2>&1 || die "git is required to clone docling.rs"
  git clone --depth 1 "$DOCLING_RS_REPO" "$DOCLING_RS"
fi

# --- 2. Runtime models ---------------------------------------------------------
say "fetching runtime models (idempotent)"
sh scripts/install/download_dependencies.sh

# --- 3. Build --------------------------------------------------------------------
BUILD_ENV=()
RPATH_LIB=""
if [ -d vendor/onnxruntime/lib ] || [ -n "${ORT_LIB_LOCATION:-}" ]; then
  ORT_DIR="${ORT_LIB_LOCATION:-$SRC_DIR/vendor/onnxruntime/lib}"
  say "linking onnxruntime dynamically from $ORT_DIR (rpath: $PREFIX/lib)"
  RPATH_LIB="$PREFIX/lib"
  BUILD_ENV=(ORT_LIB_LOCATION="$ORT_DIR" ORT_PREFER_DYNAMIC_LINK=1
             RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,$RPATH_LIB")
fi
say "building the CLI (release)"
# ${arr[@]+…} keeps the empty-array case safe under set -u on macOS's bash 3.2.
env ${BUILD_ENV[@]+"${BUILD_ENV[@]}"} cargo build --release

# --- 4. Install tree -------------------------------------------------------------
say "installing to $PREFIX"
$SUDO mkdir -p "$PREFIX/bin"
$SUDO cp target/release/docmill "$PREFIX/bin/docmill"
$SUDO mkdir -p "$PREFIX/models"
$SUDO cp -a models/. "$PREFIX/models/"
if [ -d .pdfium ]; then
  $SUDO mkdir -p "$PREFIX/.pdfium"
  $SUDO cp -a .pdfium/. "$PREFIX/.pdfium/"
fi
if [ -n "$RPATH_LIB" ] && [ -d "${ORT_DIR:-}" ]; then
  $SUDO mkdir -p "$PREFIX/lib"
  $SUDO cp -a "$ORT_DIR"/libonnxruntime.so* "$PREFIX/lib/"
fi

say "linking $BIN_DIR/docmill -> $PREFIX/bin/docmill"
$SUDO mkdir -p "$BIN_DIR"
$SUDO ln -sfn "$PREFIX/bin/docmill" "$BIN_DIR/docmill"

# --- 5. Smoke test -----------------------------------------------------------------
say "smoke test: converting a trivial Markdown document from an unrelated directory"
TMP_MD="$(mktemp --suffix=.md 2>/dev/null || mktemp -t docmill.XXXXXX.md)"
printf '# docmill\n\ninstalled.\n' > "$TMP_MD"
(cd / && "$BIN_DIR/docmill" --img-ocr-mode placeholder "$TMP_MD" >/dev/null) || die "smoke test failed"
rm -f "$TMP_MD"

say "done. Try:  docmill your.docx > out.md"
say "layout: $PREFIX  (bin/, models/, .pdfium/$( [ -n "$RPATH_LIB" ] && printf ', lib/' ))"
say "uninstall: rm -rf $PREFIX $BIN_DIR/docmill"
