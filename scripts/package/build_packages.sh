#!/usr/bin/env bash
# Build Linux distribution packages for docmill: a portable tarball,
# an .rpm (when rpmbuild is available), and a .deb (when dpkg-deb is
# available). Everything lands in dist/.
#
#   scripts/package/build_packages.sh [--models] [--with-models] [--tar] [--rpm] [--deb] [--no-build]
#
# With no format flags, every format whose tool is installed is built.
# --models additionally builds a companion `docmill-models` package
# (the checkout's .models/ + .pdfium/, several hundred MB) that installs into
# the same /opt/docmill tree — the main package stays small and merely
# suggests it. --with-models instead bundles everything into one fat package.
#
# Package layout (all formats): the same self-contained tree install.sh
# creates, rooted at /opt/docmill —
#   bin/docmill      the CLI/server binary
#   lib/libonnxruntime.so*  bundled when built against a vendored/explicit
#                           ONNX Runtime (vendor/onnxruntime or
#                           $ORT_LIB_LOCATION); the binary carries an
#                           $ORIGIN/../lib rpath, so the tree also works
#                           extracted anywhere, not just under /opt
#   scripts/…               download_dependencies.sh for fetching models
#   .models/, .pdfium/      only with --with-models (adds several hundred MB;
#                           default packages print a post-install hint to run
#                           the download script instead)
# plus a /usr/bin/docmill symlink in rpm/deb.
set -euo pipefail

say() { printf '\033[1m[package]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[package]\033[0m %s\n' "$*" >&2; exit 1; }

SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SRC_DIR"

# docling.rs v1 resolves runtime models exclusively from `.models/`. Migrate
# the old docmill layout when it is unambiguous; never merge or overwrite two
# independently populated trees.
if [ -d models ] && [ ! -e .models ]; then
  say "migrating legacy models/ to .models/"
  mv models .models
elif [ -d models ] && [ -e .models ]; then
  printf '%s\n' "[package] warning: both models/ and .models/ exist; preserving both and using .models/" >&2
fi
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
[ -n "$VERSION" ] || die "cannot read version from Cargo.toml"
ARCH="$(uname -m)"

WITH_MODELS=false
MODELS_PKG=false
DO_BUILD=true
WANT_TAR=false
WANT_RPM=false
WANT_DEB=false
for arg in "$@"; do
  case "$arg" in
    --models) MODELS_PKG=true ;;
    --with-models) WITH_MODELS=true ;;
    --no-build) DO_BUILD=false ;;
    --tar) WANT_TAR=true ;;
    --rpm) WANT_RPM=true ;;
    --deb) WANT_DEB=true ;;
    *) die "usage: build_packages.sh [--models] [--with-models] [--tar] [--rpm] [--deb] [--no-build]" ;;
  esac
done
[ "$MODELS_PKG" = true ] && [ "$WITH_MODELS" = true ] \
  && die "--models (companion package) and --with-models (one fat package) are mutually exclusive"
# Default: everything the host can build.
if [ "$WANT_TAR" = false ] && [ "$WANT_RPM" = false ] && [ "$WANT_DEB" = false ]; then
  WANT_TAR=true
  command -v rpmbuild >/dev/null 2>&1 && WANT_RPM=true
  command -v dpkg-deb >/dev/null 2>&1 && WANT_DEB=true
fi

# --- 1. Build with a relocatable rpath ---------------------------------------
ORT_DIR=""
if [ -d vendor/onnxruntime/lib ]; then
  ORT_DIR="$SRC_DIR/vendor/onnxruntime/lib"
elif [ -n "${ORT_LIB_LOCATION:-}" ]; then
  ORT_DIR="$ORT_LIB_LOCATION"
fi
if [ "$DO_BUILD" = true ]; then
  say "building release binary (version $VERSION)"
  # $ORIGIN/../lib: the loader resolves bundled libonnxruntime relative to
  # the binary, so the tree works at /opt, /usr/local, or a home directory.
  # shellcheck disable=SC2016 # $ORIGIN must reach the linker literally
  RPATH_FLAG='-C link-arg=-Wl,-rpath,$ORIGIN/../lib'
  if [ -n "$ORT_DIR" ]; then
    say "linking onnxruntime dynamically from $ORT_DIR (bundled into lib/)"
    env ORT_LIB_LOCATION="$ORT_DIR" ORT_PREFER_DYNAMIC_LINK=1 \
        RUSTFLAGS="${RUSTFLAGS:-} $RPATH_FLAG" cargo build --release
  else
    env RUSTFLAGS="${RUSTFLAGS:-} $RPATH_FLAG" cargo build --release
  fi
fi
[ -x target/release/docmill ] || die "no release binary (run without --no-build)"

# --- 2. Stage the tree ---------------------------------------------------------
WORK="target/package"
STAGE="$WORK/stage"
TREE="$STAGE/opt/docmill"
rm -rf "$WORK"
mkdir -p "$TREE/bin" "$STAGE/usr/bin" dist
cp target/release/docmill "$TREE/bin/"
cp LICENSE THIRD-PARTY-NOTICES.md README.md "$TREE/"
mkdir -p "$TREE/scripts/install"
cp scripts/install/download_dependencies.sh "$TREE/scripts/install/"
if [ -n "$ORT_DIR" ]; then
  mkdir -p "$TREE/lib"
  cp -a "$ORT_DIR"/libonnxruntime.so* "$TREE/lib/"
fi
SUFFIX=""
if [ "$WITH_MODELS" = true ]; then
  [ -d .models ] || die "--with-models: no .models/ in the checkout (run download_dependencies.sh)"
  cp -a .models "$TREE/.models"
  [ -d .pdfium ] && cp -a .pdfium "$TREE/.pdfium"
  SUFFIX="-with-models"
fi
ln -s /opt/docmill/bin/docmill "$STAGE/usr/bin/docmill"

POST_HINT="Install the docmill-models package, or run \
/opt/docmill/scripts/install/download_dependencies.sh from \
/opt/docmill to fetch the OCR/layout models (or point the binary at an \
existing model tree with --img-ocr-models-dir)."

# Companion models package: .models/ + .pdfium/ under the same /opt tree.
MSTAGE="$WORK/stage-models"
if [ "$MODELS_PKG" = true ]; then
  [ -d .models ] || die "--models: no .models/ in the checkout (run download_dependencies.sh)"
  MTREE="$MSTAGE/opt/docmill"
  mkdir -p "$MTREE"
  cp -a .models "$MTREE/.models"
  [ -d .pdfium ] && cp -a .pdfium "$MTREE/.pdfium"
fi

# --- 3. Tarball ------------------------------------------------------------------
if [ "$WANT_TAR" = true ]; then
  OUT="dist/docmill-$VERSION-linux-$ARCH$SUFFIX.tar.gz"
  say "tarball: $OUT"
  tar -C "$STAGE/opt" -czf "$OUT" docmill
  if [ "$MODELS_PKG" = true ]; then
    MOUT="dist/docmill-models-$VERSION-linux-$ARCH.tar.gz"
    say "tarball: $MOUT (extract over the same root as the main tree)"
    tar -C "$MSTAGE/opt" -czf "$MOUT" docmill
  fi
fi

# --- 4. RPM ------------------------------------------------------------------------
if [ "$WANT_RPM" = true ]; then
  command -v rpmbuild >/dev/null 2>&1 || die "--rpm: rpmbuild not installed"
  say "rpm (rpmbuild)"
  RPMTOP="$PWD/$WORK/rpm"
  mkdir -p "$RPMTOP"/{SPECS,RPMS,BUILD,BUILDROOT}
  cat > "$RPMTOP/SPECS/docmill.spec" <<EOF
Name: docmill
Version: $VERSION
Release: 1
Summary: Document conversion to Markdown with OCR over embedded pictures
License: MIT
URL: https://github.com/docling-project/docling.rs
AutoReqProv: no
Suggests: docmill-models

%description
Converts documents (DOCX/PDF/HTML/images/...) to Markdown via docling.rs and
OCRs embedded pictures with a pluggable engine chain (local PP-OCRv5 ONNX,
OpenAI-compatible VLM endpoints, PaddleOCR servers). Includes the
docmill serve HTTP service.
$([ "$WITH_MODELS" = true ] && echo "Models are bundled." || echo "Models are fetched post-install: $POST_HINT")

%install
mkdir -p %{buildroot}
cp -a $PWD/$STAGE/. %{buildroot}/

%files
/opt/docmill
/usr/bin/docmill

%post
$([ "$WITH_MODELS" = true ] || echo "echo 'docmill: $POST_HINT'")
EOF
  rpmbuild -bb --quiet --define "_topdir $RPMTOP" "$RPMTOP/SPECS/docmill.spec"
  cp "$RPMTOP"/RPMS/*/docmill-"$VERSION"-1.*.rpm dist/
  [ -n "$SUFFIX" ] && for f in dist/docmill-"$VERSION"-1.*.rpm; do
    case "$f" in *with-models*) ;; *) mv "$f" "${f%.rpm}$SUFFIX.rpm" ;; esac
  done

  if [ "$MODELS_PKG" = true ]; then
    say "rpm: docmill-models"
    cat > "$RPMTOP/SPECS/docmill-models.spec" <<EOF
Name: docmill-models
Version: $VERSION
Release: 1
Summary: OCR/layout models and pdfium for docmill
License: MIT and ASL 2.0 and BSD
URL: https://github.com/docling-project/docling.rs
AutoReqProv: no
Enhances: docmill

%description
The runtime assets docmill resolves next to its binary: PP-OCRv5 and
PP-OCRv3 recognition models (Apache-2.0, PaddleOCR), the docling layout and
TableFormer models (MIT), and libpdfium (BSD-3-Clause). Installing this
package makes the main docmill package fully offline-capable.

%install
mkdir -p %{buildroot}
cp -a $PWD/$MSTAGE/. %{buildroot}/

%files
/opt/docmill
EOF
    rpmbuild -bb --quiet --define "_topdir $RPMTOP" "$RPMTOP/SPECS/docmill-models.spec"
    cp "$RPMTOP"/RPMS/*/docmill-models-"$VERSION"-1.*.rpm dist/
  fi
fi

# --- 5. DEB ------------------------------------------------------------------------
if [ "$WANT_DEB" = true ]; then
  command -v dpkg-deb >/dev/null 2>&1 || die "--deb: dpkg-deb not installed"
  say "deb (dpkg-deb)"
  DEB_ARCH=amd64
  [ "$ARCH" = "aarch64" ] && DEB_ARCH=arm64
  mkdir -p "$STAGE/DEBIAN"
  cat > "$STAGE/DEBIAN/control" <<EOF
Package: docmill
Version: $VERSION
Section: utils
Priority: optional
Architecture: $DEB_ARCH
Maintainer: Krsto Markovic <krstom@gmail.com>
Suggests: docmill-models
Description: Document conversion to Markdown with OCR over embedded pictures
 Converts documents to Markdown via docling.rs and OCRs embedded pictures
 with a pluggable engine chain. Includes the docmill serve service.
EOF
  if [ "$WITH_MODELS" = false ]; then
    printf '#!/bin/sh\necho "docmill: %s"\n' "$POST_HINT" > "$STAGE/DEBIAN/postinst"
    chmod 755 "$STAGE/DEBIAN/postinst"
  fi
  dpkg-deb --build --root-owner-group "$STAGE" \
    "dist/docmill_${VERSION}-1_${DEB_ARCH}${SUFFIX}.deb"
  rm -rf "$STAGE/DEBIAN"

  if [ "$MODELS_PKG" = true ]; then
    say "deb: docmill-models"
    mkdir -p "$MSTAGE/DEBIAN"
    cat > "$MSTAGE/DEBIAN/control" <<EOF
Package: docmill-models
Version: $VERSION
Section: utils
Priority: optional
Architecture: $DEB_ARCH
Maintainer: Krsto Markovic <krstom@gmail.com>
Enhances: docmill
Description: OCR/layout models and pdfium for docmill
 PP-OCRv5/v3 recognition models, the docling layout and TableFormer models,
 and libpdfium — installed into /opt/docmill where the main package's
 binary resolves them. Makes docmill fully offline-capable.
EOF
    dpkg-deb --build --root-owner-group "$MSTAGE" \
      "dist/docmill-models_${VERSION}-1_${DEB_ARCH}.deb"
    rm -rf "$MSTAGE/DEBIAN"
  fi
fi

say "done:"
ls -lh dist/ | tail -n +2
