#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#
# Build the local Google Play Console upload assets for the example Android app.

set -euo pipefail

usage() {
  echo "Usage: $0 [--notes path/to/whats-new.txt]" >&2
}

NOTES_SOURCE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --notes)
      if [ "$#" -lt 2 ]; then
        usage
        exit 2
      fi
      NOTES_SOURCE="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STORE_DIR="$REPO_ROOT/play-store-assets"
EXAMPLE_DIR="$REPO_ROOT/example"

pubspec_version() {
  awk -F': *' '$1 == "version" { gsub(/"/, "", $2); print $2; exit }' "$1"
}

PACKAGE_VERSION="$(pubspec_version "$REPO_ROOT/pubspec.yaml")"
EXAMPLE_VERSION="$(pubspec_version "$EXAMPLE_DIR/pubspec.yaml")"
VERSION_NAME="${EXAMPLE_VERSION%%+*}"
VERSION_CODE="${EXAMPLE_VERSION##*+}"

if [ -z "$PACKAGE_VERSION" ] || [ -z "$EXAMPLE_VERSION" ]; then
  echo "build_play_store_assets: failed to read package versions" >&2
  exit 1
fi

if [ "$EXAMPLE_VERSION" = "$VERSION_CODE" ]; then
  echo "build_play_store_assets: example/pubspec.yaml version must include a build number, e.g. 0.5.1+5" >&2
  exit 1
fi

if [ "$PACKAGE_VERSION" != "$VERSION_NAME" ]; then
  echo "build_play_store_assets: pubspec.yaml ($PACKAGE_VERSION) does not match example version name ($VERSION_NAME)" >&2
  exit 1
fi

if ! command -v flutter > /dev/null; then
  echo "build_play_store_assets: flutter is not on PATH" >&2
  exit 1
fi

KEY_PROPERTIES="$EXAMPLE_DIR/android/key.properties"
if [ ! -f "$KEY_PROPERTIES" ]; then
  echo "build_play_store_assets: missing $KEY_PROPERTIES; release builds must use the Play upload key" >&2
  exit 1
fi

STORE_FILE="$(awk -F= '$1 == "storeFile" { print $2; exit }' "$KEY_PROPERTIES")"
if [ -n "$STORE_FILE" ] && [ ! -f "$EXAMPLE_DIR/android/$STORE_FILE" ]; then
  echo "build_play_store_assets: missing Android keystore $EXAMPLE_DIR/android/$STORE_FILE" >&2
  exit 1
fi

extract_changelog_notes() {
  awk -v version="$VERSION_NAME" '
    $0 == "## " version { in_section = 1; next }
    in_section && /^## / { exit }
    !in_section { next }
    /^- / {
      if (note != "") print note
      note = substr($0, 3)
      next
    }
    /^[[:space:]]+/ && note != "" {
      sub(/^[[:space:]]+/, "")
      note = note " " $0
      next
    }
    END {
      if (note != "") print note
    }
  ' "$REPO_ROOT/CHANGELOG.md" | sed -E \
    -e 's/\*\*([^*]+)\*\*/\1/g' \
    -e 's/`([^`]+)`/\1/g' \
    -e 's/[[:space:]]+/ /g'
}

echo "build_play_store_assets: building ultralytics_yolo $VERSION_NAME+$VERSION_CODE"

(cd "$REPO_ROOT" && flutter pub get)
(cd "$EXAMPLE_DIR" && flutter pub get)
(cd "$EXAMPLE_DIR" && flutter build appbundle --release)

mkdir -p "$STORE_DIR"
find "$STORE_DIR" -maxdepth 1 -type f \( \
  -name 'ultralytics-yolo-*-build*.aab' -o \
  -name 'ultralytics-yolo-*-whats-new.txt' -o \
  -name 'ultralytics-yolo-*-sha256.txt' \
  \) -delete

AAB_SOURCE="$EXAMPLE_DIR/build/app/outputs/bundle/release/app-release.aab"
AAB_DEST="$STORE_DIR/ultralytics-yolo-$VERSION_NAME-build$VERSION_CODE.aab"
NOTES_DEST="$STORE_DIR/ultralytics-yolo-$VERSION_NAME-whats-new.txt"
SHA_DEST="$STORE_DIR/ultralytics-yolo-$VERSION_NAME-sha256.txt"
FEATURE_GRAPHIC="$STORE_DIR/ultralytics-yolo-feature-graphic.png"

install -m 0644 "$AAB_SOURCE" "$AAB_DEST"

if [ -n "$NOTES_SOURCE" ]; then
  install -m 0644 "$NOTES_SOURCE" "$NOTES_DEST"
else
  extract_changelog_notes > "$NOTES_DEST"
fi

if [ ! -s "$NOTES_DEST" ]; then
  echo "build_play_store_assets: generated empty Play release notes at $NOTES_DEST" >&2
  exit 1
fi

NOTES_BYTES="$(wc -c < "$NOTES_DEST" | tr -d ' ')"
if [ "$NOTES_BYTES" -gt 500 ]; then
  echo "build_play_store_assets: warning: Play release notes are $NOTES_BYTES bytes; Play Console may require shortening" >&2
fi

if command -v jarsigner > /dev/null; then
  if jarsigner -verify "$AAB_DEST" > /dev/null 2>&1; then
    echo "build_play_store_assets: verified AAB signature"
  else
    echo "build_play_store_assets: AAB signature verification failed" >&2
    jarsigner -verify "$AAB_DEST" >&2
    exit 1
  fi
else
  echo "build_play_store_assets: warning: jarsigner not found; skipped signature verification" >&2
fi

if [ -f "$FEATURE_GRAPHIC" ]; then
  (
    cd "$STORE_DIR"
    shasum -a 256 "$(basename "$AAB_DEST")" "$(basename "$NOTES_DEST")" "$(basename "$FEATURE_GRAPHIC")"
  ) > "$SHA_DEST"
else
  (
    cd "$STORE_DIR"
    shasum -a 256 "$(basename "$AAB_DEST")" "$(basename "$NOTES_DEST")"
  ) > "$SHA_DEST"
fi

echo "build_play_store_assets: wrote:"
echo "  $AAB_DEST"
echo "  $NOTES_DEST"
echo "  $SHA_DEST"
if [ -f "$FEATURE_GRAPHIC" ]; then
  echo "  $FEATURE_GRAPHIC"
fi
echo
echo "Next Play Console uploads:"
echo "  App bundle: $AAB_DEST"
echo "  Release notes: $NOTES_DEST"
if [ -f "$FEATURE_GRAPHIC" ]; then
  echo "  Store listing feature graphic, if changed: $FEATURE_GRAPHIC"
fi
