#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#
# Build the local Google Play Console upload assets for the example Android app.

set -euo pipefail
shopt -s dotglob

usage() {
  echo "Usage: $0 [--notes path/to/whats-new.txt] | --verify-aab path/to/app.aab" >&2
}

NOTES_SOURCE=""
VERIFY_AAB=""
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
    --verify-aab)
      if [ "$#" -lt 2 ]; then
        usage
        exit 2
      fi
      VERIFY_AAB="$2"
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

verify_aab_entrypoint() {
  local aab="$1"
  local dex
  local tmp_dex
  local tmp_manifest

  tmp_dex="$(mktemp)"
  tmp_manifest="$(mktemp)"
  unzip -p "$aab" base/manifest/AndroidManifest.xml > "$tmp_manifest"
  if ! grep -a -q 'com.ultralytics.yolo.MainActivity' "$tmp_manifest" \
    || ! grep -a -q 'android.intent.action.MAIN' "$tmp_manifest" \
    || ! grep -a -q 'android.intent.category.LAUNCHER' "$tmp_manifest"; then
    rm -f "$tmp_dex" "$tmp_manifest"
    echo "build_play_store_assets: release AAB manifest is missing the MainActivity launcher entrypoint" >&2
    exit 1
  fi

  while IFS= read -r dex; do
    unzip -p "$aab" "$dex" > "$tmp_dex"
    if grep -a -q 'Lcom/ultralytics/yolo/MainActivity;' "$tmp_dex"; then
      rm -f "$tmp_dex" "$tmp_manifest"
      return 0
    fi
  done < <(unzip -Z1 "$aab" | grep -E '^base/dex/classes[0-9]*\.dex$')

  rm -f "$tmp_dex" "$tmp_manifest"
  echo "build_play_store_assets: release AAB is missing com.ultralytics.yolo.MainActivity" >&2
  exit 1
}

if [ -n "$VERIFY_AAB" ]; then
  verify_aab_entrypoint "$VERIFY_AAB"
  echo "build_play_store_assets: verified Android release entrypoint in $VERIFY_AAB"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STORE_DIR="$REPO_ROOT/play-store-assets"
EXAMPLE_DIR="$REPO_ROOT/example"
MODEL_DIR="$EXAMPLE_DIR/assets/models"
TMP_STORE_DIR=""
MODEL_STASH_DIR=""

cleanup() {
  local path
  if [ -n "$MODEL_STASH_DIR" ] && [ -d "$MODEL_STASH_DIR" ]; then
    for path in "$MODEL_STASH_DIR"/*; do
      [ -e "$path" ] || continue
      mv -f "$path" "$MODEL_DIR/"
    done
    rmdir "$MODEL_STASH_DIR"
  fi
  if [ -n "$TMP_STORE_DIR" ]; then
    rm -rf "$TMP_STORE_DIR"
  fi
}
trap cleanup EXIT

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

if command -v shasum > /dev/null; then
  CHECKSUM_CMD=(shasum -a 256)
elif command -v sha256sum > /dev/null; then
  CHECKSUM_CMD=(sha256sum)
else
  echo "build_play_store_assets: shasum or sha256sum is required to write the checksum manifest" >&2
  exit 1
fi

KEY_PROPERTIES="$EXAMPLE_DIR/android/key.properties"
if [ ! -f "$KEY_PROPERTIES" ]; then
  echo "build_play_store_assets: missing $KEY_PROPERTIES; release builds must use the Play upload key" >&2
  exit 1
fi

STORE_FILE="$(
  awk -F= '
    $1 ~ /^[[:space:]]*storeFile[[:space:]]*$/ {
      gsub(/\r/, "", $2)
      sub(/^[[:space:]]+/, "", $2)
      sub(/[[:space:]]+$/, "", $2)
      print $2
      exit
    }
  ' "$KEY_PROPERTIES"
)"
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

mkdir -p "$STORE_DIR"
mkdir -p "$MODEL_DIR"
MODEL_STASH_DIR="$(mktemp -d "$STORE_DIR/.tmp-model-assets.XXXXXX")"
for path in "$MODEL_DIR"/*; do
  [ -e "$path" ] || continue
  [ -f "$path" ] || continue
  case "$(basename "$path")" in
    .gitkeep | \
      yolo26n_w8a32.tflite | \
      yolo26n-seg_w8a32.tflite | \
      yolo26n-sem_w8a32.tflite | \
      yolo26n-depth_w8a32.tflite | \
      yolo26n-cls_w8a32.tflite | \
      yolo26n-pose_w8a32.tflite | \
      yolo26n-obb_w8a32.tflite) ;;
    *)
      mv "$path" "$MODEL_STASH_DIR/"
      ;;
  esac
done

(cd "$REPO_ROOT" && flutter pub get)
(cd "$EXAMPLE_DIR" && flutter pub get)
(cd "$EXAMPLE_DIR" && flutter build appbundle --release)

TMP_STORE_DIR="$(mktemp -d "$STORE_DIR/.tmp-play-store-assets.XXXXXX")"

AAB_SOURCE="$EXAMPLE_DIR/build/app/outputs/bundle/release/app-release.aab"
AAB_DEST="$STORE_DIR/ultralytics-yolo-$VERSION_NAME-build$VERSION_CODE.aab"
NOTES_DEST="$STORE_DIR/ultralytics-yolo-$VERSION_NAME-whats-new.txt"
SHA_DEST="$STORE_DIR/ultralytics-yolo-$VERSION_NAME-sha256.txt"
FEATURE_GRAPHIC="$STORE_DIR/ultralytics-yolo-feature-graphic.png"
TMP_AAB="$TMP_STORE_DIR/$(basename "$AAB_DEST")"
TMP_NOTES="$TMP_STORE_DIR/$(basename "$NOTES_DEST")"
TMP_SHA="$TMP_STORE_DIR/$(basename "$SHA_DEST")"
TMP_FEATURE_GRAPHIC="$TMP_STORE_DIR/$(basename "$FEATURE_GRAPHIC")"

install -m 0644 "$AAB_SOURCE" "$TMP_AAB"

verify_aab_entrypoint "$TMP_AAB"

if [ -n "$NOTES_SOURCE" ]; then
  install -m 0644 "$NOTES_SOURCE" "$TMP_NOTES"
else
  extract_changelog_notes > "$TMP_NOTES"
fi

if [ ! -s "$TMP_NOTES" ]; then
  echo "build_play_store_assets: generated empty Play release notes at $TMP_NOTES" >&2
  exit 1
fi

NOTES_BYTES="$(wc -c < "$TMP_NOTES" | tr -d ' ')"
if [ "$NOTES_BYTES" -gt 500 ]; then
  echo "build_play_store_assets: warning: Play release notes are $NOTES_BYTES bytes; Play Console may require shortening" >&2
fi

if command -v jarsigner > /dev/null; then
  if jarsigner -verify "$TMP_AAB" > /dev/null 2>&1; then
    echo "build_play_store_assets: verified AAB signature"
  else
    echo "build_play_store_assets: AAB signature verification failed" >&2
    jarsigner -verify "$TMP_AAB" >&2
    exit 1
  fi
else
  echo "build_play_store_assets: warning: jarsigner not found; skipped signature verification" >&2
fi

if [ -f "$FEATURE_GRAPHIC" ]; then
  install -m 0644 "$FEATURE_GRAPHIC" "$TMP_FEATURE_GRAPHIC"
  (
    cd "$TMP_STORE_DIR"
    "${CHECKSUM_CMD[@]}" "$(basename "$TMP_AAB")" "$(basename "$TMP_NOTES")" "$(basename "$TMP_FEATURE_GRAPHIC")"
  ) > "$TMP_SHA"
else
  (
    cd "$TMP_STORE_DIR"
    "${CHECKSUM_CMD[@]}" "$(basename "$TMP_AAB")" "$(basename "$TMP_NOTES")"
  ) > "$TMP_SHA"
fi

find "$STORE_DIR" -maxdepth 1 -type f \( \
  -name 'ultralytics-yolo-*-build*.aab' -o \
  -name 'ultralytics-yolo-*-whats-new.txt' -o \
  -name 'ultralytics-yolo-*-sha256.txt' \
  \) -delete
install -m 0644 "$TMP_AAB" "$AAB_DEST"
install -m 0644 "$TMP_NOTES" "$NOTES_DEST"
install -m 0644 "$TMP_SHA" "$SHA_DEST"

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
