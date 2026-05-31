#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#
# Download the nano (n) YOLO26 models into the example app's asset folder at build time so the app ships with them by
# default. The models are NOT committed to git (they are large binaries, gitignored via *.tflite / *.mlpackage.zip);
# this script fetches them on demand during the platform build. YOLOModelResolver checks assets/models/ before falling
# back to a network download, so a bundled model means no first-run download for the user.
#
# Best-effort: if a download fails (e.g. offline build machine) the script warns and exits 0 so the build still
# succeeds — the app simply falls back to the existing runtime download for any missing model.
#
# Usage: fetch_bundled_models.sh <android|ios>
#   android -> *_int8.tflite from the yolo-flutter-app release
#   ios     -> *.mlpackage.zip from the yolo-ios-app release (the resolver extracts these on first use)
#
# Keep the release tags and the nano file list in sync with lib/core/yolo_model_resolver.dart.

set -u

PLATFORM="${1:-}"
if [ "$PLATFORM" != "android" ] && [ "$PLATFORM" != "ios" ]; then
  echo "fetch_bundled_models: usage: $0 <android|ios>" >&2
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$REPO_ROOT/example/assets/models"

# Release sources (must match YOLOModelResolver constants).
ANDROID_BASE="https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5"
IOS_BASE="https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"

# The nano task family bundled by default: detect, segment, semantic, classify, pose, obb.
ANDROID_FILES=(
  "yolo26n_int8.tflite"
  "yolo26n-seg_int8.tflite"
  "yolo26n-sem_int8.tflite"
  "yolo26n-cls_int8.tflite"
  "yolo26n-pose_int8.tflite"
  "yolo26n-obb_int8.tflite"
)
IOS_FILES=(
  "yolo26n.mlpackage.zip"
  "yolo26n-seg.mlpackage.zip"
  "yolo26n-sem.mlpackage.zip"
  "yolo26n-cls.mlpackage.zip"
  "yolo26n-pose.mlpackage.zip"
  "yolo26n-obb.mlpackage.zip"
)

if [ "$PLATFORM" = "android" ]; then
  BASE="$ANDROID_BASE"
  FILES=("${ANDROID_FILES[@]}")
else
  BASE="$IOS_BASE"
  FILES=("${IOS_FILES[@]}")
fi

mkdir -p "$DEST"

fetch() {
  # Download $1 from $BASE into $DEST, atomically, skipping if already present and non-empty.
  local name="$1"
  local out="$DEST/$name"
  if [ -s "$out" ]; then
    echo "fetch_bundled_models: have $name"
    return 0
  fi
  local tmp="$out.download"
  rm -f "$tmp"
  echo "fetch_bundled_models: downloading $name"
  if curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$tmp" "$BASE/$name"; then
    if [ -s "$tmp" ]; then
      mv -f "$tmp" "$out"
    else
      echo "fetch_bundled_models: WARNING $name downloaded 0 bytes; will fall back to runtime download" >&2
      rm -f "$tmp"
    fi
  else
    echo "fetch_bundled_models: WARNING failed to download $name; will fall back to runtime download" >&2
    rm -f "$tmp"
  fi
}

for f in "${FILES[@]}"; do
  fetch "$f"
done

# Always succeed: bundling is an optimization, never a hard build dependency.
exit 0
