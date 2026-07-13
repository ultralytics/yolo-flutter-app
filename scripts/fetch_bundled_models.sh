#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#
# Download the nano (n) YOLO26 models into the example app's asset folder at build time so the app ships with them by
# default. The models are NOT committed to git (they are large binaries, gitignored via *.tflite / *.mlpackage.zip);
# this script fetches them on demand during the platform build. YOLOModelResolver checks assets/models/ before falling
# back to a network download, so a bundled model means no first-run download for the user.
#
# Best-effort: if a download fails (e.g. offline build machine) the script warns and exits 0 so the build still
# succeeds — the app simply falls back to the existing runtime download for any missing model. Bundling is skipped
# entirely under CI (CI / GITHUB_ACTIONS) to keep GitHub builds fast and off the network; set FORCE_BUNDLED_MODELS=1
# to override.
#
# Usage: fetch_bundled_models.sh <android|ios>
#   android -> *_w8a32.tflite from the yolo-flutter-app release
#   ios     -> *.mlpackage.zip from the yolo-ios-app release (the resolver extracts these on first use)
#
# Keep the release tags and the nano file list in sync with lib/core/yolo_model_resolver.dart.

set -u

PLATFORM="${1:-}"
if [ "$PLATFORM" != "android" ] && [ "$PLATFORM" != "ios" ]; then
  echo "fetch_bundled_models: usage: $0 <android|ios>" >&2
  exit 0
fi

# Skip bundling under CI. GitHub Actions builds (example-android / example-ios) don't need the models embedded, and
# downloading six of them on every run is slow and a network-flakiness risk. CI exercises the runtime-download fallback
# instead. Set FORCE_BUNDLED_MODELS=1 to bundle anyway (e.g. a release build that intentionally ships them).
if [ "${FORCE_BUNDLED_MODELS:-}" != "1" ] && { [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; }; then
  echo "fetch_bundled_models: CI detected (CI/GITHUB_ACTIONS set); skipping model bundling — app will download at runtime."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$REPO_ROOT/example/assets/models"

# Release sources (must match YOLOModelResolver constants).
ANDROID_BASE="https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6"
IOS_BASE="https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0"

# The nano task family bundled by default. Depth is Android-only until the published iOS SDK exposes its bridge.
ANDROID_FILES=(
  "yolo26n_w8a32.tflite"
  "yolo26n-seg_w8a32.tflite"
  "yolo26n-sem_w8a32.tflite"
  "yolo26n-depth_w8a32.tflite"
  "yolo26n-cls_w8a32.tflite"
  "yolo26n-pose_w8a32.tflite"
  "yolo26n-obb_w8a32.tflite"
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
  OTHER_FILES=("${IOS_FILES[@]}")
else
  BASE="$IOS_BASE"
  FILES=("${IOS_FILES[@]}")
  OTHER_FILES=("${ANDROID_FILES[@]}")
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
    if [ ! -s "$tmp" ]; then
      echo "fetch_bundled_models: WARNING $name downloaded 0 bytes; will fall back to runtime download" >&2
      rm -f "$tmp"
    elif [[ "$name" == *.zip ]] && ! unzip -tqq "$tmp" > /dev/null 2>&1; then
      # A truncated or error-page response can still be a non-empty 200; never bundle a corrupt archive.
      echo "fetch_bundled_models: WARNING $name archive is corrupt or truncated; will fall back to runtime download" >&2
      rm -f "$tmp"
    else
      mv -f "$tmp" "$out"
    fi
  else
    echo "fetch_bundled_models: WARNING failed to download $name; will fall back to runtime download" >&2
    rm -f "$tmp"
  fi
}

for f in "${FILES[@]}"; do
  fetch "$f"
done

# Keep assets/models/ single-platform: Flutter bundles the whole folder, so the other platform's bundled models would
# otherwise ship in this build — e.g. the iOS Core ML packages (~14 MB) are dead weight in the Android APK and never
# loaded. Only the known auto-fetched filenames are removed (re-fetched on demand), so any custom or benchmark model a
# developer dropped into assets/models/ is left untouched.
for f in "${OTHER_FILES[@]}"; do
  rm -f "$DEST/$f"
done

# Always succeed: bundling is an optimization, never a hard build dependency.
exit 0
