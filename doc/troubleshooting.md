---
title: Troubleshooting
description: Common installation, model-loading, and runtime issues in YOLO Flutter
path: /integrations/flutter/troubleshooting/
---

# Troubleshooting Guide

This guide focuses on the current plugin flow: metadata-first model resolution, official model IDs, and shared switching behavior across `YOLO`, `YOLOView`, and `YOLOViewController`.

## 🚨 Installation Problems

### MissingPluginException

```bash
flutter clean
flutter pub get
cd ios && pod install --repo-update
cd ..
flutter run
```

### iOS build errors

Make sure:

- the iOS deployment target is `13.0` or higher
- CocoaPods is installed
- the app was fully restarted after adding the plugin

### Android SDK / API errors

Make sure your app uses:

- `minSdkVersion 21`
- `compileSdkVersion 36`
- `targetSdkVersion 36`

## 📦 Model Loading Problems

### Model file not found

Check what the plugin can see:

```dart
final exists = await YOLO.checkModelExists('yolo26n');
print(exists);

final paths = await YOLO.getStoragePaths();
print(paths);
```

If you use a custom model:

- Android Flutter assets should point at `.tflite`
- iOS Flutter assets should point at `.mlpackage.zip`
- iOS bundled models should be added to `ios/Runner.xcworkspace`

### Model loads only on one platform

Do not assume the same official artifact exists everywhere. Use:

```dart
final models = YOLO.officialModels();
print(models);
```

That list reflects the current platform only.

### Task mismatch errors

If you pass `task`, it must match the model metadata when metadata exists.

If you are unsure, inspect the model first:

```dart
final info = await YOLO.inspectModel('assets/models/custom.tflite');
print(info);
```

Then either:

- omit `task` and trust metadata
- or pass the correct matching `task`

## 🔄 switchModel Problems

`YOLOViewController.switchModel()` now uses the same resolver as normal model loading.

That means these are all valid:

```dart
await controller.switchModel('yolo26n');
await controller.switchModel('assets/models/custom.tflite', YOLOTask.detect);
await controller.switchModel('https://example.com/model.tflite', YOLOTask.detect);
```

If switching fails:

1. verify the path or URL is real
2. check whether the custom export carries `task`
3. inspect the model metadata
4. read the thrown error instead of retrying with path hacks

## 📷 Camera Starts But No Inference Runs

If the camera appears but you get no detections:

- the model may have failed to load
- the path may be wrong
- the task may not match metadata
- the thresholds may be too aggressive

Start with a known-good path:

```dart
YOLOView(
  modelPath: 'yolo26n',
  onResult: (results) {
    print(results.length);
  },
)
```

Then switch to your custom model only after the baseline works.

## ⚡ Slow Performance

Start with the basic fixes first:

- use `yolo26n`
- increase `confidenceThreshold`
- reduce `numItemsThreshold`
- lower `maxFPS` in `YOLOStreamingConfig`
- disable masks or original frame delivery if you do not need them

Example:

```dart
await controller.setThresholds(
  confidenceThreshold: 0.6,
  iouThreshold: 0.3,
  numItemsThreshold: 10,
);
```

## 🧠 Memory Issues

If memory use grows too high:

- dispose unused `YOLO` instances
- avoid unnecessary multi-instance setups
- prefer smaller models
- keep one active camera screen at a time

```dart
await yolo.dispose();
```

## 🖥️ GPU Issues

If model loading or inference is unstable on specific devices, disable GPU:

```dart
final yolo = YOLO(
  modelPath: 'yolo26n',
  useGpu: false,
);
```

The same applies to `YOLOView`.

## ✅ Fast Debug Checklist

1. Verify the model path or official ID.
2. Check `YOLO.officialModels()` on the running platform.
3. Inspect metadata with `YOLO.inspectModel()`.
4. Omit `task` unless the export lacks metadata.
5. Test first with `yolo26n`.
