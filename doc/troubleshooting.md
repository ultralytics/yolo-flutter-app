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

- `minSdkVersion 23`
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

### Android detections run slow / not on GPU

Android inference runs on LiteRT 2.x with an automatic GPU → CPU accelerator ladder. Official int8 YOLO26 TFLite assets can compile on the LiteRT GPU path on supported devices, but int8 GPU coverage depends on the device driver and graph; unsupported graphs or ops may fall back to CPU. To compare against fp16, export a non-end-to-end TFLite model:

```python
YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
```

On a Samsung Galaxy S26, the official `yolo26n_int8.tflite` compiled with the LiteRT OpenCL GPU delegate and ran around 15 FPS / 32 ms in the live camera example. Leave `useGpu: true` (the default), inspect LiteRT logs for `LITERT_CL` or CPU fallback, and benchmark the exact model you plan to ship.

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

## 🧱 Release Build Crashes / No Detections (Android)

If everything works in debug but a release build crashes on model load or returns no detections, R8 may have stripped the LiteRT 2.x classes that the native code reaches via JNI/reflection.

The plugin ships consumer R8 rules that keep these classes automatically, so most apps need no extra setup. If you use a custom R8/ProGuard configuration that overrides them, add to `android/app/proguard-rules.pro`:

```pro
-keep class com.google.ai.edge.litert.** { *; }
-keep interface com.google.ai.edge.litert.** { *; }
-dontwarn com.google.ai.edge.litert.**
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**
```

## ✅ Fast Debug Checklist

1. Verify the model path or official ID.
2. Check `YOLO.officialModels()` on the running platform.
3. Inspect metadata with `YOLO.inspectModel()`.
4. Omit `task` unless the export lacks metadata.
5. Test first with `yolo26n`.
