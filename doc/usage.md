---
title: Usage Guide
description: Practical patterns for single-image inference, camera inference, model switching, and multi-instance use in YOLO Flutter
path: /integrations/flutter/usage/
---

# Usage Guide

This guide shows the common ways to use the plugin in real apps without reintroducing model-management logic in your own code.

## 🎯 Single-Image Inference

Use `YOLO` when you already have image bytes:

```dart
import 'dart:io';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

class ObjectDetector {
  late final YOLO yolo;

  Future<void> initialize() async {
    yolo = YOLO(modelPath: 'yolo26n');
    await yolo.loadModel();
  }

  Future<List<dynamic>> detect(File imageFile) async {
    final imageBytes = await imageFile.readAsBytes();
    final results = await yolo.predict(imageBytes);
    return results['boxes'] ?? [];
  }
}
```

## 📷 Real-Time Camera Inference

Use `YOLOView` for camera-based inference:

```dart
final controller = YOLOViewController();

YOLOView(
  modelPath: 'yolo26n',
  controller: controller,
  onResult: (results) {
    print('Detections: ${results.length}');
  },
  onPerformanceMetrics: (metrics) {
    print('FPS: ${metrics.fps.toStringAsFixed(1)}');
  },
)
```

Use `YOLOShowcase` when you want the complete Ultralytics camera UI with model/task controls, thresholds, lens controls, capture, and performance labels:

```dart
YOLOShowcase(
  initialTask: YOLOTask.detect,
  initialModelSize: 'n',
  onCapture: (bytes) {},
)
```

## 🔄 Migrating From 0.3.x Overlay/UI APIs

Version 0.4.0 removes the old Dart-side overlay/control layer. `YOLOView` is now the camera + native detection-rendering surface. The full app UI lives in `YOLOShowcase`, and reusable controls are exported as normal Flutter widgets for custom layouts.

| Removed 0.3.x API                                | 0.4.0 replacement                                                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `YOLOOverlay`, `YOLOOverlayTheme`                | Remove these widgets. Use native `YOLOView` overlays, or consume `onResult`/`YOLO.predict()` data.                        |
| `YOLOControls`                                   | Use `YOLOShowcase` for the full UI, or compose `TaskSegmentedControl`, `ThresholdSliderRow`, etc.                         |
| `YOLOView.showNativeUI`                          | Use `YOLOShowcase` for built-in controls; use bare `YOLOView` when building your own UI.                                  |
| `YOLOView.showOverlays`, `YOLOView.overlayTheme` | No constructor replacement. Camera overlay drawing is native and not themed from Dart.                                    |
| `YOLOViewController.setShowUIControls()`         | Show/hide your own Flutter controls around `YOLOView`.                                                                    |
| `YOLOViewController.setShowOverlays()`           | Still available: toggles native overlay rendering. `capturePhoto(withOverlays: false)` only affects captured JPEG output. |

Typical upgrade:

```dart
// 0.3.x: YOLOView plus package-provided Dart controls/overlays.
YOLOView(
  modelPath: 'yolo11n',
  onResult: (results) {},
)

// 0.4.0: full built-in experience.
YOLOShowcase()

// 0.4.0: custom experience.
Stack(
  children: [
    YOLOView(modelPath: 'yolo26n', controller: controller),
    Align(
      alignment: Alignment.bottomCenter,
      child: ThresholdSliderRow(
        label: 'Confidence',
        value: confidence,
        min: 0.0,
        max: 1.0,
        onChanged: controller.setConfidenceThreshold,
      ),
    ),
  ],
)
```

## 🧠 Using Custom Models

Custom models work the same way:

```dart
final yolo = YOLO(modelPath: 'assets/models/custom.tflite');
```

Task and labels are auto-detected from the model's embedded metadata (Ultralytics appended-ZIP metadata, with a TFLite FlatBuffers fallback). If metadata is missing, pass `task` explicitly:

```dart
final yolo = YOLO(
  modelPath: 'assets/models/custom.tflite',
  task: YOLOTask.detect,
);
```

On Android, inference runs on LiteRT 2.x with an automatic GPU → CPU accelerator ladder. Official int8 YOLO26 TFLite assets can compile on the LiteRT GPU path on supported devices, but int8 GPU coverage depends on the device driver and graph; unsupported graphs or ops may fall back to CPU. For GPU benchmarking, fp16 non-end-to-end exports are also useful:

```python
YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
```

Keep `useGpu: true` for automatic GPU -> CPU fallback and verify actual delegate placement on target devices.

## 🔍 Task-Specific Result Access

### Detection

```dart
final results = await yolo.predict(imageBytes);
for (final box in results['boxes'] ?? <dynamic>[]) {
  print('${box['class']}: ${box['confidence']}');
}
```

### Semantic Segmentation

```dart
final segmenter = YOLO(
  modelPath: 'yolo26n-sem',
  task: YOLOTask.semantic,
);
await segmenter.loadModel();

final results = await segmenter.predict(imageBytes);
final semanticMask = results['semanticMask'] as Map?;
print('Mask: ${semanticMask?['width']}x${semanticMask?['height']}');
```

### Classification

```dart
final classifier = YOLO(
  modelPath: 'assets/models/custom-cls.tflite',
  task: YOLOTask.classify,
);
await classifier.loadModel();

final results = await classifier.predict(imageBytes);
for (final item in results['detections'] ?? <dynamic>[]) {
  print('${item['className']}: ${item['confidence']}');
}
```

### Pose

```dart
final poseModel = YOLO(
  modelPath: 'assets/models/custom-pose.tflite',
  task: YOLOTask.pose,
);
await poseModel.loadModel();

final results = await poseModel.predict(imageBytes);
for (final pose in results['detections'] ?? <dynamic>[]) {
  print('Keypoints: ${(pose['keypoints'] as List?)?.length ?? 0}');
}
```

### OBB

```dart
final obbModel = YOLO(
  modelPath: 'assets/models/custom-obb.tflite',
  task: YOLOTask.obb,
);
await obbModel.loadModel();

final results = await obbModel.predict(imageBytes);
for (final detection in results['detections'] ?? <dynamic>[]) {
  final result = YOLOResult.fromMap(Map<String, dynamic>.from(detection));
  print('${result.className}: angle=${result.angle}');
}
```

## 🔄 Switching Models

Camera model switching uses the same resolver as normal loading:

```dart
final controller = YOLOViewController();

await controller.switchModel('yolo26n');
await controller.switchModel('assets/models/custom.tflite', YOLOTask.detect);
```

That means switching supports:

- official model IDs
- asset paths
- local file paths
- remote URLs
- metadata-inferred tasks

## 📡 Streaming Configuration

Use a throttled config when you want steadier battery usage:

```dart
YOLOView(
  modelPath: 'yolo26n',
  streamingConfig: YOLOStreamingConfig.throttled(
    maxFPS: 15,
    includeMasks: false,
    includeOriginalImage: false,
  ),
  onStreamingData: (data) {
    final detections = data['detections'] as List? ?? [];
    print('Streaming detections: ${detections.length}');
  },
)
```

## 🧩 Multi-Instance

Use multiple `YOLO` instances when you actually need more than one model loaded at once:

```dart
final detector = YOLO(
  modelPath: 'yolo26n',
  useMultiInstance: true,
);

final classifier = YOLO(
  modelPath: 'assets/models/custom-cls.tflite',
  task: YOLOTask.classify,
  useMultiInstance: true,
);

await Future.wait([
  detector.loadModel(),
  classifier.loadModel(),
]);
```

Keep this for real cases such as:

- running detection and classification side by side
- A/B model comparisons
- benchmark tooling

If you only need one active model, keep a single instance.

## 🧱 End-To-End Examples

### 1. Camera-first app with the default official model

```dart
class LiveDetectionScreen extends StatelessWidget {
  final controller = YOLOViewController();

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      modelPath: 'yolo26n',
      controller: controller,
      onResult: (results) {
        if (results.isNotEmpty) {
          print(results.first.className);
        }
      },
    );
  }
}
```

### 2. Photo picker flow with a custom bundled model

```dart
final yolo = YOLO(
  modelPath: 'assets/models/custom-seg.tflite',
  task: YOLOTask.segment,
);

await yolo.loadModel();
final results = await yolo.predict(imageBytes);
final masks = results['masks'] ?? [];
```

### 3. Runtime switching between an official model and a custom export

```dart
final controller = YOLOViewController();

await controller.switchModel('yolo26n');
await controller.switchModel('assets/models/custom-pose.tflite', YOLOTask.pose);
```

## ✅ Practical Guidance

- Start with an official ID when you want the fewest moving parts.
- Use custom asset paths when shipping your own model with the app.
- Use `YOLO.inspectModel()` when you need to see task or labels before loading.
- Let metadata drive `task` whenever possible.
- Use `YOLOViewController.switchModel()` instead of rebuilding the whole camera view just to swap models.
