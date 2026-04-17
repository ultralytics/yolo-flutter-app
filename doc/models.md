---
title: Model Integration
description: How official models, custom exports, metadata inspection, and platform asset loading work in YOLO Flutter
path: /integrations/flutter/models/
---

# Model Integration Guide

This plugin supports two model paths:

- official Ultralytics release assets resolved by model ID
- your own exported models loaded from assets, local files, or remote URLs

The plugin treats model metadata as the source of truth whenever it is available.

## 📦 Official Models

Use the default official model or a specific official model ID such as
`yolo26n`:

```dart
final yolo = YOLO(modelPath: YOLO.defaultOfficialModel() ?? 'yolo26n');
```

The plugin will:

1. resolve the current platform's artifact
2. download it if needed
3. cache it in app storage
4. read metadata to determine the task when possible

To see which official IDs exist on the current platform:

```dart
final models = YOLO.officialModels();
print(models);
```

`YOLO.officialModels()` only returns real downloadable artifacts for the running platform.

If you want the simplest “start from the default Ultralytics model” entry
point, prefer `YOLO.defaultOfficialModel()`.

## 📁 Custom Models

You can also point the plugin at your own fine-tuned exported model:

```dart
final yolo = YOLO(modelPath: 'assets/models/my-finetuned-model.tflite');
```

Supported sources:

- official model ID, for example `yolo26n`
- Flutter asset path
- local file path
- `http` or `https` URL

If the exported model metadata includes `task`, the plugin resolves it automatically. If metadata is missing or ambiguous, pass `task` explicitly:

```dart
final yolo = YOLO(
  modelPath: 'assets/models/my-finetuned-model.tflite',
  task: YOLOTask.detect,
);
```

## 🍳 Custom Model Cookbook

### 1. Bundled Flutter asset on Android

```dart
final yolo = YOLO(modelPath: 'assets/models/custom.tflite');
```

### 2. Bundled Flutter asset on iOS

```dart
final yolo = YOLO(modelPath: 'assets/models/custom.mlpackage.zip');
```

### 3. Model added directly to the iOS app bundle

```dart
final yolo = YOLO(modelPath: 'MyModel.mlpackage');
```

### 4. Local file already downloaded by your app

```dart
final yolo = YOLO(modelPath: file.path);
```

### 5. Remote URL resolved by the plugin

```dart
final yolo = YOLO(
  modelPath: 'https://example.com/models/custom.tflite',
  task: YOLOTask.detect,
);
```

Use explicit `task` only when the export does not include it or when you already know the metadata is missing.

## 🧠 Metadata Resolution

Exported metadata commonly includes:

- `task`
- `names`
- image size and stride
- author/version/export details

The plugin uses that metadata to keep the Dart API simpler:

- `task` can usually be omitted
- class names can come directly from the export
- model switching uses the same metadata-based resolution path

If you want to inspect a model without loading it for inference:

```dart
final info = await YOLO.inspectModel('assets/models/custom.tflite');
print(info['task']);
print(info['names']);
```

## 🏗️ Platform Asset Placement

### Android

You can use either:

- native assets in `android/app/src/main/assets/`
- Flutter assets such as `assets/models/custom.tflite`

Flutter asset models are copied into app storage automatically before loading.

### iOS

You can use either:

- `.mlpackage` or `.mlmodel` files added to `ios/Runner.xcworkspace`
- zipped CoreML packages in Flutter assets, for example `assets/models/custom.mlpackage.zip`

For Flutter assets on iOS, use `.mlpackage.zip` so the package can unpack the model into app storage before loading it.

## 🐍 Exporting Models

Install Ultralytics:

```bash
pip install ultralytics
```

### CoreML Export

Detection models for iOS must be exported with `nms=True`:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="coreml", nms=True, imgsz=640)
```

Other tasks can use the default export behavior:

```python
from ultralytics import YOLO

YOLO("yolo26n-seg.pt").export(format="coreml", imgsz=640)
YOLO("yolo26n-cls.pt").export(format="coreml", imgsz=640)
YOLO("yolo26n-pose.pt").export(format="coreml", imgsz=640)
YOLO("yolo26n-obb.pt").export(format="coreml", imgsz=640)
```

### TFLite Export

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", imgsz=640)
```

Quantized exports also work:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", imgsz=640, int8=True)
YOLO("yolo26n.pt").export(format="tflite", imgsz=640, half=True)
```

## 🔄 Switching Models

`YOLO`, `YOLOView`, and `YOLOViewController.switchModel()` all use the same resolver.

That means switching models supports:

- official model IDs
- asset paths
- local file paths
- remote URLs
- metadata-inferred tasks

Example:

```dart
final controller = YOLOViewController();

await controller.switchModel('yolo26n');
await controller.switchModel('assets/models/custom.tflite', YOLOTask.detect);
```

## ✅ Recommendations

- Start with `YOLO.officialModels()` when you want the simplest path.
- Use official IDs for the default app flow.
- Use custom models when you need a specific export or fine-tuned model.
- Prefer metadata-driven loading over hardcoded task/model tables.
- Pass `task` only when the export does not carry it.
