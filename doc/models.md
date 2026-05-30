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

Use the default official model or a specific official model ID such as `yolo26n`:

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

`YOLO.officialModels()` only returns real downloadable artifacts for the running platform. Official assets are downloaded on first use and cached in app storage, so model URLs stay stable across package releases and the Flutter package does not carry large model files.

Official assets are maintained in GitHub release assets:

| Platform | Format | Release | Direct URL pattern |
| --- | --- | --- | --- |
| Android | TFLite int8 | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) | `https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/<model>.tflite` |
| iOS/macOS | Core ML int8 | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0) | `https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip` |

The Flutter resolver uses the TFLite release for Android and the iOS release for Core ML. The native iOS app uses the same Core ML release through `RemoteModels.swift`.

| Property | Android TFLite official assets | iOS/macOS Core ML official assets |
| --- | --- | --- |
| Model family | YOLO26 `n/s/m/l/x` | YOLO26 `n/s/m/l/x` |
| Tasks | detect, segment, semantic, classify, pose, OBB | detect, segment, semantic, classify, pose, OBB |
| Format | `.tflite` | `.mlpackage.zip` |
| Quantization | int8 TFLite export | int8 Core ML export |
| Export size | classify: `224`; all other tasks: `640` | classify: `224`; OBB: `1024`; all other tasks: `640` |
| End-to-end / NMS export | `nms=False` | `nms=False` |
| Calibration data | `data=coco128.yaml` | Core ML exporter default calibration |
| Postprocessing | Flutter native Android postprocessing | Swift package / iOS app postprocessing |
| Hosted release | `ultralytics/yolo-flutter-app` `v0.3.5` | `ultralytics/yolo-ios-app` `v8.3.0` |

If you want the simplest “start from the default Ultralytics model” entry point, prefer `YOLO.defaultOfficialModel()`.

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
- zipped Core ML packages in Flutter assets, for example `assets/models/custom.mlpackage.zip`

For Flutter assets on iOS, use `.mlpackage.zip` so the package can unpack the model into app storage before loading it.

## 🐍 Official Asset Maintenance

Official release assets are generated from YOLO26 checkpoints with task/size loops so the app, package, and release assets use the same naming scheme.

| Asset family | Authoritative script | Hosted release | Notes |
| --- | --- | --- | --- |
| Android TFLite int8 | [`scripts/export-tflite-models.py`](../scripts/export-tflite-models.py) | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) | Exports `.tflite`, calibrates with `data=coco128.yaml`, optionally verifies one TFLite invocation per model, optionally uploads. |
| iOS/macOS Core ML int8 | `../yolo-ios-app/scripts/export-models.py` | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0) | Exports `.mlpackage`, zips to `.mlpackage.zip`, optionally copies into the iOS app, optionally uploads. |

### Export Android TFLite Assets

Use Linux Python 3.13 for TFLite export. macOS Python 3.13+ is blocked by the `ai-edge-litert` macOS wheel.

```bash
uv venv --python 3.13 .venv
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install -e "../ultralytics" "tensorflow>2.19.0" "onnx>=1.20.0" "onnxslim>=0.1.82" \
  "tf_keras>2.19.0" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" \
  "ai-edge-litert>=1.2.0" "onnxruntime" "protobuf>=6.31.1,<7.0.0" \
  --extra-index-url https://pypi.ngc.nvidia.com --index-strategy unsafe-best-match
uv pip uninstall opencv-python
uv pip install opencv-python-headless
uv pip install --no-deps "onnx2tf>=2.3.0,<2.3.16"
uv run python scripts/export-tflite-models.py --verify
```

Use `--upload --repo ultralytics/yolo-flutter-app --tag v0.3.5` to publish the generated `.tflite` assets. The script exports YOLO26 `n/s/m/l/x` models for detect, segment, semantic, classify, pose, and OBB. Output files are written under `exports/yolo26-tflite/release-assets/` and are ignored by Git.

Android inference runs on LiteRT 2.x with an automatic GPU -> CPU accelerator ladder. int8 assets are the official download artifacts for size, while fp16 non-end-to-end TFLite exports can still be useful for GPU benchmarking on devices whose delegate supports the graph:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, imgsz=640)
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
