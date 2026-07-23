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

Official assets are maintained as GitHub release assets:

| Platform    | Runtime asset                 | Release                                                                                          |
| ----------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| Android     | LiteRT w8a32 `.tflite`        | [yolo-flutter-app `v0.6.6`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6) |
| Android NPU | QNN `.onnx`                   | [yolo-flutter-app `v0.6.6`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6) |
| iOS         | Core ML int8 `.mlpackage.zip` | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |

URL patterns:

- Android LiteRT: `https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/<model>_w8a32.tflite`
- Android QNN (opt-in NPU): `https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6/<model>_v73_qnn.onnx` (Snapdragon 8 Gen 2+; `_v81` for 8 Elite Gen 5)
- iOS Core ML: `https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip`

The Flutter resolver uses the LiteRT release for Android and the Core ML release for Apple platforms. QNN models are
not auto-resolved by model ID — pass their URL or file path explicitly; any path ending in `_qnn.onnx` runs on the
Hexagon NPU via the ONNX Runtime QNN Execution Provider (see the README's NPU section for the required Gradle opt-in).
QNN assets are nano-only and use channel-last inputs with in-graph ArgMax class maps for semantic segmentation. The
native iOS app uses the same Core ML release through `RemoteModels.swift`. These release tags are intentionally pinned
in code for reproducible first-use downloads; when official assets move to a new release, update the resolver
constants, docs, and URL tests in the same PR.

Official export properties:

| Property       | TFLite                                        | Core ML                                 |
| -------------- | --------------------------------------------- | --------------------------------------- |
| Model IDs      | `yolo26{n,s,m,l,x}`                           | `yolo26{n,s,m,l,x}`                     |
| Tasks          | detect, seg, sem, depth, cls, pose, obb       | detect, seg, sem, depth, cls, pose, obb |
| Format         | `.tflite`                                     | `.mlpackage.zip`                        |
| Quantization   | w8a32 LiteRT (int8 weights, FP32 activations) | int8 Core ML                            |
| `imgsz`        | `224` cls; `640` others                       | `224` cls; `640` others                 |
| `nms`          | `False`                                       | `False`                                 |
| `end2end`      | `False`                                       | `False` cls/sem/depth; `True` others    |
| Calibration    | None (w8a32 dynamic-range)                    | exporter default                        |
| Postprocessing | Android native                                | Swift/Core ML                           |

The TFLite export script passes both `nms=False` and `end2end=False`. `nms=False` excludes an exported NMS operator,
while `end2end=False` disables the YOLO26 end-to-end head for the Android LiteRT conversion path. The shipped Core ML
assets use `end2end=True` for detect, segment, pose, and OBB and `end2end=False` for classification, semantic, and
depth. The Android `w8a32` export is dynamic-range quantization (int8 weights, FP32 activations), so it needs no
calibration data.

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
print(info['labels']);
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

| Runtime      | Existing export path                                                    | Release                                                                                          |
| ------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| LiteRT w8a32 | [`scripts/export-tflite-models.py`](../scripts/export-tflite-models.py) | [yolo-flutter-app `v0.6.6`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6) |
| QNN          | Ultralytics `YOLO.export(format="qnn", imgsz=640, ...)`                 | [yolo-flutter-app `v0.6.6`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6) |
| Core ML int8 | `../yolo-ios-app/scripts/export-models.py`                              | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |

`scripts/export-tflite-models.py` is the source of truth for Android export settings, verification, output names, and optional release upload. The Core ML counterpart in `../yolo-ios-app` owns the Apple asset export settings and packaging.

### Export Android LiteRT Assets

Use Linux x86 or macOS with Python ≥3.10 for LiteRT export.

```bash
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install "ultralytics-opencv-headless[export-litert]>=8.4.83"
uv run python scripts/export-tflite-models.py --verify
```

Use `--upload --repo ultralytics/yolo-flutter-app --tag v0.6.6` to replace the existing `.tflite` assets. The
script exports YOLO26 `n/s/m/l/x` models for every task in its `TASKS` registry, including depth. Output files are
written under `exports/yolo26-tflite/release-assets/` and are ignored by Git. The `w8a32` format (int8 weights, FP32
activations) is dynamic-range quantization, so no calibration data is required. Use Ultralytics QNN export on
Windows x64 or Linux x86-64 to export the matching nano QNN assets for HTP v73 and v81.

Android inference runs on LiteRT 2.x with an automatic GPU -> CPU accelerator ladder. w8a32 assets are the official download artifacts (the smallest GPU-compatible litert format); the GPU delegate compiles the whole graph on supported devices and otherwise falls back to CPU. GPU coverage still depends on the device driver and graph, so confirm delegate placement on your target hardware (the GPU delegate runs the graph in FP16):

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="litert", nms=False, end2end=False, imgsz=640)
# Classification models use imgsz=224.
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
