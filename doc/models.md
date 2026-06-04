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

| Platform  | Runtime asset                 | Release                                                                                          |
| --------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| Android   | TFLite int8 `.tflite`         | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) |
| iOS       | Core ML int8 `.mlpackage.zip` | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |

URL patterns:

- Android TFLite: `https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/<model>.tflite`
- iOS Core ML: `https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/<model>.mlpackage.zip`

The Flutter resolver uses the TFLite release for Android and the Core ML release for Apple platforms. The native iOS app uses the same Core ML release through `RemoteModels.swift`. These release tags are intentionally pinned in code for reproducible first-use downloads; when official assets move to a new release, update the resolver constants, docs, and URL tests in the same PR.

Official export properties:

| Property       | TFLite                                            | Core ML                             |
| -------------- | ------------------------------------------------- | ----------------------------------- |
| Model IDs      | `yolo26{n,s,m,l,x}`                               | `yolo26{n,s,m,l,x}`                 |
| Tasks          | detect, seg, sem, cls, pose, obb                  | detect, seg, sem, cls, pose, obb    |
| Format         | `.tflite`                                         | `.mlpackage.zip`                    |
| Quantization   | int8 dynamic range TFLite from `int8=True` export | int8 Core ML                        |
| `imgsz`        | `224` cls; `640` others                           | `224` cls; `1024` OBB; `640` others |
| `nms`          | `False`                                           | `False`                             |
| `end2end`      | `False`                                           | `True`                              |
| Calibration    | `ultralytics.cfg.TASK2CALIBRATIONDATA` per task   | exporter default                    |
| Postprocessing | Android native                                    | Swift/Core ML                       |

The TFLite export script passes both `nms=False` and `end2end=False`. `nms=False` excludes an exported NMS operator, while `end2end=False` disables the YOLO26 end-to-end head for the Android LiteRT conversion path. Core ML assets use `end2end=True`, which is the YOLO26 output contract consumed by the Swift decoders. For Android calibration, the script uses the Ultralytics `TASK2CALIBRATIONDATA` mapping by default: detect `coco128.yaml`, segment `coco128-seg.yaml`, classify `imagenet100`, pose `coco8-pose.yaml`, OBB `dota128.yaml`, and semantic `cityscapes8.yaml`.

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

| Runtime      | Source script                                                           | Release                                                                                          |
| ------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| TFLite int8  | [`scripts/export-tflite-models.py`](../scripts/export-tflite-models.py) | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) |
| Core ML int8 | `../yolo-ios-app/scripts/export-models.py`                              | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |

`scripts/export-tflite-models.py` is the source of truth for Android export settings, verification, output names, and optional release upload. The Core ML counterpart in `../yolo-ios-app` owns the Apple asset export settings and packaging.

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

Use `--upload --repo ultralytics/yolo-flutter-app --tag v0.3.5` to publish the generated `.tflite` assets. The script exports YOLO26 `n/s/m/l/x` models for detect, segment, semantic, classify, pose, and OBB. Output files are written under `exports/yolo26-tflite/release-assets/` and are ignored by Git. Leave `--data` unset for official exports so the script uses `ultralytics.cfg.TASK2CALIBRATIONDATA`; pass `--data` only when intentionally benchmarking a single calibration source across tasks.

Android inference runs on LiteRT 2.x with an automatic GPU -> CPU accelerator ladder. int8 assets are the official download artifacts for size, but int8 GPU coverage depends on the device driver and graph; unsupported graphs or ops may fall back to CPU. fp16 non-end-to-end TFLite exports can still be useful for GPU benchmarking on devices whose delegate supports the graph:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
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
