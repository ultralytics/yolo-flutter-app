<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 YOLO Flutter - Ultralytics Official Plugin

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/ultralytics/yolo-flutter-app/branch/main/graph/badge.svg)](https://app.codecov.io/github/ultralytics/yolo-flutter-app)
[![CocoaPods](https://img.shields.io/cocoapods/v/UltralyticsYOLO?logo=cocoapods&logoColor=white&label=CocoaPods)](https://cocoapods.org/pods/UltralyticsYOLO)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

Ultralytics YOLO Flutter is the official plugin for running YOLO models in Flutter apps on iOS and Android. It supports [detection](https://docs.ultralytics.com/tasks/detect), [instance segmentation](https://docs.ultralytics.com/tasks/segment), [semantic segmentation](https://docs.ultralytics.com/tasks/semantic), [classification](https://docs.ultralytics.com/tasks/classify), [pose](https://docs.ultralytics.com/tasks/pose), and [OBB](https://docs.ultralytics.com/tasks/obb) with two simple entry points:

- `YOLO` for single-image inference
- `YOLOView` for real-time camera inference

The main goal is simple integration: use an official model ID, or drop in your own exported model and let the plugin resolve task metadata for you.

<div align="center">
  <br>
  <a href="https://apps.apple.com/us/app/ultralytics-yolo/id1452689527" target="_blank"><img width="100%" src="https://github.com/user-attachments/assets/d5dab2e7-f473-47ce-bc63-69bef89ba52a" alt="Ultralytics YOLO iOS App previews"></a>
  <br>
  <br>
  <a href="https://apps.apple.com/us/app/ultralytics-yolo/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
  &nbsp;&nbsp;
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.yolo" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/google-play.svg" width="15%" alt="Get it on Google Play"></a>
</div>

## ✨ Features

- Official Ultralytics plugin for Flutter
- One Dart API for Android and iOS
- Metadata-first model loading with official model download and caching
- Real-time camera inference and single-image inference
- Controls for thresholds, accelerator selection, and result streaming
- YOLO26 and YOLO11 model families supported

| Feature                               | Android | iOS | Details                                                            |
| ------------------------------------- | ------- | --- | ------------------------------------------------------------------ |
| Object Detection                      | ✅      | ✅  | Bounding boxes, labels, and confidence scores                      |
| Instance Segmentation                 | ✅      | ✅  | Instance masks with boxes and classes                              |
| Semantic Segmentation                 | ✅      | ✅  | Dense class masks for every pixel                                  |
| Image Classification                  | ✅      | ✅  | Top class predictions and scores                                   |
| Pose Estimation                       | ✅      | ✅  | Keypoints with boxes and confidence scores                         |
| Oriented Bounding Box (OBB) Detection | ✅      | ✅  | Rotated boxes and polygon corners                                  |
| Real-Time Camera Inference            | ✅      | ✅  | `YOLOView` for live camera workflows                               |
| Single-Image Inference                | ✅      | ✅  | `YOLO` for image bytes                                             |
| Official Models                       | ✅      | ✅  | Discovery, download, and caching for packaged model IDs            |
| Custom Models                         | ✅      | ✅  | LiteRT (TFLite) on Android, Core ML on iOS, metadata-first tasks   |
| Qualcomm NPU (QNN)                    | ✅      | —   | Opt-in Hexagon NPU inference for `*_qnn.onnx` models on Snapdragon |

## ⚡ Quick Start

Install the package:

Package: https://pub.dev/packages/ultralytics_yolo

```yaml
dependencies:
  ultralytics_yolo: ^0.6.2
```

```bash
flutter pub get
```

Start with the default official model:

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

final modelId = YOLO.defaultOfficialModel() ?? 'yolo26n';

YOLOView(
  modelPath: modelId,
  onResult: (results) {
    for (final r in results) {
      debugPrint('${r.className}: ${r.confidence}');
    }
  },
)
```

For single-image inference:

```dart
final yolo = YOLO(modelPath: 'yolo26n');
await yolo.loadModel();
final results = await yolo.predict(imageBytes);
```

**[▶️ Example App](./example)** | **[📖 Installation Guide](doc/install.md)** | **[⚡ Quick Start Guide](doc/quickstart.md)**

## 📦 Model Loading

The plugin supports three model flows.

### 1. Official model IDs

Use the default official model or a specific official ID and let the plugin handle download and caching:

```dart
final yolo = YOLO(modelPath: YOLO.defaultOfficialModel() ?? 'yolo26n');
```

Call `YOLO.officialModels()` to see which official IDs are available on the current platform. Official assets are downloaded on first use and cached in app storage, so the app package does not carry large model files.

Official assets are maintained as GitHub release assets:

| Platform             | Runtime asset                 | Release                                                                                          |
| -------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| Android              | TFLite int8 `.tflite`         | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) |
| Android NPU (opt-in) | QNN `*_v73/_v81_qnn.onnx`     | [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5) |
| iOS                  | Core ML int8 `.mlpackage.zip` | [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)         |

The Flutter resolver uses the TFLite release for Android and the Core ML release for Apple platforms. These release tags are intentionally pinned for reproducible first-use downloads. See the [model guide](doc/models.md) for the official export matrix, URL patterns, and model properties.

### 2. Your own exported model

Pass your own exported YOLO model as a local path or Flutter asset path:

```dart
final yolo = YOLO(modelPath: 'assets/models/my-finetuned-model.tflite');
```

If the exported model includes embedded metadata, the plugin infers `task` and class labels automatically — it reads Ultralytics' appended-ZIP metadata, with a standard TFLite (FlatBuffers) metadata fallback — so drag-and-drop custom models auto-detect. If metadata is missing, pass `task` explicitly.

```dart
final yolo = YOLO(
  modelPath: 'assets/models/my-finetuned-model.tflite',
  task: YOLOTask.detect,
);
```

### 3. Remote model URL

Pass an `http` or `https` URL and the plugin will download it into app storage before loading it.

### 4. Qualcomm NPU models (Android, opt-in)

Android ships with LiteRT (TFLite) and that remains the default — nothing changes for existing apps, and the
QNN support adds zero bytes to your build. Any model path ending in `_qnn.onnx` (a Qualcomm QNN context binary
exported with `yolo export format=qnn`) is routed to the Hexagon NPU through the ONNX Runtime QNN Execution
Provider instead.

Running QNN models requires a Snapdragon device with a Hexagon HTP (Snapdragon 8 Gen 2 or newer for the official
`_v73` assets; `_v81` targets Snapdragon 8 Elite Gen 5) and three additions to your app's
`android/app/build.gradle`:

```groovy
android {
    packagingOptions {
        jniLibs {
            useLegacyPackaging = true // the Hexagon DSP loader needs real .so files, not APK-mmapped ones
        }
    }
}

dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android-qnn:1.26.0'
    implementation 'com.qualcomm.qti:qnn-runtime:2.46.0' // newer than the AAR's bundled QAIRT; required for the latest Snapdragons
}
```

```dart
final yolo = YOLO(
  modelPath: 'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/yolo26n_v73_qnn.onnx',
  task: YOLOTask.detect,
);
```

Without the Gradle opt-in, loading a `_qnn.onnx` model fails with a clear error and TFLite models are unaffected. The bundled example app follows the same opt-in — it ships without the QNN runtime to keep its download small, so build it with `ENABLE_QNN=1` (e.g. `ENABLE_QNN=1 flutter run --release`) to test the NPU path on a device.
See the [performance guide](doc/performance.md) for measured CPU/GPU/NPU numbers and tuning notes.

## 🧭 Official vs. Custom

| Use case                                              | Recommended path                  |
| ----------------------------------------------------- | --------------------------------- |
| Fastest first integration                             | Official model ID like `yolo26n`  |
| You trained or exported your own model                | Custom asset or local file        |
| You ship different models per customer or environment | Remote URL                        |
| You need the plugin to infer `task` automatically     | Any export with metadata          |
| You have an older or stripped export without metadata | Custom model plus explicit `task` |

For official models, start with `YOLO.defaultOfficialModel()` or `YOLO.officialModels()`. For custom models, start with the exported file you actually plan to ship.

## 📥 Using Your Own Model

For custom models, keep the app-side setup minimal.

- Android native assets: place `.tflite` files in `android/app/src/main/assets`
- Flutter assets on Android: place `.tflite` files in `assets/models/`
- iOS bundle: drag `.mlpackage` or `.mlmodel` into `ios/Runner.xcworkspace`
- Flutter assets on iOS: place `.mlpackage.zip` files in `assets/models/`

Then point `modelPath` at that file or asset path.

### Official asset maintenance

The Android TFLite release assets are generated by [`scripts/export-tflite-models.py`](scripts/export-tflite-models.py). The script defines the official YOLO26 task/size matrix, int8 export settings, Ultralytics task-specific calibration data, optional one-shot TFLite inference verification, and optional GitHub release upload. By default it reads `ultralytics.cfg.TASK2CALIBRATIONDATA` so each task uses the same canonical calibration dataset as the Ultralytics exporter.

Run it in a Linux Python 3.13 environment:

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

Use `--upload --repo ultralytics/yolo-flutter-app --tag v0.3.5` to publish generated `.tflite` assets to the canonical Android release. The matching Core ML assets are generated by `../yolo-ios-app/scripts/export-models.py` and hosted on the iOS `v8.3.0` release.

Android inference runs on [LiteRT](https://developers.google.com/edge/litert) 2.x through an automatic GPU -> CPU accelerator ladder. int8 assets are the official download artifacts for size, but int8 GPU coverage depends on the device driver and graph; graphs the GPU cannot compile fall back to CPU. fp16 non-end-to-end TFLite exports can still be useful for GPU benchmarking on devices whose delegate supports the graph:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
```

## 🎯 Choosing an API

Use `YOLO` when you already have image bytes and want one prediction at a time:

```dart
final yolo = YOLO(modelPath: 'yolo26n');
await yolo.loadModel();
final results = await yolo.predict(imageBytes);
```

Use `YOLOView` when you want live camera inference:

```dart
final controller = YOLOViewController();

YOLOView(
  modelPath: 'yolo26n',
  controller: controller,
  onResult: (results) {},
)

await controller.switchModel('assets/models/custom.tflite', YOLOTask.detect);
```

Use `YOLOShowcase` when you want the complete Ultralytics camera UI:

```dart
YOLOShowcase(
  initialTask: YOLOTask.detect,
  initialModelSize: 'n',
  onCapture: (bytes) {},
)
```

## 🔄 Migrating From 0.3.x UI APIs

Version 0.4.0 removes the old Dart-side overlay/control layer. Camera detections are rendered natively by `YOLOView`; Flutter now owns only the surrounding app controls.

| Removed 0.3.x API                                | 0.4.0 replacement                                                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `YOLOOverlay`, `YOLOOverlayTheme`                | Remove these widgets. Use native `YOLOView` overlays, or consume `onResult`/`YOLO.predict()` data.                        |
| `YOLOControls`                                   | Use `YOLOShowcase` for the full UI, or compose the exported Material widgets directly.                                    |
| `YOLOView.showNativeUI`                          | Use `YOLOShowcase` for built-in controls; use bare `YOLOView` when building your own UI.                                  |
| `YOLOView.showOverlays`, `YOLOView.overlayTheme` | No constructor replacement. Camera overlay drawing is native and not themed from Dart.                                    |
| `YOLOViewController.setShowUIControls()`         | Show/hide your own Flutter controls around `YOLOView`.                                                                    |
| `YOLOViewController.setShowOverlays()`           | Still available: toggles native overlay rendering. `capturePhoto(withOverlays: false)` only affects captured JPEG output. |

## 🧩 Recommended Patterns

| App type                            | Model loading pattern                                                  |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Live camera app                     | `YOLOView(modelPath: 'yolo26n')`                                       |
| Photo picker or gallery workflow    | `YOLO(modelPath: 'yolo26n')`                                           |
| App with your own bundled model     | `YOLO(modelPath: 'assets/models/custom.tflite')`                       |
| Cross-platform Core ML + TFLite app | Use platform-appropriate exported assets and let metadata drive `task` |
| App that changes models at runtime  | `YOLOViewController.switchModel(...)`                                  |

## 📚 Documentation

| Guide                                         | Description                                    |
| --------------------------------------------- | ---------------------------------------------- |
| **[Installation Guide](doc/install.md)**      | Requirements and platform setup                |
| **[Quick Start](doc/quickstart.md)**          | Minimal setup for the first working app        |
| **[Model Guide](doc/models.md)**              | Official models, custom models, export flow    |
| **[Usage Guide](doc/usage.md)**               | Common app patterns and examples               |
| **[API Reference](doc/api.md)**               | Full API surface                               |
| **[Performance Guide](doc/performance.md)**   | Tuning controls and on-device benchmark record |
| **[Troubleshooting](doc/troubleshooting.md)** | Common problems and fixes                      |

## 🤝 Community & Support

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics) [![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/) [![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

- **💬 Questions?** [Discord](https://discord.com/invite/ultralytics) | [Forums](https://community.ultralytics.com/) | [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues)
- **🐛 Found a bug?** [Report it here](https://github.com/ultralytics/yolo-flutter-app/issues/new)
- **💡 Feature request?** [Let us know](https://github.com/ultralytics/yolo-flutter-app/issues/new)

## 💡 Contribute

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's bug fixes, feature enhancements, or documentation improvements, your involvement is crucial. Please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for detailed insights on how to participate. We also encourage you to share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 goes out to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics offers two licensing options to accommodate diverse needs:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts passionate about open-source collaboration. This [OSI-approved](https://opensource.org/license/agpl-3.0) license promotes knowledge sharing and open contribution. See the [LICENSE](https://github.com/ultralytics/yolo-flutter-app/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license permits seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. For commercial use cases, please inquire about an [Enterprise License](https://www.ultralytics.com/license).

## 🔗 Related Resources

### Native iOS Development

If you're interested in using YOLO models directly in iOS applications with Swift (without Flutter), check out our dedicated iOS repository:

👉 **[Ultralytics YOLO iOS App](https://github.com/ultralytics/yolo-ios-app)** - A native iOS application for real-time detection, instance segmentation, semantic segmentation, classification, pose estimation, and OBB detection with Ultralytics YOLO models.

This repository provides:

- Pure Swift implementation for iOS
- Direct Core ML integration
- Native iOS UI components
- Example code for various YOLO tasks
- Optimized for iOS performance

> [!NOTE]
> On iOS this plugin is built on the shared [`UltralyticsYOLO` Swift package](https://github.com/ultralytics/yolo-ios-app) (`import UltralyticsYOLO`) — the same inference core used by the native iOS app — so both stay in sync from a single source of truth. The plugin's iOS sources (`ios/ultralytics_yolo/`) hold only the Flutter bridge and the camera/view components, and ship for both Swift Package Manager and CocoaPods.

## 📮 Contact

Encountering issues or have feature requests related to Ultralytics YOLO? Please report them via [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues). For broader discussions, questions, and community support, join our [Discord](https://discord.com/invite/ultralytics) server!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://x.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://space.bilibili.com/3546646073837954"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
