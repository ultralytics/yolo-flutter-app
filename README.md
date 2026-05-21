<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 YOLO Flutter - Ultralytics Official Plugin

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

Ultralytics YOLO Flutter is the official plugin for running YOLO models in Flutter apps on iOS and Android. It supports [detection](https://docs.ultralytics.com/tasks/detect/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [semantic segmentation](https://docs.ultralytics.com/tasks/semantic/), [classification](https://docs.ultralytics.com/tasks/classify/), [pose](https://docs.ultralytics.com/tasks/pose/), and [OBB](https://docs.ultralytics.com/tasks/obb/) with two simple entry points:

- `YOLO` for single-image inference
- `YOLOView` for real-time camera inference

The main goal is simple integration: use an official model ID, or drop in your own exported model and let the plugin resolve task metadata for you.

<div align="center">
  <br>
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" target="_blank"><img width="100%" src="https://github.com/user-attachments/assets/d5dab2e7-f473-47ce-bc63-69bef89ba52a" alt="Ultralytics YOLO iOS App previews"></a>
  <br>
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
</div>

## ✨ Why This Plugin

- Official Ultralytics plugin for Flutter
- One Dart API for Android and iOS
- Metadata-first model loading with official model download and caching
- Real-time camera inference and single-image inference
- Production-ready controls for thresholds, GPU use, and streaming
- YOLO26 and YOLO11 model families supported

| Feature                               | Android | iOS | Details                                                 |
| ------------------------------------- | ------- | --- | ------------------------------------------------------- |
| Object Detection                      | ✅      | ✅  | Bounding boxes, labels, and confidence scores           |
| Instance Segmentation                 | ✅      | ✅  | Instance masks with boxes and classes                   |
| Semantic Segmentation                 | ✅      | ✅  | Dense class masks for every pixel                       |
| Image Classification                  | ✅      | ✅  | Top class predictions and scores                        |
| Pose Estimation                       | ✅      | ✅  | Keypoints with boxes and confidence scores              |
| Oriented Bounding Box (OBB) Detection | ✅      | ✅  | Rotated boxes and polygon corners                       |
| Real-Time Camera Inference            | ✅      | ✅  | `YOLOView` for live camera workflows                    |
| Single-Image Inference                | ✅      | ✅  | `YOLO` for image bytes                                  |
| Official Models                       | ✅      | ✅  | Discovery, download, and caching for packaged model IDs |
| Custom Models                         | ✅      | ✅  | TFLite on Android, Core ML on iOS, metadata-first tasks |

## ⚡ Quick Start

Install the package:

Package: https://pub.dev/packages/ultralytics_yolo

```yaml
dependencies:
  ultralytics_yolo: ^0.3.5
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

Use the default official model or a specific official ID and let the plugin
handle download and caching:

```dart
final yolo = YOLO(modelPath: YOLO.defaultOfficialModel() ?? 'yolo26n');
```

Call `YOLO.officialModels()` to see which official IDs are available on the
current platform. Official assets are downloaded from the canonical `v0.2.0`
Flutter release on Android and the canonical YOLO iOS `v8.3.0` Core ML release
on iOS, so package releases do not move model URLs.

Example assets come from the same canonical locations:

- Android TFLite: [yolo-flutter-app `v0.2.0`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.2.0)
- iOS Core ML: [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)

### 2. Your own exported model

Pass your own exported YOLO model as a local path or Flutter asset path:

```dart
final yolo = YOLO(modelPath: 'assets/models/my-finetuned-model.tflite');
```

If the exported model includes metadata, the plugin infers `task` automatically. If metadata is missing, pass `task` explicitly.

```dart
final yolo = YOLO(
  modelPath: 'assets/models/my-finetuned-model.tflite',
  task: YOLOTask.detect,
);
```

### 3. Remote model URL

Pass an `http` or `https` URL and the plugin will download it into app storage before loading it.

## 🧭 Official vs. Custom

| Use case                                              | Recommended path                  |
| ----------------------------------------------------- | --------------------------------- |
| Fastest first integration                             | Official model ID like `yolo26n`  |
| You trained or exported your own model                | Custom asset or local file        |
| You ship different models per customer or environment | Remote URL                        |
| You need the plugin to infer `task` automatically     | Any export with metadata          |
| You have an older or stripped export without metadata | Custom model plus explicit `task` |

For official models, start with `YOLO.defaultOfficialModel()` or
`YOLO.officialModels()`. For custom models, start with the exported file you
actually plan to ship.

## 📥 Drop Your Own Model Into an App

For custom models, keep the app-side setup minimal.

- Android native assets: place `.tflite` files in `android/app/src/main/assets`
- Flutter assets on Android: place `.tflite` files in `assets/models/`
- iOS bundle: drag `.mlpackage` or `.mlmodel` into `ios/Runner.xcworkspace`
- Flutter assets on iOS: place `.mlpackage.zip` files in `assets/models/`

Then point `modelPath` at that file or asset path.

### iOS export note

Detection models exported to Core ML must use `nms=True`:

```python
from ultralytics import YOLO

# Square [640, 640] works best when one model must run in both portrait and landscape.
# Ultralytics imgsz order is [height, width]; use [640, 384] for portrait-only or [384, 640] for landscape-only.
YOLO("yolo26n.pt").export(format="coreml", nms=True, imgsz=[640, 640])
```

Other tasks can use the default export settings, with the same square-orientation guidance for `imgsz`.

## 🎯 Choose The Right API

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

## 🧩 Recommended Patterns

| App type                            | Model loading pattern                                                  |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Live camera app                     | `YOLOView(modelPath: 'yolo26n')`                                       |
| Photo picker or gallery workflow    | `YOLO(modelPath: 'yolo26n')`                                           |
| App with your own bundled model     | `YOLO(modelPath: 'assets/models/custom.tflite')`                       |
| Cross-platform Core ML + TFLite app | Use platform-appropriate exported assets and let metadata drive `task` |
| App that changes models at runtime  | `YOLOViewController.switchModel(...)`                                  |

## 📚 Documentation

| Guide                                         | Description                                 |
| --------------------------------------------- | ------------------------------------------- |
| **[Installation Guide](doc/install.md)**      | Requirements and platform setup             |
| **[Quick Start](doc/quickstart.md)**          | Minimal setup for the first working app     |
| **[Model Guide](doc/models.md)**              | Official models, custom models, export flow |
| **[Usage Guide](doc/usage.md)**               | Common app patterns and examples            |
| **[API Reference](doc/api.md)**               | Full API surface                            |
| **[Performance Guide](doc/performance.md)**   | Tuning and performance controls             |
| **[Troubleshooting](doc/troubleshooting.md)** | Common problems and fixes                   |

## 🤝 Community & Support

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics) [![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/) [![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

- **💬 Questions?** [Discord](https://discord.com/invite/ultralytics) | [Forums](https://community.ultralytics.com/) | [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues)
- **🐛 Found a bug?** [Report it here](https://github.com/ultralytics/yolo-flutter-app/issues/new)
- **💡 Feature request?** [Let us know](https://github.com/ultralytics/yolo-flutter-app/discussions)

## 💡 Contribute

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's bug fixes, feature enhancements, or documentation improvements, your involvement is crucial. Please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for detailed insights on how to participate. We also encourage you to share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 goes out to all our contributors!

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

## 📮 Contact

Encountering issues or have feature requests related to Ultralytics YOLO? Please report them via [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues). For broader discussions, questions, and community support, join our [Discord](https://discord.com/invite/ultralytics) server!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
