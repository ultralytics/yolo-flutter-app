<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 YOLO Flutter - Ultralytics Official Plugin

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

Ultralytics YOLO Flutter is the official plugin for running YOLO models in Flutter apps on iOS and Android. It supports [detection](https://docs.ultralytics.com/tasks/detect/), [segmentation](https://docs.ultralytics.com/tasks/segment/), [classification](https://docs.ultralytics.com/tasks/classify/), [pose](https://docs.ultralytics.com/tasks/pose/), and [OBB](https://docs.ultralytics.com/tasks/obb/) with two simple entry points:

- `YOLO` for single-image inference
- `YOLOView` for real-time camera inference

The main goal is simple integration: use an official model ID, or drop in your own exported model and let the plugin resolve task metadata for you.

## ✨ Why This Plugin

| Feature         | Android | iOS |
| --------------- | ------- | --- |
| Detection       | ✅      | ✅  |
| Classification  | ✅      | ✅  |
| Segmentation    | ✅      | ✅  |
| Pose Estimation | ✅      | ✅  |
| OBB Detection   | ✅      | ✅  |

- Official Ultralytics plugin
- One Flutter API for both Android and iOS
- Metadata-first model loading
- Official model download and caching built in
- Real-time camera inference and single-image inference
- Production-ready controls for thresholds, GPU use, and streaming

## ⚡ Quick Start

Install the package:

```yaml
dependencies:
  ultralytics_yolo: ^0.2.0
```

```bash
flutter pub get
```

Start with the default official model:

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

YOLOView(
  modelPath: 'yolo26n',
  onResult: (results) {
    for (final result in results) {
      print('${result.className}: ${result.confidence}');
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

Use an official ID such as `yolo26n` and let the plugin handle download and caching:

```dart
final yolo = YOLO(modelPath: 'yolo26n');
```

Call `YOLO.officialModels()` to see which official IDs are available on the current platform.

### 2. Your own exported model

Pass a local path or Flutter asset path:

```dart
final yolo = YOLO(modelPath: 'assets/models/custom.tflite');
```

If the exported model includes metadata, the plugin infers `task` automatically. If metadata is missing, pass `task` explicitly.

### 3. Remote model URL

Pass an `http` or `https` URL and the plugin will download it into app storage before loading it.

## 📥 Drop Your Own Model Into an App

For custom models, keep the app-side setup minimal.

- Android native assets: place `.tflite` files in `android/app/src/main/assets`
- Flutter assets on Android: place `.tflite` files in `assets/models/`
- iOS bundle: drag `.mlpackage` or `.mlmodel` into `ios/Runner.xcworkspace`
- Flutter assets on iOS: place `.mlpackage.zip` files in `assets/models/`

Then point `modelPath` at that file or asset path.

### iOS export note

Detection models exported to CoreML must use `nms=True`:

```python
from ultralytics import YOLO

YOLO("yolo11n.pt").export(format="coreml", nms=True)
```

Other tasks can use the default export settings.

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

await controller.switchModel('yolo11n');
```

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

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts passionate about open-source collaboration. This [OSI-approved](https://opensource.org/license/agpl-v3) license promotes knowledge sharing and open contribution. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license permits seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. For commercial use cases, please inquire about an [Enterprise License](https://www.ultralytics.com/license).

## 🔗 Related Resources

### Native iOS Development

If you're interested in using YOLO models directly in iOS applications with Swift (without Flutter), check out our dedicated iOS repository:

👉 **[Ultralytics YOLO iOS App](https://github.com/ultralytics/yolo-ios-app)** - A native iOS application demonstrating real-time object detection, segmentation, classification, and pose estimation using Ultralytics YOLO models.

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
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
