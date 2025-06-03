<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 YOLO Flutter - Ultralytics Official Plugin

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml) [![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app) [![pub package](https://img.shields.io/pub/v/ultralytics_yolo.svg)](https://pub.dev/packages/ultralytics_yolo)

_Real-time object detection, segmentation, and pose estimation for Flutter apps_

<!-- ![YOLO Flutter Demo](https://via.placeholder.com/600x300/1e1e1e/ffffff?text=YOLO+Flutter+Demo+GIF+Coming+Soon) -->
<!-- TODO: Add actual demo GIF/video -->

**✨ Why Choose YOLO Flutter?**

| Feature         | Android | iOS |
| --------------- | ------- | --- |
| Detection       | ✅      | ✅  |
| Classification  | ✅      | ✅  |
| Segmentation    | ✅      | ✅  |
| Pose Estimation | ✅      | ✅  |
| OBB Detection   | ✅      | ✅  |

- **Official Ultralytics Plugin** - Direct from YOLO creators
- **Real-time Performance** - Up to 30 FPS on modern devices
- **5 AI Tasks** - Detection, Segmentation, Classification, Pose, OBB
- **Cross-platform** - iOS & Android with single codebase
- **Production Ready** - Performance controls & optimization built-in

## ⚡ Quick Start (2 minutes)

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

// Add this widget and you're detecting objects!
YOLOView(
  modelPath: 'assets/yolo11n.tflite',
  task: YOLOTask.detect,
  onResult: (results) {
    print('Found ${results.length} objects!');
    for (final result in results) {
      print('${result.className}: ${result.confidence}');
    }
  },
)
```

**[▶️ Try the Live Demo](./example)** | **[📖 Full Setup Guide](./docs/install.md)**

## 🎯 What You Can Build

| Task                | Description                    | Use Cases                     | Performance |
| ------------------- | ------------------------------ | ----------------------------- | ----------- |
| **Detection**       | Find objects & their locations | Security, Inventory, Shopping | 25-30 FPS   |
| **Segmentation**    | Pixel-perfect object masks     | Photo editing,                | 15-25 FPS   |
| **Classification**  | Identify image categories      | Content moderation, Tagging   | 30+ FPS     |
| **Pose Estimation** | Human pose & keypoints         | Fitness apps, Motion capture  | 20-30 FPS   |
| **OBB Detection**   | Rotated bounding boxes         | Aerial imagery                | 20-25 FPS   |

**[📱 See Examples →](./docs/usage.md)** | **[⚡ Performance Guide →](./docs/performance.md)**

## 🚀 Installation

### 1. Add to pubspec.yaml

```yaml
dependencies:
  ultralytics_yolo: ^0.1.5
```

### 2. Install dependencies

```bash
flutter pub get
```

### 3. Add models

You can get the model in one of the following ways:

1. Download from the [release assets](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.0.0) of this repository

2. Get it from [Ultralytics Hub](https://www.ultralytics.com/hub)

3. Export it from [Ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) ([CoreML](https://docs.ultralytics.com/ja/integrations/coreml/)/[TFLite](https://docs.ultralytics.com/integrations/tflite/))

**[📥 Download Models](./docs/install.md#models)** | 

### 4. Platform-Specific Setup

**[🔧 Setup Guide](./docs/install.md)**

## 🏆 Trusted by Developers

- ✅ **Official Ultralytics Plugin** - Maintained by YOLO creators
- ✅ **Production Tested** - Used in apps with many users
- ✅ **Active Development** - Regular updates & feature additions
- ✅ **Community Driven** - Open source with responsive support

**Performance**: Up to 30 FPS on modern devices | **Model Size**: Optimized from 6MB | **Platforms**: iOS 13.0+ & Android API 21+

## 📚 Documentation

| Guide                                                 | Description                       | For             |
| ----------------------------------------------------- | --------------------------------- | --------------- |
| **[Installation Guide](./docs/install.md)**           | Installation, setup, requirements | New users       |
| **[Quick Start](./docs/quickstart.md)**               | 2-minute setup guide              | New users       |
| **[Usage Guide](./docs/usage.md)**                    | Common use cases & code samples   | All users       |
| **[Performance Optimization](./docs/performance.md)** | Inference control & tuning        | Production apps |
| **[API Reference](./docs/api.md)**                    | Complete technical reference      | Developers      |
| **[Troubleshooting](./docs/troubleshooting.md)**      | Common issues & solutions         | All users       |

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

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ultralytics/yolo-flutter-app&type=Date)](https://star-history.com/#ultralytics/yolo-flutter-app&Date)

---

<div align="center">
  <p>Made with ❤️ by <a href="https://www.ultralytics.com/">Ultralytics</a></p>
  <p>
    <a href="https://github.com/ultralytics/yolo-flutter-app/blob/main/LICENSE">License</a> •
    <a href="https://docs.ultralytics.com/">YOLO Docs</a> •
    <a href="https://www.ultralytics.com/">Ultralytics</a>
  </p>
</div>
