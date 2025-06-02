<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 YOLO Flutter - Ultralytics Official Plugin

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml) [![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app) [![pub package](https://img.shields.io/pub/v/ultralytics_yolo.svg)](https://pub.dev/packages/ultralytics_yolo)

*Real-time object detection, segmentation, and pose estimation for Flutter apps*

<!-- ![YOLO Flutter Demo](https://via.placeholder.com/600x300/1e1e1e/ffffff?text=YOLO+Flutter+Demo+GIF+Coming+Soon) -->
<!-- TODO: Add actual demo GIF/video -->

**✨ Why Choose YOLO Flutter?**
- 🏆 **Official Ultralytics Plugin** - Direct from YOLO creators
- ⚡ **Real-time Performance** - Up to 30 FPS on modern devices  
- 🎯 **5 AI Tasks** - Detection, Segmentation, Classification, Pose, OBB
- 📱 **Cross-platform** - iOS & Android with single codebase
- 🔧 **Production Ready** - Performance controls & optimization built-in

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

**[▶️ Try the Live Demo](./example)** | **[📖 Full Setup Guide](./docs/getting-started.md)**

## 🎯 What You Can Build

| Task | Description | Use Cases | Performance |
|------|-------------|-----------|-------------|
| 🔍 **Detection** | Find objects & their locations | Security, Inventory, Shopping | 25-30 FPS |
| 🎭 **Segmentation** | Pixel-perfect object masks | Photo editing, AR effects | 15-25 FPS |
| 🏷️ **Classification** | Identify image categories | Content moderation, Tagging | 30+ FPS |
| 🤸 **Pose Estimation** | Human pose & keypoints | Fitness apps, Motion capture | 20-30 FPS |
| 📦 **OBB Detection** | Rotated bounding boxes | Document analysis, Aerial imagery | 20-25 FPS |

**[📱 See Examples →](./docs/examples.md)** | **[⚡ Performance Guide →](./docs/performance.md)** | **[🎮 Streaming Demo →](./streaming_test_example)**

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

### 3. Add models & permissions
```dart
// Download models and add to assets/
// Configure camera permissions
```

**[📥 Download Models](./docs/getting-started.md#models)** | **[🔧 Setup Guide](./docs/getting-started.md)**

## 🏆 Trusted by Developers

- ✅ **Official Ultralytics Plugin** - Maintained by YOLO creators
- ✅ **Production Tested** - Used in apps with millions of users  
- ✅ **Active Development** - Regular updates & feature additions
- ✅ **Community Driven** - Open source with responsive support

**Performance**: Up to 30 FPS on modern devices | **Model Size**: Optimized from 6MB | **Platforms**: iOS 13.0+ & Android API 21+

## 📚 Documentation

| Guide | Description | For |
|-------|-------------|-----|
| **[Getting Started](./docs/getting-started.md)** | Installation, setup, first app | New users |
| **[Examples](./docs/examples.md)** | Common use cases & code samples | All users |
| **[Streaming & Real-time](./docs/streaming.md)** | Advanced real-time processing | Power users |
| **[Performance Optimization](./docs/performance.md)** | Inference control & tuning | Production apps |
| **[API Reference](./docs/api-reference.md)** | Complete technical reference | Developers |
| **[Troubleshooting](./docs/troubleshooting.md)** | Common issues & solutions | All users |

## 🤝 Community & Support

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics) [![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/) [![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

- **💬 Questions?** [Discord](https://discord.com/invite/ultralytics) | [Forums](https://community.ultralytics.com/) | [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues)
- **🐛 Found a bug?** [Report it here](https://github.com/ultralytics/yolo-flutter-app/issues/new)
- **💡 Feature request?** [Let us know](https://github.com/ultralytics/yolo-flutter-app/discussions)

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