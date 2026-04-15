---
title: Flutter YOLO Plugin
description: Official Ultralytics YOLO plugin for Flutter - Real-time object detection, segmentation, and pose estimation
path: /integrations/flutter/
---

# Ultralytics YOLO Flutter Plugin

Welcome to the official **Ultralytics YOLO Flutter Plugin** - the most comprehensive computer vision solution for Flutter applications. This plugin brings state-of-the-art AI capabilities directly to your mobile apps with real-time performance.

## 🚀 Key Features

| Feature                     | Android | iOS | Performance |
| --------------------------- | ------- | --- | ----------- |
| **Object Detection**        | ✅      | ✅  | 25-30 FPS   |
| **Instance Segmentation**   | ✅      | ✅  | 15-25 FPS   |
| **Image Classification**    | ✅      | ✅  | 30+ FPS     |
| **Pose Estimation**         | ✅      | ✅  | 20-30 FPS   |
| **Oriented Bounding Boxes** | ✅      | ✅  | 20-25 FPS   |
| **Multi-Instance Support**  | ✅      | ✅  | Variable    |

## 🎯 Why Choose YOLO Flutter?

- **Official Plugin**: Direct from the YOLO creators at Ultralytics
- **Real-time Performance**: Optimized for mobile devices with up to 30 FPS
- **Production Ready**: Built-in performance controls and memory management
- **Cross-platform**: Single codebase for iOS and Android
- **Multiple AI Tasks**: 5 different computer vision capabilities
- **Multi-Instance**: Run multiple models simultaneously (New!)

## 🎨 Supported Tasks

### 🔍 Object Detection

Detect and locate objects in images with bounding boxes.

```dart
final yolo = YOLO(modelPath: 'yolo26n');
```

### 🎭 Instance Segmentation

Get pixel-perfect masks for each detected object.

```dart
final yolo = YOLO(modelPath: 'yolo26n-seg');
```

### 🏷️ Classification

Classify entire images into categories.

```dart
final yolo = YOLO(modelPath: 'yolo26n-cls');
```

### 🤸 Pose Estimation

Detect human poses and keypoints.

```dart
final yolo = YOLO(modelPath: 'yolo26n-pose');
```

### 📦 Oriented Bounding Box (OBB)

Detect objects with rotated bounding boxes.

```dart
final yolo = YOLO(modelPath: 'yolo26n-obb');
```

## 📚 Documentation Navigation

Explore our comprehensive documentation:

- **[📦 Installation](install.md)** - Add the plugin to your Flutter project
- **[⚡ Quick Start](quickstart.md)** - Get running in 2 minutes
- **[📖 Usage Guide](usage.md)** - Comprehensive examples and patterns
- **[🔧 API Reference](api.md)** - Complete API documentation
- **[🚀 Performance](performance.md)** - Optimization tips and benchmarks
- **[🛠️ Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🏗️ Architecture Overview

The YOLO Flutter Plugin uses a hybrid architecture:

```
Flutter App Layer
    ↓
Method Channel Bridge
    ↓
Native Platform Layer (iOS/Android)
    ↓
YOLO Model Inference Engine
```

## 🎯 Use Cases

### 📱 Mobile Applications

- **Security Apps**: Real-time surveillance and monitoring
- **Retail Apps**: Product recognition and inventory management
- **Health Apps**: Pose analysis for fitness and rehabilitation

### 🏢 Enterprise Solutions

- **Quality Control**: Automated defect detection in manufacturing
- **Agriculture**: Crop monitoring and pest detection
- **Construction**: Safety equipment compliance monitoring

## 🔮 Roadmap

- ✅ Multi-instance support (v0.1.18+)
- ✅ Performance optimization (v0.1.15+)
- ✅ Streaming configuration (v0.1.18+)

## 🤝 Community & Support

- **💬 Questions?** [Discord](https://discord.com/invite/ultralytics) | [Community Forums](https://community.ultralytics.com/)
- **🐛 Found a bug?** [Report it here](https://github.com/ultralytics/yolo-flutter-app/issues/new)
- **💡 Feature request?** [Let us know](https://github.com/ultralytics/yolo-flutter-app/discussions)

## ⚡ Ready to Start?

Jump straight into our [Quick Start Guide](quickstart.md) and have YOLO running in your Flutter app within minutes!

---

<div align="center">
<strong>Made with ❤️ by <a href="https://www.ultralytics.com/">Ultralytics</a></strong>
</div>
