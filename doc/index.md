---
title: Flutter YOLO Plugin
description: Official Ultralytics YOLO plugin for Flutter - metadata-first model loading for single-image and camera inference
path: /integrations/flutter/
---

# Ultralytics YOLO Flutter Plugin

Ultralytics YOLO Flutter is the official plugin for running YOLO models in Flutter apps on iOS and Android.

It gives you two focused entry points:

- `YOLO` for single-image inference
- `YOLOView` for real-time camera inference

The plugin is built around one model-loading flow:

- use an official model ID such as `yolo26n`
- or point to your own exported model
- let the plugin resolve metadata and download/cache official assets when needed

## 🚀 What It Supports

| Task                    | Android | iOS |
| ----------------------- | ------- | --- |
| Object Detection        | ✅      | ✅  |
| Instance Segmentation   | ✅      | ✅  |
| Image Classification    | ✅      | ✅  |
| Pose Estimation         | ✅      | ✅  |
| Oriented Bounding Boxes | ✅      | ✅  |

## 🎯 Default Flow

```dart
final yolo = YOLO(modelPath: 'yolo26n');
await yolo.loadModel();
final results = await yolo.predict(imageBytes);
```

If you want live camera inference:

```dart
YOLOView(
  modelPath: 'yolo26n',
  onResult: (results) {},
)
```

For official assets, call `YOLO.officialModels()` to see which IDs are available on the current platform.

## 📚 Documentation

- [📦 Installation](install.md)
- [⚡ Quick Start](quickstart.md)
- [📖 Usage Guide](usage.md)
- [🧠 Model Guide](models.md)
- [🔧 API Reference](api.md)
- [🚀 Performance Guide](performance.md)
- [🛠️ Troubleshooting](troubleshooting.md)

## ✅ Design Principles

- Official models are discoverable and downloadable from the package, not the example app.
- Custom models stay first-class.
- `task` is optional when exported metadata already includes it.
- `YOLO` and `YOLOView` stay separate APIs, but they share the same model resolver.
