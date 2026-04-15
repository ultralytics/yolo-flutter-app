---
title: Quick Start
description: Get YOLO running in your Flutter app in under 2 minutes - minimal setup guide
path: /integrations/flutter/quickstart/
---

# Quick Start Guide

Get YOLO object detection running in your Flutter app in under 2 minutes! ⚡

## 🎯 Goal

By the end of this guide, you'll have a working Flutter app that can detect objects in images using YOLO.

## 📋 Prerequisites

- ✅ Flutter SDK installed
- ✅ Android/iOS device or emulator
- ✅ 5 minutes of your time

## 🚀 Step 1: Create New Flutter App

```bash
flutter create yolo_demo
cd yolo_demo
```

## 📦 Step 2: Add YOLO Plugin

Add to `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  ultralytics_yolo: ^0.2.0
  image_picker: ^1.2.1 # For image selection
```

Install dependencies:

```bash
flutter pub get
```

## 🎯 Step 3: Add a model

The simplest path is to use an official model ID and let the plugin download it automatically. Call `YOLO.officialModels()` to see which IDs are available on the current platform.

```dart
final yolo = YOLO(modelPath: 'yolo26n');
```

Custom local models are still supported:

- iOS: drag `.mlpackage` or `.mlmodel` into **ios/Runner.xcworkspace**, or bundle `.mlpackage.zip` in Flutter assets
- Android native assets: place `.tflite` files in **android/app/src/main/assets/**
- Flutter assets: place `.tflite` on Android or `.mlpackage.zip` on iOS in `assets/models/` and use that asset path directly

If your custom model metadata does not include `task`, pass it explicitly.

## ⚡ Step 4: Minimal Detection Code

Replace `lib/main.dart` with this complete working example:

```dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

void main() => runApp(YOLODemo());

class YOLODemo extends StatefulWidget {
  @override
  _YOLODemoState createState() => _YOLODemoState();
}

class _YOLODemoState extends State<YOLODemo> {
  YOLO? yolo;
  File? selectedImage;
  List<dynamic> results = [];
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    loadYOLO();
  }

  Future<void> loadYOLO() async {
    setState(() => isLoading = true);

    yolo = YOLO(
      modelPath: 'yolo26n',
    );

    await yolo!.loadModel();
    setState(() => isLoading = false);
  }

  Future<void> pickAndDetect() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      setState(() {
        selectedImage = File(image.path);
        isLoading = true;
      });

      final imageBytes = await selectedImage!.readAsBytes();
      final detectionResults = await yolo!.predict(imageBytes);

      setState(() {
        results = detectionResults['boxes'] ?? [];
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('YOLO Quick Demo')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (selectedImage != null)
                Container(
                  height: 300,
                  child: Image.file(selectedImage!),
                ),

              SizedBox(height: 20),

              if (isLoading)
                CircularProgressIndicator()
              else
                Text('Detected ${results.length} objects'),

              SizedBox(height: 20),

              ElevatedButton(
                onPressed: yolo != null ? pickAndDetect : null,
                child: Text('Pick Image & Detect'),
              ),

              SizedBox(height: 20),

              // Show detection results
              Expanded(
                child: ListView.builder(
                  itemCount: results.length,
                  itemBuilder: (context, index) {
                    final detection = results[index];
                    return ListTile(
                      title: Text(detection['class'] ?? 'Unknown'),
                      subtitle: Text(
                        'Confidence: ${(detection['confidence'] * 100).toStringAsFixed(1)}%'
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
```

## 🏃‍♂️ Step 5: Run Your App

```bash
flutter run
```

## 🎉 That's It!

You now have a working YOLO object detection app! The app will:

1. **Load the YOLO model** when it starts
2. **Let you pick an image** from your gallery
3. **Detect objects** in the selected image
4. **Show results** with class names and confidence scores

## 🚀 Next Steps

### Add Real-time Camera

Want real-time detection? Add the YOLOView widget:

```dart
import 'package:ultralytics_yolo/yolo_view.dart';

// Replace the Column with:
YOLOView(
  modelPath: 'yolo26n',
  onResult: (results) {
    print('Detected ${results.length} objects');
  },
)
```

### Dynamic Model Switching

Switch models without restarting the camera:

```dart
final controller = YOLOViewController();

YOLOView(
  modelPath: 'yolo26n',  // Initial model
  controller: controller,
  onResult: (results) {
    print('Detected ${results.length} objects');
  },
)

// Later, switch to another official model or a custom export
await controller.switchModel('assets/models/custom.tflite', YOLOTask.detect);
```

## 🎯 Multi-Instance Quick Example

Want to run multiple models? Try this:

```dart
// Create two YOLO instances
final detector = YOLO(
  modelPath: 'yolo26n',
  useMultiInstance: true, // Enable multi-instance
);

final classifier = YOLO(
  modelPath: 'assets/models/yolo26n-cls.tflite',
  task: YOLOTask.classify,
  useMultiInstance: true,
);

// Load both models
await detector.loadModel();
await classifier.loadModel();

// Run both on the same image
final detections = await detector.predict(imageBytes);
final classifications = await classifier.predict(imageBytes);
```

## 🛠️ Troubleshooting

**App crashes on startup?**

- Make sure the model file exists in the right place

**No detections found?**

- Try a different image with clear objects
- Check model file is not corrupted
- Verify model matches the task type

**Build errors?**

- Run `flutter clean && flutter pub get`
- Check minimum SDK versions in installation guide

## 📚 Learn More

Now that you have YOLO working, explore more features:

- **[📖 Usage Guide](usage.md)** - Advanced patterns and examples
- **[🔧 API Reference](api.md)** - Complete API documentation
- **[🚀 Performance](performance.md)** - Optimization tips
- **[🛠️ Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 💡 Pro Tips

- **Start small**: Use `yolo26n` for development, then move up in size if needed
- **Test on device**: Emulators don't show real performance
- **Monitor memory**: Watch usage when running multiple instances
- **Cache models**: Keep loaded models in memory for better performance

---

**🎉 Congratulations!** You've successfully integrated YOLO into your Flutter app. Ready to build something amazing? 🚀
