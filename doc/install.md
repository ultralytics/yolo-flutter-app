---
title: Installation
description: How to add Ultralytics YOLO to your Flutter project - complete setup guide for iOS and Android
path: /integrations/flutter/install/
---

# Installation Guide

Get the Ultralytics YOLO Flutter plugin up and running in your project with this comprehensive installation guide.

## 📦 Add to pubspec.yaml

Add the plugin to your Flutter project's `pubspec.yaml`:

Package: https://pub.dev/packages/ultralytics_yolo

```yaml
dependencies:
  flutter:
    sdk: flutter
  ultralytics_yolo: ^0.4.0 # Latest version
```

Run the installation command:

```bash
flutter pub get
```

## 📱 Platform-Specific Setup

### iOS Configuration

#### 1. Update iOS Deployment Target

Edit `ios/Podfile` and set the minimum iOS version:

```ruby
# ios/Podfile
platform :ios, '13.0'  # Minimum iOS 13.0 required
```

#### 2. Camera Permission (Optional)

If using camera features, add permissions to `ios/Info.plist`:

```xml
<!-- ios/Info.plist -->
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for real-time object detection</string>
```

#### 3. Clean and Rebuild

```bash
cd ios
flutter clean
flutter pub get
cd ios && pod install --repo-update
cd .. && flutter run
```

### Android Configuration

#### 1. Update Minimum SDK Version

Edit `android/app/build.gradle`:

```gradle
// android/app/build.gradle
android {
    compileSdkVersion 36

    defaultConfig {
        minSdkVersion 23  // Minimum API level 23 required
        targetSdkVersion 36
    }
}
```

#### 2. Camera Permission (Optional)

Add permissions to `android/app/src/main/AndroidManifest.xml`:

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
```

#### 3. ProGuard / R8 Configuration (Release Builds)

The plugin ships consumer R8 rules that keep the LiteRT 2.x classes (`com.google.ai.edge.litert.**`) and metadata classes its native code reaches via JNI/reflection, so a standard release build needs no extra configuration.

If you use a custom R8 setup that strips these rules, the app can crash on model load (or report no detections) in release builds. In that case add to `android/app/proguard-rules.pro`:

```pro
# android/app/proguard-rules.pro
-keep class com.google.ai.edge.litert.** { *; }
-keep interface com.google.ai.edge.litert.** { *; }
-dontwarn com.google.ai.edge.litert.**
-keep class org.tensorflow.** { *; }
-keep class com.ultralytics.** { *; }
-dontwarn org.tensorflow.**
```

## 🎯 Model Setup

The simplest setup is to use an official model ID:

```dart
final yolo = YOLO(modelPath: 'yolo26n');
```

Use `YOLO.officialModels()` to see which IDs are available on the current platform.

For custom models:

- Android Flutter assets: `.tflite`
- iOS Flutter assets: `.mlpackage.zip`
- iOS bundled models: `.mlpackage` or `.mlmodel`

See [Quick Start](quickstart.md#-step-3-add-a-model) for the full flow.

## ✅ Verify Installation

Create a simple test to verify everything works:

```dart
// lib/test_yolo.dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

class TestYOLO extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('YOLO Test')),
      body: Center(
        child: ElevatedButton(
          child: Text('Test YOLO'),
          onPressed: () async {
            try {
              final yolo = YOLO(
                modelPath: 'yolo26n',
              );

              await yolo.loadModel();
              debugPrint('YOLO loaded successfully');

              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('YOLO plugin working!')),
              );
            } catch (e) {
              debugPrint('Error: $e');
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Error: $e')),
              );
            }
          },
        ),
      ),
    );
  }
}
```

## 🚀 Performance Optimization

### iOS Optimization

iOS inference runs on Core ML, which automatically uses the Neural Engine and GPU when available, so no extra configuration is required. Ship a Core ML model (`.mlpackage`/`.mlmodel`, or `.mlpackage.zip` in Flutter assets) and run on a real device for accurate performance.

### Android Optimization

Android inference runs on LiteRT 2.x via `CompiledModel`, which automatically tries a GPU → CPU accelerator ladder. Official int8 YOLO26 TFLite assets can compile on the LiteRT GPU path on supported devices, but int8 GPU coverage depends on the device driver and graph; unsupported graphs or ops may fall back to CPU. Confirm actual delegate placement from device logs. fp16, non-end-to-end exports are still useful for GPU benchmarking:

```python
YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
```

Keep `useGpu: true` for the automatic LiteRT GPU -> CPU ladder. See the [Performance Guide](performance.md) for the current device results.

You can also restrict native ABIs:

```gradle
// android/app/build.gradle
android {
    defaultConfig {
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
    }
}
```

## 🛠️ Development vs Production

### Development Setup

- Use smaller models (`yolo26n`) for faster iteration
- Enable debug logging
- Test on real devices for accurate performance

### Production Setup

- Optimize model size vs accuracy trade-offs
- Enable ProGuard/R8 code shrinking (the plugin's consumer rules keep LiteRT classes automatically)
- Test memory usage under load
- On Android, keep `useGpu: true` for the LiteRT GPU -> CPU ladder and verify delegate placement on target devices

## 📋 Requirements Summary

| Platform    | Minimum Version | Recommended   |
| ----------- | --------------- | ------------- |
| **iOS**     | 13.0+           | 14.0+         |
| **Android** | API 23+         | API 28+       |
| **Flutter** | 3.32.1+         | Latest stable |
| **Dart**    | 3.8.1+          | Latest stable |

## 🔍 Troubleshooting Installation

### Common Issues

**Issue**: `MissingPluginException`

```bash
# Solution: Clean and rebuild
flutter clean
flutter pub get
flutter run
```

**Issue**: iOS build fails with "No such module"

```bash
# Solution: Update pods
cd ios && pod install --repo-update
```

**Issue**: Android build fails with "API level" error

```bash
# Solution: Update compileSdkVersion and targetSdkVersion in android/app/build.gradle to 36
```

**Issue**: Model file not found

```bash
# Solution: Verify assets are correctly configured in pubspec.yaml
flutter packages get
```

## ✨ Next Steps

Once installation is complete:

1. **[⚡ Quick Start](quickstart.md)** - Get your first YOLO detection running
2. **[📖 Usage Guide](usage.md)** - Explore advanced features
3. **[🚀 Performance](performance.md)** - Optimize for your use case

---

**Need help?** Check our [troubleshooting guide](troubleshooting.md) or reach out to the [community](https://discord.com/invite/ultralytics).
