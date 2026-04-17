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
  ultralytics_yolo: ^0.3.0 # Latest version
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
        minSdkVersion 21  // Minimum API level 21 required
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

#### 3. ProGuard Configuration (Release Builds)

For release builds, add to `android/app/proguard-rules.pro`:

```pro
# android/app/proguard-rules.pro
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
import 'package:ultralytics_yolo/yolo.dart';

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

Enable Metal Performance Shaders in `ios/Runner/Info.plist`:

```xml
<!-- ios/Runner/Info.plist -->
<key>UIRequiredDeviceCapabilities</key>
<array>
    <string>metal</string>
</array>
```

### Android Optimization

For better performance on Android, consider enabling:

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
- Enable ProGuard/R8 code shrinking
- Test memory usage under load
- Consider model quantization

## 📋 Requirements Summary

| Platform    | Minimum Version | Recommended   |
| ----------- | --------------- | ------------- |
| **iOS**     | 13.0+           | 14.0+         |
| **Android** | API 21+         | API 28+       |
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
