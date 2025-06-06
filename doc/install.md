---
title: Installation
description: How to add Ultralytics YOLO to your Flutter project - complete setup guide for iOS and Android
path: /integrations/flutter/install/
---

# Installation Guide

Get the Ultralytics YOLO Flutter plugin up and running in your project with this comprehensive installation guide.

## 📦 Add to pubspec.yaml

Add the plugin to your Flutter project's `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  ultralytics_yolo: ^0.1.18 # Latest version
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
platform :ios, '13.0'  # Minimum iOS 12.0 required
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
    compileSdkVersion 34

    defaultConfig {
        minSdkVersion 24  // Minimum API level 24 required
        targetSdkVersion 34
    }
}
```

#### 2. Camera Permission (Optional)

Add permissions to `android/app/src/main/AndroidManifest.xml`:

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

#### 3. ProGuard Configuration (Release Builds)

For release builds, add to `android/app/proguard-rules.pro`:

```pro
# android/app/proguard-rules.pro
-keep class org.tensorflow.** { *; }
-keep class com.ultralytics.** { *; }
-dontwarn org.tensorflow.**
```

## 🎯 [Model Files Setup](quickstart.md#-step-3-add-a-model)

Please check out the [quickstart.md](quickstart.md#-step-3-add-a-model)

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
                modelPath: 'yolo11n',
                task: YOLOTask.detect,
              );

              await yolo.loadModel();
              print('✅ YOLO loaded successfully!');

              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('YOLO plugin working!')),
              );
            } catch (e) {
              print('❌ Error: $e');
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

- Use smaller models (yolo11n) for faster iteration
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
| **iOS**     | 12.0+           | 14.0+         |
| **Android** | API 24+         | API 28+       |
| **Flutter** | 3.3.0+          | Latest stable |
| **Dart**    | 3.0.0+          | Latest stable |

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
# Solution: Update compileSdkVersion and targetSdkVersion in android/app/build.gradle
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
