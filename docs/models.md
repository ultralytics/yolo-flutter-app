---
title: Model Integration
description: Complete guide for integrating YOLO models - CoreML, LiteRT formats
path: /integrations/flutter/models/
---

# Model Integration Guide

Complete guide for integrating YOLO models into your Flutter app with support for CoreML (iOS), LiteRT, and TFLite formats.

## 📱 Supported Model Formats

| Format     | Platform    | Extension                | Optimized For       | Performance |
| ---------- | ----------- | ------------------------ | ------------------- | ----------- |
| **CoreML** | iOS only    | `.mlpackage`, `.mlmodel` | Apple Neural Engine | Excellent   |
| **LiteRT** | Android/iOS | `.tflite`                | TensorFlow Lite     | Very Good   |

## 🎯 Getting YOLO Models

### Option 1: Download Pre-converted Models

**Download ready-to-use models from our [releases](https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0/):**

```bash
# Download from GitHub releases
curl -L -o yolo11n.tflite \
  https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0/yolo11n.tflite

curl -L -o yolo11n.mlpackage \
  https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0/yolo11n.mlpackage
```

### Option 2: Ultralytics Hub

1. Visit [HUB](https://www.ultralytics.com/hub)
2. Select your model
3. Export as **CoreML** (iOS) or **TFLite** (Android)
4. Download the exported model

### Option 3: Export from Ultralytics Python

Install Ultralytics and export models:

```bash
pip install ultralytics
```

#### Export to CoreML (iOS)

```python
from ultralytics import YOLO

# Load a YOLOv11 model
model = YOLO("yolo11n.pt")

# Export to CoreML
model.export(
    format="coreml",
    imgsz=640,
    half=False,  # Use float32 for better compatibility
    nms=True,  # Include NMS in the model
)
```

#### Export to LiteRT/TFLite (Android/iOS)

```python
from ultralytics import YOLO

# Load a YOLOv11 model
model = YOLO("yolo11n.pt")

# Export to TFLite with quantization
model.export(
    format="tflite",
    imgsz=640,
)
```

#### Advanced Export Options

```python
# For different YOLO tasks
tasks = {"segment": "yolo11n-seg.pt", "classify": "yolo11n-cls.pt", "pose": "yolo11n-pose.pt", "obb": "yolo11n-obb.pt"}

for task, model_path in tasks.items():
    model = YOLO(model_path)

    # Export CoreML
    model.export(format="coreml", imgsz=640, half=False)

    # Export TFLite
    model.export(format="tflite", imgsz=640)
```

## 🏗️ Platform Integration

### iOS - CoreML Integration

#### 1. Add Model to Xcode Project

```bash
# Open your iOS project
open ios/Runner.xcworkspace
```

1. **Drag and drop** the `.mlpackage` or `.mlmodel` file into the Xcode project
2. **Select target**: Choose "Runner" as the target
3. **Bundle settings**: Ensure "Add to target" is checked

#### 2. Verify Model Integration

Check that the model appears in your Xcode project:

```
ios/
├── Runner.xcworkspace
├── Runner/
│   ├── Assets.xcassets/
│   ├── yolo11n.mlpackage     ← Your model here
│   └── Info.plist
```

#### 3. Use in Flutter Code

```dart
final yolo = YOLO(
  modelPath: 'yolo11n',  // CoreML model
  task: YOLOTask.detect,
);
```

#### 4. iOS Optimization Settings

Add to `ios/Runner/Info.plist`:

```xml
<!-- Enable Neural Engine -->
<key>UIRequiredDeviceCapabilities</key>
<array>
    <string>neural-engine</string>
</array>

<!-- Metal for GPU acceleration -->
<key>UIRequiredDeviceCapabilities</key>
<array>
    <string>metal</string>
</array>
```

### Android - LiteRT/TFLite Integration

#### 1. Create Assets Directory

```bash
mkdir -p android/app/src/main/assets
```

#### 2. Add Model Files

Place your `.tflite` models in the assets directory:

```
android/
├── app/
│   └── src/
│       └── main/
│           ├── assets/
│           │   ├── yolo11n.tflite          ← Your models here
│           │   ├── yolo11n-seg.tflite
│           │   └── yolo11n-cls.tflite
│           └── AndroidManifest.xml
```

#### 3. Use in Flutter Code

```dart
final yolo = YOLO(
  modelPath: 'yolo11n',  // TFLite model
  task: YOLOTask.detect,
);
```

#### 4. Android Optimization

Update `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        // Enable NNAPI for hardware acceleration
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
    }
}

dependencies {
    // Add TensorFlow Lite GPU delegate (optional)
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
}
```

### Cross-Platform Strategy

For apps targeting both iOS and Android, use conditional model loading:

```dart
class CrossPlatformYOLO {
  late YOLO _yolo;

  Future<void> initializePlatformSpecificModel() async {
    if (Platform.isIOS) {
      // Use CoreML on iOS
      _yolo = YOLO(
        modelPath: 'yolo11n',
        task: YOLOTask.detect,
      );
    } else if (Platform.isAndroid) {
      // Use TFLite on Android
      _yolo = YOLO(
        modelPath: 'yolo11n',
        task: YOLOTask.detect,
      );
    }

    await _yolo.loadModel();
  }
}
```

## 📊 Model Comparison

### Performance Characteristics

| Model Size     | iOS (CoreML) | Android (TFLite) | Memory | Use Case             |
| -------------- | ------------ | ---------------- | ------ | -------------------- |
| **Nano (n)**   | 28-32 FPS    | 25-30 FPS        | ~150MB | Real-time apps       |
| **Small (s)**  | 20-25 FPS    | 18-22 FPS        | ~200MB | Balanced performance |
| **Medium (m)** | 15-18 FPS    | 12-15 FPS        | ~300MB | High accuracy        |
| **Large (l)**  | 10-12 FPS    | 8-10 FPS         | ~450MB | Maximum accuracy     |

### File Size Comparison

| Model    | Original (.pt) | CoreML (.mlpackage) | TFLite (.tflite) | TFLite INT8 |
| -------- | -------------- | ------------------- | ---------------- | ----------- |
| YOLOv11n | 5.4 MB         | 6.8 MB              | 6.2 MB           | 3.1 MB      |
| YOLOv11s | 19.8 MB        | 24.2 MB             | 21.5 MB          | 11.2 MB     |
| YOLOv11m | 45.7 MB        | 55.1 MB             | 49.7 MB          | 25.8 MB     |

## 🔧 Model Optimization

### CoreML Optimization (iOS)

```python
# Export with optimal settings for iOS
model = YOLO("yolo11n.pt")

model.export(
    format="coreml",
    imgsz=640,
    half=False,  # Use float32 for Neural Engine
    nms=True,  # Include NMS in model
    simplify=True,  # Simplify model for better performance
)
```

### TFLite Optimization (Android)

```python
# Different quantization options
model = YOLO("yolo11n.pt")

# INT8 quantization (smallest size)
model.export(
    format="tflite",
    imgsz=640,
    int8=True,
    data="coco128.yaml",  # Calibration dataset
)

# Float16 quantization (balance)
model.export(format="tflite", imgsz=640, half=True)

# No quantization (best accuracy)
model.export(format="tflite", imgsz=640)
```

## 🚀 Advanced Model Features

### Multi-Model Setup

```dart
class MultiModelManager {
  final Map<YOLOTask, YOLO> _models = {};

  Future<void> loadAllModels() async {
    final modelConfigs = Platform.isIOS ? {
      YOLOTask.detect: 'yolo11n',
      YOLOTask.segment: 'yolo11n-seg',
      YOLOTask.classify: 'yolo11n-cls',
    } : {
      YOLOTask.detect: 'yolo11n',
      YOLOTask.segment: 'yolo11n-seg',
      YOLOTask.classify: 'yolo11n-cls',
    };

    for (final entry in modelConfigs.entries) {
      _models[entry.key] = YOLO(
        modelPath: entry.value,
        task: entry.key,
        useMultiInstance: true,
      );
      await _models[entry.key]!.loadModel();
    }
  }

  YOLO? getModel(YOLOTask task) => _models[task];
}
```

### Model Switching

```dart
class AdaptiveModelLoader {
  YOLO? _currentModel;

  Future<void> switchToOptimalModel(String deviceType) async {
    await _currentModel?.dispose();

    String modelPath;
    if (deviceType == 'high-end') {
      modelPath = Platform.isIOS ? 'yolo11s' : 'yolo11s';
    } else {
      modelPath = Platform.isIOS ? 'yolo11n' : 'yolo11n';
    }

    _currentModel = YOLO(
      modelPath: modelPath,
      task: YOLOTask.detect,
    );

    await _currentModel!.loadModel();
  }
}
```

## 🔍 Model Validation

### Verify Model Integration

```dart
class ModelValidator {
  static Future<bool> validateModel(String modelPath, YOLOTask task) async {
    try {
      // Check if model file exists
      final exists = await YOLO.checkModelExists(modelPath);
      if (!exists['exists']) {
        print('❌ Model file not found: $modelPath');
        return false;
      }

      // Try to load the model
      final yolo = YOLO(modelPath: modelPath, task: task);
      final loaded = await yolo.loadModel();

      if (!loaded) {
        print('❌ Failed to load model: $modelPath');
        return false;
      }

      // Test with dummy data
      final testResult = await _testInference(yolo);
      await yolo.dispose();

      print('✅ Model validation successful: $modelPath');
      return testResult;

    } catch (e) {
      print('❌ Model validation failed: $e');
      return false;
    }
  }

  static Future<bool> _testInference(YOLO yolo) async {
    try {
      // Create test image (1x1 white pixel)
      final testImage = Uint8List.fromList([255, 255, 255, 255]);
      await yolo.predict(testImage);
      return true;
    } catch (e) {
      print('❌ Inference test failed: $e');
      return false;
    }
  }
}
```

### Model Performance Testing

```dart
class ModelBenchmark {
  static Future<Map<String, dynamic>> benchmarkModel(
    String modelPath,
    YOLOTask task,
    List<Uint8List> testImages,
  ) async {
    final yolo = YOLO(modelPath: modelPath, task: task);
    await yolo.loadModel();

    final times = <double>[];
    final stopwatch = Stopwatch();

    for (final image in testImages) {
      stopwatch.reset();
      stopwatch.start();

      await yolo.predict(image);

      stopwatch.stop();
      times.add(stopwatch.elapsedMilliseconds.toDouble());
    }

    await yolo.dispose();

    final avgTime = times.reduce((a, b) => a + b) / times.length;

    return {
      'model': modelPath,
      'task': task.name,
      'avg_time_ms': avgTime,
      'avg_fps': 1000 / avgTime,
      'samples': testImages.length,
      'platform': Platform.isIOS ? 'iOS' : 'Android',
    };
  }
}
```

## 🛠️ Troubleshooting Models

### Common Issues

**Issue**: Model file not found

```dart
// Solution: Verify file paths and asset configuration
final exists = await YOLO.checkModelExists('your_model.tflite');
print('Model exists: ${exists['exists']}');
print('Location: ${exists['location']}');
```

**Issue**: CoreML model not loading on iOS

```bash
# Check Xcode project settings
# 1. Verify model is added to Runner target
# 2. Check Bundle Resources in Build Phases
# 3. Ensure model format is compatible
```

**Issue**: TFLite model performance is slow

```dart
// Try different quantization levels
// INT8 (smallest, fastest): model.export(format='tflite', int8=True)
// Float16 (balanced): model.export(format='tflite', half=True)
// Float32 (largest, most accurate): model.export(format='tflite')
```

**Issue**: Memory issues with large models

```dart
// Use model size appropriate for device
class DeviceOptimizer {
  static String getOptimalModel(YOLOTask task) {
    final isLowEnd = _isLowEndDevice();
    final suffix = Platform.isIOS ? '.mlpackage' : '.tflite';

    if (isLowEnd) {
      return 'yolo11n${_getTaskSuffix(task)}$suffix';  // Nano
    } else {
      return 'yolo11s${_getTaskSuffix(task)}$suffix';  // Small
    }
  }
}
```

### Debug Model Loading

```dart
class ModelDebugger {
  static Future<void> debugModelLoading(String modelPath) async {
    print('🔍 Debugging model: $modelPath');

    // Check file existence
    final exists = await YOLO.checkModelExists(modelPath);
    print('Exists: ${exists['exists']}');
    print('Location: ${exists['location']}');

    // Check storage paths
    final paths = await YOLO.getStoragePaths();
    print('Storage paths: $paths');

    // Attempt loading with error handling
    try {
      final yolo = YOLO(modelPath: modelPath, task: YOLOTask.detect);
      final loaded = await yolo.loadModel();
      print('Loading success: $loaded');
      await yolo.dispose();
    } catch (e) {
      print('Loading error: $e');
    }
  }
}
```

## 📚 Additional Resources

- **[CoreML Documentation](https://docs.ultralytics.com/integrations/coreml/)** - Official CoreML integration guide
- **[TFLite Documentation](https://docs.ultralytics.com/integrations/tflite/)** - Official TensorFlow Lite guide
- **[Model Export Guide](https://docs.ultralytics.com/modes/export/)** - Comprehensive export documentation
- **[Ultralytics Hub](https://www.ultralytics.com/hub)** - Web-based model management

## 🎯 Best Practices

1. **Choose the right format**: CoreML for iOS-only apps, TFLite for cross-platform
2. **Start with nano models**: Use yolo11n for development, upgrade for production
3. **Test on real devices**: Emulators don't reflect real performance
4. **Validate models**: Always test model loading and inference
5. **Monitor performance**: Use benchmarking to compare model variants
6. **Handle errors gracefully**: Implement fallbacks for model loading failures

---

This model integration guide covers all aspects of working with YOLO models in Flutter. For implementation examples, see our [Usage Guide](usage.md), and for performance optimization, check the [Performance Guide](performance.md).
