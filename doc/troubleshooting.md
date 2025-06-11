---
title: Troubleshooting
description: Common issues and solutions for YOLO Flutter plugin - debugging guide and FAQ
path: /integrations/flutter/troubleshooting/
---

# Troubleshooting Guide

Comprehensive solutions for common issues with the Ultralytics YOLO Flutter plugin.

## 🚨 Installation Issues

### Plugin Not Found / MissingPluginException

**Symptoms**: `MissingPluginException(No implementation found for method)`

**Solution**:

```bash
# Clean and rebuild completely
flutter clean
flutter pub get
cd ios && pod install --repo-update # iOS only
flutter run
```

**Alternative solution**:

```bash
# If above doesn't work, try hot restart instead of hot reload
# In IDE: Stop app completely and restart
# CLI: Ctrl+C then flutter run again
```

### iOS Build Failures

#### "No such module 'ultralytics_yolo'"

**Solution**:

```bash
cd ios
pod deintegrate # Remove existing pods
pod install --repo-update
cd .. && flutter run
```

#### iOS Deployment Target Error

**Symptoms**: `The iOS deployment target 'IPHONEOS_DEPLOYMENT_TARGET' is set to 8.0`

**Solution**: Update `ios/Podfile`:

```ruby
# ios/Podfile
platform :ios, '13.0'  # Change to 13.0 or higher

# Also add this if needed:
post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '12.0'
    end
  end
end
```

### Android Build Failures

#### Minimum SDK Version Error

**Symptoms**: `uses-sdk:minSdkVersion 16 cannot be smaller than version 24`

**Solution**: Update `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        minSdkVersion 24  // Change from 16 to 24
        targetSdkVersion 34
        compileSdkVersion 34
    }
}
```

#### MultiDex Issue (Large Apps)

**Symptoms**: `Cannot fit requested classes in a single dex file`

**Solution**: Enable MultiDex in `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        multiDexEnabled true
    }
}

dependencies {
    implementation 'androidx.multidex:multidex:2.0.1'
}
```

## 🎯 Model Loading Issues

### Model Not Found Error

**Symptoms**: `ModelLoadingException: Model file not found`

**Debugging Steps**:

```dart
// 1. Check if model exists
final modelExists = await YOLO.checkModelExists('yolo11n');
print('Model exists: ${modelExists['exists']}');
print('Location: ${modelExists['location']}');

// 2. List available assets
final storagePaths = await YOLO.getStoragePaths();
print('Storage paths: $storagePaths');
```

**Common Solutions**:

1. **Please check again if the model file is correctly placed.**
   **iOS: Drag and drop the mlpackage directly into ios/Runner.xcproject and set the target to Runner.**
   **Android: Place the tflite in app/src/main/assets**

2. **Please make sure you are passing the model file to YOLOView/YOLO with the correct file name and task.**

```
YOLOView(
  modelPath: 'yolo11n',
  task: YOLOTask.detect // segment, classify, pose, obb
)
```

4. **Refresh assets**:

```bash
flutter packages get
flutter clean
flutter run
```

### Model Corruption

**Symptoms**: `InferenceException: Model inference failed`

**Solutions**:

1. **Re-download model**: Corrupt download is common
2. **Verify file size**: Check against official model size
3. **Test with different model**: Try yolo11n first

```dart
// Test model loading explicitly
try {
  final yolo = YOLO(
    modelPath: 'yolo11n',
    task: YOLOTask.detect,
  );

  final success = await yolo.loadModel();
  print('Model loaded: $success');
} catch (e) {
  print('Model loading failed: $e');
}
```

## 📱 Runtime Issues

### Memory Issues

#### OutOfMemoryError (Android)

**Symptoms**: App crashes with `OutOfMemoryError`

**Solutions**:

1. **Limit concurrent instances**:

```dart
class MemoryManager {
  static const int MAX_INSTANCES = 2;
  static final Map<String, YOLO> _instances = {};

  static Future<YOLO> getOrCreateInstance(String modelPath) async {
    if (_instances.length >= MAX_INSTANCES) {
      // Dispose oldest instance
      final oldest = _instances.entries.first;
      await oldest.value.dispose();
      _instances.remove(oldest.key);
    }

    if (!_instances.containsKey(modelPath)) {
      final yolo = YOLO(
        modelPath: modelPath,
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      await yolo.loadModel();
      _instances[modelPath] = yolo;
    }

    return _instances[modelPath]!;
  }
}
```

2. **Use smaller models**:

3. **Dispose properly**:

```dart
@override
void dispose() {
  super.dispose();
  yolo?.dispose();  // Always dispose YOLO instances
}
```

#### iOS Memory Warnings

**Solution**: Monitor and respond to memory pressure:

```dart
class MemoryAwareYOLO {
  YOLO? _primaryInstance;
  List<YOLO> _secondaryInstances = [];

  Future<void> handleMemoryWarning() async {
    // Dispose secondary instances first
    for (final instance in _secondaryInstances) {
      await instance.dispose();
    }
    _secondaryInstances.clear();

    print('🔋 Memory warning: Disposed secondary instances');
  }
}
```

### Model Loading Issues

#### Model Not Loading After switchModel()

**Problem**: Calling `controller.switchModel()` doesn't seem to work.

**Solutions**:

1. **Verify model path**:

```dart
// iOS: Use model name without extension or with .mlpackage
await controller.switchModel('yolo11n', YOLOTask.detect);
// or
await controller.switchModel('yolo11n.mlpackage', YOLOTask.detect);

// Android: Use full filename with .tflite extension
await controller.switchModel('yolo11n.tflite', YOLOTask.detect);
```

2. **Check model availability**:

```dart
try {
  await controller.switchModel(modelPath, task);
} catch (e) {
  print('Model switch failed: $e');
  // Model file might not exist or be in wrong location
}
```

3. **Platform-specific paths**:

```dart
import 'dart:io' show Platform;

final modelPath = Platform.isIOS ? 'yolo11n' : 'yolo11n.tflite';
await controller.switchModel(modelPath, YOLOTask.detect);
```

#### Camera Starts But No Detections

**Problem**: YOLOView shows camera but no detection results.

**Possible causes**:

- Model file doesn't exist at specified path
- Model loading failed silently
- Invalid model format

**Solution**: As of version 0.1.25, the plugin supports camera-only mode. If model loading fails, camera will continue without inference. Check logs for model loading errors:

```
iOS: "YOLOView Warning: Model file not found"
Android: "Failed to load model: [path]. Camera will run without inference."
```

This is now an intentional feature - you can start YOLOView with an invalid model path and load a valid model later using `switchModel()`:

```dart
final controller = YOLOViewController();

// Start with camera-only mode (model doesn't exist yet)
YOLOView(
  modelPath: 'model_downloading.tflite',  // Not available yet
  task: YOLOTask.detect,
  controller: controller,
  onResult: (results) {
    // Will receive empty results until model is loaded
    print('Detections: ${results.length}');
  },
)

// Later, when model is downloaded
await controller.switchModel('yolo11n', YOLOTask.detect);
```

### Performance Issues

#### Low FPS / Slow Inference

**Debugging**:

```dart
class PerformanceDebugger {
  void onPerformanceMetrics(YOLOPerformanceMetrics metrics) {
    print('🔍 Performance Debug:');
    print('  FPS: ${metrics.fps.toStringAsFixed(1)}');
    print('  Processing: ${metrics.processingTimeMs.toStringAsFixed(1)}ms');
    print('  Rating: ${metrics.performanceRating}');

    if (metrics.hasPerformanceIssues) {
      print('⚠️ Performance issues detected!');
      _suggestOptimizations();
    }
  }

  void _suggestOptimizations() {
    print('💡 Try these optimizations:');
    print('  • Use yolo11n instead of larger models');
    print('  • Increase confidence threshold (0.6+)');
    print('  • Reduce max FPS in streaming config');
    print('  • Disable masks/poses if not needed');
  }
}
```

**Solutions**:

1. **Optimize thresholds**:

```dart
// High-performance settings
await controller.setThresholds(
  confidenceThreshold: 0.7,  // Higher = fewer detections
  iouThreshold: 0.3,         // Lower = faster NMS
  numItemsThreshold: 10,     // Limit max detections
);
```

2. **Use efficient streaming**:

```dart
final config = YOLOStreamingConfig.powerSaving(
  inferenceFrequency: 10,  // 10 FPS inference
  maxFPS: 15,              // 15 FPS display
);
```

3. **Profile on real devices**:

```bash
# Android profiling
flutter run --profile
# Monitor in Android Studio Profiler

# iOS profiling
flutter run --profile
# Monitor in Xcode Instruments
```

### Camera Issues

#### Camera Permission Denied

**Android Solution**:

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
```

Request permission in code:

```dart
import 'package:permission_handler/permission_handler.dart';

Future<void> requestCameraPermission() async {
  final status = await Permission.camera.request();
  if (status != PermissionStatus.granted) {
    print('Camera permission denied');
  }
}
```

**iOS Solution**:

```xml
<!-- ios/Runner/Info.plist -->
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for real-time object detection</string>
```

#### YOLOView Not Showing Camera

**Debugging**:

```dart
YOLOView(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  showNativeUI: true,  // Enable to see native camera controls
  onResult: (results) {
    print('Camera working: ${results.length} detections');
  },
  onStreamingData: (data) {
    print('Streaming data received: ${data.keys}');
  },
)
```

**Solutions**:

1. **Check permissions** (see above)
2. **Verify model loading** before camera starts
3. **Test on real device** (not emulator)

## 🔀 Multi-Instance Issues

### Instance ID Conflicts

**Symptoms**: Multiple instances interfering with each other

**Solution**: Use proper instance management:

```dart
class ProperInstanceManager {
  Future<YOLO> createUniqueInstance(String modelPath, YOLOTask task) async {
    final yolo = YOLO(
      modelPath: modelPath,
      task: task,
      useMultiInstance: true,  // Must be true for multi-instance
    );

    await yolo.loadModel();

    // Verify instance is registered
    final isRegistered = YOLOInstanceManager.hasInstance(yolo.instanceId);
    print('Instance ${yolo.instanceId} registered: $isRegistered');

    return yolo;
  }
}
```

### Memory Leaks in Multi-Instance

**Debugging**: Track active instances:

```dart
void debugInstances() {
  final activeIds = YOLOInstanceManager.getActiveInstanceIds();
  print('Active instances: ${activeIds.length}');
  for (final id in activeIds) {
    print('  - $id');
  }
}
```

**Solution**: Proper cleanup:

```dart
class InstanceCleanup {
  final List<YOLO> _managedInstances = [];

  Future<YOLO> createManagedInstance(String modelPath, YOLOTask task) async {
    final yolo = YOLO(
      modelPath: modelPath,
      task: task,
      useMultiInstance: true,
    );

    await yolo.loadModel();
    _managedInstances.add(yolo);

    return yolo;
  }

  Future<void> disposeAll() async {
    print('Disposing ${_managedInstances.length} instances');

    await Future.wait(
      _managedInstances.map((yolo) => yolo.dispose()),
    );

    _managedInstances.clear();

    // Verify cleanup
    final remaining = YOLOInstanceManager.getActiveInstanceIds();
    print('Remaining instances: ${remaining.length}');
  }
}
```

## 🐛 Common Runtime Errors

### "Invalid input format"

**Cause**: Wrong image format or size

**Solution**:

```dart
Future<Uint8List> prepareImageBytes(File imageFile) async {
  // Ensure image is in correct format
  final bytes = await imageFile.readAsBytes();

  // Verify it's a valid image
  if (bytes.length < 100) {
    throw InvalidInputException('Image file too small or corrupted');
  }

  return bytes;
}
```

### "Model task mismatch"

**Cause**: Using wrong model for task

**Solution**:

```dart
// ❌ Wrong: detection model with pose task
final yolo = YOLO(
  modelPath: 'yolo11n',      // Detection model
  task: YOLOTask.pose,                             // Pose task
);

// ✅ Correct: pose model with pose task
final yolo = YOLO(
  modelPath: 'yolo11n-pose', // Pose model
  task: YOLOTask.pose,                             // Pose task
);
```

### "State error: View not initialized"

**Cause**: Calling methods before view is ready

**Solution**:

```dart
class SafeViewController {
  YOLOViewController? _controller;
  bool _isViewReady = false;

  void onViewCreated(YOLOViewController controller) {
    _controller = controller;
    _isViewReady = true;
  }

  Future<void> safeSetThresholds(double confidence) async {
    if (_isViewReady && _controller != null) {
      await _controller!.setConfidenceThreshold(confidence);
    } else {
      print('⚠️ View not ready yet');
    }
  }
}
```

## 🔧 Debug Mode

### Enable Verbose Logging

```dart
import 'package:ultralytics_yolo/utils/logger.dart';

void main() {
  // Enable debug logging
  logInfo('Debug mode enabled');

  runApp(MyApp());
}
```

### Performance Profiling

```dart
class DebugProfiler {
  late Stopwatch _stopwatch;

  void startProfiling() {
    _stopwatch = Stopwatch()..start();
  }

  void logPerformance(String operation) {
    _stopwatch.stop();
    print('⏱️ $operation took: ${_stopwatch.elapsedMilliseconds}ms');
    _stopwatch.reset();
    _stopwatch.start();
  }

  Future<void> profileYOLOOperations() async {
    startProfiling();

    final yolo = YOLO(
      modelPath: 'yolo11n',
      task: YOLOTask.detect,
    );
    logPerformance('YOLO instantiation');

    await yolo.loadModel();
    logPerformance('Model loading');

    final imageBytes = await loadTestImage();
    logPerformance('Image loading');

    final results = await yolo.predict(imageBytes);
    logPerformance('Inference');

    await yolo.dispose();
    logPerformance('Disposal');
  }
}
```

## 📞 Getting Help

### Before Asking for Help

1. **Check logs**: Look for specific error messages
2. **Test on device**: Not emulator
3. **Try minimal example**: Reduce to simplest case
4. **Check versions**: Flutter, plugin, and dependencies

### Collecting Debug Information

```dart
Future<Map<String, dynamic>> collectDebugInfo() async {
  return {
    'flutter_version': 'Run: flutter --version',
    'plugin_version': '0.1.25',
    'platform': Platform.isIOS ? 'iOS' : 'Android',
    'model_path': 'yolo11n',
    'model_exists': await YOLO.checkModelExists('yolo11n'),
    'storage_paths': await YOLO.getStoragePaths(),
    'active_instances': YOLOInstanceManager.getActiveInstanceIds().length,
  };
}
```

### Community Support

- **Discord**: [Ultralytics Discord](https://discord.com/invite/ultralytics)
- **GitHub Issues**: [Flutter Plugin Issues](https://github.com/ultralytics/ultralytics/issues)
- **Documentation**: [Official Docs](https://docs.ultralytics.com)

## 🎯 Quick Fixes Checklist

When something goes wrong, try these in order:

1. ✅ **Hot Restart** (not hot reload)
2. ✅ **Flutter Clean**: `flutter clean && flutter pub get`
3. ✅ **Check Permissions**: Camera access if using YOLOView
4. ✅ **Verify Model**: Correct path and file exists
5. ✅ **Test on Device**: Real device, not emulator
6. ✅ **Update Pods** (iOS): `cd ios && pod install --repo-update`
7. ✅ **Check Logs**: Look for specific error messages
8. ✅ **Minimal Example**: Reduce to simplest working case

## 🔍 Still Having Issues?

If none of these solutions work:

1. **Create minimal reproduction**: Simplest possible example
2. **Collect debug info**: Use the `collectDebugInfo()` function above
3. **Check existing issues**: Someone might have the same problem
4. **Open new issue**: With complete debug information

Remember: Most issues are configuration or setup related. Double-check the [Installation Guide](install.md) and try the [Quick Start](quickstart.md) example first.

If still having issue - let's link this here
https://github.com/ultralytics/yolo-flutter-app/issues
