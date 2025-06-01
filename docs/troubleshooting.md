# Troubleshooting Guide

This guide helps you resolve common issues when using YOLO Flutter, from setup problems to performance optimization.

## 🚨 Quick Fixes

### Most Common Issues

| Problem | Quick Fix | Time to Fix |
|---------|-----------|-------------|
| **Model not loading** | Check file path and format | 2 minutes |
| **Camera permission denied** | Add permissions to manifest | 1 minute |
| **Low FPS performance** | Switch to `yolo11n` model | 30 seconds |
| **App crashes on detection** | Update to latest plugin version | 2 minutes |
| **No detection results** | Lower confidence threshold | 30 seconds |

## 📱 Installation & Setup Issues

### Android Setup Problems

#### Issue: "Minimum SDK version" error
```
Error: Requires minimum SDK version 21
```

**Solution:**
```gradle
// In android/app/build.gradle
android {
    defaultConfig {
        minSdkVersion 21  // Required minimum
    }
}
```

#### Issue: Camera permission denied
```
PlatformException: Camera permission denied
```

**Solution:**
```xml
<!-- Add to android/app/src/main/AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
```

#### Issue: TensorFlow Lite model loading fails
```
Error: Failed to load TensorFlow Lite model
```

**Solutions:**
1. **Check model placement:**
   ```
   android/app/src/main/assets/yolo11n.tflite
   ```

2. **Verify model format:**
   ```dart
   // Use correct model path
   YOLOView(
     modelPath: 'yolo11n',  // Without .tflite extension
     task: YOLOTask.detect,
   )
   ```

3. **Check model export:**
   ```python
   # Ensure model was exported correctly
   model.export(format="tflite", int8=True, imgsz=[320, 320], nms=False)
   ```

### iOS Setup Problems

#### Issue: Core ML model not found
```
Error: Could not load Core ML model
```

**Solution:**
1. **Add model to Xcode project:**
   - Open `ios/Runner.xcworkspace`
   - Drag `yolo11n.mlpackage` into project
   - Ensure "Add to target" includes Runner

2. **Verify model in Bundle Resources:**
   - Select Runner target
   - Build Phases → Copy Bundle Resources
   - Ensure your `.mlpackage` is listed

#### Issue: iOS deployment target too low
```
Error: iOS deployment target is below minimum required version
```

**Solution:**
```ruby
# In ios/Podfile
platform :ios, '13.0'  # Minimum required
```

#### Issue: Camera usage description missing
```
Error: This app has crashed because it attempted to access privacy-sensitive data
```

**Solution:**
```xml
<!-- Add to ios/Runner/Info.plist -->
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for object detection</string>
```

## 🏃‍♂️ Runtime Issues

### Model Loading Problems

#### Issue: Model file not found
```dart
// Common error patterns
YOLOException: Model file not found at path: assets/yolo11n.tflite
```

**Debugging Steps:**
```dart
// 1. Check if model exists
final exists = await YOLO.checkModelExists('yolo11n');
print('Model exists: $exists');

// 2. List available paths
final paths = await YOLO.getStoragePaths();
print('Available paths: $paths');

// 3. Use simple model name (recommended)
YOLOView(
  modelPath: 'yolo11n',  // Let plugin find the right file
  task: YOLOTask.detect,
)
```

#### Issue: Model format incompatible
```
Error: Unsupported model format
```

**Solution:**
```python
# Re-export with correct parameters
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# For Android
model.export(format="tflite", int8=True, imgsz=[320, 320], nms=False)

# For iOS  
model.export(format="coreml", int8=True, imgsz=[640, 384], nms=True)
```

### Performance Issues

#### Issue: Very low FPS (< 10)
```
FPS dropping below 10, app feels sluggish
```

**Solutions (in order of impact):**

1. **Switch to smaller model:**
   ```dart
   // Change from yolo11s/m to yolo11n
   YOLOView(
     modelPath: 'yolo11n',  // Smallest, fastest model
     task: YOLOTask.detect,
   )
   ```

2. **Reduce inference frequency:**
   ```dart
   YOLOView(
     streamingConfig: YOLOStreamingConfig.custom(
       inferenceFrequency: 10,  // Reduce from default 15
     ),
   )
   ```

3. **Use minimal streaming config:**
   ```dart
   YOLOView(
     streamingConfig: YOLOStreamingConfig.minimal(),
   )
   ```

#### Issue: High memory usage / crashes
```
Out of memory errors or app crashes after extended use
```

**Solutions:**
```dart
// 1. Enable frame dropping
YOLOView(
  streamingConfig: YOLOStreamingConfig.custom(
    dropFramesWhenBusy: true,
    bufferSize: 1,  // Minimal buffering
  ),
)

// 2. Implement lifecycle management
class MemoryManagedYOLO extends StatefulWidget {
  @override
  _MemoryManagedYOLOState createState() => _MemoryManagedYOLOState();
}

class _MemoryManagedYOLOState extends State<MemoryManagedYOLO> 
    with WidgetsBindingObserver {
  
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused) {
      // Reduce processing when app is backgrounded
      controller.updateStreamingConfig(
        YOLOStreamingConfig.custom(inferenceFrequency: 1)
      );
    }
  }
}
```

### Detection Issues

#### Issue: No detections appearing
```dart
// onResult callback receiving empty results
onResult: (results) {
  print('Results: ${results.length}'); // Always prints 0
}
```

**Debugging Steps:**

1. **Lower confidence threshold:**
   ```dart
   YOLOView(
     controller: controller,
     onResult: (results) => print('Found ${results.length} objects'),
   )
   
   // Set very low threshold to test
   controller.setConfidenceThreshold(0.1);
   ```

2. **Test with different lighting:**
   - Ensure good lighting conditions
   - Try different angles and distances

3. **Verify model task alignment:**
   ```dart
   // Ensure model and task match
   YOLOView(
     modelPath: 'yolo11n',        // Detection model
     task: YOLOTask.detect,       // Detection task
   )
   
   // NOT:
   // modelPath: 'yolo11n-seg',   // Segmentation model  
   // task: YOLOTask.detect,      // Detection task - MISMATCH!
   ```

#### Issue: Detections are inaccurate
```
Getting false positives or missed detections
```

**Solutions:**

1. **Adjust confidence threshold:**
   ```dart
   // Higher threshold = fewer false positives
   controller.setConfidenceThreshold(0.7);
   
   // Lower threshold = more detections (may include false positives)
   controller.setConfidenceThreshold(0.3);
   ```

2. **Adjust IoU threshold:**
   ```dart
   // Higher IoU = fewer overlapping boxes
   controller.setIoUThreshold(0.6);
   ```

3. **Consider model upgrade:**
   ```dart
   // Upgrade from nano to small for better accuracy
   YOLOView(
     modelPath: 'yolo11s',  // More accurate than yolo11n
     task: YOLOTask.detect,
   )
   ```

## 🎭 Task-Specific Issues

### Segmentation Problems

#### Issue: Mask data not appearing
```dart
// Mask field is null in results
for (final result in results) {
  print('Mask: ${result.mask}');  // Always prints null
}
```

**Solution:**
```dart
// 1. Ensure segmentation model and task
YOLOView(
  modelPath: 'yolo11n-seg',      // Segmentation model
  task: YOLOTask.segment,        // Segmentation task
  streamingConfig: YOLOStreamingConfig.withMasks(),  // Enable masks
  onResult: (results) {
    for (final result in results) {
      if (result.mask != null) {
        print('Mask size: ${result.mask!.length}x${result.mask!.first.length}');
      }
    }
  },
)
```

### Pose Estimation Problems

#### Issue: Keypoints not detected
```dart
// Keypoints field is null in results
result.keypoints == null  // Always true
```

**Solution:**
```dart
YOLOView(
  modelPath: 'yolo11n-pose',     // Pose estimation model
  task: YOLOTask.pose,           // Pose task
  streamingConfig: YOLOStreamingConfig.withPoses(),  // Enable poses
  onResult: (results) {
    for (final result in results) {
      if (result.keypoints != null) {
        print('Found ${result.keypoints!.length} keypoints');
      }
    }
  },
)
```

## 🔄 Streaming & Real-time Issues

### Streaming Configuration Problems

#### Issue: Streaming callbacks not working
```dart
// onPerformanceMetrics never called
onPerformanceMetrics: (metrics) {
  print('FPS: ${metrics.fps}');  // Never prints
}
```

**Solution:**
```dart
// Ensure streaming is properly configured
YOLOView(
  streamingConfig: YOLOStreamingConfig.balanced(),  // Enable streaming
  onPerformanceMetrics: (metrics) {
    print('FPS: ${metrics.fps?.toStringAsFixed(1)}');
  },
)
```

#### Issue: Inference frequency not taking effect
```dart
// Setting inference frequency but no performance change
YOLOStreamingConfig.custom(inferenceFrequency: 5)  // Should be slower
```

**Debugging:**
```dart
// 1. Verify configuration is applied
final controller = YOLOViewController();
YOLOView(
  controller: controller,
  streamingConfig: YOLOStreamingConfig.custom(inferenceFrequency: 5),
  onPerformanceMetrics: (metrics) {
    print('Actual inference frequency: ${metrics.inferenceFrequency}');
  },
)

// 2. Update configuration dynamically
await controller.updateStreamingConfig(
  YOLOStreamingConfig.custom(inferenceFrequency: 3)
);
```

## 🔧 Advanced Debugging

### Enable Debug Logging

```dart
// Enable verbose logging for debugging
class DebugYOLO extends StatefulWidget {
  @override
  _DebugYOLOState createState() => _DebugYOLOState();
}

class _DebugYOLOState extends State<DebugYOLO> {
  @override
  Widget build(BuildContext context) {
    return YOLOView(
      modelPath: 'yolo11n',
      task: YOLOTask.detect,
      onResult: (results) {
        // Debug detection results
        print('=== Detection Results ===');
        print('Count: ${results.length}');
        for (int i = 0; i < results.length; i++) {
          final result = results[i];
          print('[$i] ${result.className}: ${result.confidence.toStringAsFixed(3)}');
          print('    BBox: ${result.boundingBox}');
          if (result.mask != null) {
            print('    Mask: ${result.mask!.length}x${result.mask!.first.length}');
          }
          if (result.keypoints != null) {
            print('    Keypoints: ${result.keypoints!.length}');
          }
        }
      },
      onPerformanceMetrics: (metrics) {
        // Debug performance metrics
        print('=== Performance Metrics ===');
        print('FPS: ${metrics.fps?.toStringAsFixed(1)}');
        print('Processing Time: ${metrics.processingTimeMs?.toStringAsFixed(1)}ms');
        print('Memory Usage: ${metrics.memoryUsageMB?.toStringAsFixed(1)}MB');
      },
    );
  }
}
```

### Performance Profiling

```dart
class PerformanceProfiler {
  final List<double> _frameTime = [];
  final List<double> _fpsHistory = [];
  
  void profilePerformance(PerformanceMetrics metrics) {
    if (metrics.processingTimeMs != null) {
      _frameTime.add(metrics.processingTimeMs!);
    }
    if (metrics.fps != null) {
      _fpsHistory.add(metrics.fps!);
    }
    
    // Keep only recent data
    if (_frameTime.length > 100) _frameTime.removeAt(0);
    if (_fpsHistory.length > 100) _fpsHistory.removeAt(0);
    
    // Print statistics every 50 frames
    if (_frameTime.length % 50 == 0) {
      _printStatistics();
    }
  }
  
  void _printStatistics() {
    if (_frameTime.isEmpty || _fpsHistory.isEmpty) return;
    
    final avgFrameTime = _frameTime.reduce((a, b) => a + b) / _frameTime.length;
    final avgFPS = _fpsHistory.reduce((a, b) => a + b) / _fpsHistory.length;
    final maxFrameTime = _frameTime.reduce((a, b) => a > b ? a : b);
    final minFPS = _fpsHistory.reduce((a, b) => a < b ? a : b);
    
    print('=== Performance Statistics ===');
    print('Average Frame Time: ${avgFrameTime.toStringAsFixed(1)}ms');
    print('Average FPS: ${avgFPS.toStringAsFixed(1)}');
    print('Max Frame Time: ${maxFrameTime.toStringAsFixed(1)}ms');
    print('Min FPS: ${minFPS.toStringAsFixed(1)}');
    
    // Warn about performance issues
    if (avgFPS < 15) {
      print('⚠️ Warning: Low average FPS detected');
    }
    if (maxFrameTime > 100) {
      print('⚠️ Warning: High frame time spikes detected');
    }
  }
}
```

## 📋 Diagnostic Checklist

### When Things Aren't Working

Run through this checklist systematically:

#### Setup Verification
- [ ] Plugin version is latest: `flutter pub deps | grep ultralytics_yolo`
- [ ] Platform setup complete (permissions, SDK versions)
- [ ] Model files in correct locations
- [ ] Model names match exactly (no extra extensions)

#### Model Verification  
- [ ] Model exported with correct parameters
- [ ] Model format matches platform (`.tflite` for Android, `.mlpackage` for iOS)
- [ ] Model task matches YOLOTask enum
- [ ] Model exists: `await YOLO.checkModelExists('model_name')`

#### Performance Verification
- [ ] Appropriate model size for device
- [ ] Streaming configuration reasonable for hardware
- [ ] Memory usage within limits
- [ ] No blocking operations in callbacks

#### Detection Verification
- [ ] Confidence threshold appropriate (try 0.1 for testing)
- [ ] Good lighting and clear objects
- [ ] Objects within model's training classes
- [ ] Camera pointing at detectable objects

## 🆘 Getting Help

### Community Support

1. **GitHub Issues**: [Report bugs and request features](https://github.com/ultralytics/yolo-flutter-app/issues)
2. **Discord**: [Join the Ultralytics community](https://discord.com/invite/ultralytics)
3. **Forums**: [Ultralytics community forums](https://community.ultralytics.com/)

### Creating Good Bug Reports

Include this information when reporting issues:

```dart
// Include in your bug report:
void main() {
  print('Flutter version: ${Platform.version}');
  print('Plugin version: [from pubspec.yaml]');
  print('Platform: ${Platform.operatingSystem}');
  print('Device: [device model]');
  print('Model: [model name and size]');
  print('Issue: [specific problem description]');
  
  // Include minimal reproduction code
  runApp(MyApp());
}
```

### Before Posting

1. **Search existing issues** - your problem might already be solved
2. **Try the latest version** - bugs are fixed regularly
3. **Test with minimal example** - isolate the problem
4. **Include complete error messages** - don't truncate stack traces

## 🔧 Common Error Messages

### Quick Reference

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `Model file not found` | Wrong path or missing file | Check model placement and path |
| `Camera permission denied` | Missing permissions | Add camera permissions |
| `Unsupported model format` | Wrong model export | Re-export with correct parameters |
| `Out of memory` | Model too large | Use smaller model or reduce frequency |
| `Platform exception` | Setup issue | Check platform-specific setup |
| `No detections` | High confidence threshold | Lower confidence to 0.1 for testing |

## 🚀 Performance Recovery

If your app performance degrades over time:

1. **Restart detection:**
   ```dart
   await controller.stop();
   await controller.start();
   ```

2. **Clear model cache:**
   ```dart
   await YOLO.clearModelCache();
   ```

3. **Reduce streaming load:**
   ```dart
   controller.updateStreamingConfig(YOLOStreamingConfig.minimal());
   ```

4. **Monitor memory:**
   ```dart
   // Implement memory monitoring and restart if needed
   ```

Remember: Most issues are configuration-related and can be fixed quickly with the right approach!