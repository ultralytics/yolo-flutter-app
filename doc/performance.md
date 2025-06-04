---
title: Performance Guide
description: Optimization tips, benchmarks, and best practices for YOLO Flutter plugin performance
path: /integrations/flutter/performance/
---

# Performance Guide

Optimize your YOLO Flutter app for maximum performance with proven strategies and benchmarks.

## 📊 Performance Benchmarks

### Model Performance Comparison

| Model        | Size    | Android (Pixel 9) | iOS (iPhone 16) | Memory Usage |
| ------------ | ------- | ----------------- | --------------- | ------------ |
| **YOLOv11n** | 6.2 MB  | 25-30 FPS         | 28-32 FPS       | ~150 MB      |
| **YOLOv11s** | 21.5 MB | 18-22 FPS         | 20-25 FPS       | ~200 MB      |
| **YOLOv11m** | 49.7 MB | 12-15 FPS         | 15-18 FPS       | ~300 MB      |
| **YOLOv11l** | 86.9 MB | 8-10 FPS          | 10-12 FPS       | ~450 MB      |

### Task Performance Comparison

| Task               | Model         | FPS Range | Best Use Case           |
| ------------------ | ------------- | --------- | ----------------------- |
| **Detection**      | YOLOv11n      | 25-30     | Real-time applications  |
| **Segmentation**   | YOLOv11n-seg  | 15-25     | Photo editing           |
| **Classification** | YOLOv11n-cls  | 30+       | Content moderation      |
| **Pose**           | YOLOv11n-pose | 20-30     | Fitness, motion capture |
| **OBB**            | YOLOv11n-obb  | 20-25     | aerial photography      |

## 🚀 Optimization Strategies

### 1. Model Selection

#### Choose the Right Model Size

```dart
// For real-time applications - prioritize speed
final fastYOLO = YOLO(
  modelPath: 'yolo11n',  // Nano - fastest
  task: YOLOTask.detect,
);

// For accuracy-critical applications
final accurateYOLO = YOLO(
  modelPath: 'yolo11s',  // Small - balanced
  task: YOLOTask.detect,
);

// For better accuracy (use sparingly)
final preciseYOLO = YOLO(
  modelPath: 'yolo11m',  // Medium - high accuracy
  task: YOLOTask.detect,
);
```

#### Task-Specific Optimization

```dart
// Detection: Use nano for real-time
final detector = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);

// Segmentation: Consider small model for better masks
final segmenter = YOLO(modelPath: 'yolo11s-seg', task: YOLOTask.segment);

// Classification: Nano is usually sufficient
final classifier = YOLO(modelPath: 'yolo11n-cls', task: YOLOTask.classify);
```

### 2. Threshold Optimization

#### Smart Threshold Configuration

```dart
class PerformanceOptimizer {
  late YOLOViewController controller;

  Future<void> optimizeForSpeed() async {
    controller = YOLOViewController();

    // Higher confidence = fewer detections = faster processing
    await controller.setThresholds(
      confidenceThreshold: 0.6,    // Higher threshold
      iouThreshold: 0.3,           // Lower IoU for faster NMS
      numItemsThreshold: 10,       // Limit max detections
    );
  }

  Future<void> optimizeForAccuracy() async {
    controller = YOLOViewController();

    // Lower confidence = more detections = better recall
    await controller.setThresholds(
      confidenceThreshold: 0.25,   // Lower threshold
      iouThreshold: 0.5,           // Higher IoU for better precision
      numItemsThreshold: 50,       // Allow more detections
    );
  }

  Future<void> optimizeForBattery() async {
    controller = YOLOViewController();

    // Reduce processing load
    await controller.setThresholds(
      confidenceThreshold: 0.7,    // High confidence only
      iouThreshold: 0.2,           // Fast NMS
      numItemsThreshold: 5,        // Minimal detections
    );
  }
}
```

### 3. Streaming Configuration

#### Power-Saving Configurations

```dart
// Battery-efficient streaming
final powerSavingConfig = YOLOStreamingConfig.powerSaving(
  inferenceFrequency: 5,    // 5 FPS inference
  maxFPS: 10,               // 10 FPS display
);

// Balanced performance
final balancedConfig = YOLOStreamingConfig.throttled(
  maxFPS: 15,
  includeMasks: false,      // Disable expensive features
  includeOriginalImage: false,
);

// High-performance mode
final highPerfConfig = YOLOStreamingConfig.highPerformance(
  inferenceFrequency: 30,
);
```

#### Adaptive Performance

```dart
class AdaptivePerformance {
  YOLOStreamingConfig _currentConfig = YOLOStreamingConfig.minimal();
  double _avgFps = 30.0;
  int _frameCount = 0;

  void onPerformanceMetrics(YOLOPerformanceMetrics metrics) {
    _frameCount++;
    _avgFps = (_avgFps * (_frameCount - 1) + metrics.fps) / _frameCount;

    // Adapt configuration based on performance
    if (_avgFps < 15 && metrics.processingTimeMs > 100) {
      _adaptForLowPerformance();
    } else if (_avgFps > 25 && metrics.processingTimeMs < 50) {
      _adaptForHighPerformance();
    }
  }

  void _adaptForLowPerformance() {
    _currentConfig = YOLOStreamingConfig.powerSaving(
      inferenceFrequency: 8,
      maxFPS: 12,
    );
    print('🔋 Adapted to power-saving mode');
  }

  void _adaptForHighPerformance() {
    _currentConfig = YOLOStreamingConfig.throttled(
      maxFPS: 20,
      includeMasks: true,
    );
    print('🚀 Adapted to high-performance mode');
  }
}
```

### 4. Multi-Instance Optimization

#### Memory-Efficient Multi-Instance

```dart
class OptimizedMultiInstance {
  static const int MAX_CONCURRENT = 2;  // Limit concurrent instances
  final Map<String, YOLO> _instances = {};
  final Map<String, DateTime> _lastUsed = {};

  Future<YOLO> getOrCreateInstance(String modelPath, YOLOTask task) async {
    final key = '${modelPath}_${task.name}';

    // Return existing instance if available
    if (_instances.containsKey(key)) {
      _lastUsed[key] = DateTime.now();
      return _instances[key]!;
    }

    // Clean up old instances if at limit
    if (_instances.length >= MAX_CONCURRENT) {
      await _cleanupOldestInstance();
    }

    // Create new instance
    final yolo = YOLO(
      modelPath: modelPath,
      task: task,
      useMultiInstance: true,
    );

    await yolo.loadModel();
    _instances[key] = yolo;
    _lastUsed[key] = DateTime.now();

    return yolo;
  }

  Future<void> _cleanupOldestInstance() async {
    if (_lastUsed.isEmpty) return;

    // Find oldest instance
    final oldestKey = _lastUsed.entries
        .reduce((a, b) => a.value.isBefore(b.value) ? a : b)
        .key;

    // Dispose and remove
    await _instances[oldestKey]?.dispose();
    _instances.remove(oldestKey);
    _lastUsed.remove(oldestKey);

    print('🗑️ Cleaned up instance: $oldestKey');
  }

  Future<void> disposeAll() async {
    await Future.wait(_instances.values.map((yolo) => yolo.dispose()));
    _instances.clear();
    _lastUsed.clear();
  }
}
```

#### Smart Instance Management

```dart
class SmartInstanceManager {
  final Map<YOLOTask, YOLO> _taskInstances = {};
  bool _isHighMemoryPressure = false;

  Future<YOLO> getInstanceForTask(YOLOTask task) async {
    // Use task-specific instance or create new one
    if (!_taskInstances.containsKey(task)) {
      final modelPath = _getModelPathForTask(task);
      _taskInstances[task] = YOLO(
        modelPath: modelPath,
        task: task,
        useMultiInstance: true,
      );
      await _taskInstances[task]!.loadModel();
    }

    return _taskInstances[task]!;
  }

  Future<void> handleMemoryPressure() async {
    _isHighMemoryPressure = true;

    // Keep only the most recently used instance
    if (_taskInstances.length > 1) {
      final tasks = _taskInstances.keys.toList();
      for (int i = 0; i < tasks.length - 1; i++) {
        await _taskInstances[tasks[i]]?.dispose();
        _taskInstances.remove(tasks[i]);
      }
      print('🔴 Memory pressure: Reduced to ${_taskInstances.length} instances');
    }
  }

  String _getModelPathForTask(YOLOTask task) {
    switch (task) {
      case YOLOTask.detect:
        return 'yolo11n';
      case YOLOTask.segment:
        return 'yolo11n-seg';
      case YOLOTask.classify:
        return 'yolo11n-cls';
      case YOLOTask.pose:
        return 'yolo11n-pose';
      case YOLOTask.obb:
        return 'yolo11n-obb';
    }
  }
}
```

#### ProGuard Rules

```pro
# android/app/proguard-rules.pro
-keep class org.tensorflow.lite.** { *; }
-keep class com.ultralytics.** { *; }
-dontwarn org.tensorflow.**

# Optimize native libraries
-keepclassmembers class * {
    native <methods>;
}
```

## 📱 Device-Specific Optimizations

### High-End Devices (Flagship)

```dart
class HighEndOptimization {
  static bool isHighEndDevice() {
    // Implement device detection logic
    return true; // Placeholder
  }

  static YOLOStreamingConfig getOptimalConfig() {
    return YOLOStreamingConfig(
      includeDetections: true,
      includeMasks: true,           // Enable advanced features
      includePoses: true,
      maxFPS: 25,                   // Higher FPS
      inferenceFrequency: 25,
    );
  }

  static String getOptimalModelPath(YOLOTask task) {
    // Use larger models for better accuracy
    switch (task) {
      case YOLOTask.detect:
        return 'yolo11s';  // Small instead of nano
      default:
        return 'yolo11n';
    }
  }
}
```

### Mid-Range Devices

```dart
class MidRangeOptimization {
  static YOLOStreamingConfig getOptimalConfig() {
    return YOLOStreamingConfig.throttled(
      maxFPS: 15,
      includeMasks: false,          // Disable expensive features
      includeOriginalImage: false,
    );
  }

  static Map<String, double> getOptimalThresholds() {
    return {
      'confidence': 0.5,            // Balanced threshold
      'iou': 0.4,
      'maxItems': 20.0,
    };
  }
}
```

### Low-End Devices (Budget)

```dart
class LowEndOptimization {
  static YOLOStreamingConfig getOptimalConfig() {
    return YOLOStreamingConfig.powerSaving(
      inferenceFrequency: 5,        // Very low frequency
      maxFPS: 10,
    );
  }

  static Map<String, double> getOptimalThresholds() {
    return {
      'confidence': 0.7,            // High threshold for fewer detections
      'iou': 0.2,                   // Fast NMS
      'maxItems': 5.0,              // Minimal detections
    };
  }
}
```

## 🔍 Performance Monitoring

### Real-time Performance Tracking

```dart
class PerformanceTracker {
  final List<double> _fpsHistory = [];
  final List<double> _timeHistory = [];
  final int _maxHistoryLength = 100;

  Timer? _reportTimer;

  void startTracking() {
    _reportTimer = Timer.periodic(Duration(seconds: 5), (_) {
      _generatePerformanceReport();
    });
  }

  void onPerformanceUpdate(YOLOPerformanceMetrics metrics) {
    _fpsHistory.add(metrics.fps);
    _timeHistory.add(metrics.processingTimeMs);

    // Maintain history size
    if (_fpsHistory.length > _maxHistoryLength) {
      _fpsHistory.removeAt(0);
      _timeHistory.removeAt(0);
    }

    // Check for performance issues
    _checkPerformanceIssues(metrics);
  }

  void _checkPerformanceIssues(YOLOPerformanceMetrics metrics) {
    if (metrics.fps < 10) {
      print('⚠️ Low FPS detected: ${metrics.fps.toStringAsFixed(1)}');
      _suggestOptimizations();
    }

    if (metrics.processingTimeMs > 200) {
      print('⚠️ Slow processing: ${metrics.processingTimeMs.toStringAsFixed(1)}ms');
      _suggestOptimizations();
    }

    if (metrics.hasPerformanceIssues) {
      print('⚠️ Performance rating: ${metrics.performanceRating}');
    }
  }

  void _generatePerformanceReport() {
    if (_fpsHistory.isEmpty) return;

    final avgFps = _fpsHistory.reduce((a, b) => a + b) / _fpsHistory.length;
    final avgTime = _timeHistory.reduce((a, b) => a + b) / _timeHistory.length;
    final minFps = _fpsHistory.reduce(math.min);
    final maxTime = _timeHistory.reduce(math.max);

    print('📊 Performance Report:');
    print('  Average FPS: ${avgFps.toStringAsFixed(1)}');
    print('  Average Time: ${avgTime.toStringAsFixed(1)}ms');
    print('  Min FPS: ${minFps.toStringAsFixed(1)}');
    print('  Max Time: ${maxTime.toStringAsFixed(1)}ms');
    print('  Performance: ${_getOverallRating(avgFps, avgTime)}');
  }

  String _getOverallRating(double avgFps, double avgTime) {
    if (avgFps >= 20 && avgTime <= 80) return 'Excellent ⭐⭐⭐⭐⭐';
    if (avgFps >= 15 && avgTime <= 120) return 'Good ⭐⭐⭐⭐';
    if (avgFps >= 10 && avgTime <= 160) return 'Fair ⭐⭐⭐';
    return 'Poor ⭐⭐';
  }

  void _suggestOptimizations() {
    print('💡 Optimization suggestions:');
    print('  • Try a smaller model (yolo11n)');
    print('  • Increase confidence threshold');
    print('  • Reduce maxFPS in streaming config');
    print('  • Disable masks and poses if not needed');
  }

  void stopTracking() {
    _reportTimer?.cancel();
  }
}
```

### Memory Usage Monitoring

```dart
class MemoryMonitor {
  Timer? _monitorTimer;
  int _peakMemoryMB = 0;

  void startMonitoring() {
    _monitorTimer = Timer.periodic(Duration(seconds: 2), (_) {
      _checkMemoryUsage();
    });
  }

  void _checkMemoryUsage() {
    // Platform-specific memory checking would go here
    // This is a simplified example
    final estimatedMemory = _estimateCurrentMemoryUsage();

    if (estimatedMemory > _peakMemoryMB) {
      _peakMemoryMB = estimatedMemory;
    }

    if (estimatedMemory > 400) { // > 400MB
      print('⚠️ High memory usage: ${estimatedMemory}MB');
      _suggestMemoryOptimizations();
    }
  }

  int _estimateCurrentMemoryUsage() {
    // Simplified estimation based on active instances
    final activeInstances = YOLOInstanceManager.getActiveInstanceIds().length;
    return 100 + (activeInstances * 150); // Base + per instance
  }

  void _suggestMemoryOptimizations() {
    print('🔋 Memory optimization suggestions:');
    print('  • Dispose unused YOLO instances');
    print('  • Use smaller models');
    print('  • Limit concurrent instances');
    print('  • Disable original image in streaming');
  }

  void stopMonitoring() {
    _monitorTimer?.cancel();
  }
}
```

## 🎯 Performance Best Practices

### 1. Model Management

- **Use nano models** for real-time applications
- **Cache loaded models** instead of reloading
- **Dispose unused instances** promptly
- **Limit concurrent instances** to 2-3 maximum

### 2. Streaming Optimization

- **Choose appropriate FPS** based on use case
- **Disable unnecessary features** (masks, poses, original image)
- **Use throttling** for battery-sensitive applications
- **Monitor performance metrics** continuously

### 3. Memory Management

- **Monitor memory usage** in production
- **Implement memory pressure handling**
- **Use weak references** where possible
- **Clean up resources** in dispose methods

### 4. Platform Optimization

- **Enable GPU acceleration** on both platforms
- **Use appropriate build configurations**
- **Optimize for target devices**
- **Test on real devices** not emulators

### 5. Code Patterns

```dart
// ✅ Good: Reuse instances
class GoodPattern {
  late final YOLO _yolo;

  Future<void> init() async {
    _yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
    await _yolo.loadModel();
  }

  Future<List<dynamic>> predict(Uint8List image) async {
    final results = await _yolo.predict(image);
    return results['boxes'];
  }
}

// ❌ Bad: Create new instances repeatedly
class BadPattern {
  Future<List<dynamic>> predict(Uint8List image) async {
    final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
    await yolo.loadModel(); // Expensive!
    final results = await yolo.predict(image);
    await yolo.dispose();
    return results['boxes'];
  }
}
```

## 📈 Performance Testing

### Benchmark Your App

```dart
class PerformanceBenchmark {
  Future<Map<String, dynamic>> runBenchmark(
    String modelPath,
    List<Uint8List> testImages,
  ) async {
    final yolo = YOLO(modelPath: modelPath, task: YOLOTask.detect);
    await yolo.loadModel();

    final List<double> inferenceTimes = [];
    final stopwatch = Stopwatch();

    for (final image in testImages) {
      stopwatch.reset();
      stopwatch.start();

      await yolo.predict(image);

      stopwatch.stop();
      inferenceTimes.add(stopwatch.elapsedMilliseconds.toDouble());
    }

    await yolo.dispose();

    final avgTime = inferenceTimes.reduce((a, b) => a + b) / inferenceTimes.length;
    final minTime = inferenceTimes.reduce(math.min);
    final maxTime = inferenceTimes.reduce(math.max);

    return {
      'model': modelPath,
      'samples': testImages.length,
      'avg_time_ms': avgTime,
      'min_time_ms': minTime,
      'max_time_ms': maxTime,
      'avg_fps': 1000 / avgTime,
      'times': inferenceTimes,
    };
  }
}
```

This performance guide provides comprehensive strategies for optimizing YOLO Flutter applications. For implementation details, see the [Usage Guide](usage.md), and for troubleshooting performance issues, check the [Troubleshooting Guide](troubleshooting.md).
