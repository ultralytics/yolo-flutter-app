# Performance Optimization Guide

This guide provides comprehensive performance optimization strategies for YOLO Flutter applications, including benchmarks, device-specific tuning, and production deployment best practices.

## 🎯 Performance Overview

YOLO Flutter delivers real-time performance across devices, but optimization is key for production applications.

### Target Performance Metrics

| Device Tier                               | Target FPS | Target Latency | Model Recommendation |
| ----------------------------------------- | ---------- | -------------- | -------------------- |
| **High-end** (iPhone 14 Pro, Galaxy S23+) | 25-30 FPS  | <35ms          | YOLO11s, YOLO11m     |
| **Mid-range** (iPhone 12, Galaxy A54)     | 15-25 FPS  | <50ms          | YOLO11n, YOLO11s     |
| **Budget** (iPhone SE, Galaxy A34)        | 10-20 FPS  | <75ms          | YOLO11n only         |

## 📊 Benchmarks by Device & Model

### iOS Devices (Core ML)

| Model       | iPhone 14 Pro | iPhone 13     | iPhone 12     | iPhone SE 3rd |
| ----------- | ------------- | ------------- | ------------- | ------------- |
| **YOLO11n** | 30 FPS / 25ms | 28 FPS / 28ms | 25 FPS / 35ms | 20 FPS / 45ms |
| **YOLO11s** | 25 FPS / 35ms | 22 FPS / 40ms | 18 FPS / 50ms | 12 FPS / 75ms |
| **YOLO11m** | 20 FPS / 45ms | 16 FPS / 55ms | 12 FPS / 75ms | 8 FPS / 120ms |

### Android Devices (TensorFlow Lite)

| Model       | Galaxy S23 Ultra | Pixel 7 Pro   | Galaxy A54    | Galaxy A34    |
| ----------- | ---------------- | ------------- | ------------- | ------------- |
| **YOLO11n** | 28 FPS / 28ms    | 25 FPS / 32ms | 20 FPS / 45ms | 15 FPS / 60ms |
| **YOLO11s** | 22 FPS / 40ms    | 20 FPS / 45ms | 15 FPS / 60ms | 10 FPS / 90ms |
| **YOLO11m** | 18 FPS / 50ms    | 15 FPS / 60ms | 10 FPS / 90ms | 6 FPS / 150ms |

_Benchmarks measured with 640x640 input resolution, detection task_

## ⚡ Model Selection Strategy

Choose the right model for your target devices and use case:

### By Use Case

```dart
// Real-time security/surveillance
class SecurityApp {
  String getOptimalModel() {
    return 'yolo11n';  // Prioritize speed over accuracy
  }

  YOLOStreamingConfig getConfig() {
    return YOLOStreamingConfig.custom(
      maxFPS: 30,
      inferenceFrequency: 25,
      confidenceThreshold: 0.7,  // Higher threshold for security
    );
  }
}

// Photo/document analysis
class PhotoAnalysisApp {
  String getOptimalModel() {
    return 'yolo11s';  // Balance of speed and accuracy
  }

  YOLOStreamingConfig getConfig() {
    return YOLOStreamingConfig.custom(
      maxFPS: 15,
      inferenceFrequency: 10,
      confidenceThreshold: 0.5,
    );
  }
}

// High-precision industrial inspection
class IndustrialApp {
  String getOptimalModel() {
    return 'yolo11m';  // Prioritize accuracy
  }

  YOLOStreamingConfig getConfig() {
    return YOLOStreamingConfig.custom(
      maxFPS: 10,
      inferenceFrequency: 8,
      confidenceThreshold: 0.3,  // Lower threshold for detection
    );
  }
}
```

### By Device Performance

```dart
class ModelSelector {
  static Future<String> selectOptimalModel() async {
    final deviceInfo = await DeviceInfo.getInstance();
    final memoryGB = deviceInfo.totalMemoryGB;
    final cpuCores = deviceInfo.cpuCores;
    final isHighEnd = deviceInfo.isHighEndDevice;

    if (isHighEnd && memoryGB >= 6) {
      return 'yolo11m';  // High-end devices can handle larger models
    } else if (memoryGB >= 4 && cpuCores >= 6) {
      return 'yolo11s';  // Mid-range devices
    } else {
      return 'yolo11n';  // Budget devices
    }
  }
}
```

## 🔧 Optimization Techniques

### 1. Inference Frequency Optimization

Reduce computational load by controlling inference frequency:

```dart
// Adaptive inference frequency based on motion detection
class AdaptiveInference extends StatefulWidget {
  @override
  _AdaptiveInferenceState createState() => _AdaptiveInferenceState();
}

class _AdaptiveInferenceState extends State<AdaptiveInference> {
  final YOLOViewController _controller = YOLOViewController();
  int _motionLevel = 0;  // 0=static, 1=slow, 2=fast
  int _baseFrequency = 15;

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      controller: _controller,
      modelPath: 'assets/yolo11n.tflite',
      task: YOLOTask.detect,
      streamingConfig: _getAdaptiveConfig(),
      onResult: (results) {
        _analyzeMotion(results);
      },
    );
  }

  YOLOStreamingConfig _getAdaptiveConfig() {
    int frequency = _baseFrequency;

    switch (_motionLevel) {
      case 0: // Static scene - reduce frequency
        frequency = _baseFrequency ~/ 3;
        break;
      case 1: // Slow motion - normal frequency
        frequency = _baseFrequency;
        break;
      case 2: // Fast motion - increase frequency
        frequency = (_baseFrequency * 1.5).round();
        break;
    }

    return YOLOStreamingConfig.custom(
      inferenceFrequency: frequency,
      maxFPS: 30,
    );
  }

  void _analyzeMotion(List<YOLOResult> results) {
    // Implement motion detection logic
    // Update _motionLevel based on object movement
  }
}
```

### 2. Resolution Optimization

Adjust input resolution based on device capabilities:

```dart
class ResolutionOptimizer {
  static Future<Size> getOptimalResolution() async {
    final deviceInfo = await DeviceInfo.getInstance();

    if (deviceInfo.isHighEndDevice) {
      return Size(640, 640);  // Full resolution
    } else if (deviceInfo.isMidRangeDevice) {
      return Size(480, 480);  // Reduced resolution
    } else {
      return Size(320, 320);  // Minimal resolution
    }
  }
}

// Usage in model export/preprocessing
void exportOptimizedModels() {
  final resolutions = [
    Size(320, 320),  // Budget devices
    Size(480, 480),  // Mid-range devices
    Size(640, 640),  // High-end devices
  ];

  for (final res in resolutions) {
    // Export models with different input sizes
    // model.export(format="tflite", imgsz=[res.width, res.height])
  }
}
```

### 3. Memory Management

Optimize memory usage for sustained performance:

```dart
class MemoryOptimizedYOLO extends StatefulWidget {
  @override
  _MemoryOptimizedYOLOState createState() => _MemoryOptimizedYOLOState();
}

class _MemoryOptimizedYOLOState extends State<MemoryOptimizedYOLO>
    with WidgetsBindingObserver {
  final YOLOViewController _controller = YOLOViewController();
  bool _isAppInBackground = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    switch (state) {
      case AppLifecycleState.paused:
      case AppLifecycleState.inactive:
        // Reduce performance when app is backgrounded
        _controller.updateStreamingConfig(
          YOLOStreamingConfig.custom(inferenceFrequency: 1)
        );
        _isAppInBackground = true;
        break;
      case AppLifecycleState.resumed:
        // Restore performance when app is foregrounded
        if (_isAppInBackground) {
          _controller.updateStreamingConfig(
            YOLOStreamingConfig.balanced()
          );
          _isAppInBackground = false;
        }
        break;
      default:
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      controller: _controller,
      modelPath: 'assets/yolo11n.tflite',
      task: YOLOTask.detect,
      streamingConfig: YOLOStreamingConfig.custom(
        bufferSize: 1,  // Minimal buffering
        dropFramesWhenBusy: true,  // Prevent memory buildup
      ),
    );
  }
}
```

### 4. Battery Optimization

Implement battery-aware performance scaling:

```dart
import 'package:battery_plus/battery_plus.dart';

class BatteryAwareYOLO extends StatefulWidget {
  @override
  _BatteryAwareYOLOState createState() => _BatteryAwareYOLOState();
}

class _BatteryAwareYOLOState extends State<BatteryAwareYOLO> {
  final YOLOViewController _controller = YOLOViewController();
  final Battery _battery = Battery();
  YOLOStreamingConfig _currentConfig = YOLOStreamingConfig.balanced();

  @override
  void initState() {
    super.initState();
    _monitorBattery();
  }

  void _monitorBattery() {
    _battery.onBatteryStateChanged.listen((BatteryState state) {
      _updateConfigForBatteryState(state);
    });

    // Also monitor battery level
    Timer.periodic(Duration(minutes: 1), (timer) async {
      final level = await _battery.batteryLevel;
      _updateConfigForBatteryLevel(level);
    });
  }

  void _updateConfigForBatteryState(BatteryState state) {
    YOLOStreamingConfig newConfig;

    switch (state) {
      case BatteryState.charging:
        newConfig = YOLOStreamingConfig.full();  // Maximum performance
        break;
      case BatteryState.full:
        newConfig = YOLOStreamingConfig.balanced();
        break;
      default:
        newConfig = YOLOStreamingConfig.minimal();  // Power saving
        break;
    }

    if (newConfig != _currentConfig) {
      setState(() => _currentConfig = newConfig);
      _controller.updateStreamingConfig(newConfig);
    }
  }

  void _updateConfigForBatteryLevel(int batteryLevel) {
    if (batteryLevel < 20 && _currentConfig != YOLOStreamingConfig.minimal()) {
      // Switch to power saving mode when battery is low
      setState(() => _currentConfig = YOLOStreamingConfig.minimal());
      _controller.updateStreamingConfig(_currentConfig);
    }
  }

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      controller: _controller,
      modelPath: 'assets/yolo11n.tflite',
      task: YOLOTask.detect,
      streamingConfig: _currentConfig,
    );
  }
}
```

## 📱 Platform-Specific Optimizations

### iOS Optimizations

```dart
class iOSOptimizedYOLO {
  // Leverage Core ML optimizations
  static YOLOStreamingConfig getiOSConfig() {
    return YOLOStreamingConfig.custom(
      // iOS handles Core ML scheduling automatically
      useSystemOptimizations: true,

      // Core ML benefits from consistent inference frequency
      inferenceFrequency: 20,

      // iOS GPUs handle higher resolutions well
      preferHighResolution: true,
    );
  }

  // Use iOS-specific model optimizations
  static String getiOSModel(String baseModel) {
    // Prefer .mlpackage for iOS 13+
    return Platform.isIOS ? '$baseModel.mlpackage' : '$baseModel.tflite';
  }
}
```

### Android Optimizations

```dart
class AndroidOptimizedYOLO {
  // Optimize for TensorFlow Lite
  static YOLOStreamingConfig getAndroidConfig() {
    return YOLOStreamingConfig.custom(
      // Android benefits from GPU acceleration
      useGPUAcceleration: true,

      // TensorFlow Lite can handle variable frequency well
      adaptiveInferenceFrequency: true,

      // Optimize for diverse hardware
      useHardwareDetection: true,
    );
  }

  // Use Android-specific optimizations
  static Future<String> getOptimizedModel() async {
    final info = await DeviceInfoPlugin().androidInfo;

    // Use NNAPI-optimized models on supported devices
    if (info.version.sdkInt >= 27) {
      return 'yolo11n_nnapi.tflite';
    }
    return 'yolo11n.tflite';
  }
}
```

## 🎛️ Production Deployment

### Performance Monitoring

Implement comprehensive performance monitoring for production:

```dart
class ProductionYOLO extends StatefulWidget {
  @override
  _ProductionYOLOState createState() => _ProductionYOLOState();
}

class _ProductionYOLOState extends State<ProductionYOLO> {
  final YOLOViewController _controller = YOLOViewController();
  final PerformanceMonitor _monitor = PerformanceMonitor();

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      controller: _controller,
      modelPath: 'assets/yolo11n.tflite',
      task: YOLOTask.detect,
      streamingConfig: YOLOStreamingConfig.balanced(),
      onPerformanceMetrics: (metrics) {
        _monitor.recordMetrics(metrics);

        // Auto-adjust if performance degrades
        if (metrics.fps != null && metrics.fps! < 10) {
          _handlePerformanceDegradation();
        }
      },
    );
  }

  void _handlePerformanceDegradation() {
    // Step down performance automatically
    _controller.updateStreamingConfig(YOLOStreamingConfig.minimal());

    // Log for analytics
    _monitor.logPerformanceIssue('FPS dropped below threshold');
  }
}

class PerformanceMonitor {
  final List<PerformanceMetrics> _metrics = [];

  void recordMetrics(PerformanceMetrics metrics) {
    _metrics.add(metrics);

    // Keep only recent metrics
    if (_metrics.length > 100) {
      _metrics.removeAt(0);
    }

    // Send to analytics if needed
    _sendToAnalytics(metrics);
  }

  void _sendToAnalytics(PerformanceMetrics metrics) {
    // Send performance data to your analytics service
    Analytics.track('yolo_performance', {
      'fps': metrics.fps,
      'processing_time_ms': metrics.processingTimeMs,
      'device_model': DeviceInfo.model,
      'app_version': AppInfo.version,
    });
  }

  double get averageFPS {
    if (_metrics.isEmpty) return 0;
    return _metrics.map((m) => m.fps ?? 0).reduce((a, b) => a + b) / _metrics.length;
  }

  void logPerformanceIssue(String issue) {
    Analytics.track('yolo_performance_issue', {
      'issue': issue,
      'average_fps': averageFPS,
      'metrics_count': _metrics.length,
    });
  }
}
```

### A/B Testing Performance

Test different configurations with real users:

```dart
class PerformanceABTest {
  static YOLOStreamingConfig getConfigForUser(String userId) {
    final variant = _getABTestVariant(userId);

    switch (variant) {
      case 'high_performance':
        return YOLOStreamingConfig.custom(
          inferenceFrequency: 25,
          maxFPS: 30,
        );
      case 'balanced':
        return YOLOStreamingConfig.balanced();
      case 'power_saving':
        return YOLOStreamingConfig.minimal();
      default:
        return YOLOStreamingConfig.balanced();
    }
  }

  static String _getABTestVariant(String userId) {
    // Simple hash-based assignment
    final hash = userId.hashCode.abs();
    final variants = ['high_performance', 'balanced', 'power_saving'];
    return variants[hash % variants.length];
  }
}
```

## 🔍 Performance Debugging

### Identifying Bottlenecks

```dart
class PerformanceDebugger {
  static void analyzePerformance(List<PerformanceMetrics> metrics) {
    final avgFps = metrics.map((m) => m.fps ?? 0).reduce((a, b) => a + b) / metrics.length;
    final avgProcessingTime = metrics.map((m) => m.processingTimeMs ?? 0).reduce((a, b) => a + b) / metrics.length;

    print('=== Performance Analysis ===');
    print('Average FPS: ${avgFps.toStringAsFixed(1)}');
    print('Average Processing Time: ${avgProcessingTime.toStringAsFixed(1)}ms');

    if (avgFps < 15) {
      print('❌ Low FPS detected - consider:');
      print('  • Reducing inference frequency');
      print('  • Using smaller model (yolo11n)');
      print('  • Reducing input resolution');
    }

    if (avgProcessingTime > 100) {
      print('❌ High processing time - consider:');
      print('  • Enabling GPU acceleration');
      print('  • Using quantized models');
      print('  • Reducing model complexity');
    }

    _checkMemoryUsage();
    _checkCPUUsage();
  }

  static void _checkMemoryUsage() {
    // Monitor memory usage patterns
    print('Memory usage analysis...');
  }

  static void _checkCPUUsage() {
    // Monitor CPU usage patterns
    print('CPU usage analysis...');
  }
}
```

## 📋 Performance Checklist

### Pre-Production Checklist

- [ ] **Model Selection**: Right model for target devices
- [ ] **Inference Frequency**: Optimized for use case
- [ ] **Memory Management**: No memory leaks or excessive usage
- [ ] **Battery Impact**: Reasonable power consumption
- [ ] **Device Testing**: Tested on representative device range
- [ ] **Performance Monitoring**: Analytics and alerting in place
- [ ] **Graceful Degradation**: Handles performance issues automatically
- [ ] **User Controls**: Allow users to adjust quality settings

### Optimization Priority

1. **Choose appropriate model size** (biggest impact)
2. **Optimize inference frequency** (easy wins)
3. **Implement adaptive quality** (best user experience)
4. **Add performance monitoring** (production readiness)
5. **Fine-tune for specific devices** (advanced optimization)

## 🔗 Related Documentation

- **[Streaming Guide](./streaming.md)** - Real-time processing and inference control
- **[Getting Started](./getting-started.md)** - Basic setup and configuration
- **[API Reference](./api-reference.md)** - Complete technical documentation
- **[Troubleshooting](./troubleshooting.md)** - Performance issue resolution
