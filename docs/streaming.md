# Real-time Streaming & Advanced Processing

This guide covers advanced real-time processing features including inference frequency control, streaming configurations, and performance optimization for production applications.

## 🎮 Streaming Demo

The **[Streaming Test Example](../streaming_test_example/)** demonstrates all advanced features:
- Real-time inference frequency control
- Performance monitoring
- Task switching
- Data availability indicators

```bash
cd streaming_test_example
flutter run
```

## ⚡ Inference Frequency Control

Control how often YOLO runs inference to optimize performance and battery life.

### Basic Configuration

```dart
YOLOView(
  modelPath: 'assets/yolo11n.tflite',
  task: YOLOTask.detect,
  streamingConfig: YOLOStreamingConfig.custom(
    maxFPS: 30,              // Output frame rate cap
    inferenceFrequency: 15,  // Run inference 15 times per second
    includeMasks: true,      // Enable mask data streaming
    includePoses: true,      // Enable pose data streaming
  ),
  onResult: (results) {
    print('Inference ran: ${results.length} objects detected');
  },
)
```

### Pre-configured Options

```dart
// High performance (30 FPS inference)
YOLOStreamingConfig.full()

// Balanced performance (15 FPS inference)  
YOLOStreamingConfig.balanced()

// Power saving (5 FPS inference)
YOLOStreamingConfig.minimal()

// Segmentation optimized
YOLOStreamingConfig.withMasks()

// Pose estimation optimized
YOLOStreamingConfig.withPoses()
```

### Frame Skipping Control

Alternative to frequency-based control using frame skipping:

```dart
YOLOStreamingConfig.custom(
  skipFrames: 3,  // Skip 3 frames between each inference
  maxFPS: 30,     // Still output at 30 FPS
)
```

## 📊 Performance Monitoring

Track real-time performance metrics:

```dart
YOLOView(
  // ... other parameters
  onPerformanceMetrics: (metrics) {
    print('FPS: ${metrics.fps?.toStringAsFixed(1)}');
    print('Processing Time: ${metrics.processingTimeMs?.toStringAsFixed(1)}ms');
    print('Inference Frequency: ${metrics.inferenceFrequency} Hz');
  },
)
```

### Available Metrics

```dart
class PerformanceMetrics {
  final double? fps;                    // Current frames per second
  final double? processingTimeMs;       // Processing time per frame
  final int? inferenceFrequency;        // Actual inference frequency
  final double? memoryUsageMB;          // Memory usage (if available)
  final double? cpuUsagePercent;        // CPU usage (if available)
}
```

## 🎯 Task-Specific Streaming

Different YOLO tasks have different streaming considerations:

### Object Detection
```dart
YOLOView(
  task: YOLOTask.detect,
  streamingConfig: YOLOStreamingConfig.custom(
    maxFPS: 30,
    inferenceFrequency: 25,  // High frequency for smooth tracking
    includeBoundingBoxes: true,
  ),
)
```

### Segmentation
```dart
YOLOView(
  task: YOLOTask.segment,
  streamingConfig: YOLOStreamingConfig.custom(
    maxFPS: 20,
    inferenceFrequency: 15,  // Lower frequency due to complexity
    includeMasks: true,      // Essential for segmentation
    includeBoundingBoxes: true,
  ),
)
```

### Pose Estimation
```dart
YOLOView(
  task: YOLOTask.pose,
  streamingConfig: YOLOStreamingConfig.custom(
    maxFPS: 25,
    inferenceFrequency: 20,  // Smooth for motion tracking
    includePoses: true,      // Essential for pose data
    includeBoundingBoxes: false,  // Optional for pose-only apps
  ),
)
```

## 🔄 Dynamic Configuration

Adjust streaming settings at runtime based on conditions:

```dart
class AdaptiveStreamingExample extends StatefulWidget {
  @override
  _AdaptiveStreamingExampleState createState() => _AdaptiveStreamingExampleState();
}

class _AdaptiveStreamingExampleState extends State<AdaptiveStreamingExample> {
  final YOLOViewController _controller = YOLOViewController();
  YOLOStreamingConfig _config = YOLOStreamingConfig.balanced();
  double _currentFps = 0;
  bool _batteryLow = false;

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      controller: _controller,
      modelPath: 'assets/yolo11n.tflite',
      task: YOLOTask.detect,
      streamingConfig: _config,
      onPerformanceMetrics: (metrics) {
        setState(() {
          _currentFps = metrics.fps ?? 0;
        });
        
        // Auto-adjust based on performance
        _adaptConfiguration(metrics);
      },
      onResult: (results) {
        // Handle results
      },
    );
  }

  void _adaptConfiguration(PerformanceMetrics metrics) {
    // Reduce quality if performance is poor
    if (metrics.fps != null && metrics.fps! < 10 && !_batteryLow) {
      setState(() {
        _config = YOLOStreamingConfig.minimal();
      });
      _controller.updateStreamingConfig(_config);
    }
    
    // Switch to power saving if battery is low
    if (_batteryLow && _config != YOLOStreamingConfig.minimal()) {
      setState(() {
        _config = YOLOStreamingConfig.custom(
          maxFPS: 10,
          inferenceFrequency: 3,
          includeMasks: false,
          includePoses: false,
        );
      });
      _controller.updateStreamingConfig(_config);
    }
  }
}
```

## 📱 Platform-Specific Considerations

### Android Optimization
```dart
// Android benefits from higher inference frequencies
YOLOStreamingConfig.custom(
  maxFPS: 30,
  inferenceFrequency: 25,
  // Android handles GPU processing well
)
```

### iOS Optimization
```dart
// iOS Core ML optimization
YOLOStreamingConfig.custom(
  maxFPS: 25,
  inferenceFrequency: 20,
  // iOS optimizes Core ML automatically
)
```

## 🎛️ Advanced Streaming Features

### Callback Priority System

Control which data to prioritize in real-time streaming:

```dart
YOLOView(
  streamingConfig: YOLOStreamingConfig.custom(
    callbackPriority: CallbackPriority.performance,  // Prioritize speed
    // OR
    callbackPriority: CallbackPriority.accuracy,     // Prioritize data completeness
  ),
)
```

### Buffer Management

Configure internal buffering for smooth playback:

```dart
YOLOStreamingConfig.custom(
  bufferSize: 3,           // Number of frames to buffer
  dropFramesWhenBusy: true, // Drop frames if processing is slow
)
```

### Multi-Threading Control

Control how processing is distributed:

```dart
YOLOStreamingConfig.custom(
  useMultiThreading: true,    // Enable background processing
  maxConcurrentInferences: 2, // Limit concurrent operations
)
```

## 🔧 Troubleshooting Streaming Issues

### Performance Problems

**Symptoms**: Low FPS, stuttering, high CPU usage

**Solutions**:
```dart
// Reduce inference frequency
YOLOStreamingConfig.custom(inferenceFrequency: 10)

// Disable unnecessary features
YOLOStreamingConfig.custom(
  includeMasks: false,
  includePoses: false,
)

// Use smaller models
// Switch from yolo11m to yolo11n
```

### Memory Issues

**Symptoms**: App crashes, memory warnings

**Solutions**:
```dart
// Reduce buffer size
YOLOStreamingConfig.custom(bufferSize: 1)

// Enable frame dropping
YOLOStreamingConfig.custom(dropFramesWhenBusy: true)

// Lower resolution
// Use smaller input images
```

### Battery Drain

**Symptoms**: Rapid battery consumption

**Solutions**:
```dart
// Use power-saving mode
YOLOStreamingConfig.minimal()

// Reduce inference frequency significantly
YOLOStreamingConfig.custom(inferenceFrequency: 5)

// Implement adaptive frequency based on battery level
```

## 📈 Performance Benchmarks

### Typical Performance by Device Class

| Device Class | Model | Max FPS | Recommended Inference Freq |
|--------------|-------|---------|----------------------------|
| High-end (iPhone 14 Pro, Galaxy S23) | yolo11n | 30 | 25-30 |
| High-end (iPhone 14 Pro, Galaxy S23) | yolo11s | 25 | 20-25 |
| Mid-range (iPhone 12, Galaxy A54) | yolo11n | 25 | 15-20 |
| Mid-range (iPhone 12, Galaxy A54) | yolo11s | 20 | 10-15 |
| Budget (iPhone SE, Galaxy A34) | yolo11n | 20 | 10-15 |

### Task-Specific Performance Impact

| Task | Relative Speed | Memory Usage | Recommended Max Inference Freq |
|------|----------------|--------------|-------------------------------|
| Detection | 1x (baseline) | Low | 25-30 Hz |
| Classification | 1.2x (faster) | Very Low | 30+ Hz |
| Segmentation | 0.6x (slower) | High | 15-20 Hz |
| Pose Estimation | 0.8x (slower) | Medium | 20-25 Hz |
| OBB Detection | 0.9x (slower) | Medium | 20-25 Hz |

## 🚀 Production Best Practices

### 1. Start Conservative
```dart
// Begin with balanced settings
YOLOStreamingConfig.balanced()

// Monitor performance and adjust upward
```

### 2. Implement Adaptive Quality
```dart
// Automatically adjust based on device performance
void adaptToDevice() {
  final isHighEnd = await DeviceInfo.isHighEndDevice();
  final config = isHighEnd 
    ? YOLOStreamingConfig.full()
    : YOLOStreamingConfig.minimal();
}
```

### 3. Battery Awareness
```dart
// Monitor battery level and adjust accordingly
void monitorBattery() {
  Battery().onBatteryStateChanged.listen((state) {
    if (state == BatteryState.low) {
      _controller.updateStreamingConfig(YOLOStreamingConfig.minimal());
    }
  });
}
```

### 4. User Control
```dart
// Provide user settings for performance vs quality trade-off
enum QualityPreference { power_saving, balanced, high_quality }

YOLOStreamingConfig configForPreference(QualityPreference pref) {
  switch (pref) {
    case QualityPreference.power_saving:
      return YOLOStreamingConfig.minimal();
    case QualityPreference.balanced:
      return YOLOStreamingConfig.balanced();
    case QualityPreference.high_quality:
      return YOLOStreamingConfig.full();
  }
}
```

## 🔗 Related Documentation

- **[Performance Optimization](./performance.md)** - Device-specific tuning and benchmarks
- **[API Reference](./api-reference.md)** - Complete YOLOStreamingConfig documentation
- **[Examples](./examples.md)** - Real-world streaming implementations
- **[Troubleshooting](./troubleshooting.md)** - Common streaming issues and solutions