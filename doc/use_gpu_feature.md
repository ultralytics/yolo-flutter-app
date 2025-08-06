# useGpu Feature

## Overview

The `useGpu` feature allows you to control GPU acceleration for YOLO model inference. This is particularly useful for devices where GPU inference causes stability issues or crashes.

## Problem Statement

On Android, the plugin previously forced `useGpu = true` in the native implementation, and the Dart API didn't allow configuring it. On several devices, using GPU makes the app crash during model initialization — even with the sample yolo11n.tflite and custom models. This doesn't happen on iOS due to Core ML's automatic GPU/CPU selection.

## Solution

The `useGpu` parameter has been exposed through the public API, allowing developers to:

1. **Disable GPU on problematic devices** - Fall back to CPU/NNAPI for stability
2. **Control performance vs stability** - Choose between faster GPU inference or more stable CPU inference
3. **Handle device-specific issues** - Dynamically adjust based on device capabilities

## API Changes

### YOLO Class

```dart
// Constructor now accepts useGpu parameter
final yolo = YOLO(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  useGpu: false, // Disable GPU for stability
);

// withClassifierOptions also supports useGpu
final classifier = YOLO.withClassifierOptions(
  modelPath: 'assets/models/classifier.tflite',
  task: YOLOTask.classify,
  classifierOptions: options,
  useGpu: false, // Disable GPU
);
```

### YOLOView Widget

```dart
YOLOView(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  useGpu: false, // Disable GPU for stability
  onResult: (results) {
    // Handle results
  },
);
```

## Platform Implementation

### Android

- **TensorFlow Lite GPU Delegate**: When `useGpu = true`, uses GPU delegate for acceleration
- **CPU/NNAPI Fallback**: When `useGpu = false`, uses CPU or NNAPI for inference
- **Stability**: CPU mode is more stable on devices with problematic GPU drivers

### iOS

- **Core ML GPU**: When `useGpu = true`, uses `MLModelConfiguration.computeUnits = .all`
- **Core ML CPU**: When `useGpu = false`, uses `MLModelConfiguration.computeUnits = .cpuOnly`
- **Automatic Selection**: Core ML normally handles GPU/CPU selection automatically, but this gives explicit control

## Usage Examples

### Basic Usage

```dart
// Enable GPU (default behavior)
final yolo = YOLO(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  useGpu: true, // Default value
);

// Disable GPU for stability
final yolo = YOLO(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  useGpu: false, // Use CPU for stability
);
```

### Device-Specific Configuration

```dart
class DeviceAwareDetector extends StatefulWidget {
  @override
  State<DeviceAwareDetector> createState() => _DeviceAwareDetectorState();
}

class _DeviceAwareDetectorState extends State<DeviceAwareDetector> {
  bool _useGpu = true;

  @override
  void initState() {
    super.initState();
    _configureForDevice();
  }

  void _configureForDevice() {
    // Disable GPU on known problematic devices
    final deviceModel = Platform.isAndroid
        ? android_info.androidId
        : ios_info.utsname.machine;

    final problematicDevices = [
      'SM-G973F', // Samsung Galaxy S10
      'SM-G975F', // Samsung Galaxy S10+
      // Add more problematic device models
    ];

    if (problematicDevices.contains(deviceModel)) {
      setState(() {
        _useGpu = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      modelPath: 'assets/models/yolo11n.tflite',
      task: YOLOTask.detect,
      useGpu: _useGpu,
      onResult: (results) {
        // Handle results
      },
    );
  }
}
```

## Performance Considerations

### GPU Mode (useGpu = true)

- **Pros**: Faster inference, better for real-time applications
- **Cons**: May cause crashes on some devices, higher power consumption

### CPU Mode (useGpu = false)

- **Pros**: More stable, works on all devices, lower power consumption
- **Cons**: Slower inference, may not be suitable for real-time applications

## Migration Guide

### From Previous Version

If you're upgrading from a previous version, no changes are required. The default behavior remains the same (`useGpu = true`).

### Adding GPU Control

To add GPU control to existing code:

```dart
// Before
final yolo = YOLO(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
);

// After (explicit GPU control)
final yolo = YOLO(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  useGpu: false, // Add this line to disable GPU
);
```

### Debug Information

Enable debug logging to see GPU configuration:

```dart
// Android logs will show:
// "Model loaded successfully: model.tflite for task: DETECT, useGpu: false"

// iOS logs will show:
// "Model loaded with useGpu: false"
```
