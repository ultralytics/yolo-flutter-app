# API Reference

Complete technical reference for all YOLO Flutter classes, methods, and configurations.

## Core Classes

### YOLOView

The main widget for displaying YOLO detection results with camera feed.

```dart
YOLOView({
  required String modelPath,
  required YOLOTask task,
  YOLOViewController? controller,
  String cameraResolution = '720p',
  YOLOStreamingConfig? streamingConfig,
  Function(List<YOLOResult>)? onResult,
  Function(PerformanceMetrics)? onPerformanceMetrics,
  Function(double)? onZoomChanged,
  bool showNativeUI = false,
})
```

#### Parameters

| Parameter              | Type                            | Required | Default      | Description                                 |
| ---------------------- | ------------------------------- | -------- | ------------ | ------------------------------------------- |
| `modelPath`            | `String`                        | ✅       | -            | Path to YOLO model file                     |
| `task`                 | `YOLOTask`                      | ✅       | -            | YOLO task type (detect, segment, etc.)      |
| `controller`           | `YOLOViewController?`           | ❌       | `null`       | Controller for managing view settings       |
| `cameraResolution`     | `String`                        | ❌       | `'720p'`     | Camera resolution ('480p', '720p', '1080p') |
| `streamingConfig`      | `YOLOStreamingConfig?`          | ❌       | `balanced()` | Real-time streaming configuration           |
| `onResult`             | `Function(List<YOLOResult>)?`   | ❌       | `null`       | Callback for detection results              |
| `onPerformanceMetrics` | `Function(PerformanceMetrics)?` | ❌       | `null`       | Callback for performance data               |
| `onZoomChanged`        | `Function(double)?`             | ❌       | `null`       | Callback for zoom level changes             |
| `showNativeUI`         | `bool`                          | ❌       | `false`      | Show native platform UI overlays            |

#### Example

```dart
YOLOView(
  modelPath: 'yolo11n',
  task: YOLOTask.detect,
  controller: myController,
  cameraResolution: '720p',
  streamingConfig: YOLOStreamingConfig.balanced(),
  onResult: (results) {
    print('Detected ${results.length} objects');
  },
  onPerformanceMetrics: (metrics) {
    print('FPS: ${metrics.fps}');
  },
  onZoomChanged: (zoom) {
    print('Zoom: ${zoom}x');
  },
)
```

---

### YOLOViewController

Controller for managing YOLOView settings and camera controls.

#### Properties

```dart
class YOLOViewController {
  double get confidenceThreshold;
  double get iouThreshold;
  int get numItemsThreshold;
  double get currentZoom;
  bool get isCameraActive;
}
```

#### Methods

##### setConfidenceThreshold()

```dart
Future<void> setConfidenceThreshold(double threshold)
```

Sets the confidence threshold for detections.

**Parameters:**

- `threshold` (double): Confidence threshold (0.0-1.0)

**Example:**

```dart
await controller.setConfidenceThreshold(0.6);
```

##### setIoUThreshold()

```dart
Future<void> setIoUThreshold(double threshold)
```

Sets the IoU (Intersection over Union) threshold for non-maximum suppression.

**Parameters:**

- `threshold` (double): IoU threshold (0.0-1.0)

**Example:**

```dart
await controller.setIoUThreshold(0.5);
```

##### setThresholds()

```dart
Future<void> setThresholds({
  double? confidenceThreshold,
  double? iouThreshold,
  int? numItemsThreshold,
})
```

Sets multiple thresholds at once.

**Example:**

```dart
await controller.setThresholds(
  confidenceThreshold: 0.6,
  iouThreshold: 0.5,
  numItemsThreshold: 25,
);
```

##### switchCamera()

```dart
Future<void> switchCamera()
```

Switches between front and back camera.

##### setZoomLevel()

```dart
Future<void> setZoomLevel(double zoom)
```

Sets camera zoom level.

**Parameters:**

- `zoom` (double): Zoom level (1.0 = no zoom, 2.0 = 2x zoom)

##### switchModel()

```dart
Future<void> switchModel(String modelPath, YOLOTask task)
```

Dynamically switches to a different model.

**Parameters:**

- `modelPath` (String): Path to new model
- `task` (YOLOTask): Task type for new model

##### updateStreamingConfig()

```dart
Future<void> updateStreamingConfig(YOLOStreamingConfig config)
```

Updates streaming configuration at runtime.

---

### YOLO

Main class for single-image inference (non-real-time).

```dart
YOLO({
  required String modelPath,
  required YOLOTask task,
})
```

#### Methods

##### loadModel()

```dart
Future<void> loadModel()
```

Loads the YOLO model for inference.

##### predict()

```dart
Future<List<YOLOResult>> predict(
  Uint8List imageBytes, {
  double? confidenceThreshold,
  double? iouThreshold,
})
```

Runs inference on a single image.

**Parameters:**

- `imageBytes` (Uint8List): Image data as bytes
- `confidenceThreshold` (double?): Optional confidence threshold
- `iouThreshold` (double?): Optional IoU threshold

**Returns:** List of detection results

#### Static Methods

##### checkModelExists()

```dart
static Future<bool> checkModelExists(String modelPath)
```

Checks if a model file exists at the specified path.

##### getStoragePaths()

```dart
static Future<List<String>> getStoragePaths()
```

Returns available storage paths for models.

---

### YOLOResult

Contains detection/inference results.

```dart
class YOLOResult {
  final int classIndex;
  final String className;
  final double confidence;
  final Rect boundingBox;
  final List<List<double>>? mask;
  final List<Point>? keypoints;
  final double? processingTimeMs;
  final double? fps;
}
```

#### Properties

| Property           | Type                  | Description                           |
| ------------------ | --------------------- | ------------------------------------- |
| `classIndex`       | `int`                 | Numeric class identifier              |
| `className`        | `String`              | Human-readable class name             |
| `confidence`       | `double`              | Detection confidence (0.0-1.0)        |
| `boundingBox`      | `Rect`                | Object bounding rectangle             |
| `mask`             | `List<List<double>>?` | Segmentation mask (segmentation only) |
| `keypoints`        | `List<Point>?`        | Pose keypoints (pose estimation only) |
| `processingTimeMs` | `double?`             | Processing time for this frame        |
| `fps`              | `double?`             | Current frames per second             |

#### Example

```dart
void handleResults(List<YOLOResult> results) {
  for (final result in results) {
    print('Class: ${result.className}');
    print('Confidence: ${result.confidence.toStringAsFixed(3)}');
    print('BBox: ${result.boundingBox}');

    if (result.mask != null) {
      print('Mask size: ${result.mask!.length}x${result.mask!.first.length}');
    }

    if (result.keypoints != null) {
      print('Keypoints: ${result.keypoints!.length}');
    }
  }
}
```

---

### YOLOStreamingConfig

Configuration for real-time streaming and performance control.

#### Constructors

##### YOLOStreamingConfig.full()

```dart
YOLOStreamingConfig.full()
```

High-performance configuration with all features enabled.

- Max FPS: 30
- Inference Frequency: 25 Hz
- All data types enabled

##### YOLOStreamingConfig.balanced()

```dart
YOLOStreamingConfig.balanced()
```

Balanced performance configuration for most use cases.

- Max FPS: 20
- Inference Frequency: 15 Hz
- Essential data types enabled

##### YOLOStreamingConfig.minimal()

```dart
YOLOStreamingConfig.minimal()
```

Power-saving configuration for basic detection.

- Max FPS: 15
- Inference Frequency: 8 Hz
- Minimal data types

##### YOLOStreamingConfig.withMasks()

```dart
YOLOStreamingConfig.withMasks()
```

Optimized for segmentation tasks.

- Mask data enabled
- Balanced performance for segmentation

##### YOLOStreamingConfig.withPoses()

```dart
YOLOStreamingConfig.withPoses()
```

Optimized for pose estimation tasks.

- Pose data enabled
- Optimized for motion tracking

##### YOLOStreamingConfig.custom()

```dart
YOLOStreamingConfig.custom({
  int? maxFPS,
  int? inferenceFrequency,
  int? skipFrames,
  bool includeBoundingBoxes = true,
  bool includeMasks = false,
  bool includePoses = false,
  bool includeClassifications = false,
  bool includeOBB = false,
  int bufferSize = 2,
  bool dropFramesWhenBusy = false,
  CallbackPriority callbackPriority = CallbackPriority.balanced,
})
```

Custom configuration with fine-grained control.

#### Parameters

| Parameter                | Type               | Default    | Description                         |
| ------------------------ | ------------------ | ---------- | ----------------------------------- |
| `maxFPS`                 | `int?`             | `20`       | Maximum output frame rate           |
| `inferenceFrequency`     | `int?`             | `15`       | Inference frequency in Hz           |
| `skipFrames`             | `int?`             | `null`     | Skip N frames between inferences    |
| `includeBoundingBoxes`   | `bool`             | `true`     | Include bounding box data           |
| `includeMasks`           | `bool`             | `false`    | Include segmentation masks          |
| `includePoses`           | `bool`             | `false`    | Include pose keypoints              |
| `includeClassifications` | `bool`             | `false`    | Include classification scores       |
| `includeOBB`             | `bool`             | `false`    | Include oriented bounding boxes     |
| `bufferSize`             | `int`              | `2`        | Internal buffer size                |
| `dropFramesWhenBusy`     | `bool`             | `false`    | Drop frames when processing is slow |
| `callbackPriority`       | `CallbackPriority` | `balanced` | Callback priority mode              |

#### Example

```dart
final config = YOLOStreamingConfig.custom(
  maxFPS: 30,
  inferenceFrequency: 20,
  includeMasks: true,
  includePoses: false,
  bufferSize: 1,
  dropFramesWhenBusy: true,
  callbackPriority: CallbackPriority.performance,
);
```

---

### PerformanceMetrics

Performance monitoring data.

```dart
class PerformanceMetrics {
  final double? fps;
  final double? processingTimeMs;
  final int? inferenceFrequency;
  final double? memoryUsageMB;
  final double? cpuUsagePercent;
  final double? gpuUsagePercent;
  final int? droppedFrames;
}
```

#### Properties

| Property             | Type      | Description                     |
| -------------------- | --------- | ------------------------------- |
| `fps`                | `double?` | Current frames per second       |
| `processingTimeMs`   | `double?` | Processing time per frame (ms)  |
| `inferenceFrequency` | `int?`    | Actual inference frequency (Hz) |
| `memoryUsageMB`      | `double?` | Memory usage in megabytes       |
| `cpuUsagePercent`    | `double?` | CPU usage percentage            |
| `gpuUsagePercent`    | `double?` | GPU usage percentage            |
| `droppedFrames`      | `int?`    | Number of dropped frames        |

---

## Enums

### YOLOTask

YOLO task types.

```dart
enum YOLOTask {
  detect,     // Object detection
  segment,    // Instance segmentation
  classify,   // Image classification
  pose,       // Pose estimation
  obb,        // Oriented bounding box detection
}
```

### CallbackPriority

Callback execution priority modes.

```dart
enum CallbackPriority {
  performance,  // Prioritize speed over data completeness
  balanced,     // Balance speed and completeness
  accuracy,     // Prioritize data completeness over speed
}
```

---

## Platform-Specific APIs

### Android-Specific

#### TensorFlow Lite Configuration

```dart
// Android-specific optimizations (conceptual)
class AndroidYOLOConfig {
  final bool useGPUAcceleration;
  final bool useNNAPI;
  final int numThreads;

  const AndroidYOLOConfig({
    this.useGPUAcceleration = true,
    this.useNNAPI = false,
    this.numThreads = 4,
  });
}
```

### iOS-Specific

#### Core ML Configuration

```dart
// iOS-specific optimizations (conceptual)
class iOSYOLOConfig {
  final bool useANE;  // Apple Neural Engine
  final bool allowLowPrecision;
  final MLComputeUnits computeUnits;

  const iOSYOLOConfig({
    this.useANE = true,
    this.allowLowPrecision = true,
    this.computeUnits = MLComputeUnits.all,
  });
}
```

---

## Error Handling

### YOLOException

Custom exception for YOLO-specific errors.

```dart
class YOLOException implements Exception {
  final String message;
  final String? code;
  final dynamic details;

  YOLOException(this.message, {this.code, this.details});
}
```

#### Common Error Codes

| Code                 | Description                |
| -------------------- | -------------------------- |
| `MODEL_NOT_FOUND`    | Model file not found       |
| `MODEL_LOAD_FAILED`  | Failed to load model       |
| `INFERENCE_FAILED`   | Inference execution failed |
| `PERMISSION_DENIED`  | Camera permission denied   |
| `UNSUPPORTED_FORMAT` | Unsupported model format   |

#### Error Handling Example

```dart
try {
  final yolo = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);
  await yolo.loadModel();
} on YOLOException catch (e) {
  switch (e.code) {
    case 'MODEL_NOT_FOUND':
      print('Model file not found: ${e.message}');
      break;
    case 'PERMISSION_DENIED':
      print('Camera permission required: ${e.message}');
      break;
    default:
      print('YOLO error: ${e.message}');
  }
} catch (e) {
  print('Unexpected error: $e');
}
```

---

## Utility Functions

### Model Management

```dart
class ModelUtils {
  /// Download a pre-trained model
  static Future<String> downloadModel(String modelName) async {
    // Implementation for downloading models
  }

  /// Get local model path
  static Future<String?> getLocalModelPath(String modelName) async {
    // Implementation for finding local models
  }

  /// Get model info (size, task type, etc.)
  static Future<ModelInfo> getModelInfo(String modelPath) async {
    // Implementation for model metadata
  }
}
```

### Device Information

```dart
class DeviceUtils {
  /// Check if device is high-end
  static Future<bool> isHighEndDevice() async {
    // Implementation for device capability detection
  }

  /// Get optimal model for current device
  static Future<String> getOptimalModel(YOLOTask task) async {
    // Implementation for device-appropriate model selection
  }

  /// Get recommended streaming config
  static Future<YOLOStreamingConfig> getRecommendedConfig() async {
    // Implementation for device-optimized configuration
  }
}
```

---

## Constants

### Default Values

```dart
class YOLODefaults {
  static const double defaultConfidenceThreshold = 0.5;
  static const double defaultIoUThreshold = 0.45;
  static const int defaultNumItemsThreshold = 25;
  static const int defaultMaxFPS = 20;
  static const int defaultInferenceFrequency = 15;
  static const String defaultCameraResolution = '720p';
}
```

### Model Names

```dart
class YOLOModels {
  // Detection models
  static const String yolo11n = 'yolo11n';
  static const String yolo11s = 'yolo11s';
  static const String yolo11m = 'yolo11m';
  static const String yolo11l = 'yolo11l';
  static const String yolo11x = 'yolo11x';

  // Segmentation models
  static const String yolo11nSeg = 'yolo11n-seg';
  static const String yolo11sSeg = 'yolo11s-seg';

  // Pose estimation models
  static const String yolo11nPose = 'yolo11n-pose';
  static const String yolo11sPose = 'yolo11s-pose';

  // Classification models
  static const String yolo11nCls = 'yolo11n-cls';
  static const String yolo11sCls = 'yolo11s-cls';

  // OBB detection models
  static const String yolo11nObb = 'yolo11n-obb';
  static const String yolo11sObb = 'yolo11s-obb';
}
```

---

## Migration Guide

### From Version 0.1.4 to 0.1.5

#### Breaking Changes

1. **YOLOStreamingConfig constructor changes:**

    ```dart
    // Old
    YOLOStreamingConfig(maxFPS: 30, includeStreaming: true)

    // New
    YOLOStreamingConfig.custom(maxFPS: 30, inferenceFrequency: 25)
    ```

2. **Performance callback signature:**

    ```dart
    // Old
    onPerformanceMetrics: (Map<String, double> metrics) { }

    // New
    onPerformanceMetrics: (PerformanceMetrics metrics) { }
    ```

#### New Features

- Inference frequency control
- Enhanced streaming configurations
- Performance metrics class
- Battery-aware optimizations

#### Migration Steps

1. Update streaming configurations to use new constructors
2. Update performance callback to use `PerformanceMetrics` class
3. Consider using new inference frequency controls for better performance

---

This API reference covers the complete public interface of YOLO Flutter. For implementation examples, see the [Examples Guide](./examples.md) and [Getting Started Guide](./getting-started.md).
