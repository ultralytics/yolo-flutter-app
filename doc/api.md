---
title: API Reference
description: Complete API documentation for Ultralytics YOLO Flutter plugin - classes, methods, and parameters
path: /integrations/flutter/api/
---

# API Reference

Complete reference documentation for all classes, methods, and parameters in the Ultralytics YOLO Flutter plugin.

## 📚 Core Classes

### YOLO Class

The main class for YOLO model operations.

```dart
class YOLO {
  YOLO({
    required String modelPath,
    required YOLOTask task,
    bool useMultiInstance = false,
  });
}
```

#### Constructor Parameters

| Parameter          | Type       | Required | Default | Description                           |
| ------------------ | ---------- | -------- | ------- | ------------------------------------- |
| `modelPath`        | `String`   | ✅       | -       | Path to the YOLO model file (.tflite) |
| `task`             | `YOLOTask` | ✅       | -       | Type of YOLO task to perform          |
| `useMultiInstance` | `bool`     | ❌       | `false` | Enable multi-instance support         |

#### Properties

| Property     | Type       | Description                              |
| ------------ | ---------- | ---------------------------------------- |
| `instanceId` | `String`   | Unique identifier for this YOLO instance |
| `modelPath`  | `String`   | Path to the loaded model file            |
| `task`       | `YOLOTask` | Current task type                        |

#### Methods

##### `loadModel()`

Load the YOLO model for inference.

```dart
Future<bool> loadModel()
```

**Returns**: `Future<bool>` - `true` if model loaded successfully

**Throws**:

- `ModelLoadingException` - If model file cannot be found or loaded
- `PlatformException` - If platform-specific error occurs

**Example**:

```dart
final yolo = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);
final success = await yolo.loadModel();
if (success) {
  print('Model loaded successfully');
}
```

##### `predict()`

Run inference on an image.

```dart
Future<Map<String, dynamic>> predict(
  Uint8List imageBytes, {
  double? confidenceThreshold,
  double? iouThreshold,
})
```

**Parameters**:

| Parameter             | Type        | Required | Default | Description                     |
| --------------------- | ----------- | -------- | ------- | ------------------------------- |
| `imageBytes`          | `Uint8List` | ✅       | -       | Raw image data                  |
| `confidenceThreshold` | `double?`   | ❌       | `0.25`  | Confidence threshold (0.0-1.0)  |
| `iouThreshold`        | `double?`   | ❌       | `0.4`   | IoU threshold for NMS (0.0-1.0) |

**Returns**: `Future<Map<String, dynamic>>` - Prediction results

**Throws**:

- `ModelNotLoadedException` - If model not loaded
- `InvalidInputException` - If input parameters invalid
- `InferenceException` - If inference fails

**Example**:

```dart
final imageBytes = await File('image.jpg').readAsBytes();
final results = await yolo.predict(
  imageBytes,
  confidenceThreshold: 0.6,
  iouThreshold: 0.5,
);
```

##### `switchModel()`

Switch to a different model (requires viewId to be set).

```dart
Future<void> switchModel(String newModelPath, YOLOTask newTask)
```

**Parameters**:

| Parameter      | Type       | Description                 |
| -------------- | ---------- | --------------------------- |
| `newModelPath` | `String`   | Path to the new model file  |
| `newTask`      | `YOLOTask` | Task type for the new model |

**Throws**:

- `StateError` - If view not initialized
- `ModelLoadingException` - If model switch fails

##### `dispose()`

Release all resources and clean up the instance.

```dart
Future<void> dispose()
```

**Example**:

```dart
await yolo.dispose();
```

##### Static Methods

###### `checkModelExists()`

Check if a model file exists at the specified path.

```dart
static Future<Map<String, dynamic>> checkModelExists(String modelPath)
```

**Returns**: Map containing existence info and location details

###### `getStoragePaths()`

Get available storage paths for the app.

```dart
static Future<Map<String, String?>> getStoragePaths()
```

**Returns**: Map of storage location names to paths

---

### YOLOTask Enum

Defines the type of YOLO task to perform.

```dart
enum YOLOTask {
  detect,      // Object detection
  segment,     // Instance segmentation
  classify,    // Image classification
  pose,        // Pose estimation
  obb,         // Oriented bounding boxes
}
```

#### Usage

```dart
final task = YOLOTask.detect;
print(task.name); // "detect"
```

---

### YOLOView Widget

Real-time camera view with YOLO processing.

```dart
class YOLOView extends StatefulWidget {
  const YOLOView({
    Key? key,
    required this.modelPath,
    required this.task,
    this.controller,
    this.onResult,
    this.onPerformanceMetrics,
    this.onStreamingData,
    this.onZoomChanged,
    this.cameraResolution = "720p",
    this.showNativeUI = true,
    this.streamingConfig,
  }) : super(key: key);
}
```

#### Constructor Parameters

| Parameter              | Type                                | Required | Default  | Description                                             |
| ---------------------- | ----------------------------------- | -------- | -------- | ------------------------------------------------------- |
| `modelPath`            | `String`                            | ✅       | -        | Path to YOLO model file (camera starts even if invalid) |
| `task`                 | `YOLOTask`                          | ✅       | -        | YOLO task type                                          |
| `controller`           | `YOLOViewController?`               | ❌       | `null`   | Custom view controller                                  |
| `onResult`             | `Function(List<YOLOResult>)?`       | ❌       | `null`   | Detection results callback                              |
| `onPerformanceMetrics` | `Function(YOLOPerformanceMetrics)?` | ❌       | `null`   | Performance metrics callback                            |
| `onStreamingData`      | `Function(Map<String, dynamic>)?`   | ❌       | `null`   | Comprehensive streaming callback                        |
| `onZoomChanged`        | `Function(double)?`                 | ❌       | `null`   | Zoom level change callback                              |
| `cameraResolution`     | `String`                            | ❌       | `"720p"` | Camera resolution                                       |
| `showNativeUI`         | `bool`                              | ❌       | `true`   | Show native camera UI                                   |
| `streamingConfig`      | `YOLOStreamingConfig?`              | ❌       | `null`   | Streaming configuration                                 |

#### Example

```dart
// Basic usage with valid model
YOLOView(
  modelPath: 'assets/models/yolo11n.tflite',
  task: YOLOTask.detect,
  onResult: (results) {
    print('Detected ${results.length} objects');
  },
  onPerformanceMetrics: (metrics) {
    print('FPS: ${metrics.fps}');
  },
)

// Camera-only mode (v0.1.25+): starts even with invalid model path
YOLOView(
  modelPath: 'model_not_yet_downloaded.tflite',  // Model doesn't exist yet
  task: YOLOTask.detect,
  controller: controller,
  onResult: (results) {
    // Will receive empty results until model is loaded
    print('Detections: ${results.length}');
  },
)

// Later, load the model dynamically
await controller.switchModel('downloaded_model.tflite', YOLOTask.detect);
```

---

### YOLOViewController Class

Controller for managing YOLOView behavior and settings.

```dart
class YOLOViewController {
  YOLOViewController();
}
```

#### Properties

| Property              | Type     | Description                            |
| --------------------- | -------- | -------------------------------------- |
| `confidenceThreshold` | `double` | Current confidence threshold (0.0-1.0) |
| `iouThreshold`        | `double` | Current IoU threshold (0.0-1.0)        |
| `numItemsThreshold`   | `int`    | Maximum number of detections (1-100)   |
| `isInitialized`       | `bool`   | Whether controller is initialized      |

#### Methods

##### `setConfidenceThreshold()`

Set the confidence threshold for detections.

```dart
Future<void> setConfidenceThreshold(double threshold)
```

**Parameters**: `threshold` - Value between 0.0 and 1.0

##### `setIoUThreshold()`

Set the IoU threshold for non-maximum suppression.

```dart
Future<void> setIoUThreshold(double threshold)
```

**Parameters**: `threshold` - Value between 0.0 and 1.0

##### `setNumItemsThreshold()`

Set the maximum number of detections to return.

```dart
Future<void> setNumItemsThreshold(int threshold)
```

**Parameters**: `threshold` - Value between 1 and 100

##### `setThresholds()`

Set multiple thresholds at once.

```dart
Future<void> setThresholds({
  double? confidenceThreshold,
  double? iouThreshold,
  int? numItemsThreshold,
})
```

##### `switchCamera()`

Switch between front and back camera.

```dart
Future<void> switchCamera()
```

##### `switchModel()`

Dynamically switch to a different model without restarting the camera.

```dart
Future<void> switchModel(String modelPath, YOLOTask task)
```

Parameters:

- `modelPath`: Path to the new model file
- `task`: The YOLO task type for the new model

**Throws**:

- `PlatformException` - If model file cannot be found or loaded

**Note**: As of v0.1.25, YOLOView can start with an invalid model path (camera-only mode). Use this method to load a valid model later.

Example:

```dart
// Switch to a different model
await controller.switchModel('yolo11s', YOLOTask.detect);

// Platform-specific paths
await controller.switchModel(
  Platform.isIOS ? 'yolo11s' : 'yolo11s.tflite',
  YOLOTask.detect,
);

// Handle errors
try {
  await controller.switchModel('new_model.tflite', YOLOTask.detect);
} catch (e) {
  print('Failed to load model: $e');
}
```

##### `setStreamingConfig()`

Configure streaming behavior.

```dart
Future<void> setStreamingConfig(YOLOStreamingConfig config)
```

##### `captureFrame()`

Capture the current camera frame with detection overlays.

```dart
Future<Uint8List?> captureFrame()
```

**Returns**: `Future<Uint8List?>` - JPEG image data with detection overlays, or `null` if capture fails

**Description**: Captures the current camera frame including all detection visualizations (bounding boxes, masks, keypoints, etc.) as a JPEG image.

**Example**:

```dart
// Capture frame with overlays
final imageData = await controller.captureFrame();

if (imageData != null) {
  // Save to file
  final directory = await getTemporaryDirectory();
  final file = File('${directory.path}/capture_${DateTime.now().millisecondsSinceEpoch}.jpg');
  await file.writeAsBytes(imageData);

  // Or display in UI
  showDialog(
    context: context,
    builder: (context) => Image.memory(imageData),
  );
}
```

**Note**: The captured image includes:

- Camera frame
- Detection bounding boxes with labels
- Segmentation masks (for segment task)
- Pose keypoints and skeleton (for pose task)
- OBB rotated boxes (for OBB task)
- Classification results (for classify task)

---

### YOLOResult Class

Represents a single detection result.

```dart
class YOLOResult {
  final int classIndex;
  final String className;
  final double confidence;
  final Rect boundingBox;
  final Rect normalizedBox;
  final List<Offset>? keypoints;
  final Uint8List? mask;
}
```

#### Properties

| Property        | Type            | Description                           |
| --------------- | --------------- | ------------------------------------- |
| `classIndex`    | `int`           | Class index in the model              |
| `className`     | `String`        | Human-readable class name             |
| `confidence`    | `double`        | Detection confidence (0.0-1.0)        |
| `boundingBox`   | `Rect`          | Bounding box in pixel coordinates     |
| `normalizedBox` | `Rect`          | Normalized bounding box (0.0-1.0)     |
| `keypoints`     | `List<Offset>?` | Pose keypoints (pose task only)       |
| `mask`          | `Uint8List?`    | Segmentation mask (segment task only) |

---

### YOLOPerformanceMetrics Class

Performance metrics for YOLO inference.

```dart
class YOLOPerformanceMetrics {
  final double fps;
  final double processingTimeMs;
  final int frameNumber;
  final DateTime timestamp;
}
```

#### Properties

| Property           | Type       | Description                     |
| ------------------ | ---------- | ------------------------------- |
| `fps`              | `double`   | Frames per second               |
| `processingTimeMs` | `double`   | Processing time in milliseconds |
| `frameNumber`      | `int`      | Current frame number            |
| `timestamp`        | `DateTime` | Timestamp of the measurement    |

#### Methods

##### `isGoodPerformance`

Check if performance meets good thresholds.

```dart
bool get isGoodPerformance
```

**Returns**: `true` if FPS ≥ 15 and processing time ≤ 100ms

##### `hasPerformanceIssues`

Check if there are performance issues.

```dart
bool get hasPerformanceIssues
```

**Returns**: `true` if FPS < 10 or processing time > 200ms

##### `performanceRating`

Get a performance rating string.

```dart
String get performanceRating
```

**Returns**: "Excellent", "Good", "Fair", or "Poor"

##### Factory Constructors

###### `fromMap()`

Create metrics from a map.

```dart
factory YOLOPerformanceMetrics.fromMap(Map<String, dynamic> map)
```

#### Example

```dart
onPerformanceMetrics: (metrics) {
  print('Performance: ${metrics.performanceRating}');
  print('FPS: ${metrics.fps.toStringAsFixed(1)}');
  print('Processing: ${metrics.processingTimeMs.toStringAsFixed(1)}ms');

  if (metrics.hasPerformanceIssues) {
    print('⚠️ Performance issues detected');
  }
}
```

---

### YOLOStreamingConfig Class

Configuration for real-time streaming behavior.

```dart
class YOLOStreamingConfig {
  const YOLOStreamingConfig({
    this.includeDetections = true,
    this.includeClassifications = true,
    this.includeProcessingTimeMs = true,
    this.includeFps = true,
    this.includeMasks = false,
    this.includePoses = false,
    this.includeOBB = false,
    this.includeOriginalImage = false,
    this.maxFPS,
    this.throttleInterval,
    this.inferenceFrequency,
    this.skipFrames,
  });
}
```

#### Properties

| Property                  | Type        | Default | Description                      |
| ------------------------- | ----------- | ------- | -------------------------------- |
| `includeDetections`       | `bool`      | `true`  | Include detection results        |
| `includeClassifications`  | `bool`      | `true`  | Include classification results   |
| `includeProcessingTimeMs` | `bool`      | `true`  | Include processing time          |
| `includeFps`              | `bool`      | `true`  | Include FPS metrics              |
| `includeMasks`            | `bool`      | `false` | Include segmentation masks       |
| `includePoses`            | `bool`      | `false` | Include pose keypoints           |
| `includeOBB`              | `bool`      | `false` | Include oriented bounding boxes  |
| `includeOriginalImage`    | `bool`      | `false` | Include original frame data      |
| `maxFPS`                  | `int?`      | `null`  | Maximum FPS limit                |
| `throttleInterval`        | `Duration?` | `null`  | Throttling interval              |
| `inferenceFrequency`      | `int?`      | `null`  | Inference frequency (per second) |
| `skipFrames`              | `int?`      | `null`  | Number of frames to skip         |

#### Factory Constructors

##### `minimal()`

Minimal streaming configuration for best performance.

```dart
factory YOLOStreamingConfig.minimal()
```

##### `withMasks()`

Configuration including segmentation masks.

```dart
factory YOLOStreamingConfig.withMasks()
```

##### `full()`

Full configuration with all features except original image.

```dart
factory YOLOStreamingConfig.full()
```

##### `debug()`

Debug configuration including original image data.

```dart
factory YOLOStreamingConfig.debug()
```

##### `throttled()`

Throttled configuration with FPS limiting.

```dart
factory YOLOStreamingConfig.throttled({
  required int maxFPS,
  bool includeMasks = false,
  bool includePoses = false,
  int? inferenceFrequency,
  int? skipFrames,
})
```

##### `powerSaving()`

Power-saving configuration with reduced frequency.

```dart
factory YOLOStreamingConfig.powerSaving({
  int inferenceFrequency = 10,
  int maxFPS = 15,
})
```

##### `highPerformance()`

High-performance configuration for maximum throughput.

```dart
factory YOLOStreamingConfig.highPerformance({
  int inferenceFrequency = 30,
})
```

#### Example

```dart
// Power-saving configuration
final config = YOLOStreamingConfig.powerSaving(
  inferenceFrequency: 10,
  maxFPS: 15,
);

// Custom configuration
final customConfig = YOLOStreamingConfig(
  includeDetections: true,
  includeMasks: true,
  maxFPS: 20,
  skipFrames: 2,
);
```

---

### YOLOInstanceManager Class

Static class for managing multiple YOLO instances.

```dart
class YOLOInstanceManager {
  // Static methods only
}
```

#### Static Methods

##### `registerInstance()`

Register a YOLO instance.

```dart
static void registerInstance(String instanceId, YOLO instance)
```

##### `unregisterInstance()`

Unregister a YOLO instance.

```dart
static void unregisterInstance(String instanceId)
```

##### `getInstance()`

Get a registered YOLO instance.

```dart
static YOLO? getInstance(String instanceId)
```

##### `hasInstance()`

Check if an instance is registered.

```dart
static bool hasInstance(String instanceId)
```

##### `getActiveInstanceIds()`

Get list of all active instance IDs.

```dart
static List<String> getActiveInstanceIds()
```

#### Example

```dart
// Create multi-instance YOLO
final yolo = YOLO(
  modelPath: 'model.tflite',
  task: YOLOTask.detect,
  useMultiInstance: true,
);

// Check instance registration
print('Instance registered: ${YOLOInstanceManager.hasInstance(yolo.instanceId)}');
print('Active instances: ${YOLOInstanceManager.getActiveInstanceIds().length}');
```

---

## 🚨 Exception Classes

### YOLOException

Base exception class for all YOLO-related errors.

```dart
class YOLOException implements Exception {
  final String message;
  const YOLOException(this.message);
}
```

### ModelLoadingException

Thrown when model loading fails.

```dart
class ModelLoadingException extends YOLOException {
  const ModelLoadingException(String message) : super(message);
}
```

### ModelNotLoadedException

Thrown when attempting to use an unloaded model.

```dart
class ModelNotLoadedException extends YOLOException {
  const ModelNotLoadedException(String message) : super(message);
}
```

### InferenceException

Thrown when inference fails.

```dart
class InferenceException extends YOLOException {
  const InferenceException(String message) : super(message);
}
```

### InvalidInputException

Thrown when invalid input is provided.

```dart
class InvalidInputException extends YOLOException {
  const InvalidInputException(String message) : super(message);
}
```

---

## 📊 Type Definitions

### Common Types

```dart
// Callback function types
typedef YOLOResultCallback = void Function(List<YOLOResult> results);
typedef YOLOPerformanceCallback = void Function(YOLOPerformanceMetrics metrics);
typedef YOLOStreamingCallback = void Function(Map<String, dynamic> data);
typedef YOLOZoomCallback = void Function(double zoomLevel);

// Result data types
typedef DetectionBox = Map<String, dynamic>;
typedef ClassificationResult = Map<String, dynamic>;
typedef PoseKeypoints = List<Map<String, dynamic>>;
```

---

## 🔧 Constants

### Default Values

```dart
// Default thresholds
const double DEFAULT_CONFIDENCE_THRESHOLD = 0.25;
const double DEFAULT_IOU_THRESHOLD = 0.4;
const int DEFAULT_NUM_ITEMS_THRESHOLD = 30;

// Performance thresholds
const double GOOD_PERFORMANCE_FPS = 15.0;
const double GOOD_PERFORMANCE_TIME_MS = 100.0;
const double PERFORMANCE_ISSUE_FPS = 10.0;
const double PERFORMANCE_ISSUE_TIME_MS = 200.0;

// Camera resolutions
const List<String> SUPPORTED_RESOLUTIONS = [
  "480p", "720p", "1080p", "4K"
];
```

---

## 🎯 Migration Guide

### From v0.1.15 to v0.1.18+

#### Multi-Instance Support

**Old (Single Instance)**:

```dart
final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
```

**New (Multi-Instance)**:

```dart
final yolo = YOLO(
  modelPath: 'model.tflite',
  task: YOLOTask.detect,
  useMultiInstance: true, // Add this line
);
```

#### Streaming Configuration

**New Feature**:

```dart
YOLOView(
  modelPath: 'model.tflite',
  task: YOLOTask.detect,
  streamingConfig: YOLOStreamingConfig.throttled(maxFPS: 15), // New
  onStreamingData: (data) { /* New comprehensive callback */ },
)
```

---

This API reference covers all public interfaces in the YOLO Flutter plugin. For usage examples, see the [Usage Guide](usage.md), and for performance optimization, check the [Performance Guide](performance.md).
