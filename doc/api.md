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
    YOLOTask? task,
    bool useGpu = true,
    bool useMultiInstance = false,
    Map<String, dynamic>? classifierOptions,
    int? numItemsThreshold,
  });
}
```

#### Constructor Parameters

| Parameter           | Type                    | Required | Default | Description                                                                                                 |
| ------------------- | ----------------------- | -------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `modelPath`         | `String`                | ✅       | -       | Official model ID, local path, asset path, or URL                                                           |
| `task`              | `YOLOTask?`             | ❌       | `null`  | Type of YOLO task to perform when metadata is missing                                                       |
| `useGpu`            | `bool`                  | ❌       | `true`  | Allow GPU acceleration on Android (LiteRT 2.x GPU → CPU ladder); iOS uses Core ML. Set `false` to force CPU |
| `useMultiInstance`  | `bool`                  | ❌       | `false` | Enable multi-instance support                                                                               |
| `classifierOptions` | `Map<String, dynamic>?` | ❌       | `null`  | Optional classifier preprocessing and label overrides                                                       |
| `numItemsThreshold` | `int?`                  | ❌       | `30`    | Maximum number of returned detections                                                                       |

#### Properties

| Property        | Type        | Description                                        |
| --------------- | ----------- | -------------------------------------------------- |
| `instanceId`    | `String`    | Unique identifier for this YOLO instance           |
| `isInitialized` | `bool`      | Whether the model has been loaded                  |
| `modelPath`     | `String`    | Original model reference passed to the constructor |
| `task`          | `YOLOTask?` | Requested task type, if provided                   |
| `useGpu`        | `bool`      | Whether GPU acceleration is enabled                |

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
final yolo = YOLO(modelPath: 'yolo26n');
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
| `iouThreshold`        | `double?`   | ❌       | `0.7`   | IoU threshold for NMS (0.0-1.0) |

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
Future<void> switchModel(String newModelPath, [YOLOTask? newTask])
```

**Parameters**:

| Parameter      | Type        | Description                                          |
| -------------- | ----------- | ---------------------------------------------------- |
| `newModelPath` | `String`    | Path to the new model file                           |
| `newTask`      | `YOLOTask?` | Task type for the new model when metadata is missing |

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

###### `officialModels()`

List official model IDs that are downloadable on the current platform.

```dart
static List<String> officialModels({YOLOTask? task})
```

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

###### `withClassifierOptions()`

Create a YOLO instance configured for classification models that need custom preprocessing.

```dart
static YOLO withClassifierOptions({
  required String modelPath,
  YOLOTask? task,
  required Map<String, dynamic> classifierOptions,
  bool useGpu = true,
  bool useMultiInstance = false,
})
```

###### `inspectModel()`

Read exported metadata for a model without loading it for inference.

```dart
static Future<Map<String, dynamic>> inspectModel(String modelPath)
```

---

### YOLOTask Enum

Defines the type of YOLO task to perform.

```dart
enum YOLOTask {
  detect,      // Object detection
  segment,     // Instance segmentation
  semantic,    // Semantic segmentation
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
    this.task,
    this.controller,
    this.cameraResolution = "720p",
    this.onResult,
    this.onPerformanceMetrics,
    this.onStreamingData,
    this.onZoomChanged,
    this.streamingConfig,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.7,
    this.useGpu = true,
    this.lensFacing = LensFacing.back,
  }) : super(key: key);
}
```

#### Constructor Parameters

| Parameter              | Type                                | Required | Default           | Description                                                                               |
| ---------------------- | ----------------------------------- | -------- | ----------------- | ----------------------------------------------------------------------------------------- |
| `modelPath`            | `String`                            | ✅       | -                 | Official model ID, local path, asset path, or URL                                         |
| `task`                 | `YOLOTask?`                         | ❌       | `null`            | YOLO task type when metadata is missing                                                   |
| `controller`           | `YOLOViewController?`               | ❌       | `null`            | Custom view controller                                                                    |
| `cameraResolution`     | `String`                            | ❌       | `"720p"`          | Camera resolution                                                                         |
| `onResult`             | `Function(List<YOLOResult>)?`       | ❌       | `null`            | Detection results callback                                                                |
| `onPerformanceMetrics` | `Function(YOLOPerformanceMetrics)?` | ❌       | `null`            | Performance metrics callback                                                              |
| `onStreamingData`      | `Function(Map<String, dynamic>)?`   | ❌       | `null`            | Comprehensive streaming callback                                                          |
| `onZoomChanged`        | `Function(double)?`                 | ❌       | `null`            | Zoom level change callback                                                                |
| `streamingConfig`      | `YOLOStreamingConfig?`              | ❌       | `null`            | Streaming configuration                                                                   |
| `confidenceThreshold`  | `double`                            | ❌       | `0.25`            | Initial confidence threshold for YOLOView                                                 |
| `iouThreshold`         | `double`                            | ❌       | `0.7`             | Initial IoU threshold for YOLOView                                                        |
| `useGpu`               | `bool`                              | ❌       | `true`            | Allow GPU acceleration on Android (LiteRT 2.x GPU → CPU ladder); set `false` to force CPU |
| `lensFacing`           | `LensFacing`                        | ❌       | `LensFacing.back` | Initial camera lens selection                                                             |

`LensFacing.backWide` prefers the shortest-focal-length rear camera on Android and falls back to the default back camera when the device does not expose a wide rear lens. Other platforms treat it as `LensFacing.back`.

#### Example

```dart
// Basic usage
YOLOView(
  modelPath: 'yolo26n',
  controller: controller,
  onResult: (results) {
    print('Detections: ${results.length}');
  },
)
```

#### 0.4.0 Migration Notes

`YOLOView` no longer accepts `showOverlays`, `overlayTheme`, or `showNativeUI`. Camera overlay drawing is native-only, and package-provided controls moved out of `YOLOView`.

| Removed API                                | Use instead                                                                                          |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `YOLOOverlay`, `YOLOOverlayTheme`          | Native `YOLOView` overlays, or raw `onResult`/`YOLO.predict()` data for fully custom rendering.      |
| `YOLOControls`                             | `YOLOShowcase` for the full UI, or exported widgets such as `TaskSegmentedControl` and `LensPicker`. |
| `YOLOView.showNativeUI`                    | `YOLOShowcase` for built-in controls; bare `YOLOView` plus your own Flutter controls for custom UI.  |
| `YOLOView.showOverlays`, `overlayTheme`    | No constructor replacement. Native camera overlays are not themed or toggled from Dart.              |
| `setShowUIControls()`, `setShowOverlays()` | Own the surrounding Flutter controls; `capturePhoto(withOverlays: false)` only affects captures.     |

---

### YOLOViewController Class

Controller for managing YOLOView behavior and settings.

```dart
class YOLOViewController {
  YOLOViewController();
}
```

#### Properties

| Property              | Type             | Description                                                         |
| --------------------- | ---------------- | ------------------------------------------------------------------- |
| `confidenceThreshold` | `double`         | Current confidence threshold (0.0-1.0)                              |
| `iouThreshold`        | `double`         | Current IoU threshold (0.0-1.0)                                     |
| `numItemsThreshold`   | `int`            | Maximum number of detections (1-100)                                |
| `isInitialized`       | `bool`           | Whether controller is initialized                                   |
| `zoomEvents`          | `Stream<double>` | Broadcast stream of zoom-factor changes emitted by the native layer |
| `lensEvents`          | `Stream<String>` | Broadcast stream of lens-switch events emitted by the native layer  |
| `focusEvents`         | `Stream<Offset>` | Broadcast stream of tap-to-focus coordinates (view-relative)        |

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

##### `setTorchMode()`

Turn the active camera torch on or off when supported.

```dart
Future<void> setTorchMode(bool enabled)
```

**Parameters**: `enabled` - `true` to enable the torch, `false` to disable it

##### `switchModel()`

Dynamically switch to a different model without restarting the camera.

```dart
Future<void> switchModel(String modelPath, [YOLOTask? task])
```

Parameters:

- `modelPath`: Official model ID, local path, asset path, or URL
- `task`: The YOLO task type when metadata is missing

**Throws**:

- `PlatformException` - If model file cannot be found or loaded

**Note**: This method uses the same model resolver as `YOLO` and `YOLOView`, so it supports official IDs, asset paths, local files, remote URLs, and metadata-based task resolution.

Example:

```dart
// Switch to a custom model
await controller.switchModel(
  'assets/models/custom.tflite',
  YOLOTask.detect,
);

// Handle errors
try {
  await controller.switchModel('new_model.tflite');
} catch (e) {
  print('Failed to load model: $e');
}
```

##### `zoomIn()`

Increase camera zoom by one step.

```dart
Future<void> zoomIn()
```

##### `zoomOut()`

Decrease camera zoom by one step.

```dart
Future<void> zoomOut()
```

##### `setZoomLevel()`

Set the camera zoom directly.

```dart
Future<void> setZoomLevel(double zoomLevel)
```

##### `stop()`

Stop the active camera session.

```dart
Future<void> stop()
```

##### `restartCamera()`

Restart the camera session after stopping it.

```dart
Future<void> restartCamera()
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
- Instance segmentation masks (for segment task)
- Semantic segmentation masks (for semantic task)
- Pose keypoints and skeleton (for pose task)
- OBB rotated boxes (for OBB task)
- Classification results (for classify task)

##### `capturePhoto()`

Capture a composited JPEG of the current camera frame, optionally with native detection overlays drawn in.

```dart
Future<Uint8List?> capturePhoto({bool withOverlays = true})
```

**Parameters**: `withOverlays` - Include native detection overlays in the output JPEG (default `true`)

**Returns**: `Future<Uint8List?>` - JPEG image data, or `null` if capture fails

##### `getAvailableLenses()`

Return the list of physical camera lenses available on the device.

```dart
Future<List<LensInfo>> getAvailableLenses()
```

**Returns**: `Future<List<LensInfo>>` - Each entry carries a `zoomFactor` (the lens's approximate optical zoom relative to the main sensor) and a human-readable `label`.

##### `setLens()`

Switch to the physical lens whose zoom factor is nearest to the requested value.

```dart
Future<void> setLens(double zoomFactor)
```

**Parameters**: `zoomFactor` - Target zoom factor; the nearest available lens is selected.

##### `tapToFocus()`

Request a focus/exposure lock at the given view-relative coordinates.

```dart
Future<void> tapToFocus(double x, double y)
```

**Parameters**:

- `x` - Horizontal position in the range 0.0 (left) to 1.0 (right)
- `y` - Vertical position in the range 0.0 (top) to 1.0 (bottom)

##### `pause()`

Pause the active camera session. On iOS the last frame is kept frozen so `capturePhoto` returns that frame; on Android this is an alias for `stop()`.

```dart
Future<void> pause()
```

##### `resume()`

Resume after `pause()`. On iOS the cached share frame is cleared and the session restarts; on Android this is an alias for `restartCamera()`.

```dart
Future<void> resume()
```

---

### YOLOShowcase Widget

A Material 3 one-import camera screen that mirrors the layout of the native Ultralytics YOLO iOS app. All 9 exported UI widgets are composed automatically.

```dart
YOLOShowcase(
  modelPath: 'yolo26n',
  onCapture: (bytes) {},
)
```

#### Constructor Parameters

| Parameter   | Type                    | Required | Default | Description                                        |
| ----------- | ----------------------- | -------- | ------- | -------------------------------------------------- |
| `modelPath` | `String`                | ✅       | -       | Official model ID, local path, asset path, or URL  |
| `onCapture` | `Function(Uint8List?)?` | ❌       | `null`  | Callback invoked with the JPEG bytes after capture |

Import once and get the full UI:

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

YOLOShowcase(modelPath: 'yolo26n')
```

For a custom layout, compose the 9 exported Material widgets around a bare `YOLOView` instead: `TaskSegmentedControl`, `ModelSizeSegmentedControl`, `ThresholdSliderRow`, `LensPicker`, `ZoomIndicator`, `CameraToolbar`, `FocusReticle`, `LogoOverlay`, `PerformanceLabel`.

---

### YOLOModelManager Class

Static class that manages model downloads and caching.

#### Static Properties

##### `downloadProgress`

A broadcast `Stream<DownloadProgress>` that emits fractional progress (0.0–1.0) while an official model asset is downloading.

```dart
static Stream<DownloadProgress> get downloadProgress
```

**Example**:

```dart
YOLOModelManager.downloadProgress.listen((progress) {
  print('Download: ${(progress.fraction * 100).toStringAsFixed(0)}%');
});
```

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
  final List<List<double>>? mask;
  final List<Point>? keypoints;
  final double? angle;
}
```

#### Properties

| Property        | Type                  | Description                              |
| --------------- | --------------------- | ---------------------------------------- |
| `classIndex`    | `int`                 | Class index in the model                 |
| `className`     | `String`              | Human-readable class name                |
| `confidence`    | `double`              | Detection confidence (0.0-1.0)           |
| `boundingBox`   | `Rect`                | Bounding box in pixel coordinates        |
| `normalizedBox` | `Rect`                | Normalized bounding box (0.0-1.0)        |
| `mask`          | `List<List<double>>?` | Instance mask data (segment task only)   |
| `keypoints`     | `List<Point>?`        | Pose keypoints (pose task only)          |
| `angle`         | `double?`             | OBB rotation angle in radians (OBB only) |

---

Single-image semantic segmentation returns `YOLODetectionResults.semanticMask` with a row-major `classMap`, `width`, and `height`.

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
const double DEFAULT_IOU_THRESHOLD = 0.7;
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

This API reference covers all public interfaces in the YOLO Flutter plugin. For usage examples, see the [Usage Guide](usage.md), and for performance optimization, check the [Performance Guide](performance.md).
