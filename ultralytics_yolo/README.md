# Ultralytics YOLO Flutter Package

Flutter plugin for YOLO (You Only Look Once) models, supporting object detection, segmentation, classification, pose estimation and oriented bounding boxes (OBB) on both Android and iOS.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Features

- **Object Detection**: Identify and locate objects in images and camera feeds with bounding boxes
- **Segmentation**: Perform pixel-level segmentation of objects
- **Classification**: Classify objects in images
- **Pose Estimation**: Detect human poses and keypoints
- **Oriented Bounding Boxes (OBB)**: Detect rotated or oriented bounding boxes for objects
- **Cross-Platform**: Works on both Android and iOS
- **Real-time Processing**: Optimized for real-time inference on mobile devices
- **Camera Integration**: Easy integration with device cameras

## Installation

Add this to your package's `pubspec.yaml` file:

```yaml
dependencies:
  ultralytics_yolo: ^0.0.7
```

Then run:

```bash
flutter pub get
```

## Platform-Specific Setup

### Android

Add the following permissions to your `AndroidManifest.xml` file:

```xml
<!-- For camera access -->
<uses-permission android:name="android.permission.CAMERA" />

<!-- For accessing images from storage -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

Set minimum SDK version in your `android/app/build.gradle`:

```gradle
minSdkVersion 21
```

### iOS

Add these entries to your `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to detect objects</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>This app needs photos access to get images for object detection</string>
```

## Usage

### Basic Example

```dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

class YoloDemo extends StatelessWidget {
  // Create a controller to interact with the YoloView
  final controller = YoloViewController();
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('YOLO Object Detection')),
      body: Column(
        children: [
          // Controls for adjusting detection parameters
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Text('Confidence: '),
                Slider(
                  value: 0.5,
                  min: 0.1,
                  max: 0.9,
                  onChanged: (value) {
                    // Update confidence threshold
                    controller.setConfidenceThreshold(value);
                  },
                ),
              ],
            ),
          ),
          
          // YoloView with controller
          Expanded(
            child: YoloView(
              controller: controller,
              task: YOLOTask.detect,
              // Use model name only - recommended approach for cross-platform compatibility
              modelPath: 'yolo11n',
              onResult: (results) {
                // Handle detection results
                print('Detected ${results.length} objects');
              },
            ),
          ),
        ],
      ),
    );
  }
  
  @override
  void initState() {
    super.initState();
    
    // Set initial detection parameters
    controller.setThresholds(
      confidenceThreshold: 0.5,
      iouThreshold: 0.45,
    );
  }
}
```

### Object Detection with Camera Feed

There are three ways to control YoloView's detection parameters:

#### Method 1: Using a Controller (Recommended)

```dart
// Create a controller outside build method
final controller = YoloViewController();

// In your build method:
YoloView(
  controller: controller,  // Provide the controller
  task: YOLOTask.detect,
  modelPath: 'yolo11n',  // Just the model name - most reliable approach
  onResult: (results) {
    for (var result in results) {
      print('Detected: ${result.className}, Confidence: ${result.confidence}');
    }
  },
)

// Set detection parameters anywhere in your code
controller.setConfidenceThreshold(0.5);
controller.setIoUThreshold(0.45);

// Or set both at once
controller.setThresholds(
  confidenceThreshold: 0.5,
  iouThreshold: 0.45,
);
```

#### Method 2: Using GlobalKey Direct Access (Simpler)

```dart
// Create a GlobalKey to access the YoloView
final yoloViewKey = GlobalKey<YoloViewState>();

// In your build method:
YoloView(
  key: yoloViewKey,  // Important: Provide the key
  task: YOLOTask.detect,
  modelPath: 'yolo11n',  // Just the model name without extension
  onResult: (results) {
    for (var result in results) {
      print('Detected: ${result.className}, Confidence: ${result.confidence}');
    }
  },
)

// Set detection parameters directly through the key
yoloViewKey.currentState?.setConfidenceThreshold(0.6);
yoloViewKey.currentState?.setIoUThreshold(0.5);

// Or set both at once
yoloViewKey.currentState?.setThresholds(
  confidenceThreshold: 0.6,
  iouThreshold: 0.5,
);
```

#### Method 3: Automatic Controller (Simplest)

```dart
// No controller needed - just create the view
YoloView(
  task: YOLOTask.detect,
  modelPath: 'yolo11n',  // Simple model name works best across platforms
  onResult: (results) {
    for (var result in results) {
      print('Detected: ${result.className}, Confidence: ${result.confidence}');
    }
  },
)

// A controller is automatically created internally
// with default threshold values (0.5 for confidence, 0.45 for IoU)
```

### Image Segmentation

```dart
// Simplest approach - no controller needed
YoloView(
  task: YOLOTask.segment,
  modelPath: 'yolo11n-seg',  // Model name only, no extension
  onResult: (results) {
    // Process segmentation results
  },
)

// An internal controller is automatically created
// with default thresholds (0.5 confidence, 0.45 IoU)
```

### Pose Estimation

```dart
// Using the GlobalKey approach for direct access
final yoloViewKey = GlobalKey<YoloViewState>();

YoloView(
  key: yoloViewKey,
  task: YOLOTask.pose,
  modelPath: 'yolo11n-pose',  // Only model name, no path or extension
  onResult: (results) {
    // Process pose keypoints
  },
)

// Update parameters directly through the key
yoloViewKey.currentState?.setConfidenceThreshold(0.6);
```

## API Reference

### Classes

#### YOLO

Main class for YOLO operations.

```dart
YOLO({
  required String modelPath,
  required YOLOTask task,
});
```

#### YoloViewController

Controller for interacting with a YoloView, managing settings like thresholds.

```dart
// Create a controller
final controller = YoloViewController();

// Get current values
double confidence = controller.confidenceThreshold;
double iou = controller.iouThreshold;

// Set confidence threshold (0.0-1.0)
await controller.setConfidenceThreshold(0.6);

// Set IoU threshold (0.0-1.0)
await controller.setIoUThreshold(0.5);

// Set both thresholds at once
await controller.setThresholds(
  confidenceThreshold: 0.6,
  iouThreshold: 0.5,
);
```

#### YoloView

Flutter widget to display YOLO detection results.

```dart
YoloView({
  required YOLOTask task,
  required String modelPath,
  YoloViewController? controller,  // Optional: Controller for managing view settings
  Function(List<YOLOResult>)? onResult,
});

// YoloView methods (when accessed via GlobalKey<YoloViewState>)
Future<void> setConfidenceThreshold(double threshold);
Future<void> setIoUThreshold(double threshold); 
Future<void> setThresholds({
  double? confidenceThreshold,
  double? iouThreshold,
});
```

> **Note**: You can control YoloView in three ways:
> 1. Provide a controller to the constructor
> 2. Access the view directly via a GlobalKey
> 3. Don't provide anything and let the view create an internal controller
>
> See examples above for detailed usage patterns.

#### YOLOResult

Contains detection results.

```dart
class YOLOResult {
  final int classIndex;
  final String className;
  final double confidence;
  final Rect boundingBox;
  // For segmentation
  final List<List<double>>? mask;
  // For pose estimation
  final List<Point>? keypoints;
}
```

### Enums

#### YOLOTask

```dart
enum YOLOTask {
  detect,   // Object detection
  segment,  // Image segmentation
  classify, // Image classification
  pose,     // Pose estimation
  obb,      // Oriented bounding boxes
}
```

## Platform Support

| Android | iOS | Web | macOS | Windows | Linux |
|:-------:|:---:|:---:|:-----:|:-------:|:-----:|
|    ✅    |  ✅  |  ❌  |   ❌   |    ❌    |   ❌   |

## Model Loading

### Important: Recommended Approach For Both Platforms

For the most reliable cross-platform experience, the simplest approach is to:

1. **Use model name without extension** (`modelPath: 'yolo11n'`)
2. **Place platform-specific model files in the correct locations:**
   - Android: `android/app/src/main/assets/yolo11n.tflite`
   - iOS: Add `yolo11n.mlmodel` or `yolo11n.mlpackage` to your Xcode project

This approach avoids path resolution issues across platforms and lets each platform automatically find the appropriate model file without complicated path handling.

### Model Placement Options

This package supports loading models from multiple locations:

1. **Platform-Specific Native Assets (Recommended)**
   - Android: Place `.tflite` files in `android/app/src/main/assets/`
   - iOS: Add `.mlmodel` or `.mlpackage` files to your Xcode project
   - Reference in code: `modelPath: 'yolo11n'` (no extension, no path)

2. **Flutter Assets Directory (More Complex)**
   - Requires platform-specific handling in your Dart code
   - Android: Place `.tflite` files in your Flutter `assets` directory
   - iOS: Place `.mlmodel` files in your Flutter `assets` directory
   - Specify in `pubspec.yaml`:
     ```yaml
     flutter:
       assets:
         - assets/models/
     ```
   - Reference with platform detection:
     ```dart
     import 'dart:io';
     
     String modelPath = Platform.isAndroid
         ? 'assets/models/yolo11n.tflite'
         : 'assets/models/yolo11n.mlmodel';
     ```

3. **App Internal Storage**
   - Use when downloading models at runtime
   - Android path: `/data/user/0/<package_name>/app_flutter/`
   - iOS path: `/Users/<username>/Library/Application Support/<bundle_id>/`
   - Reference using the `internal://` scheme: `modelPath: 'internal://models/yolo11n.tflite'`
   - Or with absolute path: `modelPath: '/absolute/path/to/your_model.tflite'`

### Path Resolution Behavior By Platform

#### Android Path Resolution

- **Model Name Only**: `modelPath: 'yolo11n'` (RECOMMENDED)
  - Automatically appends `.tflite` extension → searches for `yolo11n.tflite`
  - First checks `android/app/src/main/assets/yolo11n.tflite`
  - Then checks Flutter assets for `yolo11n.tflite`

- **Asset Paths**: `modelPath: 'assets/models/yolo11n.tflite'`
  - CAUTION: The extension `.tflite` is expected for Android
  - If you use `.mlmodel` extension, Android will append `.tflite` to it 
    (e.g., `assets/models/yolo11n.mlmodel.tflite`) which will fail
  
- **App Internal Storage**: `modelPath: 'internal://models/yolo11n.tflite'`
  - Resolves to `/data/user/0/<package_name>/app_flutter/models/yolo11n.tflite`
  
#### iOS Path Resolution

- **Model Name Only**: `modelPath: 'yolo11n'` (RECOMMENDED)
  - Searches for resources in this order:
    1. `yolo11n.mlmodelc` in main bundle
    2. `yolo11n.mlpackage` in main bundle
    3. Various other locations including Flutter assets
  
- **Absolute Paths**: `modelPath: '/path/to/model.mlmodel'`
  - Used directly if file exists and has valid extension

### Platform-Specific Model Format Notes

- **Android**: Uses TensorFlow Lite (`.tflite`) models
  - Extension is automatically appended if missing
  - The YoloUtils class will always try to append `.tflite` to files without extension

- **iOS**: Uses Core ML models
  - Supports `.mlmodel`, `.mlmodelc` (compiled), and `.mlpackage` formats
  - `.mlpackage` files work best when added directly to the Xcode project
  - Flutter asset path resolution can be unpredictable with Core ML models

You can get the available storage paths at runtime:
```dart
final paths = await YOLO.getStoragePaths();
print("Internal storage path: ${paths['internal']}");
```

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Make sure your model file is correctly placed as described above
   - Verify that the model path is correctly specified
   - For iOS, ensure `.mlpackage` files are added directly to the Xcode project
   - Check that the model format is compatible with TFLite (Android) or Core ML (iOS)
   - Use `YOLO.checkModelExists(modelPath)` to verify if your model can be found

2. **Low performance on older devices**
   - Try using smaller models (e.g., YOLOv8n instead of YOLOv8l)
   - Reduce input image resolution
   - Increase confidence threshold to reduce the number of detections:
     ```dart
     // Using controller
     yoloController.setConfidenceThreshold(0.7); // Higher value = fewer detections
     
     // Or using GlobalKey
     yoloViewKey.currentState?.setConfidenceThreshold(0.7);
     ```
   - Adjust IoU threshold to control overlapping detections:
     ```dart
     // Using controller
     yoloController.setIoUThreshold(0.5); // Higher value = fewer merged boxes
     
     // Or using GlobalKey
     yoloViewKey.currentState?.setIoUThreshold(0.5);
     ```

3. **Camera permission issues**
   - Ensure that your app has the proper permissions in the manifest or Info.plist
   - Handle runtime permissions properly in your app

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.
