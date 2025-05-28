<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Flutter App

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

Welcome to the Ultralytics YOLO Flutter plugin! Integrate cutting-edge [Ultralytics YOLO](https://docs.ultralytics.com/) [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models seamlessly into your Flutter mobile applications. This plugin at https://pub.dev/packages/ultralytics_yolo supports both Android and iOS platforms, offering APIs for [object detection](https://docs.ultralytics.com/tasks/detect/), [image classification](https://docs.ultralytics.com/tasks/classify/), [instance segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented bounding box detection](https://docs.ultralytics.com/tasks/obb/).

## ✨ Features

| Feature         | Android | iOS |
| --------------- | ------- | --- |
| Detection       | ✅      | ✅  |
| Classification  | ✅      | ✅  |
| Segmentation    | ✅      | ✅  |
| Pose Estimation | ✅      | ✅  |
| OBB Detection   | ✅      | ✅  |

- **Real-time Processing**: Optimized for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) on mobile devices
- **Camera Integration**: Easy integration with device cameras for live detection
- **Cross-Platform**: Works seamlessly on both Android and iOS platforms
- **High Performance**: Leverages [TensorFlow Lite](https://www.ultralytics.com/glossary/tensorflow) for Android and [Core ML](https://docs.ultralytics.com/integrations/coreml/) for iOS
- **Camera Zoom Control**: Support for pinch-to-zoom and programmatic zoom control
- **Dynamic Model Switching**: Switch between different YOLO models without recreating the view

## 🚀 Installation

Add this to your package's `pubspec.yaml` file:

```yaml
dependencies:
  ultralytics_yolo: ^0.1.5
```

Then run:

```bash
flutter pub get
```

## 📱 Platform-Specific Setup

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

Additionally, modify your `Podfile` (located at `ios/Podfile`) to include permission configurations:

```ruby
post_install do |installer|
  installer.pods_project.targets.each do |target|
    flutter_additional_ios_build_settings(target)

    # Start of the permission_handler configuration
    target.build_configurations.each do |config|
      config.build_settings['GCC_PREPROCESSOR_DEFINITIONS'] ||= [
        '$(inherited)',

        ## dart: PermissionGroup.camera
        'PERMISSION_CAMERA=1',

        ## dart: PermissionGroup.photos
        'PERMISSION_PHOTOS=1',
      ]
    end
    # End of the permission_handler configuration
  end
end
```

## ✅ Prerequisites

### Export Ultralytics YOLO Models

Before integrating Ultralytics YOLO into your app, you must export the necessary models. The [export process](https://docs.ultralytics.com/modes/export/) generates `.tflite` (for Android) and `.mlpackage` (for iOS) files, which you'll include in your app. Use the Ultralytics YOLO Command Line Interface (CLI) for exporting.

> **IMPORTANT:** The parameters specified in the commands below are **mandatory**. This Flutter plugin currently only supports models exported using these exact commands. Using different parameters may cause the plugin to malfunction. We are actively working on expanding support for more models and parameters.

Use the following commands to export the required YOLO models:

```python
from ultralytics import YOLO
from ultralytics.utils.downloads import zip_directory


def export_and_zip_yolo_models(
    model_types=("", "-seg", "-cls", "-pose", "-obb"),
    model_sizes=("n",),  #  optional additional sizes are "s", "m", "l", "x"
):
    """Exports YOLO11 models to CoreML format and optionally zips the output packages."""
    for model_type in model_types:
        imgsz = [224, 224] if "cls" in model_type else [640, 384]  # default input image sizes
        nms = True if model_type == "" else False  # only apply NMS to Detect models
        for size in model_sizes:
            model_name = f"yolo11{size}{model_type}"
            model = YOLO(f"{model_name}.pt")

            # iOS Export
            model.export(format="coreml", int8=True, imgsz=imgsz, nms=nms)
            zip_directory(f"{model_name}.mlpackage").rename(f"{model_name}.mlpackage.zip")

            # TFLite Export
            model.export(format="tflite", int8=True, imgsz=[320, 320], nms=False)


# Execute with default parameters
export_and_zip_yolo_models()
```

## 👨‍💻 Usage

### Basic Example

```dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

class YoloDemo extends StatefulWidget {
  @override
  _YoloDemoState createState() => _YoloDemoState();
}

class _YoloDemoState extends State<YoloDemo> {
  // Create a controller to interact with the YoloView
  final controller = YoloViewController();
  double _confidenceValue = 0.5;
  double _currentZoom = 1.0;
  YOLOTask _currentTask = YOLOTask.detect;
  String _currentModel = 'yolo11n';

  @override
  void initState() {
    super.initState();
    // Set initial detection parameters
    controller.setThresholds(
      confidenceThreshold: 0.5,
      iouThreshold: 0.45,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLO Object Detection'),
        actions: [
          // Switch camera button
          IconButton(
            icon: Icon(Icons.flip_camera_ios),
            onPressed: () => controller.switchCamera(),
          ),
        ],
      ),
      body: Column(
        children: [
          // Controls for adjusting detection parameters
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                // Confidence threshold slider
                Row(
                  children: [
                    Text('Confidence: ${_confidenceValue.toStringAsFixed(2)}'),
                    Expanded(
                      child: Slider(
                        value: _confidenceValue,
                        min: 0.1,
                        max: 0.9,
                        onChanged: (value) {
                          setState(() => _confidenceValue = value);
                          controller.setConfidenceThreshold(value);
                        },
                      ),
                    ),
                  ],
                ),
                // Zoom control
                Row(
                  children: [
                    Text('Zoom: ${_currentZoom.toStringAsFixed(1)}x'),
                    IconButton(
                      icon: Icon(Icons.zoom_out),
                      onPressed: () => controller.setZoomLevel(1.0),
                    ),
                    IconButton(
                      icon: Icon(Icons.zoom_in),
                      onPressed: () => controller.setZoomLevel(3.0),
                    ),
                  ],
                ),
                // Model switcher
                Row(
                  children: [
                    Text('Model: '),
                    DropdownButton<String>(
                      value: _currentModel,
                      items: ['yolo11n', 'yolo11s', 'yolo11m']
                          .map((model) => DropdownMenuItem(
                                value: model,
                                child: Text(model),
                              ))
                          .toList(),
                      onChanged: (model) {
                        if (model != null) {
                          setState(() => _currentModel = model);
                          controller.switchModel(model, _currentTask);
                        }
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),

          // YoloView with controller
          Expanded(
            child: YoloView(
              controller: controller,
              task: _currentTask,
              modelPath: _currentModel,
              onResult: (results) {
                // Handle detection results
                print('Detected ${results.length} objects');
              },
              onZoomChanged: (zoom) {
                setState(() => _currentZoom = zoom);
              },
              onPerformanceMetrics: (metrics) {
                print('FPS: ${metrics['fps']?.toStringAsFixed(1)}');
              },
            ),
          ),
        ],
      ),
    );
  }
}
```

### Object Detection with Camera Feed

There are three ways to control YoloView's detection parameters:

#### Camera Zoom Control

The plugin now supports camera zoom functionality through both gestures and programmatic control:

```dart
// Enable zoom callbacks to track zoom changes
YoloView(
  controller: controller,
  task: YOLOTask.detect,
  modelPath: 'yolo11n',
  onZoomChanged: (zoomLevel) {
    print('Current zoom: ${zoomLevel}x');
  },
  onResult: (results) {
    // Handle results
  },
)

// Programmatically set zoom level
controller.setZoomLevel(2.5);  // Set to 2.5x zoom

// Users can also pinch-to-zoom on the camera view
```

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

### Dynamic Model Switching

You can switch between different YOLO models at runtime without recreating the entire view:

```dart
final controller = YoloViewController();

// Initial model
YoloView(
  controller: controller,
  task: YOLOTask.detect,
  modelPath: 'yolo11n',  // Start with nano model
  onResult: (results) {
    // Handle results
  },
)

// Switch to a different model based on user preference or device capabilities
await controller.switchModel('yolo11s', YOLOTask.detect);  // Switch to small model
await controller.switchModel('yolo11m', YOLOTask.detect);  // Switch to medium model

// Switch to a different task type
await controller.switchModel('yolo11n-seg', YOLOTask.segment);  // Switch to segmentation
await controller.switchModel('yolo11n-pose', YOLOTask.pose);    // Switch to pose estimation
```

### Model Loading

#### Important: Recommended Approach For Both Platforms

For the most reliable cross-platform experience, the simplest approach is to:

1. **Use model name without extension** (`modelPath: 'yolo11n'`)
2. **Place platform-specific model files in the correct locations:**
   - Android: `android/app/src/main/assets/yolo11n.tflite`
   - iOS: Add `yolo11n.mlmodel` or `yolo11n.mlpackage` to your Xcode project

This approach avoids path resolution issues across platforms and lets each platform automatically find the appropriate model file without complicated path handling.

## 📚 API Reference

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
int numItems = controller.numItemsThreshold;

// Set confidence threshold (0.0-1.0)
await controller.setConfidenceThreshold(0.6);

// Set IoU threshold (0.0-1.0)
await controller.setIoUThreshold(0.5);

// Set maximum number of detection items (1-100)
await controller.setNumItemsThreshold(20);

// Set multiple thresholds at once
await controller.setThresholds(
  confidenceThreshold: 0.6,
  iouThreshold: 0.5,
  numItemsThreshold: 20,
);

// Switch between front and back camera
await controller.switchCamera();

// Set camera zoom level (1.0 = no zoom, 2.0 = 2x zoom)
await controller.setZoomLevel(2.0);

// Switch to a different model dynamically
await controller.switchModel('yolo11s', YOLOTask.detect);
```

#### YoloView

Flutter widget to display YOLO detection results.

```dart
YoloView({
  required YOLOTask task,
  required String modelPath,
  YoloViewController? controller,  // Optional: Controller for managing view settings
  Function(List<YOLOResult>)? onResult,
  Function(Map<String, double>)? onPerformanceMetrics,  // Optional: Performance tracking
  Function(double)? onZoomChanged,  // Optional: Zoom level callback
  bool showNativeUI = false,  // Optional: Show native UI controls
});
```

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
  // Performance metrics
  final double? processingTimeMs; // Processing time in milliseconds for the frame
  final double? fps;              // Frames Per Second (available on Android, and iOS for real-time)
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

## 🔧 Troubleshooting

### Common Issues

1. **Model loading fails**

   - Make sure your model file is correctly placed as described above
   - Verify that the model path is correctly specified
   - For iOS, ensure `.mlpackage` files are added directly to the Xcode project and properly included in target's "Build Phases" → "Copy Bundle Resources"
   - Check that the model format is compatible with [TensorFlow Lite](https://www.ultralytics.com/glossary/tensorflow) (Android) or [Core ML](https://docs.ultralytics.com/integrations/coreml/) (iOS)
   - Use `YOLO.checkModelExists(modelPath)` to verify if your model can be found

2. **Low performance on older devices**

   - Try using smaller models (e.g., YOLO11n instead of YOLO11l)
   - Reduce input image resolution
   - Increase [confidence threshold](https://www.ultralytics.com/glossary/confidence) to reduce the number of detections
   - Adjust [IoU threshold](https://www.ultralytics.com/glossary/intersection-over-union-iou) to control overlapping detections
   - Limit the maximum number of detection items

3. **Camera permission issues**

   - Ensure that your app has the proper permissions in the manifest or Info.plist
   - Handle runtime permissions properly in your app

4. **Performance optimization tips**
   - Use [model quantization](https://www.ultralytics.com/glossary/model-quantization) for faster inference
   - Consider [edge computing](https://www.ultralytics.com/glossary/edge-computing) approaches for better performance
   - Implement proper [data preprocessing](https://www.ultralytics.com/glossary/data-preprocessing) for optimal results

## 💡 Contribute

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's bug fixes, feature enhancements, or documentation improvements, your involvement is crucial. Please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for detailed insights on how to participate. We also encourage you to share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 goes out to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics offers two licensing options to accommodate diverse needs:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts passionate about open-source collaboration. This [OSI-approved](https://opensource.org/license/agpl-v3) license promotes knowledge sharing and open contribution. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license permits seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. For commercial use cases, please inquire about an [Enterprise License](https://www.ultralytics.com/license).

## 📮 Contact

Encountering issues or have feature requests related to Ultralytics YOLO? Please report them via [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues). For broader discussions, questions, and community support, join our [Discord](https://discord.com/invite/ultralytics) server!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
