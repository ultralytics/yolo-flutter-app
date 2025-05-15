<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Flutter App

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

Welcome to the Ultralytics YOLO Flutter plugin! Integrate cutting-edge [Ultralytics YOLO](https://docs.ultralytics.com/) [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models seamlessly into your Flutter mobile applications. This plugin supports both Android and iOS platforms, offering APIs for [object detection](https://docs.ultralytics.com/tasks/detect/) and [image classification](https://docs.ultralytics.com/tasks/classify/).

## ✨ Features

| Feature         | Android | iOS |
| --------------- | ------- | --- |
| Detection       | ✅      | ✅  |
| Classification  | ✅      | ✅  |
| Pose Estimation | ❌      | ❌  |
| Segmentation    | ❌      | ❌  |
| OBB Detection   | ❌      | ❌  |

Before proceeding or reporting issues, please ensure you have read this documentation thoroughly.

## 🚀 Usage

This Ultralytics YOLO plugin is specifically designed for mobile platforms, targeting iOS and Android apps. It leverages [Flutter Platform Channels](https://docs.flutter.dev/platform-integration/platform-channels) for efficient communication between the client (your app/plugin) and the host platform (Android/iOS), ensuring seamless integration and responsiveness. All intensive processing related to Ultralytics YOLO APIs is handled natively using platform-specific APIs, with this plugin acting as a bridge.

### ✅ Prerequisites

#### Export Ultralytics YOLO Models

Before integrating Ultralytics YOLO into your app, you must export the necessary models. The [export process](https://docs.ultralytics.com/modes/export/) generates `.tflite` (for Android) and `.mlmodel` (for iOS) files, which you'll include in your app. Use the Ultralytics YOLO Command Line Interface (CLI) for exporting.

> **IMPORTANT:** The parameters specified in the commands below are **mandatory**. This Flutter plugin currently only supports models exported using these exact commands. Using different parameters may cause the plugin to malfunction. We are actively working on expanding support for more models and parameters.

Use the following commands to export the required models:

<details>
<summary><b>Android</b></summary>

#### Detection

Export the [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) detection model:

```bash
yolo export format=tflite model=yolov8n imgsz=320 int8
```

#### Classification

Export the YOLOv8n-cls classification model:

```bash
yolo export format=tflite model=yolov8n-cls imgsz=320 int8
```

After running the commands, use the generated `yolov8n_int8.tflite` or `yolov8n-cls_int8.tflite` file in your Android project.

</details>

<details>
<summary><b>iOS</b></summary>

Export the [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) detection model for iOS:

```bash
yolo export format=mlmodel model=yolov8n imgsz=[320, 192] half nms
```

Use the resulting `.mlmodel` file in your iOS project.

</details>

### 🛠️ Installation

After exporting the models, include the generated `.tflite` and `.mlmodel` files in your Flutter app's `assets` folder. Refer to the [Flutter documentation on adding assets](https://docs.flutter.dev/ui/assets/assets-and-images) for guidance.

#### Permissions

Ensure your application requests the necessary permissions to access the camera and storage.

<details>
<summary><b>Android</b></summary>

Add the following permissions to your `AndroidManifest.xml` file, typically located at `android/app/src/main/AndroidManifest.xml`. Consult the [Android developer documentation](https://developer.android.com/guide/topics/permissions/overview) for more details on permissions.

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
```

</details>

<details>
<summary><b>iOS</b></summary>

Add the following keys with descriptions to your `Info.plist` file, usually found at `ios/Runner/Info.plist`. See Apple's documentation on [protecting user privacy](https://developer.apple.com/documentation/uikit/protecting-the-user-s-privacy) for more information.

```xml
<key>NSCameraUsageDescription</key>
<string>Camera permission is required for object detection.</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Storage permission is required for object detection.</string>
```

Additionally, modify your `Podfile` (located at `ios/Podfile`) to include permission configurations for `permission_handler`:

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

</details>

### 👨‍💻 Usage Examples

#### Predictor Setup

Instantiate a predictor object using the `LocalYoloModel` class. Provide the necessary parameters:

```dart
// Define the model configuration
final model = LocalYoloModel(
  id: 'yolov8n-detect', // Unique identifier for the model
  task: Task.detect, // Specify the task (detect or classify)
  format: Format.tflite, // Specify the model format (tflite or coreml)
  modelPath: 'assets/models/yolov8n_int8.tflite', // Path to the model file in assets
  metadataPath: 'assets/models/metadata.yaml', // Path to the metadata file (if applicable)
);
```

##### Object Detector

Create and load an `ObjectDetector`:

```dart
// Initialize the ObjectDetector
final objectDetector = ObjectDetector(model: model);

// Load the model
await objectDetector.loadModel();
```

##### Image Classifier

Create and load an `ImageClassifier`:

```dart
// Initialize the ImageClassifier (adjust model details accordingly)
final imageClassifier = ImageClassifier(model: model); // Ensure 'model' is configured for classification

// Load the model
await imageClassifier.loadModel();
```

#### Camera Preview Integration

Use the `UltralyticsYoloCameraPreview` [widget](https://api.flutter.dev/flutter/widgets/Widget-class.html) to display the live camera feed and overlay prediction results.

```dart
final _controller = UltralyticsYoloCameraController(
  deferredProcessing: true, // deferred processing for better performance of android (Android only, default: false)
);
UltralyticsYoloCameraPreview(
  predictor: objectDetector, // Pass your initialized predictor (ObjectDetector or ImageClassifier)
  controller: _controller, // Pass the camera controller
  // Optional: Display a loading indicator while the model loads
  loadingPlaceholder: Center(
    child: Wrap(
      direction: Axis.vertical,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        const CircularProgressIndicator(
          color: Colors.white,
          strokeWidth: 2,
        ),
        const SizedBox(height: 20),
        Text(
          'Loading model...',
          // style: theme.typography.base.copyWith( // Adapt styling as needed
          //   color: Colors.white,
          //   fontSize: 14,
          // ),
        ),
      ],
    ),
  ),
  // Add other necessary parameters like onCameraCreated, onCameraInitialized, etc.
)
```

#### Image Prediction

Perform predictions on static images using the `detect` or `classify` methods.

```dart
// Perform object detection on an image file
final detectionResults = await objectDetector.detect(imagePath: 'path/to/your/image.jpg');
```

or

```dart
// Perform image classification on an image file
final classificationResults = await imageClassifier.classify(imagePath: 'path/to/your/image.jpg');
```

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
