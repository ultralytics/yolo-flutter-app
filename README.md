<a href="https://ultralytics.com" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO for Flutter

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/wave/actions/workflows/format.yml) <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

A Flutter plugin for integrating Ultralytics YOLO computer vision models into your mobile apps. The plugin supports both Android and iOS platforms, and provides APIs for object detection and image classification.

## Features

| Feature         | Android | iOS |
| --------------- | ------- | --- |
| Detection       | ✅      | ✅  |
| Classification  | ✅      | ✅  |
| Pose Estimation | ❌      | ❌  |
| Segmentation    | ❌      | ❌  |
| OBB Detection   | ❌      | ❌  |

Before proceeding further or reporting new issues, please ensure you read this documentation thoroughly.

## Usage

Ultralytics YOLO is designed specifically for mobile platforms, targeting iOS and Android apps. The plugin leverages Flutter Platform Channels for communication between the client (_app_/_plugin_) and host (_platform_), ensuring seamless integration and responsiveness. All processing related to Ultralytics YOLO APIs is handled natively using Flutter's native APIs, with the plugin serving as a bridge between your app and Ultralytics YOLO.

### Prerequisites

#### Export Ultralytics YOLO Models

Before you can use Ultralytics YOLO in your app, you must export the required models. The exported models are in the form of `.tflite` and `.mlmodel` files, which you can then include in your app. Use the Ultralytics YOLO CLI to export the models.

> IMPORTANT: The parameters in the commands above are mandatory. Ultralytics YOLO plugin for Flutter only supports the models exported using the commands above. If you use different parameters, the plugin will not work as expected. We're working on adding support for more models and parameters in the future.

The following commands are used to export the models:

<details>
<summary><b>Android</b></summary>

#### Detection

```bash
yolo export format=tflite model=yolov8n imgsz=320 int8
```

#### Classification

```bash
yolo export format=tflite model=yolov8n-cls imgsz=320 int8
```

Then use file `yolov8n_int8.tflite` or `yolov8n-cls_int8.tflite`

</details>

<details>
<summary><b>iOS</b></summary>
To export the YOLOv8n Detection model for iOS, use the following command:

```bash
yolo export format=mlmodel model=yolov8n imgsz=[320, 192] half nms
```

</details>

### Installation

After exporting the models, you will get the `.tflite` and `.mlmodel` files. Include these files in your app's `assets` folder.

#### Permissions

Ensure that you have the necessary permissions to access the camera and storage.

<details>
<summary><b>Android</b></summary>

Add the following permissions to your `AndroidManifest.xml` file:

```xml

<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
```

</details>

<details>
<summary><b>iOS</b></summary>
Add the following permissions to your `Info.plist` file:

```xml

<key>NSCameraUsageDescription</key>
<string>Camera permission is required for object detection.</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Storage permission is required for object detection.</string>
```

Add the following permissions to your `Podfile`:

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

### Usage

#### Predictor

Create a predictor object using the `LocalYoloModel` class. This class requires the following parameters:

```dart
final model = LocalYoloModel(
        id: id,
        task: Task.detect /* or Task.classify */,
        format: Format.tflite /* or Format.coreml*/,
        modelPath: modelPath,
        metadataPath: metadataPath,
      );
```

##### Object Detector

```dart
final objectDetector = ObjectDetector(model: model);
await objectDetector.loadModel();
```

##### Image Classifier

```dart
final imageClassifier = ImageClassifier(model: model);
await imageClassifier.loadModel();
```

#### Camera Preview

The `UltralyticsYoloCameraPreview` widget is used to display the camera preview and the results of the prediction.

```dart
final _controller = UltralyticsYoloCameraController();
UltralyticsYoloCameraPreview(
 predictor: predictor, // Your prediction model data
 controller: _controller, // Ultralytics camera controller
 // For showing any widget on screen at the time of model loading
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
           style: theme.typography.base.copyWith(
             color: Colors.white,
             fontSize: 14,
           ),
```

#### Image

Use the `detect` or `classify` methods to get the results of the prediction on an image.

```dart
objectDetector.detect(imagePath: imagePath)
```

or

```dart
imageClassifier.classify(imagePath: imagePath)
```

## 💡 Contribute

Ultralytics thrives on community collaboration; we immensely value your involvement! We urge you to peruse our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for detailed insights on how you can participate. Don't forget to share your feedback with us by contributing to our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 goes out to everyone who has already contributed!

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## 📄 License

Ultralytics presents two distinct licensing paths to accommodate a variety of scenarios:

- **AGPL-3.0 License**: This official [OSI-approved](https://opensource.org/license) open-source license is perfectly aligned with the goals of students, enthusiasts, and researchers who believe in the virtues of open collaboration and shared wisdom. Details are available in the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) document.
- **Enterprise License**: Tailored for commercial deployment, this license authorizes the unfettered integration of Ultralytics software and AI models within commercial goods and services, without the copyleft stipulations of AGPL-3.0. Should your use case demand an enterprise solution, direct your inquiries to [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📮 Contact

For bugs or feature suggestions pertaining to Ultralytics, please lodge an issue via [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues). You're also invited to participate in our [Discord](https://discord.com/invite/ultralytics) community to engage in discussions and seek advice!

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
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
