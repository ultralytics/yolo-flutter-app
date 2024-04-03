# Ultralytics YOLO for Flutter

A Flutter plugin for integrating Ultralytics YOLO computer vision models into your mobile apps. The plugin supports both Android and iOS platforms, and provides APIs for object detection and image classification. 

## Features

| Feature | Android | iOS | 
| --- | --- | --- |
| Detection | ✅ | ✅ |
| Classification | ✅ | ✅ |
| Pose Estimation | ❌ | ❌ |
| Segmentation | ❌ | ❌ |
| OBB Detection | ❌ | ❌ |

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
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
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
 predictor: predictor, // Your prediciton model data
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

## Issues
Please report any issues you find with the plugin on the [GitHub repository](https://github.com/ultralytics/ultralytics_yolo_mobile/issues). We'll do our best to address them as soon as possible.

## Contributing
We love your input! YOLOv5 and YOLOv8 would not be possible without help from our community. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started, and fill out our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experience. Thank you 🙏 to all our contributors!


## License
Ultralytics YOLO for Flutter is licensed under the [AGPL 3.0 License](https://github.com/ultralytics/ultralytics_yolo_mobile/blob/master/LICENSE). Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/licenses/) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://ultralytics.com/license).
