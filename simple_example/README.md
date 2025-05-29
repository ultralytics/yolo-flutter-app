# Simple YOLO Flutter Examples

This directory contains minimal examples of using the Ultralytics YOLO Flutter plugin.

## Setup

### 1. Add Models

You need to add YOLO models to run these examples:

#### Android
Place `.tflite` model files in `assets/models/`:
```
assets/models/yolo11n.tflite
```

#### iOS
Add `.mlpackage` or `.mlmodel` files to your Xcode project:
1. Open `ios/Runner.xcworkspace` in Xcode
2. Drag and drop your model files (e.g., `yolo11n.mlpackage`) into the project
3. Make sure "Copy items if needed" is checked
4. Add to target "Runner"

### 2. Run the Examples

```bash
flutter pub get
flutter run
```

## Examples

### Camera View Example
Minimal example showing real-time object detection using the device camera:
```dart
YOLOView(
  modelPath: 'yolo11n',
  task: YOLOTask.detect,
  onResult: (results) {
    // Handle results
  },
)
```

### Single Image Example
Minimal example for running inference on a single image:
```dart
final yolo = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);
await yolo.loadModel();
final result = await yolo.predict(imageBytes);
```

## Model Path Convention

### Android
- **Flutter assets are supported!** Place models in `assets/models/`
- Use full path: `'assets/models/yolo11n.tflite'`
- Or just model name: `'yolo11n'` (`.tflite` extension will be added automatically)

### iOS
- **Flutter assets are NOT supported** (iOS platform limitation)
- Models must be added directly to Xcode project
- Use just the model name: `'yolo11n'`
- The plugin will search for `.mlpackage` or `.mlmodel` in the app bundle

## Platform-specific Setup

For a cross-platform app, you might want to use:
```dart
import 'dart:io';

final modelPath = Platform.isIOS ? 'yolo11n' : 'assets/models/yolo11n.tflite';
```