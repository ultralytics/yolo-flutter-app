# YOLO Flutter Plugin Example

This is a minimal example demonstrating how to use the Ultralytics YOLO Flutter plugin. In just ~80 lines of code, you can add object detection to your Flutter app.

## Quick Start

```dart
// 1. Initialize YOLO
final yolo = YOLO(modelPath: 'yolo11n.tflite');

// 2. Load model
await yolo.loadModel();

// 3. Run inference
final results = await yolo.predict(imageBytes);

// 4. Parse results
final detections = (results['detections'] as List?)
    ?.map((e) => YOLOResult.fromMap(e))
    .toList();

// 5. Clean up
await yolo.dispose();
```

## Features Demonstrated

- ✅ Image picker integration
- ✅ YOLO model loading
- ✅ Object detection inference
- ✅ Result parsing
- ✅ Basic UI display

## Running the Example

1. **Add the plugin** to your `pubspec.yaml`:
   ```yaml
   dependencies:
     ultralytics_yolo: ^0.0.3
   ```

2. **Add model files**:
   - Android: Place `yolo11n.tflite` in `android/app/src/main/assets/`
   - iOS: Add `yolo11n.mlmodel` to Xcode project

3. **Run the app**:
   ```bash
   flutter run
   ```

## What's in the Example

The example consists of a single `main.dart` file that:
1. Creates a simple Flutter app
2. Allows users to pick an image from gallery
3. Runs YOLO object detection
4. Displays the number of objects found

## Next Steps

- For a comprehensive demo with camera support and all YOLO tasks, see the `demo_app/` directory
- For task-specific examples, check the `samples/` directory
- Read the full documentation at [pub.dev/packages/ultralytics_yolo](https://pub.dev/packages/ultralytics_yolo)

## Model Files

Download pre-trained models from:
- [Ultralytics Hub](https://hub.ultralytics.com)
- [GitHub Releases](https://github.com/ultralytics/yolo-flutter-app/releases)