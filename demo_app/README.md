# YOLO Flutter Demo App

This is a comprehensive demo application showcasing all features of the Ultralytics YOLO Flutter plugin. It demonstrates both single image inference and real-time camera detection across all supported YOLO tasks.

## Features

- 📷 **Camera Inference**: Real-time object detection using device camera
- 🖼️ **Single Image Inference**: Process images from gallery
- 🎯 **All YOLO Tasks**: Supports detect, segment, classify, pose, and OBB
- 📊 **Performance Metrics**: FPS counter and inference time display
- 🎨 **Rich UI**: Professional interface with animations and controls
- 🔧 **Adjustable Parameters**: Confidence and IoU thresholds

## Structure

```
demo_app/
├── lib/
│   ├── main.dart                          # App entry point
│   ├── models/                            # Data models
│   ├── presentation/                      # UI screens
│   │   ├── screens/
│   │   │   ├── camera_inference_screen.dart
│   │   │   └── single_image_screen.dart
│   └── services/                          # Business logic
│       └── model_manager.dart
├── android/                               # Android platform code
├── ios/                                   # iOS platform code
└── assets/                                # Images and models
```

## Running the Demo

1. **Prerequisites**:
   - Flutter SDK installed
   - Android Studio / Xcode configured
   - YOLO model files

2. **Install dependencies**:

   ```bash
   cd demo_app
   flutter pub get
   ```

3. **Add model files**:
   - Android: Place `.tflite` files in `android/app/src/main/assets/`
   - iOS: Add `.mlmodel` files via Xcode

4. **Run the app**:
   ```bash
   flutter run
   ```

## Model Files

The demo supports multiple YOLO models:

- `yolo11n.tflite` - Object detection
- `yolo11n-seg.tflite` - Instance segmentation
- `yolo11n-cls.tflite` - Image classification
- `yolo11n-pose.tflite` - Pose estimation
- `yolo11n-obb.tflite` - Oriented bounding boxes

## Features Demonstrated

### Camera Inference

- Real-time detection with bounding boxes
- FPS and inference time display
- Adjustable detection thresholds
- Support for all YOLO tasks

### Single Image Processing

- Gallery image selection
- Comprehensive result visualization
- Task-specific displays (masks, keypoints, etc.)
- Performance metrics

## Customization

The demo app can be customized:

- Modify detection thresholds in the UI
- Change visualization colors and styles
- Add new models to the model manager
- Extend with custom post-processing

## Note

This is a feature-rich demonstration app. For a minimal example suitable for quick integration, see the main `example/` directory.
