# Basic Object Detection Sample

This sample demonstrates the simplest way to use the Ultralytics YOLO Flutter plugin for object detection.

## Features

- Load a YOLO detection model
- Select images from device gallery
- Run object detection inference
- Display results with colored bounding boxes and labels
- Show detection confidence scores

## Getting Started

### 1. Add Model Files

Before running this sample, you need to add YOLO model files:

**Android:**
1. Create the assets directory: `android/app/src/main/assets/`
2. Copy your `.tflite` model file (e.g., `yolo11n.tflite`) to this directory

**iOS:**
1. Open `ios/Runner.xcworkspace` in Xcode
2. Drag your `.mlmodel` file (e.g., `yolo11n.mlmodel`) to the Runner target
3. Make sure "Copy items if needed" is checked

### 2. Install Dependencies

```bash
flutter pub get
```

### 3. Run the App

```bash
flutter run
```

## Usage

1. Tap "Select Image" to choose an image from your gallery
2. Tap "Detect Objects" to run YOLO inference
3. View the results:
   - Bounding boxes drawn on the image
   - Object labels with confidence scores
   - List of detected objects below the image

## Code Structure

The main code is in `lib/main.dart`:

- `DetectionScreen`: Main UI widget
- `_pickImage()`: Handles image selection from gallery
- `_runDetection()`: Performs YOLO inference
- `BoundingBoxPainter`: Custom painter for drawing bounding boxes

## Key Code Snippets

### Loading and Using YOLO

```dart
// Create YOLO instance
final yolo = YOLO(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
);

// Load model
await yolo.loadModel();

// Run inference
final response = await yolo.predict(imageBytes);

// Parse results
final detections = response['detections'] as List<dynamic>;
final results = detections
    .map((detection) => YOLOResult.fromMap(detection))
    .toList();
```

### Drawing Bounding Boxes

The sample uses normalized coordinates from the `YOLOResult` objects:

```dart
final left = result.normalizedBox.left * size.width;
final top = result.normalizedBox.top * size.height;
final right = result.normalizedBox.right * size.width;
final bottom = result.normalizedBox.bottom * size.height;
```

## Customization

You can customize this sample by:

- Changing the model file (use different YOLO variants)
- Adjusting confidence thresholds
- Modifying the UI colors and layout
- Adding image capture from camera
- Implementing real-time detection

## Troubleshooting

If you encounter issues:

1. **Model not loading**: Ensure the model file is in the correct location
2. **No detections**: Try a different image or check the model compatibility
3. **App crashes**: Check the console for error messages

## Next Steps

After mastering this basic sample, try:
- [02_pose_estimation](../02_pose_estimation/) - Detect human poses
- [06_camera_detection](../06_camera_detection/) - Real-time camera detection
- [07_multi_model](../07_multi_model/) - Switch between different models