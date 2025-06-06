# Human Pose Estimation Sample

This sample demonstrates how to use the Ultralytics YOLO Flutter plugin for human pose estimation.

## Features

- Detect human poses using YOLO pose models
- Visualize 17 keypoints (COCO format)
- Draw skeleton connections between keypoints
- Support multiple people in one image
- Toggle keypoints and skeleton visualization
- Color-coded display for different people

## Keypoints Detected

The COCO pose format includes 17 keypoints:
1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

## Getting Started

### 1. Add Model Files

Before running this sample, you need to add YOLO pose model files:

**Android:**
1. Create the assets directory: `android/app/src/main/assets/`
2. Copy your `.tflite` pose model file (e.g., `yolo11n-pose.tflite`) to this directory

**iOS:**
1. Open `ios/Runner.xcworkspace` in Xcode
2. Drag your `.mlmodel` pose file (e.g., `yolo11n-pose.mlmodel`) to the Runner target
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
2. Tap "Detect Poses" to run YOLO inference
3. View the results:
   - Keypoints shown as colored circles
   - Skeleton connections drawn between keypoints
   - Bounding box around each detected person
   - List of detected people below the image
4. Use the toggle switches to show/hide keypoints or skeleton

## Code Structure

The main code is in `lib/main.dart`:

- `PoseEstimationScreen`: Main UI widget
- `_runPoseEstimation()`: Performs YOLO pose inference
- `PosePainter`: Custom painter for drawing keypoints and skeleton
- `skeleton`: Defines connections between keypoints
- `keypointNames`: Human-readable names for keypoints

## Key Code Snippets

### Using YOLO for Pose Estimation

```dart
// Create YOLO pose instance
final yolo = YOLO(
  modelPath: 'yolo11n-pose.tflite',
  task: YOLOTask.pose,
);

// Run inference
final response = await yolo.predict(imageBytes);

// Parse results with keypoints
final detections = response['detections'] as List<dynamic>;
final results = detections
    .map((detection) => YOLOResult.fromMap(detection))
    .toList();

// Access keypoints
for (final result in results) {
  final keypoints = result.keypoints; // List<Point<double>>
  final confidences = result.keypointConfidences; // List<double>
}
```

### Drawing Skeleton Connections

```dart
// Define skeleton connections
static const List<List<int>> skeleton = [
  // Head connections
  [0, 1], [0, 2], [1, 3], [2, 4],
  // Arm connections
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  // Body connections
  [5, 11], [6, 12], [11, 12],
  // Leg connections
  [11, 13], [13, 15], [12, 14], [14, 16],
];

// Draw connections
for (final connection in skeleton) {
  final startIdx = connection[0];
  final endIdx = connection[1];
  
  if (confidences[startIdx] > 0.5 && confidences[endIdx] > 0.5) {
    canvas.drawLine(
      Offset(keypoints[startIdx].x * size.width, 
             keypoints[startIdx].y * size.height),
      Offset(keypoints[endIdx].x * size.width,
             keypoints[endIdx].y * size.height),
      paint,
    );
  }
}
```

## Customization

You can customize this sample by:

- Adjusting the confidence threshold for keypoint visibility
- Changing colors for different people
- Adding keypoint labels
- Implementing pose classification
- Adding real-time camera pose detection
- Calculating pose angles for fitness applications

## Troubleshooting

If you encounter issues:

1. **No poses detected**: Ensure the image contains visible people
2. **Missing keypoints**: Some keypoints may be occluded or have low confidence
3. **Model not loading**: Verify you're using a pose-specific model (e.g., yolo11n-pose)
4. **Poor skeleton display**: Try images with clear, unobstructed human figures

## Next Steps

After mastering pose estimation, try:
- [03_segmentation](../03_segmentation/) - Instance segmentation
- [06_camera_detection](../06_camera_detection/) - Real-time camera detection
- Building a fitness app with pose analysis