# YOLO Pose Estimation Sample

This sample demonstrates how to use the Ultralytics YOLO Flutter plugin for human pose estimation.

## Features

- Single image pose detection from camera or gallery
- Real-time visualization of detected keypoints and skeleton
- Support for multiple person detection
- Visual feedback showing:
  - 17 keypoints per person (COCO format)
  - Skeleton connections between keypoints
  - Confidence scores for each detection
  - Processing time display

## Setup

1. Add the pose model to your assets:
   - Android: Place `yolo11n-pose.tflite` in `android/app/src/main/assets/`
   - iOS: Add `yolo11n-pose.mlmodel` to the Xcode project

2. Run the app:
   ```bash
   flutter run
   ```

## Key Concepts

### Pose Keypoints

The YOLO pose model detects 17 keypoints following the COCO format:

- Head: nose, eyes, ears
- Upper body: shoulders, elbows, wrists
- Lower body: hips, knees, ankles

### Using YOLOResult for Pose

```dart
// Each YOLOResult contains:
// - keypoints: List<Point> with x,y coordinates
// - keypointConfidences: List<double> confidence for each keypoint
// - boundingBox: Person bounding box
// - confidence: Overall person detection confidence

for (final pose in poseResults) {
  if (pose.keypoints != null) {
    for (int i = 0; i < pose.keypoints!.length; i++) {
      final keypoint = pose.keypoints![i];
      final confidence = pose.keypointConfidences![i];

      if (confidence > 0.5) {
        // Keypoint is visible
        print('Keypoint $i at (${keypoint.x}, ${keypoint.y})');
      }
    }
  }
}
```

### Skeleton Visualization

The sample draws connections between keypoints to form a skeleton:

```dart
static const List<List<int>> skeleton = [
  [0, 1], [0, 2],    // Head connections
  [5, 6], [5, 11],   // Body connections
  [5, 7], [7, 9],    // Left arm
  [6, 8], [8, 10],   // Right arm
  [11, 13], [13, 15], // Left leg
  [12, 14], [14, 16], // Right leg
];
```

## Customization

- Adjust detection thresholds:

  ```dart
  final result = await _yolo.predict(
    imageBytes,
    confidenceThreshold: 0.5,  // Minimum person confidence
    iouThreshold: 0.45,         // NMS threshold
  );
  ```

- Customize visualization colors and styles in `PosePainter`
- Filter keypoints by confidence for cleaner visualization

## Model Information

The sample uses `yolo11n-pose.tflite`, which is optimized for:

- Fast inference on mobile devices
- Human pose estimation with 17 keypoints
- Multi-person detection capabilities

For better accuracy, consider using larger models like `yolo11s-pose` or `yolo11m-pose`.
