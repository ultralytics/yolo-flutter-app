# Ultralytics YOLO Flutter Sample Applications

This directory contains sample applications demonstrating various features and use cases of the Ultralytics YOLO Flutter plugin.

## Sample Applications

### 1. [Basic Detection](01_basic_detection/)

**Difficulty: Beginner**

A simple introduction to object detection with YOLO.

- Single image detection from camera/gallery
- Bounding box visualization
- Confidence threshold adjustment
- Perfect starting point for new users

### 2. [Pose Estimation](02_pose_estimation/)

**Difficulty: Intermediate**

Demonstrates human pose estimation capabilities.

- 17-keypoint detection (COCO format)
- Skeleton visualization
- Multi-person support
- Keypoint confidence display

### 3. [Instance Segmentation](03_segmentation/)

**Difficulty: Intermediate**

Shows pixel-level object segmentation.

- Object masks with transparency
- Color-coded instance visualization
- Mask overlap handling
- Performance optimization tips

### 4. [Image Classification](04_classification/)

**Difficulty: Beginner**

Basic image classification example.

- Top-5 predictions display
- Confidence bars visualization
- 1000+ class support (ImageNet)
- Simple integration pattern

### 5. [Oriented Bounding Box Detection](05_obb_detection/)

**Difficulty: Advanced**

Specialized detection for rotated objects.

- Rotated bounding box visualization
- Angle display
- Ideal for aerial/satellite imagery
- Custom drawing implementation

### 6. [Real-time Camera Detection](06_camera_detection/)

**Difficulty: Intermediate**

Live camera feed with object detection.

- Real-time processing
- FPS display
- Camera switching
- Performance metrics

### 7. [Multi-Model Inference](07_multi_model/)

**Difficulty: Advanced**

Run multiple YOLO models simultaneously.

- Model comparison
- Performance benchmarking
- Task switching
- Resource management

### 8. [Custom UI and Visualization](08_custom_ui/)

**Difficulty: Advanced**

Advanced UI customization examples.

- Multiple visualization styles (Modern, Neon, Minimal, Glass)
- Animated detections
- Heatmap overlays
- Custom detection filters

## Getting Started

1. **Prerequisites**
   - Flutter SDK installed
   - Android Studio / Xcode for platform-specific setup
   - YOLO model files (`.tflite` for Android, `.mlmodel` for iOS)

2. **Running a Sample**

   ```bash
   cd samples/[sample_name]
   flutter pub get
   flutter run
   ```

3. **Model Setup**
   - Android: Place model files in `android/app/src/main/assets/`
   - iOS: Add model files to Xcode project with target set to Runner

## Model Files

Each sample requires specific YOLO model files:

| Sample           | Model File          | Task Type               |
| ---------------- | ------------------- | ----------------------- |
| Basic Detection  | yolo11n.tflite      | Object Detection        |
| Pose Estimation  | yolo11n-pose.tflite | Pose Estimation         |
| Segmentation     | yolo11n-seg.tflite  | Instance Segmentation   |
| Classification   | yolo11n-cls.tflite  | Image Classification    |
| OBB Detection    | yolo11n-obb.tflite  | Oriented BBox Detection |
| Camera Detection | yolo11n.tflite      | Object Detection        |
| Multi-Model      | All models          | All tasks               |
| Custom UI        | yolo11n.tflite      | Object Detection        |

## Learning Path

For developers new to YOLO Flutter:

1. Start with **Basic Detection** to understand core concepts
2. Try **Classification** for a simpler task
3. Move to **Camera Detection** for real-time processing
4. Explore **Segmentation** or **Pose** for advanced features
5. Study **Multi-Model** for complex applications
6. Reference **Custom UI** for visualization ideas

## Key Concepts

### YOLO Instance Management

```dart
// Single instance (default)
final yolo = YOLO(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
);

// Multi-instance support
final yolo = YOLO(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
  useMultiInstance: true,
);
```

### Real-time Camera Processing

```dart
YOLOView(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
  onResult: (results) {
    // Process results
  },
)
```

### Performance Optimization

- Use appropriate model sizes (n/s/m/l/x variants)
- Adjust confidence thresholds
- Limit maximum detections
- Consider frame skipping for real-time processing

## Contributing

When adding new samples:

1. Follow the existing structure and naming conventions
2. Include comprehensive README documentation
3. Add appropriate comments in code
4. Test on both Android and iOS platforms
5. Ensure proper error handling

## License

These samples are provided under the same AGPL-3.0 license as the Ultralytics YOLO Flutter plugin. See the main project LICENSE for details.
