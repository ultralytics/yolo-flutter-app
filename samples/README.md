# Ultralytics YOLO Flutter Samples

This directory contains simple, focused sample applications demonstrating various features of the Ultralytics YOLO Flutter plugin. Each sample is designed to be easy to understand and focuses on a single aspect of the plugin.

## 📋 Sample Applications

### 1. [01_basic_detection](./01_basic_detection/)

**Basic Object Detection** - The simplest example of using YOLO for object detection

- Load a YOLO model
- Select an image from gallery
- Run inference and display bounding boxes
- ~100 lines of code

### 2. [02_pose_estimation](./02_pose_estimation/)

**Human Pose Estimation** - Detect human keypoints and poses

- Detect body keypoints (17 points for COCO format)
- Visualize skeleton connections
- Display confidence for each keypoint
- Handle multiple people in one image

### 3. [03_segmentation](./03_segmentation/)

**Instance Segmentation** - Pixel-level object detection

- Detect objects with pixel-precise masks
- Visualize each instance with different colors
- Overlay masks on original image
- Handle mask transparency

### 4. [04_classification](./04_classification/)

**Image Classification** - Classify entire images

- Load classification model
- Get top-5 predictions
- Display confidence scores
- Simple UI with results

### 5. [05_obb_detection](./05_obb_detection/)

**Oriented Bounding Box Detection** - Detect rotated objects

- Detect objects with rotation angles
- Draw oriented bounding boxes
- Useful for aerial/satellite imagery
- Handle arbitrary orientations

### 6. [06_camera_detection](./06_camera_detection/)

**Real-time Camera Detection** - Live object detection

- Use YOLOView widget
- Real-time camera feed
- FPS counter
- Adjustable detection thresholds

### 7. [07_multi_model](./07_multi_model/)

**Multiple Models** - Switch between different models

- Load multiple models
- Switch between tasks dynamically
- Memory management
- Performance comparison

### 8. [08_custom_ui](./08_custom_ui/)

**Custom UI and Visualization** - Advanced UI customization

- Custom drawing of results
- Animation effects
- Result filtering
- Advanced visualization techniques

## 🚀 Getting Started

### Prerequisites

1. Flutter SDK installed
2. Android Studio / Xcode for platform-specific setup
3. YOLO model files (.tflite for Android, .mlmodel for iOS)

### Running a Sample

1. Navigate to the sample directory:

   ```bash
   cd samples/01_basic_detection
   ```

2. Install dependencies:

   ```bash
   flutter pub get
   ```

3. Add your model files:

   - Android: Place `.tflite` files in `android/app/src/main/assets/`
   - iOS: Add `.mlmodel` files via Xcode to the Runner target

4. Run the sample:
   ```bash
   flutter run
   ```

## 📱 Model Files

You can obtain model files from:

1. **Ultralytics Hub**: https://hub.ultralytics.com
2. **GitHub Releases**: https://github.com/ultralytics/yolo-flutter-app/releases
3. **Export from Python**:

   ```python
   from ultralytics import YOLO

   model = YOLO("yolo11n.pt")
   model.export(format="tflite")  # For Android
   model.export(format="coreml")  # For iOS
   ```

## 📝 Code Structure

Each sample follows a similar structure:

```
sample_name/
├── lib/
│   └── main.dart          # Main application code
├── pubspec.yaml           # Dependencies
├── README.md              # Sample-specific documentation
└── screenshots/           # Sample screenshots
    └── demo.png
```

## 🎯 Learning Path

We recommend going through the samples in order:

1. Start with **01_basic_detection** to understand the basics
2. Try **06_camera_detection** for real-time detection
3. Explore task-specific samples (02-05) based on your needs
4. Advanced users can check **07_multi_model** and **08_custom_ui**

## 💡 Tips

- Each sample is self-contained and can be used as a starting point for your own app
- Code is heavily commented in both English and Japanese
- Check the individual README files in each sample for specific details
- All samples use the same model files, so you can reuse them

## 🤝 Contributing

If you have ideas for new samples or improvements, please:

1. Open an issue describing your idea
2. Submit a pull request with your sample
3. Follow the existing code style and structure

## 📄 License

These samples are provided under the same license as the main project (AGPL-3.0).
