# Ultralytics YOLO Flutter Example App

This example app demonstrates how to use the Ultralytics YOLO Flutter plugin (`ultralytics_yolo`) for various computer vision tasks such as object detection, segmentation, classification, pose estimation, and oriented bounding box detection.

## Getting Started

### Prerequisites

- Flutter SDK installed and configured
- An Android or iOS device/emulator
- YOLO model files (included in the assets)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ultralytics/yolo-flutter-app
   cd yolo-flutter-app/example
   ```

2. Install dependencies:
   ```bash
   flutter pub get
   ```

3. Run the app:
   ```bash
   flutter run
   ```

## Implementation Strategy

This document outlines the step-by-step strategy for enhancing the example app to showcase all features of the YOLO Flutter plugin.

### Phase 1: Basic Structure and Object Detection

1. **Create a task selection home screen**
   - Card-based UI for selecting different YOLO tasks
   - Add descriptive text and icons for each task
   - Add navigation to task-specific screens

2. **Improve existing object detection**
   - Enhance visualization of bounding boxes
   - Add confidence score display
   - Format detection results in a structured list view
   - Add support for resizing and processing larger images

3. **Add camera feed improvements**
   - Implement camera resolution switching
   - Add front/back camera toggle
   - Add confidence threshold slider
   - Improve real-time detection display

### Phase 2: Additional YOLO Tasks

4. **Implement segmentation screen**
   - Create a dedicated screen for segmentation
   - Visualize segmentation masks with adjustable opacity
   - Display class and confidence information
   - Support both camera feed and image picking

5. **Implement classification screen**
   - Create a dedicated screen for classification
   - Show top-N classification results with confidence bars
   - Support both camera feed and image picking

6. **Implement pose estimation screen**
   - Create a dedicated screen for pose estimation
   - Visualize keypoints and skeleton connections
   - Add different color schemes for multiple people detection
   - Display confidence scores for each keypoint

7. **Implement oriented bounding box screen**
   - Create a dedicated screen for OBB detection
   - Visualize rotated bounding boxes
   - Display orientation angles and dimensions
   - Support both camera feed and image picking

### Phase 3: Settings and Optimizations

8. **Add model settings screen**
   - Create a model selection interface
   - Bundle multiple pre-trained models
   - Display model information (size, classes, speed)
   - Allow confidence threshold adjustments globally

9. **Implement error handling**
   - Add user-friendly error messages for model loading failures
   - Handle invalid inputs gracefully
   - Add loading indicators during processing
   - Implement process cancellation mechanism

10. **Optimize performance**
    - Implement image resizing for large inputs
    - Add performance options for lower-end devices
    - Optimize memory usage during inference
    - Add performance metrics display (FPS, latency)

### Phase 4: Documentation and Polish

11. **Enhance in-app documentation**
    - Add explanation screens for each YOLO task
    - Include sample use cases for each feature
    - Provide tooltips and help buttons

12. **Visual polish**
    - Consistent color scheme and typography
    - Smooth transitions between screens
    - Professional icons and graphics
    - Responsive layout for different screen sizes

13. **Enhance README documentation**
    - Add screenshots and GIFs of the app in action
    - Include code explanations and customization guides
    - Document example app architecture
    - Add troubleshooting section

## Code Structure

Currently, the example app has a simpler structure:

```
lib/
├── main.dart               # App entry point with two demo screens:
│   ├── CameraInferenceScreen - Real-time detection with device camera
│   └── SingleImageScreen - Detection on images from gallery
```

The planned structure will include more sophisticated components as development continues:

```
lib/
├── main.dart               # App entry point
├── screens/
│   ├── home_screen.dart    # Main task selection screen
│   ├── detection_screen.dart
│   ├── segmentation_screen.dart
│   ├── classification_screen.dart
│   ├── pose_screen.dart
│   ├── obb_screen.dart
│   └── settings_screen.dart
├── widgets/
│   ├── task_card.dart      # Card widget for task selection
│   ├── result_list.dart    # List view for displaying results
│   ├── camera_feed.dart    # Reusable camera feed widget
│   └── visualization/      # Task-specific visualization widgets
├── utils/
│   ├── image_utils.dart    # Image processing utilities
│   └── model_manager.dart  # Model loading and management
└── models/
    └── result_models.dart  # Structured models for results
```

## Feature Roadmap

- [x] Basic object detection with camera
- [x] Object detection with image picker
- [ ] Segmentation visualization
- [ ] Pose estimation visualization
- [ ] Classification results display
- [ ] Oriented bounding box visualization
- [ ] Multiple model support
- [ ] Performance optimization
- [ ] UI polish and documentation

## Contributing

Contributions to improve the example app are welcome. Please feel free to submit pull requests or create issues for bugs or feature requests.