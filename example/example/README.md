<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Flutter Example App

This example app demonstrates how to use the [Ultralytics YOLO Flutter plugin](https://pub.dev/packages/ultralytics_yolo) (`ultralytics_yolo`) for various computer vision tasks such as [object detection](https://docs.ultralytics.com/tasks/detect/), [segmentation](https://docs.ultralytics.com/tasks/segment/), [classification](https://docs.ultralytics.com/tasks/classify/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), and [oriented bounding box detection](https://docs.ultralytics.com/tasks/obb/).

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

## 🚀 Getting Started

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) installed and configured
- An Android or iOS device/emulator
- [YOLO model files](https://docs.ultralytics.com/models/) (included in the assets)

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

## 📋 Implementation Strategy

This document outlines the step-by-step strategy for enhancing the example app to showcase all features of the [YOLO Flutter plugin](https://pub.dev/packages/ultralytics_yolo).

### Phase 1: Basic Structure and Object Detection

1. **Create a task selection home screen**
   - Card-based UI for selecting different [YOLO tasks](https://docs.ultralytics.com/tasks/)
   - Add descriptive text and icons for each task
   - Add navigation to task-specific screens

2. **Improve existing object detection**
   - Enhance visualization of [bounding boxes](https://www.ultralytics.com/glossary/bounding-box)
   - Add [confidence score](https://www.ultralytics.com/glossary/confidence) display
   - Format detection results in a structured list view
   - Add support for resizing and processing larger images

3. **Add camera feed improvements**
   - Implement camera resolution switching
   - Add front/back camera toggle
   - Add confidence threshold slider
   - Improve [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) display

### Phase 2: Additional YOLO Tasks

4. **Implement segmentation screen**
   - Create a dedicated screen for [instance segmentation](https://docs.ultralytics.com/tasks/segment/)
   - Visualize segmentation masks with adjustable opacity
   - Display class and confidence information
   - Support both camera feed and image picking

5. **Implement classification screen**
   - Create a dedicated screen for [image classification](https://docs.ultralytics.com/tasks/classify/)
   - Show top-N classification results with confidence bars
   - Support both camera feed and image picking

6. **Implement pose estimation screen**
   - Create a dedicated screen for [pose estimation](https://docs.ultralytics.com/tasks/pose/)
   - Visualize keypoints and skeleton connections
   - Add different color schemes for multiple people detection
   - Display confidence scores for each keypoint

7. **Implement oriented bounding box screen**
   - Create a dedicated screen for [OBB detection](https://docs.ultralytics.com/tasks/obb/)
   - Visualize rotated bounding boxes
   - Display orientation angles and dimensions
   - Support both camera feed and image picking

### Phase 3: Settings and Optimizations

8. **Add model settings screen**
   - Create a [model selection](https://docs.ultralytics.com/models/) interface
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
    - Optimize memory usage during [inference](https://www.ultralytics.com/glossary/inference-engine)
    - Add performance metrics display (FPS, latency)

### Phase 4: Documentation and Polish

11. **Enhance in-app documentation**
    - Add explanation screens for each [YOLO task](https://docs.ultralytics.com/tasks/)
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

## 🏗️ Code Structure

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

## 🗺️ Feature Roadmap

- [x] Basic [object detection](https://docs.ultralytics.com/tasks/detect/) with camera
- [x] Object detection with image picker
- [ ] [Segmentation](https://docs.ultralytics.com/tasks/segment/) visualization
- [ ] [Pose estimation](https://docs.ultralytics.com/tasks/pose/) visualization
- [ ] [Classification](https://docs.ultralytics.com/tasks/classify/) results display
- [ ] [Oriented bounding box](https://docs.ultralytics.com/tasks/obb/) visualization
- [ ] Multiple [model support](https://docs.ultralytics.com/models/)
- [ ] Performance optimization
- [ ] UI polish and documentation

## 🤝 Contributing

Contributions to improve the example app are welcome! Whether you're fixing bugs, adding new features, or improving documentation, your help makes this project better for everyone.

- 🐛 **Bug Reports**: Found an issue? [Create an issue](https://github.com/ultralytics/yolo-flutter-app/issues) with details
- 💡 **Feature Requests**: Have ideas for improvements? We'd love to hear them
- 🔧 **Pull Requests**: Ready to contribute code? Submit a pull request
- 📖 **Documentation**: Help improve our guides and examples

Please check our [contributing guidelines](https://docs.ultralytics.com/help/contributing/) before getting started. Join our vibrant community on [Discord](https://discord.com/invite/ultralytics) to connect with other developers and get support!
