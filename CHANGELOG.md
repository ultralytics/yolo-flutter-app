## 0.1.36

- **New Feature**: implement UI controls on Android platform

## 0.1.35

- **Bug Fix**: Fix iOS numItemsThreshold inconsistency with Android

## 0.1.34

- **Bug Fix**: Fix originalImage being null in onStreamingData callback
  - Fixed issue where `originalImage` was consistently null despite `includeOriginalImage: true` being set
  - Capture original image data in BasePredictor before clearing currentBuffer

## 0.1.33

- **Bug Fix**: Fix setState callback issue in YOLOView
  - Fixed issue where `onResult` and `onStreamingData` callbacks would stop working after `setState` calls
  - Improved `didUpdateWidget` logic to prevent unnecessary subscription recreation when callbacks are functionally equivalent
  - Added subscription existence check to avoid recreating working subscriptions
  - Fixed Android compilation errors related to YOLOStreamConfig and method access

## 0.1.32

- **Bug Fix**: Fix Android crash in YOLOInstanceManager
  - Added compatibility methods for better API consistency

## 0.1.31

- **New Feature**: Add `useGpu` parameter for GPU acceleration control
  - Added `useGpu` parameter to `YOLO` constructor and `YOLOView` widget
  - Allows disabling GPU acceleration on problematic devices to prevent crashes
  - Supports fallback to CPU/NNAPI on Android and CPU-only mode on iOS
  - Cross-platform consistency: parameter works on both Android and iOS
  - Default value is `true` for backward compatibility
  - Added comprehensive example demonstrating GPU control and error handling
  - Updated documentation with usage examples and troubleshooting guide

**API Changes:**

- `YOLO` constructor now accepts optional `useGpu: bool = true` parameter
- `YOLOView` widget now accepts optional `useGpu: bool = true` parameter
- `YOLO.withClassifierOptions` constructor now accepts optional `useGpu: bool = true` parameter

## 0.1.30

- Remove pubspec models

## 0.1.29

- **Bug Fix**: Clarified model placement requirements for Android
  - For Android, models **must** be placed in `android/app/src/main/assets/` due to TensorFlow Lite's asset loading limitations.
  - For iOS, add models to the Xcode project as before.
  - The code changes to YOLOFileUtils.kt for path prefix stripping are not used in the current loading flow.

## 0.1.28

- **Enhancement**: Smart label positioning for detection results
  - Labels for detected objects are now always visible and never cut off at the edges of the screen
  - Implemented boundary checks in all directions (top, bottom, left, right)
  - By default, labels are placed above the box; if off-screen, they are placed inside the box
  - Implemented as a reusable, high-performance function
  - Consistent behavior on both iOS and Android platforms
- **Bug Fix**: Accurate coordinate handling for all model sizes
  - Fixed bounding box and keypoint placement for all YOLO tasks (detect, segment, pose, OBB) when using models with input sizes other than 640x640

**Targeted tasks:**

- ✅ Object Detection (DETECT)
- ✅ Instance Segmentation (SEGMENT)
- ✅ Pose Estimation (POSE)
- ✅ Oriented Bounding Box (OBB)
- ⚠️ Classification (CLASSIFY) - Not applicable for label placement

**Upgrade note:**

- Labels for all detection results are now always visible and neatly placed
- Coordinate handling is robust for all model sizes and tasks
- Example app is more flexible and user-friendly

## 0.1.27

- **Breaking**: None - fully backward compatible
- **Bug Fix**: Fix iOS segmentation mask alignment issue
  - Masks now correctly align with detected objects in both portrait and landscape modes
  - Removed explicit `contentsGravity` settings that caused mask stretching
  - Simplified mask positioning to match yolo-ios-app reference implementation
- **Enhancement**: Add mask layer frame update during orientation changes
- **Internal**: Remove unnecessary margin calculations for mask positioning

## 0.1.26

- **Breaking**: None - fully backward compatible
- **New Feature**: Add frame capture functionality with detection overlays
  - Capture camera frames with bounding boxes, masks, poses, and other overlays
  - Save captured images to device gallery or share with other apps
  - Support for all YOLO tasks (detect, segment, pose, classify, OBB)
  - New `captureFrame()` method in YOLOViewController returns JPEG image data
- **Enhancement**: iOS capture includes all overlay types (masks, poses, OBB)
- **Enhancement**: Android capture with multiple fallback methods for reliability
- **Documentation**: Added comprehensive frame capture API documentation

## 0.1.25

- **Breaking**: None - fully backward compatible
- **New Feature**: Enable camera preview without valid model path
  - YOLOView now starts with camera-only mode when model is unavailable
  - Graceful error handling instead of crashes on both iOS and Android
- **New Feature**: Add dynamic model switching via `switchModel()` method
  - Switch between different models without restarting camera
  - Enables progressive model loading and A/B testing scenarios
- **Enhancement**: Improved error messages and logging for model loading failures
- **Documentation**: Added comprehensive examples for new features

## 0.1.24

- Fix Android landscape orientation coordinate mapping issue
- Add device orientation detection for proper image rotation
- Implement separate image processors for portrait/landscape modes
- Correct aspect ratio calculations for all YOLO tasks in landscape mode

## 0.1.23

- Add Support for Landscape Mode

## 0.1.22

- Fixed critical memory leaks in iOS YOLOView disposal and model switching
- Added proper dispose implementation for YOLOView on both iOS and Android platforms
- Fixed native rendering issues for detection visualization
- Fixed Android model label loading issues
- Enhanced single image inference result updates
- Improved resource cleanup when switching between models or tasks

## 0.1.21

- Merge example READMEs
- Rename `example/example.dart` to `example/main.dart`

## 0.1.20

- Added `example/example.dart` for usage demonstration.

## 0.1.19

- Added Dart publish dry run to CI
- Renamed incorrect docs/ directory to /doc

## 0.1.18

- Added customizable result streaming with `YOLOStreamingConfig`
  - Enable detailed control based on streaming mode
  - Enable throttling and frame dropping for performance optimization
  - Added optional support for mask and pose data in results
- Added multi-instance YOLO model support
  - Run multiple YOLO models simultaneously
  - Independent configuration for each instance
  - Efficient resource management across instances
- Enhanced Swift backward compatibility
  - Improved support for older iOS versions
  - Better compatibility with legacy Swift code
- Updated documentation
  - Added comprehensive model integration guide
  - Improved API documentation
  - Enhanced troubleshooting section

## 0.1.17

- Improved publish workflow robustness.

## 0.1.16

- Fixed publishing workflows for non-sequential version numbers.

## 0.1.15

- Added `example/main.dart` for usage demonstration.

## 0.1.13

- Updated publishing workflows.

## 0.1.12

- Added `example/main.dart` for usage demonstration.
- Created `shared_main.dart` to eliminate duplication between `example.dart` and `main.dart`.
- Resolved pub.dev warning: “No example found.”
- Improved `pubspec.yaml` to explicitly point to the example file.

## 0.1.9

- Simplified package publishing workflow
- Removed Python-based version check in favor of direct pubspec.yaml version reading
- Improved GitHub Actions workflow reliability
- Fixed tag management and release process

## 0.1.8

- Add optional confidence and IoU thresholds for single image inference
  - Thresholds can be passed to `predict()` method for temporary use
  - Does not affect subsequent predictions or camera inference
  - Useful for fine-tuning detection sensitivity per image

## 0.1.7

- Updated package topics to comply with pub.dev requirements
- Improved package validation and documentation

## 0.1.6

- Fixed CI/CD pipeline issues for pub.dev publishing

## 0.1.5

- Updated package validation and documentation
- Improved error handling and logging
- Added support for multiple model types:
  - Object Detection (YOLOv11)
  - Pose Estimation
  - Image Segmentation
  - Oriented Bounding Box (OBB) Detection
  - Image Classification
- Enhanced camera functionality:
  - Camera flipping between front and back cameras
  - Camera zooming with pinch gestures
  - Improved camera preview quality
- Updated package validation and documentation
- Improved error handling and logging
- Added comprehensive example app showcasing all features
- Enhanced documentation with detailed usage examples

## 0.1.4

- Fixed front camera orientation issue on Android where detection results were displayed upside down.
- Fixed vertical flipping for bounding boxes, segmentation masks, pose keypoints, and OBB (oriented bounding boxes) when using front camera.
- Added proper canvas transformations for segmentation mask rendering with front camera.
- Improved overall detection accuracy and visual alignment for front-facing camera usage.

## 0.1.3

- Added camera switching functionality to toggle between front and back cameras.
- Added `switchCamera()` method to YoloViewController for programmatic camera switching.
- Added `switchCamera()` method to YoloViewState for GlobalKey-based camera switching.
- Updated sample app with camera switching button in the app bar.
- Updated README documentation with examples of camera switching functionality.
- Improved code coverage with additional unit tests.
- Updated codecov badge to show coverage percentage.

## 0.1.2

- Android: Fixed pose estimation keypoints not displaying correctly by properly implementing object pooling in PoseEstimator.kt.
- Android: Improved segmentation to work with all model classes, not just early ones like "person" and "car".
- Android: Enhanced model metadata loading to extract labels from model files with fallback to COCO dataset classes.
- Android: Fixed lifecycle management in YoloView.kt with proper onLifecycleOwnerAvailable implementation.
- Android: Made Box class fields mutable (var instead of val) to properly support object pooling.
- Performance: Various optimizations for faster inference and more reliable detection.

## 0.1.0

- iOS: Implemented direct FPS (Frames Per Second) reporting to Flutter, similar to Android. Native-calculated FPS is now included in the data sent to Dart during real-time inference.
- Android: Fixed an issue where the camera preview would remain black by improving native lifecycle management and camera initialization timing. (Previously part of 0.0.9 prep)
- Android: Added detailed debug logs to `YoloPlatformView` initialization. (Previously part of 0.0.9 prep)
- `lib/yolo_view.dart`: Added debug logs for communication channel creation and improved null checks. (Previously part of 0.0.9 prep)
- `.pubignore`: Updated to optimize the content of the published package. (Previously part of 0.0.9 prep)
- General: Incorporated various improvements from previous development versions (including enhanced model path resolution and logging). (Previously part of 0.0.9 prep)

## 0.0.9

- Android: Fixed an issue where the camera preview would remain black by improving native lifecycle management and camera initialization timing.
- Android: Added detailed debug logs to `YoloPlatformView` initialization for easier troubleshooting.
- `lib/yolo_view.dart`: Added debug logs for communication channel creation and improved null checks.
- `.pubignore`: Updated to optimize the content of the published package.
- General: Incorporated various improvements from previous development versions (including enhanced model path resolution and logging).

## 0.0.7

- Fix Android implementation for inference results not displaying or updating
- Fix "Unresolved reference: setIoUThreshold" error by fixing method name casing
- Add support for both "setIoUThreshold" and "setIouThreshold" method names for robustness
- Enhance error handling and logging for event channel communication
- Improve StreamHandler implementation for more reliable event dispatching
- Add fallback mechanisms for when direct method calls fail
- Fix reflection-based sink access for CustomStreamHandler
- Add test message mechanism to verify event channel connection
- Significantly increase logging for easier troubleshooting
- Update documentation with clear guidance on model placement and path resolution
- Recommend using model name only (without extension) for best cross-platform compatibility

## 0.0.8

- Fix iOS implementation for loading .mlmodel files from Flutter assets
- Significantly improve model path resolution for different path formats
- Add extensive logging to help debug model loading issues
- Fix Flutter asset bundle path issues with nested directories

## 0.0.7

- Fix iOS implementation to properly load models from Flutter assets
- Improve asset path resolution for paths like 'assets/models/yolo11n.mlmodel'
- Fix syntax errors in YoloPlugin.swift

## 0.0.6

- Add iOS implementation for checkModelExists method
- Add iOS implementation for getStoragePaths method
- Fix cross-platform consistency for model path resolution

## 0.0.5

- Update README to match current implementation of YOLO class constructor
- Fix documentation for threshold management in the API reference
- Add optional controller-based approach for managing YoloView settings
- Make onResult callback truly optional
- Improve threshold controls with IoU threshold support
- Update code documentation with detailed examples
- Add support for direct YoloView state access via GlobalKey
- Enhance error handling and debug logging
- Translate Japanese comments to English

## 0.0.4

- Initial release
- Object detection with YOLOv8 models
- Segmentation support
- Image classification support
- Pose estimation support
- Oriented Bounding Box (OBB) detection support
- Android/iOS platform support
- Real-time detection with camera feed
- Customizable confidence threshold
- YoloView Flutter widget implementation
