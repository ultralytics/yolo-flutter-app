## 0.1.28

- **Breaking**: None - fully backward compatible
- **New Feature**: Enhanced front camera support with improved coordinate transformation
  - Fixed front camera bounding box positioning and alignment issues
  - Improved coordinate system handling for front-facing camera detection
  - Added automatic vertical flipping for bounding boxes, masks, poses, and OBB on front camera
  - Enhanced front camera detection accuracy and visual alignment
- **New Feature**: Advanced frame rate control with YOLOStreamingConfig
  - Added `inferenceFrequency` parameter to control model inference rate (5-30 FPS)
  - Added `maxFPS` parameter to limit output streaming rate
  - Added `skipFrames` parameter for frame skipping control
  - Improved performance optimization for battery saving and resource management
  - Added dynamic frame rate adjustment during runtime
- **Enhancement**: Front camera example with comprehensive controls
  - Created dedicated front camera example demonstrating proper usage
  - Added confidence threshold cycling (0.5 → 0.3 → 0.1 → 0.5)
  - Added frame rate cycling (5 → 10 → 15 → 30 → 5 FPS)
  - Added model reload functionality for troubleshooting
  - Enhanced debugging output for front camera detection issues
- **Enhancement**: Improved camera inference screen with front camera support
  - Set front camera as default for better user experience
  - Added camera indicator showing current camera mode (FRONT/BACK)
  - Added target FPS display alongside current FPS
  - Integrated streaming config for frame rate control
  - Added frame rate control button for dynamic adjustment
- **Bug Fix**: Fixed front camera detection issues in main camera screen
  - Resolved model loading conflicts between front and back camera modes
  - Fixed coordinate transformation for front camera bounding boxes
  - Improved detection sensitivity for front camera usage
- **Documentation**: Added comprehensive front camera usage guide
  - Created FRONT_CAMERA_GUIDE.md with troubleshooting steps
  - Added examples for different frame rate configurations
  - Documented front camera coordinate system handling
  - Provided performance optimization recommendations
- **Internal**: Enhanced debugging and logging for front camera issues
  - Added detailed detection result logging
  - Improved error messages for front camera problems
  - Enhanced performance metrics reporting
  - Better troubleshooting information in debug output

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
