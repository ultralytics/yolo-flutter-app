// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Ultralytics YOLO Flutter Plugin
///
/// A comprehensive Flutter plugin for integrating Ultralytics YOLO computer vision models
/// into mobile applications. This plugin provides real-time object detection, segmentation,
/// classification, pose estimation, and oriented bounding box detection capabilities.
///
/// ## Features
///
/// - **Multiple YOLO Tasks**: Supports all major YOLO tasks including detection, segmentation,
///   classification, pose estimation, and oriented bounding box (OBB) detection
/// - **Real-time Performance**: Optimized for mobile devices with hardware acceleration
/// - **Cross-platform**: Works on both iOS (Core ML) and Android (TensorFlow Lite)
/// - **Easy Integration**: Simple API for both single image prediction and live camera preview
/// - **Customizable**: Adjustable confidence thresholds, IoU thresholds, and detection limits
///
/// ## Getting Started
///
/// ### Installation
///
/// Add this to your package's `pubspec.yaml` file:
///
/// ```yaml
/// dependencies:
///   ultralytics_yolo: ^latest_version
/// ```
///
/// ### Basic Usage
///
/// #### Single Image Prediction
///
/// ```dart
/// import 'package:ultralytics_yolo/ultralytics_yolo.dart';
///
/// final yolo = YOLO();
/// await yolo.loadModel('assets/yolov8n.mlmodel', task: YOLOTask.detect);
/// final results = await yolo.predict(imageBytes);
///
/// for (var detection in results.detections) {
///   print('${detection.className}: ${detection.confidence}');
/// }
/// ```
///
/// #### Live Camera Preview
///
/// ```dart
/// import 'package:ultralytics_yolo/ultralytics_yolo.dart';
///
/// YoloView(
///   modelPath: 'assets/yolov8n.mlmodel',
///   task: YOLOTask.detect,
///   onResult: (List<YOLOResult> results) {
///     // Handle detection results
///   },
/// )
/// ```
///
/// ## Platform Setup
///
/// ### iOS
/// - Add camera usage description to `Info.plist`
/// - Minimum iOS version: 12.0
/// - Models must be in Core ML format (.mlmodel)
///
/// ### Android
/// - Add camera permission to `AndroidManifest.xml`
/// - Minimum Android SDK: 21
/// - Models must be in TensorFlow Lite format (.tflite)
///
/// ## API Overview
///
/// ### Core Classes
///
/// - [YOLO]: Main class for single image predictions
/// - [YoloView]: Widget for live camera preview with detection
/// - [YoloViewController]: Controller for adjusting detection parameters
/// - [YOLOResult]: Detection result containing bounding box, class, and confidence
/// - [YOLODetectionResults]: Collection of detection results with performance metrics
///
/// ### Task Types
///
/// - [YOLOTask.detect]: Object detection with bounding boxes
/// - [YOLOTask.segment]: Instance segmentation with masks
/// - [YOLOTask.classify]: Image classification
/// - [YOLOTask.pose]: Human pose estimation with keypoints
/// - [YOLOTask.obb]: Oriented bounding box detection
///
/// ### Error Handling
///
/// The plugin provides specific exception types for different error scenarios:
/// - [ModelLoadingException]: Model file not found or invalid format
/// - [ModelNotLoadedException]: Attempting prediction without loading model
/// - [InvalidInputException]: Invalid image data provided
/// - [InferenceException]: Error during model inference
///
/// ## Performance Tips
///
/// 1. Use appropriate model sizes (n, s, m, l, x) based on device capabilities
/// 2. Adjust confidence thresholds to balance accuracy and performance
/// 3. Limit maximum detections per frame for better real-time performance
/// 4. Consider using lower resolution inputs for faster processing
///
/// ## License
///
/// This plugin is released under the AGPL-3.0 License.
/// See [LICENSE](https://github.com/ultralytics/yolo-flutter-app/blob/main/LICENSE) for details.
library ultralytics_yolo;

export 'yolo.dart';
export 'yolo_exceptions.dart';
export 'yolo_result.dart';
export 'yolo_task.dart';
export 'yolo_view.dart';
