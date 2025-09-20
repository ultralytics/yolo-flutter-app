// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/yolo_instance_manager.dart';
import 'package:ultralytics_yolo/core/yolo_inference.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';

export 'models/yolo_task.dart';
export 'models/yolo_exceptions.dart';
export 'models/yolo_result.dart';
export 'yolo_instance_manager.dart';

/// YOLO (You Only Look Once) is a class that provides machine learning inference
/// capabilities for object detection, segmentation, classification, pose estimation,
/// and oriented bounding box detection.
///
/// This class handles the initialization of YOLO models and provides methods
/// to perform inference on images.
///
/// Example usage:
/// ```dart
/// final yolo = YOLO(
///   modelPath: 'assets/models/yolo11n.tflite',
///   task: YOLOTask.detect,
///   useGpu: false, // Disable GPU for stability on some devices
/// );
///
/// await yolo.loadModel();
/// final results = await yolo.predict(imageBytes);
/// ```
class YOLO {
  late final YOLOInference _inference;
  late final YOLOModelManager _modelManager;
  late final String _instanceId;
  bool _isInitialized = false;

  /// The unique instance ID for this YOLO instance
  String get instanceId => _instanceId;

  /// Path to the YOLO model file. This can be:
  /// - An asset path (e.g., 'assets/models/yolo11n.tflite')
  /// - An absolute file path (e.g., '/data/user/0/com.example.app/files/models/yolo11n.tflite')
  /// - An internal storage reference (e.g., 'internal://models/yolo11n.tflite')
  ///
  /// The 'internal://' prefix will be resolved to the app's internal storage directory.
  final String modelPath;

  /// The type of task this YOLO model will perform (detection, segmentation, etc.)
  final YOLOTask task;

  /// Whether to use GPU acceleration for inference.
  ///
  /// On Android, this controls TensorFlow Lite GPU delegate usage.
  /// On iOS, this controls Core ML GPU usage.
  ///
  /// Default is true for better performance, but can be set to false
  /// for stability on devices where GPU inference causes crashes.
  final bool useGpu;

  /// Classifier options for customizing preprocessing
  final Map<String, dynamic>? classifierOptions;

  /// The view ID of the associated YoloView (used for model switching)
  int? _viewId;

  /// Creates a new YOLO instance with the specified model path and task.
  ///
  /// The [modelPath] can refer to a model in assets, internal storage, or absolute path.
  /// The [task] specifies what type of inference will be performed.
  /// The [useGpu] parameter controls whether to use GPU acceleration (default: true).
  ///
  /// If [useMultiInstance] is true, each YOLO instance gets a unique ID and its own channel.
  /// If false, uses the default channel for backward compatibility.
  YOLO({
    required this.modelPath,
    required this.task,
    this.useGpu = true,
    bool useMultiInstance = false,
    this.classifierOptions,
  }) {
    if (useMultiInstance) {
      _instanceId = 'yolo_${DateTime.now().millisecondsSinceEpoch}_$hashCode';

      YOLOInstanceManager.registerInstance(_instanceId, this);
    } else {
      _instanceId = 'default';
      _isInitialized = true;
    }

    _initializeComponents();
  }

  void _initializeComponents() {
    final channel = ChannelConfig.createSingleImageChannel(
      instanceId: _instanceId,
    );

    _modelManager = YOLOModelManager(
      channel: channel,
      instanceId: _instanceId,
      modelPath: modelPath,
      task: task,
      useGpu: useGpu,
      classifierOptions: classifierOptions,
      viewId: _viewId,
    );

    _inference = YOLOInference(
      channel: channel,
      instanceId: _instanceId,
      task: task,
    );
  }

  void setViewId(int viewId) {
    _viewId = viewId;
    _modelManager.setViewId(viewId);
  }

  /// Switches the model on the associated YoloView.
  ///
  /// This method allows switching to a different model without recreating the view.
  /// The view must be initialized (have a viewId) before calling this method.
  ///
  /// [newModelPath] The path to the new model
  /// [newTask] The task type for the new model
  /// throws [StateError] if the view is not initialized
  /// throws [ModelLoadingException] if the model switch fails
  Future<void> switchModel(String newModelPath, YOLOTask newTask) async {
    await _modelManager.switchModel(newModelPath, newTask);
  }

  /// Loads the YOLO model for inference.
  ///
  /// This method must be called before [predict] to initialize the model.
  /// Returns `true` if the model was loaded successfully, `false` otherwise.
  ///
  /// Example:
  /// ```dart
  /// bool success = await yolo.loadModel();
  /// if (success) {
  ///   print('Model loaded successfully');
  /// } else {
  ///   print('Failed to load model');
  /// }
  /// ```
  ///
  /// throws [ModelLoadingException] if the model file cannot be found
  Future<bool> loadModel() async {
    if (!_isInitialized) {
      _isInitialized = true;
    }
    return await _modelManager.loadModel();
  }

  /// Runs inference on a single image.
  ///
  /// Takes raw image bytes as input and returns a map containing the inference results.
  /// The returned map contains:
  /// - 'boxes': List of detected objects with bounding box coordinates
  /// - 'detections': List of detections in YOLOResult-compatible format
  /// - Task-specific data (keypoints for pose, mask for segmentation, etc.)
  ///
  /// The model must be loaded with [loadModel] before calling this method.
  ///
  /// Example:
  /// ```dart
  /// // Basic detection usage
  /// final results = await yolo.predict(imageBytes);
  /// final boxes = results['boxes'] as List<dynamic>;
  /// for (var box in boxes) {
  ///   print('Class: ${box['class']}, Confidence: ${box['confidence']}');
  /// }
  ///
  /// // Pose estimation with YOLOResult
  /// final results = await yolo.predict(imageBytes);
  /// final detections = results['detections'] as List<dynamic>;
  /// for (var detection in detections) {
  ///   final result = YOLOResult.fromMap(detection);
  ///   if (result.keypoints != null) {
  ///     print('Found ${result.keypoints!.length} keypoints');
  ///     for (int i = 0; i < result.keypoints!.length; i++) {
  ///       final kp = result.keypoints![i];
  ///       final conf = result.keypointConfidences![i];
  ///       print('Keypoint $i: (${kp.x}, ${kp.y}) confidence: $conf');
  ///     }
  ///   }
  /// }
  /// ```
  ///
  /// [imageBytes] The raw image data as a Uint8List
  /// [confidenceThreshold] Optional confidence threshold (0.0-1.0). Defaults to 0.25 if not specified.
  /// [iouThreshold] Optional IoU threshold for NMS (0.0-1.0). Defaults to 0.4 if not specified.
  /// returns A map containing:
  ///   - 'boxes': List of bounding boxes
  ///   - 'detections': List of YOLOResult-compatible detection maps
  ///   - 'keypoints': (pose only) Raw keypoints data from platform
  /// throws [ModelNotLoadedException] if the model has not been loaded
  /// throws [InferenceException] if there's an error during inference
  Future<Map<String, dynamic>> predict(
    Uint8List imageBytes, {
    double? confidenceThreshold,
    double? iouThreshold,
  }) async {
    if (!_isInitialized) {
      await loadModel();
    }
    return await _inference.predict(
      imageBytes,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
    );
  }

  /// Checks if a model exists at the specified path.
  ///
  /// This method can check for models in assets, internal storage, or at an absolute path.
  ///
  /// [modelPath] The path to check
  /// returns A map containing information about the model existence and location
  static Future<Map<String, dynamic>> checkModelExists(String modelPath) async {
    try {
      final channel = ChannelConfig.createSingleImageChannel();
      final result = await channel.invokeMethod('checkModelExists', {
        'modelPath': modelPath,
      });

      if (result is Map) {
        return Map<String, dynamic>.fromEntries(
          result.entries.map((e) => MapEntry(e.key.toString(), e.value)),
        );
      }

      return {'exists': false, 'path': modelPath, 'location': 'unknown'};
    } catch (e) {
      String errorMessage = e.toString();
      if (e is PlatformException && e.message != null) {
        errorMessage = e.message!;
      }
      return {'exists': false, 'path': modelPath, 'error': errorMessage};
    }
  }

  /// Gets the available storage paths for the app.
  ///
  /// Returns a map containing paths to different storage locations:
  /// - 'internal': App's internal storage directory
  /// - 'cache': App's cache directory
  /// - 'external': App's external storage directory (may be null)
  /// - 'externalCache': App's external cache directory (may be null)
  ///
  /// These paths can be used to save or load models.
  static Future<Map<String, String?>> getStoragePaths() async {
    try {
      final channel = ChannelConfig.createSingleImageChannel();
      final result = await channel.invokeMethod('getStoragePaths');

      if (result is Map) {
        return Map<String, String?>.fromEntries(
          result.entries.map(
            (e) => MapEntry(e.key.toString(), e.value as String?),
          ),
        );
      }

      return {};
    } catch (e) {
      return {};
    }
  }

  /// Creates a YOLO instance with classifier options for custom preprocessing
  ///
  /// This constructor is specifically designed for classification models that
  /// need custom preprocessing, such as 1-channel grayscale models.
  ///
  /// Example:
  /// ```dart
  /// final yolo = YOLO.withClassifierOptions(
  ///   modelPath: 'assets/handwriting_model.tflite',
  ///   task: YOLOTask.classify,
  ///   classifierOptions: {
  ///     'enable1ChannelSupport': true,
  ///     'enableColorInversion': true,
  ///     'enableMaxNormalization': true,
  ///     'expectedChannels': 1,
  ///     'expectedClasses': 12,
  ///   },
  /// );
  /// ```
  ///
  /// if need custom Normalization:
  /// ```dart
  ///   final grayscaleOptions = {
  ///   'enableMaxNormalization': false,
  ///   'inputMean': 127.5,
  ///   'inputStd' : 127.5,
  ///   'expectedChannels': 1,
  ///   // labels·expectedClasses (if needed)
  /// };
  ///```

  static YOLO withClassifierOptions({
    required String modelPath,
    required YOLOTask task,
    required Map<String, dynamic> classifierOptions,
    bool useGpu = true,
    bool useMultiInstance = false,
  }) {
    return YOLO(
      modelPath: modelPath,
      task: task,
      useGpu: useGpu,
      useMultiInstance: useMultiInstance,
      classifierOptions: classifierOptions,
    );
  }

  Future<void> dispose() async {
    await _modelManager.dispose();
    YOLOInstanceManager.unregisterInstance(_instanceId);
    _isInitialized = false;
  }
}
