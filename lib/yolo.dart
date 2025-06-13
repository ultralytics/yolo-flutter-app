// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo.dart

import 'dart:async';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_exceptions.dart';
import 'package:ultralytics_yolo/yolo_instance_manager.dart';

/// Exports all YOLO-related classes and enums
export 'yolo_task.dart';
export 'yolo_exceptions.dart';
export 'yolo_result.dart';
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
/// );
///
/// await yolo.loadModel();
/// final results = await yolo.predict(imageBytes);
/// ```
class YOLO {
  // Static channel for backward compatibility
  static const _defaultChannel = MethodChannel('yolo_single_image_channel');

  // Instance-specific properties
  late final String _instanceId;
  late final MethodChannel _channel;
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

  /// The view ID of the associated YoloView (used for model switching)
  int? _viewId;

  /// Creates a new YOLO instance with the specified model path and task.
  ///
  /// The [modelPath] can refer to a model in assets, internal storage, or absolute path.
  /// The [task] specifies what type of inference will be performed.
  ///
  /// If [useMultiInstance] is true, each YOLO instance gets a unique ID and its own channel.
  /// If false, uses the default channel for backward compatibility.
  YOLO({
    required this.modelPath,
    required this.task,
    bool useMultiInstance = false,
  }) {
    if (useMultiInstance) {
      // Generate unique instance ID
      _instanceId = 'yolo_${DateTime.now().millisecondsSinceEpoch}_$hashCode';

      // Create instance-specific channel
      final channelName = 'yolo_single_image_channel_$_instanceId';
      _channel = MethodChannel(channelName);

      // Register this instance with the manager
      YOLOInstanceManager.registerInstance(_instanceId, this);
    } else {
      // Use default values for backward compatibility
      _instanceId = 'default';
      _channel = _defaultChannel;
      _isInitialized = true; // Skip initialization for default mode
    }
  }

  /// Initialize this instance on the platform side
  Future<void> _initializeInstance() async {
    try {
      // Use the default channel to create the instance (only for multi-instance mode)
      if (_instanceId != 'default') {
        await _defaultChannel.invokeMethod('createInstance', {
          'instanceId': _instanceId,
        });
      }
      _isInitialized = true;
    } catch (e) {
      throw ModelLoadingException('Failed to initialize YOLO instance: $e');
    }
  }

  /// Sets the view ID for this controller (called internally by YoloView)
  void setViewId(int viewId) {
    _viewId = viewId;
  }

  /// Switches the model on the associated YoloView.
  ///
  /// This method allows switching to a different model without recreating the view.
  /// The view must be initialized (have a viewId) before calling this method.
  ///
  /// @param newModelPath The path to the new model
  /// @param newTask The task type for the new model
  /// @throws [StateError] if the view is not initialized
  /// @throws [ModelLoadingException] if the model switch fails
  Future<void> switchModel(String newModelPath, YOLOTask newTask) async {
    if (_viewId == null) {
      throw StateError('Cannot switch model: view not initialized');
    }

    try {
      final Map<String, dynamic> arguments = {
        'viewId': _viewId,
        'modelPath': newModelPath,
        'task': newTask.name,
      };

      // Only include instanceId for multi-instance mode
      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      await _channel.invokeMethod('setModel', arguments);
    } on PlatformException catch (e) {
      if (e.code == 'MODEL_NOT_FOUND') {
        throw ModelLoadingException('Model file not found: $newModelPath');
      } else if (e.code == 'INVALID_MODEL') {
        throw ModelLoadingException('Invalid model format: $newModelPath');
      } else if (e.code == 'UNSUPPORTED_TASK') {
        throw ModelLoadingException(
          'Unsupported task type: ${newTask.name} for model: $newModelPath',
        );
      } else {
        throw ModelLoadingException('Failed to switch model: ${e.message}');
      }
    } catch (e) {
      throw ModelLoadingException('Unknown error switching model: $e');
    }
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
  /// @throws [ModelLoadingException] if the model file cannot be found
  /// @throws [PlatformException] if there's an issue with the platform-specific code
  Future<bool> loadModel() async {
    if (!_isInitialized) {
      await _initializeInstance();
    }

    try {
      final Map<String, dynamic> arguments = {
        'modelPath': modelPath,
        'task': task.name,
      };

      // Only include instanceId for multi-instance mode
      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      final result = await _channel.invokeMethod('loadModel', arguments);
      return result == true;
    } on PlatformException catch (e) {
      if (e.code == 'MODEL_NOT_FOUND') {
        throw ModelLoadingException('Model file not found: $modelPath');
      } else if (e.code == 'INVALID_MODEL') {
        throw ModelLoadingException('Invalid model format: $modelPath');
      } else if (e.code == 'UNSUPPORTED_TASK') {
        throw ModelLoadingException(
          'Unsupported task type: ${task.name} for model: $modelPath',
        );
      } else {
        throw ModelLoadingException('Failed to load model: ${e.message}');
      }
    } catch (e) {
      throw ModelLoadingException('Unknown error loading model: $e');
    }
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
  /// @param imageBytes The raw image data as a Uint8List
  /// @param confidenceThreshold Optional confidence threshold (0.0-1.0). Defaults to 0.25 if not specified.
  /// @param iouThreshold Optional IoU threshold for NMS (0.0-1.0). Defaults to 0.4 if not specified.
  /// @return A map containing:
  ///   - 'boxes': List of bounding boxes
  ///   - 'detections': List of YOLOResult-compatible detection maps
  ///   - 'keypoints': (pose only) Raw keypoints data from platform
  /// @throws [ModelNotLoadedException] if the model has not been loaded
  /// @throws [InferenceException] if there's an error during inference
  /// @throws [PlatformException] if there's an issue with the platform-specific code
  Future<Map<String, dynamic>> predict(
    Uint8List imageBytes, {
    double? confidenceThreshold,
    double? iouThreshold,
  }) async {
    if (imageBytes.isEmpty) {
      throw InvalidInputException('Image data is empty');
    }

    // Validate threshold values if provided
    if (confidenceThreshold != null &&
        (confidenceThreshold < 0.0 || confidenceThreshold > 1.0)) {
      throw InvalidInputException(
        'Confidence threshold must be between 0.0 and 1.0',
      );
    }
    if (iouThreshold != null && (iouThreshold < 0.0 || iouThreshold > 1.0)) {
      throw InvalidInputException('IoU threshold must be between 0.0 and 1.0');
    }

    try {
      final Map<String, dynamic> arguments = {'image': imageBytes};

      // Add optional thresholds if provided
      if (confidenceThreshold != null) {
        arguments['confidenceThreshold'] = confidenceThreshold;
      }
      if (iouThreshold != null) {
        arguments['iouThreshold'] = iouThreshold;
      }

      // Only include instanceId for multi-instance mode
      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      final result = await _channel.invokeMethod(
        'predictSingleImage',
        arguments,
      );

      if (result is Map) {
        // Convert Map<Object?, Object?> to Map<String, dynamic>
        final Map<String, dynamic> resultMap = Map<String, dynamic>.fromEntries(
          result.entries.map((e) => MapEntry(e.key.toString(), e.value)),
        );

        // Convert boxes list if it exists
        final List<Map<String, dynamic>> boxes = [];
        if (resultMap.containsKey('boxes') && resultMap['boxes'] is List) {
          boxes.addAll(
            (resultMap['boxes'] as List).whereType<Map>().map((item) {
              return Map<String, dynamic>.fromEntries(
                item.entries.map((e) => MapEntry(e.key.toString(), e.value)),
              );
            }),
          );

          resultMap['boxes'] = boxes;
        }

        // Create detections array with YOLOResult-compatible structure
        final List<Map<String, dynamic>> detections = [];

        // Handle different task types
        switch (task) {
          case YOLOTask.pose:
            // For pose estimation, merge boxes with keypoints
            if (resultMap.containsKey('keypoints')) {
              final keypointsList =
                  resultMap['keypoints'] as List<dynamic>? ?? [];

              for (
                int i = 0;
                i < boxes.length && i < keypointsList.length;
                i++
              ) {
                final box = boxes[i];
                final detection = _createDetectionMap(box);

                // Convert keypoints to flat array format expected by YOLOResult
                if (keypointsList[i] is Map) {
                  final personKeypoints =
                      keypointsList[i] as Map<dynamic, dynamic>;
                  final coordinates =
                      personKeypoints['coordinates'] as List<dynamic>? ?? [];

                  final flatKeypoints = <double>[];
                  for (final coord in coordinates) {
                    if (coord is Map) {
                      flatKeypoints.add(
                        (coord['x'] as num?)?.toDouble() ?? 0.0,
                      );
                      flatKeypoints.add(
                        (coord['y'] as num?)?.toDouble() ?? 0.0,
                      );
                      flatKeypoints.add(
                        (coord['confidence'] as num?)?.toDouble() ?? 0.0,
                      );
                    }
                  }

                  if (flatKeypoints.isNotEmpty) {
                    detection['keypoints'] = flatKeypoints;
                  }
                }

                detections.add(detection);
              }
            }
            break;

          case YOLOTask.segment:
            // For segmentation, include mask data with boxes
            final masks = resultMap['masks'] as List<dynamic>? ?? [];

            for (int i = 0; i < boxes.length; i++) {
              final box = boxes[i];
              final detection = _createDetectionMap(box);

              // Add mask data if available for this detection
              if (i < masks.length && masks[i] != null) {
                // masks[i] is already in the correct List<List<double>> format
                final maskData = masks[i] as List<dynamic>;
                final mask = maskData.map((row) {
                  return (row as List<dynamic>).map((val) {
                    return (val as num).toDouble();
                  }).toList();
                }).toList();

                detection['mask'] = mask;
              }

              detections.add(detection);
            }
            break;

          case YOLOTask.classify:
            // For classification, create a single detection with classification data
            if (resultMap.containsKey('classification')) {
              final classification =
                  resultMap['classification'] as Map<dynamic, dynamic>;

              // Classification doesn't have boxes, create a full-image detection
              final detection = <String, dynamic>{
                'classIndex': 0, // Would need class mapping
                'className': classification['topClass'] ?? '',
                'confidence':
                    (classification['topConfidence'] as num?)?.toDouble() ??
                    0.0,
                'boundingBox': {
                  'left': 0.0,
                  'top': 0.0,
                  'right': 1.0, // Full image
                  'bottom': 1.0,
                },
                'normalizedBox': {
                  'left': 0.0,
                  'top': 0.0,
                  'right': 1.0,
                  'bottom': 1.0,
                },
              };

              detections.add(detection);
            }
            break;

          case YOLOTask.obb:
            // For OBB, convert oriented bounding boxes
            if (resultMap.containsKey('obb')) {
              final obbList = resultMap['obb'] as List<dynamic>? ?? [];

              for (final obb in obbList) {
                if (obb is Map) {
                  final points = obb['points'] as List<dynamic>? ?? [];

                  // Calculate bounding box from OBB points
                  double minX = double.infinity, minY = double.infinity;
                  double maxX = double.negativeInfinity,
                      maxY = double.negativeInfinity;

                  for (final point in points) {
                    if (point is Map) {
                      final x = (point['x'] as num?)?.toDouble() ?? 0.0;
                      final y = (point['y'] as num?)?.toDouble() ?? 0.0;
                      minX = minX > x ? x : minX;
                      minY = minY > y ? y : minY;
                      maxX = maxX < x ? x : maxX;
                      maxY = maxY < y ? y : maxY;
                    }
                  }

                  final detection = <String, dynamic>{
                    'classIndex': 0, // Would need class mapping
                    'className': obb['class'] ?? '',
                    'confidence':
                        (obb['confidence'] as num?)?.toDouble() ?? 0.0,
                    'boundingBox': {
                      'left': minX,
                      'top': minY,
                      'right': maxX,
                      'bottom': maxY,
                    },
                    'normalizedBox': {
                      'left': minX,
                      'top': minY,
                      'right': maxX,
                      'bottom': maxY,
                    },
                  };

                  detections.add(detection);
                }
              }
            }
            break;

          case YOLOTask.detect:
            // For detection, just convert boxes
            for (final box in boxes) {
              detections.add(_createDetectionMap(box));
            }
            break;
        }

        // Add detections to result map
        resultMap['detections'] = detections;

        return resultMap;
      }

      throw InferenceException('Invalid result format returned from inference');
    } on PlatformException catch (e) {
      if (e.code == 'MODEL_NOT_LOADED') {
        throw ModelNotLoadedException(
          'Model has not been loaded. Call loadModel() first.',
        );
      } else if (e.code == 'INVALID_IMAGE') {
        throw InvalidInputException(
          'Invalid image format or corrupted image data',
        );
      } else if (e.code == 'INFERENCE_ERROR') {
        throw InferenceException('Error during inference: ${e.message}');
      } else {
        throw InferenceException(
          'Platform error during inference: ${e.message}',
        );
      }
    } catch (e) {
      throw InferenceException('Unknown error during inference: $e');
    }
  }

  /// Helper method to create a detection map from a box
  Map<String, dynamic> _createDetectionMap(Map<String, dynamic> box) {
    return {
      'classIndex': 0, // Would need class mapping for proper index
      'className': box['class'] ?? '',
      'confidence': box['confidence'] ?? 0.0,
      'boundingBox': {
        'left': box['x1'] ?? 0.0,
        'top': box['y1'] ?? 0.0,
        'right': box['x2'] ?? 0.0,
        'bottom': box['y2'] ?? 0.0,
      },
      'normalizedBox': {
        'left': box['x1_norm'] ?? 0.0,
        'top': box['y1_norm'] ?? 0.0,
        'right': box['x2_norm'] ?? 0.0,
        'bottom': box['y2_norm'] ?? 0.0,
      },
    };
  }

  /// Checks if a model exists at the specified path.
  ///
  /// This method can check for models in assets, internal storage, or at an absolute path.
  ///
  /// @param modelPath The path to check
  /// @return A map containing information about the model existence and location
  static Future<Map<String, dynamic>> checkModelExists(String modelPath) async {
    try {
      final result = await _defaultChannel.invokeMethod('checkModelExists', {
        'modelPath': modelPath,
      });

      if (result is Map) {
        return Map<String, dynamic>.fromEntries(
          result.entries.map((e) => MapEntry(e.key.toString(), e.value)),
        );
      }

      return {'exists': false, 'path': modelPath, 'location': 'unknown'};
    } on PlatformException catch (e) {
      logInfo('Failed to check model existence: ${e.message}');
      return {'exists': false, 'path': modelPath, 'error': e.message};
    } catch (e) {
      logInfo('Error checking model existence: $e');
      return {'exists': false, 'path': modelPath, 'error': e.toString()};
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
      final result = await _defaultChannel.invokeMethod('getStoragePaths');

      if (result is Map) {
        return Map<String, String?>.fromEntries(
          result.entries.map(
            (e) => MapEntry(e.key.toString(), e.value as String?),
          ),
        );
      }

      return {};
    } on PlatformException catch (e) {
      logInfo('Failed to get storage paths: ${e.message}');
      return {};
    } catch (e) {
      logInfo('Error getting storage paths: $e');
      return {};
    }
  }

  /// Disposes this YOLO instance and releases all resources
  Future<void> dispose() async {
    try {
      await _channel.invokeMethod('disposeInstance', {
        'instanceId': _instanceId,
      });
    } catch (e) {
      logInfo('Error disposing instance $_instanceId: $e');
    } finally {
      // Always remove from manager, even if platform call fails
      YOLOInstanceManager.unregisterInstance(_instanceId);
      _isInitialized = false;
    }
  }
}
