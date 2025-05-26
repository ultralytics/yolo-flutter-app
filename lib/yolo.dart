// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo.dart

import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_exceptions.dart';

/// Exports all YOLO-related classes and enums
export 'yolo_task.dart';
export 'yolo_exceptions.dart';
export 'yolo_result.dart';

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
  // We'll store a method channel for calling native code
  static const _channel = MethodChannel('yolo_single_image_channel');

  /// Path to the YOLO model file. This can be:
  /// - An asset path (e.g., 'assets/models/yolo11n.tflite')
  /// - An absolute file path (e.g., '/data/user/0/com.example.app/files/models/yolo11n.tflite')
  /// - An internal storage reference (e.g., 'internal://models/yolo11n.tflite')
  ///
  /// The 'internal://' prefix will be resolved to the app's internal storage directory.
  final String modelPath;

  /// The type of task this YOLO model will perform (detection, segmentation, etc.)
  final YOLOTask task;

  /// Creates a new YOLO instance with the specified model path and task.
  ///
  /// The [modelPath] can refer to a model in assets, internal storage, or absolute path.
  /// The [task] specifies what type of inference will be performed.
  YOLO({required this.modelPath, required this.task});

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
    try {
      final result = await _channel.invokeMethod('loadModel', {
        'modelPath': modelPath,
        'task': task.name,
      });
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
  /// The structure of the returned map depends on the [task] type:
  ///
  /// - For detection: Contains 'boxes' with class, confidence, and bounding box coordinates.
  /// - For segmentation: Contains 'boxes' with class, confidence, bounding box coordinates, and mask data.
  /// - For classification: Contains class and confidence information.
  /// - For pose estimation: Contains keypoints information for detected poses.
  /// - For OBB: Contains oriented bounding box coordinates.
  ///
  /// The model must be loaded with [loadModel] before calling this method.
  ///
  /// Example:
  /// ```dart
  /// final results = await yolo.predict(imageBytes);
  /// final boxes = results['boxes'] as List<Map<String, dynamic>>;
  /// for (var box in boxes) {
  ///   print('Class: ${box['class']}, Confidence: ${box['confidence']}');
  /// }
  /// ```
  ///
  /// Returns a map containing the inference results. If inference fails, throws an exception.
  ///
  /// @param imageBytes The raw image data as a Uint8List
  /// @return A map containing the inference results
  /// @throws [ModelNotLoadedException] if the model has not been loaded
  /// @throws [InferenceException] if there's an error during inference
  /// @throws [PlatformException] if there's an issue with the platform-specific code
  Future<Map<String, dynamic>> predict(Uint8List imageBytes) async {
    if (imageBytes.isEmpty) {
      throw InvalidInputException('Image data is empty');
    }

    try {
      final result = await _channel.invokeMethod('predictSingleImage', {
        'image': imageBytes,
      });

      if (result is Map) {
        // Convert Map<Object?, Object?> to Map<String, dynamic>
        final Map<String, dynamic> resultMap = Map<String, dynamic>.fromEntries(
          result.entries.map((e) => MapEntry(e.key.toString(), e.value)),
        );

        // Convert boxes list if it exists
        if (resultMap.containsKey('boxes') && resultMap['boxes'] is List) {
          final List<Map<String, dynamic>> boxes = (resultMap['boxes'] as List)
              .map((item) {
                if (item is Map) {
                  return Map<String, dynamic>.fromEntries(
                    item.entries.map(
                      (e) => MapEntry(e.key.toString(), e.value),
                    ),
                  );
                }
                return <String, dynamic>{};
              })
              .toList();

          resultMap['boxes'] = boxes;
        }

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

  /// Checks if a model exists at the specified path.
  ///
  /// This method can check for models in assets, internal storage, or at an absolute path.
  ///
  /// @param modelPath The path to check
  /// @return A map containing information about the model existence and location
  static Future<Map<String, dynamic>> checkModelExists(String modelPath) async {
    try {
      final result = await _channel.invokeMethod('checkModelExists', {
        'modelPath': modelPath,
      });

      if (result is Map) {
        return Map<String, dynamic>.fromEntries(
          result.entries.map((e) => MapEntry(e.key.toString(), e.value)),
        );
      }

      return {'exists': false, 'path': modelPath, 'location': 'unknown'};
    } on PlatformException catch (e) {
      debugPrint('Failed to check model existence: ${e.message}');
      return {'exists': false, 'path': modelPath, 'error': e.message};
    } catch (e) {
      debugPrint('Error checking model existence: $e');
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
      final result = await _channel.invokeMethod('getStoragePaths');

      if (result is Map) {
        return Map<String, String?>.fromEntries(
          result.entries.map(
            (e) => MapEntry(e.key.toString(), e.value as String?),
          ),
        );
      }

      return {};
    } on PlatformException catch (e) {
      debugPrint('Failed to get storage paths: ${e.message}');
      return {};
    } catch (e) {
      debugPrint('Error getting storage paths: $e');
      return {};
    }
  }
}
