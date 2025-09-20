// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';

/// Centralized error handling utility for YOLO operations.
///
/// This class provides a unified way to handle PlatformExceptions and convert
/// them to appropriate YOLO-specific exceptions.
class YOLOErrorHandler {
  /// Handles PlatformExceptions and converts them to appropriate YOLO exceptions.
  ///
  /// [e] The PlatformException to handle
  /// [context] Optional context string for more specific error messages
  ///
  /// Returns the appropriate YOLOException based on the error code
  static YOLOException handlePlatformException(
    PlatformException e, {
    String? context,
  }) {
    final contextPrefix = context != null ? '$context: ' : '';

    switch (e.code) {
      case 'MODEL_NOT_FOUND':
        return ModelLoadingException(
          '${contextPrefix}Model file not found: ${e.message}',
        );

      case 'INVALID_MODEL':
        return ModelLoadingException(
          '${contextPrefix}Invalid model format: ${e.message}',
        );

      case 'UNSUPPORTED_TASK':
        String taskName = 'unknown';
        if (context != null && context.contains('task ')) {
          final match = RegExp(r'task (\w+)').firstMatch(context);
          if (match != null) {
            taskName = match.group(1) ?? 'unknown';
          }
        }
        return ModelLoadingException(
          '${contextPrefix}Unsupported task type: $taskName',
        );

      case 'MODEL_FILE_ERROR':
        return ModelLoadingException(
          '${contextPrefix}Failed to load model: ${e.message}',
        );

      case 'MODEL_NOT_LOADED':
        return ModelNotLoadedException(
          '${contextPrefix}Model has not been loaded. Call loadModel() first.',
        );

      case 'INVALID_IMAGE':
        return InvalidInputException(
          '${contextPrefix}Invalid image format or corrupted image data',
        );

      case 'IMAGE_LOAD_ERROR':
        return InferenceException(
          '${contextPrefix}Platform error during inference: ${e.message}',
        );

      case 'INFERENCE_ERROR':
        return InferenceException(
          '${contextPrefix}Error during inference: ${e.message}',
        );

      default:
        return InferenceException(
          '${contextPrefix}Platform error: ${e.message}',
        );
    }
  }

  /// Handles generic exceptions and wraps them in appropriate YOLO exceptions.
  ///
  /// [e] The generic exception to handle
  /// [context] Optional context string for more specific error messages
  ///
  /// Returns the appropriate YOLOException
  static YOLOException handleGenericException(dynamic e, {String? context}) {
    final contextPrefix = context != null ? '$context: ' : '';

    if (e is YOLOException) {
      return e;
    }

    if (e.toString().contains('MissingPluginException')) {
      if (context != null && context.contains('load model')) {
        return ModelLoadingException(
          '${contextPrefix}Model loading failed: $e',
        );
      } else if (context != null && context.contains('switch to model')) {
        return ModelLoadingException(
          '${contextPrefix}Model switching failed: $e',
        );
      } else if (context != null && context.contains('predict')) {
        return InferenceException('${contextPrefix}Inference failed: $e');
      }
    }

    return InferenceException('${contextPrefix}Unknown error: $e');
  }

  /// Handles any exception with a custom context message.
  ///
  /// [e] The exception to handle
  /// [context] The context message describing what operation failed
  static YOLOException handleError(dynamic e, String context) {
    if (e is PlatformException) {
      return handlePlatformException(e, context: context);
    }

    return handleGenericException(e, context: context);
  }
}
