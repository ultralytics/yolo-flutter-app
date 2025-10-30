// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';

/// Centralized channel configuration for YOLO operations.
///
/// This class provides a unified way to handle method channel setup,
/// naming conventions, and communication patterns that were previously
/// duplicated across multiple files in the codebase.
class ChannelConfig {
  // Channel name constants
  static const String singleImageChannel = 'yolo_single_image_channel';
  static const String controlChannelPrefix =
      'com.ultralytics.yolo/controlChannel_';
  static const String detectionResultsPrefix =
      'com.ultralytics.yolo/detectionResults_';

  /// Creates a method channel with the standard YOLO naming convention.
  ///
  /// [channelName] The base name for the channel
  /// [instanceId] Optional instance ID for multi-instance support
  /// Returns a properly configured MethodChannel
  static MethodChannel createChannel(String channelName, {String? instanceId}) {
    final fullChannelName = instanceId != null
        ? '${channelName}_$instanceId'
        : channelName;
    return MethodChannel(fullChannelName);
  }

  /// Creates a YOLO single image channel for static operations.
  ///
  /// [instanceId] Optional instance ID for multi-instance support
  /// Returns a MethodChannel configured for single image operations
  static MethodChannel createSingleImageChannel({String? instanceId}) {
    return createChannel(singleImageChannel, instanceId: instanceId);
  }

  /// Creates a YOLO control channel for platform view operations.
  ///
  /// [viewId] The unique view ID for the platform view
  /// Returns a MethodChannel configured for control operations
  static MethodChannel createControlChannel(String viewId) {
    return MethodChannel('$controlChannelPrefix$viewId');
  }

  /// Creates a YOLO detection results event channel.
  ///
  /// [viewId] The unique view ID for the platform view
  /// Returns an EventChannel configured for detection results
  static EventChannel createDetectionResultsChannel(String viewId) {
    return EventChannel('$detectionResultsPrefix$viewId');
  }

  /// Validates method call arguments for common patterns.
  ///
  /// [call] The method call to validate
  /// [requiredKeys] List of required argument keys
  /// Throws ArgumentError if validation fails
  static void validateMethodCall(MethodCall call, List<String> requiredKeys) {
    if (call.arguments is! Map) {
      throw ArgumentError('Method ${call.method} requires Map arguments');
    }

    final args = call.arguments as Map;
    for (final key in requiredKeys) {
      if (!args.containsKey(key)) {
        throw ArgumentError('Method ${call.method} requires argument: $key');
      }
    }
  }

  /// Creates standardized method call arguments for common operations.
  ///
  /// [viewId] Optional view ID for platform view operations
  /// [modelPath] Optional model path for model operations
  /// [task] Optional task name for task-specific operations
  /// [additionalArgs] Additional arguments to include
  /// Returns a Map with standardized argument structure
  static Map<String, dynamic> createStandardArgs({
    int? viewId,
    String? modelPath,
    String? task,
    Map<String, dynamic>? additionalArgs,
  }) {
    final args = <String, dynamic>{};

    if (viewId != null) args['viewId'] = viewId;
    if (modelPath != null) args['modelPath'] = modelPath;
    if (task != null) args['task'] = task;

    if (additionalArgs != null) {
      args.addAll(additionalArgs);
    }

    return args;
  }
}
