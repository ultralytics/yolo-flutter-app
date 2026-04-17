// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';

/// Centralized channel naming and construction for YOLO platform interop.
class ChannelConfig {
  static const String singleImageChannel = 'yolo_single_image_channel';
  static const String controlChannelPrefix =
      'com.ultralytics.yolo/controlChannel_';
  static const String detectionResultsPrefix =
      'com.ultralytics.yolo/detectionResults_';

  /// Creates a method channel, suffixed with [instanceId] unless it is the
  /// `default` (single-instance) sentinel.
  static MethodChannel createChannel(String channelName, {String? instanceId}) {
    final fullChannelName = instanceId != null && instanceId != 'default'
        ? '${channelName}_$instanceId'
        : channelName;
    return MethodChannel(fullChannelName);
  }

  /// Channel for single-image inference and static plugin operations.
  static MethodChannel createSingleImageChannel({String? instanceId}) =>
      createChannel(singleImageChannel, instanceId: instanceId);

  /// Control channel for a specific YOLO platform view.
  static MethodChannel createControlChannel(String viewId) =>
      MethodChannel('$controlChannelPrefix$viewId');

  /// Event channel that streams detection results for a specific view.
  static EventChannel createDetectionResultsChannel(String viewId) =>
      EventChannel('$detectionResultsPrefix$viewId');
}
