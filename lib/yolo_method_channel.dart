// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'yolo_platform_interface.dart';

/// An implementation of [YOLOPlatform] that uses method channels.
///
/// This class provides the default implementation for communicating
/// with platform-specific YOLO implementations through Flutter's
/// method channel API. It handles single image predictions and
/// other static YOLO operations.
///
/// This implementation is automatically registered as the default
/// platform interface and should not be instantiated directly.
class YOLOMethodChannel extends YOLOPlatform {
  /// The method channel used to interact with the native platform.
  ///
  /// This channel is used for single image predictions and other
  /// operations that don't require a platform view.
  @visibleForTesting
  final methodChannel = const MethodChannel('yolo_single_image_channel');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }

  @override
  Future<void> setModel(int viewId, String modelPath, String task) async {
    await methodChannel.invokeMethod<void>('setModel', {
      'viewId': viewId,
      'modelPath': modelPath,
      'task': task,
    });
  }
}
