import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'yolo_platform_interface.dart';

/// An implementation of [YoloPlatform] that uses method channels.
class MethodChannelYolo extends YoloPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('yolo_single_image_channel');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
