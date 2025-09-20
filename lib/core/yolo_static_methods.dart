// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';

/// Static utility methods for YOLO operations
class YOLOStaticMethods {
  static final _defaultChannel = ChannelConfig.createSingleImageChannel();

  /// Checks if a model exists at the specified path
  ///
  /// [modelPath] The path to check
  /// returns A map containing information about the model existence and location
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

  /// Gets the available storage paths for the app
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
}
