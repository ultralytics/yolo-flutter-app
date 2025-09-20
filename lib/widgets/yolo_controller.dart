// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

/// Controller for managing YOLO detection settings and camera controls.
class YOLOViewController {
  MethodChannel? _methodChannel;
  int? _viewId;

  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;

  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  bool get isInitialized => _methodChannel != null && _viewId != null;

  void init(MethodChannel methodChannel, int viewId) {
    _methodChannel = methodChannel;
    _viewId = viewId;
    _applyThresholds();
  }

  Future<void> _applyThresholds() async {
    if (_methodChannel == null) return;

    try {
      await _methodChannel!.invokeMethod('setThresholds', {
        'confidenceThreshold': _confidenceThreshold,
        'iouThreshold': _iouThreshold,
        'numItemsThreshold': _numItemsThreshold,
      });
    } catch (e) {
      logInfo('YOLOViewController: Error applying thresholds: $e');
    }
  }

  Future<void> setConfidenceThreshold(double threshold) async {
    _confidenceThreshold = threshold.clamp(0.0, 1.0);
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setConfidenceThreshold', {
          'threshold': _confidenceThreshold,
        });
      } catch (e) {
        logInfo('YOLOViewController: Error setting confidence threshold: $e');
        // Fallback to _applyThresholds if individual method fails
        await _applyThresholds();
      }
    }
  }

  Future<void> setIoUThreshold(double threshold) async {
    _iouThreshold = threshold.clamp(0.0, 1.0);
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setIoUThreshold', {
          'threshold': _iouThreshold,
        });
      } catch (e) {
        logInfo('YOLOViewController: Error setting IoU threshold: $e');
        // Fallback to _applyThresholds if individual method fails
        await _applyThresholds();
      }
    }
  }

  Future<void> setNumItemsThreshold(int numItems) async {
    _numItemsThreshold = numItems.clamp(1, 100);
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setNumItemsThreshold', {
          'numItems': _numItemsThreshold,
        });
      } catch (e) {
        logInfo('YOLOViewController: Error setting num items threshold: $e');
        // Fallback to _applyThresholds if individual method fails
        await _applyThresholds();
      }
    }
  }

  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
    int? numItemsThreshold,
  }) async {
    if (confidenceThreshold != null) {
      _confidenceThreshold = confidenceThreshold.clamp(0.0, 1.0);
    }
    if (iouThreshold != null) _iouThreshold = iouThreshold.clamp(0.0, 1.0);
    if (numItemsThreshold != null) {
      _numItemsThreshold = numItemsThreshold.clamp(1, 100);
    }
    await _applyThresholds();
  }

  Future<void> switchCamera() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('switchCamera');
      } catch (e) {
        logInfo('YOLOViewController: Error switching camera: $e');
      }
    }
  }

  Future<void> zoomIn() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('zoomIn');
      } catch (e) {
        logInfo('YOLOViewController: Error zooming in: $e');
      }
    }
  }

  Future<void> zoomOut() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('zoomOut');
      } catch (e) {
        logInfo('YOLOViewController: Error zooming out: $e');
      }
    }
  }

  Future<void> setZoomLevel(double zoomLevel) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setZoomLevel', {
          'zoomLevel': zoomLevel,
        });
      } catch (e) {
        logInfo('YOLOViewController: Error setting zoom level: $e');
      }
    }
  }

  Future<void> switchModel(String modelPath, YOLOTask task) async {
    if (_methodChannel != null && _viewId != null) {
      try {
        await _methodChannel!.invokeMethod('setModel', {
          'modelPath': modelPath,
          'task': task.name,
        });
      } catch (e) {
        logInfo('YOLOViewController: Error switching model: $e');
        rethrow;
      }
    }
  }

  Future<void> setStreamingConfig(YOLOStreamingConfig config) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setStreamingConfig', {
          'includeDetections': config.includeDetections,
          'includeClassifications': config.includeClassifications,
          'includeProcessingTimeMs': config.includeProcessingTimeMs,
          'includeFps': config.includeFps,
          'includeMasks': config.includeMasks,
          'includePoses': config.includePoses,
          'includeOBB': config.includeOBB,
          'includeOriginalImage': config.includeOriginalImage,
          'maxFPS': config.maxFPS,
          'throttleInterval': config.throttleInterval?.inMilliseconds,
          'inferenceFrequency': config.inferenceFrequency,
          'skipFrames': config.skipFrames,
        });
      } catch (e) {
        logInfo('YOLOViewController: Error setting streaming config: $e');
      }
    }
  }

  Future<void> stop() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('stop');
      } catch (e) {
        logInfo('YOLOViewController: Error stopping: $e');
      }
    }
  }

  Future<void> setShowUIControls(bool show) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setShowUIControls', {'show': show});
      } catch (e) {
        logInfo('YOLOViewController: Error setting UI controls: $e');
      }
    }
  }

  Future<Uint8List?> captureFrame() async {
    if (_methodChannel == null) return null;

    try {
      final result = await _methodChannel!.invokeMethod<dynamic>(
        'captureFrame',
      );
      return result is Uint8List ? result : null;
    } catch (e) {
      logInfo('YOLOViewController: Error capturing frame: $e');
      return null;
    }
  }
}
