// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/core/yolo_model_resolver.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

/// Controls a [YOLOView] imperatively: thresholds, camera, zoom, streaming.
class YOLOViewController {
  MethodChannel? _methodChannel;
  int? _viewId;
  double _confidenceThreshold = 0.25;
  double _iouThreshold = 0.7;
  int _numItemsThreshold = 30;

  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  bool get isInitialized => _methodChannel != null && _viewId != null;

  YOLOViewController();

  void init(MethodChannel methodChannel, int viewId) {
    _methodChannel = methodChannel;
    _viewId = viewId;
    _invoke('setThresholds', {
      'confidenceThreshold': _confidenceThreshold,
      'iouThreshold': _iouThreshold,
      'numItemsThreshold': _numItemsThreshold,
    });
  }

  Future<T?> _invoke<T>(String method, [Map<String, dynamic>? args]) async {
    final channel = _methodChannel;
    if (channel == null) return null;
    try {
      return await channel.invokeMethod<T>(method, args);
    } catch (e) {
      logInfo('YOLOViewController.$method failed: $e');
      return null;
    }
  }

  Future<void> setConfidenceThreshold(double threshold) async {
    _confidenceThreshold = threshold.clamp(0.0, 1.0);
    await _invoke('setConfidenceThreshold', {
      'threshold': _confidenceThreshold,
    });
  }

  Future<void> setIoUThreshold(double threshold) async {
    _iouThreshold = threshold.clamp(0.0, 1.0);
    await _invoke('setIoUThreshold', {'threshold': _iouThreshold});
  }

  Future<void> setNumItemsThreshold(int numItems) async {
    _numItemsThreshold = numItems.clamp(1, 100);
    await _invoke('setNumItemsThreshold', {'numItems': _numItemsThreshold});
  }

  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
    int? numItemsThreshold,
  }) async {
    if (confidenceThreshold != null) {
      _confidenceThreshold = confidenceThreshold.clamp(0.0, 1.0);
    }
    if (iouThreshold != null) {
      _iouThreshold = iouThreshold.clamp(0.0, 1.0);
    }
    if (numItemsThreshold != null) {
      _numItemsThreshold = numItemsThreshold.clamp(1, 100);
    }
    await _invoke('setThresholds', {
      'confidenceThreshold': _confidenceThreshold,
      'iouThreshold': _iouThreshold,
      'numItemsThreshold': _numItemsThreshold,
    });
  }

  Future<void> switchCamera() => _invoke('switchCamera');

  Future<void> setTorchMode(bool enabled) =>
      _invoke('setTorchMode', {'enabled': enabled});

  Future<void> zoomIn() => _invoke('zoomIn');

  Future<void> zoomOut() => _invoke('zoomOut');

  Future<void> setZoomLevel(double zoomLevel) =>
      _invoke('setZoomLevel', {'zoomLevel': zoomLevel});

  Future<void> switchModel(String modelPath, [YOLOTask? task]) async {
    if (_methodChannel == null || _viewId == null) return;
    final resolvedModel = await YOLOModelResolver.resolve(
      modelPath: modelPath,
      task: task,
    );
    await _invoke('setModel', {
      'modelPath': resolvedModel.modelPath,
      'task': resolvedModel.task.name,
    });
  }

  Future<void> setStreamingConfig(YOLOStreamingConfig config) =>
      _invoke('setStreamingConfig', {
        'includeDetections': config.includeDetections,
        'includeClassifications': config.includeClassifications,
        'includeProcessingTimeMs': config.includeProcessingTimeMs,
        'includeFps': config.includeFps,
        'includeMasks': config.includeMasks,
        'includePoses': config.includePoses,
        'includeOBB': config.includeOBB,
        'includeOriginalImage': config.includeOriginalImage,
        'maxFPS': config.maxFPS,
        'throttleIntervalMs': config.throttleInterval?.inMilliseconds,
        'inferenceFrequency': config.inferenceFrequency,
        'skipFrames': config.skipFrames,
      });

  Future<void> stop() => _invoke('stop');

  Future<void> restartCamera() => _invoke('restartCamera');

  Future<void> setShowUIControls(bool show) =>
      _invoke('setShowUIControls', {'show': show});

  Future<void> setShowOverlays(bool show) =>
      _invoke('setShowOverlays', {'show': show});

  Future<Uint8List?> captureFrame() => _invoke<Uint8List>('captureFrame');
}
