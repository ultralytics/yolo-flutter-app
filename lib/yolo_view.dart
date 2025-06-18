// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_view.dart

import 'dart:async';
import 'package:flutter/foundation.dart' show defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';

/// Controller for interacting with a [YOLOView] widget.
///
/// This controller provides methods to adjust detection thresholds
/// and camera settings for real-time object detection. It manages
/// the communication with the native platform views.
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

  @visibleForTesting
  void init(MethodChannel methodChannel, int viewId) =>
      _init(methodChannel, viewId);

  void _init(MethodChannel methodChannel, int viewId) {
    _methodChannel = methodChannel;
    _viewId = viewId;
    _applyThresholds();
  }

  Future<void> _applyThresholds() async {
    if (_methodChannel == null) {
      logInfo(
        'YOLOViewController: Warning - Cannot apply thresholds, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setThresholds', {
        'confidenceThreshold': _confidenceThreshold,
        'iouThreshold': _iouThreshold,
        'numItemsThreshold': _numItemsThreshold,
      });
      logInfo(
        'YOLOViewController: Applied thresholds - confidence: $_confidenceThreshold, IoU: $_iouThreshold, numItems: $_numItemsThreshold',
      );
    } catch (e) {
      logInfo('YOLOViewController: Error applying combined thresholds: $e');
    }
  }

  Future<void> setConfidenceThreshold(double threshold) async {
    final clampedThreshold = threshold.clamp(0.0, 1.0);
    _confidenceThreshold = clampedThreshold;
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('setConfidenceThreshold', {
        'threshold': clampedThreshold,
      });
    } catch (e) {
      logInfo('YOLOViewController: Error applying confidence threshold: $e');
    }
  }

  Future<void> setIoUThreshold(double threshold) async {
    final clampedThreshold = threshold.clamp(0.0, 1.0);
    _iouThreshold = clampedThreshold;
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('setIoUThreshold', {
        'threshold': clampedThreshold,
      });
    } catch (e) {
      logInfo('YOLOViewController: Error applying IoU threshold: $e');
    }
  }

  Future<void> setNumItemsThreshold(int numItems) async {
    final clampedNumItems = numItems.clamp(1, 100);
    _numItemsThreshold = clampedNumItems;
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('setNumItemsThreshold', {
        'numItems': clampedNumItems,
      });
    } catch (e) {
      logInfo('YOLOViewController: Error applying numItems threshold: $e');
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
    return _applyThresholds();
  }

  Future<void> switchCamera() async {
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('switchCamera');
    } catch (e) {
      logInfo('YOLOViewController: Error switching camera: $e');
    }
  }

  Future<void> setZoomLevel(double zoomLevel) async {
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('setZoomLevel', {
        'zoomLevel': zoomLevel,
      });
    } catch (e) {
      logInfo('YoloViewController: Error setting zoom level: $e');
    }
  }

  Future<void> zoomIn() async {
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('zoomIn');
    } catch (e) {
      logInfo('YoloViewController: Error zooming in: $e');
    }
  }

  Future<void> zoomOut() async {
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('zoomOut');
    } catch (e) {
      logInfo('YoloViewController: Error zooming out: $e');
    }
  }

  Future<void> switchModel(String modelPath, YOLOTask task) async {
    if (_methodChannel == null || _viewId == null) {
      logInfo(
        'YoloViewController: Warning - Cannot switch model, view not yet created',
      );
      return;
    }
    try {
      logInfo('YoloViewController: Switching model with viewId: $_viewId');
      await _methodChannel!.invokeMethod('setModel', {
        'modelPath': modelPath,
        'task': task.name,
      });
      logInfo(
        'YoloViewController: Model switched successfully to $modelPath with task ${task.name}',
      );
    } catch (e) {
      logInfo('YoloViewController: Error switching model: $e');
      rethrow;
    }
  }

  Future<void> setStreamingConfig(YOLOStreamingConfig config) async {
    if (_methodChannel == null) {
      logInfo(
        'YOLOViewController: Warning - Cannot set streaming config, view not yet created',
      );
      return;
    }
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

  Future<void> stop() async {
    if (_methodChannel == null) return;
    try {
      await _methodChannel!.invokeMethod('stop');
    } catch (e) {
      logInfo('YOLOViewController: Error stopping camera and inference: $e');
    }
  }

  /// Captures the current camera frame with detection overlays.
  ///
  /// Returns the captured image as a Uint8List (JPEG format) that includes
  /// the camera frame with detection bounding boxes and labels overlaid.
  /// Returns null if capture fails.
  ///
  /// Example:
  /// ```dart
  /// // Capture frame with detection overlays
  /// final imageData = await controller.captureFrame();
  /// if (imageData != null) {
  ///   // Save to file or display
  ///   final image = Image.memory(imageData);
  /// }
  /// ```
  Future<Uint8List?> captureFrame() async {
    if (_methodChannel == null) {
      logInfo(
        'YOLOViewController: Warning - Cannot capture frame, view not yet created',
      );
      return null;
    }
    try {
      final result = await _methodChannel!.invokeMethod<dynamic>(
        'captureFrame',
      );
      if (result is Uint8List) {
        logInfo(
          'YOLOViewController: Frame captured successfully: ${result.length} bytes',
        );
        return result;
      } else {
        logInfo(
          'YOLOViewController: Unexpected capture result type: ${result.runtimeType}',
        );
        return null;
      }
    } catch (e) {
      logInfo('YOLOViewController: Error capturing frame: $e');
      return null;
    }
  }
}

class YOLOView extends StatefulWidget {
  final String modelPath;
  final YOLOTask task;
  final YOLOViewController? controller;
  final String cameraResolution;
  final Function(List<YOLOResult>)? onResult;
  final Function(YOLOPerformanceMetrics)? onPerformanceMetrics;
  final Function(Map<String, dynamic> streamData)? onStreamingData;
  final bool showNativeUI;
  final Function(double zoomLevel)? onZoomChanged;
  final YOLOStreamingConfig? streamingConfig;
  final double confidenceThreshold;
  final double iouThreshold;

  const YOLOView({
    super.key,
    required this.modelPath,
    required this.task,
    this.controller,
    this.cameraResolution = '720p',
    this.onResult,
    this.onPerformanceMetrics,
    this.onStreamingData,
    this.showNativeUI = false,
    this.onZoomChanged,
    this.streamingConfig,
    this.confidenceThreshold = 0.5,
    this.iouThreshold = 0.45,
  });

  @override
  State<YOLOView> createState() => YOLOViewState();
}

class YOLOViewState extends State<YOLOView> {
  StreamSubscription? resultSubscription;
  MethodChannel? methodChannel;
  late YOLOViewController _effectiveController;
  final String _viewId = UniqueKey().toString();
  int? _platformViewId;

  @visibleForTesting
  EventChannel? testEventChannel;

  @override
  void initState() {
    super.initState();
    _effectiveController = widget.controller ?? YOLOViewController();
  }

  @override
  void didUpdateWidget(YOLOView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      _effectiveController = widget.controller ?? YOLOViewController();
      if (_platformViewId != null) {
        _effectiveController.init(methodChannel!, _platformViewId!);
      }
    }
    if (oldWidget.onResult != widget.onResult ||
        oldWidget.onPerformanceMetrics != widget.onPerformanceMetrics ||
        oldWidget.onStreamingData != widget.onStreamingData) {
      if (_platformViewId != null) {
        subscribeToResults();
      }
    }
    if (oldWidget.showNativeUI != widget.showNativeUI) {
      methodChannel?.invokeMethod('setShowUIControls', {
        'show': widget.showNativeUI,
      });
    }
    if (_platformViewId != null &&
        (oldWidget.modelPath != widget.modelPath ||
            oldWidget.task != widget.task)) {
      _effectiveController
          .switchModel(widget.modelPath, widget.task)
          .catchError((e) {
            logInfo('YoloView: Error switching model in didUpdateWidget: $e');
          });
    }
  }

  void subscribeToResults() {
    cancelResultSubscription();
    if (widget.onResult == null &&
        widget.onPerformanceMetrics == null &&
        widget.onStreamingData == null) {
      return;
    }

    final eventChannel =
        testEventChannel ??
        EventChannel('com.ultralytics.yolo/detectionResults_$_viewId');
    resultSubscription = eventChannel.receiveBroadcastStream().listen(
      (dynamic event) {
        if (event is Map) {
          if (widget.onStreamingData != null) {
            widget.onStreamingData!(Map<String, dynamic>.from(event));
          }
          // Always process individual callbacks if they exist
          if (widget.onResult != null && event.containsKey('detections')) {
            widget.onResult!(parseDetectionResults(event));
          }
          if (widget.onPerformanceMetrics != null) {
            widget.onPerformanceMetrics!(
              YOLOPerformanceMetrics.fromMap(Map<String, dynamic>.from(event)),
            );
          }
        }
      },
      onError: (dynamic error) {
        logInfo('Error from detection results stream: $error');
      },
      onDone: () {
        logInfo('YOLOView: Event stream closed for $_viewId');
      },
    );
  }

  void cancelResultSubscription() {
    resultSubscription?.cancel();
    resultSubscription = null;
  }

  List<YOLOResult> parseDetectionResults(Map<dynamic, dynamic> event) {
    final List<dynamic> detectionsData = event['detections'] ?? [];
    try {
      return detectionsData
          .map((detection) => YOLOResult.fromMap(detection))
          .toList();
    } catch (e) {
      logInfo('YOLOView: Error parsing detections list: $e');
      return [];
    }
  }

  void triggerPlatformViewCreated(int id) {
    logInfo(
      'YOLOView: Platform view created with system id: $id, our viewId: $_viewId',
    );
    _platformViewId = id;

    // Initialize method channel if not already done
    methodChannel ??= MethodChannel(
      'com.ultralytics.yolo/controlChannel_$_viewId',
    );

    subscribeToResults();

    logInfo('YoloView: Initializing controller with platform view ID: $id');
    _effectiveController.init(methodChannel!, id);

    methodChannel?.invokeMethod('setShowUIControls', {
      'show': widget.showNativeUI,
    });
    methodChannel?.setMethodCallHandler(handleMethodCall);

    if (widget.streamingConfig != null) {
      _effectiveController.setStreamingConfig(widget.streamingConfig!);
    }
  }

  Future<dynamic> handleMethodCall(MethodCall call) async {
    switch (call.method) {
      case 'onZoomChanged':
        final zoomLevel = call.arguments as double?;
        if (zoomLevel != null) {
          widget.onZoomChanged?.call(zoomLevel);
        }
        break;
      case 'recreateEventChannel':
        subscribeToResults();
        break;
      default:
        logInfo('YOLOView: Unknown method call from native: ${call.method}');
    }
  }

  @override
  void dispose() {
    cancelResultSubscription();
    methodChannel?.setMethodCallHandler(null);
    super.dispose();
  }

  // Getters for testing
  YOLOViewController get effectiveController => _effectiveController;
  StreamSubscription? get currentResultSubscription => resultSubscription;
  MethodChannel? get currentMethodChannel => methodChannel;

  @visibleForTesting
  String get viewIdForTest => _viewId;

  @override
  Widget build(BuildContext context) {
    const viewType = 'com.ultralytics.yolo/YOLOPlatformView';
    final creationParams = <String, dynamic>{
      'modelPath': widget.modelPath,
      'task': widget.task.name,
      'confidenceThreshold': widget.confidenceThreshold,
      'iouThreshold': widget.iouThreshold,
      'numItemsThreshold': _effectiveController.numItemsThreshold,
      'viewId': _viewId,
    };

    if (widget.streamingConfig != null) {
      creationParams['streamingConfig'] = {
        'includeDetections': widget.streamingConfig!.includeDetections,
        'includeClassifications':
            widget.streamingConfig!.includeClassifications,
        'includeProcessingTimeMs':
            widget.streamingConfig!.includeProcessingTimeMs,
        'includeFps': widget.streamingConfig!.includeFps,
        'includeMasks': widget.streamingConfig!.includeMasks,
        'includePoses': widget.streamingConfig!.includePoses,
        'includeOBB': widget.streamingConfig!.includeOBB,
        'includeOriginalImage': widget.streamingConfig!.includeOriginalImage,
        'maxFPS': widget.streamingConfig!.maxFPS,
        'throttleInterval':
            widget.streamingConfig!.throttleInterval?.inMilliseconds,
        'inferenceFrequency': widget.streamingConfig!.inferenceFrequency,
        'skipFrames': widget.streamingConfig!.skipFrames,
      };
    }

    if (defaultTargetPlatform == TargetPlatform.android) {
      return AndroidView(
        viewType: viewType,
        layoutDirection: TextDirection.ltr,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: triggerPlatformViewCreated,
      );
    } else if (defaultTargetPlatform == TargetPlatform.iOS) {
      return UiKitView(
        viewType: viewType,
        layoutDirection: TextDirection.ltr,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: triggerPlatformViewCreated,
      );
    } else {
      return const Center(child: Text('Platform not supported for YOLOView'));
    }
  }

  Future<void> setConfidenceThreshold(double threshold) =>
      _effectiveController.setConfidenceThreshold(threshold);
  Future<void> setIoUThreshold(double threshold) =>
      _effectiveController.setIoUThreshold(threshold);
  Future<void> setNumItemsThreshold(int numItems) =>
      _effectiveController.setNumItemsThreshold(numItems);
  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
    int? numItemsThreshold,
  }) => _effectiveController.setThresholds(
    confidenceThreshold: confidenceThreshold,
    iouThreshold: iouThreshold,
    numItemsThreshold: numItemsThreshold,
  );
  Future<void> switchCamera() => _effectiveController.switchCamera();
  Future<void> setZoomLevel(double zoomLevel) =>
      _effectiveController.setZoomLevel(zoomLevel);
  Future<void> zoomIn() => _effectiveController.zoomIn();
  Future<void> zoomOut() => _effectiveController.zoomOut();
}
