// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart' show defaultTargetPlatform, listEquals;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/widgets/yolo_overlay.dart';
import 'package:ultralytics_yolo/models/yolo_model_spec.dart';

/// A Flutter widget that displays a real-time camera preview with YOLO object detection.
class YOLOView extends StatefulWidget {
  final List<YOLOModelSpec> models;
  final YOLOViewController? controller;
  final String cameraResolution;
  final Function(List<YOLOResult>)? onResult;
  final Function(YOLOPerformanceMetrics)? onPerformanceMetrics;
  final Function(Map<String, dynamic>)? onStreamingData;
  final bool showNativeUI;
  final Function(double zoomLevel)? onZoomChanged;
  final YOLOStreamingConfig? streamingConfig;
  final double confidenceThreshold;
  final double iouThreshold;
  final bool useGpu;
  final bool showOverlays;
  final YOLOOverlayTheme overlayTheme;

  const YOLOView({
    super.key,
    required this.models,
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
    this.useGpu = true,
    this.showOverlays = true,
    this.overlayTheme = const YOLOOverlayTheme(),
  });

  @override
  State<YOLOView> createState() => _YOLOViewState();
}

class _YOLOViewState extends State<YOLOView> {
  late YOLOViewController _effectiveController;
  late MethodChannel _methodChannel;
  late EventChannel _resultEventChannel;
  StreamSubscription<dynamic>? _resultSubscription;

  final String _viewId = UniqueKey().toString();
  int? _platformViewId;
  List<YOLOResult> _currentDetections = [];

  @override
  void initState() {
    super.initState();
    _setupController();
    _setupChannels();
  }

  void _setupController() {
    _effectiveController = widget.controller ?? YOLOViewController();
  }

  void _setupChannels() {
    _methodChannel = ChannelConfig.createControlChannel(_viewId);
    _resultEventChannel = ChannelConfig.createDetectionResultsChannel(_viewId);
  }

  void _subscribeToResults() {
    if (widget.onResult == null &&
        widget.onPerformanceMetrics == null &&
        widget.onStreamingData == null) {
      return;
    }

    _resultSubscription = _resultEventChannel.receiveBroadcastStream().listen(
      _handleEvent,
      onError: (error, stackTrace) {
        logInfo('YOLOView: Stream error: $error');
        _resubscribeAfterDelay();
      },
      onDone: () {
        logInfo('YOLOView: Stream closed');
        _resultSubscription = null;
      },
    );
  }

  void _resubscribeAfterDelay() {
    Timer(const Duration(seconds: 2), () {
      if (mounted && _resultSubscription == null) {
        _subscribeToResults();
      }
    });
  }

  void _handleEvent(dynamic event) {
    if (event is! Map) return;

    if (widget.onStreamingData != null) {
      try {
        final streamData = MapConverter.convertToTypedMap(event);
        widget.onStreamingData!(streamData);
      } catch (e) {
        logInfo('YOLOView: Error processing streaming data: $e');
      }
    } else {
      _handleDetectionResults(event);
      _handlePerformanceMetrics(event);
    }
  }

  void _handleDetectionResults(Map<dynamic, dynamic> event) {
    if (widget.onResult == null || !event.containsKey('detections')) return;

    try {
      final results = _parseDetectionResults(event);
      setState(() {
        _currentDetections = results;
      });
      widget.onResult!(results);
    } catch (e) {
      logInfo('YOLOView: Error parsing detection results: $e');
    }
  }

  void _handlePerformanceMetrics(Map<dynamic, dynamic> event) {
    if (widget.onPerformanceMetrics == null) return;

    try {
      final metrics = YOLOPerformanceMetrics.fromMap(
        MapConverter.convertToTypedMap(event),
      );
      widget.onPerformanceMetrics!(metrics);
    } catch (e) {
      logInfo('YOLOView: Error parsing performance metrics: $e');
    }
  }

  List<YOLOResult> _parseDetectionResults(Map<dynamic, dynamic> event) {
    final List<dynamic> detectionsData = event['detections'] ?? [];
    final results = <YOLOResult>[];

    for (final detection in detectionsData) {
      if (detection is! Map) continue;

      // Validate required fields
      if (!detection.containsKey('classIndex') ||
          !detection.containsKey('className') ||
          !detection.containsKey('confidence') ||
          !detection.containsKey('boundingBox') ||
          !detection.containsKey('normalizedBox')) {
        continue;
      }

      // Validate non-null values
      if (detection['classIndex'] == null ||
          detection['className'] == null ||
          detection['confidence'] == null ||
          detection['boundingBox'] == null ||
          detection['normalizedBox'] == null) {
        continue;
      }

      try {
        final result = YOLOResult.fromMap(detection);
        results.add(result);
      } catch (e) {
        logInfo('YOLOView: Error parsing detection: $e');
      }
    }

    return results;
  }

  @override
  void didUpdateWidget(YOLOView oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Handle controller changes
    if (oldWidget.controller != widget.controller) {
      _setupController();
    }

    // Handle callback changes
    final callbacksChanged =
        (oldWidget.onResult == null) != (widget.onResult == null) ||
        (oldWidget.onPerformanceMetrics == null) !=
            (widget.onPerformanceMetrics == null) ||
        (oldWidget.onStreamingData == null) != (widget.onStreamingData == null);

    if (callbacksChanged) {
      _resultSubscription?.cancel();
      _resultSubscription = null;
      _subscribeToResults();
    }

    // Handle models changes
    if (_platformViewId != null &&
        !listEquals(oldWidget.models, widget.models)) {
      _effectiveController.switchModels(widget.models);
    }
  }

  @override
  void dispose() {
    _effectiveController.stop();
    _resultSubscription?.cancel();
    _methodChannel.setMethodCallHandler(null);

    if (_platformViewId != null) {
      ChannelConfig.createSingleImageChannel()
          .invokeMethod('disposeInstance', {'instanceId': _viewId})
          .catchError((e) => logInfo('YOLOView: Error disposing model: $e'));
    }

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        _buildCameraView(),
        if (widget.showOverlays && _currentDetections.isNotEmpty)
          LayoutBuilder(
            builder: (context, constraints) {
              final viewSize = Size(
                constraints.maxWidth,
                constraints.maxHeight,
              );
              final transformed = _transformDetectionsForView(viewSize);
              return YOLOOverlay(
                detections: transformed,
                showConfidence: true,
                showClassName: true,
                theme: widget.overlayTheme,
                onDetectionTap: (detection) {
                  logInfo('YOLOView: Detection tapped: ${detection.className}');
                },
              );
            },
          ),
      ],
    );
  }

  Widget _buildCameraView() {
    const viewType = 'com.ultralytics.yolo/YOLOPlatformView';
    final creationParams = _buildCreationParams();

    if (defaultTargetPlatform == TargetPlatform.android) {
      return AndroidView(
        viewType: viewType,
        layoutDirection: TextDirection.ltr,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: _onPlatformViewCreated,
      );
    } else if (defaultTargetPlatform == TargetPlatform.iOS) {
      return UiKitView(
        viewType: viewType,
        layoutDirection: TextDirection.ltr,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: _onPlatformViewCreated,
      );
    } else {
      return const Center(child: Text('Platform not supported for YOLOView'));
    }
  }

  Map<String, dynamic> _buildCreationParams() {
    final creationParams = <String, dynamic>{
      'models': widget.models.map((m) => m.toMap()).toList(),
      'confidenceThreshold': widget.confidenceThreshold,
      'iouThreshold': widget.iouThreshold,
      'numItemsThreshold': _effectiveController.numItemsThreshold,
      'viewId': _viewId,
      'useGpu': widget.useGpu,
      'showOverlays': widget.showOverlays,
    };

    if (widget.streamingConfig != null) {
      final streamConfig = <String, dynamic>{
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
      };

      if (widget.streamingConfig!.maxFPS != null) {
        streamConfig['maxFPS'] = widget.streamingConfig!.maxFPS;
      }
      if (widget.streamingConfig!.throttleInterval != null) {
        streamConfig['throttleIntervalMs'] =
            widget.streamingConfig!.throttleInterval!.inMilliseconds;
      }
      if (widget.streamingConfig!.inferenceFrequency != null) {
        streamConfig['inferenceFrequency'] =
            widget.streamingConfig!.inferenceFrequency;
      }
      if (widget.streamingConfig!.skipFrames != null) {
        streamConfig['skipFrames'] = widget.streamingConfig!.skipFrames;
      }

      creationParams['streamingConfig'] = streamConfig;
    }

    return creationParams;
  }

  void _onPlatformViewCreated(int id) {
    _platformViewId = id;
    _effectiveController.init(_methodChannel, id);
    _methodChannel.setMethodCallHandler(_handleMethodCall);
    _methodChannel.invokeMethod('setShowUIControls', {
      'show': widget.showNativeUI,
    });

    if (widget.streamingConfig != null) {
      _effectiveController.setStreamingConfig(widget.streamingConfig!);
    }

    if (widget.onResult != null ||
        widget.onPerformanceMetrics != null ||
        widget.onStreamingData != null) {
      _subscribeToResults();
    }
  }

  Future<dynamic> _handleMethodCall(MethodCall call) async {
    switch (call.method) {
      case 'recreateEventChannel':
        _resultSubscription?.cancel();
        _resultSubscription = null;
        Timer(const Duration(milliseconds: 100), () {
          if (mounted) _subscribeToResults();
        });
        return null;
      case 'onZoomChanged':
        final zoomLevel = call.arguments as double?;
        if (zoomLevel != null && widget.onZoomChanged != null) {
          widget.onZoomChanged!(zoomLevel);
        }
        return null;
      default:
        return null;
    }
  }

  // Helper: transform detections from image-space to view-space using normalizedBox and center-crop
  List<YOLOResult> _transformDetectionsForView(Size viewSize) {
    if (_currentDetections.isEmpty) return _currentDetections;

    // Try to estimate source image width/height from first detection using normalized vs pixel box
    double? iw;
    double? ih;
    for (final d in _currentDetections) {
      final nb = d.normalizedBox;
      final bb = d.boundingBox;
      final nw = (nb.right - nb.left).abs();
      final nh = (nb.bottom - nb.top).abs();
      if (nw > 0 && nh > 0 && bb.width > 0 && bb.height > 0) {
        iw = bb.width / nw;
        ih = bb.height / nh;
        break;
      }
    }
    if (iw == null || ih == null || iw <= 0 || ih <= 0) {
      // Fallback: no transform
      return _currentDetections;
    }

    final vw = viewSize.width;
    final vh = viewSize.height;
    if (vw <= 0 || vh <= 0) return _currentDetections;

    final scale = math.max(vw / iw, vh / ih);
    final scaledW = iw * scale;
    final scaledH = ih * scale;
    final dx = (vw - scaledW) / 2.0;
    final dy = (vh - scaledH) / 2.0;

    Rect _toViewRect(Rect imgRect) {
      double left = imgRect.left * scale + dx;
      double top = imgRect.top * scale + dy;
      double right = imgRect.right * scale + dx;
      double bottom = imgRect.bottom * scale + dy;

      // If front camera mirroring is needed, we would flip horizontally here.
      // Without an explicit signal from native, keep as-is to avoid double flipping.
      // Example (if needed):
      // if (_isFrontCamera) {
      //   final flippedLeft = vw - right;
      //   final flippedRight = vw - left;
      //   left = flippedLeft;
      //   right = flippedRight;
      // }

      // Clamp to view bounds
      left = left.clamp(0.0, vw);
      right = right.clamp(0.0, vw);
      top = top.clamp(0.0, vh);
      bottom = bottom.clamp(0.0, vh);

      return Rect.fromLTRB(left, top, right, bottom);
    }

    // Create transformed copies (only boundingBox is changed, normalizedBox stays as original)
    final out = <YOLOResult>[];
    for (final d in _currentDetections) {
      final transformedBox = _toViewRect(d.boundingBox);
      out.add(
        YOLOResult(
          classIndex: d.classIndex,
          className: d.className,
          confidence: d.confidence,
          boundingBox: transformedBox,
          normalizedBox: d.normalizedBox,
          modelName: d.modelName,
          mask: d.mask,
          keypoints: d.keypoints,
          keypointConfidences: d.keypointConfidences,
        ),
      );
    }
    return out;
  }

  // Public methods for external control
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
  Future<void> setShowOverlays(bool show) =>
      _effectiveController.setShowOverlays(show);
}
