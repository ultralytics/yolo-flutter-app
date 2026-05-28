// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'package:flutter/foundation.dart' show defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/core/yolo_model_resolver.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

/// Enum for camera lens selection.
enum LensFacing {
  back,
  front,

  /// Prefer the shortest-focal-length rear camera on Android, falling back to [back].
  backWide,
}

/// A Flutter widget that displays a real-time camera preview with YOLO object detection.
class YOLOView extends StatefulWidget {
  final String modelPath;
  final YOLOTask? task;
  final YOLOViewController? controller;
  final String cameraResolution;
  final Function(List<YOLOResult>)? onResult;
  final Function(YOLOPerformanceMetrics)? onPerformanceMetrics;
  final Function(Map<String, dynamic>)? onStreamingData;
  final Function(double zoomLevel)? onZoomChanged;
  final YOLOStreamingConfig? streamingConfig;
  final double confidenceThreshold;
  final double iouThreshold;
  final bool useGpu;
  final LensFacing lensFacing;

  const YOLOView({
    super.key,
    required this.modelPath,
    this.task,
    this.controller,
    this.cameraResolution = '720p',
    this.onResult,
    this.onPerformanceMetrics,
    this.onStreamingData,
    this.onZoomChanged,
    this.streamingConfig,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.7,
    this.useGpu = true,
    this.lensFacing = LensFacing.back,
  });

  @override
  State<YOLOView> createState() => _YOLOViewState();
}

class _YOLOViewState extends State<YOLOView> {
  late YOLOViewController _effectiveController;
  bool _ownsController = false;
  late MethodChannel _methodChannel;
  late EventChannel _resultEventChannel;
  StreamSubscription<dynamic>? _resultSubscription;
  YOLOResolvedModel? _resolvedModel;
  Object? _resolutionError;
  int _resolutionRequestId = 0;

  final String _viewId = UniqueKey().toString();
  int? _platformViewId;

  @override
  void initState() {
    super.initState();
    _setupController();
    _setupChannels();
    _resolveModel();
  }

  void _setupController() {
    if (widget.controller != null) {
      _effectiveController = widget.controller!;
      _ownsController = false;
    } else {
      _effectiveController = YOLOViewController();
      _ownsController = true;
    }
  }

  void _setupChannels() {
    _methodChannel = ChannelConfig.createControlChannel(_viewId);
    _resultEventChannel = ChannelConfig.createDetectionResultsChannel(_viewId);
  }

  void _subscribeToResults() {
    // Subscribe unconditionally — the controller's zoom/lens/focus streams ride the same event channel as detection
    // results, so consumers that only listen via the controller still need an active subscription.
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

    // Typed native events (`zoom`, `lens`, `focus`) coexist with detection payloads on the same channel; dispatch them
    // to the controller's streams before falling through to detection/performance handling.
    if (event['type'] is String) {
      _effectiveController.onNativeEvent(event);
      return;
    }

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
      widget.onResult!(_parseDetectionResults(event));
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
    final detectionsData = event['detections'] as List<dynamic>? ?? const [];
    final results = <YOLOResult>[];
    for (final detection in detectionsData) {
      if (detection is! Map) continue;
      if (detection['classIndex'] == null ||
          detection['className'] == null ||
          detection['confidence'] == null ||
          detection['boundingBox'] == null ||
          detection['normalizedBox'] == null) {
        continue;
      }
      try {
        results.add(YOLOResult.fromMap(detection));
      } catch (e) {
        logInfo('YOLOView: Error parsing detection: $e');
      }
    }
    return results;
  }

  Future<void> _resolveModel({bool switchExisting = false}) async {
    final requestId = ++_resolutionRequestId;
    await _performModelResolution(
      requestId: requestId,
      switchExisting: switchExisting,
      modelPath: widget.modelPath,
      task: widget.task,
    );
  }

  Future<void> _performModelResolution({
    required int requestId,
    required bool switchExisting,
    required String modelPath,
    required YOLOTask? task,
  }) async {
    if (mounted) {
      setState(() {
        _resolutionError = null;
      });
    }

    try {
      final resolvedModel = await YOLOModelResolver.resolve(
        modelPath: modelPath,
        task: task,
      );
      if (!mounted || requestId != _resolutionRequestId) return;

      final previousResolvedModel = _resolvedModel;
      final didChange =
          previousResolvedModel?.modelPath != resolvedModel.modelPath ||
          previousResolvedModel?.task != resolvedModel.task;

      setState(() {
        _resolvedModel = resolvedModel;
      });

      if (switchExisting && didChange && _platformViewId != null) {
        await _effectiveController.switchModel(
          resolvedModel.modelPath,
          resolvedModel.task,
        );
      }
    } catch (error) {
      if (!mounted || requestId != _resolutionRequestId) return;
      setState(() {
        _resolutionError = error;
        if (!switchExisting) {
          _resolvedModel = null;
        }
      });
    }
  }

  @override
  void didUpdateWidget(YOLOView oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Handle controller changes
    if (oldWidget.controller != widget.controller) {
      final previousController = _effectiveController;
      final previouslyOwned = _ownsController;
      _setupController();
      if (previouslyOwned && previousController != _effectiveController) {
        previousController.dispose();
      }
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

    // Handle model or task changes
    if (oldWidget.modelPath != widget.modelPath ||
        oldWidget.task != widget.task) {
      _resolveModel(switchExisting: _platformViewId != null);
    }
  }

  @override
  void dispose() {
    _effectiveController.stop();
    _resultSubscription?.cancel();
    _methodChannel.setMethodCallHandler(null);
    if (_ownsController) {
      _effectiveController.dispose();
    }

    if (_platformViewId != null) {
      ChannelConfig.createSingleImageChannel()
          .invokeMethod('disposeInstance', {'instanceId': _viewId})
          .catchError((e) => logInfo('YOLOView: Error disposing model: $e'));
    }

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (defaultTargetPlatform != TargetPlatform.android &&
        defaultTargetPlatform != TargetPlatform.iOS) {
      return const Center(child: Text('Platform not supported for YOLOView'));
    }
    if (_resolutionError != null) {
      return Center(child: Text('Failed to load model: $_resolutionError'));
    }
    if (_resolvedModel == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return _buildCameraView();
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
    final resolvedModel = _resolvedModel!;
    final creationParams = <String, dynamic>{
      'modelPath': resolvedModel.modelPath,
      'task': resolvedModel.task.name,
      'confidenceThreshold': widget.confidenceThreshold,
      'iouThreshold': widget.iouThreshold,
      'numItemsThreshold': _effectiveController.numItemsThreshold,
      'viewId': _viewId,
      'useGpu': widget.useGpu,
      'lensFacing': widget.lensFacing.name,
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

    if (widget.streamingConfig != null) {
      _effectiveController.setStreamingConfig(widget.streamingConfig!);
    }

    _subscribeToResults();
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
}
