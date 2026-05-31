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

  /// Called when an in-place model switch fails while a previously loaded model is still running natively. Carries the
  /// `modelPath`/`task` of the request that failed so the host can ignore stale failures and only react to the request
  /// that matches its current selection. Lets the host clear transient UI without the view tearing itself down into a
  /// full-screen error.
  final void Function(Object error, String modelPath, YOLOTask? task)?
  onModelError;

  /// Called after a model is successfully loaded (initial load) or switched in place (native `setModel` succeeded).
  /// Carries the loaded `modelPath`/`task` so the host can record exactly which model is running rather than reading
  /// its (possibly already-changed) optimistic selection.
  final void Function(String modelPath, YOLOTask? task)? onModelLoad;
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
    this.onModelError,
    this.onModelLoad,
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

  // Serializes native `switchModel` calls. Each setModel completes asynchronously on the native side (Android spins a
  // fresh executor, iOS model creation is async), so two overlapping calls could finish out of order and leave native
  // state behind the latest selection. Chaining them — and skipping any request superseded before its turn — keeps the
  // most recently requested model as the last one actually applied.
  Future<void> _nativeSwitchChain = Future<void>.value();

  // The model the native side was last asked to load (or settled on after a failed switch reverted it). Tracks native
  // intent independently of `_resolvedModel`, which is committed only after a switch succeeds and therefore lags an
  // in-flight switch. Used to decide whether a native switch is actually needed and to dedupe redundant reloads.
  YOLOResolvedModel? _nativeSwitchTarget;

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

      if (switchExisting && _platformViewId != null) {
        // Run the native switch AND the `_resolvedModel` commit inside the serialized chain so overlapping requests
        // apply strictly in order. The decision to hit the native side compares against `_nativeSwitchTarget` (what
        // native was last asked / settled on), NOT `_resolvedModel` — `_resolvedModel` is committed only after a
        // successful switch, so it lags an in-flight switch and would wrongly skip re-applying the latest selection
        // (e.g. switch to `s` in flight, user taps back to `n`). A request superseded before its turn skips the
        // native call so the latest selection always wins.
        final pending = _nativeSwitchChain.then((_) async {
          if (!mounted || requestId != _resolutionRequestId) return;
          final alreadyTargeting =
              _nativeSwitchTarget?.modelPath == resolvedModel.modelPath &&
              _nativeSwitchTarget?.task == resolvedModel.task;
          if (!alreadyTargeting) {
            // Switch BEFORE committing `_resolvedModel`. On failure the native side keeps the previously loaded model
            // (verified on both platforms), so restore the intent and rethrow — the catch keeps the live camera.
            final previousTarget = _nativeSwitchTarget;
            _nativeSwitchTarget = resolvedModel;
            try {
              await _effectiveController.switchModel(
                resolvedModel.modelPath,
                resolvedModel.task,
              );
            } catch (_) {
              _nativeSwitchTarget = previousTarget;
              rethrow;
            }
          }
          if (!mounted || requestId != _resolutionRequestId) return;
          setState(() {
            _resolvedModel = resolvedModel;
          });
          widget.onModelLoad?.call(modelPath, task);
        });
        // Keep the chain alive past a failed/superseded switch so later requests still run after this one.
        _nativeSwitchChain = pending.catchError((_) {});
        await pending; // surface this request's own native failure to the catch below
      } else {
        // Initial load / pre-platform-view: no native switch happens here (creationParams carry the model). Commit
        // directly and record it as the native target so the first in-place switch compares against the right model.
        setState(() {
          _resolvedModel = resolvedModel;
        });
        _nativeSwitchTarget = resolvedModel;
        widget.onModelLoad?.call(modelPath, task);
      }
    } catch (error) {
      if (!mounted || requestId != _resolutionRequestId) return;
      if (switchExisting && _resolvedModel != null) {
        // An in-place switch failed (e.g. the requested model isn't published at the release tag yet, or the device
        // went offline mid-download) but the previously loaded model is still running natively. Keep the live camera
        // and report the failure out-of-band instead of tearing the view down into a full-screen error.
        FlutterError.reportError(
          FlutterErrorDetails(
            exception: error,
            library: 'ultralytics_yolo',
            context: ErrorDescription('switching model to $modelPath'),
          ),
        );
        widget.onModelError?.call(error, modelPath, task);
      } else {
        logInfo('YOLOView: Failed to load model $modelPath: $error');
        setState(() {
          _resolutionError = error;
          _resolvedModel = null;
        });
        widget.onModelError?.call(error, modelPath, task);
      }
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
      // Show a neutral message on the same dark veil as the loading state — never surface the raw exception to users.
      return const ColoredBox(
        color: Colors.black,
        child: Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Text(
              'Unable to load the model. Please try again.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white70),
            ),
          ),
        ),
      );
    }
    if (_resolvedModel == null) {
      // Match the in-app model-loading veil so the first load and subsequent switches look consistent.
      return const ColoredBox(
        color: Colors.black,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 36,
                height: 36,
                child: CircularProgressIndicator(
                  strokeWidth: 2.5,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              ),
              SizedBox(height: 14),
              Text(
                'Loading model…',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      );
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
  Future<void> setShowOverlays(bool visible) =>
      _effectiveController.setShowOverlays(visible);
}
