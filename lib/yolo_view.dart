// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_view.dart

import 'dart:async';
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, debugPrint; // Explicitly import debugPrint
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

/// Controller for interacting with a [YOLOView] widget.
///
/// This controller provides methods to adjust detection thresholds
/// and camera settings for real-time object detection. It manages
/// the communication with the native platform views.
///
/// Example:
/// ```dart
/// class MyDetectorScreen extends StatefulWidget {
///   @override
///   State<MyDetectorScreen> createState() => _MyDetectorScreenState();
/// }
///
/// class _MyDetectorScreenState extends State<MyDetectorScreen> {
///   final controller = YOLOViewController();
///
///   @override
///   Widget build(BuildContext context) {
///     return Column(
///       children: [
///         Expanded(
///           child: YOLOView(
///             modelPath: 'assets/yolov8n.mlmodel',
///             task: YOLOTask.detect,
///             controller: controller,
///             onResult: (results) {
///               print('Detected ${results.length} objects');
///             },
///           ),
///         ),
///         ElevatedButton(
///           onPressed: () => controller.switchCamera(),
///           child: Text('Switch Camera'),
///         ),
///       ],
///     );
///   }
/// }
/// ```
class YOLOViewController {
  MethodChannel? _methodChannel;

  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;

  /// The current confidence threshold for detections.
  ///
  /// Only detections with confidence scores above this threshold
  /// will be returned. Default is 0.5 (50%).
  double get confidenceThreshold => _confidenceThreshold;

  /// The current Intersection over Union (IoU) threshold.
  ///
  /// Used for non-maximum suppression to filter overlapping
  /// detections. Default is 0.45.
  double get iouThreshold => _iouThreshold;

  /// The maximum number of items to detect per frame.
  ///
  /// Limits the number of detections returned to improve
  /// performance. Default is 30.
  int get numItemsThreshold => _numItemsThreshold;

  void _init(MethodChannel methodChannel) {
    _methodChannel = methodChannel;
    _applyThresholds();
  }

  Future<void> _applyThresholds() async {
    if (_methodChannel == null) {
      debugPrint(
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
      debugPrint(
        'YOLOViewController: Applied thresholds - confidence: $_confidenceThreshold, IoU: $_iouThreshold, numItems: $_numItemsThreshold',
      );
    } catch (e) {
      debugPrint('YOLOViewController: Error applying combined thresholds: $e');
      try {
        debugPrint(
          'YOLOViewController: Trying individual threshold methods as fallback',
        );
        await _methodChannel!.invokeMethod('setConfidenceThreshold', {
          'threshold': _confidenceThreshold,
        });
        debugPrint(
          'YOLOViewController: Applied confidence threshold: $_confidenceThreshold',
        );
        await _methodChannel!.invokeMethod('setIoUThreshold', {
          'threshold': _iouThreshold,
        });
        debugPrint('YOLOViewController: Applied IoU threshold: $_iouThreshold');
        await _methodChannel!.invokeMethod('setNumItemsThreshold', {
          'numItems': _numItemsThreshold,
        });
        debugPrint(
          'YOLOViewController: Applied numItems threshold: $_numItemsThreshold',
        );
      } catch (e2) {
        debugPrint(
          'YOLOViewController: Error applying individual thresholds: $e2',
        );
      }
    }
  }

  /// Sets the confidence threshold for object detection.
  ///
  /// Only detections with confidence scores above [threshold] will be
  /// returned. The value is automatically clamped between 0.0 and 1.0.
  ///
  /// Example:
  /// ```dart
  /// // Only show detections with 70% confidence or higher
  /// await controller.setConfidenceThreshold(0.7);
  /// ```
  Future<void> setConfidenceThreshold(double threshold) async {
    final clampedThreshold = threshold.clamp(0.0, 1.0);
    _confidenceThreshold = clampedThreshold;
    if (_methodChannel == null) {
      debugPrint(
        'YOLOViewController: Warning - Cannot apply confidence threshold, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setConfidenceThreshold', {
        'threshold': clampedThreshold,
      });
      debugPrint(
        'YOLOViewController: Applied confidence threshold: $_confidenceThreshold',
      );
    } catch (e) {
      debugPrint('YOLOViewController: Error applying confidence threshold: $e');
      return _applyThresholds();
    }
  }

  /// Sets the Intersection over Union (IoU) threshold.
  ///
  /// This threshold is used for non-maximum suppression to filter
  /// overlapping detections. Lower values result in fewer overlapping
  /// boxes. The value is automatically clamped between 0.0 and 1.0.
  ///
  /// Example:
  /// ```dart
  /// // Use stricter overlap filtering
  /// await controller.setIoUThreshold(0.3);
  /// ```
  Future<void> setIoUThreshold(double threshold) async {
    final clampedThreshold = threshold.clamp(0.0, 1.0);
    _iouThreshold = clampedThreshold;
    if (_methodChannel == null) {
      debugPrint(
        'YOLOViewController: Warning - Cannot apply IoU threshold, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setIoUThreshold', {
        'threshold': clampedThreshold,
      });
      debugPrint('YOLOViewController: Applied IoU threshold: $_iouThreshold');
    } catch (e) {
      debugPrint('YOLOViewController: Error applying IoU threshold: $e');
      return _applyThresholds();
    }
  }

  /// Sets the maximum number of items to detect per frame.
  ///
  /// Limiting the number of detections can improve performance,
  /// especially on lower-end devices. The value is automatically
  /// clamped between 1 and 100.
  ///
  /// Example:
  /// ```dart
  /// // Only detect up to 10 objects per frame
  /// await controller.setNumItemsThreshold(10);
  /// ```
  Future<void> setNumItemsThreshold(int numItems) async {
    final clampedValue = numItems.clamp(1, 100);
    _numItemsThreshold = clampedValue;
    if (_methodChannel == null) {
      debugPrint(
        'YOLOViewController: Warning - Cannot apply numItems threshold, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setNumItemsThreshold', {
        'numItems': clampedValue,
      });
      debugPrint(
        'YOLOViewController: Applied numItems threshold: $_numItemsThreshold',
      );
    } catch (e) {
      debugPrint('YOLOViewController: Error applying numItems threshold: $e');
      return _applyThresholds();
    }
  }

  /// Sets multiple thresholds at once.
  ///
  /// This is more efficient than calling individual threshold setters
  /// when you need to update multiple values. Only non-null parameters
  /// will be updated.
  ///
  /// Example:
  /// ```dart
  /// await controller.setThresholds(
  ///   confidenceThreshold: 0.6,
  ///   iouThreshold: 0.4,
  ///   numItemsThreshold: 20,
  /// );
  /// ```
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
    return _applyThresholds();
  }

  /// Switches between front and back camera.
  ///
  /// This method toggles the camera between front-facing and back-facing modes.
  /// Returns a [Future] that completes when the camera has been switched.
  ///
  /// Example:
  /// ```dart
  /// // Create a controller
  /// final controller = YOLOViewController();
  ///
  /// // Switch between front and back camera
  /// await controller.switchCamera();
  /// ```
  Future<void> switchCamera() async {
    if (_methodChannel == null) {
      debugPrint(
        'YOLOViewController: Warning - Cannot switch camera, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('switchCamera');
      debugPrint('YOLOViewController: Camera switched successfully');
    } catch (e) {
      debugPrint('YOLOViewController: Error switching camera: $e');
    }
  }
}

/// A Flutter widget that displays a real-time camera preview with YOLO object detection.
///
/// This widget creates a platform view that runs YOLO inference on camera frames
/// and provides detection results through callbacks. It supports various YOLO tasks
/// including object detection, segmentation, classification, pose estimation, and
/// oriented bounding box detection.
///
/// Example:
/// ```dart
/// YOLOView(
///   modelPath: 'assets/models/yolov8n.mlmodel',
///   task: YOLOTask.detect,
///   onResult: (List<YOLOResult> results) {
///     // Handle detection results
///     for (var result in results) {
///       print('Detected ${result.className} with ${result.confidence}');
///     }
///   },
///   onPerformanceMetrics: (Map<String, double> metrics) {
///     print('FPS: ${metrics['fps']}');
///   },
/// )
/// ```
///
/// The widget requires camera permissions to be granted before use.
/// On iOS, add NSCameraUsageDescription to Info.plist.
/// On Android, add CAMERA permission to AndroidManifest.xml.
class YOLOView extends StatefulWidget {
  /// Path to the YOLO model file.
  ///
  /// The model should be placed in the app's assets folder and
  /// included in pubspec.yaml. Supported formats:
  /// - iOS: .mlmodel (Core ML)
  /// - Android: .tflite (TensorFlow Lite)
  final String modelPath;

  /// The type of YOLO task to perform.
  ///
  /// This must match the task the model was trained for.
  /// See [YOLOTask] for available options.
  final YOLOTask task;

  /// Optional controller for managing detection settings.
  ///
  /// If not provided, a default controller will be created internally.
  /// Use a controller when you need to adjust thresholds or switch cameras.
  final YOLOViewController? controller;

  /// The camera resolution to use.
  ///
  /// Currently not implemented. Reserved for future use.
  final String cameraResolution;

  /// Callback invoked when new detection results are available.
  ///
  /// This callback is called for each processed frame that contains
  /// detections. The frequency depends on the device's processing speed.
  final Function(List<YOLOResult>)? onResult;

  /// Callback invoked with performance metrics.
  ///
  /// Provides real-time performance data including:
  /// - 'processingTimeMs': Time to process a single frame
  /// - 'fps': Current frames per second
  final Function(Map<String, double> metrics)? onPerformanceMetrics;

  /// Whether to show native UI controls on the camera preview.
  ///
  /// When true, platform-specific UI elements may be displayed,
  /// such as bounding boxes and labels drawn natively.
  final bool showNativeUI;

  const YOLOView({
    super.key,
    required this.modelPath,
    required this.task,
    this.controller,
    this.cameraResolution = '720p',
    this.onResult,
    this.onPerformanceMetrics,
    this.showNativeUI = false,
  });

  @override
  State<YOLOView> createState() => YOLOViewState();
}

/// State for the [YOLOView] widget.
///
/// Manages platform view creation, event channel subscriptions,
/// and communication with native YOLO implementations.
class YOLOViewState extends State<YOLOView> {
  late EventChannel _resultEventChannel;
  StreamSubscription<dynamic>? _resultSubscription;
  late MethodChannel _methodChannel;

  late YOLOViewController _effectiveController;

  final String _viewId = UniqueKey().toString();

  @override
  void initState() {
    super.initState();

    debugPrint(
      'YOLOView (Dart initState): Creating channels with _viewId: $_viewId',
    );

    final resultChannelName = 'com.ultralytics.yolo/detectionResults_$_viewId';
    _resultEventChannel = EventChannel(resultChannelName);
    debugPrint(
      'YOLOView (Dart initState): Result EventChannel created: $resultChannelName',
    );

    final controlChannelName = 'com.ultralytics.yolo/controlChannel_$_viewId';
    _methodChannel = MethodChannel(controlChannelName);
    debugPrint(
      'YOLOView (Dart initState): Control MethodChannel created: $controlChannelName',
    );

    _setupController();

    if (widget.onResult != null || widget.onPerformanceMetrics != null) {
      _subscribeToResults();
    }
  }

  void _setupController() {
    if (widget.controller != null) {
      _effectiveController = widget.controller!;
    } else {
      _effectiveController = YOLOViewController();
    }
    _effectiveController._init(_methodChannel);
  }

  @override
  void didUpdateWidget(YOLOView oldWidget) {
    super.didUpdateWidget(oldWidget);

    if (oldWidget.controller != widget.controller) {
      _setupController();
    }

    if (oldWidget.onResult != widget.onResult ||
        oldWidget.onPerformanceMetrics != widget.onPerformanceMetrics) {
      if (widget.onResult == null && widget.onPerformanceMetrics == null) {
        _cancelResultSubscription();
      } else {
        // If at least one callback is now non-null, ensure subscription
        _subscribeToResults();
      }
    }

    if (oldWidget.showNativeUI != widget.showNativeUI) {
      _methodChannel.invokeMethod('setShowUIControls', {
        'show': widget.showNativeUI,
      });
    }
  }

  @override
  void dispose() {
    _cancelResultSubscription();
    super.dispose();
  }

  void _subscribeToResults() {
    _cancelResultSubscription();

    debugPrint(
      'YOLOView: Setting up event stream listener for channel: ${_resultEventChannel.name}',
    );

    _resultSubscription = _resultEventChannel.receiveBroadcastStream().listen(
      (dynamic event) {
        debugPrint('YOLOView: Received event from native platform: $event');

        if (event is Map && event.containsKey('test')) {
          debugPrint('YOLOView: Received test message: ${event['test']}');
          return;
        }

        if (event is Map) {
          // Handle detection results
          if (widget.onResult != null && event.containsKey('detections')) {
            try {
              final List<dynamic> detections = event['detections'] ?? [];
              debugPrint('YOLOView: Received ${detections.length} detections');

              for (var i = 0; i < detections.length && i < 3; i++) {
                final detection = detections[i];
                final className = detection['className'] ?? 'unknown';
                final confidence = detection['confidence'] ?? 0.0;
                debugPrint(
                  'YOLOView: Detection $i - $className (${(confidence * 100).toStringAsFixed(1)}%)',
                );
              }

              final results = _parseDetectionResults(event);
              debugPrint('YOLOView: Parsed results count: ${results.length}');
              widget.onResult!(results);
              debugPrint('YOLOView: Called onResult callback with results');
            } catch (e, s) {
              debugPrint('Error parsing detection results: $e');
              debugPrint('Stack trace for detection error: $s');
              debugPrint(
                'YOLOView: Event keys for detection error: ${event.keys.toList()}',
              );
              if (event.containsKey('detections')) {
                final detections = event['detections'];
                debugPrint(
                  'YOLOView: Detections type for error: ${detections.runtimeType}',
                );
                if (detections is List && detections.isNotEmpty) {
                  debugPrint(
                    'YOLOView: First detection keys for error: ${detections.first?.keys?.toList()}',
                  );
                }
              }
            }
          }

          // Handle performance metrics
          if (widget.onPerformanceMetrics != null) {
            try {
              final double? processingTimeMs =
                  (event['processingTimeMs'] as num?)?.toDouble();
              final double? fps = (event['fps'] as num?)?.toDouble();

              if (processingTimeMs != null && fps != null) {
                widget.onPerformanceMetrics!({
                  'processingTimeMs': processingTimeMs,
                  'fps': fps,
                });
                debugPrint(
                  'YOLOView: Called onPerformanceMetrics callback with: processingTimeMs=$processingTimeMs, fps=$fps',
                );
              }
            } catch (e, s) {
              debugPrint('Error parsing performance metrics: $e');
              debugPrint('Stack trace for metrics error: $s');
              debugPrint(
                'YOLOView: Event keys for metrics error: ${event.keys.toList()}',
              );
            }
          }
        } else {
          debugPrint(
            'YOLOView: Received invalid event format or no relevant callbacks are set. Event type: ${event.runtimeType}',
          );
        }
      },
      onError: (dynamic error, StackTrace stackTrace) {
        // Added StackTrace
        debugPrint('Error from detection results stream: $error');
        debugPrint('Stack trace from stream error: $stackTrace');

        Future.delayed(const Duration(seconds: 2), () {
          if (_resultSubscription != null && mounted) {
            // Check mounted before resubscribing
            debugPrint('YOLOView: Attempting to resubscribe after error');
            _subscribeToResults();
          } else {
            debugPrint(
              'YOLOView: Not resubscribing (stream already null or widget disposed)',
            );
          }
        });
      },
      onDone: () {
        debugPrint('YOLOView: Event stream closed for $_viewId');
        _resultSubscription = null;
      },
    );
    debugPrint('YOLOView: Event stream listener setup complete for $_viewId');
  }

  void _cancelResultSubscription() {
    if (_resultSubscription != null) {
      debugPrint(
        'YOLOView: Cancelling existing result subscription for $_viewId',
      );
      _resultSubscription!.cancel();
      _resultSubscription = null;
    }
  }

  List<YOLOResult> _parseDetectionResults(Map<dynamic, dynamic> event) {
    final List<dynamic> detectionsData = event['detections'] ?? [];
    debugPrint('YOLOView: Parsing ${detectionsData.length} detections');

    if (detectionsData.isNotEmpty) {
      final first = detectionsData.first;
      debugPrint(
        'YOLOView: First detection structure: ${first.runtimeType} with keys: ${first is Map ? first.keys.toList() : "not a map"}',
      );

      if (first is Map) {
        debugPrint('YOLOView: ClassIndex: ${first["classIndex"]}');
        debugPrint('YOLOView: ClassName: ${first["className"]}');
        debugPrint('YOLOView: Confidence: ${first["confidence"]}');
        debugPrint('YOLOView: BoundingBox: ${first["boundingBox"]}');
        debugPrint('YOLOView: NormalizedBox: ${first["normalizedBox"]}');
      }
    }

    try {
      final results = detectionsData.map((detection) {
        try {
          return YOLOResult.fromMap(detection);
        } catch (e) {
          debugPrint('YOLOView: Error parsing single detection: $e');
          debugPrint('YOLOView: Problem detection data: $detection');
          rethrow;
        }
      }).toList();

      debugPrint('YOLOView: Successfully parsed ${results.length} results');
      return results;
    } catch (e) {
      debugPrint('YOLOView: Error parsing detections list: $e');
      return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    const viewType = 'com.ultralytics.yolo/YOLOPlatformView';
    final creationParams = <String, dynamic>{
      'modelPath': widget.modelPath,
      'task': widget.task.name,
      'confidenceThreshold': _effectiveController.confidenceThreshold,
      'iouThreshold': _effectiveController.iouThreshold,
      'numItemsThreshold': _effectiveController.numItemsThreshold,
      'viewId': _viewId,
    };

    // This was causing issues in initState/didUpdateWidget, better to call once after view created.
    // WidgetsBinding.instance.addPostFrameCallback((_) {
    //   if (mounted) { // Ensure widget is still mounted
    //    _methodChannel.invokeMethod('setShowUIControls', {'show': widget.showNativeUI});
    //   }
    // });

    Widget platformView;
    if (defaultTargetPlatform == TargetPlatform.android) {
      platformView = AndroidView(
        viewType: viewType,
        layoutDirection: TextDirection.ltr,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: _onPlatformViewCreated,
      );
    } else if (defaultTargetPlatform == TargetPlatform.iOS) {
      platformView = UiKitView(
        viewType: viewType,
        layoutDirection: TextDirection.ltr,
        creationParams: creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        onPlatformViewCreated: _onPlatformViewCreated,
      );
    } else {
      platformView = const Center(
        child: Text('Platform not supported for YOLOView'),
      );
    }
    return platformView;
  }

  void _onPlatformViewCreated(int id) {
    debugPrint(
      'YOLOView: Platform view created with system id: $id, our viewId: $_viewId',
    );

    // _cancelResultSubscription(); // Already called in _subscribeToResults if needed

    if (widget.onResult != null || widget.onPerformanceMetrics != null) {
      debugPrint(
        'YOLOView: Re-subscribing to results after platform view creation for $_viewId',
      );
      _subscribeToResults();
    }

    _effectiveController._init(
      _methodChannel,
    ); // Re-init controller with the now valid method channel

    _methodChannel.invokeMethod('setShowUIControls', {
      'show': widget.showNativeUI,
    });

    _methodChannel.setMethodCallHandler((call) async {
      debugPrint(
        'YOLOView: Received method call from platform: ${call.method} for $_viewId',
      );

      switch (call.method) {
        case 'recreateEventChannel':
          debugPrint(
            'YOLOView: Platform requested recreation of event channel for $_viewId',
          );
          _cancelResultSubscription();
          Future.delayed(const Duration(milliseconds: 100), () {
            if (mounted &&
                (widget.onResult != null ||
                    widget.onPerformanceMetrics != null)) {
              _subscribeToResults();
              debugPrint('YOLOView: Event channel recreated for $_viewId');
            }
          });
          return null;
        default:
          debugPrint('YOLOView: Unknown method call: ${call.method}');
          return null;
      }
    });
  }

  // Methods to be called via GlobalKey
  /// Sets the confidence threshold through the widget's state.
  ///
  /// This method can be called using a GlobalKey to access the state:
  /// ```dart
  /// final key = GlobalKey<YOLOViewState>();
  /// // Later...
  /// key.currentState?.setConfidenceThreshold(0.7);
  /// ```
  Future<void> setConfidenceThreshold(double threshold) {
    return _effectiveController.setConfidenceThreshold(threshold);
  }

  /// Sets the IoU threshold through the widget's state.
  ///
  /// This method can be called using a GlobalKey to access the state.
  Future<void> setIoUThreshold(double threshold) {
    return _effectiveController.setIoUThreshold(threshold);
  }

  /// Sets the maximum number of items threshold through the widget's state.
  ///
  /// This method can be called using a GlobalKey to access the state.
  Future<void> setNumItemsThreshold(int numItems) {
    return _effectiveController.setNumItemsThreshold(numItems);
  }

  /// Sets multiple thresholds through the widget's state.
  ///
  /// This method can be called using a GlobalKey to access the state.
  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
    int? numItemsThreshold,
  }) {
    return _effectiveController.setThresholds(
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      numItemsThreshold: numItemsThreshold,
    );
  }

  /// Switches between front and back camera.
  ///
  /// This method toggles the camera between front-facing and back-facing modes.
  /// It delegates to the effective controller's switchCamera method.
  /// Returns a [Future] that completes when the camera has been switched.
  Future<void> switchCamera() {
    return _effectiveController.switchCamera();
  }
}
