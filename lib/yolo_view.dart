// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_view.dart

import 'dart:async';
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, debugPrint; // Explicitly import debugPrint
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

/// Controller for interacting with a [YoloView] widget.
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
///   final controller = YoloViewController();
///
///   @override
///   Widget build(BuildContext context) {
///     return Column(
///       children: [
///         Expanded(
///           child: YoloView(
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
class YoloViewController {
  MethodChannel? _methodChannel;
  int? _viewId;

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

  void _init(MethodChannel methodChannel, int viewId) {
    _methodChannel = methodChannel;
    _viewId = viewId;
    _applyThresholds();
  }

  Future<void> _applyThresholds() async {
    if (_methodChannel == null) {
      debugPrint(
        'YoloViewController: Warning - Cannot apply thresholds, view not yet created',
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
        'YoloViewController: Applied thresholds - confidence: $_confidenceThreshold, IoU: $_iouThreshold, numItems: $_numItemsThreshold',
      );
    } catch (e) {
      debugPrint('YoloViewController: Error applying combined thresholds: $e');
      try {
        debugPrint(
          'YoloViewController: Trying individual threshold methods as fallback',
        );
        await _methodChannel!.invokeMethod('setConfidenceThreshold', {
          'threshold': _confidenceThreshold,
        });
        debugPrint(
          'YoloViewController: Applied confidence threshold: $_confidenceThreshold',
        );
        await _methodChannel!.invokeMethod('setIoUThreshold', {
          'threshold': _iouThreshold,
        });
        debugPrint('YoloViewController: Applied IoU threshold: $_iouThreshold');
        await _methodChannel!.invokeMethod('setNumItemsThreshold', {
          'numItems': _numItemsThreshold,
        });
        debugPrint(
          'YoloViewController: Applied numItems threshold: $_numItemsThreshold',
        );
      } catch (e2) {
        debugPrint(
          'YoloViewController: Error applying individual thresholds: $e2',
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
        'YoloViewController: Warning - Cannot apply confidence threshold, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setConfidenceThreshold', {
        'threshold': clampedThreshold,
      });
      debugPrint(
        'YoloViewController: Applied confidence threshold: $_confidenceThreshold',
      );
    } catch (e) {
      debugPrint('YoloViewController: Error applying confidence threshold: $e');
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
        'YoloViewController: Warning - Cannot apply IoU threshold, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setIoUThreshold', {
        'threshold': clampedThreshold,
      });
      debugPrint('YoloViewController: Applied IoU threshold: $_iouThreshold');
    } catch (e) {
      debugPrint('YoloViewController: Error applying IoU threshold: $e');
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
        'YoloViewController: Warning - Cannot apply numItems threshold, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setNumItemsThreshold', {
        'numItems': clampedValue,
      });
      debugPrint(
        'YoloViewController: Applied numItems threshold: $_numItemsThreshold',
      );
    } catch (e) {
      debugPrint('YoloViewController: Error applying numItems threshold: $e');
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
  /// final controller = YoloViewController();
  ///
  /// // Switch between front and back camera
  /// await controller.switchCamera();
  /// ```
  Future<void> switchCamera() async {
    if (_methodChannel == null) {
      debugPrint(
        'YoloViewController: Warning - Cannot switch camera, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('switchCamera');
      debugPrint('YoloViewController: Camera switched successfully');
    } catch (e) {
      debugPrint('YoloViewController: Error switching camera: $e');
    }
  }

  /// Sets the camera zoom level to a specific value.
  ///
  /// The zoom level must be within the supported range of the camera.
  /// Typical values are 0.5x, 1.0x, 2.0x, 3.0x, etc.
  ///
  /// Example:
  /// ```dart
  /// // Set zoom to 2x
  /// await controller.setZoomLevel(2.0);
  /// ```
  Future<void> setZoomLevel(double zoomLevel) async {
    if (_methodChannel == null) {
      debugPrint(
        'YoloViewController: Warning - Cannot set zoom level, view not yet created',
      );
      return;
    }
    try {
      await _methodChannel!.invokeMethod('setZoomLevel', {
        'zoomLevel': zoomLevel,
      });
      debugPrint('YoloViewController: Zoom level set to $zoomLevel');
    } catch (e) {
      debugPrint('YoloViewController: Error setting zoom level: $e');
    }
  }

  /// Switches the YOLO model on the existing view.
  ///
  /// This method allows changing the model without recreating the entire view.
  /// The view must be created before calling this method.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the new model file
  /// - [task]: The YOLO task type for the new model
  ///
  /// Example:
  /// ```dart
  /// await controller.switchModel(
  ///   'assets/models/yolov8s.mlmodel',
  ///   YOLOTask.segment,
  /// );
  /// ```
  Future<void> switchModel(String modelPath, YOLOTask task) async {
    if (_methodChannel == null || _viewId == null) {
      debugPrint(
        'YoloViewController: Warning - Cannot switch model, view not yet created',
      );
      return;
    }
    try {
      debugPrint('YoloViewController: Switching model with viewId: $_viewId');

      // Call the platform method to switch model
      await MethodChannel('yolo_single_image_channel').invokeMethod(
        'setModel',
        {'viewId': _viewId, 'modelPath': modelPath, 'task': task.name},
      );

      debugPrint(
        'YoloViewController: Model switched successfully to $modelPath with task ${task.name}',
      );
    } catch (e) {
      debugPrint('YoloViewController: Error switching model: $e');
      rethrow;
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
/// YoloView(
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
class YoloView extends StatefulWidget {
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
  final YoloViewController? controller;

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

  /// Callback invoked when the camera zoom level changes.
  ///
  /// Provides the current zoom level as a double value (e.g., 1.0, 2.0, 3.5).
  final Function(double zoomLevel)? onZoomChanged;

  const YoloView({
    super.key,
    required this.modelPath,
    required this.task,
    this.controller,
    this.cameraResolution = '720p',
    this.onResult,
    this.onPerformanceMetrics,
    this.showNativeUI = false,
    this.onZoomChanged,
  });

  @override
  State<YoloView> createState() => YoloViewState();
}

/// State for the [YoloView] widget.
///
/// Manages platform view creation, event channel subscriptions,
/// and communication with native YOLO implementations.
class YoloViewState extends State<YoloView> {
  late EventChannel _resultEventChannel;
  StreamSubscription<dynamic>? _resultSubscription;
  late MethodChannel _methodChannel;

  late YoloViewController _effectiveController;

  final String _viewId = UniqueKey().toString();
  int? _platformViewId;

  @override
  void initState() {
    super.initState();

    debugPrint(
      'YoloView (Dart initState): Creating channels with _viewId: $_viewId',
    );

    final resultChannelName = 'com.ultralytics.yolo/detectionResults_$_viewId';
    _resultEventChannel = EventChannel(resultChannelName);
    debugPrint(
      'YoloView (Dart initState): Result EventChannel created: $resultChannelName',
    );

    final controlChannelName = 'com.ultralytics.yolo/controlChannel_$_viewId';
    _methodChannel = MethodChannel(controlChannelName);
    debugPrint(
      'YoloView (Dart initState): Control MethodChannel created: $controlChannelName',
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
      _effectiveController = YoloViewController();
    }
    // Don't initialize here since we don't have the platform view ID yet
    // It will be initialized in _onPlatformViewCreated
  }

  @override
  void didUpdateWidget(YoloView oldWidget) {
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

    // Handle model or task changes
    if (_platformViewId != null &&
        (oldWidget.modelPath != widget.modelPath ||
            oldWidget.task != widget.task)) {
      debugPrint('YoloView: Model or task changed, switching model');
      _effectiveController
          .switchModel(widget.modelPath, widget.task)
          .catchError((e) {
            debugPrint(
              'YoloView: Error switching model in didUpdateWidget: $e',
            );
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
      'YoloView: Setting up event stream listener for channel: ${_resultEventChannel.name}',
    );

    _resultSubscription = _resultEventChannel.receiveBroadcastStream().listen(
      (dynamic event) {
        debugPrint('YoloView: Received event from native platform: $event');

        if (event is Map && event.containsKey('test')) {
          debugPrint('YoloView: Received test message: ${event['test']}');
          return;
        }

        if (event is Map) {
          // Handle detection results
          if (widget.onResult != null && event.containsKey('detections')) {
            try {
              final List<dynamic> detections = event['detections'] ?? [];
              debugPrint('YoloView: Received ${detections.length} detections');

              for (var i = 0; i < detections.length && i < 3; i++) {
                final detection = detections[i];
                final className = detection['className'] ?? 'unknown';
                final confidence = detection['confidence'] ?? 0.0;
                debugPrint(
                  'YoloView: Detection $i - $className (${(confidence * 100).toStringAsFixed(1)}%)',
                );
              }

              final results = _parseDetectionResults(event);
              debugPrint('YoloView: Parsed results count: ${results.length}');
              widget.onResult!(results);
              debugPrint('YoloView: Called onResult callback with results');
            } catch (e, s) {
              debugPrint('Error parsing detection results: $e');
              debugPrint('Stack trace for detection error: $s');
              debugPrint(
                'YoloView: Event keys for detection error: ${event.keys.toList()}',
              );
              if (event.containsKey('detections')) {
                final detections = event['detections'];
                debugPrint(
                  'YoloView: Detections type for error: ${detections.runtimeType}',
                );
                if (detections is List && detections.isNotEmpty) {
                  debugPrint(
                    'YoloView: First detection keys for error: ${detections.first?.keys?.toList()}',
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
                  'YoloView: Called onPerformanceMetrics callback with: processingTimeMs=$processingTimeMs, fps=$fps',
                );
              }
            } catch (e, s) {
              debugPrint('Error parsing performance metrics: $e');
              debugPrint('Stack trace for metrics error: $s');
              debugPrint(
                'YoloView: Event keys for metrics error: ${event.keys.toList()}',
              );
            }
          }
        } else {
          debugPrint(
            'YoloView: Received invalid event format or no relevant callbacks are set. Event type: ${event.runtimeType}',
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
            debugPrint('YoloView: Attempting to resubscribe after error');
            _subscribeToResults();
          } else {
            debugPrint(
              'YoloView: Not resubscribing (stream already null or widget disposed)',
            );
          }
        });
      },
      onDone: () {
        debugPrint('YoloView: Event stream closed for $_viewId');
        _resultSubscription = null;
      },
    );
    debugPrint('YoloView: Event stream listener setup complete for $_viewId');
  }

  @visibleForTesting
  void cancelResultSubscription() {
    _cancelResultSubscription();
  }

  void _cancelResultSubscription() {
    if (_resultSubscription != null) {
      debugPrint(
        'YoloView: Cancelling existing result subscription for $_viewId',
      );
      _resultSubscription!.cancel();
      _resultSubscription = null;
    }
  }

  @visibleForTesting
  List<YOLOResult> parseDetectionResults(Map<dynamic, dynamic> event) {
    return _parseDetectionResults(event);
  }

  List<YOLOResult> _parseDetectionResults(Map<dynamic, dynamic> event) {
    final List<dynamic> detectionsData = event['detections'] ?? [];
    debugPrint('YoloView: Parsing ${detectionsData.length} detections');

    if (detectionsData.isNotEmpty) {
      final first = detectionsData.first;
      debugPrint(
        'YoloView: First detection structure: ${first.runtimeType} with keys: ${first is Map ? first.keys.toList() : "not a map"}',
      );

      if (first is Map) {
        debugPrint('YoloView: ClassIndex: ${first["classIndex"]}');
        debugPrint('YoloView: ClassName: ${first["className"]}');
        debugPrint('YoloView: Confidence: ${first["confidence"]}');
        debugPrint('YoloView: BoundingBox: ${first["boundingBox"]}');
        debugPrint('YoloView: NormalizedBox: ${first["normalizedBox"]}');
      }
    }

    try {
      final results = detectionsData.map((detection) {
        try {
          return YOLOResult.fromMap(detection);
        } catch (e) {
          debugPrint('YoloView: Error parsing single detection: $e');
          debugPrint('YoloView: Problem detection data: $detection');
          rethrow;
        }
      }).toList();

      debugPrint('YoloView: Successfully parsed ${results.length} results');
      return results;
    } catch (e) {
      debugPrint('YoloView: Error parsing detections list: $e');
      return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    const viewType = 'com.ultralytics.yolo/YoloPlatformView';
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
        child: Text('Platform not supported for YoloView'),
      );
    }
    return platformView;
  }

  void _onPlatformViewCreated(int id) {
    debugPrint(
      'YoloView: Platform view created with system id: $id, our viewId: $_viewId',
    );

    _platformViewId = id;

    // _cancelResultSubscription(); // Already called in _subscribeToResults if needed

    if (widget.onResult != null || widget.onPerformanceMetrics != null) {
      debugPrint(
        'YoloView: Re-subscribing to results after platform view creation for $_viewId',
      );
      _subscribeToResults();
    }

    debugPrint('YoloView: Initializing controller with platform view ID: $id');
    _effectiveController._init(
      _methodChannel,
      id,
    ); // Re-init controller with the now valid method channel

    _methodChannel.invokeMethod('setShowUIControls', {
      'show': widget.showNativeUI,
    });

    _methodChannel.setMethodCallHandler((call) async {
      debugPrint(
        'YoloView: Received method call from platform: ${call.method} for $_viewId',
      );

      switch (call.method) {
        case 'recreateEventChannel':
          debugPrint(
            'YoloView: Platform requested recreation of event channel for $_viewId',
          );
          _cancelResultSubscription();
          Future.delayed(const Duration(milliseconds: 100), () {
            if (mounted &&
                (widget.onResult != null ||
                    widget.onPerformanceMetrics != null)) {
              _subscribeToResults();
              debugPrint('YoloView: Event channel recreated for $_viewId');
            }
          });
          return null;
        case 'onZoomChanged':
          final zoomLevel = call.arguments as double?;
          if (zoomLevel != null && widget.onZoomChanged != null) {
            debugPrint('YoloView: Zoom level changed to $zoomLevel');
            widget.onZoomChanged!(zoomLevel);
          }
          return null;
        default:
          debugPrint('YoloView: Unknown method call: ${call.method}');
          return null;
      }
    });
  }

  // Methods to be called via GlobalKey
  /// Sets the confidence threshold through the widget's state.
  ///
  /// This method can be called using a GlobalKey to access the state:
  /// ```dart
  /// final key = GlobalKey<YoloViewState>();
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

  /// Sets the camera zoom level to a specific value.
  ///
  /// The zoom level must be within the supported range of the camera.
  /// Typical values are 0.5x, 1.0x, 2.0x, 3.0x, etc.
  /// It delegates to the effective controller's setZoomLevel method.
  /// Returns a [Future] that completes when the zoom level has been set.
  Future<void> setZoomLevel(double zoomLevel) {
    return _effectiveController.setZoomLevel(zoomLevel);
  }
}
