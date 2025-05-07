// lib/yolo_view.dart

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

/// Controller for interacting with a YoloView.
///
/// This controller provides methods to control the YoloView after it has been created,
/// such as setting confidence and IoU thresholds.
///
/// Usage:
/// ```dart
/// final controller = YoloViewController();
/// YoloView(
///   controller: controller,
///   modelPath: 'assets/models/yolo11n.tflite',
///   task: YOLOTask.detect,
/// )
///
/// // Set thresholds
/// controller.setConfidenceThreshold(0.6);
/// controller.setIoUThreshold(0.4);
/// ```
class YoloViewController {
  /// The method channel used to communicate with the platform view
  MethodChannel? _methodChannel;

  /// Default values for thresholds
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;

  /// Current confidence threshold value (0.0 to 1.0)
  double get confidenceThreshold => _confidenceThreshold;
  
  /// Current IoU threshold value (0.0 to 1.0)
  double get iouThreshold => _iouThreshold;

  /// Internal method to initialize the controller with a method channel
  void _init(MethodChannel methodChannel) {
    _methodChannel = methodChannel;
    
    // Apply any thresholds that were set before the view was created
    _applyThresholds();
  }

  /// Apply the current threshold values to the platform view
  Future<void> _applyThresholds() async {
    if (_methodChannel == null) {
      debugPrint('YoloViewController: Warning - Cannot apply thresholds, view not yet created');
      return;
    }

    try {
      // First try the combined method
      await _methodChannel!.invokeMethod('setThresholds', {
        'confidenceThreshold': _confidenceThreshold,
        'iouThreshold': _iouThreshold,
      });
      debugPrint('YoloViewController: Applied thresholds - confidence: $_confidenceThreshold, IoU: $_iouThreshold');
    } catch (e) {
      debugPrint('YoloViewController: Error applying combined thresholds: $e');
      
      // Fall back to individual methods if combined fails
      try {
        debugPrint('YoloViewController: Trying individual threshold methods as fallback');
        
        // Set confidence threshold
        await _methodChannel!.invokeMethod('setConfidenceThreshold', {
          'threshold': _confidenceThreshold,
        });
        debugPrint('YoloViewController: Applied confidence threshold: $_confidenceThreshold');
        
        // Set IoU threshold
        await _methodChannel!.invokeMethod('setIoUThreshold', {
          'threshold': _iouThreshold,
        });
        debugPrint('YoloViewController: Applied IoU threshold: $_iouThreshold');
      } catch (e2) {
        debugPrint('YoloViewController: Error applying individual thresholds: $e2');
      }
    }
  }

  /// Set the confidence threshold for detections
  /// 
  /// The confidence threshold determines the minimum confidence score 
  /// required for a detection to be included in results. 
  /// Value should be between 0.0 and 1.0.
  Future<void> setConfidenceThreshold(double threshold) async {
    // Clamp value between 0.0 and 1.0
    final clampedThreshold = threshold.clamp(0.0, 1.0);
    _confidenceThreshold = clampedThreshold;
    
    if (_methodChannel == null) {
      debugPrint('YoloViewController: Warning - Cannot apply confidence threshold, view not yet created');
      return;
    }

    try {
      // Call the specific method directly to avoid any issues
      await _methodChannel!.invokeMethod('setConfidenceThreshold', {
        'threshold': clampedThreshold,
      });
      debugPrint('YoloViewController: Applied confidence threshold: $_confidenceThreshold');
    } catch (e) {
      debugPrint('YoloViewController: Error applying confidence threshold: $e');
      // Try the fallback on error
      return _applyThresholds();
    }
  }

  /// Set the IoU (Intersection over Union) threshold for Non-Maximum Suppression
  /// 
  /// The IoU threshold determines how much bounding boxes can overlap before 
  /// they're merged in Non-Maximum Suppression algorithm.
  /// Higher values result in fewer merged boxes.
  /// Value should be between 0.0 and 1.0.
  Future<void> setIoUThreshold(double threshold) async {
    // Clamp value between 0.0 and 1.0
    final clampedThreshold = threshold.clamp(0.0, 1.0);
    _iouThreshold = clampedThreshold;
    
    if (_methodChannel == null) {
      debugPrint('YoloViewController: Warning - Cannot apply IoU threshold, view not yet created');
      return;
    }

    try {
      // Call the specific method directly to avoid any issues
      await _methodChannel!.invokeMethod('setIoUThreshold', {
        'threshold': clampedThreshold,
      });
      debugPrint('YoloViewController: Applied IoU threshold: $_iouThreshold');
    } catch (e) {
      debugPrint('YoloViewController: Error applying IoU threshold: $e');
      // Try the fallback on error
      return _applyThresholds();
    }
  }

  /// Set both confidence and IoU thresholds at once
  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
  }) async {
    if (confidenceThreshold != null) {
      _confidenceThreshold = confidenceThreshold.clamp(0.0, 1.0);
    }
    if (iouThreshold != null) {
      _iouThreshold = iouThreshold.clamp(0.0, 1.0);
    }
    return _applyThresholds();
  }
}

/// A Flutter widget that displays a platform view for YOLO object detection.
///
/// This widget creates a native view that runs YOLO models directly using the device's
/// camera or for processing static images. It supports object detection, segmentation,
/// classification, pose estimation, and oriented bounding box detection.
///
/// The widget handles platform-specific implementations for both Android and iOS.
///
/// There are two ways to control the YoloView after it's created:
///
/// 1. Using a controller (recommended for reusable code):
/// ```dart
/// // Create a controller to interact with the view
/// final yoloController = YoloViewController();
/// 
/// YoloView(
///   controller: yoloController,
///   modelPath: 'assets/models/yolo11n.tflite',
///   task: YOLOTask.detect,
///   onResult: (results) {
///     // Process detection results
///     print('Detected ${results.length} objects');
///   },
/// )
/// 
/// // Set thresholds using the controller
/// yoloController.setConfidenceThreshold(0.5);
/// yoloController.setIoUThreshold(0.45);
/// ```
///
/// 2. Using the widget directly with a GlobalKey (simpler for single instance):
/// ```dart
/// // Create a GlobalKey to access the YoloView
/// final yoloViewKey = GlobalKey<YoloViewState>();
/// 
/// YoloView(
///   key: yoloViewKey,  // Important: Set the key
///   modelPath: 'assets/models/yolo11n.tflite',
///   task: YOLOTask.detect,
///   onResult: (results) {
///     // Process detection results
///   },
/// )
/// 
/// // Later, update thresholds directly
/// yoloViewKey.currentState?.setConfidenceThreshold(0.7);
/// // Or update both thresholds at once
/// yoloViewKey.currentState?.setThresholds(
///   confidenceThreshold: 0.6,
///   iouThreshold: 0.5,
/// );
/// ```
///
/// You can choose whichever approach works best for your application.
/// For most cases, using a controller is recommended as it's a more standard pattern in Flutter.
/// The GlobalKey approach is provided as a convenience for simple use cases.
/// 
/// If no controller is provided, YoloView will create an internal controller automatically.
/// You can still adjust settings using the GlobalKey approach if needed.
class YoloView extends StatefulWidget {
  /// Path to the YOLO model file. This should be a TFLite model file.
  final String modelPath;
  
  /// The type of task this YOLO model will perform (detection, segmentation, etc.)
  final YOLOTask task;
  
  /// Controller for interacting with the YoloView.
  /// Use this to control the view after it's been created.
  final YoloViewController? controller;
  
  /// Camera resolution string (e.g. "720p", "1080p").
  /// Note: This parameter is not currently implemented.
  final String cameraResolution;
  
  /// Callback function that receives detection results.
  /// Called each time new detection results are available.
  final Function(List<YOLOResult>)? onResult;

  /// Creates a YoloView widget that displays a platform-specific native view
  /// for YOLO object detection and other computer vision tasks.
  ///
  /// The [modelPath] should point to a valid TFLite model file in the assets.
  /// The [task] specifies what type of inference will be performed.
  /// The [controller] provides methods to control the view after creation.
  /// If not provided, an internal controller will be created automatically.
  /// The [onResult] callback provides detection results.
  const YoloView({
    super.key,
    required this.modelPath,
    required this.task,
    this.controller,
    this.cameraResolution = '720p',
    this.onResult,
  });
  
  /// Sets the confidence threshold for detections.
  ///
  /// The confidence threshold determines the minimum confidence score for a detection
  /// to be included in results. Value should be between 0.0 and 1.0.
  ///
  /// This is a convenience method that forwards to the controller.
  Future<void> setConfidenceThreshold(double threshold) {
    // This method is implemented as a wrapper around the YoloViewState
    // This allows users to call methods directly on the YoloView widget
    // The actual state will be accessed in a different way at runtime
    throw UnimplementedError('Cannot call on the widget directly - this is implemented by the state');
  }

  /// Sets the IoU threshold for Non-Maximum Suppression.
  ///
  /// The IoU threshold determines how much bounding boxes can overlap before they're merged.
  /// Higher values result in fewer merged boxes. Value should be between 0.0 and 1.0.
  ///
  /// This is a convenience method that forwards to the controller.
  Future<void> setIoUThreshold(double threshold) {
    // See above comment
    throw UnimplementedError('Cannot call on the widget directly - this is implemented by the state');
  }
  
  /// Sets both confidence and IoU thresholds at once.
  ///
  /// This is a convenience method that forwards to the controller.
  Future<void> setThresholds({double? confidenceThreshold, double? iouThreshold}) {
    // See above comment
    throw UnimplementedError('Cannot call on the widget directly - this is implemented by the state');
  }

  @override
  State<YoloView> createState() => YoloViewState();
}

// Public state class to enable GlobalKey access
// Note: We remove the underscore to make this truly public
class YoloViewState extends State<YoloView> {
  // Event channel for receiving detection results
  late EventChannel _resultEventChannel;
  StreamSubscription<dynamic>? _resultSubscription;
  // Method channel for controlling the view
  late MethodChannel _methodChannel;
  
  // The controller that will be used (either from props or internal)
  late YoloViewController _effectiveController;
  // Flag to track if we created the controller internally
  bool _isInternalController = false;
  
  // Keep an ID to uniquely identify this widget instance
  final String _viewId = UniqueKey().toString();
  
  @override
  void initState() {
    super.initState();
    // Initialize the event channel for receiving detection results
    _resultEventChannel = EventChannel('com.ultralytics.yolo/detectionResults_$_viewId');
    // Initialize the method channel for sending control messages
    _methodChannel = MethodChannel('com.ultralytics.yolo/controlChannel_$_viewId');
    
    // Set up the controller (either use provided one or create internal)
    _setupController();
    
    // Listen for detection results if a callback is provided
    if (widget.onResult != null) {
      _subscribeToResults();
    }
  }
  
  void _setupController() {
    if (widget.controller != null) {
      // Use externally provided controller
      _effectiveController = widget.controller!;
      _isInternalController = false;
    } else {
      // Create an internal controller if none provided
      _effectiveController = YoloViewController();
      _isInternalController = true;
    }
    
    // Initialize the controller with our method channel
    _effectiveController._init(_methodChannel);
  }
  
  @override
  void didUpdateWidget(YoloView oldWidget) {
    super.didUpdateWidget(oldWidget);
    
    // Update controller if it changed
    if (oldWidget.controller != widget.controller) {
      // If we were using an internal controller before, we don't need to do anything special
      // as we'll create a new one if needed in _setupController
      _setupController();
    }
    
    // Update detection results callback subscription if needed
    if (oldWidget.onResult != widget.onResult) {
      if (widget.onResult == null) {
        _cancelResultSubscription();
      } else if (oldWidget.onResult == null) {
        _subscribeToResults();
      }
    }
  }
  
  @override
  void dispose() {
    // Clean up the internal controller if we created one
    if (_isInternalController) {
      // No explicit cleanup needed for now, but can add it if required
    }
    
    // Clean up resources
    _cancelResultSubscription();
    super.dispose();
  }
  
  // Start listening for detection results
  void _subscribeToResults() {
    _cancelResultSubscription(); // Cancel any existing subscription first
    
    debugPrint('YoloView: Setting up event stream listener for channel: ${_resultEventChannel.name}');
    
    _resultSubscription = _resultEventChannel.receiveBroadcastStream().listen(
      (dynamic event) {
        debugPrint('YoloView: Received event from native platform: $event');
        
        // Check for test messages from the platform
        if (event is Map && event.containsKey('test')) {
          debugPrint('YoloView: Received test message: ${event['test']}');
          return;
        }
        
        if (event is Map && widget.onResult != null) {
          try {
            // Validate the event structure
            if (!event.containsKey('detections')) {
              debugPrint('YoloView: Warning - Event missing "detections" key: $event');
              return;
            }
            
            final List<dynamic> detections = event['detections'] ?? [];
            debugPrint('YoloView: Received ${detections.length} detections');
            
            // For each detection, print basic info for debugging
            for (var i = 0; i < detections.length && i < 3; i++) {
              final detection = detections[i];
              final className = detection['className'] ?? 'unknown';
              final confidence = detection['confidence'] ?? 0.0;
              debugPrint('YoloView: Detection $i - $className (${(confidence * 100).toStringAsFixed(1)}%)');
            }
            
            final results = _parseDetectionResults(event);
            debugPrint('YoloView: Parsed results count: ${results.length}');
            
            // Invoke callback with the results
            widget.onResult!(results);
            debugPrint('YoloView: Called onResult callback with results');
          } catch (e) {
            debugPrint('Error parsing detection results: $e');
            debugPrint('Error stack trace: ${StackTrace.current}');
            
            // Try to provide more context about the error
            if (event is Map) {
              debugPrint('YoloView: Event keys: ${event.keys.toList()}');
              if (event.containsKey('detections')) {
                final detections = event['detections'];
                debugPrint('YoloView: Detections type: ${detections.runtimeType}');
                if (detections is List && detections.isNotEmpty) {
                  debugPrint('YoloView: First detection keys: ${detections.first?.keys?.toList()}');
                }
              }
            }
          }
        } else {
          debugPrint('YoloView: Received invalid event format or onResult is null');
          debugPrint('YoloView: Event type: ${event.runtimeType}, onResult null: ${widget.onResult == null}');
        }
      },
      onError: (dynamic error) {
        debugPrint('Error from detection results stream: $error');
        debugPrint('Error stack trace: ${StackTrace.current}');
        
        // Try to resubscribe after a delay on error
        Future.delayed(const Duration(seconds: 2), () {
          if (_resultSubscription == null) {
            debugPrint('YoloView: Attempting to resubscribe after error');
            _subscribeToResults();
          }
        });
      },
      onDone: () {
        debugPrint('YoloView: Event stream closed');
        _resultSubscription = null;
      }
    );
    
    debugPrint('YoloView: Event stream listener setup complete');
  }
  
  // Cancel the detection results subscription
  void _cancelResultSubscription() {
    if (_resultSubscription != null) {
      debugPrint('YoloView: Cancelling existing result subscription');
      _resultSubscription!.cancel();
      _resultSubscription = null;
    }
  }
  
  
  // Parse detection results from platform event
  List<YOLOResult> _parseDetectionResults(Map<dynamic, dynamic> event) {
    final List<dynamic> detectionsData = event['detections'] ?? [];
    debugPrint('YoloView: Parsing ${detectionsData.length} detections');
    
    // Check detection data structure in detail
    if (detectionsData.isNotEmpty) {
      final first = detectionsData.first;
      debugPrint('YoloView: First detection structure: ${first.runtimeType} with keys: ${first is Map ? first.keys.toList() : "not a map"}');
      
      if (first is Map) {
        // Check expected fields with detailed values
        debugPrint('YoloView: ClassIndex: ${first["classIndex"]}');
        debugPrint('YoloView: ClassName: ${first["className"]}');
        debugPrint('YoloView: Confidence: ${first["confidence"]}');
        debugPrint('YoloView: BoundingBox: ${first["boundingBox"]}');
        debugPrint('YoloView: NormalizedBox: ${first["normalizedBox"]}');
      }
    }
    
    // Convert each detection to YOLOResult
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
      debugPrint('YoloView: Error parsing detections: $e');
      // Return empty list on error
      return [];
    }
  }

  @override
  Widget build(BuildContext context) {
    const viewType = 'com.ultralytics.yolo/YoloPlatformView';
    final creationParams = <String, dynamic>{
      'modelPath': widget.modelPath,
      'task': widget.task.name, // "detect" / "classify" etc.
      'confidenceThreshold': _effectiveController.confidenceThreshold,
      'iouThreshold': _effectiveController.iouThreshold,
      'viewId': _viewId, // Pass the unique ID to correlate with event channel
    };

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
    }
    // fallback for unsupported platforms
    return const Text('Platform not supported for YoloView');
  }
  
  // Called when the platform view is created
  void _onPlatformViewCreated(int id) {
    debugPrint('YoloView: Platform view created with id: $id and viewId: $_viewId');
    
    // Clean up existing subscription
    _cancelResultSubscription();
    
    // Re-setup event channel if result callback is provided
    if (widget.onResult != null) {
      debugPrint('YoloView: Re-subscribing to results after platform view creation');
      _subscribeToResults();
    }
    
    // Re-initialize controller (method channel becomes available)
    _effectiveController._init(_methodChannel);
    
    // Set up method channel handler for platform messages
    _methodChannel.setMethodCallHandler((call) async {
      debugPrint('YoloView: Received method call from platform: ${call.method}');
      
      switch (call.method) {
        case 'recreateEventChannel':
          debugPrint('YoloView: Platform requested recreation of event channel');
          // Recreate subscription
          _cancelResultSubscription();
          Future.delayed(const Duration(milliseconds: 100), () {
            if (mounted && widget.onResult != null) {
              _subscribeToResults();
              debugPrint('YoloView: Event channel recreated');
            }
          });
          return null;
        default:
          debugPrint('YoloView: Unknown method call: ${call.method}');
          return null;
      }
    });
  }
  
  /// Sets the confidence threshold for detections.
  ///
  /// The confidence threshold determines the minimum confidence score for a detection
  /// to be included in results. Value should be between 0.0 and 1.0.
  Future<void> setConfidenceThreshold(double threshold) {
    return _effectiveController.setConfidenceThreshold(threshold);
  }

  /// Sets the IoU threshold for Non-Maximum Suppression.
  ///
  /// The IoU threshold determines how much bounding boxes can overlap before they're merged.
  /// Higher values result in fewer merged boxes. Value should be between 0.0 and 1.0.
  Future<void> setIoUThreshold(double threshold) {
    return _effectiveController.setIoUThreshold(threshold);
  }
  
  /// Sets both confidence and IoU thresholds at once.
  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
  }) {
    return _effectiveController.setThresholds(
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
    );
  }
}

