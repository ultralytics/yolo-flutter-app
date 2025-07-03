// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import '../../models/model_type.dart';
import '../../services/model_manager.dart';

/// Example demonstrating front camera usage with frame rate control
class FrontCameraExample extends StatefulWidget {
  const FrontCameraExample({super.key});

  @override
  State<FrontCameraExample> createState() => _FrontCameraExampleState();
}

class _FrontCameraExampleState extends State<FrontCameraExample> {
  final _controller = YOLOViewController();
  int _detectionCount = 0;
  double _currentFps = 0.0;
  int _targetFps = 5;
  String? _modelPath;
  bool _isModelLoading = false;
  String _loadingMessage = '';
  bool _isFrontCamera =
      false; // Track camera state - YOLOView starts with back camera

  // Streaming config for slower detection
  late YOLOStreamingConfig _streamingConfig;
  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();

    // Initialize ModelManager
    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        debugPrint(
          'Download progress: ${(progress * 100).toStringAsFixed(1)}%',
        );
      },
      onStatusUpdate: (message) {
        debugPrint('Model status: $message');
      },
    );

    _updateStreamingConfig(_targetFps);
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() {
      _isModelLoading = true;
      _loadingMessage = 'Loading detection model...';
    });

    try {
      // Use the same model as the working camera inference screen
      final modelPath = await _modelManager.getModelPath(ModelType.detect);

      if (mounted) {
        setState(() {
          _modelPath = modelPath;
          _isModelLoading = false;
          _loadingMessage = '';
        });

        if (modelPath != null) {
          debugPrint(
            'FrontCameraExample: Model loaded successfully: $modelPath',
          );
        } else {
          debugPrint('FrontCameraExample: Failed to load model');
        }
      }
    } catch (e) {
      debugPrint('FrontCameraExample: Error loading model: $e');
      if (mounted) {
        setState(() {
          _isModelLoading = false;
          _loadingMessage = 'Failed to load model';
        });
      }
    }
  }

  void _updateStreamingConfig(int fps) {
    _streamingConfig = YOLOStreamingConfig(
      inferenceFrequency: fps, // Control how often inference runs
      maxFPS: fps, // Control how often results are sent to Flutter
      includeDetections: true,
      includeClassifications: true,
      includeProcessingTimeMs: true,
      includeFps: true,
      includeMasks: false,
      includePoses: false,
      includeOBB: false,
      includeOriginalImage: false,
    );

    _controller.setStreamingConfig(_streamingConfig);
    setState(() {
      _targetFps = fps;
    });
  }

  void _onDetectionResults(List<YOLOResult> results) {
    setState(() {
      _detectionCount = results.length;
    });

    // Enhanced debugging
    debugPrint('=== FRONT CAMERA DETECTION ===');
    debugPrint('Total detections: ${results.length}');
    debugPrint('Target FPS: $_targetFps');
    debugPrint('Current FPS: ${_currentFps.toStringAsFixed(1)}');

    if (results.isEmpty) {
      debugPrint('⚠️ NO DETECTIONS - Possible issues:');
      debugPrint('   - No objects in view');
      debugPrint('   - Confidence threshold too high (0.5)');
      debugPrint('   - Model not suitable for front camera');
      debugPrint('   - Frame rate too low');
    } else {
      debugPrint('✅ DETECTIONS FOUND:');
      // Debug first few detections
      for (var i = 0; i < results.length && i < 3; i++) {
        final r = results[i];
        debugPrint(
          '   Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
        );
      }
    }
    debugPrint('=============================');
  }

  void _onPerformanceMetrics(YOLOPerformanceMetrics metrics) {
    setState(() {
      _currentFps = metrics.fps;
    });
  }

  @override
  Widget build(BuildContext context) {
    final orientation = MediaQuery.of(context).orientation;
    final isLandscape = orientation == Orientation.landscape;

    return Scaffold(
      body: Stack(
        children: [
          // YOLO View: must be at back
          if (_modelPath != null && !_isModelLoading)
            YOLOView(
              controller: _controller,
              streamingConfig: _streamingConfig,
              modelPath: _modelPath!,
              task: YOLOTask.detect,
              confidenceThreshold: 0.5,
              onResult: _onDetectionResults,
              onPerformanceMetrics: _onPerformanceMetrics,
            )
          else if (_isModelLoading)
            IgnorePointer(
              child: Container(
                color: Colors.black87,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Ultralytics logo
                      Image.asset(
                        'assets/logo.png',
                        width: 120,
                        height: 120,
                        color: Colors.white.withValues(alpha: 0.8),
                      ),
                      const SizedBox(height: 32),
                      // Loading message
                      Text(
                        _loadingMessage,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
            )
          else
            const Center(
              child: Text(
                'No model loaded',
                style: TextStyle(color: Colors.white),
              ),
            ),

          // Top info pills (detection, FPS, and current threshold)
          Positioned(
            top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16),
            left: isLandscape ? 8 : 16,
            right: isLandscape ? 8 : 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Title
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    'FRONT CAMERA EXAMPLE',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                SizedBox(height: isLandscape ? 8 : 12),
                // Camera indicator
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    _isFrontCamera ? 'FRONT CAMERA' : 'BACK CAMERA',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                SizedBox(height: isLandscape ? 8 : 12),
                IgnorePointer(
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'DETECTIONS: $_detectionCount',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Text(
                        'FPS: ${_currentFps.toStringAsFixed(1)}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Text(
                        'TARGET: $_targetFps',
                        style: const TextStyle(
                          color: Colors.yellow,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // Center logo - only show when camera is active
          if (_modelPath != null && !_isModelLoading)
            Positioned.fill(
              child: IgnorePointer(
                child: Align(
                  alignment: Alignment.center,
                  child: FractionallySizedBox(
                    widthFactor: isLandscape ? 0.3 : 0.5,
                    heightFactor: isLandscape ? 0.3 : 0.5,
                    child: Image.asset(
                      'assets/logo.png',
                      color: Colors.white.withValues(alpha: 0.4),
                    ),
                  ),
                ),
              ),
            ),

          // Control buttons
          Positioned(
            bottom: isLandscape ? 16 : 32,
            right: isLandscape ? 8 : 16,
            child: Column(
              children: [
                _buildIconButton(Icons.tune, () {
                  // Cycle through confidence thresholds: 0.5 -> 0.3 -> 0.1 -> 0.5
                  double currentThreshold =
                      0.5; // You might want to track this in state
                  double nextThreshold;
                  if (currentThreshold >= 0.5) {
                    nextThreshold = 0.3;
                  } else if (currentThreshold >= 0.3) {
                    nextThreshold = 0.1;
                  } else {
                    nextThreshold = 0.5;
                  }
                  _controller.setConfidenceThreshold(nextThreshold);
                  debugPrint(
                    'FrontCameraExample: Confidence threshold set to $nextThreshold',
                  );
                }),
                SizedBox(height: isLandscape ? 8 : 12),
                _buildIconButton(Icons.speed, () {
                  // Cycle through frame rates: 5 -> 10 -> 15 -> 30 -> 5
                  int nextFps;
                  if (_targetFps <= 5) {
                    nextFps = 10;
                  } else if (_targetFps <= 10) {
                    nextFps = 15;
                  } else if (_targetFps <= 15) {
                    nextFps = 30;
                  } else {
                    nextFps = 5;
                  }
                  _updateStreamingConfig(nextFps);
                }),
                SizedBox(height: isLandscape ? 8 : 12),
                _buildIconButton(Icons.switch_camera, () {
                  setState(() {
                    _isFrontCamera = !_isFrontCamera;
                  });
                  _controller.switchCamera();
                  debugPrint(
                    'FrontCameraExample: Camera switched to ${_isFrontCamera ? "FRONT" : "BACK"}',
                  );
                }),
                SizedBox(height: isLandscape ? 8 : 12),
                _buildIconButton(Icons.refresh, () {
                  _loadModel();
                }),
                SizedBox(height: isLandscape ? 16 : 40),
              ],
            ),
          ),

          // Camera flip top-left
          Positioned(
            bottom:
                MediaQuery.of(context).padding.top + (isLandscape ? 32 : 16),
            left: isLandscape ? 32 : 16,
            child: CircleAvatar(
              radius: isLandscape ? 20 : 24,
              backgroundColor: Colors.black.withValues(alpha: 0.5),
              child: IconButton(
                icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
                onPressed: () {
                  setState(() {
                    _isFrontCamera = !_isFrontCamera;
                  });
                  _controller.switchCamera();
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Builds a circular button with an icon
  Widget _buildIconButton(IconData icon, VoidCallback onPressed) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withValues(alpha: 0.2),
      child: IconButton(
        icon: Icon(icon, color: Colors.white),
        onPressed: onPressed,
      ),
    );
  }
}
