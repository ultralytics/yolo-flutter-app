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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Front Camera Example'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        children: [
          // YOLO View with front camera configuration
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
            Container(
              color: Colors.black87,
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      _loadingMessage,
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                  ],
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

          // Top info overlay
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.7),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Detections: $_detectionCount',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    'FPS: ${_currentFps.toStringAsFixed(1)}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    'Target: $_targetFps',
                    style: const TextStyle(
                      color: Colors.yellow,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Control buttons
          Positioned(
            bottom: 32,
            right: 16,
            child: Column(
              children: [
                // Confidence threshold control
                FloatingActionButton(
                  heroTag: 'confidence',
                  onPressed: () {
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
                  },
                  backgroundColor: Colors.orange,
                  child: const Icon(Icons.tune, color: Colors.white),
                ),
                const SizedBox(height: 16),

                // Frame rate control
                FloatingActionButton(
                  heroTag: 'fps',
                  onPressed: () {
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
                  },
                  backgroundColor: Colors.blue,
                  child: const Icon(Icons.speed, color: Colors.white),
                ),
                const SizedBox(height: 16),

                // Camera switch
                FloatingActionButton(
                  heroTag: 'camera',
                  onPressed: () {
                    _controller.switchCamera();
                  },
                  backgroundColor: Colors.green,
                  child: const Icon(Icons.switch_camera, color: Colors.white),
                ),
                const SizedBox(height: 16),

                // Reload model
                FloatingActionButton(
                  heroTag: 'reload',
                  onPressed: () {
                    _loadModel();
                  },
                  backgroundColor: Colors.red,
                  child: const Icon(Icons.refresh, color: Colors.white),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
