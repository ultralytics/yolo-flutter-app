// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';

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

  // Streaming config for slower detection
  late YOLOStreamingConfig _streamingConfig;

  @override
  void initState() {
    super.initState();
    _updateStreamingConfig(_targetFps);
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

    // Debug output
    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        'Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }
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
          YOLOView(
            controller: _controller,
            streamingConfig: _streamingConfig,
            modelPath:
                'assets/models/yolov8n.tflite', // Update with your model path
            task: YOLOTask.detect,
            confidenceThreshold: 0.5,
            onResult: _onDetectionResults,
            onPerformanceMetrics: _onPerformanceMetrics,
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
              ],
            ),
          ),
        ],
      ),
    );
  }
}
