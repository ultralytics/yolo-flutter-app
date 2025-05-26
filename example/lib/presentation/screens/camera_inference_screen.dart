import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  int _detectionCount = 0;
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  String _lastDetection = "";
  double _currentProcessingTimeMs = 0.0;
  double _currentFps = 0.0;

  final _yoloController = YoloViewController();
  final _yoloViewKey = GlobalKey<YoloViewState>();
  bool _useController = true;

  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;

    debugPrint('_onDetectionResults called with ${results.length} results');

    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        '  Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }

    setState(() {
      _detectionCount = results.length;
      if (results.isNotEmpty) {
        final topDetection = results.reduce(
          (a, b) => a.confidence > b.confidence ? a : b,
        );
        _lastDetection =
            "${topDetection.className} (${(topDetection.confidence * 100).toStringAsFixed(1)}%)";
        debugPrint(
          'Updated state: count=$_detectionCount, top=$_lastDetection',
        );
      } else {
        _lastDetection = "None";
        debugPrint('Updated state: No detections');
      }
    });
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_useController) {
        _yoloController.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      } else {
        _yoloViewKey.currentState?.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Fullscreen YOLO Camera View
          YoloView(
            key: _useController ? null : _yoloViewKey,
            controller: _useController ? _yoloController : null,
            modelPath: 'yolo11s-pose',
            task: YOLOTask.pose,
            onResult: _onDetectionResults,
            onPerformanceMetrics: (metrics) {
              if (mounted) {
                setState(() {
                  _currentProcessingTimeMs = metrics['processingTimeMs'] ?? 0.0;
                  _currentFps = metrics['fps'] ?? 0.0;
                });
              }
            },
          ),

          // Center Logo Overlay
          Positioned.fill(
            child: Align(
              alignment: Alignment.center,
              child: FractionallySizedBox(
                widthFactor: 0.5,
                heightFactor: 0.5,
                child: Image.asset('assets/logo.png',
                    color: Colors.white.withAlpha((0.4 * 255).toInt())),
              ),
            ),
          ),

          // Vertical control buttons (bottom right)
          Positioned(
            bottom: 32,
            right: 16,
            child: Column(
              children: [
                _buildCircleButton('1.0x', onPressed: () {
                  // handle zoom
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.settings, onPressed: () {
                  // open settings
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.switch_camera, onPressed: () {
                  if (_useController) {
                    _yoloController.switchCamera();
                  } else {
                    _yoloViewKey.currentState?.switchCamera();
                  }
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.logout, onPressed: () {
                  Navigator.of(context).pop();
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

// Helper: Icon button builder
  Widget _buildIconButton(IconData icon, {required VoidCallback onPressed}) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withAlpha((0.5 * 255).toInt()),
      child: IconButton(
        icon: Icon(icon, color: Colors.white),
        onPressed: onPressed,
      ),
    );
  }

// Helper: Zoom button
  Widget _buildCircleButton(String label, {required VoidCallback onPressed}) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withOpacity(0.5),
      child: TextButton(
        onPressed: onPressed,
        child: Text(label, style: const TextStyle(color: Colors.white)),
      ),
    );
  }
}
