// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

enum SliderType { none, numItems, confidence, iou }

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  int _detectionCount = 0;
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  SliderType _activeSlider = SliderType.none;

  final _yoloController = YoloViewController();
  final _yoloViewKey = GlobalKey<YoloViewState>();
  final bool _useController = true;

  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;

    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;

    if (elapsed >= 1000) {
      final calculatedFps = _frameCount * 1000 / elapsed;
      debugPrint('Calculated FPS: ${calculatedFps.toStringAsFixed(1)}');

      _currentFps = calculatedFps;
      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    // Still update detection count in the UI
    setState(() {
      _detectionCount = results.length;
    });

    // Debug first few detections
    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        'Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }
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
          // YOLO View: must be at back
          YoloView(
            key: _useController ? null : _yoloViewKey,
            controller: _useController ? _yoloController : null,
            modelPath: 'yolo11s-pose',
            task: YOLOTask.pose,
            onResult: _onDetectionResults,
            onPerformanceMetrics: (metrics) {
              if (mounted) {
                setState(() {
                  if (metrics['fps'] != null) {
                    _currentFps = metrics['fps']!;
                  }
                });
              }
            },
          ),

          // Top info pills (detection, FPS, and current threshold)
          Positioned(
            top: MediaQuery.of(context).padding.top + 16, // Safe area + spacing
            left: 16,
            right: 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Row(
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
                  ],
                ),
                const SizedBox(height: 8),
                if (_activeSlider == SliderType.confidence)
                  _buildTopPill(
                    'CONFIDENCE THRESHOLD: ${_confidenceThreshold.toStringAsFixed(2)}',
                  ),
                if (_activeSlider == SliderType.iou)
                  _buildTopPill(
                    'IOU THRESHOLD: ${_iouThreshold.toStringAsFixed(2)}',
                  ),
                if (_activeSlider == SliderType.numItems)
                  _buildTopPill('ITEMS MAX: $_numItemsThreshold'),
              ],
            ),
          ),

          // Center logo
          Positioned.fill(
            child: Align(
              alignment: Alignment.center,
              child: FractionallySizedBox(
                widthFactor: 0.5,
                heightFactor: 0.5,
                child: Image.asset(
                  'assets/logo.png',
                  color: Colors.white.withOpacity(0.4),
                ),
              ),
            ),
          ),

          // Control buttons
          Positioned(
            bottom: 32,
            right: 16,
            child: Column(
              children: [
                _buildCircleButton(
                  '1.0',
                  onPressed: () {
                    // TODO: Implement zoom logic
                  },
                ),
                const SizedBox(height: 12),
                _buildIconButton(Icons.layers, () {
                  _toggleSlider(SliderType.numItems);
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.adjust, () {
                  _toggleSlider(SliderType.confidence);
                }),
                const SizedBox(height: 12),
                _buildIconButton('assets/iou.png', () {
                  _toggleSlider(SliderType.iou);
                }),
                const SizedBox(height: 40),
              ],
            ),
          ),

          // Bottom slider overlay
          if (_activeSlider != SliderType.none)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                color: Colors.black.withOpacity(0.8),
                child: SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    activeTrackColor: Colors.yellow,
                    inactiveTrackColor: Colors.white.withOpacity(0.3),
                    thumbColor: Colors.yellow,
                    overlayColor: Colors.yellow.withOpacity(0.2),
                  ),
                  child: Slider(
                    value: _getSliderValue(),
                    min: _getSliderMin(),
                    max: _getSliderMax(),
                    divisions: _getSliderDivisions(),
                    label: _getSliderLabel(),
                    onChanged: (value) {
                      setState(() {
                        _updateSliderValue(value);
                      });
                    },
                  ),
                ),
              ),
            ),
          // Camera flip top-right
          Positioned(
            bottom: MediaQuery.of(context).padding.top + 16,
            left: 16,
            child: CircleAvatar(
              radius: 24,
              backgroundColor: Colors.black.withOpacity(0.5),
              child: IconButton(
                icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
                onPressed: () {
                  if (_useController) {
                    _yoloController.switchCamera();
                  } else {
                    _yoloViewKey.currentState?.switchCamera();
                  }
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton(dynamic iconOrAsset, VoidCallback onPressed) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withOpacity(0.2),
      child: IconButton(
        icon: iconOrAsset is IconData
            ? Icon(iconOrAsset, color: Colors.white)
            : Image.asset(
                iconOrAsset,
                width: 24,
                height: 24,
                color: Colors.white,
              ),
        onPressed: onPressed,
      ),
    );
  }

  // Zoom circle button
  Widget _buildCircleButton(String label, {required VoidCallback onPressed}) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withOpacity(0.2),
      child: TextButton(
        onPressed: onPressed,
        child: Text(label, style: const TextStyle(color: Colors.white)),
      ),
    );
  }

  // Toggle slider state
  void _toggleSlider(SliderType type) {
    setState(() {
      _activeSlider = (_activeSlider == type) ? SliderType.none : type;
    });
  }

  Widget _buildTopPill(String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  // Slider value helpers
  double _getSliderValue() {
    switch (_activeSlider) {
      case SliderType.numItems:
        return _numItemsThreshold.toDouble();
      case SliderType.confidence:
        return _confidenceThreshold;
      case SliderType.iou:
        return _iouThreshold;
      default:
        return 0;
    }
  }

  double _getSliderMin() => _activeSlider == SliderType.numItems ? 5 : 0.1;

  double _getSliderMax() => _activeSlider == SliderType.numItems ? 50 : 0.9;

  int _getSliderDivisions() => _activeSlider == SliderType.numItems ? 9 : 8;

  String _getSliderLabel() {
    switch (_activeSlider) {
      case SliderType.numItems:
        return '$_numItemsThreshold';
      case SliderType.confidence:
        return _confidenceThreshold.toStringAsFixed(1);
      case SliderType.iou:
        return _iouThreshold.toStringAsFixed(1);
      default:
        return '';
    }
  }

  // 🧠 Slider value update logic
  void _updateSliderValue(double value) {
    switch (_activeSlider) {
      case SliderType.numItems:
        _numItemsThreshold = value.toInt();
        if (_useController) {
          _yoloController.setNumItemsThreshold(_numItemsThreshold);
        } else {
          _yoloViewKey.currentState?.setNumItemsThreshold(_numItemsThreshold);
        }
        break;
      case SliderType.confidence:
        _confidenceThreshold = value;
        if (_useController) {
          _yoloController.setConfidenceThreshold(value);
        } else {
          _yoloViewKey.currentState?.setConfidenceThreshold(value);
        }
        break;
      case SliderType.iou:
        _iouThreshold = value;
        if (_useController) {
          _yoloController.setIoUThreshold(value);
        } else {
          _yoloViewKey.currentState?.setIoUThreshold(value);
        }
        break;
      default:
        break;
    }
  }
}
