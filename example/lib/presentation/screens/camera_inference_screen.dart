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
  double _currentProcessingTimeMs = 0.0;
  double _currentFps = 0.0;

  SliderType _activeSlider = SliderType.none;

  final _yoloController = YoloViewController();
  final _yoloViewKey = GlobalKey<YoloViewState>();
  bool _useController = true;

  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;

    setState(() {
      _detectionCount = results.length;
    });

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
          // 🟢 YOLO View: must be at back
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

          // 🟡 Center logo
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

          // 🔵 Control buttons
          Positioned(
            bottom: 32,
            right: 16,
            child: Column(
              children: [
                _buildCircleButton('1.0x', onPressed: () {
                  // Zoom logic (static for now)
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.layers, () {
                  _toggleSlider(SliderType.numItems);
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.adjust, () {
                  _toggleSlider(SliderType.confidence);
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.filter_alt, () {
                  _toggleSlider(SliderType.iou);
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.logout, () {
                  Navigator.of(context).pop();
                }),
              ],
            ),
          ),

          // 🔻 Bottom slider overlay
          if (_activeSlider != SliderType.none)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
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
        ],
      ),
    );
  }

  // 🔘 Round icon button
  Widget _buildIconButton(IconData icon, VoidCallback onPressed) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withOpacity(0.5),
      child: IconButton(
        icon: Icon(icon, color: Colors.white),
        onPressed: onPressed,
      ),
    );
  }

  // 🔘 Zoom circle button
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

  // 🧠 Toggle slider state
  void _toggleSlider(SliderType type) {
    setState(() {
      _activeSlider = (_activeSlider == type) ? SliderType.none : type;
    });
  }

  // 📊 Slider value helpers
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
