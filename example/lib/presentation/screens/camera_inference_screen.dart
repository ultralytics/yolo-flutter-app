// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../../models/models.dart';
import '../../services/model_manager.dart';

/// A screen that demonstrates real-time YOLO inference using the device camera.
///
/// This screen provides:
/// - Live camera feed with YOLO object detection
/// - Model selection (detect, segment, classify, pose, obb)
/// - Adjustable thresholds (confidence, IoU, max detections)
/// - Camera controls (flip, zoom)
/// - Performance metrics (FPS)
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
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();
  SliderType _activeSlider = SliderType.none;
  ModelType _selectedModel = ModelType.detect;
  bool _isModelLoading = false;
  String? _modelPath;
  String _loadingMessage = '';
  double _downloadProgress = 0.0;
  double _currentZoomLevel = 1.0;
  bool _isFrontCamera = false;
  final _yoloController = YOLOViewController();
  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();

    _modelManager = ModelManager(
      onDownloadProgress: (progress) =>
          mounted ? setState(() => _downloadProgress = progress) : null,
      onStatusUpdate: (message) =>
          mounted ? setState(() => _loadingMessage = message) : null,
    );
    _loadModelForPlatform();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _yoloController.setThresholds(
        confidenceThreshold: _confidenceThreshold,
        iouThreshold: _iouThreshold,
        numItemsThreshold: _numItemsThreshold,
      );
    });
  }

  /// Called when new detection results are available
  ///
  /// Updates the UI with:
  /// - Number of detections
  /// - FPS calculation
  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;
    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;
    if (elapsed >= 1000) {
      _currentFps = _frameCount * 1000 / elapsed;
      _frameCount = 0;
      _lastFpsUpdate = now;
    }
    setState(() => _detectionCount = results.length);
  }

  @override
  Widget build(BuildContext context) {
    final isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;
    final gap = SizedBox(height: isLandscape ? 8 : 12);

    return Scaffold(
      body: Stack(
        children: [
          if (_modelPath != null && !_isModelLoading)
            YOLOView(
              key: const ValueKey('yolo_view_static'),
              controller: _yoloController,
              modelPath: _modelPath!,
              task: _selectedModel.task,
              streamingConfig: const YOLOStreamingConfig.minimal(),
              onResult: _onDetectionResults,
              onPerformanceMetrics: (metrics) =>
                  mounted ? setState(() => _currentFps = metrics.fps) : null,
              onZoomChanged: (zoomLevel) => mounted
                  ? setState(() => _currentZoomLevel = zoomLevel)
                  : null,
            )
          else if (_isModelLoading)
            Container(
              color: Colors.black87,
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Image.asset(
                      'assets/logo.png',
                      width: 120,
                      height: 120,
                      color: Colors.white.withValues(alpha: 0.8),
                    ),
                    const SizedBox(height: 32),
                    Text(
                      _loadingMessage,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    if (_downloadProgress > 0) ...[
                      const SizedBox(height: 24),
                      SizedBox(
                        width: 200,
                        child: LinearProgressIndicator(
                          value: _downloadProgress,
                          backgroundColor: Colors.white24,
                          valueColor: const AlwaysStoppedAnimation(
                            Colors.white,
                          ),
                          minHeight: 4,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        '${(_downloadProgress * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                        ),
                      ),
                    ],
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

          // Top info pills (detection, FPS, and current threshold)
          Positioned(
            top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16),
            left: isLandscape ? 8 : 16,
            right: isLandscape ? 8 : 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Model selector
                _buildModelSelector(),
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
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                if (_activeSlider == SliderType.confidence)
                  _pill(
                    'CONFIDENCE THRESHOLD: ${_confidenceThreshold.toStringAsFixed(2)}',
                  ),
                if (_activeSlider == SliderType.iou)
                  _pill('IOU THRESHOLD: ${_iouThreshold.toStringAsFixed(2)}'),
                if (_activeSlider == SliderType.numItems)
                  _pill('ITEMS MAX: $_numItemsThreshold'),
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
                if (!_isFrontCamera)
                  _btn(
                    '${_currentZoomLevel.toStringAsFixed(1)}x',
                    () => _setZoomLevel(
                      _currentZoomLevel < 0.75
                          ? 1.0
                          : _currentZoomLevel < 2.0
                          ? 3.0
                          : 0.5,
                    ),
                  ),
                gap,
                _btn(Icons.layers, () => _toggleSlider(SliderType.numItems)),
                gap,
                _btn(Icons.adjust, () => _toggleSlider(SliderType.confidence)),
                gap,
                _btn('assets/iou.png', () => _toggleSlider(SliderType.iou)),
                SizedBox(height: isLandscape ? 16 : 40),
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
                padding: EdgeInsets.symmetric(
                  horizontal: isLandscape ? 16 : 24,
                  vertical: isLandscape ? 8 : 12,
                ),
                color: Colors.black.withValues(alpha: 0.8),
                child: SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    activeTrackColor: Colors.yellow,
                    inactiveTrackColor: Colors.white.withValues(alpha: 0.3),
                    thumbColor: Colors.yellow,
                    overlayColor: Colors.yellow.withValues(alpha: 0.2),
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
                    if (_isFrontCamera) _currentZoomLevel = 1.0;
                  });
                  _yoloController.switchCamera();
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// Builds a circular button with an icon, image or text
  Widget _btn(dynamic content, VoidCallback onPressed) => CircleAvatar(
    radius: 24,
    backgroundColor: Colors.black.withValues(alpha: 0.2),
    child: content is IconData
        ? IconButton(
            icon: Icon(content, color: Colors.white),
            onPressed: onPressed,
          )
        : content.toString().contains('assets/')
        ? IconButton(
            icon: Image.asset(
              content,
              width: 24,
              height: 24,
              color: Colors.white,
            ),
            onPressed: onPressed,
          )
        : TextButton(
            onPressed: onPressed,
            child: Text(
              content,
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
  );

  /// Toggles the active slider type
  void _toggleSlider(SliderType type) => setState(
    () => _activeSlider = _activeSlider == type ? SliderType.none : type,
  );

  /// Builds a pill-shaped container with text
  ///
  /// [label] is the text to display in the pill
  Widget _pill(String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
    decoration: BoxDecoration(
      color: Colors.black.withValues(alpha: 0.6),
      borderRadius: BorderRadius.circular(24),
    ),
    child: Text(
      label,
      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
    ),
  );

  /// Gets slider configuration based on active type
  double _getSliderValue() => switch (_activeSlider) {
    SliderType.numItems => _numItemsThreshold.toDouble(),
    SliderType.confidence => _confidenceThreshold,
    SliderType.iou => _iouThreshold,
    _ => 0,
  };
  double _getSliderMin() => _activeSlider == SliderType.numItems ? 5 : 0.1;
  double _getSliderMax() => _activeSlider == SliderType.numItems ? 50 : 0.9;
  int _getSliderDivisions() => _activeSlider == SliderType.numItems ? 9 : 8;
  String _getSliderLabel() => switch (_activeSlider) {
    SliderType.numItems => '$_numItemsThreshold',
    SliderType.confidence => _confidenceThreshold.toStringAsFixed(1),
    SliderType.iou => _iouThreshold.toStringAsFixed(1),
    _ => '',
  };

  /// Updates the value of the active slider
  void _updateSliderValue(double value) {
    switch (_activeSlider) {
      case SliderType.numItems:
        _numItemsThreshold = value.toInt();
        _yoloController.setNumItemsThreshold(_numItemsThreshold);
      case SliderType.confidence:
        _confidenceThreshold = value;
        _yoloController.setConfidenceThreshold(value);
      case SliderType.iou:
        _iouThreshold = value;
        _yoloController.setIoUThreshold(value);
      default:
        break;
    }
  }

  /// Sets the camera zoom level
  void _setZoomLevel(double zoomLevel) {
    setState(() => _currentZoomLevel = zoomLevel);
    _yoloController.setZoomLevel(zoomLevel);
  }

  /// Builds the model selector widget
  ///
  /// Creates a row of buttons for selecting different YOLO model types.
  /// Each button shows the model type name and highlights the selected model.
  Widget _buildModelSelector() {
    return Container(
      height: 36,
      padding: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: ModelType.values.map((model) {
          final isSelected = _selectedModel == model;
          return GestureDetector(
            onTap: () {
              if (!_isModelLoading && model != _selectedModel) {
                setState(() {
                  _selectedModel = model;
                });
                _loadModelForPlatform();
              }
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: isSelected ? Colors.white : Colors.transparent,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                model.name.toUpperCase(),
                style: TextStyle(
                  color: isSelected ? Colors.black : Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  void _showError(String title, String message) => showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('OK'),
        ),
      ],
    ),
  );

  Future<void> _loadModelForPlatform() async {
    setState(() {
      _isModelLoading = true;
      _loadingMessage = 'Loading ${_selectedModel.modelName} model...';
      _downloadProgress = 0.0;
      _detectionCount = 0;
      _currentFps = 0.0;
    });

    try {
      final modelPath = await _modelManager.getModelPath(_selectedModel);
      if (mounted) {
        setState(() {
          _modelPath = modelPath;
          _isModelLoading = false;
          _loadingMessage = '';
          _downloadProgress = 0.0;
        });
        if (modelPath == null && mounted) {
          _showError(
            'Model Not Available',
            'Failed to load ${_selectedModel.modelName} model. Please check your internet connection and try again.',
          );
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isModelLoading = false;
          _loadingMessage = 'Failed to load model';
          _downloadProgress = 0.0;
        });
        _showError(
          'Model Loading Error',
          'Failed to load ${_selectedModel.modelName} model: $e',
        );
      }
    }
  }
}
