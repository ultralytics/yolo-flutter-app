// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

enum SliderType { none, numItems, confidence, iou }

/// Controller that manages the state and business logic for camera inference.
class CameraInferenceController extends ChangeNotifier {
  int _detectionCount = 0;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  double _confidenceThreshold = 0.25;
  double _iouThreshold = 0.7;
  int _numItemsThreshold = 30;
  SliderType _activeSlider = SliderType.none;

  YOLOTask _selectedTask = YOLOTask.detect;
  String _selectedModel = _defaultModelForTask(YOLOTask.detect);

  double _currentZoomLevel = 1.0;
  LensFacing _lensFacing = LensFacing.front;
  bool _isFrontCamera = false;

  final _yoloController = YOLOViewController();
  bool _isDisposed = false;

  int get detectionCount => _detectionCount;
  double get currentFps => _currentFps;
  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  SliderType get activeSlider => _activeSlider;
  YOLOTask get selectedTask => _selectedTask;
  String get selectedModel => _selectedModel;
  List<YOLOTask> get availableTasks => YOLOTask.values
      .where((task) => YOLO.officialModels(task: task).isNotEmpty)
      .toList(growable: false);
  List<String> get availableModels => YOLO.officialModels(task: _selectedTask);
  String get modelPath => _selectedModel;
  double get currentZoomLevel => _currentZoomLevel;
  bool get isFrontCamera => _isFrontCamera;
  LensFacing get lensFacing => _lensFacing;
  YOLOViewController get yoloController => _yoloController;

  static String _defaultModelForTask(YOLOTask task) {
    final defaultModel = YOLO.defaultOfficialModel(task: task);
    if (defaultModel != null) return defaultModel;
    final models = YOLO.officialModels(task: task);
    return models.isEmpty ? '' : models.first;
  }

  Future<void> initialize() async {
    _isFrontCamera = _lensFacing == LensFacing.front;
    await _yoloController.setThresholds(
      confidenceThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
      numItemsThreshold: _numItemsThreshold,
    );
  }

  void onDetectionResults(List<YOLOResult> results) {
    if (_isDisposed) return;

    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;

    if (elapsed >= 1000) {
      _currentFps = _frameCount * 1000 / elapsed;
      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    if (_detectionCount != results.length) {
      _detectionCount = results.length;
      notifyListeners();
    }
  }

  void onPerformanceMetrics(double fps) {
    if (_isDisposed) return;
    if ((_currentFps - fps).abs() > 0.1) {
      _currentFps = fps;
      notifyListeners();
    }
  }

  void onZoomChanged(double zoomLevel) {
    if (_isDisposed) return;
    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      notifyListeners();
    }
  }

  void toggleSlider(SliderType type) {
    if (_isDisposed) return;
    _activeSlider = _activeSlider == type ? SliderType.none : type;
    notifyListeners();
  }

  void updateSliderValue(double value) {
    if (_isDisposed) return;

    var changed = false;
    switch (_activeSlider) {
      case SliderType.numItems:
        final newValue = value.toInt();
        if (_numItemsThreshold != newValue) {
          _numItemsThreshold = newValue;
          _yoloController.setNumItemsThreshold(_numItemsThreshold);
          changed = true;
        }
        break;
      case SliderType.confidence:
        if ((_confidenceThreshold - value).abs() > 0.01) {
          _confidenceThreshold = value;
          _yoloController.setConfidenceThreshold(value);
          changed = true;
        }
        break;
      case SliderType.iou:
        if ((_iouThreshold - value).abs() > 0.01) {
          _iouThreshold = value;
          _yoloController.setIoUThreshold(value);
          changed = true;
        }
        break;
      case SliderType.none:
        break;
    }

    if (changed) notifyListeners();
  }

  void setZoomLevel(double zoomLevel) {
    if (_isDisposed) return;
    if ((_currentZoomLevel - zoomLevel).abs() > 0.01) {
      _currentZoomLevel = zoomLevel;
      _yoloController.setZoomLevel(zoomLevel);
      notifyListeners();
    }
  }

  void flipCamera() {
    if (_isDisposed) return;
    _isFrontCamera = !_isFrontCamera;
    _lensFacing = _isFrontCamera ? LensFacing.front : LensFacing.back;
    if (_isFrontCamera) _currentZoomLevel = 1.0;
    _yoloController.switchCamera();
    notifyListeners();
  }

  void changeTask(YOLOTask task) {
    if (_isDisposed ||
        _selectedTask == task ||
        !availableTasks.contains(task)) {
      return;
    }
    _selectedTask = task;
    _selectedModel = _defaultModelForTask(task);
    _detectionCount = 0;
    _currentFps = 0.0;
    notifyListeners();
  }

  void changeModel(String modelId) {
    if (_isDisposed || _selectedModel == modelId) return;
    _selectedModel = modelId;
    _detectionCount = 0;
    _currentFps = 0.0;
    notifyListeners();
  }

  @override
  void dispose() {
    _isDisposed = true;
    _yoloController.stop();
    super.dispose();
  }
}
