// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import '../../models/models.dart';
import '../../services/model_manager.dart';

/// Controller that manages the state and business logic for camera inference
class CameraInferenceController extends ChangeNotifier {
  // Detection state
  int _detectionCount = 0;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  // Threshold state
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  SliderType _activeSlider = SliderType.none;

  // Model state
  ModelType _selectedModel = ModelType.detect;
  bool _isModelLoading = false;
  String? _modelPath;
  String _loadingMessage = '';
  double _downloadProgress = 0.0;

  // Camera state
  double _currentZoomLevel = 1.0;
  bool _isFrontCamera = false;

  // Controllers
  final _yoloController = YOLOViewController();
  late final ModelManager _modelManager;

  // Performance optimization
  bool _isDisposed = false;
  Future<void>? _loadingFuture;

  // Getters
  int get detectionCount => _detectionCount;
  double get currentFps => _currentFps;
  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  SliderType get activeSlider => _activeSlider;
  ModelType get selectedModel => _selectedModel;
  bool get isModelLoading => _isModelLoading;
  String? get modelPath => _modelPath;
  String get loadingMessage => _loadingMessage;
  double get downloadProgress => _downloadProgress;
  double get currentZoomLevel => _currentZoomLevel;
  bool get isFrontCamera => _isFrontCamera;
  YOLOViewController get yoloController => _yoloController;

  CameraInferenceController() {
    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        _downloadProgress = progress;
        notifyListeners();
      },
      onStatusUpdate: (message) {
        _loadingMessage = message;
        notifyListeners();
      },
    );
  }

  /// Initialize the controller
  Future<void> initialize() async {
    await _loadModelForPlatform();
    _yoloController.setThresholds(
      confidenceThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
      numItemsThreshold: _numItemsThreshold,
    );
  }

  /// Handle detection results and calculate FPS
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

  /// Handle performance metrics
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

    if (_activeSlider != type) {
      _activeSlider = _activeSlider == type ? SliderType.none : type;
      notifyListeners();
    }
  }

  void updateSliderValue(double value) {
    if (_isDisposed) return;

    bool changed = false;
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
      default:
        break;
    }

    if (changed) {
      notifyListeners();
    }
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
    if (_isFrontCamera) _currentZoomLevel = 1.0;
    _yoloController.switchCamera();
    notifyListeners();
  }

  void changeModel(ModelType model) {
    if (_isDisposed) return;

    if (!_isModelLoading && model != _selectedModel) {
      _selectedModel = model;
      _loadModelForPlatform();
    }
  }

  Future<void> _loadModelForPlatform() async {
    if (_isDisposed) return;

    if (_loadingFuture != null) {
      await _loadingFuture;
      return;
    }

    _loadingFuture = _performModelLoading();
    try {
      await _loadingFuture;
    } finally {
      _loadingFuture = null;
    }
  }

  Future<void> _performModelLoading() async {
    if (_isDisposed) return;

    _isModelLoading = true;
    _loadingMessage = 'Loading ${_selectedModel.modelName} model...';
    _downloadProgress = 0.0;
    _detectionCount = 0;
    _currentFps = 0.0;
    notifyListeners();

    try {
      final modelPath = await _modelManager.getModelPath(_selectedModel);

      if (_isDisposed) return;

      _modelPath = modelPath;
      _isModelLoading = false;
      _loadingMessage = '';
      _downloadProgress = 0.0;
      notifyListeners();

      if (modelPath == null) {
        throw Exception('Failed to load ${_selectedModel.modelName} model');
      }
    } catch (e) {
      if (_isDisposed) return;

      final error = YOLOErrorHandler.handleError(
        e,
        'Failed to load model ${_selectedModel.modelName} for task ${_selectedModel.task.name}',
      );

      _isModelLoading = false;
      _loadingMessage = 'Failed to load model: ${error.message}';
      _downloadProgress = 0.0;
      notifyListeners();
      rethrow;
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    super.dispose();
  }
}
