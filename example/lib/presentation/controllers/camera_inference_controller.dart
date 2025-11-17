// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_model_spec.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';

import '../../models/models.dart';
import '../../services/model_manager.dart';

/// Controller that manages state and business logic for camera inference:
/// - Computes and loads model paths per ModelType (platform-aware)
/// - Supports multiple active models simultaneously
/// - Exposes modelsForView: List<YOLOModelSpec> for YOLOView
class CameraInferenceController extends ChangeNotifier {
  // -------------------------
  // Detection / Performance state
  // -------------------------
  int _detectionCount = 0;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  // -------------------------
  // Threshold state
  // -------------------------
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  SliderType _activeSlider = SliderType.none;

  // -------------------------
  // Model state
  // -------------------------
  // Active models selection (multi-select)
  final Set<ModelType> _activeModels = {
    // Default to a useful multi-model demo (segment + pose)
    ModelType.segment,
  };

  // Loaded model paths per model type (computed via ModelManager)
  final Map<ModelType, String> _modelPaths = {};

  // Loading state
  bool _isModelLoading = false;
  String _loadingMessage = '';
  double _downloadProgress = 0.0;

  // -------------------------
  // Camera state
  // -------------------------
  double _currentZoomLevel = 1.0;
  bool _isFrontCamera = false;

  // -------------------------
  // Controllers and helpers
  // -------------------------
  final YOLOViewController _yoloController = YOLOViewController();
  late final ModelManager _modelManager;

  // Avoid re-entrancy during async loads
  Future<void>? _loadingFuture;
  bool _isDisposed = false;

  // -------------------------
  // Getters
  // -------------------------
  int get detectionCount => _detectionCount;
  double get currentFps => _currentFps;

  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  SliderType get activeSlider => _activeSlider;

  Set<ModelType> get activeModels => Set.unmodifiable(_activeModels);

  bool get isModelLoading => _isModelLoading;
  String get loadingMessage => _loadingMessage;
  double get downloadProgress => _downloadProgress;

  double get currentZoomLevel => _currentZoomLevel;
  bool get isFrontCamera => _isFrontCamera;

  YOLOViewController get yoloController => _yoloController;

  /// Expose currently loaded and active models as YOLOModelSpec list for YOLOView.
  /// Only includes models that have successfully resolved modelPath.
  List<YOLOModelSpec> get modelsForView {
    final specs = <YOLOModelSpec>[];
    for (final model in _activeModels) {
      final path = _modelPaths[model];
      if (path != null && path.isNotEmpty) {
        specs.add(YOLOModelSpec(modelPath: path, task: model.task));
      }
    }
    return specs;
  }

  CameraInferenceController() {
    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        _downloadProgress = progress;
        if (!_isDisposed) notifyListeners();
      },
      onStatusUpdate: (message) {
        _loadingMessage = message;
        if (!_isDisposed) notifyListeners();
      },
    );
  }

  // -------------------------
  // Lifecycle
  // -------------------------
  Future<void> initialize() async {
    await _loadModelsForActiveSelection();
    _yoloController.setThresholds(
      confidenceThreshold: _confidenceThreshold,
      iouThreshold: _iouThreshold,
      numItemsThreshold: _numItemsThreshold,
    );
  }

  @override
  void dispose() {
    _isDisposed = true;
    super.dispose();
  }

  // -------------------------
  // Detection callbacks
  // -------------------------
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

  // -------------------------
  // UI interactions
  // -------------------------
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

  /// Replace active models with a new selection and (re)load their paths.
  Future<void> setActiveModels(Iterable<ModelType> models) async {
    if (_isDisposed) return;

    final newSet = Set<ModelType>.from(models);
    if (_activeModels.containsAll(newSet) &&
        newSet.containsAll(_activeModels)) {
      // No change
      return;
    }

    _activeModels
      ..clear()
      ..addAll(newSet);

    // Reset paths for models that are no longer active
    _modelPaths.removeWhere((key, _) => !_activeModels.contains(key));

    notifyListeners();
    await _loadModelsForActiveSelection();
  }

  /// Toggle a single model in/out of the active set.
  /// If [enabled] is provided, forces the new state; otherwise toggles.
  Future<void> toggleActiveModel(ModelType model, {bool? enabled}) async {
    if (_isDisposed) return;

    final shouldEnable = enabled ?? !_activeModels.contains(model);

    if (shouldEnable) {
      final changed = _activeModels.add(model);
      if (changed) {
        notifyListeners();
        await _loadModelsForActiveSelection(targets: {model});
      }
    } else {
      final changed = _activeModels.remove(model);
      if (changed) {
        // Remove path to prevent stale entries
        _modelPaths.remove(model);
        notifyListeners();
      }
    }
  }

  /// Legacy-style "single model" switcher: activates only the provided model.
  Future<void> changeModel(ModelType model) async {
    await setActiveModels({model});
  }

  // -------------------------
  // Model loading
  // -------------------------
  /// Load model paths for the currently active selection.
  /// If [targets] provided, loads only those (useful for incremental toggles).
  Future<void> _loadModelsForActiveSelection({Set<ModelType>? targets}) async {
    if (_isDisposed) return;

    // Prevent overlapping loads
    if (_loadingFuture != null) {
      await _loadingFuture;
      return;
    }

    _loadingFuture = _performModelLoading(targets: targets);
    try {
      await _loadingFuture;
    } finally {
      _loadingFuture = null;
    }
  }

  Future<void> _performModelLoading({Set<ModelType>? targets}) async {
    if (_isDisposed) return;

    _isModelLoading = true;
    _loadingMessage = 'Preparing models...';
    _downloadProgress = 0.0;
    _detectionCount = 0;
    _currentFps = 0.0;
    notifyListeners();

    final toLoad = (targets == null || targets.isEmpty)
        ? _activeModels
        : _activeModels.where((m) => targets.contains(m)).toSet();

    try {
      // Load each model sequentially to reuse progress UI clearly
      for (final modelType in toLoad) {
        if (_isDisposed) return;

        _loadingMessage = 'Loading ${modelType.modelName} model...';
        _downloadProgress = 0.0;
        notifyListeners();

        final modelPath = await _modelManager.getModelPath(modelType);

        if (_isDisposed) return;

        if (modelPath == null || modelPath.isEmpty) {
          throw Exception('Failed to resolve path for ${modelType.modelName}');
        }
        _modelPaths[modelType] = modelPath;
        // Emit after each model for responsive UI
        notifyListeners();
      }

      _isModelLoading = false;
      _loadingMessage = '';
      _downloadProgress = 0.0;
      notifyListeners();
    } catch (e) {
      if (_isDisposed) return;

      final error = YOLOErrorHandler.handleError(e, 'Failed to load models');
      _isModelLoading = false;
      _loadingMessage = 'Failed to load model(s): ${error.message}';
      _downloadProgress = 0.0;
      notifyListeners();
      rethrow;
    }
  }
}
