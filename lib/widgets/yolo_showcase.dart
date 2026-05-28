// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/widgets/camera_toolbar.dart';
import 'package:ultralytics_yolo/widgets/focus_reticle.dart';
import 'package:ultralytics_yolo/widgets/lens_picker.dart';
import 'package:ultralytics_yolo/widgets/logo_overlay.dart';
import 'package:ultralytics_yolo/widgets/model_size_segmented_control.dart';
import 'package:ultralytics_yolo/widgets/performance_label.dart';
import 'package:ultralytics_yolo/widgets/task_segmented_control.dart';
import 'package:ultralytics_yolo/widgets/threshold_slider_row.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/widgets/zoom_indicator.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

/// One-import camera UI matching the iOS showcase layout. Composes every
/// widget under `lib/widgets/`, owns gestures (pinch + tap-to-focus), drives
/// the controller, and persists the last task/size across launches.
class YOLOShowcase extends StatefulWidget {
  /// Task to load on first launch (overridden by stored preference).
  final YOLOTask initialTask;

  /// Model size (`n/s/m/l/x`) to load on first launch (overridden by
  /// stored preference).
  final String initialModelSize;

  /// When `false`, the Semantic task chip is hidden — useful while semantic
  /// models for the current release are still missing.
  final bool showSemanticTask;

  /// Invoked with a composited JPEG when the user taps the share button.
  final void Function(Uint8List bytes)? onCapture;

  /// Optional controller; one is created internally if `null`.
  final YOLOViewController? controller;

  /// Optional theme override; defaults to dark Material 3.
  final ThemeData? theme;

  const YOLOShowcase({
    super.key,
    this.initialTask = YOLOTask.detect,
    this.initialModelSize = 'n',
    this.showSemanticTask = true,
    this.onCapture,
    this.controller,
    this.theme,
  });

  @override
  State<YOLOShowcase> createState() => _YOLOShowcaseState();
}

class _YOLOShowcaseState extends State<YOLOShowcase> {
  static const _prefsTaskKey = 'ultralytics_yolo.showcase.task';
  static const _prefsSizeKey = 'ultralytics_yolo.showcase.size';

  late YOLOViewController _controller;
  bool _ownsController = false;

  late YOLOTask _currentTask;
  late String _currentSize;
  Set<String> _availableSizes = {};
  String? _downloadingSize;
  double? _downloadFraction;

  List<LensInfo> _lenses = const [];
  double _currentZoom = 1;
  String _currentLensLabel = '';

  double _confidence = 0.25;
  double _iou = 0.7;

  double _fps = 0;
  double _inferenceMs = 0;

  bool _isPaused = false;
  Offset? _focusPosition;
  double _baseScale = 1;
  Size _viewSize = Size.zero;

  StreamSubscription<double>? _zoomSub;
  StreamSubscription<String>? _lensSub;
  StreamSubscription<Offset>? _focusSub;
  StreamSubscription<DownloadProgress>? _progressSub;

  @override
  void initState() {
    super.initState();
    _currentTask = widget.initialTask;
    _currentSize = widget.initialModelSize;

    if (widget.controller != null) {
      _controller = widget.controller!;
    } else {
      _controller = YOLOViewController();
      _ownsController = true;
    }

    _zoomSub = _controller.zoomEvents.listen((z) {
      if (mounted) setState(() => _currentZoom = z);
    });
    _lensSub = _controller.lensEvents.listen((label) {
      if (mounted) setState(() => _currentLensLabel = label);
    });
    _focusSub = _controller.focusEvents.listen((offset) {
      // Native-side focus events fire view-relative 0..1 coords; translate to
      // pixels using the most recent LayoutBuilder size tracked in build().
      if (!mounted || _viewSize == Size.zero) return;
      setState(() {
        _focusPosition = Offset(
          offset.dx * _viewSize.width,
          offset.dy * _viewSize.height,
        );
      });
    });
    _progressSub = YOLOModelManager.downloadProgress.listen((progress) {
      // Match the model id to a size chip; ignore foreign downloads.
      final size = _sizeForModelId(progress.modelId, _currentTask);
      if (size == null || !mounted) return;
      setState(() {
        _downloadingSize = progress.fraction >= 1 ? null : size;
        _downloadFraction = progress.fraction >= 1 ? null : progress.fraction;
        if (progress.fraction >= 1) {
          _availableSizes = {..._availableSizes, size};
        }
      });
    });

    WakelockPlus.enable();
    unawaited(_bootstrap());
  }

  Future<void> _bootstrap() async {
    final prefs = await SharedPreferences.getInstance();
    final storedTask = prefs.getString(_prefsTaskKey);
    final storedSize = prefs.getString(_prefsSizeKey);
    if (mounted) {
      setState(() {
        if (storedTask != null) {
          final parsed = YOLOTaskParsing.tryParse(storedTask);
          if (parsed != null) _currentTask = parsed;
        }
        if (storedSize != null && _allSizes.contains(storedSize)) {
          _currentSize = storedSize;
        }
        _availableSizes = _scanAvailableSizes(_currentTask);
      });
    }
    // Defer lens enumeration to after the platform view is initialized.
    unawaited(_refreshLenses());
  }

  Future<void> _refreshLenses() async {
    // The native side may not be ready instantly on first frame; one retry
    // covers the common race without an exponential loop.
    for (var attempt = 0; attempt < 2; attempt++) {
      final lenses = await _controller.getAvailableLenses();
      if (lenses.isNotEmpty) {
        if (mounted) setState(() => _lenses = lenses);
        return;
      }
      await Future<void>.delayed(const Duration(milliseconds: 300));
    }
  }

  @override
  void dispose() {
    _zoomSub?.cancel();
    _lensSub?.cancel();
    _focusSub?.cancel();
    _progressSub?.cancel();
    WakelockPlus.disable();
    if (_ownsController) _controller.dispose();
    super.dispose();
  }

  static const List<String> _allSizes = ['n', 's', 'm', 'l', 'x'];

  String get _currentModelId =>
      _composeModelId(task: _currentTask, size: _currentSize);

  static String _composeModelId({required YOLOTask task, required String size}) {
    const suffixes = {
      YOLOTask.detect: '',
      YOLOTask.segment: '-seg',
      YOLOTask.semantic: '-sem',
      YOLOTask.classify: '-cls',
      YOLOTask.pose: '-pose',
      YOLOTask.obb: '-obb',
    };
    return 'yolo26$size${suffixes[task] ?? ''}';
  }

  /// Maps `yolo26<size><suffix>` back to its size letter — used to route
  /// download-progress events to the matching chip.
  static String? _sizeForModelId(String id, YOLOTask task) {
    final expectedSuffix = _composeModelId(task: task, size: '').substring(6);
    for (final size in _allSizes) {
      if (id == 'yolo26$size$expectedSuffix') return size;
    }
    return null;
  }

  /// Filters `YOLO.officialModels(task: ...)` (which already platform-filters
  /// by tflite vs mlpackage availability) down to a set of size letters.
  Set<String> _scanAvailableSizes(YOLOTask task) {
    final ids = YOLO.officialModels(task: task);
    final set = <String>{};
    for (final id in ids) {
      final size = _sizeForModelId(id, task);
      if (size != null) set.add(size);
    }
    return set;
  }

  Future<void> _persistTask(YOLOTask task) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsTaskKey, task.name);
  }

  Future<void> _persistSize(String size) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsSizeKey, size);
  }

  Future<void> _switchToCurrentModel() async {
    await _controller.switchModel(_currentModelId, _currentTask);
  }

  void _onTaskChanged(YOLOTask task) {
    if (task == _currentTask) return;
    setState(() {
      _currentTask = task;
      _availableSizes = _scanAvailableSizes(task);
    });
    unawaited(_persistTask(task));
    unawaited(_switchToCurrentModel());
  }

  void _onSizeChanged(String size) {
    if (size == _currentSize) return;
    setState(() => _currentSize = size);
    unawaited(_persistSize(size));
    unawaited(_switchToCurrentModel());
  }

  void _onLensSelected(LensInfo lens) {
    unawaited(_controller.setLens(lens.zoomFactor));
  }

  void _onPlayPause() {
    setState(() => _isPaused = !_isPaused);
    // Native pause is implemented by stopping inference; resume restarts.
    if (_isPaused) {
      unawaited(_controller.stop());
    } else {
      unawaited(_controller.restartCamera());
    }
  }

  Future<void> _onShare() async {
    final bytes = await _controller.capturePhoto(withOverlays: true);
    if (bytes != null) widget.onCapture?.call(bytes);
  }

  void _onScaleStart(ScaleStartDetails _) {
    _baseScale = _currentZoom;
  }

  void _onScaleUpdate(ScaleUpdateDetails details) {
    if (details.pointerCount < 2) return; // tap → onTapDown handles it
    final target = (_baseScale * details.scale).clamp(0.5, 8.0);
    unawaited(_controller.setZoomLevel(target));
  }

  void _onTapDown(TapDownDetails details, Size viewSize) {
    final local = details.localPosition;
    final nx = (local.dx / viewSize.width).clamp(0.0, 1.0);
    final ny = (local.dy / viewSize.height).clamp(0.0, 1.0);
    setState(() => _focusPosition = local);
    unawaited(_controller.tapToFocus(nx, ny));
  }

  void _onPerformanceMetrics(YOLOPerformanceMetrics metrics) {
    if (!mounted) return;
    setState(() {
      _fps = metrics.fps;
      _inferenceMs = metrics.processingTimeMs;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = widget.theme ?? ThemeData.dark(useMaterial3: true);
    return Theme(
      data: theme,
      child: Material(
        color: Colors.black,
        child: LayoutBuilder(
          builder: (context, constraints) {
            final viewSize = Size(constraints.maxWidth, constraints.maxHeight);
            // Cache for the focus-stream listener (registered in initState,
            // fires later — needs a synchronous view-size lookup).
            _viewSize = viewSize;
            return Stack(
              fit: StackFit.expand,
              children: [
                YOLOView(
                  modelPath: _currentModelId,
                  task: _currentTask,
                  controller: _controller,
                  showOverlays: false,
                  showNativeUI: false,
                  onPerformanceMetrics: _onPerformanceMetrics,
                ),
                // Gesture layer above YOLOView but behind controls so taps on
                // segmented buttons / sliders still reach those widgets first.
                Positioned.fill(
                  child: GestureDetector(
                    behavior: HitTestBehavior.translucent,
                    onScaleStart: _onScaleStart,
                    onScaleUpdate: _onScaleUpdate,
                    onTapDown: (d) => _onTapDown(d, viewSize),
                  ),
                ),
                FocusReticle(position: _focusPosition),
                SafeArea(child: _buildOverlay(context)),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildOverlay(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Column(
        children: [
          Align(
            alignment: Alignment.centerLeft,
            child: PerformanceLabel(
              modelName: 'YOLO26${_currentSize.toUpperCase()}',
              fps: _fps,
              inferenceMs: _inferenceMs,
            ),
          ),
          const SizedBox(height: 8),
          TaskSegmentedControl(
            currentTask: _currentTask,
            onTaskChanged: _onTaskChanged,
            showSemanticTask: widget.showSemanticTask,
          ),
          const SizedBox(height: 8),
          ModelSizeSegmentedControl(
            currentSize: _currentSize,
            availableSizes: _availableSizes,
            onSizeChanged: _onSizeChanged,
            downloadingSize: _downloadingSize,
            downloadFraction: _downloadFraction,
          ),
          const Spacer(),
          ThresholdSliderRow(
            label: 'Confidence Threshold',
            value: _confidence,
            min: 0,
            max: 1,
            onChanged: (v) {
              setState(() => _confidence = v);
              unawaited(_controller.setConfidenceThreshold(v));
            },
          ),
          ThresholdSliderRow(
            label: 'IoU Threshold',
            value: _iou,
            min: 0,
            max: 1,
            onChanged: (v) {
              setState(() => _iou = v);
              unawaited(_controller.setIoUThreshold(v));
            },
          ),
          const Align(
            alignment: Alignment.bottomRight,
            child: Padding(
              padding: EdgeInsets.only(bottom: 8),
              child: LogoOverlay(),
            ),
          ),
          ZoomIndicator(
            currentZoom: _currentZoom,
            lensLabel: _currentLensLabel,
          ),
          const SizedBox(height: 4),
          LensPicker(
            lenses: _lenses,
            currentZoomFactor: _currentZoom,
            onLensSelected: _onLensSelected,
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'v0.3.5',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.white.withValues(alpha: 0.7),
                    ),
              ),
              CameraToolbar(
                isPaused: _isPaused,
                onPlayPause: _onPlayPause,
                onSwitchCamera: () => unawaited(_controller.switchCamera()),
                onShare: () => unawaited(_onShare()),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
