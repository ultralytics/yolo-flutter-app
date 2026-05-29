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

/// One-import camera UI matching the iOS showcase layout. Composes every widget under `lib/widgets/`, owns gestures
/// (pinch + tap-to-focus), drives the controller, and persists the last task/size across launches.
class YOLOShowcase extends StatefulWidget {
  /// Task to load on first launch (overridden by stored preference).
  final YOLOTask initialTask;

  /// Model size (`n/s/m/l/x`) to load on first launch (overridden by stored preference).
  final String initialModelSize;

  /// When `false`, the Semantic task chip is hidden — useful while semantic models for the current release are still
  /// missing.
  final bool showSemanticTask;

  /// Invoked with a composited JPEG when the user taps the share button.
  final void Function(Uint8List bytes)? onCapture;

  /// Optional controller; one is created internally if `null`.
  final YOLOViewController? controller;

  /// Optional theme override; defaults to dark Material 3.
  final ThemeData? theme;

  /// Optional app version label shown in the bottom-left. Hidden when null. Pass the consuming app's
  /// `package_info_plus` version, e.g. `'v${info.version}'`.
  final String? versionLabel;

  const YOLOShowcase({
    super.key,
    this.initialTask = YOLOTask.detect,
    this.initialModelSize = 'n',
    this.showSemanticTask = true,
    this.onCapture,
    this.controller,
    this.theme,
    this.versionLabel,
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
  // The task/size the native side is actually running (last switch that succeeded). `_current*` is the optimistic
  // selection shown in the controls; on a failed switch we revert `_current*` back to these so the chips never claim a
  // model that isn't loaded.
  late YOLOTask _runningTask;
  late String _runningSize;
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
    _runningTask = _currentTask;
    _runningSize = _currentSize;

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
      // Native-side focus events fire view-relative 0..1 coords; translate to pixels using the most recent
      // LayoutBuilder size tracked in build().
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
      final done = progress.fraction >= 1;
      setState(() {
        if (done) {
          _availableSizes = {..._availableSizes, size};
        }
        // Only drive the active download spinner for the size the user is currently waiting on. A download abandoned
        // by a later size tap keeps streaming until it finishes/fails, but must not keep (or resurrect) the chip for a
        // selection the user has moved off of — `_onSizeChanged`/`_onTaskChanged` already clear the chip on change.
        if (size == _currentSize) {
          _downloadingSize = done ? null : size;
          _downloadFraction = done ? null : progress.fraction;
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
        // Clamp the restored size to whatever the resolver actually publishes for the active platform (Android only
        // hosts the `n` variants at v0.2.0, for example). Otherwise we'd hand `YOLOView` a `_currentModelId` that
        // 404s on first launch before the chip even renders.
        _currentSize = _clampSizeToSupported(_currentSize, _currentTask);
      });
    }
    final initial = await _scanAvailableSizes(_currentTask);
    if (mounted) setState(() => _availableSizes = initial);
    // Defer lens enumeration to after the platform view is initialized.
    unawaited(_refreshLenses());
  }

  /// Returns `size` if it's supported for `task`; otherwise the first supported size, falling back to `'n'` when even
  /// that's missing.
  String _clampSizeToSupported(String size, YOLOTask task) {
    final supported = _supportedSizesForTask(task);
    if (supported.contains(size)) return size;
    if (supported.isEmpty) return 'n';
    for (final s in _allSizes) {
      if (supported.contains(s)) return s;
    }
    return 'n';
  }

  Future<void> _refreshLenses() async {
    // Wait for the platform view (and its native YOLOView) to come up before enumerating lenses. Model download/compile
    // on cold launch can push view creation well past a few hundred ms; cap at ~30s but keep the body cheap so the cost
    // of waiting is negligible.
    const deadline = Duration(seconds: 30);
    final sw = Stopwatch()..start();
    while (sw.elapsed < deadline) {
      if (!mounted) return;
      if (_controller.isInitialized) {
        final lenses = await _controller.getAvailableLenses();
        if (lenses.isNotEmpty) {
          if (mounted) setState(() => _lenses = lenses);
          return;
        }
      }
      await Future<void>.delayed(const Duration(milliseconds: 250));
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

  static String _composeModelId({
    required YOLOTask task,
    required String size,
  }) {
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

  /// Maps `yolo26<size><suffix>` back to its size letter — used to route download-progress events to the matching
  /// chip.
  static String? _sizeForModelId(String id, YOLOTask task) {
    final expectedSuffix = _composeModelId(task: task, size: '').substring(6);
    for (final size in _allSizes) {
      if (id == 'yolo26$size$expectedSuffix') return size;
    }
    return null;
  }

  /// Sizes the resolver can fetch for `task` on the current platform. Drives chip visibility — sizes outside this set
  /// are hidden so the user can't tap a chip that would 404 at download time.
  Set<String> _supportedSizesForTask(YOLOTask task) {
    final declared = YOLO.officialModels(task: task);
    final sizes = <String>{};
    for (final size in _allSizes) {
      if (declared.contains(_composeModelId(task: task, size: size))) {
        sizes.add(size);
      }
    }
    return sizes;
  }

  /// Probes each declared `yolo26<size><suffix>` for the active task to decide which chips are "downloaded" vs need a
  /// `⤓` glyph. Only models the resolver declares (`YOLO.officialModels`) are probed.
  Future<Set<String>> _scanAvailableSizes(YOLOTask task) async {
    final supported = _supportedSizesForTask(task);
    final present = <String>{};
    for (final size in supported) {
      final id = _composeModelId(task: task, size: size);
      final info = await YOLO.checkModelExists(id);
      if (info['exists'] == true) present.add(size);
    }
    return present;
  }

  Future<void> _persistTask(YOLOTask task) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsTaskKey, task.name);
  }

  Future<void> _persistSize(String size) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsSizeKey, size);
  }

  /// `YOLOView` confirms a model is actually loaded (initial load or a successful in-place switch). Record the loaded
  /// model's identity (not the possibly-already-changed optimistic selection) as the "running" baseline, so a later
  /// failed switch reverts to a model that truly loaded.
  void _onModelLoaded(String modelPath, YOLOTask? task) {
    if (!mounted) return;
    final loadedTask = task ?? _currentTask;
    final loadedSize = _sizeForModelId(modelPath, loadedTask);
    if (loadedSize == null) return;
    _runningTask = loadedTask;
    _runningSize = loadedSize;
  }

  /// An in-place model switch failed (`YOLOView` kept the previously loaded model running). Ignore stale failures from
  /// a request the user has already moved off of — the current chip belongs to a newer request. When the failed
  /// request is still the current selection, clear the transient download chip and revert `_current*` back to the
  /// model that's actually running so the controls don't claim a model that never loaded. Reverting re-points
  /// `YOLOView` at the running model, which resolves to the same already-loaded path (no native re-switch).
  void _onModelError(Object error, String modelPath, YOLOTask? task) {
    if (!mounted || modelPath != _currentModelId) return;
    setState(() {
      _downloadingSize = null;
      _downloadFraction = null;
      _currentSize = _runningSize;
      _currentTask = _runningTask;
    });
  }

  void _onTaskChanged(YOLOTask task) {
    if (task == _currentTask) return;
    setState(() {
      _currentTask = task;
      // The supported set differs per task on Android (only `n` for everything in v0.2.0). Clamp here too so a
      // task switch never hands `YOLOView` a model id that doesn't exist on the active platform.
      _currentSize = _clampSizeToSupported(_currentSize, task);
      // Abandon any in-flight download chip for the previous selection (the progress listener only resurrects it for
      // the new current size).
      _downloadingSize = null;
      _downloadFraction = null;
    });
    unawaited(_persistTask(task));
    unawaited(_refreshAvailableSizes(task));
    // The `setState` above changes `YOLOView`'s `modelPath`/`task`, so its `didUpdateWidget` performs the single
    // resolve + native `switchModel`. Don't switch here too — that double-resolves and can race the download.
  }

  Future<void> _refreshAvailableSizes(YOLOTask task) async {
    final sizes = await _scanAvailableSizes(task);
    if (!mounted || task != _currentTask) return;
    setState(() => _availableSizes = sizes);
  }

  void _onSizeChanged(String size) {
    if (size == _currentSize) return;
    setState(() {
      _currentSize = size;
      // Abandon any in-flight download chip for the previous selection (see `_onTaskChanged`).
      _downloadingSize = null;
      _downloadFraction = null;
    });
    unawaited(_persistSize(size));
    // `YOLOView.didUpdateWidget` handles the resolve + native switch off the changed `modelPath` prop (see above).
  }

  void _onLensSelected(LensInfo lens) {
    unawaited(_controller.setLens(lens.zoomFactor));
  }

  void _onPlayPause() {
    setState(() => _isPaused = !_isPaused);
    // iOS `pause` snapshots the next frame into the native share cache before stopping; sharing while paused returns
    // that frame. Android aliases to stop/start. resume() clears the cached frame and restarts.
    if (_isPaused) {
      unawaited(_controller.pause());
    } else {
      unawaited(_controller.resume());
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
            // Cache for the focus-stream listener (registered in initState, fires later — needs a synchronous
            // view-size lookup).
            _viewSize = viewSize;
            return Stack(
              fit: StackFit.expand,
              children: [
                YOLOView(
                  modelPath: _currentModelId,
                  task: _currentTask,
                  controller: _controller,
                  onPerformanceMetrics: _onPerformanceMetrics,
                  onModelError: _onModelError,
                  onModelLoad: _onModelLoaded,
                ),
                // Gesture layer above YOLOView but behind controls so taps on segmented buttons / sliders still reach
                // those widgets first.
                Positioned.fill(
                  child: GestureDetector(
                    behavior: HitTestBehavior.translucent,
                    onScaleStart: _onScaleStart,
                    onScaleUpdate: _onScaleUpdate,
                    onTapDown: (d) => _onTapDown(d, viewSize),
                  ),
                ),
                FocusReticle(position: _focusPosition),
                _ShowcaseOverlay(
                  modelName: 'YOLO26$_currentSize',
                  fps: _fps,
                  inferenceMs: _inferenceMs,
                  task: _currentTask,
                  size: _currentSize,
                  availableSizes: _availableSizes,
                  supportedSizes: _supportedSizesForTask(_currentTask),
                  downloadingSize: _downloadingSize,
                  downloadFraction: _downloadFraction,
                  confidence: _confidence,
                  iou: _iou,
                  zoom: _currentZoom,
                  lensLabel: _currentLensLabel,
                  lenses: _lenses,
                  isPaused: _isPaused,
                  versionLabel: widget.versionLabel,
                  showSemanticTask: widget.showSemanticTask,
                  onTaskChanged: _onTaskChanged,
                  onSizeChanged: _onSizeChanged,
                  onConfidenceChanged: (v) {
                    setState(() => _confidence = v);
                    unawaited(_controller.setConfidenceThreshold(v));
                  },
                  onIouChanged: (v) {
                    setState(() => _iou = v);
                    unawaited(_controller.setIoUThreshold(v));
                  },
                  onLensSelected: _onLensSelected,
                  onPlayPause: _onPlayPause,
                  onSwitchCamera: () => unawaited(_controller.switchCamera()),
                  onShare: () => unawaited(_onShare()),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

/// Stateless overlay sandwich. Layout mirrors `yolo-ios-app/Sources/YOLO/YOLOView.swift#layoutPortrait` (lines 749–798)
/// and the storyboard segmented-control frames in `Main.storyboard`:
///   * Top: 20pt side padding, centered model name (10% view-height), centered FPS line (4%), 8pt gap, task control
///     (32pt), 4pt gap, model-size control (32pt).
///   * Middle: empty so the camera shows through.
///   * Bottom: confidence + IoU sliders (46% view width, 20pt left padding), then zoom indicator, then lens picker,
///     then the 66pt full-width toolbar. The Ultralytics logo + optional version label float to the right and left of
///     the zoom HUD.
class _ShowcaseOverlay extends StatelessWidget {
  const _ShowcaseOverlay({
    required this.modelName,
    required this.fps,
    required this.inferenceMs,
    required this.task,
    required this.size,
    required this.availableSizes,
    required this.supportedSizes,
    required this.downloadingSize,
    required this.downloadFraction,
    required this.confidence,
    required this.iou,
    required this.zoom,
    required this.lensLabel,
    required this.lenses,
    required this.isPaused,
    required this.versionLabel,
    required this.showSemanticTask,
    required this.onTaskChanged,
    required this.onSizeChanged,
    required this.onConfidenceChanged,
    required this.onIouChanged,
    required this.onLensSelected,
    required this.onPlayPause,
    required this.onSwitchCamera,
    required this.onShare,
  });

  final String modelName;
  final double fps;
  final double inferenceMs;
  final YOLOTask task;
  final String size;
  final Set<String> availableSizes;
  final Set<String> supportedSizes;
  final String? downloadingSize;
  final double? downloadFraction;
  final double confidence;
  final double iou;
  final double zoom;
  final String lensLabel;
  final List<LensInfo> lenses;
  final bool isPaused;
  final String? versionLabel;
  final bool showSemanticTask;
  final ValueChanged<YOLOTask> onTaskChanged;
  final ValueChanged<String> onSizeChanged;
  final ValueChanged<double> onConfidenceChanged;
  final ValueChanged<double> onIouChanged;
  final ValueChanged<LensInfo> onLensSelected;
  final VoidCallback onPlayPause;
  final VoidCallback onSwitchCamera;
  final VoidCallback onShare;

  // iOS YOLOView ports — kept as constants so the layout reads like the Swift source.
  static const double _sidePadding = 20;
  static const double _topGap = 8;
  static const double _sliderRowGap = 14;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      // Bottom toolbar extends to the device edge; opt the toolbar out of the bottom safe-area pad so it sits flush.
      bottom: false,
      child: Column(
        children: [
          // -- Top stack ----------------------------------------------------------------------------------------
          Padding(
            padding: const EdgeInsets.fromLTRB(
              _sidePadding,
              8,
              _sidePadding,
              0,
            ),
            child: Column(
              children: [
                PerformanceLabel(
                  modelName: modelName,
                  fps: fps,
                  inferenceMs: inferenceMs,
                ),
                const SizedBox(height: _topGap),
                TaskSegmentedControl(
                  currentTask: task,
                  onTaskChanged: onTaskChanged,
                  showSemanticTask: showSemanticTask,
                ),
                const SizedBox(height: 4),
                ModelSizeSegmentedControl(
                  currentSize: size,
                  availableSizes: availableSizes,
                  supportedSizes: supportedSizes,
                  onSizeChanged: onSizeChanged,
                  downloadingSize: downloadingSize,
                  downloadFraction: downloadFraction,
                ),
              ],
            ),
          ),

          // -- Free camera area --------------------------------------------------------------------------------
          const Spacer(),

          // -- Sliders ------------------------------------------------------------------------------------------
          Padding(
            padding: const EdgeInsets.fromLTRB(
              _sidePadding,
              0,
              _sidePadding,
              0,
            ),
            child: Column(
              children: [
                _SliderConstrained(
                  child: ThresholdSliderRow(
                    label: 'Confidence Threshold',
                    value: confidence,
                    min: 0,
                    max: 1,
                    onChanged: onConfidenceChanged,
                  ),
                ),
                const SizedBox(height: _sliderRowGap),
                _SliderConstrained(
                  child: ThresholdSliderRow(
                    label: 'IoU Threshold',
                    value: iou,
                    min: 0,
                    max: 1,
                    onChanged: onIouChanged,
                  ),
                ),
              ],
            ),
          ),

          // -- Zoom HUD + Logo ---------------------------------------------------------------------------------
          Padding(
            padding: const EdgeInsets.fromLTRB(
              _sidePadding,
              8,
              _sidePadding,
              0,
            ),
            child: Stack(
              alignment: Alignment.center,
              children: [
                ZoomIndicator(currentZoom: zoom, lensLabel: lensLabel),
                const Align(
                  alignment: Alignment.centerRight,
                  child: LogoOverlay(),
                ),
              ],
            ),
          ),

          // -- Lens picker -------------------------------------------------------------------------------------
          Padding(
            padding: const EdgeInsets.fromLTRB(
              _sidePadding,
              6,
              _sidePadding,
              6,
            ),
            child: LensPicker(
              lenses: lenses,
              currentZoomFactor: zoom,
              onLensSelected: onLensSelected,
            ),
          ),

          // -- Version + toolbar (full-bleed) ------------------------------------------------------------------
          if (versionLabel != null)
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: _sidePadding,
                vertical: 4,
              ),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  versionLabel!,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.7),
                    fontSize: 12,
                  ),
                ),
              ),
            ),
          CameraToolbar(
            isPaused: isPaused,
            onPlayPause: onPlayPause,
            onSwitchCamera: onSwitchCamera,
            onShare: onShare,
          ),
          // Extend the same translucent black band under the home-indicator inset so the toolbar reads as a single
          // flush bottom bar (matches `toolbar.frame = ... height - 66, width: width, height: 66` in
          // `yolo-ios-app/Sources/YOLO/YOLOView.swift:806`).
          Container(
            height: MediaQuery.of(context).padding.bottom,
            color: Colors.black.withValues(alpha: 0.7),
          ),
        ],
      ),
    );
  }
}

/// Constrains the iOS-style sliders to 46% of the view width on the left side, mirroring
/// `layoutPortrait` line 763 (`sliderWidth = width * 0.46`).
class _SliderConstrained extends StatelessWidget {
  const _SliderConstrained({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: FractionallySizedBox(widthFactor: 0.46, child: child),
    );
  }
}
