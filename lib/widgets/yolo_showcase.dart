// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/core/yolo_model_resolver.dart';
import 'package:ultralytics_yolo/widgets/camera_toolbar.dart';
import 'package:ultralytics_yolo/widgets/focus_reticle.dart';
import 'package:ultralytics_yolo/widgets/lens_picker.dart';
import 'package:ultralytics_yolo/widgets/logo_overlay.dart';
import 'package:ultralytics_yolo/widgets/model_loading_status.dart';
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

  /// When `false`, the Semantic task chip is hidden for hosts that do not want to expose semantic segmentation.
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

  /// Invoked once when the camera is up and the first inference result has arrived — i.e. the live view is fully
  /// ready. Hosts can use this to dismiss a native splash (e.g. `FlutterNativeSplash.remove()`) so the splash covers
  /// the model-compile + camera-bind gap instead of showing a black screen with controls over it.
  final VoidCallback? onReady;

  const YOLOShowcase({
    super.key,
    this.initialTask = YOLOTask.detect,
    this.initialModelSize = 'n',
    this.showSemanticTask = true,
    this.onCapture,
    this.controller,
    this.theme,
    this.versionLabel,
    this.onReady,
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
  String? _loadingStatusText;
  String? _modelErrorMessage;
  int _selectionRequestId = 0;

  List<LensInfo> _lenses = const [];

  // High-frequency values driven by native event streams (zoom/lens) and per-inference-frame metrics. These are
  // ValueNotifiers — NOT setState fields — so updating them rebuilds only the small leaf widgets that display them
  // (FPS/ms line, zoom HUD, lens highlight) instead of the whole tree (camera platform view + every control) ~30x/sec.
  final ValueNotifier<({double fps, double ms})> _metrics = ValueNotifier((
    fps: 0,
    ms: 0,
  ));
  final ValueNotifier<double> _zoom = ValueNotifier(1);
  final ValueNotifier<String> _lensLabel = ValueNotifier('');

  double _confidence = 0.25;
  double _iou = 0.7;

  // True while a model is downloading/loading after a size or task tap, so the UI can show a clear loading overlay
  // instead of looking frozen.
  bool _isModelLoading = false;

  // False until the very first model finishes loading. Until then an opaque splash covers the camera so the user sees
  // a seamless splash -> camera+predictions transition, instead of camera -> black (during the first GPU compile) ->
  // camera. Stays true afterwards (later switches use the translucent veil instead).
  bool _initialModelLoaded = false;

  // Fires onReady exactly once, on the first inference result.
  bool _readyFired = false;

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

    // Drive the notifiers directly (no setState) — these fire continuously while pinching, so a full rebuild here is
    // exactly what made the UI feel laggy.
    _zoomSub = _controller.zoomEvents.listen((z) => _zoom.value = z);
    _lensSub = _controller.lensEvents.listen(
      (label) => _lensLabel.value = label,
    );
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
          _loadingStatusText = done
              ? 'Loading ${_displayModelNameFor(_currentTask, size)}'
              : 'Downloading ${(progress.fraction * 100).clamp(0, 99).round()}%';
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
        // Clamp the restored size to whatever the resolver actually publishes for the active platform. Otherwise we'd
        // hand `YOLOView` a `_currentModelId` that 404s on first launch before the chip even renders.
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
    _metrics.dispose();
    _zoom.dispose();
    _lensLabel.dispose();
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

  static String _displayModelNameFor(YOLOTask task, String size) {
    final id = _composeModelId(task: task, size: size);
    return id.replaceFirst('yolo', 'YOLO');
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
      // Use the resolver's download cache (app-documents) — `YOLO.checkModelExists` only sees bundle/asset paths and
      // wrongly reports already-downloaded official models as missing, so the ↓ arrow never cleared.
      if (await YOLOModelResolver.isOfficialModelCached(id)) present.add(size);
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
    final preserveError =
        _modelErrorMessage != null &&
        loadedTask == _runningTask &&
        loadedSize == _runningSize;
    setState(() {
      _isModelLoading = false;
      _initialModelLoaded = true;
      _loadingStatusText = null;
      if (!preserveError) _modelErrorMessage = null;
      if (loadedSize != null) {
        _availableSizes = {..._availableSizes, loadedSize};
        _runningTask = loadedTask;
        _runningSize = loadedSize;
        unawaited(_persistTask(loadedTask));
        unawaited(_persistSize(loadedSize));
      }
    });
  }

  /// An in-place model switch failed (`YOLOView` kept the previously loaded model running). Ignore stale failures from
  /// a request the user has already moved off of — the current chip belongs to a newer request. When the failed
  /// request is still the current selection, clear the transient download chip and revert `_current*` back to the
  /// model that's actually running so the controls don't claim a model that never loaded. Reverting re-points
  /// `YOLOView` at the running model, which resolves to the same already-loaded path (no native re-switch).
  void _onModelError(Object error, String modelPath, YOLOTask? task) {
    if (!mounted || modelPath != _currentModelId) return;
    setState(() {
      _isModelLoading = false;
      _initialModelLoaded = true;
      _downloadingSize = null;
      _downloadFraction = null;
      _loadingStatusText = null;
      _modelErrorMessage = _modelSwitchErrorMessage(error);
      _currentSize = _runningSize;
      _currentTask = _runningTask;
    });
    unawaited(_refreshAvailableSizes(_runningTask));
  }

  String _modelSwitchErrorMessage(Object error) {
    final text = error.toString().toLowerCase();
    if (text.contains('failed host lookup') ||
        text.contains('no address associated with hostname') ||
        text.contains('unknown host') ||
        text.contains('unknownhostexception') ||
        text.contains('unable to resolve host') ||
        text.contains('hostname could not be found') ||
        text.contains('cannot find host') ||
        text.contains('nodename nor servname') ||
        text.contains('-1003')) {
      return 'Model download failed: device cannot resolve the release host. Check network or preload the model.';
    }
    return 'Model switch failed. Check network, model asset availability, or device logs.';
  }

  void _onTaskChanged(YOLOTask task) {
    if (task == _currentTask || _isModelLoading) return;
    HapticFeedback.selectionClick();
    unawaited(_changeTask(task));
  }

  Future<void> _changeTask(YOLOTask task) async {
    final targetSize = _clampSizeToSupported(_currentSize, task);
    final readiness = await _prepareModelSelection(task, targetSize);
    if (readiness == null || !mounted) return;

    setState(() {
      _currentTask = task;
      // The supported set can differ by platform. Clamp here too so a task switch never hands `YOLOView` a model id
      // that doesn't exist on the active platform.
      _currentSize = targetSize;
      // Abandon any in-flight download chip for the previous selection (the progress listener only resurrects it for
      // the new current size).
      _downloadingSize = readiness.isCached ? null : targetSize;
      _downloadFraction = readiness.isCached ? null : 0;
      _modelErrorMessage = null;
      _loadingStatusText = readiness.isCached
          ? 'Loading ${_displayModelNameFor(task, targetSize)}'
          : 'Downloading 0%';
      // Show the progress/status strip until `YOLOView` reports the new model loaded (onModelLoad) or failed
      // (onModelError).
      _isModelLoading = true;
    });
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
    if (size == _currentSize || _isModelLoading) return;
    HapticFeedback.selectionClick();
    unawaited(_changeSize(size));
  }

  Future<void> _changeSize(String size) async {
    final readiness = await _prepareModelSelection(_currentTask, size);
    if (readiness == null || !mounted) return;

    setState(() {
      _currentSize = size;
      // Abandon any in-flight download chip for the previous selection (see `_onTaskChanged`).
      _downloadingSize = readiness.isCached ? null : size;
      _downloadFraction = readiness.isCached ? null : 0;
      _modelErrorMessage = null;
      _loadingStatusText = readiness.isCached
          ? 'Loading ${_displayModelNameFor(_currentTask, size)}'
          : 'Downloading 0%';
      _isModelLoading = true;
    });
    // `YOLOView.didUpdateWidget` handles the resolve + native switch off the changed `modelPath` prop (see above).
  }

  Future<_ModelSelectionReadiness?> _prepareModelSelection(
    YOLOTask task,
    String size,
  ) async {
    final requestId = ++_selectionRequestId;
    final modelId = _composeModelId(task: task, size: size);
    final displayName = _displayModelNameFor(task, size);
    setState(() {
      _modelErrorMessage = null;
      _downloadingSize = null;
      _downloadFraction = null;
      _loadingStatusText = 'Checking $displayName';
      _isModelLoading = true;
    });

    final isCached = await YOLOModelResolver.isOfficialModelCached(modelId);
    if (!mounted || requestId != _selectionRequestId) return null;
    if (isCached) return const _ModelSelectionReadiness(isCached: true);

    setState(() {
      _downloadingSize = size;
      _downloadFraction = 0;
      _loadingStatusText = 'Checking network for $displayName';
    });

    if (await _canResolveReleaseHost()) {
      if (!mounted || requestId != _selectionRequestId) return null;
      return const _ModelSelectionReadiness(isCached: false);
    }

    if (!mounted || requestId != _selectionRequestId) return null;
    setState(() {
      _isModelLoading = false;
      _downloadingSize = null;
      _downloadFraction = null;
      _loadingStatusText = null;
      _modelErrorMessage =
          'Offline: connect to the internet to download $displayName, or preload the model.';
    });
    return null;
  }

  Future<bool> _canResolveReleaseHost() async {
    try {
      final addresses = await InternetAddress.lookup(
        'github.com',
      ).timeout(const Duration(seconds: 3));
      return addresses.isNotEmpty &&
          addresses.any((address) => address.rawAddress.isNotEmpty);
    } catch (_) {
      return false;
    }
  }

  void _onLensSelected(LensInfo lens) {
    HapticFeedback.selectionClick();
    unawaited(_controller.setLens(lens.zoomFactor));
  }

  void _onPlayPause() {
    HapticFeedback.lightImpact();
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
    HapticFeedback.mediumImpact();
    final bytes = await _controller.capturePhoto(withOverlays: true);
    if (bytes != null) widget.onCapture?.call(bytes);
  }

  void _onScaleStart(ScaleStartDetails _) {
    _baseScale = _zoom.value;
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
    // Update the notifier, NOT setState: this fires every inference frame (~30 fps). A setState here rebuilt the whole
    // tree — camera platform view and all controls — 30x/sec, which was the primary source of the lag.
    _metrics.value = (fps: metrics.fps, ms: metrics.processingTimeMs);
    // First inference result → the live view is fully up. Let the host dismiss its native splash now (covers the
    // model-compile + camera-bind window so startup is splash -> camera+detections, no black gap).
    if (!_readyFired) {
      _readyFired = true;
      widget.onReady?.call();
    }
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
                if (_isModelLoading)
                  Positioned.fill(
                    child: IgnorePointer(
                      child: ColoredBox(
                        color: Colors.black.withValues(alpha: 0.28),
                      ),
                    ),
                  ),
                _ShowcaseOverlay(
                  modelName: 'YOLO26$_currentSize',
                  metrics: _metrics,
                  task: _currentTask,
                  size: _currentSize,
                  availableSizes: _availableSizes,
                  supportedSizes: _supportedSizesForTask(_currentTask),
                  downloadingSize: _downloadingSize,
                  downloadFraction: _downloadFraction,
                  isModelLoading: _isModelLoading,
                  loadingStatusText: _loadingStatusText,
                  modelErrorMessage: _modelErrorMessage,
                  confidence: _confidence,
                  iou: _iou,
                  zoom: _zoom,
                  lensLabel: _lensLabel,
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
                // Ultralytics logotype, bottom-right — matches `Main.storyboard` logoImage (frame x215 y625 w159 h67
                // on a 393x852 canvas, anchored bottom-right ~19pt from the edge, sitting above the toolbar).
                const Positioned(
                  right: 19,
                  bottom: 160,
                  child: LogoOverlay(width: 159),
                ),
                // Opaque startup splash (logotype on white, matching the native launch screen) held over the camera
                // until the first model finishes loading — hides the camera-start + first GPU-compile black flash.
                if (!_initialModelLoaded)
                  const Positioned.fill(
                    child: ColoredBox(
                      color: Colors.white,
                      child: Center(child: LogoOverlay(width: 220)),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _ModelSelectionReadiness {
  const _ModelSelectionReadiness({required this.isCached});

  final bool isCached;
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
    required this.metrics,
    required this.task,
    required this.size,
    required this.availableSizes,
    required this.supportedSizes,
    required this.downloadingSize,
    required this.downloadFraction,
    required this.isModelLoading,
    required this.loadingStatusText,
    required this.modelErrorMessage,
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
  final ValueListenable<({double fps, double ms})> metrics;
  final YOLOTask task;
  final String size;
  final Set<String> availableSizes;
  final Set<String> supportedSizes;
  final String? downloadingSize;
  final double? downloadFraction;
  final bool isModelLoading;
  final String? loadingStatusText;
  final String? modelErrorMessage;
  final double confidence;
  final double iou;
  final ValueListenable<double> zoom;
  final ValueListenable<String> lensLabel;
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
  static const double _sliderRowGap = 8;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      // iOS: opt out of the bottom inset so the toolbar sits flush under the translucent home indicator (matches the
      // iOS reference). Android: the system navigation bar is opaque and always present, so respect the bottom inset
      // and let the toolbar start above it instead of being hidden underneath.
      bottom: defaultTargetPlatform == TargetPlatform.android,
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
                ValueListenableBuilder<({double fps, double ms})>(
                  valueListenable: metrics,
                  builder: (context, m, _) => PerformanceLabel(
                    modelName: modelName,
                    fps: m.fps,
                    inferenceMs: m.ms,
                  ),
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
                ModelLoadingStatus(
                  statusText: isModelLoading ? loadingStatusText : null,
                  progress: isModelLoading ? downloadFraction : null,
                  errorMessage: modelErrorMessage,
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
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ThresholdSliderRow(
                  label: 'Confidence Threshold',
                  value: confidence,
                  min: 0,
                  max: 1,
                  onChanged: onConfidenceChanged,
                  sliderWidthFactor: 0.46,
                ),
                const SizedBox(height: _sliderRowGap),
                ThresholdSliderRow(
                  label: 'IoU Threshold',
                  value: iou,
                  min: 0,
                  max: 1,
                  onChanged: onIouChanged,
                  sliderWidthFactor: 0.46,
                ),
              ],
            ),
          ),

          // -- Zoom HUD ----------------------------------------------------------------------------------------
          // Logo is NOT here — it sits bottom-right per `Main.storyboard` (handled in the outer Stack).
          Padding(
            padding: const EdgeInsets.fromLTRB(
              _sidePadding,
              4,
              _sidePadding,
              0,
            ),
            child: ValueListenableBuilder<double>(
              valueListenable: zoom,
              builder: (context, z, _) => ValueListenableBuilder<String>(
                valueListenable: lensLabel,
                builder: (context, label, _) =>
                    ZoomIndicator(currentZoom: z, lensLabel: label),
              ),
            ),
          ),

          // -- Lens picker -------------------------------------------------------------------------------------
          Padding(
            padding: const EdgeInsets.fromLTRB(
              _sidePadding,
              2,
              _sidePadding,
              2,
            ),
            child: ValueListenableBuilder<double>(
              valueListenable: zoom,
              builder: (context, z, _) => LensPicker(
                lenses: lenses,
                currentZoomFactor: z,
                onLensSelected: onLensSelected,
              ),
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
          // 66pt toolbar flush at the very bottom (its black band covers the home-indicator inset). Matches
          // `toolbar.frame = (0, height - 66, width, 66)` in `yolo-ios-app/Sources/YOLO/YOLOView.swift:806`. The
          // earlier safe-area fill Container below this pushed the buttons ~34pt too high.
          CameraToolbar(
            isPaused: isPaused,
            onPlayPause: onPlayPause,
            onSwitchCamera: onSwitchCamera,
            onShare: onShare,
          ),
        ],
      ),
    );
  }
}
