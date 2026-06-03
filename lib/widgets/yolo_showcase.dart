// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'package:flutter/cupertino.dart' show CupertinoIcons, CupertinoColors;
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
import 'package:ultralytics_yolo/widgets/model_size_segmented_control.dart';
import 'package:ultralytics_yolo/widgets/performance_label.dart';
import 'package:ultralytics_yolo/widgets/task_segmented_control.dart';
import 'package:ultralytics_yolo/widgets/threshold_slider_row.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/widgets/zoom_indicator.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

/// One-import camera UI matching the iOS showcase layout. Composes every widget under `lib/widgets/`, owns gestures
/// (pinch + tap-to-focus), drives the controller, and persists the last task across launches.
class YOLOShowcase extends StatefulWidget {
  /// Task to load on first launch (overridden by stored preference).
  final YOLOTask initialTask;

  /// Model size (`n/s/m/l/x`) to load on first launch.
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

  bool _isPaused = false;
  bool _torchOn = false;
  Offset? _focusPosition;
  double _baseScale = 1;
  Size _viewSize = Size.zero;

  StreamSubscription<double>? _zoomSub;
  StreamSubscription<String>? _lensSub;
  StreamSubscription<Offset>? _focusSub;
  StreamSubscription<DownloadProgress>? _progressSub;

  static const _infoResources =
      <({String title, String subtitle, IconData icon, String url})>[
        (
          title: 'Ultralytics Docs',
          subtitle: 'Training, prediction, export, and deployment guides.',
          icon: Icons.menu_book_outlined,
          url: 'https://docs.ultralytics.com',
        ),
        (
          title: 'YOLO Models',
          subtitle: 'Explore YOLO26 tasks, sizes, and performance.',
          icon: Icons.view_in_ar_outlined,
          url: 'https://platform.ultralytics.com/ultralytics/yolo26',
        ),
        (
          title: 'GitHub',
          subtitle: 'Source code, releases, and open-source tools.',
          icon: Icons.code,
          url: 'https://github.com/ultralytics/ultralytics',
        ),
        (
          title: 'Licensing',
          subtitle: 'AGPL-3.0 and Enterprise License options.',
          icon: Icons.description_outlined,
          url: 'https://www.ultralytics.com/license',
        ),
      ];

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
              ? (_isModelLoading
                    ? _loadingTextFor(_currentTask, _currentSize)
                    : null)
              : _loadingTextFor(
                  _currentTask,
                  size,
                  progress: progress.fraction,
                );
          _isModelLoading = done ? _isModelLoading : true;
        }
      });
    });

    WakelockPlus.enable();
    unawaited(_bootstrap());
  }

  Future<void> _bootstrap() async {
    final prefs = await SharedPreferences.getInstance();
    final storedTask = prefs.getString(_prefsTaskKey);
    if (mounted) {
      setState(() {
        if (storedTask != null) {
          final parsed = YOLOTaskParsing.tryParse(storedTask);
          if (parsed != null) _currentTask = parsed;
        }
        // The model size always starts at nano (matches the native iOS app) and is never restored from prefs.
        // Clamp to whatever the resolver actually publishes for the active platform so we never hand `YOLOView`
        // a `_currentModelId` that 404s on first launch before the chip even renders.
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
  /// `↓` glyph. Only models the resolver declares (`YOLO.officialModels`) are probed.
  Future<Set<String>> _scanAvailableSizes(YOLOTask task) async {
    final supported = _supportedSizesForTask(task);
    final present = <String>{};
    for (final size in supported) {
      final id = _composeModelId(task: task, size: size);
      if (await YOLOModelResolver.isOfficialModelAvailableLocally(id)) {
        present.add(size);
      }
    }
    return present;
  }

  Future<void> _persistTask(YOLOTask task) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsTaskKey, task.name);
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
      _loadingStatusText = null;
      if (!preserveError) _modelErrorMessage = null;
      if (loadedSize != null) {
        _availableSizes = {..._availableSizes, loadedSize};
        _runningTask = loadedTask;
        _runningSize = loadedSize;
        unawaited(_persistTask(loadedTask));
      }
    });
    _restoreInferenceAfterModelSwitch();
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
      _downloadingSize = null;
      _downloadFraction = null;
      _loadingStatusText = null;
      _modelErrorMessage = _modelSwitchErrorMessage(error);
      _currentSize = _runningSize;
      _currentTask = _runningTask;
    });
    _restoreInferenceAfterModelSwitch();
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
        text.contains('network is unreachable') ||
        text.contains('network unreachable') ||
        text.contains('connection timed out') ||
        text.contains('connection timeout') ||
        text.contains('connection failed') ||
        text.contains('software caused connection abort') ||
        text.contains('nodename nor servname') ||
        RegExp(r'(^|[^0-9])-1003([^0-9]|$)').hasMatch(text)) {
      return 'Model download failed: device cannot resolve the release host. Check network or preload the model.';
    }
    return 'Model switch failed. Check network, model asset availability, or device logs.';
  }

  void _onTaskChanged(YOLOTask task) {
    if (task == _currentTask) return;
    HapticFeedback.selectionClick();
    // Match the native iOS app: switching tasks always resets the model size to nano (the smallest/first size)
    // rather than carrying over the previously selected size.
    final targetSize = _clampSizeToSupported('n', task);
    YOLOModelManager.clearDownloadCancellation(
      _composeModelId(task: task, size: targetSize),
    );
    _suppressInferenceForModelSwitch();
    setState(() {
      _currentTask = task;
      // The supported set can differ by platform. Clamp here too so a task switch never hands `YOLOView` a model id
      // that doesn't exist on the active platform.
      _currentSize = targetSize;
      // Abandon any in-flight download chip for the previous selection (the progress listener only resurrects it for
      // the new current size).
      _downloadingSize = null;
      _downloadFraction = null;
      _availableSizes = {};
      _modelErrorMessage = null;
      _loadingStatusText = _loadingTextFor(task, targetSize);
      _isModelLoading = true;
    });
    unawaited(_updateTargetDownloadState(task, targetSize));
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
    HapticFeedback.selectionClick();
    final isCached = _availableSizes.contains(size);
    YOLOModelManager.clearDownloadCancellation(
      _composeModelId(task: _currentTask, size: size),
    );
    _suppressInferenceForModelSwitch();
    setState(() {
      _currentSize = size;
      // Abandon any in-flight download chip for the previous selection (see `_onTaskChanged`).
      _downloadingSize = isCached ? null : size;
      _downloadFraction = isCached ? null : 0;
      _modelErrorMessage = null;
      _loadingStatusText = isCached
          ? _loadingTextFor(_currentTask, size)
          : _loadingTextFor(_currentTask, size, progress: 0);
      _isModelLoading = true;
    });
    // `YOLOView.didUpdateWidget` handles the resolve + native switch off the changed `modelPath` prop (see above).
  }

  Future<void> _updateTargetDownloadState(YOLOTask task, String size) async {
    final local = await YOLOModelResolver.isOfficialModelAvailableLocally(
      _composeModelId(task: task, size: size),
    );
    if (!mounted ||
        !_isModelLoading ||
        _currentTask != task ||
        _currentSize != size) {
      return;
    }
    setState(() {
      _downloadingSize = local ? null : size;
      _downloadFraction = local ? null : 0;
      _loadingStatusText = local
          ? _loadingTextFor(task, size)
          : _loadingTextFor(task, size, progress: 0);
    });
  }

  void _cancelModelSwitch() {
    if (!_isModelLoading) return;
    YOLOModelManager.cancelDownload(_currentModelId);
    setState(() {
      _currentTask = _runningTask;
      _currentSize = _runningSize;
      _isModelLoading = false;
      _downloadingSize = null;
      _downloadFraction = null;
      _loadingStatusText = null;
      _modelErrorMessage = null;
    });
    _restoreInferenceAfterModelSwitch();
    unawaited(_refreshAvailableSizes(_runningTask));
  }

  void _suppressInferenceForModelSwitch() {
    unawaited(_controller.setShowOverlays(false));
  }

  void _restoreInferenceAfterModelSwitch() {
    unawaited(_controller.setShowOverlays(true));
  }

  void _onLensSelected(LensInfo lens) {
    HapticFeedback.selectionClick();
    unawaited(_controller.setLens(lens.zoomFactor));
  }

  Future<void> _onSwitchCamera() async {
    setState(() => _lenses = const [LensInfo(zoomFactor: 1, label: 'Camera')]);
    _zoom.value = 1;
    _lensLabel.value = '';
    // Switching the camera input drops the torch (the new device may not have one); reset both the controller's
    // cached state and the UI flag so they stay in sync across the switch.
    _controller.resetTorchState();
    if (_torchOn) setState(() => _torchOn = false);
    await _controller.switchCamera();
    await Future<void>.delayed(const Duration(milliseconds: 300));
    await _refreshLenses();
  }

  Future<void> _onToggleTorch() async {
    HapticFeedback.selectionClick();
    await _controller.toggleTorch();
    if (mounted) setState(() => _torchOn = _controller.isTorchEnabled);
  }

  void _onPlayPause() {
    if (_isModelLoading) return;
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

  void _showInfoSheet(BuildContext context) {
    HapticFeedback.selectionClick();
    unawaited(
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          fullscreenDialog: true,
          builder: (pageContext) => Scaffold(
            appBar: AppBar(title: const Text('About YOLO')),
            body: ListView(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 28),
              children: [
                const ListTile(
                  leading: Icon(Icons.center_focus_strong),
                  title: Text('Ultralytics YOLO'),
                  subtitle: Text('Real-time AI vision on-device'),
                ),
                const ListTile(
                  title: Text('The App'),
                  subtitle: Text(
                    'Try detection, segmentation, classification, pose estimation, and oriented bounding box models directly on-device.',
                  ),
                ),
                const ListTile(
                  title: Text('YOLO Models'),
                  subtitle: Text(
                    'Compare nano models bundled with the app and larger official models downloaded when selected.',
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                  child: Text(
                    'Continue Learning',
                    style: Theme.of(pageContext).textTheme.titleMedium,
                  ),
                ),
                for (final resource in _infoResources)
                  ListTile(
                    leading: Icon(resource.icon, color: Colors.blue),
                    title: Text(
                      resource.title,
                      style: const TextStyle(color: Colors.blue),
                    ),
                    subtitle: Text(resource.subtitle),
                    trailing: const Icon(Icons.open_in_new, size: 18),
                    onTap: () {
                      unawaited(_openInfoUrl(resource.url));
                    },
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _openInfoUrl(String url) async {
    final uri = Uri.parse(url);
    try {
      final launched = await launchUrl(
        uri,
        mode: LaunchMode.externalApplication,
      );
      if (!launched) debugPrint('Unable to open $uri');
    } catch (error) {
      debugPrint('Unable to open $uri: $error');
    }
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
  }

  static String _taskLabel(YOLOTask task) {
    return switch (task) {
      YOLOTask.detect => 'Detect',
      YOLOTask.segment => 'Segment',
      YOLOTask.semantic => 'Semantic',
      YOLOTask.classify => 'Classify',
      YOLOTask.pose => 'Pose',
      YOLOTask.obb => 'OBB',
    };
  }

  static String _loadingTextFor(
    YOLOTask task,
    String size, {
    double? progress,
  }) {
    final model = 'YOLO26$size ${_taskLabel(task)}';
    if (progress == null) return 'Loading $model';
    final percent = (progress * 100).clamp(0, 99).round();
    return 'Downloading $model $percent%';
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
            final isLandscape = constraints.maxWidth > constraints.maxHeight;
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
                  metrics: _metrics,
                  task: _currentTask,
                  size: _currentSize,
                  availableSizes: _availableSizes,
                  supportedSizes: _supportedSizesForTask(_currentTask),
                  downloadingSize: _downloadingSize,
                  downloadFraction: _downloadFraction,
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
                  onSwitchCamera: () => unawaited(_onSwitchCamera()),
                  isTorchOn: _torchOn,
                  onToggleTorch: () => unawaited(_onToggleTorch()),
                  onShare: () => unawaited(_onShare()),
                  onInfo: () => _showInfoSheet(context),
                ),
                // Ultralytics logotype, bottom-right — matches `Main.storyboard` logoImage (frame x215 y625 w159 h67
                // on a 393x852 canvas, anchored bottom-right ~19pt from the edge, sitting above the toolbar).
                Positioned(
                  right: 19,
                  bottom: isLandscape ? 100 : 160,
                  child: const LogoOverlay(width: 159),
                ),
                Positioned.fill(
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 180),
                    reverseDuration: const Duration(milliseconds: 120),
                    switchInCurve: Curves.easeOutCubic,
                    switchOutCurve: Curves.easeInCubic,
                    child: _isModelLoading
                        ? _ModelSwitchLoadingOverlay(
                            statusText: _loadingStatusText,
                            progress: _downloadFraction,
                            onCancel: _downloadingSize == null
                                ? null
                                : _cancelModelSwitch,
                          )
                        : const SizedBox.shrink(),
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

class _ModelSwitchLoadingOverlay extends StatelessWidget {
  const _ModelSwitchLoadingOverlay({
    this.statusText,
    this.progress,
    this.onCancel,
  });

  final String? statusText;
  final double? progress;
  final VoidCallback? onCancel;

  @override
  Widget build(BuildContext context) {
    final value = progress?.clamp(0.0, 1.0);
    return Stack(
      fit: StackFit.expand,
      children: [
        ModalBarrier(
          dismissible: false,
          color: Colors.black.withValues(alpha: 0.72),
        ),
        SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 320),
              child: Card(
                margin: const EdgeInsets.symmetric(horizontal: 24),
                color: const Color(0xFF202124),
                elevation: 10,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                  side: BorderSide(color: Colors.white.withValues(alpha: 0.12)),
                ),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(22, 20, 22, 14),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (value == null)
                        const SizedBox(
                          width: 34,
                          height: 34,
                          child: CircularProgressIndicator(strokeWidth: 3),
                        )
                      else
                        LinearProgressIndicator(value: value),
                      const SizedBox(height: 16),
                      Text(
                        statusText ?? 'Loading model',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                            ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        value == null
                            ? 'Preparing inference'
                            : 'Downloading model weights',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: Colors.white.withValues(alpha: 0.72),
                        ),
                      ),
                      if (onCancel != null) ...[
                        const SizedBox(height: 10),
                        TextButton(
                          onPressed: onCancel,
                          child: const Text('Cancel'),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// Stateless overlay sandwich. Layout mirrors `yolo-ios-app/Sources/UltralyticsYOLO/YOLOView.swift#layoutPortrait`
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
    required this.isTorchOn,
    required this.onToggleTorch,
    required this.onShare,
    required this.onInfo,
  });

  final String modelName;
  final ValueListenable<({double fps, double ms})> metrics;
  final YOLOTask task;
  final String size;
  final Set<String> availableSizes;
  final Set<String> supportedSizes;
  final String? downloadingSize;
  final double? downloadFraction;
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
  final bool isTorchOn;
  final VoidCallback onToggleTorch;
  final VoidCallback onShare;
  final VoidCallback onInfo;

  // iOS YOLOView ports — kept as constants so the layout reads like the Swift source.
  static const double _sidePadding = 20;
  static const double _topGap = 8;
  static const double _sliderRowGap = 2;

  @override
  Widget build(BuildContext context) {
    final topInset = MediaQuery.paddingOf(context).top;
    return SafeArea(
      left: false,
      right: false,
      bottom: defaultTargetPlatform == TargetPlatform.android,
      child: LayoutBuilder(
        builder: (context, constraints) {
          return constraints.maxWidth > constraints.maxHeight
              ? _buildLandscape(constraints, topInset)
              : _buildPortrait();
        },
      ),
    );
  }

  Widget _buildPortrait() {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(_sidePadding, 8, _sidePadding, 0),
          child: _topControls(),
        ),
        const Spacer(),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: _sidePadding),
          child: _thresholdControls(sliderWidthFactor: 0.46),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(_sidePadding, 4, _sidePadding, 0),
          child: _zoomLabel(),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(_sidePadding, 2, _sidePadding, 2),
          child: Stack(
            alignment: Alignment.center,
            clipBehavior: Clip.none,
            children: [
              _lensPicker(),
              // Version stamp at the lens-tabs vertical level, tucked into the bottom-left corner (negative left
              // escapes the row's side padding so it sits closer to the screen edge than the controls).
              if (versionLabel != null)
                Positioned(
                  left: -(_sidePadding - 8),
                  bottom: 0,
                  child: _version(),
                ),
            ],
          ),
        ),
        _toolbar(),
      ],
    );
  }

  Widget _buildLandscape(BoxConstraints constraints, double topInset) {
    final topWidth = (constraints.maxWidth * 0.62).clamp(360.0, 620.0);
    final sliderWidth = (constraints.maxWidth * 0.28).clamp(220.0, 300.0);
    final hasLenses = lenses.isNotEmpty;
    return Stack(
      fit: StackFit.expand,
      children: [
        Align(
          alignment: Alignment.topCenter,
          child: Transform.translate(
            offset: Offset(0, -topInset),
            child: Padding(
              padding: const EdgeInsets.only(top: 6),
              child: SizedBox(width: topWidth, child: _topControls()),
            ),
          ),
        ),
        Positioned(
          left: _sidePadding,
          bottom: CameraToolbar.height + (hasLenses ? 18 : 0),
          width: sliderWidth,
          child: _thresholdControls(sliderWidthFactor: 1),
        ),
        Positioned(
          left: _sidePadding,
          right: _sidePadding,
          bottom: CameraToolbar.height + (hasLenses ? 42 : 8),
          child: _zoomLabel(),
        ),
        if (hasLenses)
          Positioned(
            left: _sidePadding,
            right: _sidePadding,
            bottom: CameraToolbar.height + 2,
            child: _lensPicker(),
          ),
        if (versionLabel != null)
          Positioned(
            left: _sidePadding,
            bottom: CameraToolbar.height + 4,
            child: _version(),
          ),
        Positioned(left: 0, right: 0, bottom: 0, child: _toolbar()),
      ],
    );
  }

  Widget _topControls() {
    return Column(
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
        if (modelErrorMessage != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.72),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
              ),
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 8,
                ),
                child: Text(
                  modelErrorMessage!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _thresholdControls({required double sliderWidthFactor}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        ThresholdSliderRow(
          label: 'Confidence Threshold',
          value: confidence,
          min: 0,
          max: 1,
          onChanged: onConfidenceChanged,
          sliderWidthFactor: sliderWidthFactor,
        ),
        const SizedBox(height: _sliderRowGap),
        ThresholdSliderRow(
          label: 'IoU Threshold',
          value: iou,
          min: 0,
          max: 1,
          onChanged: onIouChanged,
          sliderWidthFactor: sliderWidthFactor,
        ),
      ],
    );
  }

  Widget _zoomLabel() {
    return ValueListenableBuilder<double>(
      valueListenable: zoom,
      builder: (context, z, _) => ValueListenableBuilder<String>(
        valueListenable: lensLabel,
        builder: (context, label, _) =>
            ZoomIndicator(currentZoom: z, lensLabel: label),
      ),
    );
  }

  Widget _lensPicker() {
    return ValueListenableBuilder<double>(
      valueListenable: zoom,
      builder: (context, z, _) => LensPicker(
        lenses: lenses,
        currentZoomFactor: z,
        onLensSelected: onLensSelected,
        trailing: _torchControl(),
      ),
    );
  }

  /// Torch chip plus its "Torch on" note to the right. The note's space is reserved (shown/hidden in place) so
  /// toggling the torch never shifts the chip or the centered zoom options.
  Widget _torchControl() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _torchChip(),
        const SizedBox(width: 6),
        Visibility(
          visible: isTorchOn,
          maintainSize: true,
          maintainAnimation: true,
          maintainState: true,
          child: const Text(
            'Torch on',
            style: TextStyle(
              color: CupertinoColors.systemYellow,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }

  /// Torch toggle styled as a chip the same size as the lens (zoom) chips, sitting directly next to them. Uses the
  /// native lightning glyph (Cupertino has no flashlight icon).
  Widget _torchChip() {
    return Semantics(
      button: true,
      label: isTorchOn ? 'Turn torch off' : 'Turn torch on',
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onToggleTorch,
        child: Container(
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.38),
            borderRadius: BorderRadius.circular(8),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Icon(
            isTorchOn
                ? CupertinoIcons.bolt_fill
                : CupertinoIcons.bolt_slash_fill,
            color: isTorchOn ? CupertinoColors.systemYellow : Colors.white,
            size: 17,
          ),
        ),
      ),
    );
  }

  Widget _version() {
    return Text(
      versionLabel!,
      style: TextStyle(
        color: Colors.white.withValues(alpha: 0.7),
        fontSize: 10,
      ),
    );
  }

  Widget _toolbar() {
    return CameraToolbar(
      isPaused: isPaused,
      onPlayPause: onPlayPause,
      onSwitchCamera: onSwitchCamera,
      onShare: onShare,
      onInfo: onInfo,
    );
  }
}
