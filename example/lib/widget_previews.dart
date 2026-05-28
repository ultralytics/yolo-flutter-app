// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Widget previews for the plugin's Material 3 widgets. Run with:
//
//   cd example && flutter widget-preview start
//
// Opens a browser tab where each annotated function is rendered standalone with no platform-view dependency, so this
// lets you iterate on visuals (TaskSegmentedControl, sliders, lens picker, toolbar, etc.) without an iOS Simulator or
// Android emulator. YOLOView itself can't render here — it needs a real platform view — so the live camera widget is
// not previewed. Compose them in YOLOShowcase to see the full screen on device.

import 'package:flutter/material.dart';
import 'package:flutter/widget_previews.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
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

Widget darkBg(Widget child) => Container(
  color: Colors.black,
  padding: const EdgeInsets.all(16),
  child: Center(child: child),
);

// region Task selector

@Preview(name: 'TaskSegmentedControl — Detect', group: 'Task', wrapper: darkBg)
Widget taskDetect() =>
    TaskSegmentedControl(currentTask: YOLOTask.detect, onTaskChanged: _noop);

@Preview(name: 'TaskSegmentedControl — Segment', group: 'Task', wrapper: darkBg)
Widget taskSegment() =>
    TaskSegmentedControl(currentTask: YOLOTask.segment, onTaskChanged: _noop);

@Preview(name: 'TaskSegmentedControl — Pose', group: 'Task', wrapper: darkBg)
Widget taskPose() =>
    TaskSegmentedControl(currentTask: YOLOTask.pose, onTaskChanged: _noop);

@Preview(
  name: 'TaskSegmentedControl — no Semantic',
  group: 'Task',
  wrapper: darkBg,
)
Widget taskNoSemantic() => TaskSegmentedControl(
  currentTask: YOLOTask.detect,
  showSemanticTask: false,
  onTaskChanged: _noop,
);

// endregion

// region Model size

@Preview(
  name: 'ModelSize — only n available',
  group: 'Model size',
  wrapper: darkBg,
)
Widget modelOnlyN() => ModelSizeSegmentedControl(
  currentSize: 'n',
  availableSizes: const {'n'},
  onSizeChanged: _noopStr,
);

@Preview(
  name: 'ModelSize — all available',
  group: 'Model size',
  wrapper: darkBg,
)
Widget modelAllAvailable() => ModelSizeSegmentedControl(
  currentSize: 's',
  availableSizes: const {'n', 's', 'm', 'l', 'x'},
  onSizeChanged: _noopStr,
);

@Preview(
  name: 'ModelSize — downloading m at 42%',
  group: 'Model size',
  wrapper: darkBg,
)
Widget modelDownloading() => ModelSizeSegmentedControl(
  currentSize: 'm',
  availableSizes: const {'n'},
  downloadingSize: 'm',
  downloadFraction: 0.42,
  onSizeChanged: _noopStr,
);

@Preview(
  name: 'ModelSize — supported subset (Android v0.2.0)',
  group: 'Model size',
  wrapper: darkBg,
)
Widget modelSupportedSubset() => ModelSizeSegmentedControl(
  currentSize: 'n',
  availableSizes: const {'n'},
  supportedSizes: const {'n'},
  onSizeChanged: _noopStr,
);

// endregion

// region Threshold sliders

@Preview(
  name: 'ThresholdSliderRow — Confidence 0.25',
  group: 'Sliders',
  wrapper: darkBg,
)
Widget sliderConfidence() => ThresholdSliderRow(
  label: 'Confidence Threshold',
  value: 0.25,
  min: 0,
  max: 1,
  onChanged: _noopD,
);

@Preview(
  name: 'ThresholdSliderRow — IoU 0.70',
  group: 'Sliders',
  wrapper: darkBg,
)
Widget sliderIou() => ThresholdSliderRow(
  label: 'IoU Threshold',
  value: 0.7,
  min: 0,
  max: 1,
  onChanged: _noopD,
);

@Preview(
  name: 'ThresholdSliderRow — numItems 30',
  group: 'Sliders',
  wrapper: darkBg,
)
Widget sliderNumItems() => ThresholdSliderRow(
  label: 'Max Detections',
  value: 30,
  min: 5,
  max: 50,
  divisions: 9,
  isInt: true,
  onChanged: _noopD,
);

// endregion

// region Lens picker + zoom indicator

@Preview(name: 'LensPicker — 0.5x / 1 / 2', group: 'Lens', wrapper: darkBg)
Widget lensTriple() => LensPicker(
  lenses: const [
    LensInfo(zoomFactor: 0.5, label: 'Ultra wide camera'),
    LensInfo(zoomFactor: 1, label: 'Wide camera'),
    LensInfo(zoomFactor: 2, label: 'Telephoto camera'),
  ],
  currentZoomFactor: 1,
  onLensSelected: _noopLens,
);

@Preview(
  name: 'LensPicker — 0.5x / 1 / 4 (iPhone 17 Pro)',
  group: 'Lens',
  wrapper: darkBg,
)
Widget lensIPhone17() => LensPicker(
  lenses: const [
    LensInfo(zoomFactor: 0.5, label: 'Ultra wide camera'),
    LensInfo(zoomFactor: 1, label: 'Wide camera'),
    LensInfo(zoomFactor: 4, label: 'Telephoto camera'),
  ],
  currentZoomFactor: 0.5,
  onLensSelected: _noopLens,
);

@Preview(
  name: 'LensPicker — single lens fallback',
  group: 'Lens',
  wrapper: darkBg,
)
Widget lensSingle() => LensPicker(
  lenses: const [],
  currentZoomFactor: 1,
  onLensSelected: _noopLens,
);

@Preview(
  name: 'ZoomIndicator — 0.50x Ultra wide',
  group: 'Lens',
  wrapper: darkBg,
)
Widget zoomUltrawide() =>
    const ZoomIndicator(currentZoom: 0.5, lensLabel: 'Ultra wide camera');

@Preview(
  name: 'ZoomIndicator — 2.00x Telephoto',
  group: 'Lens',
  wrapper: darkBg,
)
Widget zoomTele() =>
    const ZoomIndicator(currentZoom: 2, lensLabel: 'Telephoto camera');

// endregion

// region Toolbar

@Preview(name: 'CameraToolbar — playing', group: 'Toolbar', wrapper: darkBg)
Widget toolbarPlaying() => CameraToolbar(
  isPaused: false,
  onPlayPause: _noop0,
  onSwitchCamera: _noop0,
  onShare: _noop0,
);

@Preview(name: 'CameraToolbar — paused', group: 'Toolbar', wrapper: darkBg)
Widget toolbarPaused() => CameraToolbar(
  isPaused: true,
  onPlayPause: _noop0,
  onSwitchCamera: _noop0,
  onShare: _noop0,
);

// endregion

// region Performance label

@Preview(
  name: 'PerformanceLabel — YOLO26n 29.9 FPS',
  group: 'Performance',
  wrapper: darkBg,
)
Widget perfDetect() =>
    const PerformanceLabel(modelName: 'YOLO26n', fps: 29.9, inferenceMs: 11.1);

@Preview(
  name: 'PerformanceLabel — YOLO26x 12.4 FPS',
  group: 'Performance',
  wrapper: darkBg,
)
Widget perfHeavy() =>
    const PerformanceLabel(modelName: 'YOLO26x', fps: 12.4, inferenceMs: 78.6);

// endregion

// region Overlays

@Preview(name: 'LogoOverlay', group: 'Overlays', wrapper: darkBg)
Widget logo() => const LogoOverlay();

// endregion

// region Full showcase mock

/// Renders the YOLOShowcase composition with a placeholder where the live camera (a platform view) would be. Lets you
/// see the assembled iPhone-style layout — top labels + task/model segmented controls + bottom sliders + lens picker +
/// toolbar + logo + zoom indicator — without an iOS Simulator or Android emulator. The visuals match the real
/// YOLOShowcase build() one-for-one; the only thing fake is the camera region (`_CameraStub`) and the controller-driven
/// state, which is hard-coded so the layout is fully realised.
@Preview(name: 'YOLOShowcase — Detect / YOLO26n / iPhone-style', size: Size(393, 852))
Widget showcaseDetectIphone() => const _ShowcaseMock(
      task: YOLOTask.detect,
      modelSize: 'n',
      fps: 29.9,
      inferenceMs: 11.1,
      confidence: 0.25,
      iou: 0.7,
      currentZoom: 1,
      currentLensLabel: 'Wide camera',
      lenses: [
        LensInfo(zoomFactor: 0.5, label: 'Ultra wide camera'),
        LensInfo(zoomFactor: 1, label: 'Wide camera'),
        LensInfo(zoomFactor: 4, label: 'Telephoto camera'),
      ],
    );

@Preview(name: 'YOLOShowcase — Segment / YOLO26s downloading', size: Size(393, 852))
Widget showcaseSegmentDownloading() => const _ShowcaseMock(
      task: YOLOTask.segment,
      modelSize: 's',
      availableSizes: {'n'},
      downloadingSize: 's',
      downloadFraction: 0.42,
      fps: 18.3,
      inferenceMs: 22.5,
      confidence: 0.4,
      iou: 0.55,
      currentZoom: 0.5,
      currentLensLabel: 'Ultra wide camera',
      lenses: [
        LensInfo(zoomFactor: 0.5, label: 'Ultra wide camera'),
        LensInfo(zoomFactor: 1, label: 'Wide camera'),
        LensInfo(zoomFactor: 2, label: 'Telephoto camera'),
      ],
    );

@Preview(name: 'YOLOShowcase — Pose / YOLO26x / Tele 2x', size: Size(393, 852))
Widget showcasePoseTele() => const _ShowcaseMock(
      task: YOLOTask.pose,
      modelSize: 'x',
      availableSizes: {'n', 's', 'm', 'l', 'x'},
      fps: 12.4,
      inferenceMs: 78.6,
      confidence: 0.3,
      iou: 0.6,
      currentZoom: 2,
      currentLensLabel: 'Telephoto camera',
      lenses: [
        LensInfo(zoomFactor: 0.5, label: 'Ultra wide camera'),
        LensInfo(zoomFactor: 1, label: 'Wide camera'),
        LensInfo(zoomFactor: 2, label: 'Telephoto camera'),
      ],
    );

@Preview(name: 'FocusReticle — at center', group: 'Overlays', wrapper: darkBg)
Widget reticle() => const SizedBox(
  width: 300,
  height: 300,
  child: Stack(children: [FocusReticle(position: Offset(150, 150))]),
);

// endregion

void _noop(YOLOTask _) {}
void _noopStr(String _) {}
void _noopD(double _) {}
void _noopLens(LensInfo _) {}
void _noop0() {}

/// Mocks the [YOLOShowcase] layout for widget previews. Stubs out the live camera (a platform view) with a static dark
/// region so the rest of the iPhone-style overlay (top labels + segmented controls + sliders + lens picker + zoom
/// indicator + toolbar + logo) renders standalone. Visuals mirror `lib/widgets/yolo_showcase.dart#build`.
class _ShowcaseMock extends StatelessWidget {
  const _ShowcaseMock({
    required this.task,
    required this.modelSize,
    required this.fps,
    required this.inferenceMs,
    required this.confidence,
    required this.iou,
    required this.currentZoom,
    required this.currentLensLabel,
    required this.lenses,
    this.availableSizes = const {'n'},
    this.downloadingSize,
    this.downloadFraction,
  });

  final YOLOTask task;
  final String modelSize;
  final double fps;
  final double inferenceMs;
  final double confidence;
  final double iou;
  final double currentZoom;
  final String currentLensLabel;
  final List<LensInfo> lenses;
  final Set<String> availableSizes;
  final String? downloadingSize;
  final double? downloadFraction;

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: ThemeData.dark(useMaterial3: true),
      child: Material(
        color: Colors.black,
        child: Stack(
          fit: StackFit.expand,
          children: [
            _CameraStub(),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                child: Column(
                  children: [
                    Align(
                      alignment: Alignment.centerLeft,
                      child: PerformanceLabel(
                        modelName: 'YOLO26$modelSize',
                        fps: fps,
                        inferenceMs: inferenceMs,
                      ),
                    ),
                    const SizedBox(height: 8),
                    TaskSegmentedControl(currentTask: task, onTaskChanged: _noop),
                    const SizedBox(height: 8),
                    ModelSizeSegmentedControl(
                      currentSize: modelSize,
                      availableSizes: availableSizes,
                      downloadingSize: downloadingSize,
                      downloadFraction: downloadFraction,
                      onSizeChanged: _noopStr,
                    ),
                    const Spacer(),
                    ThresholdSliderRow(
                      label: 'Confidence Threshold',
                      value: confidence,
                      min: 0,
                      max: 1,
                      onChanged: _noopD,
                    ),
                    ThresholdSliderRow(
                      label: 'IoU Threshold',
                      value: iou,
                      min: 0,
                      max: 1,
                      onChanged: _noopD,
                    ),
                    const Align(
                      alignment: Alignment.bottomRight,
                      child: Padding(
                        padding: EdgeInsets.only(bottom: 8),
                        child: LogoOverlay(),
                      ),
                    ),
                    ZoomIndicator(currentZoom: currentZoom, lensLabel: currentLensLabel),
                    const SizedBox(height: 4),
                    LensPicker(
                      lenses: lenses,
                      currentZoomFactor: currentZoom,
                      onLensSelected: _noopLens,
                    ),
                    const SizedBox(height: 8),
                    CameraToolbar(
                      isPaused: false,
                      onPlayPause: _noop0,
                      onSwitchCamera: _noop0,
                      onShare: _noop0,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Stand-in for the live camera region used in widget previews. Renders a dark gradient with a faint label so it's
/// obvious this is a mock and not a black bug.
class _CameraStub extends StatelessWidget {
  const _CameraStub();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF111418), Color(0xFF1F2933)],
        ),
      ),
      child: Center(
        child: Text(
          'Live camera renders here on device',
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.white.withValues(alpha: 0.25),
              ),
        ),
      ),
    );
  }
}
