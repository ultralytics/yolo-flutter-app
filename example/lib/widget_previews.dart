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
