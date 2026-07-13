// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

/// Picks the active [YOLOTask] using a `CupertinoSlidingSegmentedControl` styled to match
/// `yolo-ios-app`'s task control (`Det Seg Sem Depth Cls Pose OBB`).
///
/// Material 3's `SegmentedButton` was used previously but its pill-shaped chips don't match the iOS reference's
/// inset rounded-rect look. The Cupertino variant gives us:
///   * white text on transparent default segments
///   * a translucent white "thumb" on the selected segment (matching iOS' `white.withAlphaComponent(0.18)`)
///   * a thin rounded background (`black.withAlphaComponent(0.7)`).
class TaskSegmentedControl extends StatelessWidget {
  /// Currently-selected task; drives which segment is highlighted.
  final YOLOTask currentTask;

  /// Invoked with the user's selection.
  final ValueChanged<YOLOTask> onTaskChanged;

  /// When `false`, the `Semantic` segment is hidden for hosts that do not want to expose semantic segmentation.
  final bool showSemanticTask;

  /// Whether to show the Android-only `Depth` segment.
  final bool showDepthTask;

  const TaskSegmentedControl({
    super.key,
    required this.currentTask,
    required this.onTaskChanged,
    this.showSemanticTask = true,
    this.showDepthTask = false,
  });

  @override
  Widget build(BuildContext context) {
    final tasks = YOLOTask.values
        .where((t) => showSemanticTask || t != YOLOTask.semantic)
        .where((t) => showDepthTask || t != YOLOTask.depth)
        .toList(growable: false);

    // Content-hug + centered (NOT full-width) so the control only uses the width it needs, like the iOS app. Wrapped in
    // a scale-down FittedBox: on narrow screens the 6 segments can exceed the available width, which makes
    // CupertinoSlidingSegmentedControl compute a negative per-segment width and crash (BoxConstraints NOT NORMALIZED,
    // w=-0.8). FittedBox lays the control out at its natural (unbounded) width, then scales it down to fit, so the
    // per-segment width never goes negative.
    return Center(
      child: FittedBox(
        fit: BoxFit.scaleDown,
        child: CupertinoSlidingSegmentedControl<YOLOTask>(
          groupValue: tasks.contains(currentTask) ? currentTask : tasks.first,
          // black @ 70% — matches `toolbar.backgroundColor = .black.withAlphaComponent(0.7)` from setupToolbar.
          backgroundColor: Colors.black.withValues(alpha: 0.7),
          // selected thumb at 18% white, matching the iOS lensControl selectedSegmentTintColor.
          thumbColor: Colors.white.withValues(alpha: 0.18),
          padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 2),
          onValueChanged: (task) {
            if (task != null) onTaskChanged(task);
          },
          children: {
            for (final task in tasks)
              task: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 3),
                child: Text(
                  task.shortLabel,
                  style: TextStyle(
                    color: Colors.white,
                    // 11pt + tighter padding keeps the task strip compact on narrow phones.
                    fontSize: 11,
                    fontWeight: task == currentTask
                        ? FontWeight.w600
                        : FontWeight.w400,
                  ),
                ),
              ),
          },
        ),
      ),
    );
  }
}
