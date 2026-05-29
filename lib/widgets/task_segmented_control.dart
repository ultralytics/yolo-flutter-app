// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

/// Picks the active [YOLOTask] using a `CupertinoSlidingSegmentedControl` styled to match
/// `yolo-ios-app/Sources/YOLO/YOLOView.swift`'s storyboard-driven `UISegmentedControl` (`Det Seg Sem Cls Pose OBB`).
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

  /// When `false`, the `Semantic` segment is hidden (used on builds where no semantic models exist yet for the chosen
  /// size).
  final bool showSemanticTask;

  const TaskSegmentedControl({
    super.key,
    required this.currentTask,
    required this.onTaskChanged,
    this.showSemanticTask = true,
  });

  static const Map<YOLOTask, String> _shortLabels = {
    YOLOTask.detect: 'Det',
    YOLOTask.segment: 'Seg',
    YOLOTask.semantic: 'Sem',
    YOLOTask.classify: 'Cls',
    YOLOTask.pose: 'Pose',
    YOLOTask.obb: 'OBB',
  };

  // Canonical iOS task order (Det Seg Sem Cls Pose OBB).
  static const List<YOLOTask> _order = [
    YOLOTask.detect,
    YOLOTask.segment,
    YOLOTask.semantic,
    YOLOTask.classify,
    YOLOTask.pose,
    YOLOTask.obb,
  ];

  @override
  Widget build(BuildContext context) {
    final tasks = _order
        .where((t) => showSemanticTask || t != YOLOTask.semantic)
        .toList(growable: false);

    return SizedBox(
      width: double.infinity,
      child: CupertinoSlidingSegmentedControl<YOLOTask>(
        groupValue: tasks.contains(currentTask) ? currentTask : tasks.first,
        // black @ 70% — matches `toolbar.backgroundColor = .black.withAlphaComponent(0.7)` from setupToolbar.
        backgroundColor: Colors.black.withValues(alpha: 0.7),
        // selected thumb at 18% white, matching the iOS lensControl selectedSegmentTintColor.
        thumbColor: Colors.white.withValues(alpha: 0.18),
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
        onValueChanged: (task) {
          if (task != null) onTaskChanged(task);
        },
        children: {
          for (final task in tasks)
            task: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              child: Text(
                _shortLabels[task] ?? task.name,
                style: TextStyle(
                  color: Colors.white,
                  // iOS segmented controls use the system font — regular, with only a light weight bump when selected.
                  // The previous w700/w600 read as too heavy versus the native control.
                  fontSize: 13,
                  fontWeight: task == currentTask
                      ? FontWeight.w600
                      : FontWeight.w400,
                ),
              ),
            ),
        },
      ),
    );
  }
}
