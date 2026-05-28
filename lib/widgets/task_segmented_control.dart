// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

/// Material 3 [SegmentedButton] that picks the active [YOLOTask].
///
/// Order and labels mirror the iOS showcase (`Det Seg Sem Cls Pose OBB`).
class TaskSegmentedControl extends StatelessWidget {
  /// Currently-selected task; drives which segment is highlighted.
  final YOLOTask currentTask;

  /// Invoked with the user's selection.
  final ValueChanged<YOLOTask> onTaskChanged;

  /// When `false`, the `Semantic` segment is hidden (used on builds where
  /// no semantic models exist yet for the chosen size).
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

    return SegmentedButton<YOLOTask>(
      segments: [
        for (final task in tasks)
          ButtonSegment<YOLOTask>(
            value: task,
            label: Text(_shortLabels[task] ?? task.name),
          ),
      ],
      selected: {currentTask},
      showSelectedIcon: false,
      onSelectionChanged: (selection) {
        if (selection.isEmpty) return;
        onTaskChanged(selection.first);
      },
    );
  }
}
