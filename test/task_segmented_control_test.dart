// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/widgets/task_segmented_control.dart';

void main() {
  // On narrow screens the six task segments (Det/Seg/Sem/Cls/Pose/OBB) can exceed the available width, which made
  // CupertinoSlidingSegmentedControl compute a negative per-segment width and crash with
  // `BoxConstraints(w=-0.8) NOT NORMALIZED`. The FittedBox(scaleDown) wrapper must keep it laying out without throwing.
  testWidgets('does not throw when constrained to a narrow width', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Center(
            // Deliberately tight — narrower than the natural width of six segments.
            child: SizedBox(
              width: 180,
              child: TaskSegmentedControl(
                currentTask: YOLOTask.detect,
                onTaskChanged: (_) {},
              ),
            ),
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    expect(find.text('Det'), findsOneWidget);
    expect(find.text('Seg'), findsOneWidget);
    expect(find.text('Sem'), findsOneWidget);
    expect(find.text('Cls'), findsOneWidget);
    expect(find.text('Pose'), findsOneWidget);
    expect(find.text('OBB'), findsOneWidget);
  });

  testWidgets('hides the Semantic segment when showSemanticTask is false', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: TaskSegmentedControl(
            currentTask: YOLOTask.detect,
            onTaskChanged: (_) {},
            showSemanticTask: false,
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    expect(find.text('Sem'), findsNothing);
    expect(find.text('Det'), findsOneWidget);
  });
}
