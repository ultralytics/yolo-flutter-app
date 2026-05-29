// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/widgets/model_size_segmented_control.dart';

void main() {
  // CupertinoSlidingSegmentedControl asserts >= 2 segments; on Android only the `n` size ships, which previously threw
  // a red-screen build error. The control must degrade to a single static chip instead.
  testWidgets('single supported size renders a chip, does not throw', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ModelSizeSegmentedControl(
            currentSize: 'n',
            availableSizes: const {'n'},
            supportedSizes: const {'n'},
            onSizeChanged: (_) {},
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    expect(find.text('YOLO26n'), findsOneWidget);
  });

  testWidgets('multiple supported sizes render the segmented control', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ModelSizeSegmentedControl(
            currentSize: 'n',
            availableSizes: const {'n'},
            supportedSizes: const {'n', 's', 'm', 'l', 'x'},
            onSizeChanged: (_) {},
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    expect(find.text('YOLO26n'), findsOneWidget);
  });
}
