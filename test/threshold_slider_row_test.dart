// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/widgets/threshold_slider_row.dart';

void main() {
  testWidgets(
    'ThresholdSliderRow is left-justified (not centered) inside a center-aligned parent',
    (tester) async {
      // Mirrors the showcase: an outer Column with the DEFAULT (center) crossAxisAlignment, a 20pt side padding, and an
      // inner start-aligned Column. The old FractionallySizedBox shrink-wrapped here and got centered; the Row fill must
      // keep the caption + slider hard against the left.
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: Column(
              children: [
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [_Probe()],
                  ),
                ),
              ],
            ),
          ),
        ),
      );

      final screenWidth = tester.getSize(find.byType(Scaffold)).width;
      final sliderRect = tester.getRect(find.byType(CupertinoSlider));
      final captionRect = tester.getRect(
        find.text('0.25 Confidence Threshold'),
      );

      // Slider must live in the left ~46%: its left edge near the 20pt pad, its center well left of screen center.
      expect(
        sliderRect.left,
        lessThan(screenWidth * 0.12),
        reason:
            'slider left edge should hug the left padding, got ${sliderRect.left} of $screenWidth',
      );
      expect(
        sliderRect.center.dx,
        lessThan(screenWidth * 0.45),
        reason:
            'slider should be left-justified, got center ${sliderRect.center.dx} of $screenWidth',
      );
      // Caption must also start at the left, not be centered.
      expect(
        captionRect.left,
        lessThan(screenWidth * 0.12),
        reason:
            'caption should start at the left, got ${captionRect.left} of $screenWidth',
      );
    },
  );
}

class _Probe extends StatelessWidget {
  const _Probe();
  @override
  Widget build(BuildContext context) {
    return ThresholdSliderRow(
      label: 'Confidence Threshold',
      value: 0.25,
      min: 0,
      max: 1,
      onChanged: (_) {},
      sliderWidthFactor: 0.46,
    );
  }
}
