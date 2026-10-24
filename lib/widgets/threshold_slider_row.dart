// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

/// One labeled slider row: a caption spanning the slider's width, with the name (`Confidence`) on the left and the
/// live value (`0.25`) right-aligned in tabular figures, over the slider.
///
/// Styling mirrors the iOS app's threshold rows:
///   * 15pt (`subheadline`) caption, name in 62% white, value in white medium weight.
///   * `slider.minimumTrackTintColor = .white`.
///   * `slider.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)`.
/// `CupertinoSlider` is used over Material's `Slider` because the iOS-style thin track with a circular thumb is what
/// the showcase reference uses; `SliderTheme` can't reshape Material's wider track to match without re-painting.
class ThresholdSliderRow extends StatelessWidget {
  /// Slider name shown left of the value (e.g. `Confidence`).
  final String label;

  /// Current slider value.
  final double value;

  /// Slider minimum.
  final double min;

  /// Slider maximum.
  final double max;

  /// Discrete divisions, or `null` for a continuous slider.
  final int? divisions;

  /// Invoked on every drag tick.
  final ValueChanged<double> onChanged;

  /// When `true` the value is rendered with no decimal places (numItems).
  final bool isInt;

  /// Fraction of the available width the slider and its caption occupy (iOS `sliderWidth = width * 0.46`).
  final double sliderWidthFactor;

  const ThresholdSliderRow({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
    this.divisions,
    this.isInt = false,
    this.sliderWidthFactor = 1.0,
  });

  @override
  Widget build(BuildContext context) {
    final valueText = isInt
        ? value.round().toString()
        : value.toStringAsFixed(2);
    final leftFlex = (sliderWidthFactor.clamp(0.05, 1.0) * 100).round();
    final rightFlex = 100 - leftFlex;
    final slider = Semantics(
      // CupertinoSlider carries no descriptive label; name it + read out the live value so screen readers announce
      // "Confidence, 0.25" instead of an anonymous adjustable.
      label: label,
      value: valueText,
      slider: true,
      child: CupertinoSlider(
        value: value.clamp(min, max),
        min: min,
        max: max,
        divisions: divisions,
        // iOS YOLOView uses pure white for the filled portion; CupertinoSlider's default `activeColor` is the system
        // accent, which is blue on iOS. Force white to match the reference.
        activeColor: Colors.white,
        thumbColor: Colors.white,
        onChanged: onChanged,
      ),
    );
    return Row(
      // Left-justified: the caption and slider hug the left edge (matching yolo-ios-app); the Spacer fills the rest.
      children: [
        Expanded(
          flex: leftFlex,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisSize: MainAxisSize.min,
            children: [
              // CupertinoSlider insets its track 8 px, so the caption does too and spans exactly the track.
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.62),
                          fontSize: 15,
                        ),
                      ),
                    ),
                    Text(
                      valueText,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ],
                ),
              ),
              Transform.translate(offset: const Offset(0, -6), child: slider),
            ],
          ),
        ),
        if (rightFlex > 0) Spacer(flex: rightFlex),
      ],
    );
  }
}
