// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

/// One labeled slider row — the prefix (`0.25 Confidence Threshold`) shows the live value with the label baked in.
///
/// Styling mirrors `yolo-ios-app/Sources/YOLO/YOLOView.swift#setupUI`:
///   * `labelSliderConf.font = UIFont.preferredFont(forTextStyle: .subheadline)` (lines 577/586) — `subheadline` in
///     iOS is ~15pt regular, so we use `bodyMedium`/15 with `FontWeight.w400`.
///   * `slider.minimumTrackTintColor = .white` (line 630).
///   * `slider.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)` (line 631).
/// `CupertinoSlider` is used over Material's `Slider` because the iOS-style thin track with a circular thumb is what
/// the showcase reference uses; `SliderTheme` can't reshape Material's wider track to match without re-painting.
class ThresholdSliderRow extends StatelessWidget {
  /// Slider label suffix (e.g. `Confidence Threshold`).
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

  /// Fraction of the available width the slider track occupies (iOS `sliderWidth = width * 0.46`). The caption above
  /// always spans full width so it never wraps.
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
    final prefix = isInt ? value.round().toString() : value.toStringAsFixed(2);
    final leftFlex = (sliderWidthFactor.clamp(0.05, 1.0) * 100).round();
    final rightFlex = 100 - leftFlex;
    final slider = Semantics(
      // CupertinoSlider carries no descriptive label; name it + read out the live value so screen readers announce
      // "Confidence Threshold, 0.25" instead of an anonymous adjustable. Visuals are unchanged.
      label: label,
      value: prefix,
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
    return Column(
      // Left-justified: the caption and the slider both hug the left edge (matching yolo-ios-app). The Row below
      // fills the full width, which also forces this Column to full width so the parent can't center it.
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$prefix $label',
          maxLines: 1,
          overflow: TextOverflow.visible,
          softWrap: false,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w400,
          ),
        ),
        // The slider track occupies the left `sliderWidthFactor` of the width; the Spacer fills the rest. A Row fills
        // its width (unlike FractionallySizedBox, which shrink-wrapped and let the whole row get center-aligned).
        Transform.translate(
          offset: const Offset(0, -6),
          child: Row(
            children: [
              Expanded(flex: leftFlex, child: slider),
              if (rightFlex > 0) Spacer(flex: rightFlex),
            ],
          ),
        ),
      ],
    );
  }
}
