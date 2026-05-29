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

  const ThresholdSliderRow({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
    this.divisions,
    this.isInt = false,
  });

  @override
  Widget build(BuildContext context) {
    final prefix = isInt ? value.round().toString() : value.toStringAsFixed(2);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$prefix $label',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w400,
          ),
        ),
        const SizedBox(height: 3),
        CupertinoSlider(
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
      ],
    );
  }
}
