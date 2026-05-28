// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// One labeled slider row — the prefix (`0.25 Confidence Threshold`) shows
/// the live value with the label baked in.
///
/// Style mirrors the iOS showcase: white track + thumb, faded gray inactive
/// track. `isInt` formats the prefix as a whole number (numItems) rather than
/// a two-decimal float.
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
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            '$prefix $label',
            style: Theme.of(
              context,
            ).textTheme.bodyMedium?.copyWith(color: Colors.white),
          ),
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: Colors.white,
            thumbColor: Colors.white,
            inactiveTrackColor: Colors.white.withValues(alpha: 0.3),
            overlayColor: Colors.white.withValues(alpha: 0.12),
          ),
          child: Slider(
            value: value.clamp(min, max),
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }
}
