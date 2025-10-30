// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../../models/models.dart';

/// A slider widget for adjusting threshold values
class ThresholdSlider extends StatelessWidget {
  const ThresholdSlider({
    super.key,
    required this.activeSlider,
    required this.confidenceThreshold,
    required this.iouThreshold,
    required this.numItemsThreshold,
    required this.onValueChanged,
    required this.isLandscape,
  });

  final SliderType activeSlider;
  final double confidenceThreshold;
  final double iouThreshold;
  final int numItemsThreshold;
  final ValueChanged<double> onValueChanged;
  final bool isLandscape;

  @override
  Widget build(BuildContext context) {
    if (activeSlider == SliderType.none) {
      return const SizedBox.shrink();
    }

    return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: isLandscape ? 16 : 24,
          vertical: isLandscape ? 8 : 12,
        ),
        color: Colors.black.withValues(alpha: 0.8),
        child: SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: Colors.yellow,
            inactiveTrackColor: Colors.white.withValues(alpha: 0.3),
            thumbColor: Colors.yellow,
            overlayColor: Colors.yellow.withValues(alpha: 0.2),
          ),
          child: Slider(
            value: _getSliderValue(),
            min: _getSliderMin(),
            max: _getSliderMax(),
            divisions: _getSliderDivisions(),
            label: _getSliderLabel(),
            onChanged: onValueChanged,
          ),
        ),
      ),
    );
  }

  double _getSliderValue() => switch (activeSlider) {
    SliderType.numItems => numItemsThreshold.toDouble(),
    SliderType.confidence => confidenceThreshold,
    SliderType.iou => iouThreshold,
    _ => 0,
  };

  double _getSliderMin() => activeSlider == SliderType.numItems ? 5 : 0.1;
  double _getSliderMax() => activeSlider == SliderType.numItems ? 50 : 0.9;
  int _getSliderDivisions() => activeSlider == SliderType.numItems ? 9 : 8;
  String _getSliderLabel() => switch (activeSlider) {
    SliderType.numItems => '$numItemsThreshold',
    SliderType.confidence => confidenceThreshold.toStringAsFixed(1),
    SliderType.iou => iouThreshold.toStringAsFixed(1),
    _ => '',
  };
}
