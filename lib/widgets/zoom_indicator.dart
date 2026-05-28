// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Two-line zoom HUD: numeric zoom factor on top, human-readable lens label
/// below (e.g. `Wide camera`, `Ultra wide camera`). Sits above the lens
/// picker in [YOLOShowcase].
class ZoomIndicator extends StatelessWidget {
  /// Current native zoom factor, e.g. `0.5` or `2.34`.
  final double currentZoom;

  /// Lens label sourced from `LensInfo.label`. May be empty.
  final String lensLabel;

  const ZoomIndicator({
    super.key,
    required this.currentZoom,
    required this.lensLabel,
  });

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '${currentZoom.toStringAsFixed(2)}x',
          style: textTheme.titleLarge?.copyWith(color: Colors.white),
        ),
        if (lensLabel.isNotEmpty)
          Text(
            lensLabel,
            style: textTheme.bodySmall?.copyWith(
              color: Colors.white.withValues(alpha: 0.7),
            ),
          ),
      ],
    );
  }
}
