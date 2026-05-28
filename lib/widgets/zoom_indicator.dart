// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Two-line zoom HUD: numeric zoom factor on top, human-readable lens label below (e.g. `Wide camera`, `Ultra wide
/// camera`). Sits above the lens picker in [YOLOShowcase].
///
/// Styling pulled from `yolo-ios-app/Sources/YOLO/YOLOView.swift#setupUI`:
///   * `labelZoom.font = UIFont.systemFont(ofSize: 12, weight: .semibold)` (line 597) — numeric reading.
///   * `lensCaptionLabel.font = UIFont.systemFont(ofSize: 11, weight: .medium)` (line 665) — caption.
///   * `lensCaptionLabel.textColor = UIColor.white.withAlphaComponent(0.78)` (line 664).
class ZoomIndicator extends StatelessWidget {
  /// Current effective zoom factor, e.g. `0.5` or `2.34`.
  final double currentZoom;

  /// Lens label sourced from `LensInfo.label`. May be empty.
  final String lensLabel;

  const ZoomIndicator({super.key, required this.currentZoom, required this.lensLabel});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '${currentZoom.toStringAsFixed(2)}x',
          style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
        ),
        if (lensLabel.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 1),
            child: Text(
              lensLabel,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.78),
                fontSize: 11,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
      ],
    );
  }
}
