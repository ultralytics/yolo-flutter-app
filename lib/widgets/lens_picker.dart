// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

/// Lens picker mirroring `yolo-ios-app/Sources/YOLO/YOLOView.swift#setupLensControl`.
///
/// Visual tokens:
///   * background: `UIColor.black.withAlphaComponent(0.38)` (line 647)
///   * selected thumb: `UIColor.white.withAlphaComponent(0.18)` (line 648)
///   * normal label: 13pt semibold white (lines 651–653)
///   * selected label: 13pt **bold** `systemYellow` (lines 654–658)
///
/// Labels render `0.5`, `1`, `2`, `4` (sub-1× lenses get one decimal). The chip whose zoom factor is closest to
/// [currentZoomFactor] is selected, so a pinch-zoom past a lens threshold visually snaps the picker.
class LensPicker extends StatelessWidget {
  /// Lenses the active camera exposes (back lens cluster on iOS, focal-length enumeration on Android). Empty for
  /// single-lens devices.
  final List<LensInfo> lenses;

  /// Current effective zoom factor; drives selection.
  final double currentZoomFactor;

  /// Invoked when the user taps a lens chip.
  final ValueChanged<LensInfo> onLensSelected;

  const LensPicker({
    super.key,
    required this.lenses,
    required this.currentZoomFactor,
    required this.onLensSelected,
  });

  static const LensInfo _defaultLens = LensInfo(
    zoomFactor: 1,
    label: 'Default',
  );

  @override
  Widget build(BuildContext context) {
    final effectiveLenses = lenses.isEmpty ? const [_defaultLens] : lenses;
    final selected = _closestLens(effectiveLenses, currentZoomFactor);

    return SizedBox(
      width: double.infinity,
      child: CupertinoSlidingSegmentedControl<double>(
        groupValue: selected.zoomFactor,
        backgroundColor: Colors.black.withValues(alpha: 0.38),
        thumbColor: Colors.white.withValues(alpha: 0.18),
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
        onValueChanged: (zoom) {
          if (zoom == null) return;
          final picked = effectiveLenses.firstWhere(
            (l) => l.zoomFactor == zoom,
            orElse: () => selected,
          );
          onLensSelected(picked);
        },
        children: {
          for (final lens in effectiveLenses)
            lens.zoomFactor: Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Text(
                _formatZoom(lens),
                style: TextStyle(
                  // systemYellow when selected; white otherwise — matches the iOS reference exactly.
                  color: lens.zoomFactor == selected.zoomFactor
                      ? CupertinoColors.systemYellow
                      : Colors.white,
                  fontSize: 13,
                  fontWeight: lens.zoomFactor == selected.zoomFactor
                      ? FontWeight.w700
                      : FontWeight.w600,
                ),
              ),
            ),
        },
      ),
    );
  }

  static String _formatZoom(LensInfo lens) {
    return lens.zoomFactor < 1
        ? lens.zoomFactor.toStringAsFixed(1)
        : lens.zoomFactor.toStringAsFixed(0);
  }

  static LensInfo _closestLens(List<LensInfo> lenses, double zoom) {
    var best = lenses.first;
    var bestDelta = (best.zoomFactor - zoom).abs();
    for (final lens in lenses.skip(1)) {
      final delta = (lens.zoomFactor - zoom).abs();
      if (delta < bestDelta) {
        best = lens;
        bestDelta = delta;
      }
    }
    return best;
  }
}
