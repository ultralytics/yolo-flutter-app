// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

/// Material 3 [SegmentedButton] over the device's available lenses.
///
/// Labels render as `0.5`, `1`, `2`, `4` (sub-1× lenses use one decimal). The
/// chip whose zoom factor is closest to [currentZoomFactor] is selected, so
/// a pinch-zoom past a lens threshold visually snaps the picker.
class LensPicker extends StatelessWidget {
  /// Lenses the active camera exposes (back lens cluster on iOS, focal-length
  /// enumeration on Android). Empty for single-lens devices.
  final List<LensInfo> lenses;

  /// Current native zoom factor; drives selection.
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

    return SegmentedButton<double>(
      segments: [
        for (final lens in effectiveLenses)
          ButtonSegment<double>(
            value: lens.zoomFactor,
            label: Text(_formatZoom(lens)),
          ),
      ],
      selected: {selected.zoomFactor},
      showSelectedIcon: false,
      onSelectionChanged: (selection) {
        if (selection.isEmpty) return;
        final picked = effectiveLenses.firstWhere(
          (l) => l.zoomFactor == selection.first,
          orElse: () => selected,
        );
        onLensSelected(picked);
      },
    );
  }

  static String _formatZoom(LensInfo lens) {
    // Sub-1× lenses (e.g. 0.5×) need one decimal so users can distinguish
    // ultra-wide from wide; 1×/2×/4× read cleaner as whole numbers.
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
