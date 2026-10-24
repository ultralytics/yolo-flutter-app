// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

/// Lens picker mirroring `yolo-ios-app/Sources/UltralyticsYOLO/YOLOView.swift#setupLensControl`.
///
/// Visual tokens:
///   * background: `UIColor.black.withAlphaComponent(0.38)`
///   * selected thumb: `UIColor.white.withAlphaComponent(0.18)`
///   * normal label: 13pt semibold white
///   * selected label: 13pt **bold** `systemYellow`
///
/// Labels render `0.5`, `1`, `2`, `4` (sub-1× lenses get one decimal). The chip whose zoom factor is closest to
/// [currentZoomFactor] is selected, so a pinch-zoom past a lens threshold visually snaps the picker.
class LensPicker extends StatelessWidget {
  /// Lenses the active camera exposes (back lens cluster on iOS, focal-length enumeration on Android).
  final List<LensInfo> lenses;

  /// Current effective zoom factor; drives selection.
  final double currentZoomFactor;

  /// Invoked when the user taps a lens chip.
  final ValueChanged<LensInfo> onLensSelected;

  /// Optional control rendered immediately to the right of the lens pill (e.g. a torch toggle), sharing the same
  /// centered row so it sits directly next to the zoom options.
  final Widget? trailing;

  const LensPicker({
    super.key,
    required this.lenses,
    required this.currentZoomFactor,
    required this.onLensSelected,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    final pill = _pill(context);
    if (pill == null && trailing == null) return const SizedBox.shrink();
    if (trailing == null) return Center(child: pill);
    if (pill == null) return Center(child: trailing);
    // Keep the lens pill centered on screen: balance the trailing control (torch) on the right with an invisible
    // copy of equal width on the left, so the zoom options stay centered while the torch sits directly next to them.
    return Center(
      child: FittedBox(
        fit: BoxFit.scaleDown,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Opacity(opacity: 0, child: IgnorePointer(child: trailing)),
            const SizedBox(width: 6),
            pill,
            const SizedBox(width: 6),
            trailing!,
          ],
        ),
      ),
    );
  }

  /// The lens pill itself (single chip or sliding segmented control), un-centered; `null` when there are no lenses.
  Widget? _pill(BuildContext context) {
    if (lenses.isEmpty) return null;
    final selected = _closestLens(lenses, currentZoomFactor);
    if (lenses.length == 1) {
      return Semantics(
        button: true,
        selected: true,
        label: selected.label.isNotEmpty
            ? selected.label
            : '${_formatZoom(selected)}x zoom',
        child: GestureDetector(
          onTap: () => onLensSelected(selected),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.38),
              borderRadius: BorderRadius.circular(8),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 9),
            child: Text(
              _formatZoom(selected),
              style: const TextStyle(
                color: CupertinoColors.systemYellow,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ),
      );
    }

    // Content-hug pill like the iOS app, not a screen-wide bar.
    return CameraSegmentedControl<LensInfo>(
      values: lenses,
      selected: selected,
      label: _formatZoom,
      // Name each chip for screen readers (the bare `0.5`/`1`/`2` glyphs are ambiguous out of context).
      semanticsLabel: (lens) =>
          lens.label.isNotEmpty ? lens.label : '${_formatZoom(lens)}x zoom',
      onSelected: onLensSelected,
    );
  }

  static String _formatZoom(LensInfo lens) {
    return lens.zoomFactor < 1
        ? lens.zoomFactor.toStringAsFixed(1)
        : lens.zoomFactor.toStringAsFixed(0);
  }

  /// Mirrors `yolo-ios-app/Sources/UltralyticsYOLO/YOLOView.swift#updateSelectedLens`: pick the largest-zoom lens
  /// whose threshold is `<= zoom + 0.01`, falling back to the smallest lens when zoom is below every threshold. The
  /// previous "closest absolute delta" heuristic switched the yellow selection too early around midpoints (a 1.5x
  /// zoom on a 1×/2× picker would jump to 2x before the camera actually rebound).
  static LensInfo _closestLens(List<LensInfo> lenses, double zoom) {
    final sorted = [...lenses]
      ..sort((a, b) => a.zoomFactor.compareTo(b.zoomFactor));
    LensInfo? best;
    for (final lens in sorted) {
      if (zoom + 0.01 >= lens.zoomFactor) best = lens;
    }
    return best ?? sorted.first;
  }
}

/// Camera-overlay segmented control with the lens picker's tokens above, shared by [LensPicker] and the showcase
/// model sizes so both read as one control family. [isDimmed] values (e.g. sizes not on the device) render at 50% white
/// and stay tappable.
class CameraSegmentedControl<T extends Object> extends StatelessWidget {
  /// Segments in display order.
  final List<T> values;

  /// Selected segment, or `null` for none.
  final T? selected;

  /// Visible segment text.
  final String Function(T value) label;

  /// Screen-reader name for each segment.
  final String Function(T value) semanticsLabel;

  /// Invoked with the tapped segment.
  final ValueChanged<T> onSelected;

  /// Dims segments that are not ready yet; `null` dims none.
  final bool Function(T value)? isDimmed;

  const CameraSegmentedControl({
    super.key,
    required this.values,
    required this.selected,
    required this.label,
    required this.semanticsLabel,
    required this.onSelected,
    this.isDimmed,
  });

  @override
  Widget build(BuildContext context) {
    return CupertinoSlidingSegmentedControl<T>(
      groupValue: selected,
      backgroundColor: Colors.black.withValues(alpha: 0.38),
      thumbColor: Colors.white.withValues(alpha: 0.18),
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
      onValueChanged: (value) {
        if (value != null) onSelected(value);
      },
      children: {
        for (final value in values)
          value: Semantics(
            button: true,
            selected: value == selected,
            label: semanticsLabel(value),
            child: ExcludeSemantics(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Text(
                  label(value),
                  style: TextStyle(
                    // systemYellow when selected; white otherwise — matches the iOS reference exactly.
                    color: value == selected
                        ? CupertinoColors.systemYellow
                        : Colors.white.withValues(
                            alpha: isDimmed?.call(value) ?? false ? 0.5 : 1,
                          ),
                    fontSize: 13,
                    fontWeight: value == selected
                        ? FontWeight.w700
                        : FontWeight.w600,
                  ),
                ),
              ),
            ),
          ),
      },
    );
  }
}
