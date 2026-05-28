// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

/// Picks the active YOLO26 model size (n/s/m/l/x) using a `CupertinoSlidingSegmentedControl` styled to match
/// `yolo-ios-app`'s storyboard `modelSegmentedControl`. Downloaded chips read `YOLO26<size>`; missing chips are
/// prefixed `⤓ ` to signal a download-on-tap. When [downloadingSize] matches a chip a thin
/// [LinearProgressIndicator] tracks [downloadFraction] under the label.
///
/// Visual tokens pulled from `yolo-ios-app/YOLOiOSApp/ModelSelectionManager.swift` + `YOLOView.swift#setupLensControl`
/// (the segmented controls share styling): translucent black background, 18% white selected thumb, 13pt semibold
/// labels in white (bold + larger when active).
class ModelSizeSegmentedControl extends StatelessWidget {
  /// Currently-selected size (one of `n s m l x`).
  final String currentSize;

  /// Sizes already present on-disk; missing sizes still appear with the `⤓` prefix so the user can tap to start a
  /// download.
  final Set<String> availableSizes;

  /// Sizes the resolver can fetch on the current platform. Sizes outside this set are hidden from the segmented control
  /// so users can't tap a chip that has no asset to download. Defaults to all five.
  final Set<String> supportedSizes;

  /// Invoked with the tapped size. Tapping a missing size is treated as a download request — the parent kicks off the
  /// resolver.
  final ValueChanged<String> onSizeChanged;

  /// Size currently being downloaded (renders a progress indicator on that chip). `null` when no download is in-flight.
  final String? downloadingSize;

  /// Fraction in `[0,1]` for the active download. `null` while indeterminate.
  final double? downloadFraction;

  const ModelSizeSegmentedControl({
    super.key,
    required this.currentSize,
    required this.availableSizes,
    required this.onSizeChanged,
    this.downloadingSize,
    this.downloadFraction,
    this.supportedSizes = const {'n', 's', 'm', 'l', 'x'},
  });

  static const List<String> _sizes = ['n', 's', 'm', 'l', 'x'];

  @override
  Widget build(BuildContext context) {
    final visibleSizes = _sizes.where(supportedSizes.contains).toList(growable: false);
    if (visibleSizes.isEmpty) return const SizedBox.shrink();
    final effectiveCurrent = visibleSizes.contains(currentSize) ? currentSize : visibleSizes.first;

    return SizedBox(
      width: double.infinity,
      child: CupertinoSlidingSegmentedControl<String>(
        groupValue: effectiveCurrent,
        backgroundColor: Colors.black.withValues(alpha: 0.7),
        thumbColor: Colors.white.withValues(alpha: 0.18),
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
        onValueChanged: (size) {
          if (size != null) onSizeChanged(size);
        },
        children: {
          for (final size in visibleSizes)
            size: _SegmentLabel(
              size: size,
              isSelected: size == effectiveCurrent,
              isAvailable: availableSizes.contains(size),
              isDownloading: downloadingSize == size,
              fraction: downloadingSize == size ? downloadFraction : null,
            ),
        },
      ),
    );
  }
}

class _SegmentLabel extends StatelessWidget {
  final String size;
  final bool isSelected;
  final bool isAvailable;
  final bool isDownloading;
  final double? fraction;

  const _SegmentLabel({
    required this.size,
    required this.isSelected,
    required this.isAvailable,
    required this.isDownloading,
    required this.fraction,
  });

  @override
  Widget build(BuildContext context) {
    final label = isAvailable ? 'YOLO26$size' : '⤓ YOLO26$size';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.white,
              // Match iOS' 13pt segmented control labels; bump weight when selected to mirror UISegmentedControl's
              // selected-state delta.
              fontSize: 13,
              fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
            ),
          ),
          if (isDownloading)
            Padding(
              padding: const EdgeInsets.only(top: 3),
              child: SizedBox(
                width: 36,
                height: 2,
                child: LinearProgressIndicator(
                  value: fraction,
                  color: Colors.white,
                  backgroundColor: Colors.white.withValues(alpha: 0.2),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
