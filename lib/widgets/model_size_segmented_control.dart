// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

/// Picks the active YOLO26 model size using a `CupertinoSlidingSegmentedControl` styled to match `yolo-ios-app`'s
/// `Main.storyboard` modelSegmentedControl. Chips read `nano / small / medium / large / xlarge` (the storyboard
/// titles); sizes not yet on disk are dimmed to signal a download-on-tap. When [downloadingSize] matches a chip a thin
/// [LinearProgressIndicator] tracks [downloadFraction] under the label.
///
/// Visual tokens: translucent black background, 18% white selected thumb, 13pt system-weight labels in white (light
/// weight bump when active).
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
    final visibleSizes = _sizes
        .where(supportedSizes.contains)
        .toList(growable: false);
    if (visibleSizes.isEmpty) return const SizedBox.shrink();
    final effectiveCurrent = visibleSizes.contains(currentSize)
        ? currentSize
        : visibleSizes.first;

    // CupertinoSlidingSegmentedControl asserts it has >= 2 segments. If a host limits official assets to one size, show
    // one static chip instead of crashing with a red error screen.
    if (visibleSizes.length < 2) {
      return Center(
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.18),
            borderRadius: BorderRadius.circular(8),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
          child: Text(
            'YOLO26$effectiveCurrent',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        // Android can deliver an initial zero-width warm-up layout before viewport metrics arrive. A full-width
        // CupertinoSlidingSegmentedControl under those constraints computes a negative per-segment width and prevents
        // the first Flutter frame from painting. Reserve the row height until the real width arrives.
        if (constraints.hasBoundedWidth && constraints.maxWidth <= 0) {
          return const SizedBox(height: 32);
        }

        final control = CupertinoSlidingSegmentedControl<String>(
          groupValue: effectiveCurrent,
          backgroundColor: Colors.black.withValues(alpha: 0.7),
          thumbColor: Colors.white.withValues(alpha: 0.18),
          padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 2),
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
        );

        // Full-width when there is enough room, matching the iOS model row. On constrained layouts, lay the control
        // out at its natural width and scale down so Cupertino never receives impossible per-segment constraints.
        if (!constraints.hasBoundedWidth || constraints.maxWidth >= 280) {
          return SizedBox(width: double.infinity, child: control);
        }

        return Center(
          child: FittedBox(fit: BoxFit.scaleDown, child: control),
        );
      },
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
    // Matches the iOS app's modelSegmentedControl: downloaded sizes read `YOLO26<size>`; sizes not yet on disk get a
    // `↓` download-on-tap prefix. FittedBox keeps a long title from clipping its (equal-width) segment.
    final percent = fraction == null
        ? null
        : '${(fraction!.clamp(0.0, 0.99) * 100).round()}%';
    final label = isDownloading && percent != null
        ? '$percent YOLO26$size'
        : isAvailable
        ? 'YOLO26$size'
        : '↓ YOLO26$size';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Text(
              label,
              maxLines: 1,
              softWrap: false,
              style: TextStyle(
                color: Colors.white,
                // 11pt so the longer `↓ YOLO26x` labels fit each segment without the FittedBox shrinking them.
                fontSize: 11,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
          ),
          if (isDownloading)
            Padding(
              padding: const EdgeInsets.only(top: 3),
              child: SizedBox(
                width: 36,
                height: 2,
                child: LinearProgressIndicator(
                  // Clamp to [0,1] so a fraction over 1 (or below 0) never paints an out-of-range / asserting bar.
                  value: fraction?.clamp(0.0, 1.0),
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
