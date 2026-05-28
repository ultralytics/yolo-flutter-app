// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Material 3 [SegmentedButton] over the five YOLO26 model sizes.
///
/// Downloaded chips read `YOLO26<size>`; missing chips are prefixed `⤓ ` to
/// signal a download-on-tap. When [downloadingSize] matches a chip a thin
/// [LinearProgressIndicator] tracks [downloadFraction] under the label.
class ModelSizeSegmentedControl extends StatelessWidget {
  /// Currently-selected size (one of `n s m l x`).
  final String currentSize;

  /// Sizes already present on-disk; missing sizes still appear with the `⤓`
  /// prefix so the user can tap to start a download.
  final Set<String> availableSizes;

  /// Sizes the resolver can fetch on the current platform. Sizes outside this
  /// set are hidden from the segmented control so users can't tap a chip
  /// that has no asset to download. Defaults to all five.
  final Set<String> supportedSizes;

  /// Invoked with the tapped size. Tapping a missing size is treated as a
  /// download request — the parent kicks off the resolver.
  final ValueChanged<String> onSizeChanged;

  /// Size currently being downloaded (renders a progress indicator on that
  /// chip). `null` when no download is in-flight.
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
    return SegmentedButton<String>(
      segments: [
        for (final size in _sizes)
          if (supportedSizes.contains(size))
            ButtonSegment<String>(
              value: size,
              label: _SegmentLabel(
                size: size,
                isAvailable: availableSizes.contains(size),
                isDownloading: downloadingSize == size,
                fraction: downloadingSize == size ? downloadFraction : null,
              ),
            ),
      ],
      selected: {currentSize},
      showSelectedIcon: false,
      // SegmentedButton clears the selection if the user re-taps the active
      // segment; preserve `currentSize` so a re-tap on the same chip still
      // routes (and so missing-chip taps that fail to switch don't unselect
      // the previously-active chip).
      onSelectionChanged: (selection) {
        final next = selection.isEmpty ? currentSize : selection.first;
        onSizeChanged(next);
      },
    );
  }
}

class _SegmentLabel extends StatelessWidget {
  final String size;
  final bool isAvailable;
  final bool isDownloading;
  final double? fraction;

  const _SegmentLabel({
    required this.size,
    required this.isAvailable,
    required this.isDownloading,
    required this.fraction,
  });

  @override
  Widget build(BuildContext context) {
    final label = isAvailable ? 'YOLO26$size' : '⤓ YOLO26$size';
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label),
        if (isDownloading)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: SizedBox(
              width: 48,
              height: 2,
              child: LinearProgressIndicator(value: fraction),
            ),
          ),
      ],
    );
  }
}
