// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// Bottom action toolbar (play/pause, switch camera, share, optional QR scan, info). Matches `yolo-ios-app/Sources/UltralyticsYOLO/YOLOView.swift`'s
/// toolbar layout:
///   * `toolbar.backgroundColor = .black.withAlphaComponent(0.7)`.
///   * `toolBarHeight: CGFloat = 66` — the toolbar is 66pt tall.
///   * `buttonHeight = toolBarHeight * 0.75 = 49.5pt` — each button is roughly square at 49.5pt.
///   * SF Symbol `pointSize: 20` — icons render at ~20pt.
///   * Buttons spread evenly across the toolbar width (`layoutToolbarButtons`).
class CameraToolbar extends StatelessWidget {
  // 66pt matches the iOS reference toolbar; Android's bar is shortened to the 44pt min touch target since it also sits
  // above the system nav bar.
  static double get height =>
      defaultTargetPlatform == TargetPlatform.android ? 44 : 66;
  static const double iconSize = 20;

  /// Inference paused state; controls the play/pause icon.
  final bool isPaused;

  /// Fires when the user taps the play/pause button.
  final VoidCallback onPlayPause;

  /// Fires on the camera-flip button.
  final VoidCallback onSwitchCamera;

  /// Fires on the share button (capture + system share / save).
  final VoidCallback onShare;

  /// Fires on the info button.
  final VoidCallback onInfo;

  /// Starts QR scanning through the current camera stream, or cancels the active scan.
  final VoidCallback? onScanQR;

  /// Highlights the QR button while scanning; tapping it again cancels.
  final bool isScanningQR;

  const CameraToolbar({
    super.key,
    required this.isPaused,
    required this.onPlayPause,
    required this.onSwitchCamera,
    required this.onShare,
    required this.onInfo,
    this.onScanQR,
    this.isScanningQR = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      child: ColoredBox(
        color: Colors.black.withValues(alpha: 0.7),
        child: Row(
          children: [
            _ToolbarButton(
              // play.fill / pause.fill on iOS → matching Cupertino glyphs for the closest visual to SF Symbols.
              icon: isPaused
                  ? CupertinoIcons.play_fill
                  : CupertinoIcons.pause_fill,
              onPressed: onPlayPause,
              semanticLabel: isPaused ? 'Resume' : 'Pause',
            ),
            _ToolbarButton(
              icon: CupertinoIcons.camera_rotate,
              onPressed: onSwitchCamera,
              semanticLabel: 'Switch camera',
            ),
            _ToolbarButton(
              icon: CupertinoIcons.share,
              onPressed: onShare,
              semanticLabel: 'Share',
            ),
            if (onScanQR != null)
              _ToolbarButton(
                icon: CupertinoIcons.qrcode_viewfinder,
                onPressed: onScanQR!,
                semanticLabel: 'Scan QR',
                selected: isScanningQR,
              ),
            _ToolbarButton(
              icon: CupertinoIcons.info,
              onPressed: onInfo,
              semanticLabel: 'Ultralytics',
            ),
          ],
        ),
      ),
    );
  }
}

class _ToolbarButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onPressed;
  final String semanticLabel;

  /// Toggle state for mode buttons such as QR scanning; null for plain actions.
  final bool? selected;

  const _ToolbarButton({
    required this.icon,
    required this.onPressed,
    required this.semanticLabel,
    this.selected,
  });

  static const _activeColor = Color(0xFFFFD60A);

  @override
  Widget build(BuildContext context) {
    final active = selected ?? false;
    return Expanded(
      child: Semantics(
        selected: selected,
        child: CupertinoButton(
          padding: EdgeInsets.zero,
          onPressed: onPressed,
          child: DecoratedBox(
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: active ? _activeColor.withValues(alpha: 0.16) : null,
            ),
            child: SizedBox.square(
              dimension: 44,
              child: Icon(
                icon,
                color: active ? _activeColor : Colors.white,
                size: CameraToolbar.iconSize,
                semanticLabel: semanticLabel,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
