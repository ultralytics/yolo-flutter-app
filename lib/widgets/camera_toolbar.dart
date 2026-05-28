// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Bottom action toolbar (play/pause, switch camera, share). Implemented as a
/// row of `IconButton.filledTonal` inside a translucent pill — the M3 stand-in
/// for the iOS `UIToolbar` blur bar.
class CameraToolbar extends StatelessWidget {
  /// Inference paused state; controls the play/pause icon.
  final bool isPaused;

  /// Fires when the user taps the play/pause button.
  final VoidCallback onPlayPause;

  /// Fires on the camera-flip button.
  final VoidCallback onSwitchCamera;

  /// Fires on the share button (capture + system share / save).
  final VoidCallback onShare;

  const CameraToolbar({
    super.key,
    required this.isPaused,
    required this.onPlayPause,
    required this.onSwitchCamera,
    required this.onShare,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton.filledTonal(
            onPressed: onPlayPause,
            icon: Icon(isPaused ? Icons.play_arrow : Icons.pause),
            tooltip: isPaused ? 'Resume' : 'Pause',
          ),
          const SizedBox(width: 8),
          IconButton.filledTonal(
            onPressed: onSwitchCamera,
            icon: const Icon(Icons.cameraswitch),
            tooltip: 'Switch camera',
          ),
          const SizedBox(width: 8),
          IconButton.filledTonal(
            onPressed: onShare,
            icon: const Icon(Icons.ios_share),
            tooltip: 'Share',
          ),
        ],
      ),
    );
  }
}
