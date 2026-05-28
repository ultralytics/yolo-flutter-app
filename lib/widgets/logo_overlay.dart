// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Watermark Ultralytics logotype. Bundled inside the plugin so consumers
/// don't have to copy the asset into their own app.
class LogoOverlay extends StatelessWidget {
  /// Logical-pixel width; the asset preserves aspect ratio at any size.
  final double width;

  const LogoOverlay({super.key, this.width = 120});

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: 0.9,
      child: Image.asset(
        'assets/ultralytics_yolo_logotype.png',
        package: 'ultralytics_yolo',
        width: width,
      ),
    );
  }
}
