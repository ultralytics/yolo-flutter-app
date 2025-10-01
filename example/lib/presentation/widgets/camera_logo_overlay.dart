// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../controllers/camera_inference_controller.dart';

/// Center logo overlay widget
class CameraLogoOverlay extends StatelessWidget {
  const CameraLogoOverlay({
    super.key,
    required this.controller,
    required this.isLandscape,
  });

  final CameraInferenceController controller;
  final bool isLandscape;

  @override
  Widget build(BuildContext context) {
    if (controller.modelPath == null || controller.isModelLoading) {
      return const SizedBox.shrink();
    }

    return Positioned.fill(
      child: IgnorePointer(
        child: Align(
          alignment: Alignment.center,
          child: FractionallySizedBox(
            widthFactor: isLandscape ? 0.3 : 0.5,
            heightFactor: isLandscape ? 0.3 : 0.5,
            child: Image.asset(
              'assets/logo.png',
              color: Colors.white.withValues(alpha: 0.4),
            ),
          ),
        ),
      ),
    );
  }
}
