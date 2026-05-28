// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Top-left HUD: model name (`YOLO26n`) over an FPS / inference-time line.
class PerformanceLabel extends StatelessWidget {
  /// Active model display name, e.g. `YOLO26n`.
  final String modelName;

  /// Smoothed frames-per-second from [YOLOPerformanceMetrics].
  final double fps;

  /// Per-inference time in milliseconds.
  final double inferenceMs;

  const PerformanceLabel({
    super.key,
    required this.modelName,
    required this.fps,
    required this.inferenceMs,
  });

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          modelName,
          style: textTheme.titleLarge?.copyWith(color: Colors.white),
        ),
        Text(
          '${fps.toStringAsFixed(1)} FPS - ${inferenceMs.toStringAsFixed(1)} ms',
          style: textTheme.bodyMedium?.copyWith(
            color: Colors.white.withValues(alpha: 0.7),
          ),
        ),
      ],
    );
  }
}
