// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Top HUD: model name (`YOLO26n`) over an FPS / inference-time line.
///
/// Centered, matching `yolo-ios-app/Sources/UltralyticsYOLO/YOLOView.swift#setupUI`:
///   * `labelName.font = UIFont.preferredFont(forTextStyle: .title1)` — iOS `title1` ≈ 28pt regular.
///   * `labelFPS.font = UIFont.preferredFont(forTextStyle: .body)` — iOS `body` ≈ 17pt regular.
///   * Both `textAlignment = .center` and `textColor = .white`.
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          modelName,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.w400,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          '${fps.toStringAsFixed(1)} FPS - ${inferenceMs.toStringAsFixed(1)} ms',
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 17,
            fontWeight: FontWeight.w400,
          ),
        ),
      ],
    );
  }
}
