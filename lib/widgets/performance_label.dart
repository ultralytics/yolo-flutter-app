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

  /// Total per-frame processing time in milliseconds (pre + inference + post).
  final double inferenceMs;

  /// Preprocessing time in milliseconds, shown in the breakdown line when > 0.
  final double preMs;

  /// Model inference time in milliseconds.
  final double modelMs;

  /// Postprocessing time in milliseconds.
  final double postMs;

  const PerformanceLabel({
    super.key,
    required this.modelName,
    required this.fps,
    required this.inferenceMs,
    this.preMs = 0,
    this.modelMs = 0,
    this.postMs = 0,
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
        if (preMs + modelMs + postMs > 0)
          Text(
            '${preMs.toStringAsFixed(1)} pre · ${modelMs.toStringAsFixed(1)} inference · ${postMs.toStringAsFixed(1)} post',
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 12,
              fontWeight: FontWeight.w400,
            ),
          ),
      ],
    );
  }
}
