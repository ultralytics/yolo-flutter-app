// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/cupertino.dart' show CupertinoIcons;
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

  /// Replaces inference metrics while the camera is in another mode, such as QR scanning.
  final String? status;

  /// Cancels QR scanning when shown beside [status].
  final VoidCallback? onCancel;

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
    this.status,
    this.onCancel,
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
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.w400,
          ),
        ),
        const SizedBox(height: 2),
        // Both timing lines keep their height before the first result arrives (blank text, not absent
        // widgets), so the rows below don't shift down when inference starts.
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Flexible(
              child: Text(
                status ??
                    (fps > 0
                        ? '${fps.toStringAsFixed(1)} FPS - ${inferenceMs.toStringAsFixed(1)} ms'
                        : ' '),
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.w400,
                ),
              ),
            ),
            if (onCancel != null)
              IconButton(
                tooltip: 'Cancel QR scan',
                constraints: const BoxConstraints.tightFor(
                  width: 44,
                  height: 44,
                ),
                onPressed: onCancel,
                icon: const Icon(
                  CupertinoIcons.xmark,
                  color: Colors.white,
                  size: 20,
                ),
              ),
          ],
        ),
        Text(
          status == null && preMs + modelMs + postMs > 0
              ? '${preMs.toStringAsFixed(1)} pre · ${modelMs.toStringAsFixed(1)} inference · ${postMs.toStringAsFixed(1)} post'
              : ' ',
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
