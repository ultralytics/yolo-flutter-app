// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// iOS-style model load/download status shown below the model selector.
///
/// The status is intentionally non-interactive. A DNS-stalled or slow model download should keep the current camera
/// usable and leave the user able to choose another model.
class ModelLoadingStatus extends StatelessWidget {
  const ModelLoadingStatus({
    super.key,
    this.statusText,
    this.progress,
    this.errorMessage,
  });

  final String? statusText;
  final double? progress;
  final String? errorMessage;

  @override
  Widget build(BuildContext context) {
    final message = errorMessage ?? statusText;
    if (message == null || message.isEmpty) return const SizedBox.shrink();

    final isError = errorMessage != null;
    final normalizedProgress = progress?.clamp(0.0, 1.0);

    return IgnorePointer(
      child: Padding(
        padding: const EdgeInsets.only(top: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (!isError)
              SizedBox(
                width: 200,
                height: 2,
                child: LinearProgressIndicator(
                  value: normalizedProgress,
                  color: Colors.white,
                  backgroundColor: Colors.white.withValues(alpha: 0.28),
                ),
              ),
            if (!isError) const SizedBox(height: 8),
            ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 280),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: isError
                      ? Colors.black.withValues(alpha: 0.72)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(8),
                  border: isError
                      ? Border.all(color: Colors.white.withValues(alpha: 0.18))
                      : null,
                ),
                child: Padding(
                  padding: EdgeInsets.symmetric(
                    horizontal: isError ? 10 : 0,
                    vertical: isError ? 8 : 0,
                  ),
                  child: Text(
                    message,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: isError
                          ? Colors.white
                          : Colors.white.withValues(alpha: 0.72),
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
