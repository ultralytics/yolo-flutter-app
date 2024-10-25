// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/foundation.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_interface.dart';

/// The state of the camera
class UltralyticsYoloCameraValue {
  /// Constructor to create an instance of [UltralyticsYoloCameraValue]
  UltralyticsYoloCameraValue({
    required this.lensDirection,
    required this.strokeWidth,
    required this.deferredProcessing,
  });

  /// The direction of the camera lens
  final int lensDirection;

  /// The width of the stroke used to draw the bounding boxes
  final double strokeWidth;

  /// Whether the processing of the frames should be deferred (android only)
  final bool deferredProcessing;

  /// Creates a copy of this [UltralyticsYoloCameraValue] but with
  /// the given fields
  UltralyticsYoloCameraValue copyWith({
    int? lensDirection,
    double? strokeWidth,
    bool? deferredProcessing,
  }) => UltralyticsYoloCameraValue(
    lensDirection: lensDirection ?? this.lensDirection,
    strokeWidth: strokeWidth ?? this.strokeWidth,
    deferredProcessing: deferredProcessing ?? this.deferredProcessing,
  );
}

/// ValueNotifier that holds the state of the camera
class UltralyticsYoloCameraController
    extends ValueNotifier<UltralyticsYoloCameraValue> {
  /// Constructor to create an instance of [UltralyticsYoloCameraController]
  UltralyticsYoloCameraController({bool deferredProcessing = false})
    : super(
        UltralyticsYoloCameraValue(
          lensDirection: 1,
          strokeWidth: 2.5,
          deferredProcessing: deferredProcessing,
        ),
      );

  final _ultralyticsYoloPlatform = UltralyticsYoloPlatform.instance;

  /// Toggles the direction of the camera lens
  Future<void> toggleLensDirection() async {
    try {
      // Update state first to show loading state if needed
      final newLensDirection = value.lensDirection == 0 ? 1 : 0;
      value = value.copyWith(lensDirection: newLensDirection);

      // Request camera switch
      final result = await _ultralyticsYoloPlatform.setLensDirection(
        newLensDirection,
      );

      if (result != 'Success') {
        // Revert state if failed
        value = value.copyWith(lensDirection: value.lensDirection == 0 ? 1 : 0);
        throw Exception('Failed to switch camera: $result');
      }
    } catch (e) {
      // Handle errors and revert state
      value = value.copyWith(lensDirection: value.lensDirection == 0 ? 1 : 0);
      rethrow;
    }
  }

  /// Sets the width of the stroke used to draw the bounding boxes
  void setStrokeWidth(double strokeWidth) {
    value = value.copyWith(strokeWidth: strokeWidth);
  }

  /// Closes the camera
  Future<void> closeCamera() async {
    await _ultralyticsYoloPlatform.closeCamera();
  }

  /// Starts the camera
  Future<void> startCamera() async {
    await _ultralyticsYoloPlatform.startCamera();
  }

  /// Captures the camera
  Future<Uint8List?> captureCamera({int timeoutSec = 3}) async {
    return _ultralyticsYoloPlatform.captureCamera(timeoutSec);
  }

  /// Stops the camera
  Future<void> pauseLivePrediction() async {
    await _ultralyticsYoloPlatform.pauseLivePrediction();
  }
}
