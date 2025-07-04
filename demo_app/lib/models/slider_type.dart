// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Enum representing different slider types for threshold adjustments.
///
/// Each slider type corresponds to a different parameter that can be adjusted
/// during inference to control the model's behavior.
enum SliderType {
  /// No active slider
  none,

  /// Slider for maximum number of detections
  numItems,

  /// Slider for confidence threshold
  confidence,

  /// Slider for IoU (Intersection over Union) threshold
  iou,
}
