// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_task.dart

/// Represents the different types of tasks that can be performed by YOLO models.
///
/// YOLO models can be trained for various computer vision tasks, and this enum
/// allows specifying which task a particular model is designed to perform.
enum YOLOTask {
  /// Object detection - identifies objects and their locations with bounding boxes
  detect,

  /// Instance segmentation - provides pixel-level masks for detected objects
  segment,

  /// Image classification - categorizes the main subject of an image
  classify,

  /// Pose estimation - detects human body keypoints and poses
  pose,

  /// Oriented Bounding Box detection - detects rotated bounding boxes for objects
  obb,
}
