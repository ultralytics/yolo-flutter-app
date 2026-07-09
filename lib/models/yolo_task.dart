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

  /// Semantic segmentation - assigns a class label to each image pixel
  semantic,

  /// Monocular depth estimation - predicts metric distance for each image pixel
  depth,

  /// Image classification - categorizes the main subject of an image
  classify,

  /// Pose estimation - detects human body keypoints and poses
  pose,

  /// Oriented Bounding Box detection - detects rotated bounding boxes for objects
  obb,
}

extension YOLOTaskParsing on YOLOTask {
  String get modelSuffix => switch (this) {
    YOLOTask.detect => '',
    YOLOTask.segment => '-seg',
    YOLOTask.semantic => '-sem',
    YOLOTask.depth => '-depth',
    YOLOTask.classify => '-cls',
    YOLOTask.pose => '-pose',
    YOLOTask.obb => '-obb',
  };

  String get shortLabel => switch (this) {
    YOLOTask.detect => 'Det',
    YOLOTask.segment => 'Seg',
    YOLOTask.semantic => 'Sem',
    YOLOTask.depth => 'Depth',
    YOLOTask.classify => 'Cls',
    YOLOTask.pose => 'Pose',
    YOLOTask.obb => 'OBB',
  };

  String get label => switch (this) {
    YOLOTask.detect => 'Detect',
    YOLOTask.segment => 'Segment',
    YOLOTask.semantic => 'Semantic',
    YOLOTask.depth => 'Depth',
    YOLOTask.classify => 'Classify',
    YOLOTask.pose => 'Pose',
    YOLOTask.obb => 'OBB',
  };

  static YOLOTask? tryParse(String? value) {
    if (value == null) return null;
    for (final task in YOLOTask.values) {
      if (task.name == value.toLowerCase()) {
        return task;
      }
    }
    return null;
  }
}
