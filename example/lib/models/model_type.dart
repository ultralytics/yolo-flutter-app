// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/yolo_task.dart';

/// Each model type corresponds to a specific YOLO task and model variant.
/// The model names follow the format 'yolo11n-{task}' where:
/// - '11n' indicates the model size (nano)
/// - {task} indicates the specific task (detect, segment, classify, pose, obb)
enum ModelType {
  /// Object detection model
  detect('yolo11n', YOLOTask.detect),

  /// Instance segmentation model
  segment('yolo11n-seg', YOLOTask.segment),

  /// Image classification model
  classify('yolo11n-cls', YOLOTask.classify),

  /// Pose estimation model
  pose('yolo11n-pose', YOLOTask.pose),

  /// Oriented bounding box detection model
  obb('yolo11n-obb', YOLOTask.obb);

  /// The name of the model file (without extension)
  final String modelName;

  /// The YOLO task type this model performs
  final YOLOTask task;

  const ModelType(this.modelName, this.task);
}
