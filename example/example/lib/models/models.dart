// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_task.dart';

enum ModelType {
  detect('yolo11n', YOLOTask.detect),
  segment('yolo11n-seg', YOLOTask.segment),
  classify('yolo11n-cls', YOLOTask.classify),
  pose('yolo11n-pose', YOLOTask.pose),
  obb('yolo11n-obb', YOLOTask.obb);

  final String modelName;

  final YOLOTask task;

  const ModelType(this.modelName, this.task);
}

enum SliderType { none, numItems, confidence, iou }
