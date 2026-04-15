// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_task.dart';

enum ModelFamily { yolo11, yolo26 }

enum ModelType {
  detect('yolo11n', ModelFamily.yolo11, YOLOTask.detect),
  detect26('yolo26n', ModelFamily.yolo26, YOLOTask.detect),
  segment('yolo11n-seg', ModelFamily.yolo11, YOLOTask.segment),
  segment26('yolo26n-seg', ModelFamily.yolo26, YOLOTask.segment),
  classify('yolo11n-cls', ModelFamily.yolo11, YOLOTask.classify),
  classify26('yolo26n-cls', ModelFamily.yolo26, YOLOTask.classify),
  pose('yolo11n-pose', ModelFamily.yolo11, YOLOTask.pose),
  pose26('yolo26n-pose', ModelFamily.yolo26, YOLOTask.pose),
  obb('yolo11n-obb', ModelFamily.yolo11, YOLOTask.obb),
  obb26('yolo26n-obb', ModelFamily.yolo26, YOLOTask.obb);

  final String modelName;

  final ModelFamily family;

  final YOLOTask task;

  const ModelType(this.modelName, this.family, this.task);

  static ModelType forSelection(ModelFamily family, YOLOTask task) => values
      .firstWhere((model) => model.family == family && model.task == task);
}

enum SliderType { none, numItems, confidence, iou }

extension ModelFamilyDisplay on ModelFamily {
  String get label => switch (this) {
    ModelFamily.yolo11 => 'YOLO11',
    ModelFamily.yolo26 => 'YOLO26',
  };
}

extension YOLOTaskDisplay on YOLOTask {
  String get label => name.toUpperCase();
}
