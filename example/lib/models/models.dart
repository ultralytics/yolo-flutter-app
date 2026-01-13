// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_task.dart';

enum ModelFamily { yolo11, yolo26 }

enum ModelTask { detect, segment, classify, pose, obb }

enum ModelType {
  detect('yolo11n', YOLOTask.detect),
  detect26('yolo26n', YOLOTask.detect),
  segment('yolo11n-seg', YOLOTask.segment),
  segment26('yolo26n-seg', YOLOTask.segment),
  classify('yolo11n-cls', YOLOTask.classify),
  classify26('yolo26n-cls', YOLOTask.classify),
  pose('yolo11n-pose', YOLOTask.pose),
  pose26('yolo26n-pose', YOLOTask.pose),
  obb('yolo11n-obb', YOLOTask.obb),
  obb26('yolo26n-obb', YOLOTask.obb);

  final String modelName;
  final YOLOTask task;

  const ModelType(this.modelName, this.task);

  static ModelType forFamilyAndTask(ModelFamily family, ModelTask task) {
    switch ((family, task)) {
      case (ModelFamily.yolo11, ModelTask.detect):
        return ModelType.detect;
      case (ModelFamily.yolo11, ModelTask.segment):
        return ModelType.segment;
      case (ModelFamily.yolo11, ModelTask.classify):
        return ModelType.classify;
      case (ModelFamily.yolo11, ModelTask.pose):
        return ModelType.pose;
      case (ModelFamily.yolo11, ModelTask.obb):
        return ModelType.obb;
      case (ModelFamily.yolo26, ModelTask.detect):
        return ModelType.detect26;
      case (ModelFamily.yolo26, ModelTask.segment):
        return ModelType.segment26;
      case (ModelFamily.yolo26, ModelTask.classify):
        return ModelType.classify26;
      case (ModelFamily.yolo26, ModelTask.pose):
        return ModelType.pose26;
      case (ModelFamily.yolo26, ModelTask.obb):
        return ModelType.obb26;
    }
  }
}

extension ModelFamilyDisplay on ModelFamily {
  String get label => switch (this) {
    ModelFamily.yolo11 => 'YOLO11',
    ModelFamily.yolo26 => 'YOLO26',
  };
}

extension ModelTaskDisplay on ModelTask {
  String get label => name.toUpperCase();
}

enum SliderType { none, numItems, confidence, iou }
