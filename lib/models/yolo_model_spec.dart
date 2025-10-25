/// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/models/yolo_model_spec.dart
//
// A simple data model representing a YOLO model specification used by YOLOView.
// It encapsulates the model path and the associated task type.
//
// This model includes helpers for (de)serialization and a derived modelName
// (basename of the file without extension), which is useful for tagging results
// with their source model in multi-model inference.

import 'package:meta/meta.dart';
import 'yolo_task.dart';

@immutable
class YOLOModelSpec {
  /// Absolute or bundle-relative path to the model file.
  final String modelPath;

  /// The YOLO task type that the model performs.
  final YOLOTask task;

  const YOLOModelSpec({required this.modelPath, required this.task});

  /// Returns the basename of [modelPath] without file extension.
  ///
  /// Examples:
  /// - "/path/to/yolo11n.tflite" -> "yolo11n"
  /// - "assets/models/my_model.mlmodelc" -> "my_model"
  String get modelName {
    // Normalize slashes for cross-platform paths
    final normalized = modelPath.replaceAll('\\', '/');
    final base = normalized.split('/').isNotEmpty
        ? normalized.split('/').last
        : normalized;

    final dotIdx = base.lastIndexOf('.');
    if (dotIdx > 0) {
      return base.substring(0, dotIdx);
    }
    return base;
  }

  YOLOModelSpec copyWith({String? modelPath, YOLOTask? task}) {
    return YOLOModelSpec(
      modelPath: modelPath ?? this.modelPath,
      task: task ?? this.task,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'modelPath': modelPath,
      'task': task.name, // dart enum name: detect/segment/classify/pose/obb
    };
  }

  factory YOLOModelSpec.fromMap(Map<dynamic, dynamic> map) {
    final path = map['modelPath'] as String? ?? '';
    final taskRaw = (map['task'] as String? ?? '').toLowerCase();

    return YOLOModelSpec(modelPath: path, task: _parseTask(taskRaw));
  }

  static YOLOTask _parseTask(String value) {
    switch (value) {
      case 'detect':
        return YOLOTask.detect;
      case 'segment':
        return YOLOTask.segment;
      case 'classify':
        return YOLOTask.classify;
      case 'pose':
        return YOLOTask.pose;
      case 'obb':
        return YOLOTask.obb;
      default:
        // Default to detect if not recognizable; upstream should validate
        return YOLOTask.detect;
    }
  }

  static List<YOLOModelSpec> listFromDynamic(dynamic value) {
    if (value is List) {
      return value
          .whereType<Map>()
          .map((m) => YOLOModelSpec.fromMap(m))
          .toList(growable: false);
    }
    return const [];
  }

  @override
  String toString() =>
      'YOLOModelSpec(modelPath: $modelPath, task: ${task.name})';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is YOLOModelSpec &&
        other.modelPath == modelPath &&
        other.task == task;
  }

  @override
  int get hashCode => Object.hash(modelPath, task);
}
