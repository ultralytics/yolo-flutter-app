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
  /// Absolute or bundle-relative path to the model file. Optional when [type] is provided.
  final String? modelPath;

  /// Logical model type/name (e.g. "yolo11n"). Optional when [modelPath] is provided.
  final String? type;

  /// The YOLO task type that the model performs.
  final YOLOTask task;

  const YOLOModelSpec({this.modelPath, this.type, required this.task})
    : assert(
        modelPath != null || type != null,
        'Provide either modelPath or type',
      );

  /// Returns the effective model name (type).
  ///
  /// If [type] is provided, it is returned as-is.
  /// Otherwise, derives the basename from [modelPath] without extension.
  String get modelName {
    if (type != null && type!.isNotEmpty) return type!;
    final path = modelPath ?? '';
    // Normalize slashes for cross-platform paths
    final normalized = path.replaceAll('\\', '/');
    final base = normalized.split('/').isNotEmpty
        ? normalized.split('/').last
        : normalized;

    final dotIdx = base.lastIndexOf('.');
    return dotIdx > 0 ? base.substring(0, dotIdx) : base;
  }

  YOLOModelSpec copyWith({String? modelPath, String? type, YOLOTask? task}) {
    return YOLOModelSpec(
      modelPath: modelPath ?? this.modelPath,
      type: type ?? this.type,
      task: task ?? this.task,
    );
  }

  Map<String, dynamic> toMap() {
    final map = <String, dynamic>{
      'modelName': modelName,
      'task': task.name, // dart enum name: detect/segment/classify/pose/obb
    };
    if (modelPath != null && modelPath!.isNotEmpty) {
      map['modelPath'] = modelPath;
    }
    return map;
  }

  factory YOLOModelSpec.fromMap(Map<dynamic, dynamic> map) {
    final path = map['modelPath'] as String?;
    // Prefer explicit modelName/type if provided
    final type = (map['modelName'] as String?) ?? (map['type'] as String?);
    final taskRaw = (map['task'] as String? ?? '').toLowerCase();

    return YOLOModelSpec(
      modelPath: path,
      type: type,
      task: _parseTask(taskRaw),
    );
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
      'YOLOModelSpec(modelPath: $modelPath, type: $type, task: ${task.name})';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is YOLOModelSpec &&
        other.modelPath == modelPath &&
        other.type == type &&
        other.task == task;
  }

  @override
  int get hashCode => Object.hash(modelPath, type, task);
}
