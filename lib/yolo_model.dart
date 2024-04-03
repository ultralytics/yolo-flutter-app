/// Type of [YoloModel].
enum Type {
  /// Custom model stored locally.
  local,

  /// Custom model stored remotely.
  remote,
}

/// Task of [YoloModel].
enum Task {
  /// Image Classification task.
  classify('classify'),

  /// Object Detection task.
  detect('detect'),

  /// Pose estimation task.
  pose('pose');

  const Task(this.name);

  factory Task.fromString(String task) =>
      values.singleWhere((element) => element.name == task);

  /// Returns the string representation of the [Task].
  final String name;
}

/// Format of [YoloModel].
enum Format {
  /// CoreML format.
  coreml('coreml', '.mlmodel'),

  /// TensorFlow Lite format.
  tflite('tflite', '.tflite');

  const Format(this.name, this.extension);

  /// Returns the string representation of the [Format].
  final String name;

  /// Returns the file extension of the [Format].
  final dynamic extension;
}

/// Base class for YOLO models.
abstract class YoloModel {
  /// Constructor to create an instance of [YoloModel].
  YoloModel({
    required this.id,
    required this.type,
    required this.task,
    required this.format,
  });

  /// Unique identifier for the model.
  final String id;

  /// Type of the model.
  final Type type;

  /// Task of the model.
  final Task task;

  /// Format of the model.
  final Format format;

  /// Returns a json representation of an instance of [YoloModel].
  Map<String, dynamic> toJson() => {
        'id': id,
        'type': type.name,
        'task': task.name,
        'format': format.name,
      };
}

/// Options to configure the detector while using a local
/// Ultralytics YOLO model.
class LocalYoloModel extends YoloModel {
  /// Constructor to create an instance of [LocalYoloModel].
  LocalYoloModel({
    required super.id,
    required this.modelPath,
    required super.task,
    required super.format,
    this.metadataPath,
  }) : super(type: Type.local);

  /// Path where the local custom model is stored.
  final String modelPath;

  /// Path where the local custom model metadata is stored.
  final String? metadataPath;

  /// Returns a json representation of an instance of [LocalYoloModel].
  @override
  Map<String, dynamic> toJson() => {
        ...super.toJson(),
        'modelPath': modelPath,
        'metadataPath': metadataPath,
      };
}

/// Options to configure the detector while using a remote
/// Ultralytics YOLO model.
class RemoteYoloModel extends YoloModel {
  /// Constructor to create an instance of [RemoteYoloModel].
  RemoteYoloModel({
    required super.id,
    required this.modelUrl,
    required super.task,
    required super.format,
  }) : super(type: Type.remote);

  /// Path where the local custom model is stored.
  final String modelUrl;

  /// Returns a json representation of an instance of [RemoteYoloModel].
  @override
  Map<String, dynamic> toJson() => {
        ...super.toJson(),
        'modelUrl': modelUrl,
      };
}
