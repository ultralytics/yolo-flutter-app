// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_result.dart

import 'dart:typed_data';
import 'dart:ui';
import '../utils/map_converter.dart';

/// Represents a detection result from YOLO models.
///
/// This class encapsulates all the information returned by YOLO models
/// for a single detected object, including its location, classification,
/// and task-specific data like segmentation masks or pose keypoints.
///
/// Example:
/// ```dart
/// final result = YOLOResult(
///   classIndex: 0,
///   className: 'person',
///   confidence: 0.95,
///   boundingBox: Rect.fromLTWH(100, 100, 200, 300),
///   normalizedBox: Rect.fromLTWH(0.1, 0.1, 0.2, 0.3),
/// );
/// ```
class YOLOResult {
  /// The index of the detected class in the model's class list.
  ///
  /// This corresponds to the position of the class in the model's
  /// training dataset labels.
  final int classIndex;

  /// The human-readable name of the detected class.
  ///
  /// Examples: "person", "car", "dog", "chair", etc.
  /// The exact names depend on the model's training dataset.
  final String className;

  /// The confidence score of the detection.
  ///
  /// A value between 0.0 and 1.0 representing the model's
  /// confidence in this detection. Higher values indicate
  /// more confident detections.
  final double confidence;

  /// The bounding box of the detected object in pixel coordinates.
  ///
  /// This rectangle defines the location and size of the detected
  /// object within the original image, using absolute pixel values.
  final Rect boundingBox;

  /// The normalized bounding box coordinates.
  ///
  /// All values are between 0.0 and 1.0, representing the relative
  /// position and size within the image. This is useful for
  /// resolution-independent processing.
  final Rect normalizedBox;

  /// The segmentation mask for instance segmentation tasks.
  ///
  /// Only available when using segmentation models (YOLOTask.segment).
  /// Each inner list represents a row of mask values.
  final List<List<double>>? mask;

  /// The detected keypoints for pose estimation tasks.
  ///
  /// Only available when using pose models (YOLOTask.pose).
  /// Coordinates are in image pixel space.
  /// Common keypoints include body joints like shoulders, elbows, knees, etc.
  final List<Keypoint>? keypoints;

  /// The confidence values for each detected keypoint.
  ///
  /// Only available when using pose models (YOLOTask.pose).
  /// Each value corresponds to a keypoint in the [keypoints] list
  /// and ranges from 0.0 to 1.0.
  final List<double>? keypointConfidences;

  /// The rotation angle for oriented bounding boxes in radians.
  ///
  /// Only available when using OBB models (YOLOTask.obb).
  final double? angle;

  /// Polygon corner points for oriented bounding boxes, in pixel coordinates of the camera frame.
  final List<Map<String, num>>? obbPoints;

  /// Polygon corner points for oriented bounding boxes, normalized to 0.0-1.0.
  ///
  /// Multiply by the frame dimensions (`imageWidth`/`imageHeight` in streaming data) or your own
  /// render size to draw custom OBB overlays.
  final List<Map<String, num>>? obbPointsNormalized;

  YOLOResult({
    required this.classIndex,
    required this.className,
    required this.confidence,
    required this.boundingBox,
    required this.normalizedBox,
    this.mask,
    this.keypoints,
    this.keypointConfidences,
    this.angle,
    this.obbPoints,
    this.obbPointsNormalized,
  });

  /// Creates a [YOLOResult] from a map representation.
  ///
  /// This factory constructor is primarily used for deserializing results
  /// received from the platform channel. The map should contain keys:
  /// - 'classIndex': int
  /// - 'className': String
  /// - 'confidence': double
  /// - 'boundingBox': Map with 'left', 'top', 'right', 'bottom'
  /// - 'normalizedBox': Map with 'left', 'top', 'right', 'bottom'
  /// - 'mask': (optional) List of List of double
  /// - 'keypoints': (optional) List of double in pixel x,y,confidence triplets
  factory YOLOResult.fromMap(Map<dynamic, dynamic> map) {
    final classIndex = MapConverter.safeGetInt(map, 'classIndex');
    final className = MapConverter.safeGetString(map, 'className');
    final confidence = MapConverter.safeGetDouble(map, 'confidence');

    final boxMap = MapConverter.convertToTypedMapSafe(
      map['boundingBox'] as Map<dynamic, dynamic>?,
    );
    final boundingBox = boxMap != null
        ? MapConverter.convertBoundingBox(boxMap)
        : Rect.zero;

    final normalizedBoxMap = MapConverter.convertToTypedMapSafe(
      map['normalizedBox'] as Map<dynamic, dynamic>?,
    );
    final normalizedBox = normalizedBoxMap != null
        ? MapConverter.convertBoundingBox(normalizedBoxMap)
        : Rect.zero;

    List<List<double>>? mask;
    if (map.containsKey('mask') && map['mask'] != null) {
      final maskData = map['mask'] as List<dynamic>;
      mask = MapConverter.convertMaskData(maskData);
    }

    List<Keypoint>? keypoints;
    List<double>? keypointConfidences;
    if (map.containsKey('keypoints') && map['keypoints'] != null) {
      final keypointsData = map['keypoints'] as List<dynamic>;
      final keypointResult = MapConverter.convertKeypoints(keypointsData);
      keypoints = keypointResult.keypoints;
      keypointConfidences = keypointResult.confidences;
    }

    final angle = map['angle'] is num ? (map['angle'] as num).toDouble() : null;
    final obbPoints = _parsePolygonPoints(map['polygon'] ?? map['obbPoints']);
    final obbPointsNormalized = _parsePolygonPoints(map['polygonNormalized']);

    return YOLOResult(
      classIndex: classIndex,
      className: className,
      confidence: confidence,
      boundingBox: boundingBox,
      normalizedBox: normalizedBox,
      mask: mask,
      keypoints: keypoints,
      keypointConfidences: keypointConfidences,
      angle: angle,
      obbPoints: obbPoints,
      obbPointsNormalized: obbPointsNormalized,
    );
  }

  /// Parses a list of `{x, y}` maps, keeping only valid numeric points.
  static List<Map<String, num>>? _parsePolygonPoints(dynamic raw) {
    if (raw is! List) return null;
    final points = <Map<String, num>>[];
    for (final point in raw) {
      if (point is! Map) continue;
      final x = point['x'];
      final y = point['y'];
      if (x is! num || y is! num) continue;
      points.add({'x': x.toDouble(), 'y': y.toDouble()});
    }
    return points;
  }

  Map<String, dynamic> toMap() {
    final map = <String, dynamic>{
      'classIndex': classIndex,
      'className': className,
      'confidence': confidence,
      'boundingBox': {
        'left': boundingBox.left,
        'top': boundingBox.top,
        'right': boundingBox.right,
        'bottom': boundingBox.bottom,
      },
      'normalizedBox': {
        'left': normalizedBox.left,
        'top': normalizedBox.top,
        'right': normalizedBox.right,
        'bottom': normalizedBox.bottom,
      },
    };

    if (mask != null) {
      map['mask'] = mask;
    }

    if (keypoints != null && keypointConfidences != null) {
      final keypointsData = <double>[];
      for (var i = 0; i < keypoints!.length; i++) {
        keypointsData.add(keypoints![i].x);
        keypointsData.add(keypoints![i].y);
        keypointsData.add(keypointConfidences![i]);
      }
      map['keypoints'] = keypointsData;
    }

    if (angle != null) {
      map['angle'] = angle;
    }

    if (obbPoints != null) {
      map['polygon'] = obbPoints;
    }

    if (obbPointsNormalized != null) {
      map['polygonNormalized'] = obbPointsNormalized;
    }

    return map;
  }

  @override
  String toString() {
    return 'YOLOResult{classIndex: $classIndex, className: $className, confidence: $confidence, boundingBox: $boundingBox}';
  }
}

/// Complete output from a YOLO inference.
///
/// Includes detections, optional task-level outputs such as semantic masks,
/// depth maps, an optional annotated image, and processing time in milliseconds.
class YOLODetectionResults {
  /// List of all objects detected in the image.
  ///
  /// Each [YOLOResult] in this list represents a single detected object
  /// with its location, classification, and confidence score.
  final List<YOLOResult> detections;

  /// The original image with detection visualizations overlaid.
  ///
  /// This annotated image includes bounding boxes, class labels,
  /// confidence scores, and other task-specific visualizations
  /// (masks for segmentation, keypoints for pose estimation).
  final Uint8List? annotatedImage;

  /// Dense class map for semantic segmentation tasks.
  final YOLOSemanticMask? semanticMask;

  /// Dense metric distance map for depth estimation tasks.
  final YOLODepthMap? depthMap;

  /// The time taken to process the image in milliseconds.
  ///
  /// This includes model inference time and post-processing,
  /// but excludes image preprocessing and annotation rendering.
  final double processingTimeMs;

  YOLODetectionResults({
    required this.detections,
    this.annotatedImage,
    this.semanticMask,
    this.depthMap,
    required this.processingTimeMs,
  });

  /// Creates [YOLODetectionResults] from a map representation.
  ///
  /// This factory constructor deserializes results received from
  /// the platform channel. The map should contain:
  /// - 'detections': List of detection maps
  /// - 'annotatedImage': (optional) Uint8List of image data
  /// - 'semanticMask': (optional) semantic class map data
  /// - 'processingTimeMs': double representing processing time
  factory YOLODetectionResults.fromMap(Map<dynamic, dynamic> map) {
    final detectionsData = map['detections'] as List<dynamic>?;
    final detections = detectionsData != null
        ? detectionsData
              .map((detection) => YOLOResult.fromMap(detection))
              .toList()
        : <YOLOResult>[];

    final annotatedImage = MapConverter.safeGetUint8List(map, 'annotatedImage');
    final semanticMaskRaw = map['semanticMask'];
    final semanticMask = semanticMaskRaw is Map
        ? YOLOSemanticMask.fromMap(semanticMaskRaw)
        : null;
    final depthMapRaw = map['depthMap'];
    final depthMap = depthMapRaw is Map
        ? YOLODepthMap.fromMap(depthMapRaw)
        : null;

    final processingTimeMs = MapConverter.safeGetDouble(
      map,
      'processingTimeMs',
    );

    return YOLODetectionResults(
      detections: detections,
      annotatedImage: annotatedImage,
      semanticMask: semanticMask,
      depthMap: depthMap,
      processingTimeMs: processingTimeMs,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'detections': detections.map((detection) => detection.toMap()).toList(),
      'annotatedImage': annotatedImage,
      'semanticMask': semanticMask?.toMap(),
      'depthMap': depthMap?.toMap(),
      'processingTimeMs': processingTimeMs,
    };
  }
}

/// Dense row-major metric depth output.
class YOLODepthMap {
  /// Distance in meters for every map pixel.
  final Float32List values;

  final int width;
  final int height;
  final double minDepth;
  final double maxDepth;

  YOLODepthMap({
    required this.values,
    required this.width,
    required this.height,
    required this.minDepth,
    required this.maxDepth,
  });

  factory YOLODepthMap.fromMap(Map<dynamic, dynamic> map) {
    final raw = map['values'];
    final values = raw is Float32List
        ? raw
        : Float32List.fromList(
            (raw as List<dynamic>? ?? const [])
                .whereType<num>()
                .map((value) => value.toDouble())
                .toList(),
          );
    return YOLODepthMap(
      values: values,
      width: MapConverter.safeGetInt(map, 'width'),
      height: MapConverter.safeGetInt(map, 'height'),
      minDepth: MapConverter.safeGetDouble(map, 'minDepth'),
      maxDepth: MapConverter.safeGetDouble(map, 'maxDepth'),
    );
  }

  Map<String, dynamic> toMap() => {
    'values': values,
    'width': width,
    'height': height,
    'minDepth': minDepth,
    'maxDepth': maxDepth,
  };
}

/// Dense full-image semantic segmentation result.
class YOLOSemanticMask {
  /// Row-major class index for each mask pixel.
  final List<int> classMap;

  /// Mask width in pixels.
  final int width;

  /// Mask height in pixels.
  final int height;

  YOLOSemanticMask({
    required this.classMap,
    required this.width,
    required this.height,
  });

  factory YOLOSemanticMask.fromMap(Map<dynamic, dynamic> map) {
    final classMapRaw = map['classMap'] as List<dynamic>? ?? const [];
    return YOLOSemanticMask(
      classMap: classMapRaw
          .whereType<num>()
          .map((value) => value.toInt())
          .toList(),
      width: MapConverter.safeGetInt(map, 'width'),
      height: MapConverter.safeGetInt(map, 'height'),
    );
  }

  Map<String, dynamic> toMap() => {
    'classMap': classMap,
    'width': width,
    'height': height,
  };
}

/// A single 2D pose keypoint (joint) in image/pixel space.
class Keypoint {
  final double x;
  final double y;

  Keypoint(this.x, this.y);

  Map<String, double> toMap() => {'x': x, 'y': y};

  factory Keypoint.fromMap(Map<dynamic, dynamic> map) {
    return Keypoint(
      MapConverter.safeGetDouble(map, 'x'),
      MapConverter.safeGetDouble(map, 'y'),
    );
  }

  @override
  String toString() => 'Keypoint($x, $y)';
}
