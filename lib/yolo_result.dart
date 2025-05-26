// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// lib/yolo_result.dart

import 'dart:typed_data';
import 'dart:ui';

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
  /// Common keypoints include body joints like shoulders, elbows, knees, etc.
  final List<Point>? keypoints;

  /// The confidence values for each detected keypoint.
  ///
  /// Only available when using pose models (YOLOTask.pose).
  /// Each value corresponds to a keypoint in the [keypoints] list
  /// and ranges from 0.0 to 1.0.
  final List<double>? keypointConfidences;

  YOLOResult({
    required this.classIndex,
    required this.className,
    required this.confidence,
    required this.boundingBox,
    required this.normalizedBox,
    this.mask,
    this.keypoints,
    this.keypointConfidences,
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
  /// - 'keypoints': (optional) List of double in x,y,confidence triplets
  factory YOLOResult.fromMap(Map<dynamic, dynamic> map) {
    final classIndex = map['classIndex'] as int;
    final className = map['className'] as String;
    final confidence = (map['confidence'] as num).toDouble();

    // Parse bounding box
    final boxMap = map['boundingBox'] as Map<dynamic, dynamic>;
    final boundingBox = Rect.fromLTRB(
      (boxMap['left'] as num).toDouble(),
      (boxMap['top'] as num).toDouble(),
      (boxMap['right'] as num).toDouble(),
      (boxMap['bottom'] as num).toDouble(),
    );

    // Parse normalized bounding box
    final normalizedBoxMap = map['normalizedBox'] as Map<dynamic, dynamic>;
    final normalizedBox = Rect.fromLTRB(
      (normalizedBoxMap['left'] as num).toDouble(),
      (normalizedBoxMap['top'] as num).toDouble(),
      (normalizedBoxMap['right'] as num).toDouble(),
      (normalizedBoxMap['bottom'] as num).toDouble(),
    );

    // Parse mask if available
    List<List<double>>? mask;
    if (map.containsKey('mask') && map['mask'] != null) {
      final maskData = map['mask'] as List<dynamic>;
      mask = maskData
          .map(
            (row) => (row as List<dynamic>)
                .map((val) => (val as num).toDouble())
                .toList(),
          )
          .toList();
    }

    // Parse keypoints if available
    List<Point>? keypoints;
    List<double>? keypointConfidences;
    if (map.containsKey('keypoints') && map['keypoints'] != null) {
      final keypointsData = map['keypoints'] as List<dynamic>;
      keypoints = [];
      keypointConfidences = [];

      for (var i = 0; i < keypointsData.length; i += 3) {
        keypoints.add(
          Point(
            (keypointsData[i] as num).toDouble(),
            (keypointsData[i + 1] as num).toDouble(),
          ),
        );
        keypointConfidences.add((keypointsData[i + 2] as num).toDouble());
      }
    }

    return YOLOResult(
      classIndex: classIndex,
      className: className,
      confidence: confidence,
      boundingBox: boundingBox,
      normalizedBox: normalizedBox,
      mask: mask,
      keypoints: keypoints,
      keypointConfidences: keypointConfidences,
    );
  }

  /// Converts this [YOLOResult] to a map representation.
  ///
  /// This method is used for serializing the result for platform channel
  /// communication. The returned map contains all the properties of this
  /// result in a format suitable for transmission across platform channels.
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

    return map;
  }

  @override
  String toString() {
    return 'YOLOResult{classIndex: $classIndex, className: $className, confidence: $confidence, boundingBox: $boundingBox}';
  }
}

/// Represents a collection of detection results from YOLO models.
///
/// This class encapsulates the complete output from a YOLO inference,
/// including all detected objects, an optional annotated image showing
/// the detections, and performance metrics.
///
/// Example:
/// ```dart
/// final results = await yolo.predict(imageBytes);
/// print('Found ${results.detections.length} objects');
/// print('Processing took ${results.processingTimeMs}ms');
/// if (results.annotatedImage != null) {
///   // Display or save the annotated image
/// }
/// ```
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
  /// May be null if annotation was disabled.
  final Uint8List? annotatedImage;

  /// The time taken to process the image in milliseconds.
  ///
  /// This includes model inference time and post-processing,
  /// but excludes image preprocessing and annotation rendering.
  final double processingTimeMs;

  YOLODetectionResults({
    required this.detections,
    this.annotatedImage,
    required this.processingTimeMs,
  });

  /// Creates [YOLODetectionResults] from a map representation.
  ///
  /// This factory constructor deserializes results received from
  /// the platform channel. The map should contain:
  /// - 'detections': List of detection maps
  /// - 'annotatedImage': (optional) Uint8List of image data
  /// - 'processingTimeMs': double representing processing time
  factory YOLODetectionResults.fromMap(Map<dynamic, dynamic> map) {
    // Parse detections
    final detectionsData = map['detections'] as List<dynamic>?;
    final detections = detectionsData != null
        ? detectionsData
              .map((detection) => YOLOResult.fromMap(detection))
              .toList()
        : <YOLOResult>[];

    // Parse annotated image if available
    final annotatedImage = map['annotatedImage'] as Uint8List?;

    // Parse processing time
    final processingTimeMs = map.containsKey('processingTimeMs')
        ? (map['processingTimeMs'] as num).toDouble()
        : 0.0;

    return YOLODetectionResults(
      detections: detections,
      annotatedImage: annotatedImage,
      processingTimeMs: processingTimeMs,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'detections': detections.map((detection) => detection.toMap()).toList(),
      'annotatedImage': annotatedImage,
      'processingTimeMs': processingTimeMs,
    };
  }
}

/// Represents a point in 2D space.
///
/// Used primarily for representing keypoint locations in pose estimation
/// results. Coordinates are typically in pixel space relative to the
/// original image dimensions.
///
/// Example:
/// ```dart
/// final point = Point(150.5, 200.0);
/// print('Point at (${point.x}, ${point.y})');
/// ```
class Point {
  /// The x-coordinate of the point.
  final double x;

  /// The y-coordinate of the point.
  final double y;

  Point(this.x, this.y);

  Map<String, double> toMap() => {'x': x, 'y': y};

  factory Point.fromMap(Map<dynamic, dynamic> map) {
    return Point((map['x'] as num).toDouble(), (map['y'] as num).toDouble());
  }

  @override
  String toString() => 'Point($x, $y)';
}
