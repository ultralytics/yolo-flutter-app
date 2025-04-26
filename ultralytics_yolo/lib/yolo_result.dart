// lib/yolo_result.dart

import 'dart:typed_data';
import 'dart:ui';

/// Represents a detection result from YOLO models
class YOLOResult {
  /// Index of the detected class
  final int classIndex;
  
  /// Name of the detected class (e.g. "person", "car")
  final String className;
  
  /// Confidence score between 0.0 and 1.0
  final double confidence;
  
  /// Bounding box of the detected object
  final Rect boundingBox;
  
  /// Normalized bounding box coordinates (values between 0.0 and 1.0)
  final Rect normalizedBox;
  
  /// Segmentation mask for segmentation tasks (nullable)
  final List<List<double>>? mask;
  
  /// Keypoints for pose estimation tasks (nullable)
  final List<Point>? keypoints;
  
  /// Keypoint confidence values for pose estimation tasks (nullable)
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

  /// Create a YOLOResult from a map (used for platform channel communication)
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
      mask = maskData.map((row) => 
        (row as List<dynamic>).map((val) => (val as num).toDouble()).toList()
      ).toList();
    }
    
    // Parse keypoints if available
    List<Point>? keypoints;
    List<double>? keypointConfidences;
    if (map.containsKey('keypoints') && map['keypoints'] != null) {
      final keypointsData = map['keypoints'] as List<dynamic>;
      keypoints = [];
      keypointConfidences = [];
      
      for (var i = 0; i < keypointsData.length; i += 3) {
        keypoints.add(Point(
          (keypointsData[i] as num).toDouble(),
          (keypointsData[i + 1] as num).toDouble(),
        ));
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
  
  /// Convert this YOLOResult to a map (used for platform channel communication)
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

/// Represents a detection results collection from YOLO models
class YOLODetectionResults {
  /// List of detected objects
  final List<YOLOResult> detections;
  
  /// Annotated image with visualized detections
  final Uint8List? annotatedImage;
  
  /// Processing speed in milliseconds
  final double processingTimeMs;
  
  YOLODetectionResults({
    required this.detections,
    this.annotatedImage,
    required this.processingTimeMs,
  });
  
  /// Create detection results from a map
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

/// Represents a point in 2D space
class Point {
  final double x;
  final double y;
  
  Point(this.x, this.y);
  
  Map<String, double> toMap() => {'x': x, 'y': y};
  
  factory Point.fromMap(Map<dynamic, dynamic> map) {
    return Point(
      (map['x'] as num).toDouble(),
      (map['y'] as num).toDouble(),
    );
  }
  
  @override
  String toString() => 'Point($x, $y)';
}