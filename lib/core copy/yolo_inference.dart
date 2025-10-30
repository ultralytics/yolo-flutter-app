// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';

/// Inference functionality for YOLO models
class YOLOInference {
  final MethodChannel _channel;
  final String _instanceId;
  final YOLOTask _task;

  YOLOInference({
    required MethodChannel channel,
    required String instanceId,
    required YOLOTask task,
  }) : _channel = channel,
       _instanceId = instanceId,
       _task = task;

  Future<Map<String, dynamic>> predict(
    Uint8List imageBytes, {
    double? confidenceThreshold,
    double? iouThreshold,
  }) async {
    if (imageBytes.isEmpty) {
      throw InvalidInputException('Image data is empty');
    }

    if (confidenceThreshold != null &&
        (confidenceThreshold < 0.0 || confidenceThreshold > 1.0)) {
      throw InvalidInputException(
        'Confidence threshold must be between 0.0 and 1.0',
      );
    }
    if (iouThreshold != null && (iouThreshold < 0.0 || iouThreshold > 1.0)) {
      throw InvalidInputException('IoU threshold must be between 0.0 and 1.0');
    }

    try {
      final Map<String, dynamic> arguments = {'image': imageBytes};

      if (confidenceThreshold != null) {
        arguments['confidenceThreshold'] = confidenceThreshold;
      }
      if (iouThreshold != null) {
        arguments['iouThreshold'] = iouThreshold;
      }

      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      final result = await _channel.invokeMethod(
        'predictSingleImage',
        arguments,
      );

      if (result is Map) {
        return _processInferenceResult(result);
      }

      throw InferenceException('Invalid result format returned from inference');
    } on PlatformException catch (e) {
      throw YOLOErrorHandler.handleError(e, 'Error during image prediction');
    } catch (e) {
      throw YOLOErrorHandler.handleError(e, 'Error during image prediction');
    }
  }

  Map<String, dynamic> _processInferenceResult(Map<dynamic, dynamic> result) {
    final Map<String, dynamic> resultMap = MapConverter.convertToTypedMap(
      result,
    );

    final List<Map<String, dynamic>> boxes = [];
    if (resultMap.containsKey('boxes') && resultMap['boxes'] is List) {
      boxes.addAll(MapConverter.convertBoxesList(resultMap['boxes'] as List));
      resultMap['boxes'] = boxes;
    }

    final List<Map<String, dynamic>> detections = [];

    switch (_task) {
      case YOLOTask.pose:
        detections.addAll(_processPoseResults(resultMap, boxes));
        break;
      case YOLOTask.segment:
        detections.addAll(_processSegmentResults(resultMap, boxes));
        break;
      case YOLOTask.classify:
        detections.addAll(_processClassifyResults(resultMap));
        break;
      case YOLOTask.obb:
        detections.addAll(_processObbResults(resultMap));
        break;
      case YOLOTask.detect:
        detections.addAll(_processDetectResults(boxes));
        break;
    }

    resultMap['detections'] = detections;

    return resultMap;
  }

  List<Map<String, dynamic>> _processPoseResults(
    Map<String, dynamic> resultMap,
    List<Map<String, dynamic>> boxes,
  ) {
    final List<Map<String, dynamic>> detections = [];

    if (resultMap.containsKey('keypoints')) {
      final keypointsList = resultMap['keypoints'] as List<dynamic>? ?? [];

      for (int i = 0; i < boxes.length && i < keypointsList.length; i++) {
        final box = boxes[i];
        final detection = _createDetectionMap(box);

        if (keypointsList[i] is Map) {
          final personKeypoints = keypointsList[i] as Map<dynamic, dynamic>;
          final coordinates =
              personKeypoints['coordinates'] as List<dynamic>? ?? [];

          final flatKeypoints = <double>[];
          for (final coord in coordinates) {
            if (coord is Map) {
              final coordMap = MapConverter.convertToTypedMap(coord);
              flatKeypoints.add(MapConverter.safeGetDouble(coordMap, 'x'));
              flatKeypoints.add(MapConverter.safeGetDouble(coordMap, 'y'));
              flatKeypoints.add(
                MapConverter.safeGetDouble(coordMap, 'confidence'),
              );
            }
          }

          if (flatKeypoints.isNotEmpty) {
            detection['keypoints'] = flatKeypoints;
          }
        }

        detections.add(detection);
      }
    }

    return detections;
  }

  List<Map<String, dynamic>> _processSegmentResults(
    Map<String, dynamic> resultMap,
    List<Map<String, dynamic>> boxes,
  ) {
    final List<Map<String, dynamic>> detections = [];
    final masks = resultMap['masks'] as List<dynamic>? ?? [];

    for (int i = 0; i < boxes.length; i++) {
      final box = boxes[i];
      final detection = _createDetectionMap(box);

      if (i < masks.length && masks[i] != null) {
        final maskData = masks[i] as List<dynamic>;
        final mask = MapConverter.convertMaskData(maskData);
        detection['mask'] = mask;
      }

      detections.add(detection);
    }

    return detections;
  }

  List<Map<String, dynamic>> _processClassifyResults(
    Map<String, dynamic> resultMap,
  ) {
    final List<Map<String, dynamic>> detections = [];

    if (resultMap.containsKey('classification')) {
      final classification =
          resultMap['classification'] as Map<dynamic, dynamic>;

      final detection = <String, dynamic>{
        'classIndex': 0,
        'className': classification['topClass'] ?? '',
        'confidence': MapConverter.safeGetDouble(
          MapConverter.convertToTypedMap(classification),
          'topConfidence',
        ),
        'boundingBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
        'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
      };

      detections.add(detection);
    }

    return detections;
  }

  List<Map<String, dynamic>> _processObbResults(
    Map<String, dynamic> resultMap,
  ) {
    final List<Map<String, dynamic>> detections = [];

    if (resultMap.containsKey('obb')) {
      final obbList = resultMap['obb'] as List<dynamic>? ?? [];

      for (final obb in obbList) {
        if (obb is Map) {
          final points = obb['points'] as List<dynamic>? ?? [];

          double minX = double.infinity, minY = double.infinity;
          double maxX = double.negativeInfinity, maxY = double.negativeInfinity;

          for (final point in points) {
            if (point is Map) {
              final pointMap = MapConverter.convertToTypedMap(point);
              final x = MapConverter.safeGetDouble(pointMap, 'x');
              final y = MapConverter.safeGetDouble(pointMap, 'y');
              minX = minX > x ? x : minX;
              minY = minY > y ? y : minY;
              maxX = maxX < x ? x : maxX;
              maxY = maxY < y ? y : maxY;
            }
          }

          final detection = <String, dynamic>{
            'classIndex': 0,
            'className': obb['class'] ?? '',
            'confidence': MapConverter.safeGetDouble(
              MapConverter.convertToTypedMap(obb),
              'confidence',
            ),
            'boundingBox': {
              'left': minX,
              'top': minY,
              'right': maxX,
              'bottom': maxY,
            },
            'normalizedBox': {
              'left': minX,
              'top': minY,
              'right': maxX,
              'bottom': maxY,
            },
          };

          detections.add(detection);
        }
      }
    }

    return detections;
  }

  List<Map<String, dynamic>> _processDetectResults(
    List<Map<String, dynamic>> boxes,
  ) {
    final List<Map<String, dynamic>> detections = [];

    for (final box in boxes) {
      detections.add(_createDetectionMap(box));
    }

    return detections;
  }

  Map<String, dynamic> _createDetectionMap(Map<String, dynamic> box) {
    return {
      'classIndex': 0,
      'className': MapConverter.safeGetString(box, 'class'),
      'confidence': MapConverter.safeGetDouble(box, 'confidence'),
      'boundingBox': {
        'left': MapConverter.safeGetDouble(box, 'x1'),
        'top': MapConverter.safeGetDouble(box, 'y1'),
        'right': MapConverter.safeGetDouble(box, 'x2'),
        'bottom': MapConverter.safeGetDouble(box, 'y2'),
      },
      'normalizedBox': {
        'left': MapConverter.safeGetDouble(box, 'x1_norm'),
        'top': MapConverter.safeGetDouble(box, 'y1_norm'),
        'right': MapConverter.safeGetDouble(box, 'x2_norm'),
        'bottom': MapConverter.safeGetDouble(box, 'y2_norm'),
      },
    };
  }
}
