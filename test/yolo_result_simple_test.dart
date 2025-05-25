// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  group('YOLOResult', () {
    final testBoundingBox = const Rect.fromLTRB(10, 10, 110, 210);
    final testNormalizedBox = const Rect.fromLTRB(0.1, 0.1, 0.5, 0.9);

    test('fromMap creates instance with detection data', () {
      final map = {
        'classIndex': 1,
        'className': 'car',
        'confidence': 0.85,
        'boundingBox': {
          'left': 10.0,
          'top': 10.0,
          'right': 110.0,
          'bottom': 210.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
      };

      final result = YOLOResult.fromMap(map);

      expect(result.classIndex, 1);
      expect(result.className, 'car');
      expect(result.confidence, 0.85);
      expect(result.boundingBox, testBoundingBox);
      expect(result.normalizedBox, testNormalizedBox);
    });

    test('fromMap handles segmentation mask data', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 10.0,
          'right': 110.0,
          'bottom': 210.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'mask': [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
      };

      final result = YOLOResult.fromMap(map);

      expect(result.mask, isNotNull);
      expect(result.mask!.length, 2);
      expect(result.mask![0], [0.1, 0.2]);
    });

    test('fromMap handles pose keypoints data', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 10.0,
          'right': 110.0,
          'bottom': 210.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'keypoints': [100.0, 200.0, 0.9, 150.0, 250.0, 0.8],
      };

      final result = YOLOResult.fromMap(map);

      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, 2);
      expect(result.keypoints![0], Point(100.0, 200.0));
      expect(result.keypointConfidences![0], 0.9);
    });

    test('toMap serializes basic detection data', () {
      final result = YOLOResult(
        classIndex: 1,
        className: 'car',
        confidence: 0.85,
        boundingBox: testBoundingBox,
        normalizedBox: testNormalizedBox,
      );

      final map = result.toMap();

      expect(map['classIndex'], 1);
      expect(map['className'], 'car');
      expect(map['confidence'], 0.85);
      expect(map['boundingBox']['left'], 10.0);
    });

    test('toMap includes mask when present', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: testBoundingBox,
        normalizedBox: testNormalizedBox,
        mask: [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
      );

      final map = result.toMap();

      expect(map.containsKey('mask'), true);
      expect(map['mask'], [
        [0.1, 0.2],
        [0.3, 0.4],
      ]);
    });
  });

  group('YOLODetectionResults', () {
    test('fromMap creates instance with detections', () {
      final map = {
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
          },
        ],
        'processingTimeMs': 25.5,
        'annotatedImage': Uint8List.fromList([1, 2, 3]),
      };

      final results = YOLODetectionResults.fromMap(map);

      expect(results.detections.length, 1);
      expect(results.processingTimeMs, 25.5);
      expect(results.annotatedImage, isNotNull);
    });

    test('toMap serializes detection results', () {
      final detection = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTRB(10, 10, 110, 210),
        normalizedBox: const Rect.fromLTRB(0.1, 0.1, 0.5, 0.9),
      );

      final results = YOLODetectionResults(
        detections: [detection],
        processingTimeMs: 25.5,
      );

      final map = results.toMap();

      expect(map['detections'], hasLength(1));
      expect(map['processingTimeMs'], 25.5);
    });
  });

  group('Point', () {
    test('fromMap creates point from coordinates', () {
      final map = {'x': 10.5, 'y': 20.7};
      final point = Point.fromMap(map);

      expect(point.x, 10.5);
      expect(point.y, 20.7);
    });

    test('toMap serializes point coordinates', () {
      final point = Point(15.2, 30.8);
      final map = point.toMap();

      expect(map['x'], 15.2);
      expect(map['y'], 30.8);
    });
  });
}
