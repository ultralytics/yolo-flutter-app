// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:ui';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

void main() {
  group('YOLOResult', () {
    const testBoundingBox = Rect.fromLTRB(10, 10, 110, 210);
    const testNormalizedBox = Rect.fromLTRB(0.1, 0.1, 0.5, 0.9);

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

      expect(result.classIndex, 0);
      expect(result.className, 'person');
      expect(result.confidence, 0.95);
      expect(result.mask, isNotNull);
      expect(result.mask!.length, 2);
    });

    test('fromMap handles pose keypoints data', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.9,
        'boundingBox': {
          'left': 10.0,
          'top': 10.0,
          'right': 110.0,
          'bottom': 210.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'keypoints': [0.5, 0.3, 0.8, 0.6, 0.4, 0.9],
      };

      final result = YOLOResult.fromMap(map);

      expect(result.classIndex, 0);
      expect(result.className, 'person');
      expect(result.confidence, 0.9);
      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, 2);
      expect(result.keypoints![0].x, 0.5);
      expect(result.keypoints![0].y, 0.3);
      expect(result.keypointConfidences, isNotNull);
      expect(result.keypointConfidences![0], 0.8);
    });

    test('fromMap handles minimal data', () {
      final map = {'classIndex': 0, 'className': 'object', 'confidence': 0.5};

      final result = YOLOResult.fromMap(map);

      expect(result.classIndex, 0);
      expect(result.className, 'object');
      expect(result.confidence, 0.5);
      expect(result.boundingBox, Rect.zero);
      expect(result.normalizedBox, Rect.zero);
      expect(result.mask, isNull);
      expect(result.keypoints, isNull);
      expect(result.keypointConfidences, isNull);
    });

    test('fromMap handles null values gracefully', () {
      final map = <String, dynamic>{
        'classIndex': 0,
        'className': 'object',
        'confidence': 0.5,
        'boundingBox': null,
        'normalizedBox': null,
        'mask': null,
        'keypoints': null,
      };

      final result = YOLOResult.fromMap(map);

      expect(result.classIndex, 0);
      expect(result.className, 'object');
      expect(result.confidence, 0.5);
      expect(result.boundingBox, Rect.zero);
      expect(result.normalizedBox, Rect.zero);
      expect(result.mask, isNull);
      expect(result.keypoints, isNull);
      expect(result.keypointConfidences, isNull);
    });

    test('constructor creates instance with all parameters', () {
      final keypoints = [Point(0.5, 0.3), Point(0.6, 0.4)];
      final keypointConfidences = [0.8, 0.9];
      final mask = [
        [0.1, 0.2],
        [0.3, 0.4],
      ];

      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.9,
        boundingBox: testBoundingBox,
        normalizedBox: testNormalizedBox,
        keypoints: keypoints,
        keypointConfidences: keypointConfidences,
        mask: mask,
      );

      expect(result.classIndex, 0);
      expect(result.className, 'person');
      expect(result.confidence, 0.9);
      expect(result.boundingBox, testBoundingBox);
      expect(result.normalizedBox, testNormalizedBox);
      expect(result.keypoints, keypoints);
      expect(result.keypointConfidences, keypointConfidences);
      expect(result.mask, mask);
    });

    test('toMap converts instance to map', () {
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
      expect(map['boundingBox'], isNotNull);
      expect(map['normalizedBox'], isNotNull);
    });
  });
}
