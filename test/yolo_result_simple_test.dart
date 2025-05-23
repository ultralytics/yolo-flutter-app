// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  group('YOLOResult', () {
    test('constructor creates instance with required parameters', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTRB(10, 10, 110, 210),
        normalizedBox: const Rect.fromLTRB(0.1, 0.1, 0.5, 0.9),
      );

      expect(result.classIndex, 0);
      expect(result.className, 'person');
      expect(result.confidence, 0.95);
      expect(result.boundingBox, const Rect.fromLTRB(10, 10, 110, 210));
      expect(result.normalizedBox, const Rect.fromLTRB(0.1, 0.1, 0.5, 0.9));
      expect(result.mask, isNull);
      expect(result.keypoints, isNull);
      expect(result.keypointConfidences, isNull);
    });

    test('toString returns a string representation', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTRB(10, 10, 110, 210),
        normalizedBox: const Rect.fromLTRB(0.1, 0.1, 0.5, 0.9),
      );

      final string = result.toString();

      expect(string, contains('classIndex: 0'));
      expect(string, contains('className: person'));
      expect(string, contains('confidence: 0.95'));
      expect(
        string,
        contains('boundingBox: Rect.fromLTRB(10.0, 10.0, 110.0, 210.0)'),
      );
    });
  });

  group('YOLODetectionResults', () {
    test('constructor creates instance with required parameters', () {
      final results = YOLODetectionResults(
        detections: [
          YOLOResult(
            classIndex: 0,
            className: 'person',
            confidence: 0.95,
            boundingBox: const Rect.fromLTRB(10, 10, 110, 210),
            normalizedBox: const Rect.fromLTRB(0.1, 0.1, 0.5, 0.9),
          ),
        ],
        processingTimeMs: 25.5,
      );

      expect(results.detections.length, 1);
      expect(results.annotatedImage, isNull);
      expect(results.processingTimeMs, 25.5);
    });
  });

  group('Point', () {
    test('constructor creates instance', () {
      final point = Point(10.0, 20.0);

      expect(point.x, 10.0);
      expect(point.y, 20.0);
    });

    test('toString returns a string representation', () {
      final point = Point(10.0, 20.0);

      final string = point.toString();

      expect(string, 'Point(10.0, 20.0)');
    });
  });
}
