import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  group('Mask Data Parsing Tests', () {
    test('YOLOResult.fromMap should correctly parse mask data', () {
      // Create test data with mask information
      final testData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 100.0,
          'top': 150.0,
          'right': 300.0,
          'bottom': 400.0,
        },
        'normalizedBox': {
          'left': 0.1,
          'top': 0.15,
          'right': 0.3,
          'bottom': 0.4,
        },
        'mask': [
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9],
        ],
      };

      // Parse the result
      final result = YOLOResult.fromMap(testData);

      // Verify basic properties
      expect(result.classIndex, equals(0));
      expect(result.className, equals('person'));
      expect(result.confidence, equals(0.95));

      // Verify mask data is correctly parsed
      expect(result.mask, isNotNull);
      expect(result.mask!.length, equals(3));
      expect(result.mask![0].length, equals(3));
      expect(result.mask![0][0], equals(0.1));
      expect(result.mask![1][1], equals(0.5));
      expect(result.mask![2][2], equals(0.9));
    });

    test('YOLOResult.fromMap should handle missing mask data', () {
      // Create test data without mask information
      final testData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 100.0,
          'top': 150.0,
          'right': 300.0,
          'bottom': 400.0,
        },
        'normalizedBox': {
          'left': 0.1,
          'top': 0.15,
          'right': 0.3,
          'bottom': 0.4,
        },
      };

      // Parse the result
      final result = YOLOResult.fromMap(testData);

      // Verify basic properties
      expect(result.classIndex, equals(0));
      expect(result.className, equals('person'));
      expect(result.confidence, equals(0.95));

      // Verify mask data is null when not provided
      expect(result.mask, isNull);
    });

    test('YOLOResult.fromMap should handle keypoints data', () {
      // Create test data with keypoints information
      final testData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 100.0,
          'top': 150.0,
          'right': 300.0,
          'bottom': 400.0,
        },
        'normalizedBox': {
          'left': 0.1,
          'top': 0.15,
          'right': 0.3,
          'bottom': 0.4,
        },
        'keypoints': [
          100.0, 200.0, 0.9, // x, y, confidence for first keypoint
          150.0, 250.0, 0.8, // x, y, confidence for second keypoint
        ],
      };

      // Parse the result
      final result = YOLOResult.fromMap(testData);

      // Verify basic properties
      expect(result.classIndex, equals(0));
      expect(result.className, equals('person'));
      expect(result.confidence, equals(0.95));

      // Verify keypoints data is correctly parsed
      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, equals(2));
      expect(result.keypoints![0].x, equals(100.0));
      expect(result.keypoints![0].y, equals(200.0));
      expect(result.keypointConfidences![0], equals(0.9));
      expect(result.keypoints![1].x, equals(150.0));
      expect(result.keypoints![1].y, equals(250.0));
      expect(result.keypointConfidences![1], equals(0.8));
    });
  });
}