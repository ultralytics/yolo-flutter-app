// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

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
      expect(result.keypoints![0].x, 100.0);
      expect(result.keypoints![0].y, 200.0);
      expect(result.keypointConfidences![0], 0.9);
      expect(result.keypoints![1].x, 150.0);
      expect(result.keypoints![1].y, 250.0);
      expect(result.keypointConfidences![1], 0.8);
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

    test('toMap includes keypoints when present', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: testBoundingBox,
        normalizedBox: testNormalizedBox,
        keypoints: [Point(100.0, 200.0), Point(150.0, 250.0)],
        keypointConfidences: [0.9, 0.8],
      );

      final map = result.toMap();

      expect(map.containsKey('keypoints'), true);
      expect(map['keypoints'], [100.0, 200.0, 0.9, 150.0, 250.0, 0.8]);
    });

    test('fromMap handles OBB detection data', () {
      final map = {
        'classIndex': 2,
        'className': 'car',
        'confidence': 0.88,
        'boundingBox': {
          'left': 20.0,
          'top': 30.0,
          'right': 120.0,
          'bottom': 180.0,
        },
        'normalizedBox': {'left': 0.2, 'top': 0.3, 'right': 0.6, 'bottom': 0.8},
        'orientation': 45.0, // OBB-specific field
      };

      final result = YOLOResult.fromMap(map);

      expect(result.classIndex, 2);
      expect(result.className, 'car');
      expect(result.confidence, 0.88);
    });

    test('fromMap handles classification data structure', () {
      final map = {
        'classIndex': 1,
        'className': 'dog',
        'confidence': 0.92,
        'boundingBox': {'left': 0.0, 'top': 0.0, 'right': 0.0, 'bottom': 0.0},
        'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 0.0, 'bottom': 0.0},
      };

      final result = YOLOResult.fromMap(map);

      expect(result.className, 'dog');
      expect(result.confidence, 0.92);
      expect(result.boundingBox, const Rect.fromLTRB(0.0, 0.0, 0.0, 0.0));
    });

    test('fromMap handles missing optional fields gracefully', () {
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
        // No mask, keypoints, etc.
      };

      final result = YOLOResult.fromMap(map);

      expect(result.mask, isNull);
      expect(result.keypoints, isNull);
      expect(result.keypointConfidences, isNull);
    });

    test('fromMap handles empty mask data', () {
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
        'mask': [],
      };

      final result = YOLOResult.fromMap(map);

      expect(result.mask, isEmpty);
    });

    test('fromMap handles empty keypoints data', () {
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
        'keypoints': [],
      };

      final result = YOLOResult.fromMap(map);

      expect(result.keypoints, isEmpty);
      expect(result.keypointConfidences, isEmpty);
    });

    test('fromMap handles edge case confidence values', () {
      final cases = [
        {'confidence': 0.0, 'expected': 0.0},
        {'confidence': 1.0, 'expected': 1.0},
        {'confidence': 0.999999, 'expected': 0.999999},
        {'confidence': 0.000001, 'expected': 0.000001},
      ];

      for (final testCase in cases) {
        final map = {
          'classIndex': 0,
          'className': 'test',
          'confidence': testCase['confidence'],
          'boundingBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
          'normalizedBox': {
            'left': 0.0,
            'top': 0.0,
            'right': 1.0,
            'bottom': 1.0,
          },
        };

        final result = YOLOResult.fromMap(map);
        expect(result.confidence, testCase['expected']);
      }
    });

    test('fromMap handles very large bounding boxes', () {
      final map = {
        'classIndex': 0,
        'className': 'large_object',
        'confidence': 0.95,
        'boundingBox': {
          'left': 0.0,
          'top': 0.0,
          'right': 4000.0,
          'bottom': 3000.0,
        },
        'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
      };

      final result = YOLOResult.fromMap(map);

      expect(result.boundingBox.width, 4000.0);
      expect(result.boundingBox.height, 3000.0);
    });

    test('fromMap handles negative coordinates', () {
      final map = {
        'classIndex': 0,
        'className': 'test',
        'confidence': 0.95,
        'boundingBox': {
          'left': -10.0,
          'top': -5.0,
          'right': 100.0,
          'bottom': 200.0,
        },
        'normalizedBox': {
          'left': -0.1,
          'top': -0.05,
          'right': 0.5,
          'bottom': 0.9,
        },
      };

      final result = YOLOResult.fromMap(map);

      expect(result.boundingBox.left, -10.0);
      expect(result.boundingBox.top, -5.0);
      expect(result.normalizedBox.left, -0.1);
    });

    test('fromMap handles many keypoints (full body pose)', () {
      // Simulate 17 keypoints (COCO pose format: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
      final keypointsData = <double>[];
      for (var i = 0; i < 17; i++) {
        keypointsData.addAll([
          i * 10.0,
          i * 15.0,
          0.8 + (i * 0.01),
        ]); // x, y, confidence
      }

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
        'keypoints': keypointsData,
      };

      final result = YOLOResult.fromMap(map);

      expect(result.keypoints!.length, 17);
      expect(result.keypointConfidences!.length, 17);
      expect(result.keypoints![0].x, 0.0);
      expect(result.keypoints![0].y, 0.0);
      expect(result.keypointConfidences![0], 0.8);
      expect(result.keypoints![16].x, 160.0);
      expect(result.keypoints![16].y, 240.0);
      expect(
        result.keypointConfidences![16],
        closeTo(0.96, 0.001),
      ); // Use closeTo for floating point
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

    test('fromMap handles empty detections list', () {
      final map = {'detections': <dynamic>[], 'processingTimeMs': 15.0};

      final results = YOLODetectionResults.fromMap(map);

      expect(results.detections, isEmpty);
      expect(results.processingTimeMs, 15.0);
      expect(results.annotatedImage, isNull);
    });

    test('fromMap handles null detections', () {
      final map = {'processingTimeMs': 20.0};

      final results = YOLODetectionResults.fromMap(map);

      expect(results.detections, isEmpty);
      expect(results.processingTimeMs, 20.0);
    });

    test('fromMap handles missing processingTimeMs', () {
      final map = {'detections': <dynamic>[]};

      final results = YOLODetectionResults.fromMap(map);

      expect(results.processingTimeMs, 0.0);
    });

    test('fromMap handles multiple detection types', () {
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
            'keypoints': [100.0, 200.0, 0.9],
          },
          {
            'classIndex': 1,
            'className': 'car',
            'confidence': 0.88,
            'boundingBox': {
              'left': 200.0,
              'top': 100.0,
              'right': 300.0,
              'bottom': 250.0,
            },
            'normalizedBox': {
              'left': 0.6,
              'top': 0.2,
              'right': 0.8,
              'bottom': 0.7,
            },
            'mask': [
              [0.1, 0.2],
              [0.3, 0.4],
            ],
          },
        ],
        'processingTimeMs': 45.2,
      };

      final results = YOLODetectionResults.fromMap(map);

      expect(results.detections.length, 2);
      expect(results.detections[0].className, 'person');
      expect(results.detections[0].keypoints, isNotNull);
      expect(results.detections[1].className, 'car');
      expect(results.detections[1].mask, isNotNull);
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
      expect(map['annotatedImage'], isNull);
    });

    test('toMap includes annotated image when present', () {
      final imageData = Uint8List.fromList([255, 128, 64, 32]);
      final results = YOLODetectionResults(
        detections: [],
        processingTimeMs: 30.0,
        annotatedImage: imageData,
      );

      final map = results.toMap();

      expect(map['annotatedImage'], equals(imageData));
    });

    test('handles very fast processing times', () {
      final results = YOLODetectionResults(
        detections: [],
        processingTimeMs: 0.001, // 1 microsecond
      );

      expect(results.processingTimeMs, 0.001);
    });

    test('handles very slow processing times', () {
      final results = YOLODetectionResults(
        detections: [],
        processingTimeMs: 5000.0, // 5 seconds
      );

      expect(results.processingTimeMs, 5000.0);
    });
  });

  group('Point', () {
    test('fromMap creates point from coordinates', () {
      final map = {'x': 10.5, 'y': 20.7};
      final point = Point.fromMap(map);

      expect(point.x, 10.5);
      expect(point.y, 20.7);
    });

    test('fromMap handles integer coordinates', () {
      final map = {'x': 15, 'y': 25};
      final point = Point.fromMap(map);

      expect(point.x, 15.0);
      expect(point.y, 25.0);
    });

    test('fromMap handles zero coordinates', () {
      final map = {'x': 0, 'y': 0};
      final point = Point.fromMap(map);

      expect(point.x, 0.0);
      expect(point.y, 0.0);
    });

    test('fromMap handles negative coordinates', () {
      final map = {'x': -10.5, 'y': -20.3};
      final point = Point.fromMap(map);

      expect(point.x, -10.5);
      expect(point.y, -20.3);
    });

    test('fromMap handles very large coordinates', () {
      final map = {'x': 9999999.99, 'y': 8888888.88};
      final point = Point.fromMap(map);

      expect(point.x, 9999999.99);
      expect(point.y, 8888888.88);
    });

    test('fromMap handles very small coordinates', () {
      final map = {'x': 0.000001, 'y': 0.000002};
      final point = Point.fromMap(map);

      expect(point.x, 0.000001);
      expect(point.y, 0.000002);
    });

    test('toMap serializes point coordinates', () {
      final point = Point(15.2, 30.8);
      final map = point.toMap();

      expect(map['x'], 15.2);
      expect(map['y'], 30.8);
    });

    test('toMap handles zero coordinates', () {
      final point = Point(0.0, 0.0);
      final map = point.toMap();

      expect(map['x'], 0.0);
      expect(map['y'], 0.0);
    });

    test('toMap handles negative coordinates', () {
      final point = Point(-5.5, -10.2);
      final map = point.toMap();

      expect(map['x'], -5.5);
      expect(map['y'], -10.2);
    });

    test('toString formats correctly', () {
      final point = Point(42.5, 37.8);
      expect(point.toString(), 'Point(42.5, 37.8)');
    });

    test('toString handles integer-like values', () {
      final point = Point(10.0, 20.0);
      expect(point.toString(), 'Point(10.0, 20.0)');
    });

    test('toString handles negative values', () {
      final point = Point(-15.3, -25.7);
      expect(point.toString(), 'Point(-15.3, -25.7)');
    });

    test('round-trip serialization preserves values', () {
      final original = Point(123.456, 789.012);
      final map = original.toMap();
      final restored = Point.fromMap(map);

      expect(restored.x, original.x);
      expect(restored.y, original.y);
    });
  });

  group('YOLOResult toString formatting', () {
    test('toString formats with all fields correctly', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTRB(10.0, 20.0, 110.0, 220.0),
        normalizedBox: const Rect.fromLTRB(0.1, 0.1, 0.9, 0.9),
        keypoints: [Point(50, 60), Point(55, 65)],
        mask: [
          [0.0, 1.0],
          [1.0, 0.0],
        ],
      );

      final str = result.toString();
      expect(str, contains('YOLOResult'));
      expect(str, contains('person'));
      expect(str, contains('0.95'));
      expect(str, contains('Rect.fromLTRB(10.0, 20.0, 110.0, 220.0)'));
      // toString doesn't include keypoints or mask
      expect(result.keypoints!.length, 2);
      expect(result.mask!.length, 2);
    });

    test('toString formats without optional fields', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'car',
        confidence: 0.85,
        boundingBox: const Rect.fromLTRB(0.0, 0.0, 100.0, 100.0),
        normalizedBox: const Rect.fromLTRB(0.0, 0.0, 1.0, 1.0),
      );

      final str = result.toString();
      expect(str, contains('YOLOResult'));
      expect(str, contains('car'));
      expect(str, contains('0.85'));
      expect(str, isNot(contains('keypoints')));
      expect(str, isNot(contains('mask')));
    });

    test('toString with different values', () {
      final result = YOLOResult(
        classIndex: 2,
        className: 'car',
        confidence: 0.87,
        boundingBox: const Rect.fromLTRB(50.0, 75.0, 250.0, 225.0),
        normalizedBox: const Rect.fromLTRB(0.2, 0.3, 0.4, 0.5),
      );

      final str = result.toString();
      expect(
        str,
        'YOLOResult{classIndex: 2, className: car, confidence: 0.87, boundingBox: Rect.fromLTRB(50.0, 75.0, 250.0, 225.0)}',
      );
    });
  });

  group('YOLOResult edge cases', () {
    test('fromMap handles missing optional fields', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
        // No optional fields
      };

      final result = YOLOResult.fromMap(map);
      expect(result.classIndex, 0);
      expect(result.className, 'person');
      expect(result.confidence, 0.95);
      expect(result.keypoints, isNull);
      expect(result.keypointConfidences, isNull);
      expect(result.mask, isNull);
    });

    test('BoundingBox getters calculate correctly', () {
      final map = {
        'classIndex': 0,
        'className': 'test',
        'confidence': 0.9,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
      };

      final result = YOLOResult.fromMap(map);
      final box = result.boundingBox;

      expect(box.left, 10);
      expect(box.top, 20);
      expect(box.width, 100);
      expect(box.height, 200);
      expect(box.center.dx, 60);
      expect(box.center.dy, 120);
    });

    test('Keypoint parsing from flat array', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.9,
        'boundingBox': {
          'left': 0.0,
          'top': 0.0,
          'right': 100.0,
          'bottom': 100.0,
        },
        'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
        'keypoints': [10.5, 20.5, 0.9], // x, y, confidence format
      };

      final result = YOLOResult.fromMap(map);
      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, 1);
      final keypoint = result.keypoints!.first;
      expect(keypoint.x, 10.5);
      expect(keypoint.y, 20.5);
    });

    test('BoundingBox toString formats correctly', () {
      final map = {
        'classIndex': 0,
        'className': 'test',
        'confidence': 0.9,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
      };

      final result = YOLOResult.fromMap(map);
      final str = result.boundingBox.toString();
      expect(str, contains('Rect.fromLTRB(10.0, 20.0, 110.0, 220.0)'));
    });
  });

  group('YOLOResult toString formatting (moved from task-specific)', () {
    test('YOLOResult toString formats correctly', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTWH(10, 20, 100, 200),
        normalizedBox: const Rect.fromLTWH(0.1, 0.1, 0.8, 0.8),
        keypoints: [Point(50, 60), Point(55, 65)],
        mask: [
          [0.0, 1.0],
          [1.0, 0.0],
        ],
      );

      final str = result.toString();
      expect(
        str,
        'YOLOResult{classIndex: 0, className: person, confidence: 0.95, boundingBox: Rect.fromLTRB(10.0, 20.0, 110.0, 220.0)}',
      );
    });

    test('Point toString formats correctly', () {
      final point = Point(123.45, 678.90);
      expect(point.toString(), 'Point(123.45, 678.9)');
    });

    test('YOLOResult toString with different values', () {
      final result = YOLOResult(
        classIndex: 2,
        className: 'car',
        confidence: 0.87,
        boundingBox: const Rect.fromLTWH(50, 75, 200, 150),
        normalizedBox: const Rect.fromLTWH(0.2, 0.3, 0.4, 0.5),
      );

      final str = result.toString();
      expect(
        str,
        'YOLOResult{classIndex: 2, className: car, confidence: 0.87, boundingBox: Rect.fromLTRB(50.0, 75.0, 250.0, 225.0)}',
      );
    });
  });

  group('Task-specific result handling (moved from task-specific)', () {
    test('predict handles mask data correctly for segmentation', () async {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
        'mask': [
          [0.0, 0.5, 1.0],
          [0.2, 0.8, 0.3],
          [1.0, 0.5, 0.0],
        ],
      };

      final result = YOLOResult.fromMap(map);
      expect(result.mask, isNotNull);
      expect(result.mask!.length, 3);
      expect(result.mask![0].length, 3);
      expect(result.mask![0][0], 0.0);
      expect(result.mask![0][1], 0.5);
      expect(result.mask![0][2], 1.0);
    });

    test('predict handles keypoints data correctly for pose estimation', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
        'keypoints': [
          100.0, 50.0, 0.9, // nose
          95.0, 55.0, 0.85, // left eye
          105.0, 55.0, 0.87, // right eye
        ],
      };

      final result = YOLOResult.fromMap(map);
      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, 3);
      expect(result.keypointConfidences, isNotNull);
      expect(result.keypointConfidences!.length, 3);

      // Check first keypoint
      expect(result.keypoints![0].x, 100.0);
      expect(result.keypoints![0].y, 50.0);
      expect(result.keypointConfidences![0], 0.9);
    });
  });
}
