import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/predict/segment/detected_segment.dart';

void main() {
  group('DetectedSegment - Object Creation', () {
    test('should create instance with valid data', () {
      final segment = DetectedSegment(
        label: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4),
        polygons: [
          [
            const Offset(0.1, 0.2),
            const Offset(0.3, 0.2),
            const Offset(0.3, 0.4),
            const Offset(0.1, 0.4),
          ],
        ],
        index: 0,
      );

      expect(segment.label, equals('person'));
      expect(segment.confidence, equals(0.95));
      expect(
        segment.boundingBox,
        equals(
          const Rect.fromLTWH(
            0.1,
            0.2,
            0.3,
            0.4,
          ),
        ),
      );
      expect(segment.polygons, hasLength(1));
      expect(segment.polygons[0], hasLength(4));
    });

    test('should create instance with boundary confidence values', () {
      final segment = DetectedSegment(
        label: 'person',
        confidence: 1,
        boundingBox: const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4),
        polygons: [],
        index: 0,
      );

      expect(segment.confidence, equals(1.0));
    });
  });

  group('DetectedSegment - Edge Cases', () {
    test('should create instance with empty polygons', () {
      final segment = DetectedSegment(
        label: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4),
        polygons: [],
        index: 0,
      );

      expect(segment.polygons, isEmpty);
    });

    test('should create instance with empty label', () {
      final segment = DetectedSegment(
        label: '',
        confidence: 0.95,
        boundingBox: const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4),
        polygons: [],
        index: 0,
      );

      expect(segment.label, isEmpty);
    });
  });

  group('DetectedSegment - JSON Parsing', () {
    test('should parse valid JSON data', () {
      final json = {
        'label': 'person',
        'confidence': 0.95,
        'x': 0.1,
        'y': 0.2,
        'width': 0.3,
        'height': 0.4,
        'index': 0,
        'polygons': [
          [
            [0.1, 0.2],
            [0.3, 0.2],
            [0.3, 0.4],
            [0.1, 0.4],
          ]
        ],
      };

      final segment = DetectedSegment.fromJson(json);

      expect(segment.label, equals('person'));
      expect(segment.confidence, equals(0.95));
      expect(
        segment.boundingBox,
        equals(const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4)),
      );
      expect(segment.index, equals(0));
      expect(segment.polygons, hasLength(1));
      expect(segment.polygons[0], hasLength(4));
    });

    test('should handle missing optional fields in JSON', () {
      final json = {
        'label': 'person',
        'confidence': 0.95,
        'x': 0.1,
        'y': 0.2,
        'width': 0.3,
        'height': 0.4,
        'index': 0,
      };

      final segment = DetectedSegment.fromJson(json);

      expect(segment.label, equals('person'));
      expect(segment.confidence, equals(0.95));
      expect(
        segment.boundingBox,
        equals(const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4)),
      );
      expect(segment.index, equals(0));
      expect(segment.polygons, isEmpty);
    });
  });
}
