import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/predict/detect/detected_object.dart';
import 'package:ultralytics_yolo/predict/detect/object_detector.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

class MockYoloModel extends Fake implements YoloModel {}

class MockPlatform extends UltralyticsYoloPlatform {
  Stream<List<DetectedObject?>?>? _stream;
  Future<List<DetectedObject?>?> Function(String)? _detectImage;
  double? confidence;
  double? iou;
  int? numItems;

  @override
  Stream<List<DetectedObject?>?> get detectionResultStream =>
      _stream ?? Stream.value([]);

  @override
  Future<List<DetectedObject?>?> detectImage(String imagePath) =>
      _detectImage != null ? _detectImage!(imagePath) : Future.value([]);

  @override
  Future<String?> setConfidenceThreshold(double c) async {
    confidence = c;
    return 'ok';
  }

  @override
  Future<String?> setIouThreshold(double i) async {
    iou = i;
    return 'ok';
  }

  @override
  Future<String?> setNumItemsThreshold(int n) async {
    numItems = n;
    return 'ok';
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  group('ObjectDetector', () {
    late MockYoloModel mockModel;
    late MockPlatform mockPlatform;
    late ObjectDetector detector;
    late List<DetectedObject?> mockResults;

    setUp(() {
      mockModel = MockYoloModel();
      mockPlatform = MockPlatform();
      mockResults = [
        DetectedObject(
          label: 'person',
          confidence: 0.9,
          boundingBox: const Rect.fromLTWH(0.1, 0.2, 0.3, 0.4),
          index: 0,
        ),
      ];
      mockPlatform
        .._stream = Stream.value(mockResults)
        .._detectImage = (imagePath) async => mockResults;
      UltralyticsYoloPlatform.instance = mockPlatform;
      detector = ObjectDetector(model: mockModel);
    });

    group('Construction', () {
      test('should create instance with valid model', () {
        expect(detector, isA<ObjectDetector>());
      });
    });

    group('Streams', () {
      test('should emit detection results from platform stream', () {
        expect(detector.detectionResultStream, emits(mockResults));
      });
    });

    group('Thresholds', () {
      test('should set confidence threshold', () {
        detector.setConfidenceThreshold(0.5);
        expect(mockPlatform.confidence, 0.5);
      });

      test('should set IOU threshold', () {
        detector.setIouThreshold(0.7);
        expect(mockPlatform.iou, 0.7);
      });

      test('should set number of items threshold', () {
        detector.setNumItemsThreshold(3);
        expect(mockPlatform.numItems, 3);
      });
    });

    group('Detection', () {
      test('should detect objects in image', () async {
        final result = await detector.detect(imagePath: 'test.jpg');
        expect(result, mockResults);
      });
    });
  });
}
