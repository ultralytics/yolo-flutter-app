// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/predict/classify/image_classifier.dart';
import 'package:ultralytics_yolo/predict/detect/object_detector.dart';
import 'package:ultralytics_yolo/predict/predictor.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

class MockYoloModel extends Fake implements YoloModel {
  @override
  String get id => 'mock-id';

  @override
  Map<String, dynamic> toJson() => {'id': id};
}

class MockPlatform extends UltralyticsYoloPlatform {
  String? loadedModelId;
  bool? loadedUseGpu;
  @override
  Future<String?> loadModel(
    Map<String, dynamic> model, {
    bool useGpu = false,
  }) async {
    loadedModelId = model['id'] as String?;
    loadedUseGpu = useGpu;
    return 'loaded';
  }

  @override
  Stream<double>? get inferenceTimeStream => Stream.value(42);
  @override
  Stream<double>? get fpsRateStream => Stream.value(24);
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class TestPredictor extends Predictor {
  TestPredictor(super.model);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('Predictor Tests', () {
    late LocalYoloModel localModel;
    late MockYoloModel mockModel;
    late MockPlatform mockPlatform;

    setUp(() {
      debugPrint('Setting up predictor test...');
      localModel = LocalYoloModel(
        id: 'test-model',
        task: Task.detect,
        format: Format.coreml,
        modelPath: 'test/assets/model.mlmodel',
        metadataPath: 'test/assets/metadata.json',
      );
      mockModel = MockYoloModel();
      mockPlatform = MockPlatform();
      UltralyticsYoloPlatform.instance = mockPlatform;
    });

    group('Abstract Predictor', () {
      late TestPredictor predictor;

      setUp(() {
        predictor = TestPredictor(mockModel);
      });

      group('Construction', () {
        test('should create instance with valid model', () {
          expect(predictor, isA<TestPredictor>());
        });
      });

      group('Model Loading', () {
        test('should load model with correct parameters', () async {
          final result = await predictor.loadModel(useGpu: true);
          expect(result, 'loaded');
          expect(mockPlatform.loadedModelId, mockModel.id);
          expect(mockPlatform.loadedUseGpu, true);
        });
      });

      group('Streams', () {
        test('should emit inference time from platform stream', () async {
          expect(predictor.inferenceTime, emits(42.0));
        });

        test('should emit FPS rate from platform stream', () async {
          expect(predictor.fpsRate, emits(24.0));
        });
      });
    });

    group('Concrete Predictors', () {
      test('should create ObjectDetector with LocalYoloModel', () {
        debugPrint('Running: ObjectDetector creation with LocalYoloModel');
        try {
          final predictor = ObjectDetector(model: localModel);
          expect(predictor.model, equals(localModel));
          debugPrint('✓ ObjectDetector creation passed');
        } catch (e) {
          debugPrint('Error in test: $e');
          rethrow;
        }
      });

      test('should create ImageClassifier with RemoteYoloModel', () {
        debugPrint('Running: ImageClassifier creation with RemoteYoloModel');
        try {
          final remoteModel = RemoteYoloModel(
            id: 'remote-model',
            modelUrl: 'https://example.com/model.mlmodel',
            task: Task.classify,
            format: Format.coreml,
          );
          final predictor = ImageClassifier(model: remoteModel);
          expect(predictor.model, equals(remoteModel));
          debugPrint('✓ ImageClassifier creation passed');
        } catch (e) {
          debugPrint('Error in test: $e');
          rethrow;
        }
      });
    });
  });
}
