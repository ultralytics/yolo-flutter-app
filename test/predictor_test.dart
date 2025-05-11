import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/predict/classify/image_classifier.dart';
import 'package:ultralytics_yolo/predict/detect/object_detector.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('Predictor Tests', () {
    late LocalYoloModel model;

    setUp(() {
      debugPrint('Setting up predictor test...');
      model = LocalYoloModel(
        id: 'test-model',
        task: Task.detect,
        format: Format.coreml,
        modelPath: 'test/assets/model.mlmodel',
        metadataPath: 'test/assets/metadata.json',
      );
    });

    test('ObjectDetector creation with LocalYoloModel', () {
      debugPrint('Running: ObjectDetector creation with LocalYoloModel');
      try {
        final predictor = ObjectDetector(model: model);
        expect(predictor.model, equals(model));
        debugPrint('✓ ObjectDetector creation passed');
      } catch (e) {
        debugPrint('Error in test: $e');
        rethrow;
      }
    });

    test('ImageClassifier creation with RemoteYoloModel', () {
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
}
