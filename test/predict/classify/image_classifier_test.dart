import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/predict/classify/classification_result.dart';
import 'package:ultralytics_yolo/predict/classify/image_classifier.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

class MockYoloModel extends Fake implements YoloModel {}

class MockPlatform extends UltralyticsYoloPlatform {
  Stream<List<ClassificationResult?>?>? _stream;
  Future<List<ClassificationResult?>?> Function(String)? _classifyImage;

  @override
  Stream<List<ClassificationResult?>?> get classificationResultStream =>
      _stream ?? Stream.value([]);

  @override
  Future<List<ClassificationResult?>?> classifyImage(String imagePath) =>
      _classifyImage != null ? _classifyImage!(imagePath) : Future.value([]);

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  group('ImageClassifier', () {
    late MockYoloModel mockModel;
    late MockPlatform mockPlatform;
    late ImageClassifier classifier;
    late List<ClassificationResult?> mockResults;

    setUp(() {
      mockModel = MockYoloModel();
      mockPlatform = MockPlatform();
      mockResults = [
        ClassificationResult(label: 'cat', confidence: 0.9, index: 0),
        ClassificationResult(label: 'dog', confidence: 0.8, index: 1),
      ];
      mockPlatform
        .._stream = Stream.value(mockResults)
        .._classifyImage = (imagePath) async => mockResults;
      UltralyticsYoloPlatform.instance = mockPlatform;
      classifier = ImageClassifier(model: mockModel);
    });

    group('Construction', () {
      test('should create instance with valid model', () {
        expect(classifier, isA<ImageClassifier>());
      });
    });

    group('Streams', () {
      test('should emit classification results from platform stream', () {
        expect(classifier.classificationResultStream, emits(mockResults));
      });
    });

    group('Classification', () {
      test('should classify image and return results', () async {
        final result = await classifier.classify(imagePath: 'test.jpg');
        expect(result, mockResults);
      });
    });
  });
}
