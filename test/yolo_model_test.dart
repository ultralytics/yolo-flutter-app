import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

void main() {
  group('YoloModel Tests', () {
    test('LocalYoloModel creation', () {
      final model = LocalYoloModel(
        id: 'test-model',
        task: Task.detect,
        format: Format.coreml,
        modelPath: 'test/assets/model.mlmodel',
        metadataPath: 'test/assets/metadata.json',
      );

      expect(model.task, equals(Task.detect));
      expect(model.format, equals(Format.coreml));
      expect(model.modelPath, equals('test/assets/model.mlmodel'));
      expect(model.metadataPath, equals('test/assets/metadata.json'));
      expect(model.id, equals('test-model'));
      expect(model.type, equals(Type.local));
    });

    test('RemoteYoloModel creation', () {
      final model = RemoteYoloModel(
        id: 'remote-model',
        modelUrl: 'https://example.com/model.mlmodel',
        task: Task.classify,
        format: Format.coreml,
      );

      expect(model.task, equals(Task.classify));
      expect(model.modelUrl, equals('https://example.com/model.mlmodel'));
      expect(model.id, equals('remote-model'));
      expect(model.type, equals(Type.remote));
      expect(model.format, equals(Format.coreml));
    });
  });
}
