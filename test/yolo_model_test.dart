// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

void main() {
  group('YoloModel', () {
    group('LocalYoloModel', () {
      test('should create instance with valid parameters', () {
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

      test('should create instance without metadata path', () {
        final model = LocalYoloModel(
          id: 'test-model',
          task: Task.detect,
          format: Format.coreml,
          modelPath: 'test/assets/model.mlmodel',
        );

        expect(model.metadataPath, isNull);
      });

      test('should convert to JSON with all fields', () {
        final model = LocalYoloModel(
          id: 'abc',
          modelPath: '/path/model.tflite',
          task: Task.detect,
          format: Format.tflite,
          metadataPath: '/path/metadata.yaml',
        );
        final json = model.toJson();
        expect(json['id'], 'abc');
        expect(json['type'], 'local');
        expect(json['task'], 'detect');
        expect(json['format'], 'tflite');
        expect(json['modelPath'], '/path/model.tflite');
        expect(json['metadataPath'], '/path/metadata.yaml');
      });

      test('should convert to JSON without metadata path', () {
        final model = LocalYoloModel(
          id: 'abc',
          modelPath: '/path/model.tflite',
          task: Task.detect,
          format: Format.tflite,
        );
        final json = model.toJson();
        expect(json['metadataPath'], isNull);
      });
    });

    group('RemoteYoloModel', () {
      test('should create instance with valid parameters', () {
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

      test('should convert to JSON with all fields', () {
        final model = RemoteYoloModel(
          id: 'xyz',
          modelUrl: 'https://example.com/model.tflite',
          task: Task.classify,
          format: Format.tflite,
        );
        final json = model.toJson();
        expect(json['id'], 'xyz');
        expect(json['type'], 'remote');
        expect(json['task'], 'classify');
        expect(json['format'], 'tflite');
        expect(json['modelUrl'], 'https://example.com/model.tflite');
      });
    });

    group('Enums', () {
      group('Task', () {
        test('should parse valid task strings', () {
          expect(Task.fromString('detect'), Task.detect);
          expect(Task.fromString('classify'), Task.classify);
          expect(Task.fromString('pose'), Task.pose);
          expect(Task.fromString('segment'), Task.segment);
        });

        test('should throw for invalid task string', () {
          expect(() => Task.fromString('invalid'), throwsStateError);
        });

        test('should convert to string', () {
          expect(Task.detect.toString(), 'Task.detect');
          expect(Task.classify.toString(), 'Task.classify');
          expect(Task.pose.toString(), 'Task.pose');
          expect(Task.segment.toString(), 'Task.segment');
        });
      });

      group('Format', () {
        test('should have correct names and extensions', () {
          expect(Format.coreml.name, 'coreml');
          expect(Format.coreml.extension, '.mlmodel');
          expect(Format.tflite.name, 'tflite');
          expect(Format.tflite.extension, '.tflite');
        });
      });

      group('Type', () {
        test('should have correct values', () {
          expect(Type.local.toString(), 'Type.local');
          expect(Type.remote.toString(), 'Type.remote');
        });
      });
    });
  });
}
