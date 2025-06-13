// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Static Methods', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      log.clear();
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            log.add(methodCall);

            switch (methodCall.method) {
              case 'checkModelExists':
                return {
                  'exists': true,
                  'path': methodCall.arguments['modelPath'],
                  'location': 'assets',
                };
              case 'getStoragePaths':
                return {
                  'internal': '/data/internal',
                  'cache': '/data/cache',
                  'external': '/storage/external',
                };
              default:
                return null;
            }
          });
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('checkModelExists returns model information', () async {
      final result = await YOLO.checkModelExists('test_model.tflite');

      expect(result['exists'], true);
      expect(result['path'], 'test_model.tflite');
      expect(result['location'], 'assets');
      expect(log.last.method, 'checkModelExists');
      expect(log.last.arguments['modelPath'], 'test_model.tflite');
    });

    test('getStoragePaths returns storage locations', () async {
      final paths = await YOLO.getStoragePaths();

      expect(paths['internal'], '/data/internal');
      expect(paths['cache'], '/data/cache');
      expect(paths['external'], '/storage/external');
      expect(log.last.method, 'getStoragePaths');
    });

    test('checkModelExists handles platform exceptions gracefully', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            throw PlatformException(code: 'ERROR', message: 'Test error');
          });

      final result = await YOLO.checkModelExists('test_model.tflite');

      expect(result['exists'], false);
      expect(result['error'], 'Test error');
    });

    test('getStoragePaths handles platform exceptions gracefully', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            throw PlatformException(code: 'ERROR', message: 'Storage error');
          });

      final paths = await YOLO.getStoragePaths();

      expect(paths, isEmpty);
    });
  });

  group('YOLO Error Handling', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      log.clear();
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test(
      'loadModel throws ModelLoadingException for MODEL_NOT_FOUND',
      () async {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
              throw PlatformException(
                code: 'MODEL_NOT_FOUND',
                message: 'Model file not found',
              );
            });

        final yolo = YOLO(
          modelPath: 'missing_model.tflite',
          task: YOLOTask.detect,
        );

        expect(() => yolo.loadModel(), throwsA(isA<ModelLoadingException>()));
      },
    );

    test('loadModel throws ModelLoadingException for INVALID_MODEL', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            throw PlatformException(
              code: 'INVALID_MODEL',
              message: 'Invalid model format',
            );
          });

      final yolo = YOLO(
        modelPath: 'invalid_model.tflite',
        task: YOLOTask.detect,
      );

      expect(() => yolo.loadModel(), throwsA(isA<ModelLoadingException>()));
    });

    test('predict throws InvalidInputException for empty image data', () async {
      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      final emptyImage = Uint8List(0);

      expect(
        () => yolo.predict(emptyImage),
        throwsA(isA<InvalidInputException>()),
      );
    });

    test(
      'predict throws ModelNotLoadedException for MODEL_NOT_LOADED',
      () async {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
              throw PlatformException(
                code: 'MODEL_NOT_LOADED',
                message: 'Model not loaded',
              );
            });

        final yolo = YOLO(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
        );
        final image = Uint8List.fromList([1, 2, 3, 4]);

        expect(
          () => yolo.predict(image),
          throwsA(isA<ModelNotLoadedException>()),
        );
      },
    );

    test('predict throws InvalidInputException for INVALID_IMAGE', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            throw PlatformException(
              code: 'INVALID_IMAGE',
              message: 'Invalid image format',
            );
          });

      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3, 4]);

      expect(() => yolo.predict(image), throwsA(isA<InvalidInputException>()));
    });

    test('predict throws InferenceException for INFERENCE_ERROR', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            throw PlatformException(
              code: 'INFERENCE_ERROR',
              message: 'Inference failed',
            );
          });

      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3, 4]);

      expect(() => yolo.predict(image), throwsA(isA<InferenceException>()));
    });

    test('predict handles invalid result format', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            return 'invalid_result_format';
          });

      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3, 4]);

      expect(() => yolo.predict(image), throwsA(isA<InferenceException>()));
    });
  });

  group('YOLO additional static methods', () {
    test('checkModelExists handles PlatformException', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('yolo_single_image_channel'),
            (MethodCall methodCall) async {
              if (methodCall.method == 'checkModelExists') {
                throw PlatformException(
                  code: 'ERROR',
                  message: 'Platform error',
                );
              }
              return null;
            },
          );

      final result = await YOLO.checkModelExists('model.tflite');
      expect(result['exists'], false);
      expect(result['path'], 'model.tflite');
      expect(result['error'], contains('Platform error'));
    });
  });
}
