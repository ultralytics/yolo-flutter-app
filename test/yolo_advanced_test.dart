// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Advanced Tests', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');

    setUp(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('loadModel handles all error codes', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);

      // Test MODEL_NOT_FOUND error
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        throw PlatformException(code: 'MODEL_NOT_FOUND');
      });

      expect(
        () => yolo.loadModel(),
        throwsA(isA<ModelLoadingException>().having(
            (e) => e.message, 'message', contains('Model file not found'))),
      );

      // Test INVALID_MODEL error
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        throw PlatformException(code: 'INVALID_MODEL');
      });

      expect(
        () => yolo.loadModel(),
        throwsA(isA<ModelLoadingException>().having(
            (e) => e.message, 'message', contains('Invalid model format'))),
      );

      // Test UNSUPPORTED_TASK error
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        throw PlatformException(code: 'UNSUPPORTED_TASK');
      });

      expect(
        () => yolo.loadModel(),
        throwsA(isA<ModelLoadingException>().having(
            (e) => e.message, 'message', contains('Unsupported task type'))),
      );

      // Test generic PlatformException
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        throw PlatformException(
            code: 'GENERIC_ERROR', message: 'Something went wrong');
      });

      expect(
        () => yolo.loadModel(),
        throwsA(isA<ModelLoadingException>().having(
            (e) => e.message, 'message', contains('Failed to load model'))),
      );

      // Test non-PlatformException
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        throw Exception('Unexpected error');
      });

      expect(
        () => yolo.loadModel(),
        throwsA(isA<ModelLoadingException>().having((e) => e.message, 'message',
            contains('Unknown error loading model'))),
      );
    });

    test('predict handles all error codes', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);
      final imageBytes = Uint8List.fromList([1, 2, 3, 4]);

      // Test MODEL_NOT_LOADED error
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          throw PlatformException(code: 'MODEL_NOT_LOADED');
        }
        return null;
      });

      expect(
        () => yolo.predict(imageBytes),
        throwsA(isA<ModelNotLoadedException>()),
      );

      // Test INVALID_IMAGE error
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          throw PlatformException(code: 'INVALID_IMAGE');
        }
        return null;
      });

      expect(
        () => yolo.predict(imageBytes),
        throwsA(isA<InvalidInputException>()),
      );

      // Test INFERENCE_ERROR
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          throw PlatformException(
              code: 'INFERENCE_ERROR', message: 'GPU error');
        }
        return null;
      });

      expect(
        () => yolo.predict(imageBytes),
        throwsA(isA<InferenceException>().having(
            (e) => e.message, 'message', contains('Error during inference'))),
      );

      // Test generic PlatformException
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          throw PlatformException(
              code: 'UNKNOWN', message: 'Unknown platform error');
        }
        return null;
      });

      expect(
        () => yolo.predict(imageBytes),
        throwsA(isA<InferenceException>().having((e) => e.message, 'message',
            contains('Platform error during inference'))),
      );

      // Test non-PlatformException
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          throw Exception('Network error');
        }
        return null;
      });

      expect(
        () => yolo.predict(imageBytes),
        throwsA(isA<InferenceException>().having((e) => e.message, 'message',
            contains('Unknown error during inference'))),
      );
    });

    test('predict handles empty image data', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);
      final emptyBytes = Uint8List(0);

      expect(
        () => yolo.predict(emptyBytes),
        throwsA(isA<InvalidInputException>().having(
            (e) => e.message, 'message', contains('Image data is empty'))),
      );
    });

    test('predict handles invalid result format', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);
      final imageBytes = Uint8List.fromList([1, 2, 3, 4]);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          // Return invalid format (not a Map)
          return 'invalid result';
        }
        return null;
      });

      expect(
        () => yolo.predict(imageBytes),
        throwsA(isA<InferenceException>().having(
            (e) => e.message, 'message', contains('Invalid result format'))),
      );
    });

    test('predict handles boxes list conversion', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);
      final imageBytes = Uint8List.fromList([1, 2, 3, 4]);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'predictSingleImage') {
          return {
            'boxes': [
              {'class': 'person', 'confidence': 0.9, 'x': 10, 'y': 20},
              'invalid box', // This should be handled gracefully
              {'class': 'car', 'confidence': 0.8, 'x': 30, 'y': 40},
            ],
            'processingTime': 25.5,
          };
        }
        return null;
      });

      final result = await yolo.predict(imageBytes);

      expect(result, isA<Map<String, dynamic>>());
      expect(result['boxes'], isA<List>());
      expect((result['boxes'] as List).length, 3);
      expect((result['boxes'] as List)[0]['class'], 'person');
      expect((result['boxes'] as List)[1],
          isEmpty); // Invalid box becomes empty map
      expect((result['boxes'] as List)[2]['class'], 'car');
    });

    test('checkModelExists handles all response types', () async {
      // Test successful response
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'checkModelExists') {
          return {
            'exists': true,
            'path': '/path/to/model',
            'location': 'internal'
          };
        }
        return null;
      });

      var result = await YOLO.checkModelExists('test.tflite');
      expect(result['exists'], true);
      expect(result['path'], '/path/to/model');
      expect(result['location'], 'internal');

      // Test PlatformException
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'checkModelExists') {
          throw PlatformException(code: 'ERROR', message: 'File system error');
        }
        return null;
      });

      result = await YOLO.checkModelExists('test.tflite');
      expect(result['exists'], false);
      expect(result['error'], 'File system error');

      // Test generic exception
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'checkModelExists') {
          throw Exception('Unexpected error');
        }
        return null;
      });

      result = await YOLO.checkModelExists('test.tflite');
      expect(result['exists'], false);
      expect(result['error'], contains('Exception: Unexpected error'));

      // Test invalid response format
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'checkModelExists') {
          return 'invalid response';
        }
        return null;
      });

      result = await YOLO.checkModelExists('test.tflite');
      expect(result['exists'], false);
      expect(result['location'], 'unknown');
    });

    test('getStoragePaths handles all response types', () async {
      // Test successful response
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'getStoragePaths') {
          return {
            'internal': '/data/user/0/com.example/files',
            'cache': '/data/user/0/com.example/cache',
            'external': null,
            'externalCache': null,
          };
        }
        return null;
      });

      var result = await YOLO.getStoragePaths();
      expect(result['internal'], '/data/user/0/com.example/files');
      expect(result['cache'], '/data/user/0/com.example/cache');
      expect(result['external'], isNull);
      expect(result['externalCache'], isNull);

      // Test PlatformException
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'getStoragePaths') {
          throw PlatformException(code: 'PERMISSION_DENIED');
        }
        return null;
      });

      result = await YOLO.getStoragePaths();
      expect(result, isEmpty);

      // Test generic exception
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'getStoragePaths') {
          throw Exception('Storage error');
        }
        return null;
      });

      result = await YOLO.getStoragePaths();
      expect(result, isEmpty);

      // Test invalid response format
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'getStoragePaths') {
          return 'invalid response';
        }
        return null;
      });

      result = await YOLO.getStoragePaths();
      expect(result, isEmpty);
    });

    test('loadModel returns false on failure', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'loadModel') {
          return false;
        }
        return null;
      });

      final result = await yolo.loadModel();
      expect(result, false);
    });

    test('loadModel returns true on success', () async {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'loadModel') {
          return true;
        }
        return null;
      });

      final result = await yolo.loadModel();
      expect(result, true);
    });
  });
}
