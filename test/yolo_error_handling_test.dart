// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Error Handling', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('loadModel handles INVALID_MODEL error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'INVALID_MODEL',
                message: 'Invalid model format',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'invalid.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Invalid model format'),
          ),
        ),
      );
    });

    test('loadModel handles UNSUPPORTED_TASK error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'UNSUPPORTED_TASK',
                message: 'Task not supported',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.obb);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Unsupported task type: obb'),
          ),
        ),
      );
    });

    test('loadModel handles MODEL_FILE_ERROR correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'MODEL_FILE_ERROR',
                message: 'Corrupted model file',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to load model: Corrupted model file'),
          ),
        ),
      );
    });

    test('predict handles MODEL_NOT_LOADED error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'MODEL_NOT_LOADED',
                message: 'Model not loaded',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<ModelNotLoadedException>().having(
            (e) => e.message,
            'message',
            contains('Model has not been loaded'),
          ),
        ),
      );
    });

    test('predict handles INVALID_IMAGE error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'INVALID_IMAGE',
                message: 'Invalid image data',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('Invalid image format or corrupted image data'),
          ),
        ),
      );
    });

    test('predict handles INFERENCE_ERROR correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'INFERENCE_ERROR',
                message: 'GPU out of memory',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<InferenceException>().having(
            (e) => e.message,
            'message',
            contains('Error during inference: GPU out of memory'),
          ),
        ),
      );
    });

    test('loadModel handles MODEL_FILE_ERROR correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'MODEL_FILE_ERROR',
                message: 'Corrupted model file',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to load model: Corrupted model file'),
          ),
        ),
      );
    });

    test('predict handles image load error correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'IMAGE_LOAD_ERROR',
                message: 'Failed to decode image',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<InferenceException>().having(
            (e) => e.message,
            'message',
            contains('Platform error during inference: Failed to decode image'),
          ),
        ),
      );
    });

    test('predict handles inference error correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'INFERENCE_ERROR',
                message: 'GPU out of memory',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<InferenceException>().having(
            (e) => e.message,
            'message',
            contains('Error during inference: GPU out of memory'),
          ),
        ),
      );
    });

    test('switchModel handles MODEL_NOT_FOUND error', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      yolo.setViewId(1);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'MODEL_NOT_FOUND',
                message: 'Model not found',
              );
            }
            return null;
          });

      expect(
        () => yolo.switchModel('missing.tflite', YOLOTask.detect),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Model file not found'),
          ),
        ),
      );
    });

    test('switchModel handles INVALID_MODEL error', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      yolo.setViewId(1);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'INVALID_MODEL',
                message: 'Invalid model',
              );
            }
            return null;
          });

      expect(
        () => yolo.switchModel('invalid.tflite', YOLOTask.detect),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Invalid model format'),
          ),
        ),
      );
    });

    test('switchModel handles UNSUPPORTED_TASK error', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      yolo.setViewId(1);

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'UNSUPPORTED_TASK',
                message: 'Unsupported task',
              );
            }
            return null;
          });

      expect(
        () => yolo.switchModel('model.tflite', YOLOTask.pose),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Unsupported task type'),
          ),
        ),
      );
    });

    test('switchModel without viewId throws StateError', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.switchModel('new.tflite', YOLOTask.detect),
        throwsStateError,
      );
    });
  });
}
