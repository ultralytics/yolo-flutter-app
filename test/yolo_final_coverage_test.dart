// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_exceptions.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Final Coverage Tests', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('getStoragePaths returns paths correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'getStoragePaths') {
          return {
            'documents': '/data/documents',
            'cache': '/data/cache',
            'external': '/storage/external',
          };
        }
        return null;
      });

      final paths = await YOLO.getStoragePaths();
      expect(paths['documents'], '/data/documents');
      expect(paths['cache'], '/data/cache');
      expect(paths['external'], '/storage/external');
    });

    test('checkModelExists returns true when model exists', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'checkModelExists') {
          return {'exists': true, 'path': '/path/to/model.tflite'};
        }
        return null;
      });

      final result = await YOLO.checkModelExists('model.tflite');
      expect(result['exists'], true);
      expect(result['path'], '/path/to/model.tflite');
    });

    test('predict handles missing confidenceThreshold in args', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'loadModel') {
          return true;
        } else if (methodCall.method == 'predictSingleImage') {
          // Verify confidenceThreshold is not in args when not provided
          expect(methodCall.arguments.containsKey('confidenceThreshold'), false);
          return {
            'boxes': [],
            'detections': [],
          };
        }
        return null;
      });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();
      
      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image); // No thresholds provided
    });

    test('predict handles missing iouThreshold in args', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
        if (methodCall.method == 'loadModel') {
          return true;
        } else if (methodCall.method == 'predictSingleImage') {
          // Verify iouThreshold is not in args when not provided
          expect(methodCall.arguments.containsKey('iouThreshold'), false);
          return {
            'boxes': [],
            'detections': [],
          };
        }
        return null;
      });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();
      
      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image); // No thresholds provided
    });
  });
}