// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// dart:typed_data is already imported via flutter/services.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:flutter/services.dart';

class MockYoloPlatform with MockPlatformInterfaceMixin implements YoloPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<void> setModel(int viewId, String modelPath, String task) =>
      Future.value();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Set up mock method channel
  const MethodChannel channel = MethodChannel('yolo_single_image_channel');
  final List<MethodCall> log = <MethodCall>[];

  setUp(() {
    // Configure mock response for the channel
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);

          if (methodCall.method == 'loadModel') {
            return true;
          } else if (methodCall.method == 'predictSingleImage') {
            // Return mock detection result
            return {
              'boxes': [
                {
                  'class': 'person',
                  'confidence': 0.95,
                  'x': 10,
                  'y': 10,
                  'width': 100,
                  'height': 200,
                },
              ],
              'annotatedImage': Uint8List.fromList(List.filled(100, 0)),
            };
          }
          return null;
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
    log.clear();
  });

  // Start the tests
  final YoloPlatform initialPlatform = YoloPlatform.instance;

  test('$MethodChannelYolo is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelYolo>());
  });

  group('YOLO Model Loading', () {
    test('loadModel success', () async {
      // Create a YOLO instance for testing
      final testYolo = YOLO(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      // Execute the loadModel method
      final result = await testYolo.loadModel();

      // Verify result
      expect(result, isTrue);

      // Verify the correct method was called with proper parameters
      expect(log, hasLength(1));
      expect(log[0].method, 'loadModel');
      expect(log[0].arguments['modelPath'], 'test_model.tflite');
      expect(log[0].arguments['task'], 'detect');
    });
  });

  group('YOLO Inference', () {
    test('predict returns valid result structure', () async {
      // Create a YOLO instance for testing
      final testYolo = YOLO(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      // Create a dummy image
      final Uint8List dummyImage = Uint8List.fromList(List.filled(100, 0));

      // Execute predict method
      final result = await testYolo.predict(dummyImage);

      // Verify result
      expect(result, isA<Map<String, dynamic>>());
      expect(result.containsKey('boxes'), isTrue);
      expect(result['boxes'], isA<List>());
      expect(result['boxes'].length, 1);
      expect(result['boxes'][0]['class'], 'person');
      expect(result['boxes'][0]['confidence'], 0.95);

      // Verify the correct method was called with proper parameters
      expect(log, hasLength(1));
      expect(log[0].method, 'predictSingleImage');
      expect(log[0].arguments['image'], isA<Uint8List>());
    });
  });

  group('YOLOTask', () {
    test('All task types can be converted to string', () {
      expect(YOLOTask.detect.toString(), contains('detect'));
      expect(YOLOTask.segment.toString(), contains('segment'));
      expect(YOLOTask.classify.toString(), contains('classify'));
      expect(YOLOTask.pose.toString(), contains('pose'));
      expect(YOLOTask.obb.toString(), contains('obb'));
    });

    test('All task types have a valid name', () {
      expect(YOLOTask.detect.name, equals('detect'));
      expect(YOLOTask.segment.name, equals('segment'));
      expect(YOLOTask.classify.name, equals('classify'));
      expect(YOLOTask.pose.name, equals('pose'));
      expect(YOLOTask.obb.name, equals('obb'));
    });
  });
}
