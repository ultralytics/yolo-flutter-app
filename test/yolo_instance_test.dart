import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Instance ID Tests', () {
    const MethodChannel defaultChannel = MethodChannel('yolo_single_image_channel');
    final List<MethodCall> methodCalls = [];
    
    setUp(() {
      methodCalls.clear();
      
      // Set up default channel mock
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(defaultChannel, (MethodCall methodCall) async {
        methodCalls.add(methodCall);
        
        switch (methodCall.method) {
          case 'createInstance':
            return null;
          case 'loadModel':
            return true;
          case 'predictSingleImage':
            return {
              'boxes': [
                {
                  'x1': 100.0,
                  'y1': 100.0,
                  'x2': 200.0,
                  'y2': 200.0,
                  'class': 0,
                  'confidence': 0.95
                }
              ],
              'speed': {'preprocess': 2.0, 'inference': 15.0, 'postprocess': 3.0}
            };
          case 'disposeInstance':
            return null;
          default:
            return null;
        }
      });
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(defaultChannel, null);
    });

    test('Create and manage multiple instances', () async {
      // Create first instance
      final instanceId1 = await YOLO.createInstance();
      expect(instanceId1, isNotNull);
      expect(instanceId1, startsWith('yolo_'));
      
      // Create second instance
      final instanceId2 = await YOLO.createInstance();
      expect(instanceId2, isNotNull);
      expect(instanceId2, isNot(equals(instanceId1)));
      
      // Verify createInstance was called for each
      final createCalls = methodCalls.where((call) => call.method == 'createInstance').toList();
      expect(createCalls.length, 2);
      expect(createCalls[0].arguments['instanceId'], instanceId1);
      expect(createCalls[1].arguments['instanceId'], instanceId2);
    });

    test('Load models for different instances', () async {
      // Create instances
      final instanceId1 = await YOLO.createInstance();
      final instanceId2 = await YOLO.createInstance();
      
      // Load different models
      final success1 = await YOLO.loadModelWithInstance(
        instanceId: instanceId1,
        model: 'yolov8n.tflite',
        task: YOLOTask.detect,
      );
      expect(success1, isTrue);
      
      final success2 = await YOLO.loadModelWithInstance(
        instanceId: instanceId2,
        model: 'yolov8s-seg.tflite',
        task: YOLOTask.segment,
      );
      expect(success2, isTrue);
      
      // Verify loadModel calls
      final loadCalls = methodCalls.where((call) => call.method == 'loadModel').toList();
      expect(loadCalls.length, 2);
      expect(loadCalls[0].arguments['instanceId'], instanceId1);
      expect(loadCalls[0].arguments['modelPath'], 'yolov8n.tflite');
      expect(loadCalls[0].arguments['task'], 'detect');
      expect(loadCalls[1].arguments['instanceId'], instanceId2);
      expect(loadCalls[1].arguments['modelPath'], 'yolov8s-seg.tflite');
      expect(loadCalls[1].arguments['task'], 'segment');
    });

    test('Run inference on different instances', () async {
      // Create and load instances
      final instanceId1 = await YOLO.createInstance();
      final instanceId2 = await YOLO.createInstance();
      
      await YOLO.loadModelWithInstance(
        instanceId: instanceId1,
        model: 'yolov8n.tflite',
        task: YOLOTask.detect,
      );
      
      await YOLO.loadModelWithInstance(
        instanceId: instanceId2,
        model: 'yolov8s.tflite',
        task: YOLOTask.detect,
      );
      
      // Test image data
      final testImage = Uint8List.fromList([0, 1, 2, 3, 4, 5]);
      
      // Run inference on both instances
      final result1 = await YOLO.detectImageWithInstance(
        instanceId: instanceId1,
        imageBytes: testImage,
        confidenceThreshold: 0.3,
      );
      
      final result2 = await YOLO.detectImageWithInstance(
        instanceId: instanceId2,
        imageBytes: testImage,
        confidenceThreshold: 0.5,
        iouThreshold: 0.6,
      );
      
      // Verify results
      expect(result1, isNotNull);
      expect(result1['boxes'], isNotNull);
      expect(result2, isNotNull);
      expect(result2['boxes'], isNotNull);
      
      // Verify inference calls
      final predictCalls = methodCalls.where((call) => call.method == 'predictSingleImage').toList();
      expect(predictCalls.length, 2);
      expect(predictCalls[0].arguments['instanceId'], instanceId1);
      expect(predictCalls[0].arguments['confidenceThreshold'], 0.3);
      expect(predictCalls[1].arguments['instanceId'], instanceId2);
      expect(predictCalls[1].arguments['confidenceThreshold'], 0.5);
      expect(predictCalls[1].arguments['iouThreshold'], 0.6);
    });

    test('Dispose instances', () async {
      // Create instances
      final instanceId1 = await YOLO.createInstance();
      final instanceId2 = await YOLO.createInstance();
      
      // Dispose first instance
      await YOLO.disposeInstance(instanceId1);
      
      // Verify dispose was called
      final disposeCalls = methodCalls.where((call) => call.method == 'disposeInstance').toList();
      expect(disposeCalls.length, 1);
      expect(disposeCalls[0].arguments['instanceId'], instanceId1);
      
      // Dispose second instance
      await YOLO.disposeInstance(instanceId2);
      
      // Verify second dispose
      final allDisposeCalls = methodCalls.where((call) => call.method == 'disposeInstance').toList();
      expect(allDisposeCalls.length, 2);
      expect(allDisposeCalls[1].arguments['instanceId'], instanceId2);
    });

    test('Handle invalid instance ID', () async {
      final testImage = Uint8List.fromList([0, 1, 2, 3, 4, 5]);
      
      // Try to use non-existent instance
      expect(
        () => YOLO.detectImageWithInstance(
          instanceId: 'invalid_instance',
          imageBytes: testImage,
        ),
        throwsA(isA<InvalidInputException>()),
      );
      
      expect(
        () => YOLO.loadModelWithInstance(
          instanceId: 'invalid_instance',
          model: 'yolov8n.tflite',
          task: YOLOTask.detect,
        ),
        throwsA(isA<InvalidInputException>()),
      );
    });

    test('Backward compatibility with default instance', () async {
      // Create YOLO instance using old API
      final yolo = YOLO(
        modelPath: 'yolov8n.tflite',
        task: YOLOTask.detect,
      );
      
      // Load model should work
      final success = await yolo.loadModel();
      expect(success, isTrue);
      
      // Run inference should work
      final testImage = Uint8List.fromList([0, 1, 2, 3, 4, 5]);
      final result = await yolo.predict(testImage);
      expect(result, isNotNull);
      expect(result['boxes'], isNotNull);
    });

    test('YOLOInstanceManager maintains instance list', () async {
      // Create multiple instances
      final instanceId1 = await YOLO.createInstance();
      final instanceId2 = await YOLO.createInstance();
      final instanceId3 = await YOLO.createInstance();
      
      // Get active instances
      final activeIds = YOLOInstanceManager.getActiveInstanceIds();
      expect(activeIds.length, greaterThanOrEqualTo(3));
      expect(activeIds, contains(instanceId1));
      expect(activeIds, contains(instanceId2));
      expect(activeIds, contains(instanceId3));
      
      // Check instance existence
      expect(YOLOInstanceManager.hasInstance(instanceId1), isTrue);
      expect(YOLOInstanceManager.hasInstance('non_existent'), isFalse);
      
      // Dispose one instance
      await YOLO.disposeInstance(instanceId2);
      
      // Verify instance was removed
      final updatedIds = YOLOInstanceManager.getActiveInstanceIds();
      expect(updatedIds, contains(instanceId1));
      expect(updatedIds, isNot(contains(instanceId2)));
      expect(updatedIds, contains(instanceId3));
    });
  });
}