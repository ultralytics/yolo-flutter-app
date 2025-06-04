// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOInstanceManager', () {
    setUp(() {
      // Clear all instances before each test
      final activeIds = YOLOInstanceManager.getActiveInstanceIds();
      for (final id in activeIds) {
        YOLOInstanceManager.unregisterInstance(id);
      }
    });

    test('should start with no active instances', () {
      expect(YOLOInstanceManager.getActiveInstanceIds(), isEmpty);
    });

    test('should register and retrieve an instance', () {
      final yolo = YOLO(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final instanceId = yolo.instanceId;

      // Instance should be auto-registered
      expect(YOLOInstanceManager.hasInstance(instanceId), isTrue);
      expect(YOLOInstanceManager.getInstance(instanceId), equals(yolo));
    });

    test('should register multiple instances', () {
      final yolo1 = YOLO(
        modelPath: 'model1.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final yolo2 = YOLO(
        modelPath: 'model2.tflite',
        task: YOLOTask.segment,
        useMultiInstance: true,
      );

      expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(2));
      expect(YOLOInstanceManager.hasInstance(yolo1.instanceId), isTrue);
      expect(YOLOInstanceManager.hasInstance(yolo2.instanceId), isTrue);
    });

    test('should unregister instance', () {
      final yolo = YOLO(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final instanceId = yolo.instanceId;

      expect(YOLOInstanceManager.hasInstance(instanceId), isTrue);

      YOLOInstanceManager.unregisterInstance(instanceId);

      expect(YOLOInstanceManager.hasInstance(instanceId), isFalse);
      expect(YOLOInstanceManager.getInstance(instanceId), isNull);
      expect(YOLOInstanceManager.getActiveInstanceIds(), isEmpty);
    });

    test('should return null for non-existent instance', () {
      expect(YOLOInstanceManager.getInstance('non_existent_id'), isNull);
      expect(YOLOInstanceManager.hasInstance('non_existent_id'), isFalse);
    });

    test('should handle multiple registrations and unregistrations', () {
      final instances = <YOLO>[];

      // Register multiple instances
      for (int i = 0; i < 5; i++) {
        instances.add(
          YOLO(
            modelPath: 'model_$i.tflite',
            task: YOLOTask.detect,
            useMultiInstance: true,
          ),
        );
      }

      expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(5));

      // Unregister some instances
      YOLOInstanceManager.unregisterInstance(instances[1].instanceId);
      YOLOInstanceManager.unregisterInstance(instances[3].instanceId);

      expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(3));
      expect(YOLOInstanceManager.hasInstance(instances[0].instanceId), isTrue);
      expect(YOLOInstanceManager.hasInstance(instances[1].instanceId), isFalse);
      expect(YOLOInstanceManager.hasInstance(instances[2].instanceId), isTrue);
      expect(YOLOInstanceManager.hasInstance(instances[3].instanceId), isFalse);
      expect(YOLOInstanceManager.hasInstance(instances[4].instanceId), isTrue);
    });

    test('should not break when unregistering non-existent instance', () {
      // This should not throw an error
      expect(
        () => YOLOInstanceManager.unregisterInstance('non_existent_id'),
        returnsNormally,
      );
    });

    test('should return unique instance IDs', () {
      final yolo1 = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final yolo2 = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(yolo1.instanceId, isNot(equals(yolo2.instanceId)));
      expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(2));
    });

    test('should handle re-registration of same instance ID', () {
      final yolo1 = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final instanceId = yolo1.instanceId;

      // Create a new instance and manually register with same ID
      final yolo2 = YOLO(
        modelPath: 'model2.tflite',
        task: YOLOTask.segment,
        useMultiInstance: true,
      );

      // Re-register with same ID (this should replace the instance)
      YOLOInstanceManager.registerInstance(instanceId, yolo2);

      expect(YOLOInstanceManager.getInstance(instanceId), equals(yolo2));
      expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(2));
    });

    test('default YOLO instances should not be registered', () {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        // useMultiInstance = false by default
      );

      expect(yolo.instanceId, equals('default'));
      expect(YOLOInstanceManager.hasInstance('default'), isFalse);
      expect(YOLOInstanceManager.getActiveInstanceIds(), isEmpty);
    });
  });
}
