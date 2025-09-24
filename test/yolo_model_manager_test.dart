// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'utils/test_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOModelManager', () {
    late MethodChannel mockChannel;
    late List<MethodCall> log;

    setUp(() {
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      mockChannel = setup.$1;
      log = setup.$2;
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(mockChannel, null);
      log.clear();
    });

    test('constructor initializes correctly', () {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      expect(manager, isNotNull);
    });

    test('constructor with classifier options', () {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'classifier_model.tflite',
        task: YOLOTask.classify,
        useGpu: false,
        classifierOptions: {
          'enable1ChannelSupport': true,
          'expectedChannels': 1,
        },
        viewId: 123,
      );

      expect(manager, isNotNull);
    });

    test('initializeInstance for non-default instance', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'custom_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      await manager.initializeInstance();

      YOLOTestHelpers.assertMethodCalled(
        log,
        'createInstance',
        arguments: {'instanceId': 'custom_instance'},
      );
    });

    test('initializeInstance for default instance', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'default',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      await manager.initializeInstance();

      // Should not call createInstance for default instance
      expect(log.where((call) => call.method == 'createInstance'), isEmpty);
    });

    test('loadModel works correctly', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      final result = await manager.loadModel();

      expect(result, isTrue);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'loadModel',
        arguments: {
          'modelPath': 'test_model.tflite',
          'task': 'detect',
          'useGpu': true,
          'instanceId': 'test_instance',
        },
      );
    });

    test('loadModel with classifier options', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'classifier_model.tflite',
        task: YOLOTask.classify,
        useGpu: false,
        classifierOptions: {
          'enable1ChannelSupport': true,
          'expectedChannels': 1,
        },
      );

      await manager.loadModel();

      YOLOTestHelpers.assertMethodCalled(
        log,
        'loadModel',
        arguments: {
          'modelPath': 'classifier_model.tflite',
          'task': 'classify',
          'useGpu': false,
          'classifierOptions': {
            'enable1ChannelSupport': true,
            'expectedChannels': 1,
          },
          'instanceId': 'test_instance',
        },
      );
    });

    test('loadModel handles platform exceptions', () async {
      final errorChannel = YOLOTestHelpers.setupMockChannel(
        customResponses: {
          'loadModel': (_) => throw PlatformException(
            code: 'MODEL_NOT_FOUND',
            message: 'Model not found',
          ),
        },
      );

      final manager = YOLOModelManager(
        channel: errorChannel,
        instanceId: 'test_instance',
        modelPath: 'nonexistent_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      expect(() => manager.loadModel(), throwsA(isA<YOLOException>()));
    });

    test('switchModel works correctly', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'old_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
        viewId: 123,
      );

      await manager.switchModel('new_model.tflite', YOLOTask.segment);

      YOLOTestHelpers.assertMethodCalled(
        log,
        'setModel',
        arguments: {
          'viewId': 123,
          'modelPath': 'new_model.tflite',
          'task': 'segment',
          'useGpu': true,
          'instanceId': 'test_instance',
        },
      );
    });

    test('switchModel throws StateError when view not initialized', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      expect(
        () => manager.switchModel('new_model.tflite', YOLOTask.segment),
        throwsA(isA<StateError>()),
      );
    });

    test('switchModel handles platform exceptions', () async {
      final errorChannel = YOLOTestHelpers.setupMockChannel(
        customResponses: {
          'setModel': (_) => throw PlatformException(
            code: 'MODEL_SWITCH_ERROR',
            message: 'Failed to switch model',
          ),
        },
      );

      final manager = YOLOModelManager(
        channel: errorChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
        viewId: 123,
      );

      expect(
        () => manager.switchModel('new_model.tflite', YOLOTask.segment),
        throwsA(isA<YOLOException>()),
      );
    });

    test('setViewId works correctly', () {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      manager.setViewId(456);
      expect(manager, isNotNull);
    });

    test('dispose works correctly', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      await manager.dispose();

      YOLOTestHelpers.assertMethodCalled(
        log,
        'disposeInstance',
        arguments: {'instanceId': 'test_instance'},
      );
    });

    test('dispose handles errors gracefully', () async {
      final errorChannel = YOLOTestHelpers.setupMockChannel(
        customResponses: {
          'disposeInstance': (_) => throw PlatformException(
            code: 'DISPOSE_ERROR',
            message: 'Failed to dispose',
          ),
        },
      );

      final manager = YOLOModelManager(
        channel: errorChannel,
        instanceId: 'test_instance',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      // Should not throw exception
      await manager.dispose();
      expect(true, isTrue);
    });

    test('loadModel with default instance ID', () async {
      final manager = YOLOModelManager(
        channel: mockChannel,
        instanceId: 'default',
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );

      await manager.loadModel();

      YOLOTestHelpers.assertMethodCalled(
        log,
        'loadModel',
        arguments: {
          'modelPath': 'test_model.tflite',
          'task': 'detect',
          'useGpu': true,
        },
      );
    });
  });
}
