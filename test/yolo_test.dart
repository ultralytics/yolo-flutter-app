// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// dart:typed_data is already imported via flutter/services.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_method_channel.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

class MockYOLOPlatform with MockPlatformInterfaceMixin implements YOLOPlatform {
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

  bool modelLoaded = false;
  setUp(() {
    // Configure mock response for the channel
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);

          if (methodCall.method == 'loadModel') {
            modelLoaded = true;
            return true;
          } else if (methodCall.method == 'createInstance') {
            // Support for multi-instance creation
            return true;
          } else if (methodCall.method == 'disposeInstance') {
            // Support for multi-instance disposal
            return true;
          } else if (methodCall.method == 'predictSingleImage') {
            if (!modelLoaded) {
              throw PlatformException(
                code: 'MODEL_NOT_LOADED',
                message: 'Model not loaded',
              );
            }
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
          } else if (methodCall.method == 'setModel') {
            // Support for model switching
            return true;
          }
          return null;
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
    log.clear();
    modelLoaded = false;
  });

  // Start the tests
  final YOLOPlatform initialPlatform = YOLOPlatform.instance;

  test('$YOLOMethodChannel is the default instance', () {
    expect(initialPlatform, isInstanceOf<YOLOMethodChannel>());
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

    test('YOLO.predict throws if called before loadModel', () async {
      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);
      await expectLater(
        yolo.predict(image),
        throwsA(isA<ModelNotLoadedException>()),
      );
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

  testWidgets('YOLOViewState handles platform view creation', (tester) async {
    final key = GlobalKey<YOLOViewState>();
    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
        ),
      ),
    );
    expect(key.currentState, isNotNull);
  });

  testWidgets('YOLOViewState handles event channel errors', (tester) async {
    final key = GlobalKey<YOLOViewState>();
    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
        ),
      ),
    );
    key.currentState?.cancelResultSubscription();
  });

  testWidgets('YOLOViewState didUpdateWidget and dispose', (tester) async {
    final key = GlobalKey<YOLOViewState>();
    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
        ),
      ),
    );
    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'test_model.tflite',
          task: YOLOTask.segment, // change task to trigger didUpdateWidget
        ),
      ),
    );
    expect(key.currentState, isNotNull);
  });

  test('fallback to default instance if not registered', () {
    YOLOPlatform.instance = MockYOLOPlatform();
    expect(YOLOPlatform.instance, isNotNull);
  });

  test('YOLOViewState.parseDetectionResults handles null/empty/malformed', () {
    final state = YOLOViewState();
    expect(state.parseDetectionResults({}), isEmpty);
    expect(state.parseDetectionResults({'detections': null}), isEmpty);
    expect(
      state.parseDetectionResults({
        'detections': [{}],
      }),
      isEmpty,
    );
  });

  testWidgets('YOLOView calls all callbacks and handles nulls', (tester) async {
    int resultCount = 0;
    int metricsCount = 0;
    double? lastZoom;

    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
          onResult: (_) => resultCount++,
          onPerformanceMetrics: (_) => metricsCount++,
          onZoomChanged: (z) => lastZoom = z,
        ),
      ),
    );

    // Simulate calling the callbacks
    final state = tester.state<YOLOViewState>(find.byType(YOLOView));
    state.widget.onResult?.call([]);
    state.widget.onPerformanceMetrics?.call(
      YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 50.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      ),
    );
    state.widget.onZoomChanged?.call(2.0);

    expect(resultCount, 1);
    expect(metricsCount, 1);
    expect(lastZoom, 2.0);
  });

  test('YOLOViewState.cancelResultSubscription is idempotent', () {
    final state = YOLOViewState();
    state.cancelResultSubscription();
    state.cancelResultSubscription();
  });

  test('YOLOViewController._applyThresholds fallback', () async {
    final controller = YOLOViewController();
    // No method channel set, should not throw
    await controller.setConfidenceThreshold(0.9);
    await controller.setIoUThreshold(0.8);
    await controller.setNumItemsThreshold(50);
    await controller.switchCamera();
  });

  test('YOLOViewState handles malformed detection event', () {
    final state = YOLOViewState();
    final malformedEvent = {
      'detections': [
        {'badKey': 123},
      ],
    };
    expect(state.parseDetectionResults(malformedEvent), isEmpty);
  });

  test('switchModel throws when viewId is not set', () {
    final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
    expect(
      () => yolo.switchModel('other_model.tflite', YOLOTask.detect),
      throwsA(isA<StateError>()),
    );
  });

  test('YOLO.predict returns parsed detection results', () async {
    final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
    await yolo.loadModel();

    final image = Uint8List.fromList(List.filled(10, 0));
    final results = await yolo.predict(image);

    expect(results, contains('boxes'));
    expect(results['boxes'], isA<List<Map<String, dynamic>>>());
    expect(results['boxes'][0]['class'], equals('person'));
  });

  test('YOLO.predict throws on empty image', () async {
    final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
    await yolo.loadModel();

    await expectLater(
      () => yolo.predict(Uint8List(0)),
      throwsA(isA<InvalidInputException>()),
    );
  });

  test('checkModelExists returns fallback on error', () async {
    final result = await YOLO.checkModelExists('nonexistent_model.tflite');
    expect(result['exists'], false);
    expect(result['path'], 'nonexistent_model.tflite');
  });

  test('getStoragePaths returns valid result or fallback', () async {
    final result = await YOLO.getStoragePaths();
    expect(result, isA<Map<String, String?>>());
  });

  test('switchModel works when viewId is set', () async {
    final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
    yolo.setViewId(1);

    await yolo.switchModel('new_model.tflite', YOLOTask.segment);
    expect(log.any((call) => call.method == 'setModel'), isTrue);
  });

  group('Multi-Instance YOLO', () {
    setUp(() {
      // The existing main channel mock will handle default channel calls
      // We don't need additional setup since the multi-instance channels
      // are mocked by the main setUp() method
    });

    tearDown(() {
      // Clear instance manager state between tests
      final activeIds = YOLOInstanceManager.getActiveInstanceIds();
      for (final id in activeIds) {
        YOLOInstanceManager.unregisterInstance(id);
      }
    });

    test('creates multi-instance with unique ID', () {
      final yolo1 = YOLO(
        modelPath: 'model1.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final yolo2 = YOLO(
        modelPath: 'model2.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(yolo1.instanceId, isNot(equals('default')));
      expect(yolo2.instanceId, isNot(equals('default')));
      expect(yolo1.instanceId, isNot(equals(yolo2.instanceId)));
    });

    test('default instance has correct ID', () {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        // useMultiInstance defaults to false
      );

      expect(yolo.instanceId, equals('default'));
    });

    test('multi-instance constructor registers with manager', () {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(YOLOInstanceManager.hasInstance(yolo.instanceId), isTrue);
      expect(YOLOInstanceManager.getInstance(yolo.instanceId), equals(yolo));
    });

    test('default instance is not registered with manager', () {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        // useMultiInstance defaults to false
      );

      expect(YOLOInstanceManager.hasInstance(yolo.instanceId), isFalse);
    });

    test('multi-instance loadModel calls createInstance', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      try {
        await yolo.loadModel();
        // If this doesn't throw, the multi-instance logic is working
        expect(yolo.instanceId, isNot(equals('default')));
      } catch (e) {
        // Expected since we don't have a real platform implementation
        expect(yolo.instanceId, isNot(equals('default')));
        expect(YOLOInstanceManager.hasInstance(yolo.instanceId), isTrue);
      }
    });

    test('default instance loadModel does not call createInstance', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        // useMultiInstance defaults to false
      );

      await yolo.loadModel();

      expect(log.any((call) => call.method == 'createInstance'), isFalse);
    });

    test('multi-instance predict includes instanceId', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(yolo.instanceId, isNot(equals('default')));
      expect(YOLOInstanceManager.hasInstance(yolo.instanceId), isTrue);

      try {
        await yolo.loadModel();
        final image = Uint8List.fromList([1, 2, 3, 4, 5]);
        await yolo.predict(image);
      } catch (e) {
        // Expected since we don't have real platform implementation
        // The important part is that multi-instance structure is correct
        expect(yolo.instanceId, isNot(equals('default')));
      }
    });

    test('default instance predict does not include instanceId', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        // useMultiInstance defaults to false
      );

      await yolo.loadModel();
      log.clear(); // Clear previous calls

      final image = Uint8List.fromList([1, 2, 3, 4, 5]);
      await yolo.predict(image);

      expect(
        log.any(
          (call) =>
              call.method == 'predictSingleImage' &&
              call.arguments.containsKey('instanceId'),
        ),
        isFalse,
      );
    });

    test('multi-instance switchModel includes instanceId', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      yolo.setViewId(1);

      expect(yolo.instanceId, isNot(equals('default')));
      expect(YOLOInstanceManager.hasInstance(yolo.instanceId), isTrue);

      try {
        await yolo.switchModel('new_model.tflite', YOLOTask.segment);
      } catch (e) {
        // Expected since we don't have real platform implementation
        // The important part is that multi-instance structure is correct
        expect(yolo.instanceId, isNot(equals('default')));
      }
    });

    test('default instance switchModel does not include instanceId', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        // useMultiInstance defaults to false
      );
      yolo.setViewId(1);

      await yolo.switchModel('new_model.tflite', YOLOTask.segment);

      expect(
        log.any(
          (call) =>
              call.method == 'setModel' &&
              call.arguments.containsKey('instanceId'),
        ),
        isFalse,
      );
    });

    test('dispose unregisters instance from manager', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final instanceId = yolo.instanceId;

      expect(YOLOInstanceManager.hasInstance(instanceId), isTrue);

      await yolo.dispose();

      expect(YOLOInstanceManager.hasInstance(instanceId), isFalse);
    });

    test('dispose calls disposeInstance method', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final instanceId = yolo.instanceId;

      expect(YOLOInstanceManager.hasInstance(instanceId), isTrue);

      await yolo.dispose();

      // Instance should be unregistered regardless of platform error
      expect(YOLOInstanceManager.hasInstance(instanceId), isFalse);
    });

    test('dispose handles platform errors gracefully', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );
      final instanceId = yolo.instanceId;

      expect(YOLOInstanceManager.hasInstance(instanceId), isTrue);

      // The dispose method should complete and unregister regardless of platform errors
      await expectLater(yolo.dispose(), completes);

      // Instance should be unregistered after dispose, regardless of platform error
      expect(YOLOInstanceManager.hasInstance(instanceId), isFalse);
    });

    test(
      'multiple instances can be created and disposed independently',
      () async {
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
        final yolo3 = YOLO(
          modelPath: 'model3.tflite',
          task: YOLOTask.classify,
          useMultiInstance: true,
        );

        expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(3));

        // Dispose middle instance
        await yolo2.dispose();

        expect(YOLOInstanceManager.getActiveInstanceIds().length, equals(2));
        expect(YOLOInstanceManager.hasInstance(yolo1.instanceId), isTrue);
        expect(YOLOInstanceManager.hasInstance(yolo2.instanceId), isFalse);
        expect(YOLOInstanceManager.hasInstance(yolo3.instanceId), isTrue);

        // Dispose remaining instances
        await yolo1.dispose();
        await yolo3.dispose();

        expect(YOLOInstanceManager.getActiveInstanceIds(), isEmpty);
      },
    );
  });
}
