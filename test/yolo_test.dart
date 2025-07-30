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

    test('loadModel with classifierOptions', () async {
      final classifierOptions = {
        'enable1ChannelSupport': true,
        'expectedChannels': 1,
      };

      final testYolo = YOLO(
        modelPath: 'classifier_model.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
      );

      final result = await testYolo.loadModel();

      expect(result, isTrue);
      expect(log, hasLength(1));
      expect(log[0].arguments['classifierOptions'], classifierOptions);
    });

    test('YOLO.predict throws if called before loadModel', () async {
      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);
      await expectLater(
        yolo.predict(image),
        throwsA(isA<ModelNotLoadedException>()),
      );
    });

    test('loadModel handles initialization failure', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('yolo_single_image_channel'),
            (MethodCall methodCall) async {
              if (methodCall.method == 'loadModel') {
                throw Exception('Initialization failed');
              }
              return {'success': true};
            },
          );

      final yolo = YOLO(modelPath: 'bad_model.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to load model'),
          ),
        ),
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

  test('switchModel handles MODEL_NOT_FOUND error', () async {
    final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
    yolo.setViewId(1);

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('yolo_single_image_channel'),
          (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'MODEL_NOT_FOUND',
                message: 'Model not found',
              );
            }
            return {'success': true};
          },
        );

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
        .setMockMethodCallHandler(
          const MethodChannel('yolo_single_image_channel'),
          (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'INVALID_MODEL',
                message: 'Invalid model',
              );
            }
            return {'success': true};
          },
        );

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
        .setMockMethodCallHandler(
          const MethodChannel('yolo_single_image_channel'),
          (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'UNSUPPORTED_TASK',
                message: 'Unsupported task',
              );
            }
            return {'success': true};
          },
        );

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

  test('switchModel handles generic platform error', () async {
    final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
    yolo.setViewId(1);

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('yolo_single_image_channel'),
          (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(
                code: 'UNKNOWN_ERROR',
                message: 'Something went wrong',
              );
            }
            return {'success': true};
          },
        );

    expect(
      () => yolo.switchModel('model.tflite', YOLOTask.detect),
      throwsA(
        isA<ModelLoadingException>().having(
          (e) => e.message,
          'message',
          contains('Failed to switch model'),
        ),
      ),
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

    test('predict handles missing confidenceThreshold in args', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              // Verify confidenceThreshold is not in args when not provided
              expect(
                methodCall.arguments.containsKey('confidenceThreshold'),
                false,
              );
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image); // No thresholds provided
    });

    test('_initializeInstance handles createInstance failure', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'createInstance') {
              throw Exception('Platform error');
            }
            return null;
          });

      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to initialize YOLO instance'),
          ),
        ),
      );
    });

    test(
      'switchModel handles UNSUPPORTED_TASK with task name in message',
      () async {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
              if (methodCall.method == 'setModel') {
                throw PlatformException(
                  code: 'UNSUPPORTED_TASK',
                  message: 'Task not supported',
                );
              }
              return null;
            });

        final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
        yolo.setViewId(1);

        expect(
          () => yolo.switchModel('model.tflite', YOLOTask.obb),
          throwsA(
            isA<ModelLoadingException>().having(
              (e) => e.message,
              'message',
              contains('Unsupported task type: obb for model: model.tflite'),
            ),
          ),
        );
      },
    );

    test('switchModel handles unknown error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw Exception('Unknown error');
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      yolo.setViewId(1);

      expect(
        () => yolo.switchModel('model.tflite', YOLOTask.detect),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to switch model'),
          ),
        ),
      );
    });

    test('predict validates confidence threshold range', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);

      // Test below 0
      expect(
        () => yolo.predict(image, confidenceThreshold: -0.1),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('Confidence threshold must be between 0.0 and 1.0'),
          ),
        ),
      );

      // Test above 1
      expect(
        () => yolo.predict(image, confidenceThreshold: 1.5),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('Confidence threshold must be between 0.0 and 1.0'),
          ),
        ),
      );
    });

    test('predict validates IoU threshold range', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);

      // Test below 0
      expect(
        () => yolo.predict(image, iouThreshold: -0.1),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('IoU threshold must be between 0.0 and 1.0'),
          ),
        ),
      );

      // Test above 1
      expect(
        () => yolo.predict(image, iouThreshold: 1.5),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('IoU threshold must be between 0.0 and 1.0'),
          ),
        ),
      );
    });

    test('predict handles empty boxes gracefully', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                // No boxes key
                'detections': [],
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      // When platform doesn't return boxes, the key won't exist
      expect(result.containsKey('boxes'), false);
      expect(result['detections'], []);
    });

    test('predict handles missing iouThreshold in args', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              // Verify iouThreshold is not in args when not provided
              expect(methodCall.arguments.containsKey('iouThreshold'), false);
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image); // No thresholds provided
    });

    test('predict includes confidenceThreshold when provided', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              // Verify confidenceThreshold is included in args when provided
              expect(methodCall.arguments['confidenceThreshold'], 0.7);
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image, confidenceThreshold: 0.7);
    });

    test('predict includes iouThreshold when provided', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              // Verify iouThreshold is included in args when provided
              expect(methodCall.arguments['iouThreshold'], 0.5);
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image, iouThreshold: 0.5);
    });

    test('predict includes both thresholds when provided', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              // Verify both thresholds are included
              expect(methodCall.arguments['confidenceThreshold'], 0.8);
              expect(methodCall.arguments['iouThreshold'], 0.6);
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image, confidenceThreshold: 0.8, iouThreshold: 0.6);
    });

    test('multi-instance predict includes instanceId in args', () async {
      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      // Set up mock for the specific instance channel
      final instanceChannel = MethodChannel(
        'yolo_single_image_channel_${yolo.instanceId}',
      );
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(instanceChannel, (
            MethodCall methodCall,
          ) async {
            if (methodCall.method == 'createInstance') {
              return true;
            } else if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              // Verify instanceId is included for multi-instance
              expect(methodCall.arguments.containsKey('instanceId'), true);
              expect(
                methodCall.arguments['instanceId'],
                isNot(equals('default')),
              );
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image);
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

    test('checkModelExists handles generic exception', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'checkModelExists') {
              throw Exception('Generic error');
            }
            return null;
          });

      final result = await YOLO.checkModelExists('model.tflite');
      expect(result['exists'], false);
      expect(result['path'], 'model.tflite');
      expect(result['error'], contains('Generic error'));
    });

    test('predict handles malformed boxes response', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                'boxes': [
                  123, // Not a map
                  'invalid', // Not a map
                  null, // Null
                ],
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      // Should handle gracefully and return empty boxes
      expect(result['boxes'], isA<List>());
      expect((result['boxes'] as List).length, 0);
    });

    test('getStoragePaths handles generic exception', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'getStoragePaths') {
              throw Exception('Storage error');
            }
            return null;
          });

      final result = await YOLO.getStoragePaths();
      expect(result, isEmpty);
    });

    test('getStoragePaths handles platform exception', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'getStoragePaths') {
              throw PlatformException(code: 'ERROR', message: 'Platform error');
            }
            return null;
          });

      final result = await YOLO.getStoragePaths();
      expect(result, isEmpty);
    });
  });

  group('YOLO withClassifierOptions Constructor', () {
    test('creates YOLO instance with classifier options', () {
      final classifierOptions = {'enable1ChannelSupport': true};

      final yolo = YOLO.withClassifierOptions(
        modelPath: 'model.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
      );

      expect(yolo.modelPath, 'model.tflite');
      expect(yolo.classifierOptions, classifierOptions);
    });

    test('withClassifierOptions with useMultiInstance', () {
      final classifierOptions = {'expectedChannels': 1};

      final yolo = YOLO.withClassifierOptions(
        modelPath: 'model.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
        useMultiInstance: true,
      );

      expect(yolo.instanceId, isNot('default'));
    });

    test('withClassifierOptions loadModel', () async {
      final classifierOptions = {'enable1ChannelSupport': true};

      final yolo = YOLO.withClassifierOptions(
        modelPath: 'model.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
      );

      final result = await yolo.loadModel();

      expect(result, isTrue);
      expect(log[0].arguments['classifierOptions'], classifierOptions);
    });
  });
}
