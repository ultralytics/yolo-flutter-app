// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// dart:typed_data is already imported via flutter/services.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_method_channel.dart';
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
    state.widget.onPerformanceMetrics?.call({'fps': 30.0});
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
}
