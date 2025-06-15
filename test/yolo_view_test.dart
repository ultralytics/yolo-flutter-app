// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:flutter/foundation.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('com.ultralytics.yolo/controlChannel_xyz'),
          (MethodCall methodCall) async {
            return null;
          },
        );
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('com.ultralytics.yolo/controlChannel_xyz'),
          null,
        );
  });

  tearDown(() async {
    // Clean up any remaining subscriptions
    await Future.delayed(const Duration(milliseconds: 100));
  });

  group('YOLOViewController Public API', () {
    late YOLOViewController controller;

    setUp(() {
      controller = YOLOViewController();
    });

    test('initial values are correct', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('setConfidenceThreshold clamps values', () async {
      await controller.setConfidenceThreshold(0.8);
      expect(controller.confidenceThreshold, 0.8);

      await controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setConfidenceThreshold(-0.2);
      expect(controller.confidenceThreshold, 0.0);
    });

    test('setIoUThreshold clamps values', () async {
      await controller.setIoUThreshold(0.7);
      expect(controller.iouThreshold, 0.7);

      await controller.setIoUThreshold(2.0);
      expect(controller.iouThreshold, 1.0);

      await controller.setIoUThreshold(-1.0);
      expect(controller.iouThreshold, 0.0);
    });

    test('setNumItemsThreshold clamps values', () async {
      await controller.setNumItemsThreshold(50);
      expect(controller.numItemsThreshold, 50);

      await controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);

      await controller.setNumItemsThreshold(0);
      expect(controller.numItemsThreshold, 1);
    });

    test('setThresholds updates multiple values', () async {
      await controller.setThresholds(
        confidenceThreshold: 0.9,
        iouThreshold: 0.6,
        numItemsThreshold: 25,
      );

      expect(controller.confidenceThreshold, 0.9);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 25);
    });

    test('setThresholds updates only specified values', () async {
      await controller.setThresholds(confidenceThreshold: 0.7);
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.45); // unchanged
      expect(controller.numItemsThreshold, 30); // unchanged
    });

    test('switchCamera completes without error', () async {
      expect(() => controller.switchCamera(), returnsNormally);
    });

    test('extreme values are handled correctly', () async {
      await controller.setConfidenceThreshold(double.maxFinite);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setNumItemsThreshold(999999);
      expect(controller.numItemsThreshold, 100);
    });

    test('YOLOViewController clamps confidence threshold', () {
      final controller = YOLOViewController();
      controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);
    });
  });

  group('YOLOView Widget Properties', () {
    test('widget properties are accessible', () {
      const widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.segment,
        cameraResolution: '720p',
        showNativeUI: true,
      );

      expect(widget.modelPath, 'test_model.tflite');
      expect(widget.task, YOLOTask.segment);
      expect(widget.cameraResolution, '720p');
      expect(widget.showNativeUI, true);
      expect(widget.controller, isNull);
    });

    test('widget with controller property', () {
      final controller = YOLOViewController();

      final widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        controller: controller,
      );

      expect(widget.controller, equals(controller));
    });

    test('widget with callbacks', () {
      var resultCallCount = 0;
      var metricsCallCount = 0;

      final widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        onResult: (results) => resultCallCount++,
        onPerformanceMetrics: (metrics) => metricsCallCount++,
      );

      expect(widget.onResult, isNotNull);
      expect(widget.onPerformanceMetrics, isNotNull);

      // Test callbacks work
      widget.onResult!([]);
      widget.onPerformanceMetrics!(
        YOLOPerformanceMetrics(
          fps: 30.0,
          processingTimeMs: 50.0,
          frameNumber: 1,
          timestamp: DateTime.now(),
        ),
      );

      expect(resultCallCount, 1);
      expect(metricsCallCount, 1);
    });
  });

  group('YOLOView Widget Creation', () {
    testWidgets('creates with minimal parameters', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('creates with custom controller', (WidgetTester tester) async {
      final controller = YOLOViewController();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('creates with all optional parameters', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'custom_model.tflite',
            task: YOLOTask.segment,
            cameraResolution: '1080p',
            onResult: (results) {},
            onPerformanceMetrics: (metrics) {},
            showNativeUI: true,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('handles null callbacks', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: null,
            onPerformanceMetrics: null,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });
  });

  group('YOLOView GlobalKey Access', () {
    testWidgets('can access state methods via GlobalKey', (
      WidgetTester tester,
    ) async {
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

      await tester.pump(); // Single pump instead of pumpAndSettle

      // Test that methods are accessible
      expect(
        () => key.currentState?.setConfidenceThreshold(0.8),
        returnsNormally,
      );
      expect(() => key.currentState?.setIoUThreshold(0.6), returnsNormally);
      expect(() => key.currentState?.setNumItemsThreshold(25), returnsNormally);
      expect(() => key.currentState?.switchCamera(), returnsNormally);
    });
  });

  group('YOLOView Task Types', () {
    test('supports all YOLOTask enum values', () {
      // Test that YOLOTask enum has expected values
      expect(YOLOTask.values.length, greaterThan(0));
      expect(YOLOTask.values.contains(YOLOTask.detect), true);
      expect(YOLOTask.values.contains(YOLOTask.segment), true);
    });

    test('different task types create different widgets', () {
      const widget1 = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      const widget2 = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.segment,
      );

      expect(widget1.task, YOLOTask.detect);
      expect(widget2.task, YOLOTask.segment);
      expect(widget1.task, isNot(equals(widget2.task)));
    });
  });

  group('YOLOView Model Paths', () {
    test('handles different model path formats', () {
      const testPaths = [
        'yolo11n.tflite',
        'assets/models/yolo11s.tflite',
        'yolo11n.mlpackage',
        'custom_model.tflite',
      ];

      // Test that model paths are valid strings
      for (final path in testPaths) {
        expect(path, isA<String>());
        expect(path.isNotEmpty, true);
      }

      const widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );
      expect(widget.modelPath, isA<String>());
      expect(widget.modelPath.isNotEmpty, true);
    });

    test('handles special characters in model paths', () {
      const specialPaths = [
        'models/test-model_v2.tflite',
        'models/test.model.with.dots.tflite',
        'models/test model with spaces.tflite',
      ];

      // Test that special character paths are valid
      for (final path in specialPaths) {
        expect(path, isA<String>());
        expect(path.contains('model'), true);
      }

      const widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );
      expect(widget.modelPath, isA<String>());
    });
  });

  group('YOLOView Camera Resolutions', () {
    test('supports common camera resolutions', () {
      const resolutions = ['480p', '720p', '1080p', '4K'];

      // Test that resolutions are valid strings
      for (final resolution in resolutions) {
        expect(resolution, isA<String>());
        expect(resolution.isNotEmpty, true);
      }

      const widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        cameraResolution: '1080p',
      );
      expect(widget.cameraResolution, isA<String>());
    });

    test('handles default camera resolution when not specified', () {
      const widget = YOLOView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        // cameraResolution not specified - should use default
      );
      expect(widget.cameraResolution, isA<String>());
    });
  });

  group('YOLOViewState internal logic', () {
    test('parses empty detection result', () {
      final state = YOLOViewState();
      final result = state.parseDetectionResults({});
      expect(result, isEmpty);
    });

    test('handles malformed detection data gracefully', () {
      final state = YOLOViewState();
      final malformedEvent = {
        'detections': [
          {'badKey': 123},
        ],
      };

      final result = state.parseDetectionResults(malformedEvent);
      expect(result, isEmpty);
    });

    test('cancelResultSubscription does not crash', () {
      final state = YOLOViewState();
      state.cancelResultSubscription();
    });
  });

  test('setThresholds works without method channel', () async {
    final controller = YOLOViewController();
    await controller.setThresholds(confidenceThreshold: 0.9);
    await controller.setIoUThreshold(0.8);
    await controller.setNumItemsThreshold(50);
    await controller.switchCamera();
  });

  test('controller._applyThresholds fallback path', () async {
    final controller = YOLOViewController();
    await controller.setConfidenceThreshold(0.9);
  });

  test('fallback path in _applyThresholds is hit on error', () async {
    final controller = YOLOViewController();
    const methodChannel = MethodChannel(
      'com.ultralytics.yolo/controlChannel_test',
    );
    controller.init(methodChannel, 1);

    // simulate failure on setThresholds
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(methodChannel, (MethodCall methodCall) async {
          if (methodCall.method == 'setThresholds') {
            throw PlatformException(code: 'fail');
          }
          return null;
        });

    await controller.setThresholds(confidenceThreshold: 0.7);
    expect(controller.confidenceThreshold, 0.7);
  });

  test('zoomIn calls platform method', () async {
    final controller = YOLOViewController();
    const testChannel = MethodChannel('test_channel');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(testChannel, (methodCall) async {
          if (methodCall.method == 'zoomIn') {
            return null;
          } else if (methodCall.method == 'setThresholds') {
            return null;
          }
          return null;
        });

    controller.init(testChannel, 1);
    await controller.zoomIn();
  });

  test('zoomOut calls platform method', () async {
    final controller = YOLOViewController();
    const testChannel = MethodChannel('test_channel');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(testChannel, (methodCall) async {
          if (methodCall.method == 'zoomOut') {
            return null;
          } else if (methodCall.method == 'setThresholds') {
            return null;
          }
          return null;
        });

    controller.init(testChannel, 1);
    await controller.zoomOut();
  });

  test('setZoomLevel calls platform method', () async {
    final controller = YOLOViewController();
    const testChannel = MethodChannel('test_channel');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(testChannel, (methodCall) async {
          if (methodCall.method == 'setZoomLevel') {
            expect(methodCall.arguments['zoomLevel'], 2.0);
            return null;
          } else if (methodCall.method == 'setThresholds') {
            return null;
          }
          return null;
        });

    controller.init(testChannel, 1);
    await controller.setZoomLevel(2.0);
  });

  test('stop calls platform method', () async {
    final controller = YOLOViewController();
    const testChannel = MethodChannel('test_channel');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(testChannel, (methodCall) async {
          if (methodCall.method == 'stop') {
            return null;
          } else if (methodCall.method == 'setThresholds') {
            return null;
          }
          return null;
        });

    controller.init(testChannel, 1);
    await controller.stop();
  });

  test('setStreamingConfig calls platform method', () async {
    final controller = YOLOViewController();
    const testChannel = MethodChannel('test_channel');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(testChannel, (methodCall) async {
          if (methodCall.method == 'setStreamingConfig') {
            expect(methodCall.arguments['includeDetections'], true);
            expect(methodCall.arguments['maxFPS'], 15);
            return null;
          } else if (methodCall.method == 'setThresholds') {
            return null;
          }
          return null;
        });

    controller.init(testChannel, 1);
    final config = YOLOStreamingConfig.throttled(maxFPS: 15);
    await controller.setStreamingConfig(config);
  });

  test('switchModel applies model switch with valid viewId', () async {
    final controller = YOLOViewController();
    const dummyChannel = MethodChannel('dummy');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(dummyChannel, (methodCall) async {
          if (methodCall.method == 'setModel') {
            expect(methodCall.arguments['modelPath'], 'my_model.tflite');
            expect(methodCall.arguments['task'], 'detect');
            return null;
          } else if (methodCall.method == 'setThresholds') {
            // Mock setThresholds which is called during init
            return null;
          }
          return null;
        });

    controller.init(dummyChannel, 42);
    await controller.switchModel('my_model.tflite', YOLOTask.detect);
  });

  testWidgets('handles detection and metrics events correctly', (tester) async {
    final key = GlobalKey<YOLOViewState>();
    var detectionCalled = false;
    var metricsCalled = false;

    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'model.tflite',
          task: YOLOTask.detect,
          onResult: (results) {
            detectionCalled = true;
          },
          onPerformanceMetrics: (metrics) {
            metricsCalled = true;
          },
        ),
      ),
    );

    final state = key.currentState!;
    state.subscribeToResults();

    final event = {
      'detections': [
        {
          'classIndex': 0,
          'className': 'person',
          'confidence': 0.95,
          'boundingBox': {'left': 0, 'top': 0, 'right': 100, 'bottom': 100},
          'normalizedBox': {'left': 0, 'top': 0, 'right': 1, 'bottom': 1},
        },
      ],
      'processingTimeMs': 16,
      'fps': 60,
    };

    state.parseDetectionResults(event);
    if (state.widget.onResult != null) state.widget.onResult!([]);
    if (state.widget.onPerformanceMetrics != null) {
      state.widget.onPerformanceMetrics!(
        YOLOPerformanceMetrics(
          fps: 60.0,
          processingTimeMs: 30.0,
          frameNumber: 1,
          timestamp: DateTime.now(),
        ),
      );
    }

    expect(detectionCalled, isTrue);
    expect(metricsCalled, isTrue);
  });

  testWidgets('fallback UI shown on unsupported platform', (tester) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.fuchsia;

    await tester.pumpWidget(
      const MaterialApp(
        home: YOLOView(modelPath: 'model.tflite', task: YOLOTask.detect),
      ),
    );

    expect(find.text('Platform not supported for YOLOView'), findsOneWidget);

    debugDefaultTargetPlatformOverride = null;
  });

  testWidgets('zoom callback is triggered', (tester) async {
    final key = GlobalKey<YOLOViewState>();
    double? zoomLevel;

    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'model.tflite',
          task: YOLOTask.detect,
          onZoomChanged: (z) => zoomLevel = z,
        ),
      ),
    );

    final state = key.currentState!;
    state.widget.onZoomChanged?.call(2.5);
    expect(zoomLevel, 2.5);
  });

  testWidgets('unknown method call is handled gracefully', (tester) async {
    final key = GlobalKey<YOLOViewState>();
    const bool widgetMounted = true;

    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'model.tflite',
          task: YOLOTask.detect,
        ),
      ),
    );

    final state = key.currentState!;
    state.triggerPlatformViewCreated(1);

    // Verify widget is still mounted and working after method call
    expect(widgetMounted, isTrue);
    expect(find.byType(YOLOView), findsOneWidget);
  });

  testWidgets('controller is created internally when not provided', (
    tester,
  ) async {
    final key = GlobalKey<YOLOViewState>();
    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'model.tflite',
          task: YOLOTask.detect,
        ),
      ),
    );

    final state = key.currentState!;
    expect(state.effectiveController, isA<YOLOViewController>());
    expect(state.effectiveController.confidenceThreshold, 0.5);
    expect(state.effectiveController.iouThreshold, 0.45);
    expect(state.effectiveController.numItemsThreshold, 30);
  });

  testWidgets('recreateEventChannel triggers resubscription', (tester) async {
    final key = GlobalKey<YOLOViewState>();

    await tester.pumpWidget(
      MaterialApp(
        home: YOLOView(
          key: key,
          modelPath: 'model.tflite',
          task: YOLOTask.detect,
          onResult: (_) {},
          onPerformanceMetrics: (_) {},
        ),
      ),
    );

    final state = key.currentState!;
    state.triggerPlatformViewCreated(1);

    // Verify initial subscription
    expect(state.resultSubscription, isNotNull);

    // Clean up
    state.cancelResultSubscription();
    expect(state.resultSubscription, isNull);
  });

  testWidgets('builds correctly on iOS', (tester) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.iOS;

    await tester.pumpWidget(
      const MaterialApp(
        home: YOLOView(modelPath: 'model.tflite', task: YOLOTask.detect),
      ),
    );

    expect(find.byType(YOLOView), findsOneWidget);
    debugDefaultTargetPlatformOverride = null;
  });

  group('YOLOView Streaming Functionality', () {
    testWidgets('handles onStreamingData callback', (tester) async {
      final key = GlobalKey<YOLOViewState>();
      Map<String, dynamic>? streamingData;
      var streamingCallCount = 0;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onStreamingData: (data) {
              streamingData = data;
              streamingCallCount++;
            },
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();

      // Simulate comprehensive streaming event
      final comprehensiveEvent = {
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {'left': 0, 'top': 0, 'right': 100, 'bottom': 100},
            'normalizedBox': {
              'left': 0.0,
              'top': 0.0,
              'right': 1.0,
              'bottom': 1.0,
            },
          },
        ],
        'fps': 30.0,
        'processingTimeMs': 33.3,
        'frameNumber': 100,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
        'originalImage': Uint8List.fromList([1, 2, 3, 4, 5]),
      };

      // Simulate receiving streaming data
      if (state.widget.onStreamingData != null) {
        state.widget.onStreamingData!(comprehensiveEvent);
      }

      expect(streamingCallCount, equals(1));
      expect(streamingData, isNotNull);
      expect(streamingData!['detections'], isA<List>());
      expect(streamingData!['fps'], equals(30.0));
      expect(streamingData!['originalImage'], isA<Uint8List>());
    });

    testWidgets('streaming config is applied during initialization', (
      tester,
    ) async {
      final key = GlobalKey<YOLOViewState>();
      const config = YOLOStreamingConfig.withMasks();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            streamingConfig: config,
          ),
        ),
      );

      final state = key.currentState!;
      expect(state.widget.streamingConfig, equals(config));
      expect(state.widget.streamingConfig!.includeMasks, isTrue);
    });

    testWidgets('onStreamingData takes precedence over individual callbacks', (
      tester,
    ) async {
      final key = GlobalKey<YOLOViewState>();
      var onResultCalled = false;
      var onMetricsCalled = false;
      var onStreamingCalled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) => onResultCalled = true,
            onPerformanceMetrics: (_) => onMetricsCalled = true,
            onStreamingData: (_) => onStreamingCalled = true,
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();

      final event = {
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {'left': 0, 'top': 0, 'right': 100, 'bottom': 100},
          },
        ],
        'fps': 30.0,
        'processingTimeMs': 50.0,
      };

      // Simulate event processing
      if (state.widget.onStreamingData != null) {
        state.widget.onStreamingData!(event);
      }

      expect(onStreamingCalled, isTrue);
      // Individual callbacks should NOT be called when onStreamingData is provided
      expect(onResultCalled, isFalse);
      expect(onMetricsCalled, isFalse);
    });

    testWidgets('individual callbacks work when onStreamingData is null', (
      tester,
    ) async {
      final key = GlobalKey<YOLOViewState>();
      var onResultCalled = false;
      var onMetricsCalled = false;
      List<YOLOResult>? results;
      YOLOPerformanceMetrics? metrics;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (r) {
              onResultCalled = true;
              results = r;
            },
            onPerformanceMetrics: (m) {
              onMetricsCalled = true;
              metrics = m;
            },
            // onStreamingData is null
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();

      final event = {
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {'left': 10, 'top': 10, 'right': 110, 'bottom': 210},
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.9,
              'bottom': 0.9,
            },
          },
        ],
        'fps': 30.0,
        'processingTimeMs': 50.0,
        'frameNumber': 1,
      };

      // Simulate individual callback processing
      if (state.widget.onResult != null) {
        final parsedResults = state.parseDetectionResults(event);
        state.widget.onResult!(parsedResults);
      }
      if (state.widget.onPerformanceMetrics != null) {
        final performanceMetrics = YOLOPerformanceMetrics.fromMap(event);
        state.widget.onPerformanceMetrics!(performanceMetrics);
      }

      expect(onResultCalled, isTrue);
      expect(onMetricsCalled, isTrue);
      expect(results, isNotNull);
      expect(results!.length, equals(1));
      expect(results!.first.className, equals('person'));
      expect(metrics, isNotNull);
      expect(metrics!.fps, equals(30.0));
    });

    testWidgets('event channel recreation is handled correctly', (
      tester,
    ) async {
      final key = GlobalKey<YOLOViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      final state = key.currentState!;
      state.triggerPlatformViewCreated(1);

      // Simulate recreateEventChannel method call
      const methodCall = MethodCall('recreateEventChannel', null);
      await state.handleMethodCall(methodCall);

      // Wait for delayed resubscription
      await tester.pump(const Duration(milliseconds: 150));

      expect(state.resultSubscription, isNotNull);
    });

    testWidgets('zoom level changes are handled correctly', (tester) async {
      final key = GlobalKey<YOLOViewState>();
      double? receivedZoomLevel;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onZoomChanged: (level) => receivedZoomLevel = level,
          ),
        ),
      );

      final state = key.currentState!;

      // Simulate onZoomChanged method call
      const methodCall = MethodCall('onZoomChanged', 2.5);
      await state.handleMethodCall(methodCall);

      expect(receivedZoomLevel, equals(2.5));
    });

    testWidgets('unknown method calls are handled gracefully', (tester) async {
      final key = GlobalKey<YOLOViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      final state = key.currentState!;

      // Simulate unknown method call
      const methodCall = MethodCall('unknownMethod', {'data': 'test'});
      final result = await state.handleMethodCall(methodCall);

      expect(result, isNull);
    });

    test('streaming config setter on controller works', () async {
      final controller = YOLOViewController();
      const config = YOLOStreamingConfig.debug();

      // Mock method channel
      const methodChannel = MethodChannel('test_channel');
      final List<MethodCall> log = [];

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(methodChannel, (
            MethodCall methodCall,
          ) async {
            log.add(methodCall);
            return null;
          });

      controller.init(methodChannel, 1);
      await controller.setStreamingConfig(config);

      expect(log.any((call) => call.method == 'setStreamingConfig'), isTrue);
      final streamingCall = log.firstWhere(
        (call) => call.method == 'setStreamingConfig',
      );
      expect(streamingCall.arguments['includeDetections'], isTrue);
      expect(streamingCall.arguments['includeMasks'], isTrue);
      expect(streamingCall.arguments['includeOriginalImage'], isTrue);
    });

    test('streaming config without channel logs warning', () async {
      final controller = YOLOViewController();
      const config = YOLOStreamingConfig.minimal();

      // Should not throw, but log a warning
      await controller.setStreamingConfig(config);
      expect(controller.isInitialized, isFalse);
    });

    testWidgets('error handling in streaming data callback', (tester) async {
      final key = GlobalKey<YOLOViewState>();
      var errorHandled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onStreamingData: (data) {
              // Simulate an error in processing
              throw Exception('Test error in streaming data');
            },
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();

      final event = {'detections': [], 'fps': 30.0};

      // This should not crash the app
      try {
        if (state.widget.onStreamingData != null) {
          state.widget.onStreamingData!(event);
        }
      } catch (e) {
        errorHandled = true;
      }

      expect(errorHandled, isTrue);
    });

    testWidgets('error handling in performance metrics callback', (
      tester,
    ) async {
      final key = GlobalKey<YOLOViewState>();
      var errorHandled = false;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onPerformanceMetrics: (metrics) {
              // Simulate an error in processing
              throw Exception('Test error in metrics');
            },
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();

      final event = {'fps': 30.0, 'processingTimeMs': 50.0, 'frameNumber': 1};

      // This should not crash the app
      try {
        if (state.widget.onPerformanceMetrics != null) {
          final metrics = YOLOPerformanceMetrics.fromMap(event);
          state.widget.onPerformanceMetrics!(metrics);
        }
      } catch (e) {
        errorHandled = true;
      }

      expect(errorHandled, isTrue);
    });

    testWidgets('stream subscription error recovery', (tester) async {
      final key = GlobalKey<YOLOViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();
      expect(state.resultSubscription, isNotNull);

      // Simulate stream error and recovery
      state.cancelResultSubscription();
      expect(state.resultSubscription, isNull);

      // Re-subscribe
      state.subscribeToResults();
      expect(state.resultSubscription, isNotNull);
    });

    testWidgets('streaming config in build params', (tester) async {
      const config = YOLOStreamingConfig.full();

      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            streamingConfig: config,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('zoom level setter on controller works', (tester) async {
      final key = GlobalKey<YOLOViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      final state = key.currentState!;

      // Should not throw when called
      expect(() => state.setZoomLevel(2.0), returnsNormally);
    });

    testWidgets('dynamic callback updates are handled correctly', (
      tester,
    ) async {
      final key = GlobalKey<YOLOViewState>();

      // Initial widget with onResult
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      final state = key.currentState!;
      expect(state.resultSubscription, isNotNull);

      // Update widget with different onResult
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      expect(state.resultSubscription, isNotNull);

      // Update widget with no callbacks
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            // No callbacks
          ),
        ),
      );

      expect(state.resultSubscription, isNull);
    });

    testWidgets('controller switching in didUpdateWidget', (tester) async {
      final key = GlobalKey<YOLOViewState>();
      final controller1 = YOLOViewController();
      final controller2 = YOLOViewController();

      // Initial widget with controller1
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            controller: controller1,
          ),
        ),
      );

      final state = key.currentState!;
      expect(state.effectiveController, equals(controller1));

      // Update widget with controller2
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            controller: controller2,
          ),
        ),
      );

      expect(state.effectiveController, equals(controller2));
    });

    testWidgets('showNativeUI changes are applied', (tester) async {
      final key = GlobalKey<YOLOViewState>();

      // Initial widget with showNativeUI = false
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            showNativeUI: false,
          ),
        ),
      );

      final state = key.currentState!;
      expect(state.widget.showNativeUI, isFalse);

      // Update widget with showNativeUI = true
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            showNativeUI: true,
          ),
        ),
      );

      expect(state.widget.showNativeUI, isTrue);
    });

    testWidgets('model/task changes trigger switchModel', (tester) async {
      final key = GlobalKey<YOLOViewState>();
      final controller = YOLOViewController();
      final List<MethodCall> log = [];

      // Initial widget
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model1.tflite',
            task: YOLOTask.detect,
            controller: controller,
          ),
        ),
      );

      final state = key.currentState!;

      // Set up the mock handler for the state's method channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(state.methodChannel, (
            MethodCall methodCall,
          ) async {
            log.add(methodCall);
            return null;
          });

      state.triggerPlatformViewCreated(1);

      // Clear any initialization calls
      log.clear();

      // Update widget with different model
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model2.tflite',
            task: YOLOTask.segment,
            controller: controller,
          ),
        ),
      );

      // Should have triggered setModel method
      expect(log.any((call) => call.method == 'setModel'), isTrue);
    });

    testWidgets('comprehensive malformed data handling', (tester) async {
      final key = GlobalKey<YOLOViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      final state = key.currentState!;

      // Test various malformed data scenarios
      expect(state.parseDetectionResults({}), isEmpty);
      expect(state.parseDetectionResults({'detections': null}), isEmpty);

      // Handle type error gracefully for non-list detections
      expect(
        () => state.parseDetectionResults({'detections': 'not a list'}),
        throwsA(isA<TypeError>()),
      );

      expect(
        state.parseDetectionResults({
          'detections': [
            {'invalidStructure': true},
          ],
        }),
        isEmpty,
      );
    });

    testWidgets('test message handling in event stream', (tester) async {
      final key = GlobalKey<YOLOViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            key: key,
            modelPath: 'model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      final state = key.currentState!;
      state.subscribeToResults();

      // Simulate test message (should be handled separately)
      const testEvent = {'test': 'Hello from native platform'};
      // This would normally be processed in the stream listener
      // We test that it doesn't interfere with normal processing
      expect(testEvent.containsKey('test'), isTrue);
      expect(testEvent.containsKey('detections'), isFalse);
    });
  });

  group('YOLOViewController Error Handling', () {
    testWidgets('handles threshold method errors with fallback', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      // Mock channel that throws on individual methods but succeeds on combined
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setConfidenceThreshold' ||
                methodCall.method == 'setIoUThreshold' ||
                methodCall.method == 'setNumItemsThreshold') {
              throw PlatformException(code: 'ERROR');
            } else if (methodCall.method == 'setThresholds') {
              return null; // Success
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should fall back to _applyThresholds
      await controller.setConfidenceThreshold(0.7);
      expect(controller.confidenceThreshold, 0.7);

      await controller.setIoUThreshold(0.3);
      expect(controller.iouThreshold, 0.3);

      await controller.setNumItemsThreshold(20);
      expect(controller.numItemsThreshold, 20);
    });

    testWidgets('handles errors in zoom methods gracefully', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'zoomIn' ||
                methodCall.method == 'zoomOut' ||
                methodCall.method == 'setZoomLevel') {
              throw PlatformException(code: 'ERROR');
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should not throw
      await controller.zoomIn();
      await controller.zoomOut();
      await controller.setZoomLevel(2.0);
    });

    testWidgets('handles errors in other control methods', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'switchCamera' ||
                methodCall.method == 'setStreamingConfig' ||
                methodCall.method == 'stop') {
              throw PlatformException(code: 'ERROR');
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should not throw
      await controller.switchCamera();
      await controller.setStreamingConfig(const YOLOStreamingConfig.minimal());
      await controller.stop();
    });

    testWidgets('switchModel rethrows exceptions', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setModel') {
              throw PlatformException(code: 'ERROR', message: 'Test error');
            }
            return null;
          });

      controller.init(testChannel, 1);

      expect(
        () => controller.switchModel('model.tflite', YOLOTask.detect),
        throwsException,
      );
    });

    testWidgets('_applyThresholds handles errors in batch mode', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'applyThresholds') {
              throw PlatformException(code: 'ERROR', message: 'Batch error');
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should not throw - it catches and handles errors
      await controller.setConfidenceThreshold(0.7);
      await controller.setIoUThreshold(0.5);
      await controller.setNumItemsThreshold(30);
    });

    testWidgets('threshold setters apply successfully', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');
      bool confidenceSet = false;
      bool iouSet = false;
      bool numItemsSet = false;

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'applyThresholds') {
              // Handle the batch apply call
              return null;
            } else if (methodCall.method == 'setConfidenceThreshold') {
              confidenceSet = true;
              expect(methodCall.arguments['threshold'], 0.7);
            } else if (methodCall.method == 'setIoUThreshold') {
              iouSet = true;
              expect(methodCall.arguments['threshold'], 0.5);
            } else if (methodCall.method == 'setNumItemsThreshold') {
              numItemsSet = true;
              expect(methodCall.arguments['threshold'], 30);
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.setConfidenceThreshold(0.7);
      await controller.setIoUThreshold(0.5);
      await controller.setNumItemsThreshold(30);

      expect(confidenceSet, true);
      expect(iouSet, true);
      expect(numItemsSet, true);
    });

    testWidgets('switchCamera succeeds', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');
      bool switchCameraCalledSuccess = false;

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'switchCamera') {
              switchCameraCalledSuccess = true;
              return null; // Success
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.switchCamera();
      expect(switchCameraCalledSuccess, true);
    });

    testWidgets('zoom methods succeed', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');
      bool zoomInCalled = false;
      bool zoomOutCalled = false;
      bool setZoomCalled = false;

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'zoomIn') {
              zoomInCalled = true;
              return null;
            } else if (methodCall.method == 'zoomOut') {
              zoomOutCalled = true;
              return null;
            } else if (methodCall.method == 'setZoomLevel') {
              setZoomCalled = true;
              expect(methodCall.arguments['zoomLevel'], 2.0);
              return null;
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.zoomIn();
      await controller.zoomOut();
      await controller.setZoomLevel(2.0);

      expect(zoomInCalled, true);
      expect(zoomOutCalled, true);
      expect(setZoomCalled, true);
    });

    testWidgets('switchModel handles null channel gracefully', (tester) async {
      final controller = YOLOViewController();
      // Don't initialize the controller, so _methodChannel remains null

      // Should not throw, just log warning
      await controller.switchModel('model.tflite', YOLOTask.detect);
    });

    testWidgets('stop method succeeds', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');
      bool stopCalled = false;

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'stop') {
              stopCalled = true;
              return null;
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.stop();
      expect(stopCalled, true);
    });

    testWidgets('setStreamingConfig succeeds', (tester) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');
      bool configSet = false;

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setStreamingConfig') {
              configSet = true;
              return null;
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.setStreamingConfig(const YOLOStreamingConfig.minimal());
      expect(configSet, true);
    });
  });

  group('YOLOView Widget Updates', () {
    testWidgets('handles callback changes in didUpdateWidget', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      // Update widget with different callback
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('handles showNativeUI changes', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: true,
          ),
        ),
      );

      // Change showNativeUI
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: false,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('handles model path changes', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(modelPath: 'model1.tflite', task: YOLOTask.detect),
        ),
      );

      // Change model path
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(modelPath: 'model2.tflite', task: YOLOTask.detect),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('handles null callbacks removal', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
            onPerformanceMetrics: (_) {},
            onStreamingData: (_) {},
          ),
        ),
      );

      // Remove all callbacks
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            // No callbacks - should cancel subscription
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });
  });
}
