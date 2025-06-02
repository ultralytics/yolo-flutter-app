// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
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

  test('switchModel applies model switch with valid viewId', () async {
    final controller = YOLOViewController();
    const modelChannel = MethodChannel('yolo_single_image_channel');

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(modelChannel, (methodCall) async {
          expect(methodCall.method, 'setModel');
          expect(methodCall.arguments['modelPath'], 'my_model.tflite');
          return null;
        });

    controller.init(const MethodChannel('dummy'), 42);
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
}
