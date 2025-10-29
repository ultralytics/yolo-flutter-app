import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_model_spec.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

import 'utils/test_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOViewController - method channel interactions', () {
    late YOLOViewController controller;
    late MethodChannel channel;
    late List<MethodCall> log;

    setUp(() {
      controller = YOLOViewController();
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      channel = setup.$1;
      log = setup.$2;
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
      log.clear();
    });

    test('init applies thresholds via setThresholds', () async {
      controller.init(channel, 42);
      // init triggers _applyThresholds -> setThresholds once
      YOLOTestHelpers.assertMethodCallCount(log, 'setThresholds', 1);
    });

    test(
      'setThresholds helpers invoke correct methods and update state',
      () async {
        controller.init(channel, 1);
        // Uses shared helper to validate confidence/IoU/numItems
        YOLOTestHelpers.validateThresholdBehavior(controller, log, channel);
      },
    );

    test('setThresholds combined updates and sends method call', () async {
      controller.init(channel, 1);

      await controller.setThresholds(
        confidenceThreshold: 0.7,
        iouThreshold: 0.55,
        numItemsThreshold: 25,
      );

      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.55);
      expect(controller.numItemsThreshold, 25);

      YOLOTestHelpers.assertMethodCalled(log, 'setThresholds');
    });

    test('camera controls and overlay/ui toggles', () async {
      controller.init(channel, 7);

      await controller.switchCamera();
      YOLOTestHelpers.assertMethodCalled(log, 'switchCamera');

      await controller.zoomIn();
      YOLOTestHelpers.assertMethodCalled(log, 'zoomIn');

      await controller.zoomOut();
      YOLOTestHelpers.assertMethodCalled(log, 'zoomOut');

      await controller.setZoomLevel(2.5);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setZoomLevel',
        arguments: {'zoomLevel': 2.5},
      );

      await controller.setShowUIControls(true);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setShowUIControls',
        arguments: {'show': true},
      );

      await controller.setShowOverlays(false);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setShowOverlays',
        arguments: {'show': false},
      );

      await controller.stop();
      YOLOTestHelpers.assertMethodCalled(log, 'stop');
    });

    test('setStreamingConfig sends method call', () async {
      controller.init(channel, 2);

      await controller.setStreamingConfig(
        const YOLOStreamingConfig.withPoses(),
      );
      // We only assert that the method was called; arguments are validated by platform tests
      YOLOTestHelpers.assertMethodCalled(log, 'setStreamingConfig');
    });

    test('captureFrame returns bytes', () async {
      controller.init(channel, 3);

      final Uint8List? bytes = await controller.captureFrame();
      expect(bytes, isNotNull);
      expect(bytes!.length, 100);
      YOLOTestHelpers.assertMethodCalled(log, 'captureFrame');
    });

    test('methods on uninitialized controller do not throw', () async {
      final c = YOLOViewController();
      expect(() => c.setConfidenceThreshold(0.5), returnsNormally);
      expect(() => c.setIoUThreshold(0.5), returnsNormally);
      expect(() => c.setNumItemsThreshold(20), returnsNormally);
      expect(() => c.switchCamera(), returnsNormally);
      expect(() => c.setZoomLevel(1.0), returnsNormally);
      expect(() => c.setShowUIControls(true), returnsNormally);
      expect(() => c.setShowOverlays(true), returnsNormally);
      expect(() => c.stop(), returnsNormally);
      // captureFrame should return null when not initialized
      final res = await c.captureFrame();
      expect(res, isNull);
    });
  });

  group('YOLOView - widget integration (smoke tests)', () {
    testWidgets('YOLOView builds with provided models', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            models: [
              YOLOModelSpec(
                modelPath: 'assets/models/yolo11n.tflite',
                task: YOLOTask.detect,
              ),
            ],
            streamingConfig: YOLOStreamingConfig.minimal(),
          ),
        ),
      );
      expect(find.byType(YOLOView), findsOneWidget);
    });

    testWidgets('YOLOView with controller builds and callbacks are assignable', (
      WidgetTester tester,
    ) async {
      // Prepare mock channel and init controller to avoid platform dependency
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      final channel = setup.$1;
      final controller = YOLOViewController()..init(channel, 1001);

      final List<YOLOResult> received = [];
      double? fpsValue;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            controller: controller,
            models: const [
              YOLOModelSpec(
                modelPath: 'assets/models/yolo11n.tflite',
                task: YOLOTask.detect,
              ),
            ],
            streamingConfig: const YOLOStreamingConfig.minimal(),
            onResult: (results) => received.addAll(results),
            onPerformanceMetrics: (metrics) => fpsValue = metrics.fps,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
      // We didn't push any real events into the EventChannel, so no results yet
      expect(received, isEmpty);
      expect(fpsValue, isNull);
    });

    testWidgets('YOLOView can rebuild with same controller without crashes', (
      WidgetTester tester,
    ) async {
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      final channel = setup.$1;
      final controller = YOLOViewController()..init(channel, 999);

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            controller: controller,
            models: const [
              YOLOModelSpec(
                modelPath: 'assets/models/yolo11n.tflite',
                task: YOLOTask.detect,
              ),
            ],
          ),
        ),
      );
      expect(find.byType(YOLOView), findsOneWidget);

      // Rebuild with different models to simulate didUpdateWidget()
      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            controller: controller,
            models: const [
              YOLOModelSpec(
                modelPath: 'assets/models/yolo11n-seg.tflite',
                task: YOLOTask.segment,
              ),
            ],
          ),
        ),
      );
      expect(find.byType(YOLOView), findsOneWidget);
    });
  });
}
