// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:flutter/foundation.dart';
import 'utils/test_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOViewController', () {
    late YOLOViewController controller;
    late MethodChannel mockChannel;
    late List<MethodCall> log;

    setUp(() {
      controller = YOLOViewController();
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      mockChannel = setup.$1;
      log = setup.$2;
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(mockChannel, null);
      log.clear();
    });

    test('default values and threshold clamping', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);

      // Test clamping
      controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);
      controller.setConfidenceThreshold(-0.2);
      expect(controller.confidenceThreshold, 0.0);

      controller.setIoUThreshold(2.0);
      expect(controller.iouThreshold, 1.0);
      controller.setIoUThreshold(-1.0);
      expect(controller.iouThreshold, 0.0);

      controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);
      controller.setNumItemsThreshold(0);
      expect(controller.numItemsThreshold, 1);
    });

    test('setThresholds updates values correctly', () async {
      await controller.setThresholds(
        confidenceThreshold: 0.9,
        iouThreshold: 0.6,
        numItemsThreshold: 25,
      );

      expect(controller.confidenceThreshold, 0.9);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 25);

      // Test partial updates
      await controller.setThresholds(confidenceThreshold: 0.7);
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.6); // unchanged
      expect(controller.numItemsThreshold, 25); // unchanged
    });

    test('platform methods work with initialized channel', () async {
      controller.init(mockChannel, 1);

      // Test threshold methods
      YOLOTestHelpers.validateThresholdBehavior(controller, log, mockChannel);

      // Test camera controls
      await controller.switchCamera();
      YOLOTestHelpers.assertMethodCalled(log, 'switchCamera');

      await controller.zoomIn();
      YOLOTestHelpers.assertMethodCalled(log, 'zoomIn');

      await controller.zoomOut();
      YOLOTestHelpers.assertMethodCalled(log, 'zoomOut');

      await controller.setZoomLevel(2.0);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setZoomLevel',
        arguments: {'zoomLevel': 2.0},
      );

      // Test capture frame
      final result = await controller.captureFrame();
      expect(result, isA<Uint8List>());
      YOLOTestHelpers.assertMethodCalled(log, 'captureFrame');
    });

    test('methods handle uninitialized channel gracefully', () async {
      final uninitializedController = YOLOViewController();
      expect(
        () => uninitializedController.setConfidenceThreshold(0.8),
        returnsNormally,
      );
      expect(() => uninitializedController.switchCamera(), returnsNormally);
    });

    test('error handling works correctly', () async {
      controller.init(mockChannel, 1);

      // Test that errors are handled gracefully
      await controller.setConfidenceThreshold(0.8);
      await controller.setIoUThreshold(0.6);
      await controller.setNumItemsThreshold(50);
      await controller.switchCamera();
      await controller.zoomIn();
      await controller.zoomOut();
      await controller.setZoomLevel(2.0);
      await controller.stop();
      await controller.setShowUIControls(true);

      // All methods should complete without throwing
      expect(true, isTrue);
    });
  });

  group('YOLOView Widget', () {
    testWidgets('creates with various configurations', (
      WidgetTester tester,
    ) async {
      // Test minimal parameters
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );
      expect(find.byType(YOLOView), findsOneWidget);

      // Test with custom controller
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

      // Test with all optional parameters
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

    test('supports different task types and model paths', () {
      expect(YOLOTask.values.length, greaterThan(0));
      expect(YOLOTask.values.contains(YOLOTask.detect), true);
      expect(YOLOTask.values.contains(YOLOTask.segment), true);

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
      expect(widget1.modelPath, isA<String>());
      expect(widget1.modelPath.isNotEmpty, true);
    });

    testWidgets('handles callbacks correctly', (WidgetTester tester) async {
      final mockChannel = YOLOTestHelpers.setupMockChannel();
      final List<YOLOResult> capturedResults = [];
      YOLOPerformanceMetrics? capturedMetrics;

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'assets/yolo.tflite',
            task: YOLOTask.detect,
            controller: YOLOViewController()..init(mockChannel, 1),
            onResult: (results) {
              capturedResults.addAll(results);
            },
            onPerformanceMetrics: (metrics) {
              capturedMetrics = metrics;
            },
          ),
        ),
      );

      // Test that callbacks can be set without errors
      expect(find.byType(YOLOView), findsOneWidget);
      expect(capturedResults, isEmpty);
      expect(capturedMetrics, isNull);
    });

    testWidgets('handles widget updates correctly', (
      WidgetTester tester,
    ) async {
      final mockChannel = YOLOTestHelpers.setupMockChannel(
        customResponses: {'setModel': (_) => Future.value(null)},
      );
      final controller = YOLOViewController()..init(mockChannel, 1);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: YOLOView(
              modelPath: 'assets/yolo_old.tflite',
              task: YOLOTask.detect,
              controller: controller,
            ),
          ),
        ),
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: YOLOView(
              modelPath: 'assets/yolo_new.tflite',
              task: YOLOTask.segment,
              controller: controller,
            ),
          ),
        ),
      );

      // Test passes if no exceptions are thrown
      expect(true, isTrue);
    });

    testWidgets('handles disposal correctly', (WidgetTester tester) async {
      final mockChannel = YOLOTestHelpers.setupMockChannel(
        customResponses: {
          'stop': (_) => Future.value(null),
          'disposeInstance': (_) => Future.value(null),
        },
      );
      final controller = YOLOViewController()..init(mockChannel, 1);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: YOLOView(
              modelPath: 'assets/yolo.tflite',
              task: YOLOTask.detect,
              controller: controller,
            ),
          ),
        ),
      );

      await tester.pumpAndSettle();
      await tester.pumpWidget(Container()); // Dispose the widget

      // Test passes if no exceptions are thrown
      expect(true, isTrue);
    });

    testWidgets('fallback UI shown on unsupported platform', (
      WidgetTester tester,
    ) async {
      debugDefaultTargetPlatformOverride = TargetPlatform.fuchsia;

      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(modelPath: 'model.tflite', task: YOLOTask.detect),
        ),
      );

      expect(find.text('Platform not supported for YOLOView'), findsOneWidget);

      debugDefaultTargetPlatformOverride = null;
    });

    testWidgets('handles streaming data correctly', (
      WidgetTester tester,
    ) async {
      final mockChannel = YOLOTestHelpers.setupMockChannel();
      final List<Map<String, dynamic>> capturedStreamData = [];

      await tester.pumpWidget(
        MaterialApp(
          home: YOLOView(
            modelPath: 'assets/yolo.tflite',
            task: YOLOTask.detect,
            controller: YOLOViewController()..init(mockChannel, 1),
            onStreamingData: (data) {
              capturedStreamData.add(data);
            },
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
      expect(capturedStreamData, isEmpty);
    });

    testWidgets('handles different camera resolutions', (
      WidgetTester tester,
    ) async {
      const resolutions = ['720p', '1080p', '4K'];

      for (final resolution in resolutions) {
        await tester.pumpWidget(
          MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              cameraResolution: resolution,
            ),
          ),
        );

        expect(find.byType(YOLOView), findsOneWidget);
      }
    });

    testWidgets('handles showNativeUI parameter', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YOLOView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: true,
          ),
        ),
      );

      expect(find.byType(YOLOView), findsOneWidget);
    });
  });
}
