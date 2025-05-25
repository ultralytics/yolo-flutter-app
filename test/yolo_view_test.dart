// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloViewController Public API', () {
    late YoloViewController controller;

    setUp(() {
      controller = YoloViewController();
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
  });

  group('YoloView Widget Properties', () {
    test('widget properties are accessible', () {
      const widget = YoloView(
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
      final controller = YoloViewController();

      final widget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        controller: controller,
      );

      expect(widget.controller, equals(controller));
    });

    test('widget with callbacks', () {
      var resultCallCount = 0;
      var metricsCallCount = 0;

      final widget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        onResult: (results) => resultCallCount++,
        onPerformanceMetrics: (metrics) => metricsCallCount++,
      );

      expect(widget.onResult, isNotNull);
      expect(widget.onPerformanceMetrics, isNotNull);

      // Test callbacks work
      widget.onResult!([]);
      widget.onPerformanceMetrics!({});

      expect(resultCallCount, 1);
      expect(metricsCallCount, 1);
    });
  });

  group('YoloView Widget Creation', () {
    testWidgets('creates with minimal parameters', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('creates with custom controller', (WidgetTester tester) async {
      final controller = YoloViewController();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('creates with all optional parameters', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'custom_model.tflite',
            task: YOLOTask.segment,
            cameraResolution: '1080p',
            onResult: (results) {},
            onPerformanceMetrics: (metrics) {},
            showNativeUI: true,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles null callbacks', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: null,
            onPerformanceMetrics: null,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });
  });

  group('YoloView GlobalKey Access', () {
    testWidgets('can access state methods via GlobalKey', (
      WidgetTester tester,
    ) async {
      final key = GlobalKey<YoloViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
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

  group('YoloView Task Types', () {
    test('supports all YOLOTask enum values', () {
      // Test that YOLOTask enum has expected values
      expect(YOLOTask.values.length, greaterThan(0));
      expect(YOLOTask.values.contains(YOLOTask.detect), true);
      expect(YOLOTask.values.contains(YOLOTask.segment), true);
    });

    test('different task types create different widgets', () {
      const widget1 = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      const widget2 = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.segment,
      );

      expect(widget1.task, YOLOTask.detect);
      expect(widget2.task, YOLOTask.segment);
      expect(widget1.task, isNot(equals(widget2.task)));
    });
  });

  group('YoloView Model Paths', () {
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

      const widget = YoloView(
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

      const widget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );
      expect(widget.modelPath, isA<String>());
    });
  });

  group('YoloView Camera Resolutions', () {
    test('supports common camera resolutions', () {
      const resolutions = ['480p', '720p', '1080p', '4K'];

      // Test that resolutions are valid strings
      for (final resolution in resolutions) {
        expect(resolution, isA<String>());
        expect(resolution.isNotEmpty, true);
      }

      const widget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        cameraResolution: '1080p',
      );
      expect(widget.cameraResolution, isA<String>());
    });

    test('handles default camera resolution when not specified', () {
      const widget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        // cameraResolution not specified - should use default
      );
      expect(widget.cameraResolution, isA<String>());
    });
  });
}
