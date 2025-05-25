// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloViewController Public API', () {
    late YoloViewController controller;

    setUp(() {
      controller = YoloViewController();
    });

    test('initial values are set correctly', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('setConfidenceThreshold clamps and updates value', () async {
      await controller.setConfidenceThreshold(0.8);
      expect(controller.confidenceThreshold, 0.8);

      await controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setConfidenceThreshold(-0.2);
      expect(controller.confidenceThreshold, 0.0);

      await controller.setConfidenceThreshold(0.001);
      expect(controller.confidenceThreshold, 0.001);

      await controller.setConfidenceThreshold(0.999);
      expect(controller.confidenceThreshold, 0.999);
    });

    test('setIoUThreshold clamps and updates value', () async {
      await controller.setIoUThreshold(0.7);
      expect(controller.iouThreshold, 0.7);

      await controller.setIoUThreshold(2.0);
      expect(controller.iouThreshold, 1.0);

      await controller.setIoUThreshold(-1.0);
      expect(controller.iouThreshold, 0.0);
    });

    test('setNumItemsThreshold clamps and updates value', () async {
      await controller.setNumItemsThreshold(50);
      expect(controller.numItemsThreshold, 50);

      await controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);

      await controller.setNumItemsThreshold(-5);
      expect(controller.numItemsThreshold, 1);

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

      await controller.setThresholds(iouThreshold: 0.3);
      expect(controller.confidenceThreshold, 0.7); // unchanged
      expect(controller.iouThreshold, 0.3);
      expect(controller.numItemsThreshold, 30); // unchanged

      await controller.setThresholds(numItemsThreshold: 15);
      expect(controller.confidenceThreshold, 0.7); // unchanged
      expect(controller.iouThreshold, 0.3); // unchanged
      expect(controller.numItemsThreshold, 15);
    });

    test('setThresholds clamps values', () async {
      await controller.setThresholds(
        confidenceThreshold: 2.0,
        iouThreshold: -0.5,
        numItemsThreshold: 200,
      );

      expect(controller.confidenceThreshold, 1.0);
      expect(controller.iouThreshold, 0.0);
      expect(controller.numItemsThreshold, 100);
    });

    test('switchCamera completes without error', () async {
      // Should not throw even without method channel
      expect(() => controller.switchCamera(), returnsNormally);
    });

    test('boundary value testing for thresholds', () async {
      final testCases = [
        {'input': 0.0, 'expected': 0.0},
        {'input': 1.0, 'expected': 1.0},
        {'input': 0.5, 'expected': 0.5},
        {'input': 0.0001, 'expected': 0.0001},
        {'input': 0.9999, 'expected': 0.9999},
      ];

      for (final testCase in testCases) {
        await controller.setConfidenceThreshold(testCase['input']!);
        expect(controller.confidenceThreshold, testCase['expected']);

        await controller.setIoUThreshold(testCase['input']!);
        expect(controller.iouThreshold, testCase['expected']);
      }
    });

    test('boundary value testing for numItems', () async {
      final testCases = [
        {'input': 1, 'expected': 1},
        {'input': 100, 'expected': 100},
        {'input': 50, 'expected': 50},
        {'input': 99, 'expected': 99},
        {'input': 2, 'expected': 2},
      ];

      for (final testCase in testCases) {
        await controller.setNumItemsThreshold(testCase['input']!);
        expect(controller.numItemsThreshold, testCase['expected']);
      }
    });
  });

  group('YoloView Widget Creation', () {
    testWidgets('creates with minimal required parameters', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('creates with all parameters', (WidgetTester tester) async {
      final controller = YoloViewController();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'custom_model.tflite',
            task: YOLOTask.segment,
            controller: controller,
            cameraResolution: '1080p',
            onResult: (results) {
              // Callback provided but we're just testing widget creation
            },
            onPerformanceMetrics: (metrics) {
              // Callback provided but we're just testing widget creation
            },
            showNativeUI: true,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('creates with different tasks', (WidgetTester tester) async {
      for (final task in YOLOTask.values) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: 'test_model.tflite',
              task: task,
            ),
          ),
        );

        expect(find.byType(YoloView), findsOneWidget);
        await tester.pumpAndSettle();
      }
    });

    testWidgets('creates with different camera resolutions', (WidgetTester tester) async {
      const resolutions = ['480p', '720p', '1080p', '4K'];

      for (final resolution in resolutions) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              cameraResolution: resolution,
            ),
          ),
        );

        expect(find.byType(YoloView), findsOneWidget);
        await tester.pumpAndSettle();
      }
    });

    testWidgets('creates with showNativeUI variations', (WidgetTester tester) async {
      // Test with showNativeUI = false
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: false,
          ),
        ),
      );
      expect(find.byType(YoloView), findsOneWidget);

      // Test with showNativeUI = true
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: true,
          ),
        ),
      );
      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('widget properties are accessible', (WidgetTester tester) async {
      final controller = YoloViewController();
      final onResultCalled = <YOLOResult>[];
      final onMetricsCalled = <Map<String, double>>[];

      final widget = YoloView(
        modelPath: 'specific_model.mlpackage',
        task: YOLOTask.pose,
        controller: controller,
        cameraResolution: '720p',
        onResult: (results) => onResultCalled.add(results.first),
        onPerformanceMetrics: (metrics) => onMetricsCalled.add(metrics),
        showNativeUI: true,
      );

      expect(widget.modelPath, 'specific_model.mlpackage');
      expect(widget.task, YOLOTask.pose);
      expect(widget.controller, equals(controller));
      expect(widget.cameraResolution, '720p');
      expect(widget.onResult, isNotNull);
      expect(widget.onPerformanceMetrics, isNotNull);
      expect(widget.showNativeUI, true);
    });
  });

  group('YoloView GlobalKey Access', () {
    testWidgets('can access state methods via GlobalKey', (WidgetTester tester) async {
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

      // These methods should be accessible via GlobalKey
      expect(() => key.currentState?.setConfidenceThreshold(0.8), returnsNormally);
      expect(() => key.currentState?.setIoUThreshold(0.6), returnsNormally);
      expect(() => key.currentState?.setNumItemsThreshold(25), returnsNormally);
      expect(() => key.currentState?.setThresholds(confidenceThreshold: 0.7), returnsNormally);
      expect(() => key.currentState?.switchCamera(), returnsNormally);
    });

    testWidgets('GlobalKey methods delegate correctly', (WidgetTester tester) async {
      final key = GlobalKey<YoloViewState>();
      final controller = YoloViewController();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            key: key,
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller,
          ),
        ),
      );

      await tester.pumpAndSettle();

      // Call methods via GlobalKey
      await key.currentState?.setConfidenceThreshold(0.9);
      await key.currentState?.setIoUThreshold(0.7);
      await key.currentState?.setNumItemsThreshold(40);

      // Values should be updated in the controller
      expect(controller.confidenceThreshold, 0.9);
      expect(controller.iouThreshold, 0.7);
      expect(controller.numItemsThreshold, 40);
    });
  });

  group('YoloView Callback Handling', () {
    testWidgets('handles null callbacks gracefully', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
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

    testWidgets('stores callback references correctly', (WidgetTester tester) async {
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

      // Simulate callback calls
      widget.onResult!([]);
      widget.onPerformanceMetrics!({});

      expect(resultCallCount, 1);
      expect(metricsCallCount, 1);
    });
  });

  group('YoloView Model Path Variations', () {
    testWidgets('handles different model path formats', (WidgetTester tester) async {
      final modelPaths = [
        'yolo11n.tflite',
        'assets/models/yolo11s.tflite',
        '/data/app/models/custom.tflite',
        'internal://models/yolo11n-seg.tflite',
        'yolo11n.mlpackage',
        'models/pose/yolo11n-pose.mlpackage',
        'classification_model.tflite',
        'obb_detection.tflite',
      ];

      for (final modelPath in modelPaths) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: modelPath,
              task: YOLOTask.detect,
            ),
          ),
        );

        expect(find.byType(YoloView), findsOneWidget);
        await tester.pumpAndSettle();
      }
    });

    testWidgets('handles unicode model paths', (WidgetTester tester) async {
      const unicodePaths = [
        'models/测试模型.tflite',
        'модели/yolo11n.tflite',
        'モデル/detection.tflite',
        'models/émojí_🤖_model.tflite',
      ];

      for (final modelPath in unicodePaths) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: modelPath,
              task: YOLOTask.detect,
            ),
          ),
        );

        expect(find.byType(YoloView), findsOneWidget);
        await tester.pumpAndSettle();
      }
    });
  });

  group('YoloView Edge Cases', () {
    testWidgets('handles controller changes', (WidgetTester tester) async {
      final controller1 = YoloViewController();
      final controller2 = YoloViewController();

      await controller1.setConfidenceThreshold(0.8);
      await controller2.setConfidenceThreshold(0.6);

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller1,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);

      // Change controller
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller2,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles rapid widget rebuilds', (WidgetTester tester) async {
      for (var i = 0; i < 10; i++) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: 'test_model_$i.tflite',
              task: YOLOTask.values[i % YOLOTask.values.length],
              showNativeUI: i % 2 == 0,
            ),
          ),
        );

        expect(find.byType(YoloView), findsOneWidget);
        await tester.pump();
      }
    });

    testWidgets('handles extreme threshold values through widget', (WidgetTester tester) async {
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

      // Test extreme values
      await controller.setConfidenceThreshold(double.maxFinite);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setConfidenceThreshold(double.negativeInfinity);
      expect(controller.confidenceThreshold, 0.0);

      await controller.setNumItemsThreshold(999999);
      expect(controller.numItemsThreshold, 100);
    });
  });
}
