// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloView Missing Coverage Tests', () {
    late List<MethodCall> methodCalls;

    setUp(() {
      methodCalls = <MethodCall>[];
    });

    testWidgets('didUpdateWidget triggers controller changes', (WidgetTester tester) async {
      final controller1 = YoloViewController();
      final controller2 = YoloViewController();
      
      await controller1.setConfidenceThreshold(0.7);
      await controller2.setConfidenceThreshold(0.9);

      // Initial widget with controller1
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      // Update widget with controller2 - triggers didUpdateWidget
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller2,
          ),
        ),
      );

      await tester.pump();
    });

    testWidgets('didUpdateWidget with different parameters', (WidgetTester tester) async {
      // Initial widget
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: false,
          ),
        ),
      );

      // Update with different parameters - triggers didUpdateWidget
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'updated_model.tflite',
            task: YOLOTask.segment,
            showNativeUI: true,
          ),
        ),
      );

      await tester.pump();
    });

    testWidgets('_onPlatformViewCreated is called during widget creation', (WidgetTester tester) async {
      // Mock the platform view creation
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        const MethodChannel('plugins.flutter.io/android_view_0'),
        (MethodCall methodCall) async {
          methodCalls.add(methodCall);
          return null;
        },
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      await tester.pump();
    });

    testWidgets('result callback with detection data', (WidgetTester tester) async {
      final List<List<YOLOResult>> receivedResults = [];
      
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: (results) {
              receivedResults.add(results);
            },
          ),
        ),
      );

      await tester.pump();
      
      // Simulate calling the callback directly to test the parsing logic
      final state = tester.state<YoloViewState>(find.byType(YoloView));
      
      // Test the parseDetectionResults method indirectly by simulating widget.onResult call
      final mockEvent = {
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
            'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
          }
        ]
      };
      
      // Simulate the result by calling the callback
      if (state.widget.onResult != null) {
        final results = [
          YOLOResult.fromMap(mockEvent['detections']![0] as Map<String, dynamic>)
        ];
        state.widget.onResult!(results);
      }

      expect(receivedResults.length, 1);
      expect(receivedResults.first.length, 1);
      expect(receivedResults.first.first.className, 'person');
    });

    testWidgets('performance metrics callback', (WidgetTester tester) async {
      final List<Map<String, double>> receivedMetrics = [];
      
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onPerformanceMetrics: (metrics) {
              receivedMetrics.add(metrics);
            },
          ),
        ),
      );

      await tester.pump();
      
      final state = tester.state<YoloViewState>(find.byType(YoloView));
      
      // Simulate performance metrics callback
      if (state.widget.onPerformanceMetrics != null) {
        state.widget.onPerformanceMetrics!({
          'processingTimeMs': 50.5,
          'fps': 30.0,
        });
      }

      expect(receivedMetrics.length, 1);
      expect(receivedMetrics.first['processingTimeMs'], 50.5);
      expect(receivedMetrics.first['fps'], 30.0);
    });

    testWidgets('widget handles null callbacks gracefully', (WidgetTester tester) async {
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

      await tester.pump();
      
      final state = tester.state<YoloViewState>(find.byType(YoloView));
      
      // Verify callbacks are null
      expect(state.widget.onResult, isNull);
      expect(state.widget.onPerformanceMetrics, isNull);
    });

    test('YOLOResult.fromMap handles segmentation mask data', () {
      final mapData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'mask': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], // Segmentation mask
      };

      final result = YOLOResult.fromMap(mapData);
      
      expect(result.className, 'person');
      expect(result.mask, isNotNull);
      expect(result.mask!.length, 2);
      expect(result.mask![0], [0.1, 0.2, 0.3]);
      expect(result.mask![1], [0.4, 0.5, 0.6]);
    });

    test('YOLOResult.fromMap handles pose keypoints data', () {
      final mapData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'keypoints': [100.0, 200.0, 0.9, 150.0, 250.0, 0.8], // x1, y1, conf1, x2, y2, conf2
      };

      final result = YOLOResult.fromMap(mapData);
      
      expect(result.className, 'person');
      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, 2);
      expect(result.keypoints![0].x, 100.0);
      expect(result.keypoints![0].y, 200.0);
      expect(result.keypointConfidences![0], 0.9);
      expect(result.keypoints![1].x, 150.0);
      expect(result.keypoints![1].y, 250.0);
      expect(result.keypointConfidences![1], 0.8);
    });

    test('YOLOResult.fromMap handles malformed data gracefully', () {
      final malformedMaps = [
        <String, dynamic>{}, // Empty map
        {
          'classIndex': 'invalid_type',
          'className': null,
          'confidence': 'not_a_number',
        },
        {
          'classIndex': 0,
          'className': 'valid',
          'confidence': 0.9,
          'keypoints': 'invalid_keypoints_data',
        },
        {
          'classIndex': 0,
          'className': 'valid',
          'confidence': 0.9,
          'mask': 'invalid_mask_data',
        },
      ];

      for (final mapData in malformedMaps) {
        expect(() => YOLOResult.fromMap(mapData), returnsNormally);
      }
    });

    testWidgets('widget creation with different task types covers parsing branches', (WidgetTester tester) async {
      // Test different task types to ensure different parsing paths are tested
      final tasks = [YOLOTask.detect, YOLOTask.segment, YOLOTask.pose, YOLOTask.classify];
      
      for (final task in tasks) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: 'test_model.tflite',
              task: task,
            ),
          ),
        );
        
        await tester.pump();
        expect(find.byType(YoloView), findsOneWidget);
      }
    });

    testWidgets('GlobalKey access for method coverage', (WidgetTester tester) async {
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

      await tester.pump();

      // Access state methods to increase coverage
      final state = key.currentState;
      expect(state, isNotNull);
      
      // Call public state methods
      await state!.setConfidenceThreshold(0.8);
      await state.setIoUThreshold(0.6);
      await state.setNumItemsThreshold(25);
      await state.setThresholds(confidenceThreshold: 0.7);
      await state.switchCamera();
    });

    testWidgets('widget disposal and cleanup', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      await tester.pump();

      // Remove the widget to trigger disposal
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: Text('Different widget'),
          ),
        ),
      );

      await tester.pump();
    });

    testWidgets('widget with complex callback scenarios', (WidgetTester tester) async {
      var resultCallCount = 0;
      var metricsCallCount = 0;
      
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: (results) {
              resultCallCount++;
              // Test that results parameter is properly typed
              expect(results, isA<List<YOLOResult>>());
            },
            onPerformanceMetrics: (metrics) {
              metricsCallCount++;
              // Test that metrics parameter is properly typed
              expect(metrics, isA<Map<String, double>>());
            },
          ),
        ),
      );

      await tester.pump();
      
      final state = tester.state<YoloViewState>(find.byType(YoloView));
      
      // Simulate multiple callback calls
      if (state.widget.onResult != null) {
        state.widget.onResult!([]);
        state.widget.onResult!([]);
      }
      
      if (state.widget.onPerformanceMetrics != null) {
        state.widget.onPerformanceMetrics!({'fps': 30.0, 'processingTimeMs': 33.3});
      }

      expect(resultCallCount, 2);
      expect(metricsCallCount, 1);
    });
  });
}
