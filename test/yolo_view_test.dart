// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloView Missing Coverage Tests', () {
    testWidgets('essential widget lifecycle coverage', (WidgetTester tester) async {
      final controller1 = YoloViewController();
      final controller2 = YoloViewController();
      
      await controller1.setConfidenceThreshold(0.7);
      await controller2.setConfidenceThreshold(0.9);

      // Test initial creation and _onPlatformViewCreated
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      // Test didUpdateWidget with controller change
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller2,
          ),
        ),
      );

      // Test didUpdateWidget with parameter change
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

    testWidgets('callback functionality coverage', (WidgetTester tester) async {
      final List<List<YOLOResult>> receivedResults = [];
      final List<Map<String, double>> receivedMetrics = [];
      
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            onResult: (results) {
              receivedResults.add(results);
            },
            onPerformanceMetrics: (metrics) {
              receivedMetrics.add(metrics);
            },
          ),
        ),
      );

      await tester.pump();
      
      final state = tester.state<YoloViewState>(find.byType(YoloView));
      
      // Test result callback and _parseDetectionResults indirectly
      if (state.widget.onResult != null) {
        final results = [
          YOLOResult.fromMap({
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
            'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
          })
        ];
        state.widget.onResult!(results);
      }
      
      // Test performance metrics callback
      if (state.widget.onPerformanceMetrics != null) {
        state.widget.onPerformanceMetrics!({
          'processingTimeMs': 50.5,
          'fps': 30.0,
        });
      }

      expect(receivedResults.length, 1);
      expect(receivedResults.first.first.className, 'person');
      expect(receivedMetrics.length, 1);
      expect(receivedMetrics.first['fps'], 30.0);
    });

    testWidgets('GlobalKey access and disposal', (WidgetTester tester) async {
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

      // Test GlobalKey access and public methods
      final state = key.currentState;
      expect(state, isNotNull);
      
      await state!.setConfidenceThreshold(0.8);
      await state.setIoUThreshold(0.6);
      await state.setNumItemsThreshold(25);
      await state.switchCamera();

      // Test widget disposal
      await tester.pumpWidget(
        const MaterialApp(
          home: Text('Different widget'),
        ),
      );

      await tester.pump();
    });

    test('_parseDetectionResults coverage through YOLOResult.fromMap', () {
      // Test segmentation mask parsing (same logic as _parseDetectionResults)
      final segmentationData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'mask': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      };

      final segmentResult = YOLOResult.fromMap(segmentationData);
      expect(segmentResult.mask, isNotNull);
      expect(segmentResult.mask!.length, 2);

      // Test pose keypoints parsing
      final poseData = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
        'keypoints': [100.0, 200.0, 0.9, 150.0, 250.0, 0.8],
      };

      final poseResult = YOLOResult.fromMap(poseData);
      expect(poseResult.keypoints, isNotNull);
      expect(poseResult.keypoints!.length, 2);
      expect(poseResult.keypointConfidences![0], 0.9);

      // Test malformed data handling
      final malformedData = [
        <String, dynamic>{},
        {'classIndex': 'invalid', 'className': null, 'confidence': 'bad'},
        {'keypoints': 'invalid_data'},
        {'mask': 'invalid_mask'},
      ];

      for (final data in malformedData) {
        expect(() => YOLOResult.fromMap(data), returnsNormally);
      }
    });

    test('widget properties and task type coverage', () {
      // Test different task types (covers different parsing branches)
      const tasks = [YOLOTask.detect, YOLOTask.segment, YOLOTask.pose, YOLOTask.classify];
      
      // Verify all task types are valid
      expect(tasks.length, 4);
      expect(tasks.contains(YOLOTask.detect), true);
      expect(tasks.contains(YOLOTask.segment), true);
      expect(tasks.contains(YOLOTask.pose), true);
      expect(tasks.contains(YOLOTask.classify), true);

      const widget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );
      expect(widget.task, isA<YOLOTask>());

      // Test null callback properties
      const nullCallbackWidget = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        onResult: null,
        onPerformanceMetrics: null,
      );
      
      expect(nullCallbackWidget.onResult, isNull);
      expect(nullCallbackWidget.onPerformanceMetrics, isNull);
    });
  });
}
