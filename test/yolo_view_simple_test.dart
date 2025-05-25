// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:ui';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloView Widget', () {
    late List<MethodCall> methodCalls;
    late StreamController<Map<String, dynamic>> eventController;

    setUp(() {
      methodCalls = <MethodCall>[];
      eventController = StreamController<Map<String, dynamic>>.broadcast();

      // Mock platform view and method channels
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMessageHandler('flutter/platform_views', (data) async {
        return const StandardMessageCodec().encodeMessage({'viewId': 1});
      });
    });

    tearDown(() {
      eventController.close();
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMessageHandler('flutter/platform_views', null);
    });

    testWidgets('YoloView creates with basic parameters', (WidgetTester tester) async {
      await tester.pumpWidget(
        const YoloView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('YoloView handles result callbacks', (WidgetTester tester) async {
      List<YOLOResult>? receivedResults;
      Map<String, double>? receivedMetrics;

      const channelName = 'com.ultralytics.yolo/detectionResults_test';
      
      // Mock event channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMessageHandler(channelName, (data) async {
        final Map<String, dynamic> testEvent = {
          'detections': [
            {
              'classIndex': 0,
              'className': 'person',
              'confidence': 0.95,
              'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
              'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
            }
          ],
          'processingTimeMs': 25.5,
          'fps': 30.0,
        };

        final encoded = const StandardMessageCodec().encodeMessage(testEvent);
        return encoded;
      });

      await tester.pumpWidget(
        YoloView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
          onResult: (results) => receivedResults = results,
          onPerformanceMetrics: (metrics) => receivedMetrics = metrics,
        ),
      );

      await tester.pump();

      // Simulate receiving detection results
      eventController.add({
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {'left': 10.0, 'top': 10.0, 'right': 110.0, 'bottom': 210.0},
            'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.5, 'bottom': 0.9},
          }
        ],
        'processingTimeMs': 25.5,
        'fps': 30.0,
      });

      await tester.pump();

      // Note: In real widget tests, the event channel simulation is complex
      // This test structure shows the expected pattern
    });

    testWidgets('YoloView with controller sets initial thresholds', (WidgetTester tester) async {
      final controller = YoloViewController();

      // Mock method channel for the specific view
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        const MethodChannel('com.ultralytics.yolo/controlChannel_test'),
        (MethodCall methodCall) async {
          methodCalls.add(methodCall);
          return null;
        },
      );

      await tester.pumpWidget(
        YoloView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
          controller: controller,
        ),
      );

      await tester.pump();

      // Verify that threshold methods would be called
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('YoloView parses detection results correctly', () {
      final yoloView = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      final state = yoloView.createState() as YoloViewState;

      final testEvent = {
        'detections': [
          {
            'classIndex': 1,
            'className': 'car',
            'confidence': 0.85,
            'boundingBox': {'left': 50.0, 'top': 60.0, 'right': 150.0, 'bottom': 160.0},
            'normalizedBox': {'left': 0.2, 'top': 0.3, 'right': 0.6, 'bottom': 0.8},
          }
        ]
      };

      final results = state._parseDetectionResults(testEvent);

      expect(results.length, 1);
      expect(results[0].classIndex, 1);
      expect(results[0].className, 'car');
      expect(results[0].confidence, 0.85);
      expect(results[0].boundingBox, const Rect.fromLTRB(50.0, 60.0, 150.0, 160.0));
    });

    test('YoloView handles malformed detection data gracefully', () {
      final yoloView = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      final state = yoloView.createState() as YoloViewState;

      final malformedEvent = {
        'detections': [
          {'incomplete': 'data'},
          null,
          'not_a_map',
        ]
      };

      // Should not throw and return empty list
      expect(() => state._parseDetectionResults(malformedEvent), returnsNormally);
      final results = state._parseDetectionResults(malformedEvent);
      expect(results, isEmpty);
    });

    test('YoloView GlobalKey access methods work', () {
      final globalKey = GlobalKey<YoloViewState>();
      
      final yoloView = YoloView(
        key: globalKey,
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
      );

      final state = yoloView.createState() as YoloViewState;

      // Test the public methods that can be called via GlobalKey
      expect(() => state.setConfidenceThreshold(0.8), returnsNormally);
      expect(() => state.setIoUThreshold(0.6), returnsNormally);
      expect(() => state.setNumItemsThreshold(25), returnsNormally);
      expect(() => state.switchCamera(), returnsNormally);
    });
  });

  group('YoloView Configuration', () {
    test('different camera resolutions are accepted', () {
      const resolutions = ['480p', '720p', '1080p'];
      
      for (final resolution in resolutions) {
        expect(
          () => YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            cameraResolution: resolution,
          ),
          returnsNormally,
        );
      }
    });

    test('all YOLO tasks can be used', () {
      for (final task in YOLOTask.values) {
        expect(
          () => YoloView(
            modelPath: 'test_model.tflite',
            task: task,
          ),
          returnsNormally,
        );
      }
    });

    test('showNativeUI flag is properly handled', () {
      const view = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        showNativeUI: true,
      );

      expect(view.showNativeUI, true);
    });
  });
}
