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
    late StreamController<dynamic> mockStreamController;

    setUp(() {
      methodCalls = <MethodCall>[];
      mockStreamController = StreamController<dynamic>.broadcast();
    });

    tearDown(() {
      mockStreamController.close();
    });

    testWidgets('didUpdateWidget triggers controller changes', (
      WidgetTester tester,
    ) async {
      final controller1 = YoloViewController();
      final controller2 = YoloViewController();

      await controller1.setConfidenceThreshold(0.7);
      await controller2.setConfidenceThreshold(0.9);

      // Initial widget with controller1
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            controller: controller1,
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

    testWidgets('didUpdateWidget with different parameters', (
      WidgetTester tester,
    ) async {
      // Initial widget
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.detect,
            showNativeUI: false,
          ),
        ),
      );

      // Update with different parameters - triggers didUpdateWidget
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'updated_model.tflite',
            task: YOLOTask.segment,
            showNativeUI: true,
          ),
        ),
      );

      await tester.pump();
    });

    testWidgets('_onPlatformViewCreated is called during widget creation', (
      WidgetTester tester,
    ) async {
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
        MaterialApp(
          home: YoloView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      await tester.pump();
    });

    testWidgets(
      'result stream handles detection events with onResult callback',
      (WidgetTester tester) async {
        final List<List<YOLOResult>> receivedResults = [];

        // Mock event channel
        const eventChannelName = 'ultralytics_yolo/yolo_results_0';
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockStreamHandler(
              const EventChannel(eventChannelName),
              MockStreamHandler(mockStreamController.stream),
            );

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

        // Simulate detection event
        mockStreamController.add({
          'detections': [
            {
              'classIndex': 0,
              'className': 'person',
              'confidence': 0.95,
              'boundingBox': {
                'left': 10.0,
                'top': 10.0,
                'right': 110.0,
                'bottom': 210.0,
              },
              'normalizedBox': {
                'left': 0.1,
                'top': 0.1,
                'right': 0.5,
                'bottom': 0.9,
              },
            },
          ],
        });

        await tester.pump();
        expect(receivedResults.length, 1);
        expect(receivedResults.first.length, 1);
        expect(receivedResults.first.first.className, 'person');
      },
    );

    testWidgets('result stream handles performance metrics', (
      WidgetTester tester,
    ) async {
      final List<Map<String, double>> receivedMetrics = [];

      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

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

      // Simulate performance metrics event
      mockStreamController.add({'processingTimeMs': 50.5, 'fps': 30.0});

      await tester.pump();
      expect(receivedMetrics.length, 1);
      expect(receivedMetrics.first['processingTimeMs'], 50.5);
      expect(receivedMetrics.first['fps'], 30.0);
    });

    testWidgets('result stream handles test messages', (
      WidgetTester tester,
    ) async {
      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      await tester.pump();

      // Simulate test message - should be handled without errors
      mockStreamController.add({'test': 'test message from platform'});

      await tester.pump();
    });

    testWidgets('result stream handles malformed detection data', (
      WidgetTester tester,
    ) async {
      final List<List<YOLOResult>> receivedResults = [];

      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

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

      // Simulate malformed detection data - should trigger error handling
      mockStreamController.add({
        'detections': [
          {
            'classIndex': 'invalid_type',
            'className': null,
            'confidence': 'not_a_number',
            // Missing required fields
          },
          null, // Null detection
          'not_a_map', // Invalid type
        ],
      });

      await tester.pump();
      // Should handle gracefully, possibly with empty results
    });

    testWidgets('result stream handles malformed performance metrics', (
      WidgetTester tester,
    ) async {
      final List<Map<String, double>> receivedMetrics = [];

      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

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

      // Simulate malformed metrics - should trigger error handling
      mockStreamController.add({
        'processingTimeMs': 'invalid_number',
        'fps': null,
      });

      await tester.pump();
      // Should handle gracefully, no callback should be called with invalid data
      expect(receivedMetrics.length, 0);
    });

    testWidgets('result stream handles invalid event types', (
      WidgetTester tester,
    ) async {
      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      await tester.pump();

      // Simulate invalid event types
      mockStreamController.add('invalid_string_event');
      mockStreamController.add(123);
      mockStreamController.add(null);
      mockStreamController.add([1, 2, 3]);

      await tester.pump();
    });

    testWidgets('result stream handles errors and resubscription', (
      WidgetTester tester,
    ) async {
      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      await tester.pump();

      // Simulate stream error
      mockStreamController.addError('Stream error occurred');

      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
    });

    testWidgets('result stream onDone callback', (WidgetTester tester) async {
      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(modelPath: 'test_model.tflite', task: YOLOTask.detect),
        ),
      );

      await tester.pump();

      // Close the stream to trigger onDone
      mockStreamController.close();

      await tester.pump();
    });

    testWidgets('_parseDetectionResults handles segmentation masks', (
      WidgetTester tester,
    ) async {
      final List<List<YOLOResult>> receivedResults = [];

      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.segment,
            onResult: (results) {
              receivedResults.add(results);
            },
          ),
        ),
      );

      await tester.pump();

      // Simulate segmentation detection with mask
      mockStreamController.add({
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
            'mask': [
              [0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
            ], // Segmentation mask
          },
        ],
      });

      await tester.pump();
      expect(receivedResults.length, 1);
      expect(receivedResults.first.first.mask, isNotNull);
      expect(receivedResults.first.first.mask!.length, 2);
    });

    testWidgets('_parseDetectionResults handles pose keypoints', (
      WidgetTester tester,
    ) async {
      final List<List<YOLOResult>> receivedResults = [];

      const eventChannelName = 'ultralytics_yolo/yolo_results_0';
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
            const EventChannel(eventChannelName),
            MockStreamHandler(mockStreamController.stream),
          );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test_model.tflite',
            task: YOLOTask.pose,
            onResult: (results) {
              receivedResults.add(results);
            },
          ),
        ),
      );

      await tester.pump();

      // Simulate pose detection with keypoints
      mockStreamController.add({
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'left': 10.0,
              'top': 10.0,
              'right': 110.0,
              'bottom': 210.0,
            },
            'normalizedBox': {
              'left': 0.1,
              'top': 0.1,
              'right': 0.5,
              'bottom': 0.9,
            },
            'keypoints': [
              100.0,
              200.0,
              0.9,
              150.0,
              250.0,
              0.8,
            ], // x1, y1, conf1, x2, y2, conf2
          },
        ],
      });

      await tester.pump();
      expect(receivedResults.length, 1);
      expect(receivedResults.first.first.keypoints, isNotNull);
      expect(receivedResults.first.first.keypoints!.length, 2);
      expect(receivedResults.first.first.keypointConfidences!.length, 2);
    });
  });
}

class MockStreamHandler implements MethodCallHandler {
  final Stream<dynamic> stream;

  MockStreamHandler(this.stream);

  @override
  Future<dynamic> call(MethodCall call) async {
    if (call.method == 'listen') {
      stream.listen(
        (event) {
          // Send events to Flutter
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .handlePlatformMessage(
                'ultralytics_yolo/yolo_results_0',
                const StandardMethodCodec().encodeSuccessEnvelope(event),
                (data) {},
              );
        },
        onError: (error) {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .handlePlatformMessage(
                'ultralytics_yolo/yolo_results_0',
                const StandardMethodCodec().encodeErrorEnvelope(
                  code: 'STREAM_ERROR',
                  message: error.toString(),
                ),
                (data) {},
              );
        },
        onDone: () {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .handlePlatformMessage(
                'ultralytics_yolo/yolo_results_0',
                null, // null indicates stream is done
                (data) {},
              );
        },
      );
      return null;
    } else if (call.method == 'cancel') {
      return null;
    }
    throw PlatformException(code: 'UNIMPLEMENTED');
  }
}
