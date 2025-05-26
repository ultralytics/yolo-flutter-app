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

  group('YoloView Comprehensive Coverage', () {
    // Helper to create a mock stream handler
    StreamController<dynamic> createMockEventStream(String channelName) {
      final controller = StreamController<dynamic>.broadcast();
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockStreamHandler(
        EventChannel(channelName),
        TestMockStreamHandler(streamController: controller),
      );
      return controller;
    }

    testWidgets('event stream with all data types', (tester) async {
      final uniqueKey = UniqueKey().toString();
      final eventChannelName =
          'com.ultralytics.yolo/detectionResults_$uniqueKey';
      final streamController = createMockEventStream(eventChannelName);

      final receivedResults = <YOLOResult>[];
      final receivedMetrics = <String, double>{};

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: (results) => receivedResults.addAll(results),
            onPerformanceMetrics: (metrics) => receivedMetrics.addAll(metrics),
          ),
        ),
      );

      // Send valid detection with all optional fields
      streamController.add({
        'detections': [
          {
            'classIndex': 0,
            'className': 'person',
            'confidence': 0.95,
            'boundingBox': {
              'x': 10.0,
              'y': 20.0,
              'width': 100.0,
              'height': 200.0
            },
            'normalizedBox': {'x': 0.1, 'y': 0.2, 'width': 0.5, 'height': 0.8},
            'mask': [
              [0.1, 0.2],
              [0.3, 0.4]
            ],
            'keypoints': [
              {'x': 15.0, 'y': 25.0},
              {'x': 20.0, 'y': 30.0},
            ],
          },
        ],
        'processingTimeMs': 15.5,
        'fps': 60.0,
      });

      await tester.pump();

      expect(receivedResults.length, 1);
      expect(receivedResults[0].className, 'person');
      expect(receivedResults[0].confidence, 0.95);
      expect(receivedMetrics['processingTimeMs'], 15.5);
      expect(receivedMetrics['fps'], 60.0);

      // Send test message
      streamController.add({'test': 'debugging'});
      await tester.pump();

      // Send invalid detection to trigger error handling
      streamController.add({
        'detections': [
          {'invalid': 'data'}, // Missing required fields
        ],
      });
      await tester.pump();

      // Send only performance metrics
      streamController.add({
        'processingTimeMs': 20.0,
        'fps': 45.0,
      });
      await tester.pump();

      expect(receivedMetrics['processingTimeMs'], 20.0);
      expect(receivedMetrics['fps'], 45.0);

      streamController.close();
    });

    testWidgets('error handling and recovery', (tester) async {
      final uniqueKey = UniqueKey().toString();
      final eventChannelName =
          'com.ultralytics.yolo/detectionResults_$uniqueKey';
      final streamController = createMockEventStream(eventChannelName);

      int errorCount = 0;

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      // Add error to stream
      streamController.addError('Connection lost', StackTrace.current);
      await tester.pump();

      // Wait for automatic reconnection attempt
      await tester.pump(const Duration(seconds: 2, milliseconds: 100));

      streamController.close();
    });

    testWidgets('platform view creation and method handling', (tester) async {
      final methodCalls = <MethodCall>[];
      final uniqueKey = UniqueKey().toString();
      final controlChannelName =
          'com.ultralytics.yolo/controlChannel_$uniqueKey';

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        MethodChannel(controlChannelName),
        (MethodCall methodCall) async {
          methodCalls.add(methodCall);
          return null;
        },
      );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.pose,
            showNativeUI: true,
            onResult: (_) {},
          ),
        ),
      );

      await tester.pumpAndSettle();

      // Simulate platform requesting channel recreation
      final message = const StandardMethodCodec().encodeMethodCall(
        const MethodCall('recreateEventChannel'),
      );

      await TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .handlePlatformMessage(
        controlChannelName,
        message,
        (ByteData? data) {},
      );

      await tester.pump(const Duration(milliseconds: 150));

      // Check that setShowUIControls was called
      expect(
        methodCalls.any((call) => call.method == 'setShowUIControls'),
        isTrue,
      );
    });

    testWidgets('empty and null detection handling', (tester) async {
      final uniqueKey = UniqueKey().toString();
      final eventChannelName =
          'com.ultralytics.yolo/detectionResults_$uniqueKey';
      final streamController = createMockEventStream(eventChannelName);

      final receivedResults = <YOLOResult>[];

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.segment,
            onResult: (results) => receivedResults.addAll(results),
          ),
        ),
      );

      // Send empty detections list
      streamController.add({
        'detections': [],
      });
      await tester.pump();

      expect(receivedResults.isEmpty, true);

      // Send null detections
      streamController.add({
        'detections': null,
      });
      await tester.pump();

      expect(receivedResults.isEmpty, true);

      // Send mixed valid and invalid detections
      streamController.add({
        'detections': [
          null,
          {'invalid': 'data'},
          {
            'classIndex': 2,
            'className': 'dog',
            'confidence': 0.7,
            'boundingBox': {'x': 0.0, 'y': 0.0, 'width': 50.0, 'height': 50.0},
          },
        ],
      });
      await tester.pump();

      streamController.close();
    });

    testWidgets('performance metrics only without detections', (tester) async {
      final uniqueKey = UniqueKey().toString();
      final eventChannelName =
          'com.ultralytics.yolo/detectionResults_$uniqueKey';
      final streamController = createMockEventStream(eventChannelName);

      final receivedMetrics = <String, double>{};

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.obb,
            onPerformanceMetrics: (metrics) => receivedMetrics.addAll(metrics),
          ),
        ),
      );

      // Send only metrics without detections
      streamController.add({
        'processingTimeMs': 12.5,
        'fps': 75.0,
      });
      await tester.pump();

      expect(receivedMetrics['processingTimeMs'], 12.5);
      expect(receivedMetrics['fps'], 75.0);

      // Send metrics with null values
      streamController.add({
        'processingTimeMs': null,
        'fps': null,
      });
      await tester.pump();

      // Send partial metrics
      streamController.add({
        'processingTimeMs': 8.0,
      });
      await tester.pump();

      streamController.close();
    });

    testWidgets('stream done callback handling', (tester) async {
      final uniqueKey = UniqueKey().toString();
      final eventChannelName =
          'com.ultralytics.yolo/detectionResults_$uniqueKey';
      final streamController = createMockEventStream(eventChannelName);

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.classify,
            onResult: (_) {},
          ),
        ),
      );

      // Close stream to trigger onDone
      streamController.close();
      await tester.pump();
    });

    testWidgets('unknown method call handling', (tester) async {
      final uniqueKey = UniqueKey().toString();
      final controlChannelName =
          'com.ultralytics.yolo/controlChannel_$uniqueKey';

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        MethodChannel(controlChannelName),
        (MethodCall methodCall) async {
          return null;
        },
      );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: (_) {},
          ),
        ),
      );

      // Send unknown method call
      final message = const StandardMethodCodec().encodeMethodCall(
        const MethodCall('unknownMethod', {'param': 'value'}),
      );

      await TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .handlePlatformMessage(
        controlChannelName,
        message,
        (ByteData? data) {},
      );

      await tester.pump();
    });
  });
}

class TestMockStreamHandler extends MockStreamHandler {
  final StreamController<dynamic>? streamController;

  TestMockStreamHandler({this.streamController});

  @override
  void onListen(Object? arguments, MockStreamHandlerEventSink events) {
    streamController?.stream.listen(
      events.success,
      onError: events.error,
      onDone: events.endOfStream,
    );
  }

  @override
  void onCancel(Object? arguments) {}
}
