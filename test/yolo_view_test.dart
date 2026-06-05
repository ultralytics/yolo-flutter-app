// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
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
      expect(controller.confidenceThreshold, 0.25);
      expect(controller.iouThreshold, 0.7);
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

      expect(find.byType(YOLOView), findsOneWidget);
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

      expect(find.byType(YOLOView), findsNothing);
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

    testWidgets(
      'drives platform view, event stream, and controller callbacks',
      (WidgetTester tester) async {
        debugDefaultTargetPlatformOverride = TargetPlatform.android;
        addTearDown(() => debugDefaultTargetPlatformOverride = null);

        final singleImageChannel = YOLOTestHelpers.setupMockChannel();
        addTearDown(() {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .setMockMethodCallHandler(singleImageChannel, null);
        });

        final results = <YOLOResult>[];
        final metrics = <YOLOPerformanceMetrics>[];
        final streamingData = <Map<String, dynamic>>[];
        final zoomLevels = <double>[];
        final loadedModels = <String>[];
        final controlCalls = <MethodCall>[];
        MockStreamHandlerEventSink? events;

        await tester.pumpWidget(
          MaterialApp(
            home: YOLOView(
              modelPath: 'camera_model.tflite',
              task: YOLOTask.detect,
              confidenceThreshold: 0.4,
              iouThreshold: 0.6,
              useGpu: false,
              lensFacing: LensFacing.front,
              streamingConfig: const YOLOStreamingConfig(
                includeDetections: true,
                includeClassifications: true,
                includeProcessingTimeMs: true,
                includeFps: true,
                includeMasks: true,
                includePoses: true,
                includeOBB: true,
                includeOriginalImage: true,
                maxFPS: 15,
                throttleInterval: Duration(milliseconds: 120),
                inferenceFrequency: 3,
                skipFrames: 2,
              ),
              onResult: results.addAll,
              onPerformanceMetrics: metrics.add,
              onZoomChanged: zoomLevels.add,
              onModelLoad: (modelPath, _) => loadedModels.add(modelPath),
            ),
          ),
        );
        await tester.pump();

        final androidView = tester.widget<AndroidView>(
          find.byType(AndroidView),
        );
        final creationParams =
            androidView.creationParams! as Map<dynamic, dynamic>;
        expect(creationParams['modelPath'], 'camera_model.tflite');
        expect(creationParams['task'], 'detect');
        expect(creationParams['confidenceThreshold'], 0.4);
        expect(creationParams['iouThreshold'], 0.6);
        expect(creationParams['useGpu'], isFalse);
        expect(creationParams['lensFacing'], 'front');
        expect(creationParams['streamingConfig'], {
          'includeDetections': true,
          'includeClassifications': true,
          'includeProcessingTimeMs': true,
          'includeFps': true,
          'includeMasks': true,
          'includePoses': true,
          'includeOBB': true,
          'includeOriginalImage': true,
          'maxFPS': 15,
          'throttleIntervalMs': 120,
          'inferenceFrequency': 3,
          'skipFrames': 2,
        });
        expect(loadedModels, ['camera_model.tflite']);

        final viewId = creationParams['viewId'] as String;
        final controlChannel = MethodChannel(
          'com.ultralytics.yolo/controlChannel_$viewId',
        );
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(controlChannel, (call) async {
              controlCalls.add(call);
              return true;
            });
        addTearDown(() {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .setMockMethodCallHandler(controlChannel, null);
        });

        final eventChannel = EventChannel(
          'com.ultralytics.yolo/detectionResults_$viewId',
        );
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockStreamHandler(
              eventChannel,
              MockStreamHandler.inline(
                onListen: (_, eventSink) {
                  events = eventSink;
                },
              ),
            );
        addTearDown(() {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .setMockStreamHandler(eventChannel, null);
        });

        androidView.onPlatformViewCreated?.call(7);
        await tester.pump();

        expect(
          controlCalls.map((call) => call.method),
          contains('setStreamingConfig'),
        );

        events!.success({'type': 'zoom', 'value': 1.8});
        await tester.pump();
        expect(zoomLevels, isEmpty);

        await tester.binding.defaultBinaryMessenger.handlePlatformMessage(
          controlChannel.name,
          const StandardMethodCodec().encodeMethodCall(
            const MethodCall('onZoomChanged', 2.5),
          ),
          (_) {},
        );
        expect(zoomLevels, [2.5]);

        events!.success({
          'detections': [
            {
              'classIndex': 0,
              'className': 'person',
              'confidence': 0.9,
              'boundingBox': {
                'left': 10.0,
                'top': 20.0,
                'right': 30.0,
                'bottom': 40.0,
              },
              'normalizedBox': {
                'left': 0.1,
                'top': 0.2,
                'right': 0.3,
                'bottom': 0.4,
              },
            },
          ],
          'fps': 30.0,
          'processingTimeMs': 12.0,
          'preprocessTimeMs': 3.0,
          'inferenceTimeMs': 7.0,
          'postprocessTimeMs': 2.0,
        });
        await tester.pump();

        expect(results.single.className, 'person');
        expect(metrics.single.fps, 30.0);
        expect(streamingData, isEmpty);

        await tester.pumpWidget(
          MaterialApp(
            home: YOLOView(
              modelPath: 'camera_model.tflite',
              task: YOLOTask.detect,
              onStreamingData: streamingData.add,
            ),
          ),
        );
        await tester.pump();

        final streamingView = tester.widget<AndroidView>(
          find.byType(AndroidView),
        );
        final streamingParams =
            streamingView.creationParams! as Map<dynamic, dynamic>;
        final streamingViewId = streamingParams['viewId'] as String;
        MockStreamHandlerEventSink? streamingEvents;
        final streamingEventChannel = EventChannel(
          'com.ultralytics.yolo/detectionResults_$streamingViewId',
        );
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockStreamHandler(
              streamingEventChannel,
              MockStreamHandler.inline(
                onListen: (_, eventSink) {
                  streamingEvents = eventSink;
                },
              ),
            );
        addTearDown(() {
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
              .setMockStreamHandler(streamingEventChannel, null);
        });
        streamingView.onPlatformViewCreated?.call(8);
        await tester.pump();

        streamingEvents!.success({'fps': 60.0, 'frameId': 42});
        await tester.pump();
        expect(streamingData.single, {'fps': 60.0, 'frameId': 42});
        debugDefaultTargetPlatformOverride = null;
      },
    );

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

    group('lensFacing parameter', () {
      test('YOLOView defaults to LensFacing.back', () {
        const widget = YOLOView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
        );
        expect(widget.lensFacing, LensFacing.back);
      });

      test('YOLOView accepts LensFacing.back explicitly', () {
        const widget = YOLOView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
          lensFacing: LensFacing.back,
        );
        expect(widget.lensFacing, LensFacing.back);
      });

      test('YOLOView accepts LensFacing.front', () {
        const widget = YOLOView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
          lensFacing: LensFacing.front,
        );
        expect(widget.lensFacing, LensFacing.front);
      });

      test('YOLOView accepts LensFacing.backWide', () {
        const widget = YOLOView(
          modelPath: 'test_model.tflite',
          task: YOLOTask.detect,
          lensFacing: LensFacing.backWide,
        );
        expect(widget.lensFacing, LensFacing.backWide);
      });

      testWidgets('creates widget with front camera', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              lensFacing: LensFacing.front,
            ),
          ),
        );

        expect(find.byType(YOLOView), findsOneWidget);
        final yoloView = tester.widget<YOLOView>(find.byType(YOLOView));
        expect(yoloView.lensFacing, LensFacing.front);
      });

      testWidgets('creates widget with back camera', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              lensFacing: LensFacing.back,
            ),
          ),
        );

        expect(find.byType(YOLOView), findsOneWidget);
        final yoloView = tester.widget<YOLOView>(find.byType(YOLOView));
        expect(yoloView.lensFacing, LensFacing.back);
      });

      testWidgets('creates widget with wide back camera preference', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              lensFacing: LensFacing.backWide,
            ),
          ),
        );

        expect(find.byType(YOLOView), findsOneWidget);
        final yoloView = tester.widget<YOLOView>(find.byType(YOLOView));
        expect(yoloView.lensFacing, LensFacing.backWide);
      });

      testWidgets('lensFacing parameter does not force widget recreation', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              lensFacing: LensFacing.back,
            ),
          ),
        );

        final widget1 = tester.widget<YOLOView>(find.byType(YOLOView));
        expect(widget1.lensFacing, LensFacing.back);

        await tester.pumpWidget(
          const MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.detect,
              lensFacing: LensFacing.front,
            ),
          ),
        );

        final widget2 = tester.widget<YOLOView>(find.byType(YOLOView));
        expect(widget2.lensFacing, LensFacing.front);
      });

      testWidgets('handles lensFacing with other parameters', (
        WidgetTester tester,
      ) async {
        final controller = YOLOViewController();
        await tester.pumpWidget(
          MaterialApp(
            home: YOLOView(
              modelPath: 'test_model.tflite',
              task: YOLOTask.segment,
              controller: controller,
              lensFacing: LensFacing.front,
              confidenceThreshold: 0.7,
              iouThreshold: 0.5,
            ),
          ),
        );

        expect(find.byType(YOLOView), findsOneWidget);
        final yoloView = tester.widget<YOLOView>(find.byType(YOLOView));
        expect(yoloView.lensFacing, LensFacing.front);
        expect(yoloView.confidenceThreshold, 0.7);
        expect(yoloView.iouThreshold, 0.5);
      });
    });
  });
}
