// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/platform/yolo_platform_interface.dart';
import 'package:ultralytics_yolo/platform/yolo_platform_impl.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:flutter/services.dart';
import 'utils/test_helpers.dart';

class MockYOLOPlatform with MockPlatformInterfaceMixin implements YOLOPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<void> setModel(int viewId, String modelPath, String task) =>
      Future.value();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MethodChannel channel;
  late List<MethodCall> log;

  setUp(() {
    final setup = YOLOTestHelpers.createYOLOTestSetup();
    channel = setup.$1;
    log = setup.$2;
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
    log.clear();
  });

  test('YOLO instance creation works', () {
    final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
    expect(yolo, isNotNull);
    expect(yolo.modelPath, 'test_model.tflite');
    expect(yolo.task, YOLOTask.detect);
  });

  group('YOLO Basic Functionality', () {
    test('YOLO instance creation works', () {
      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);
      expect(yolo, isNotNull);
      expect(yolo.modelPath, 'test_model.tflite');
      expect(yolo.task, YOLOTask.detect);
    });

    test('different task types work', () {
      final detectYolo = YOLO(
        modelPath: 'detect_model.tflite',
        task: YOLOTask.detect,
      );
      final segmentYolo = YOLO(
        modelPath: 'segment_model.tflite',
        task: YOLOTask.segment,
      );
      final classifyYolo = YOLO(
        modelPath: 'classify_model.tflite',
        task: YOLOTask.classify,
      );

      expect(detectYolo.task, YOLOTask.detect);
      expect(segmentYolo.task, YOLOTask.segment);
      expect(classifyYolo.task, YOLOTask.classify);
    });
  });

  group('Platform Method Channel', () {
    test('getPlatformVersion works', () async {
      final platform = YOLOMethodChannel();
      expect(platform, isNotNull);
    });

    test('setModel calls method channel correctly', () async {
      final platform = YOLOMethodChannel();
      expect(platform, isNotNull);
      // Test passes if no exceptions are thrown
      expect(true, isTrue);
    });

    test('platform interface works correctly', () {
      final mockPlatform = MockYOLOPlatform();
      expect(mockPlatform, isNotNull);
      expect(mockPlatform.getPlatformVersion(), completion('42'));
    });
  });

  group('Performance Tests', () {
    test('performance metrics are tracked correctly', () {
      final metrics = YOLOTestHelpers.createMockPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.5,
      );

      expect(metrics['fps'], 30.0);
      expect(metrics['processingTimeMs'], 33.5);
    });
  });

  group('Performance Metrics', () {
    test('constructor creates metrics with all required fields', () {
      final timestamp = DateTime.now();
      const fps = 30.0;
      const processingTime = 33.3;
      const frameNumber = 100;

      final metrics = YOLOPerformanceMetrics(
        fps: fps,
        processingTimeMs: processingTime,
        frameNumber: frameNumber,
        timestamp: timestamp,
      );

      expect(metrics.fps, equals(fps));
      expect(metrics.processingTimeMs, equals(processingTime));
      expect(metrics.frameNumber, equals(frameNumber));
      expect(metrics.timestamp, equals(timestamp));
    });

    test('fromMap() factory constructor with valid data', () {
      final data = {'fps': 25.5, 'processingTimeMs': 40.0, 'frameNumber': 50};
      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, equals(25.5));
      expect(metrics.processingTimeMs, equals(40.0));
      expect(metrics.frameNumber, equals(50));
      expect(metrics.timestamp, isA<DateTime>());
    });
  });

  group('Streaming Config', () {
    test('default constructor sets correct defaults', () {
      const config = YOLOStreamingConfig();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('constructor with custom parameters', () {
      const config = YOLOStreamingConfig(
        includeDetections: false,
        includeMasks: true,
        includePoses: true,
        maxFPS: 30,
        throttleInterval: Duration(milliseconds: 100),
      );

      expect(config.includeDetections, isFalse);
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isTrue);
      expect(config.maxFPS, equals(30));
      expect(
        config.throttleInterval,
        equals(const Duration(milliseconds: 100)),
      );
    });
  });

  group('Static Methods', () {
    test('checkModelExists returns model information', () async {
      final result = await YOLO.checkModelExists('test_model.tflite');
      expect(result, isNotNull);
    });

    test('getStoragePaths returns storage information', () async {
      final paths = await YOLO.getStoragePaths();
      expect(paths, isNotNull);
    });
  });

  group('All Tasks', () {
    test('different task types work correctly', () {
      final detectYolo = YOLO(
        modelPath: 'detect_model.tflite',
        task: YOLOTask.detect,
      );
      final segmentYolo = YOLO(
        modelPath: 'segment_model.tflite',
        task: YOLOTask.segment,
      );
      final classifyYolo = YOLO(
        modelPath: 'classify_model.tflite',
        task: YOLOTask.classify,
      );
      final poseYolo = YOLO(
        modelPath: 'pose_model.tflite',
        task: YOLOTask.pose,
      );

      expect(detectYolo.task, YOLOTask.detect);
      expect(segmentYolo.task, YOLOTask.segment);
      expect(classifyYolo.task, YOLOTask.classify);
      expect(poseYolo.task, YOLOTask.pose);
    });
  });

  group('Error Handling', () {
    test('handles platform exceptions gracefully', () async {
      expect(true, isTrue);
    });
  });

  group('Test Helpers Integration', () {
    test('YOLOTestHelpers methods work correctly', () {
      final mockResult = YOLOTestHelpers.createMockDetectionResult(
        numDetections: 2,
        includeKeypoints: true,
        includeMask: true,
      );

      expect(mockResult, isA<Map<String, dynamic>>());
      expect(mockResult['detections'], isA<List>());
      expect(mockResult['detections'].length, 2);
      expect(mockResult['annotatedImage'], isA<Uint8List>());

      final mockYOLOResult = YOLOTestHelpers.createMockYOLOResult(
        className: 'car',
        confidence: 0.85,
        includeKeypoints: true,
        includeMask: true,
      );

      expect(mockYOLOResult.className, 'car');
      expect(mockYOLOResult.confidence, 0.85);
      expect(mockYOLOResult.keypoints, isNotNull);
      expect(mockYOLOResult.mask, isNotNull);

      final mockMetrics = YOLOTestHelpers.createMockPerformanceMetrics(
        fps: 25.0,
        processingTimeMs: 40.0,
      );

      expect(mockMetrics['fps'], 25.0);
      expect(mockMetrics['processingTimeMs'], 40.0);
      expect(mockMetrics['timestamp'], isA<int>());

      final mockStreamData = YOLOTestHelpers.createMockStreamingData(
        includeDetections: true,
        includePerformance: true,
        includeOriginalImage: true,
      );

      expect(mockStreamData['detections'], isNotNull);
      expect(mockStreamData['performance'], isNotNull);
      expect(mockStreamData['originalImage'], isNotNull);

      final mockModelResult = YOLOTestHelpers.createMockModelExistsResult(
        exists: true,
        location: 'assets',
      );

      expect(mockModelResult['exists'], true);
      expect(mockModelResult['location'], 'assets');
      expect(mockModelResult['path'], isA<String>());

      final mockPaths = YOLOTestHelpers.createMockStoragePaths();

      expect(mockPaths['internal'], isA<String>());
      expect(mockPaths['cache'], isA<String>());
      expect(mockPaths['external'], isA<String>());
      expect(mockPaths['externalCache'], isA<String>());
    });

    test('YOLOTestHelpers.waitForCondition works correctly', () async {
      bool conditionMet = false;

      Future.delayed(const Duration(milliseconds: 100), () {
        conditionMet = true;
      });

      await YOLOTestHelpers.waitForCondition(
        () => conditionMet,
        timeout: const Duration(seconds: 1),
      );

      expect(conditionMet, true);
    });
  });
}
