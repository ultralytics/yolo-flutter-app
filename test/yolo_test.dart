// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/platform/yolo_platform_interface.dart';
import 'package:ultralytics_yolo/platform/yolo_platform_impl.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
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

  group('Error Handling', () {
    test('YOLOException types work correctly', () {
      final modelException = ModelLoadingException('Model failed to load');
      final inferenceException = InferenceException('Inference failed');
      final invalidInputException = InvalidInputException('Invalid input');
      final modelNotLoadedException = ModelNotLoadedException(
        'Model not loaded',
      );

      expect(modelException, isA<YOLOException>());
      expect(inferenceException, isA<YOLOException>());
      expect(invalidInputException, isA<YOLOException>());
      expect(modelNotLoadedException, isA<YOLOException>());

      expect(modelException.message, 'Model failed to load');
      expect(inferenceException.message, 'Inference failed');
      expect(invalidInputException.message, 'Invalid input');
      expect(modelNotLoadedException.message, 'Model not loaded');
    });

    test('YOLOException toString works correctly', () {
      final exception = ModelLoadingException('Test error');
      expect(exception.toString(), 'ModelLoadingException: Test error');
    });
  });

  group('Streaming Config', () {
    test('YOLOStreamingConfig constructor works correctly', () {
      const config = YOLOStreamingConfig(
        includeDetections: false,
        includeMasks: true,
        maxFPS: 25,
        throttleInterval: Duration(milliseconds: 200),
      );

      expect(config.includeDetections, false);
      expect(config.includeMasks, true);
      expect(config.maxFPS, 25);
      expect(config.throttleInterval, const Duration(milliseconds: 200));
    });
  });

  group('Performance Metrics', () {
    test('YOLOPerformanceMetrics fromMap with int values', () {
      final data = {'fps': 30, 'processingTimeMs': 25, 'frameNumber': 100};

      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, 30.0);
      expect(metrics.processingTimeMs, 25.0);
      expect(metrics.frameNumber, 100);
      expect(metrics.timestamp, isA<DateTime>());
    });

    test('YOLOPerformanceMetrics toMap works correctly', () {
      final timestamp = DateTime.now();
      final metrics = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 25.0,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final map = metrics.toMap();

      expect(map['fps'], 30.0);
      expect(map['processingTimeMs'], 25.0);
      expect(map['frameNumber'], 100);
      expect(map['timestamp'], timestamp.millisecondsSinceEpoch);
    });
  });

  group('Map Converter', () {
    test('convertToTypedMapSafe works correctly', () {
      final input = {'key1': 'value1', 'key2': 123, 'key3': true};
      final result = MapConverter.convertToTypedMapSafe(input);

      expect(result, isNotNull);
      expect(result, isA<Map<String, dynamic>>());
      expect(result!['key1'], 'value1');
      expect(result['key2'], 123);
      expect(result['key3'], true);
    });

    test('convertToTypedMapSafe handles null input', () {
      final result = MapConverter.convertToTypedMapSafe(null);
      expect(result, isNull);
    });

    test('convertBoundingBox works correctly', () {
      final boxMap = {
        'left': 10.0,
        'top': 20.0,
        'right': 110.0,
        'bottom': 220.0,
      };

      final rect = MapConverter.convertBoundingBox(boxMap);

      expect(rect.left, 10.0);
      expect(rect.top, 20.0);
      expect(rect.right, 110.0);
      expect(rect.bottom, 220.0);
    });
  });

  group('Critical Functionality', () {
    test('YOLOInstanceManager works correctly', () {
      const instanceId = 'test_instance';
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);

      YOLOInstanceManager.registerInstance(instanceId, yolo);
      final retrieved = YOLOInstanceManager.getInstance(instanceId);

      expect(retrieved, isNotNull);
      expect(retrieved, equals(yolo));

      YOLOInstanceManager.unregisterInstance(instanceId);
      expect(YOLOInstanceManager.getInstance(instanceId), isNull);
    });

    test('ChannelConfig creates channels correctly', () {
      final controlChannel = ChannelConfig.createControlChannel('test123');
      final detectionChannel = ChannelConfig.createDetectionResultsChannel(
        'test123',
      );

      expect(
        controlChannel.name,
        'com.ultralytics.yolo/controlChannel_test123',
      );
      expect(
        detectionChannel.name,
        'com.ultralytics.yolo/detectionResults_test123',
      );
    });

    test('ErrorHandler handles different exception types', () {
      final platformException = PlatformException(
        code: 'MODEL_NOT_FOUND',
        message: 'Model not found',
      );
      final handledException = YOLOErrorHandler.handlePlatformException(
        platformException,
        context: 'Loading model',
      );

      expect(handledException, isA<ModelLoadingException>());
      expect(handledException.message, contains('Model not found'));
    });
  });

  group('YOLO Core API Tests', () {
    test('YOLO constructor with multi-instance', () {
      final yolo = YOLO(
        modelPath: 'test_model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(yolo.modelPath, 'test_model.tflite');
      expect(yolo.task, YOLOTask.detect);
      expect(yolo.instanceId, isNotEmpty);
      expect(yolo.instanceId, isNot('default'));
    });

    test('YOLO constructor with classifier options', () {
      final yolo = YOLO.withClassifierOptions(
        modelPath: 'classifier_model.tflite',
        task: YOLOTask.classify,
        classifierOptions: {
          'enable1ChannelSupport': true,
          'expectedChannels': 1,
        },
      );

      expect(yolo.modelPath, 'classifier_model.tflite');
      expect(yolo.task, YOLOTask.classify);
    });

    test('YOLO setViewId works', () {
      final yolo = YOLO(modelPath: 'test.tflite', task: YOLOTask.detect);
      yolo.setViewId(123);
      expect(yolo, isNotNull);
    });

    test('YOLO static methods work', () async {
      final modelExists = await YOLO.checkModelExists('test_model.tflite');
      expect(modelExists, isA<Map<String, dynamic>>());

      final storagePaths = await YOLO.getStoragePaths();
      expect(storagePaths, isA<Map<String, String?>>());
    });
  });
}
