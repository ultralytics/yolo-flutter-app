// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/core/yolo_model_resolver.dart';
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

class BareYOLOPlatform extends YOLOPlatform {}

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

    test('official models are available per task', () {
      const expected = {
        YOLOTask.detect: [
          'yolo26n',
          'yolo26s',
          'yolo26m',
          'yolo26l',
          'yolo26x',
        ],
        YOLOTask.segment: [
          'yolo26n-seg',
          'yolo26s-seg',
          'yolo26m-seg',
          'yolo26l-seg',
          'yolo26x-seg',
        ],
        YOLOTask.semantic: [
          'yolo26n-sem',
          'yolo26s-sem',
          'yolo26m-sem',
          'yolo26l-sem',
          'yolo26x-sem',
        ],
        YOLOTask.classify: [
          'yolo26n-cls',
          'yolo26s-cls',
          'yolo26m-cls',
          'yolo26l-cls',
          'yolo26x-cls',
        ],
        YOLOTask.pose: [
          'yolo26n-pose',
          'yolo26s-pose',
          'yolo26m-pose',
          'yolo26l-pose',
          'yolo26x-pose',
        ],
        YOLOTask.obb: [
          'yolo26n-obb',
          'yolo26s-obb',
          'yolo26m-obb',
          'yolo26l-obb',
          'yolo26x-obb',
        ],
      };

      for (final entry in expected.entries) {
        final yolo26Models = YOLO
            .officialModels(task: entry.key)
            .where((id) => id.startsWith('yolo26'))
            .toList(growable: false);
        expect(yolo26Models, entry.value);
      }
    });

    test('official YOLO26 Android and Apple URLs cover every task and size', () {
      const expected = [
        'yolo26n',
        'yolo26s',
        'yolo26m',
        'yolo26l',
        'yolo26x',
        'yolo26n-seg',
        'yolo26s-seg',
        'yolo26m-seg',
        'yolo26l-seg',
        'yolo26x-seg',
        'yolo26n-sem',
        'yolo26s-sem',
        'yolo26m-sem',
        'yolo26l-sem',
        'yolo26x-sem',
        'yolo26n-cls',
        'yolo26s-cls',
        'yolo26m-cls',
        'yolo26l-cls',
        'yolo26x-cls',
        'yolo26n-pose',
        'yolo26s-pose',
        'yolo26m-pose',
        'yolo26l-pose',
        'yolo26x-pose',
        'yolo26n-obb',
        'yolo26s-obb',
        'yolo26m-obb',
        'yolo26l-obb',
        'yolo26x-obb',
      ];

      for (final modelId in expected) {
        expect(
          YOLOModelResolver.officialModelDownloadUrlForTesting(
            modelId,
            iosLike: false,
          ),
          'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/${modelId}_int8.tflite',
        );
        expect(
          YOLOModelResolver.officialModelDownloadUrlForTesting(
            modelId,
            iosLike: true,
          ),
          'https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0/$modelId.mlpackage.zip',
        );
      }
    });

    test('default official model returns the first supported ID', () {
      expect(
        YOLO.defaultOfficialModel(task: YOLOTask.detect),
        YOLO.officialModels(task: YOLOTask.detect).first,
      );
    });

    test('task order follows Ultralytics docs navigation', () {
      expect(YOLOTask.values, [
        YOLOTask.detect,
        YOLOTask.segment,
        YOLOTask.semantic,
        YOLOTask.classify,
        YOLOTask.pose,
        YOLOTask.obb,
      ]);
    });

    test('task can be inferred from model metadata', () async {
      final yolo = YOLO(modelPath: 'test_model.tflite');
      await yolo.loadModel();

      expect(yolo.resolvedTask, YOLOTask.detect);
      expect(log.any((call) => call.method == 'inspectModel'), isTrue);
    });

    test('task mismatch from model metadata throws', () async {
      final setup = YOLOTestHelpers.createYOLOTestSetup(
        customResponses: {
          'inspectModel': (_) => {
            'path': 'test_model.tflite',
            'task': 'segment',
            'labels': ['person'],
          },
          'loadModel': (_) => true,
        },
      );
      channel = setup.$1;
      log = setup.$2;
      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);

      await expectLater(
        yolo.loadModel(),
        throwsA(isA<ModelLoadingException>()),
      );
    });

    test('missing task metadata requires explicit task', () async {
      final setup = YOLOTestHelpers.createYOLOTestSetup(
        customResponses: {
          'inspectModel': (_) => {'path': 'test_model.tflite', 'labels': []},
          'loadModel': (_) => true,
        },
      );
      channel = setup.$1;
      log = setup.$2;
      final yolo = YOLO(modelPath: 'test_model.tflite');

      await expectLater(
        yolo.loadModel(),
        throwsA(isA<ModelLoadingException>()),
      );
    });

    test(
      'single-image API resolves, loads, predicts, switches, and inspects',
      () async {
        final calls = <MethodCall>[];
        final setup = YOLOTestHelpers.createYOLOTestSetup(
          customResponses: {
            'inspectModel': (call) {
              calls.add(call);
              final path = (call.arguments as Map)['modelPath'] as String;
              return {
                'path': path,
                'task': path.contains('segment') ? 'segment' : 'detect',
                'labels': ['person'],
              };
            },
            'loadModel': (call) {
              calls.add(call);
              return true;
            },
            'predictorInstance': (call) {
              calls.add(call);
              return null;
            },
            'predictSingleImage': (call) {
              calls.add(call);
              return {
                'boxes': [
                  {
                    'x1': 10,
                    'y1': 20,
                    'x2': 30,
                    'y2': 40,
                    'x1_norm': 0.1,
                    'y1_norm': 0.2,
                    'x2_norm': 0.3,
                    'y2_norm': 0.4,
                    'class': 'person',
                    'confidence': 0.9,
                  },
                ],
              };
            },
            'setModel': (call) {
              calls.add(call);
              return true;
            },
            'disposeInstance': (call) {
              calls.add(call);
              return true;
            },
          },
        );
        channel = setup.$1;
        log = setup.$2;

        final yolo = YOLO(
          modelPath: 'detect_model.tflite',
          task: YOLOTask.detect,
          useGpu: false,
          classifierOptions: {'expectedClasses': 80},
          numItemsThreshold: 5,
        )..setViewId(42);

        expect(await yolo.loadModel(), isTrue);
        expect(yolo.isInitialized, isTrue);
        await yolo.predictorInstance();

        final prediction = await yolo.predict(
          Uint8List.fromList([1, 2, 3]),
          confidenceThreshold: 0.4,
          iouThreshold: 0.6,
        );
        expect(prediction['detections'], hasLength(1));
        expect(prediction['detections'].first['className'], 'person');

        final metadata = await YOLO.inspectModel('detect_model.tflite');
        expect(metadata['task'], 'detect');

        await yolo.switchModel('segment_model.tflite', YOLOTask.segment);
        expect(yolo.resolvedTask, YOLOTask.segment);
        await yolo.dispose();
        expect(yolo.isInitialized, isFalse);

        final loadArgs =
            calls.firstWhere((call) => call.method == 'loadModel').arguments
                as Map;
        expect(loadArgs['modelPath'], 'detect_model.tflite');
        expect(loadArgs['task'], 'detect');
        expect(loadArgs['useGpu'], isFalse);
        expect(loadArgs['classifierOptions'], {'expectedClasses': 80});
        expect(loadArgs['numItemsThreshold'], 5);

        final predictArgs =
            calls
                    .firstWhere((call) => call.method == 'predictSingleImage')
                    .arguments
                as Map;
        expect(predictArgs['confidenceThreshold'], 0.4);
        expect(predictArgs['iouThreshold'], 0.6);
        expect(predictArgs['image'], isA<Uint8List>());

        final switchArgs =
            calls.firstWhere((call) => call.method == 'setModel').arguments
                as Map;
        expect(switchArgs['viewId'], 42);
        expect(switchArgs['modelPath'], 'segment_model.tflite');
        expect(switchArgs['task'], 'segment');
      },
    );
  });

  group('Platform Method Channel', () {
    test('getPlatformVersion works', () async {
      final platform = YOLOMethodChannel();
      expect(platform, isNotNull);
    });

    test('platform interface works correctly', () {
      final mockPlatform = MockYOLOPlatform();
      YOLOPlatform.instance = mockPlatform;

      expect(mockPlatform, isNotNull);
      expect(YOLOPlatform.instance.getPlatformVersion(), completion('42'));
      expect(
        YOLOPlatform.instance.setModel(1, 'model.tflite', 'detect'),
        completes,
      );
    });

    test('base platform reports unimplemented operations', () {
      final platform = BareYOLOPlatform();

      expect(platform.getPlatformVersion, throwsUnimplementedError);
      expect(
        () => platform.setModel(1, 'model.tflite', 'detect'),
        throwsUnimplementedError,
      );
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

    test('ratings and copyWith classify real-time performance', () {
      final timestamp = DateTime.fromMillisecondsSinceEpoch(1234);
      final excellent = YOLOPerformanceMetrics(
        fps: 30,
        processingTimeMs: 40,
        frameNumber: 7,
        timestamp: timestamp,
      );
      final fair = excellent.copyWith(fps: 12, processingTimeMs: 140);
      final poor = excellent.copyWith(fps: 8, processingTimeMs: 250);

      expect(excellent.isGoodPerformance, isTrue);
      expect(excellent.hasPerformanceIssues, isFalse);
      expect(excellent.performanceRating, 'Excellent');
      expect(excellent.toString(), contains('fps: 30.0'));
      expect(fair.isGoodPerformance, isFalse);
      expect(fair.hasPerformanceIssues, isFalse);
      expect(fair.performanceRating, 'Fair');
      expect(poor.hasPerformanceIssues, isTrue);
      expect(poor.performanceRating, 'Poor');
      expect(poor.frameNumber, 7);
      expect(poor.timestamp, timestamp);
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

    test('preset constructors encode streaming tradeoffs', () {
      // Keep these runtime-constructed so LCOV records each public constructor.
      // ignore: prefer_const_constructors
      final minimal = YOLOStreamingConfig.minimal();
      // ignore: prefer_const_constructors
      final masks = YOLOStreamingConfig.withMasks();
      // ignore: prefer_const_constructors
      final poses = YOLOStreamingConfig.withPoses();
      // ignore: prefer_const_constructors
      final full = YOLOStreamingConfig.full();
      // ignore: prefer_const_constructors
      final debug = YOLOStreamingConfig.debug();
      final throttled = YOLOStreamingConfig.throttled(
        maxFPS: 12,
        includeOBB: true,
        skipFrames: 2,
      );
      final powerSaving = YOLOStreamingConfig.powerSaving(
        inferenceFrequency: 6,
        maxFPS: 9,
      );
      final highPerformance = YOLOStreamingConfig.highPerformance(
        inferenceFrequency: 45,
      );
      // ignore: prefer_const_constructors
      final custom = YOLOStreamingConfig.custom(includeOriginalImage: true);

      expect(minimal.includeMasks, isFalse);
      expect(masks.includeMasks, isTrue);
      expect(masks.includePoses, isFalse);
      expect(poses.includePoses, isTrue);
      expect(full.includeOBB, isTrue);
      expect(debug.includeOriginalImage, isTrue);
      expect(throttled.maxFPS, 12);
      expect(throttled.includeOBB, isTrue);
      expect(throttled.skipFrames, 2);
      expect(powerSaving.maxFPS, 9);
      expect(powerSaving.inferenceFrequency, 6);
      expect(highPerformance.inferenceFrequency, 45);
      expect(custom.includeDetections, isTrue);
      expect(custom.includeOriginalImage, isTrue);
      expect(debug.toString(), contains('originalImage: true'));
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

    test('ChannelConfig handles default instance ID correctly', () {
      // Test that 'default' instance ID does not append suffix
      final defaultChannel = ChannelConfig.createSingleImageChannel(
        instanceId: 'default',
      );
      expect(
        defaultChannel.name,
        'yolo_single_image_channel',
        reason: 'Default instance should use base channel name without suffix',
      );

      // Test that null instance ID also uses base channel name
      final nullChannel = ChannelConfig.createSingleImageChannel();
      expect(
        nullChannel.name,
        'yolo_single_image_channel',
        reason: 'Null instance should use base channel name without suffix',
      );

      // Test that custom instance ID appends suffix
      final customChannel = ChannelConfig.createSingleImageChannel(
        instanceId: 'custom_123',
      );
      expect(
        customChannel.name,
        'yolo_single_image_channel_custom_123',
        reason: 'Custom instance should append instance ID as suffix',
      );
    });

    test('ChannelConfig validates legacy method-call arguments', () {
      const validCall = MethodCall('loadModel', {
        'modelPath': 'model.tflite',
        'task': 'detect',
      });
      expect(
        () =>
            ChannelConfig.validateMethodCall(validCall, ['modelPath', 'task']),
        returnsNormally,
      );
      expect(
        () => ChannelConfig.validateMethodCall(
          const MethodCall('loadModel', 'not a map'),
          ['modelPath'],
        ),
        throwsArgumentError,
      );
      expect(
        () => ChannelConfig.validateMethodCall(validCall, ['missing']),
        throwsArgumentError,
      );

      expect(
        ChannelConfig.createStandardArgs(
          viewId: 1,
          modelPath: 'model.tflite',
          task: 'detect',
          additionalArgs: {'useGpu': false},
        ),
        {
          'viewId': 1,
          'modelPath': 'model.tflite',
          'task': 'detect',
          'useGpu': false,
        },
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

    test('ErrorHandler maps native and generic failures by context', () {
      final platformCases = {
        'INVALID_MODEL': isA<ModelLoadingException>(),
        'UNSUPPORTED_TASK': isA<ModelLoadingException>(),
        'MODEL_FILE_ERROR': isA<ModelLoadingException>(),
        'MODEL_NOT_LOADED': isA<ModelNotLoadedException>(),
        'INVALID_IMAGE': isA<InvalidInputException>(),
        'IMAGE_LOAD_ERROR': isA<InferenceException>(),
        'INFERENCE_ERROR': isA<InferenceException>(),
        'OTHER': isA<InferenceException>(),
      };

      for (final entry in platformCases.entries) {
        expect(
          YOLOErrorHandler.handlePlatformException(
            PlatformException(code: entry.key, message: 'message'),
            context: 'switch to model task pose',
          ),
          entry.value,
        );
      }

      final existing = InferenceException('already wrapped');
      expect(YOLOErrorHandler.handleGenericException(existing), same(existing));
      expect(
        YOLOErrorHandler.handleGenericException(
          MissingPluginException('missing'),
          context: 'load model',
        ),
        isA<ModelLoadingException>(),
      );
      expect(
        YOLOErrorHandler.handleGenericException(
          MissingPluginException('missing'),
          context: 'predict',
        ),
        isA<InferenceException>(),
      );
      expect(
        YOLOErrorHandler.handleError(
          PlatformException(code: 'INVALID_IMAGE', message: 'bad'),
          'predict',
        ),
        isA<InvalidInputException>(),
      );
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

    test(
      'YOLOModelResolver resolves metadata and normalizes official IDs',
      () async {
        final setup = YOLOTestHelpers.createYOLOTestSetup(
          customResponses: {
            'inspectModel': (_) => {
              'task': 'pose',
              'labels': ['person'],
            },
          },
        );
        channel = setup.$1;
        log = setup.$2;

        final resolved = await YOLOModelResolver.resolve(
          modelPath: 'custom-model.tflite',
        );

        expect(resolved.modelPath, 'custom-model.tflite');
        expect(resolved.task, YOLOTask.pose);
        expect(resolved.metadata['labels'], ['person']);
        expect(
          YOLOModelResolver.isOfficialModel('models/yolo26n.tflite'),
          isTrue,
        );
        expect(
          YOLOModelResolver.isOfficialModel('yolo26n.mlpackage.zip'),
          isTrue,
        );
        expect(
          YOLOModelResolver.isOfficialModel('not-a-model.tflite'),
          isFalse,
        );
        expect(
          YOLOModelResolver.officialModelDownloadUrlForTesting(
            'missing',
            iosLike: false,
          ),
          isNull,
        );
      },
    );
  });
}
