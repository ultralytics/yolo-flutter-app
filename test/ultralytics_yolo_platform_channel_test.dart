import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_channel.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('PlatformChannelUltralyticsYolo - Model Loading', () {
    late PlatformChannelUltralyticsYolo platform;
    late List<MethodCall> log;

    setUp(() {
      platform = PlatformChannelUltralyticsYolo();
      log = <MethodCall>[];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        platform.methodChannel,
        (MethodCall methodCall) async {
          log.add(methodCall);
          switch (methodCall.method) {
            case 'loadModel':
              return 'success';
            case 'detectImage':
              return [
                {
                  'label': 'person',
                  'confidence': 0.95,
                  'x': 0.1,
                  'y': 0.2,
                  'width': 0.3,
                  'height': 0.4,
                  'index': 0,
                }
              ];
            case 'classifyImage':
              return [
                {
                  'label': 'person',
                  'confidence': 0.95,
                  'index': 0,
                }
              ];
            case 'segmentImage':
              return [
                {
                  'label': 'person',
                  'confidence': 0.95,
                  'x': 0.1,
                  'y': 0.2,
                  'width': 0.3,
                  'height': 0.4,
                  'index': 0,
                  'polygons': [
                    [
                      [0.1, 0.2],
                      [0.3, 0.2],
                      [0.3, 0.4],
                      [0.1, 0.4],
                    ]
                  ],
                }
              ];
            default:
              return 'success';
          }
        },
      );
    });

    test('should load model with correct parameters', () async {
      final result = await platform.loadModel(
        {
          'modelPath': 'test_model.tflite',
          'metadataPath': 'test_metadata.yaml',
        },
        useGpu: true,
      );

      expect(log, hasLength(1));
      expect(log.first.method, equals('loadModel'));
      expect(log.first.arguments, {
        'model': {
          'modelPath': 'test_model.tflite',
          'metadataPath': 'test_metadata.yaml',
        },
        'useGpu': true,
      });
      expect(result, equals('success'));
    });

    test('should set confidence threshold', () async {
      final result = await platform.setConfidenceThreshold(0.5);

      expect(log, hasLength(1));
      expect(log.first.method, equals('setConfidenceThreshold'));
      expect(log.first.arguments, {'confidence': 0.5});
      expect(result, equals('success'));
    });

    test('should set IOU threshold', () async {
      final result = await platform.setIouThreshold(0.5);

      expect(log, hasLength(1));
      expect(log.first.method, equals('setIouThreshold'));
      expect(log.first.arguments, {'iou': 0.5});
      expect(result, equals('success'));
    });

    test('should set number of items threshold', () async {
      final result = await platform.setNumItemsThreshold(10);

      expect(log, hasLength(1));
      expect(log.first.method, equals('setNumItemsThreshold'));
      expect(log.first.arguments, {'numItems': 10});
      expect(result, equals('success'));
    });

    test('should set zoom ratio', () async {
      final result = await platform.setZoomRatio(1.5);

      expect(log, hasLength(1));
      expect(log.first.method, equals('setZoomRatio'));
      expect(log.first.arguments, {'ratio': 1.5});
      expect(result, equals('success'));
    });

    test('should set lens direction', () async {
      final result = await platform.setLensDirection(1);

      expect(log, hasLength(1));
      expect(log.first.method, equals('setLensDirection'));
      expect(log.first.arguments, {'direction': 1});
      expect(result, equals('success'));
    });

    test('should control camera', () async {
      final startResult = await platform.startCamera();
      expect(startResult, equals('success'));

      final pauseResult = await platform.pauseLivePrediction();
      expect(pauseResult, equals('success'));

      final resumeResult = await platform.resumeLivePrediction();
      expect(resumeResult, equals('success'));

      final closeResult = await platform.closeCamera();
      expect(closeResult, equals('success'));
    });
  });

  group('PlatformChannelUltralyticsYolo - Image Processing', () {
    late PlatformChannelUltralyticsYolo platform;
    late List<MethodCall> log;

    setUp(() {
      platform = PlatformChannelUltralyticsYolo();
      log = <MethodCall>[];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        platform.methodChannel,
        (MethodCall methodCall) async {
          log.add(methodCall);
          switch (methodCall.method) {
            case 'detectImage':
              return [
                {
                  'label': 'person',
                  'confidence': 0.95,
                  'x': 0.1,
                  'y': 0.2,
                  'width': 0.3,
                  'height': 0.4,
                  'index': 0,
                }
              ];
            case 'classifyImage':
              return [
                {
                  'label': 'person',
                  'confidence': 0.95,
                  'index': 0,
                }
              ];
            case 'segmentImage':
              return [
                {
                  'label': 'person',
                  'confidence': 0.95,
                  'x': 0.1,
                  'y': 0.2,
                  'width': 0.3,
                  'height': 0.4,
                  'index': 0,
                  'polygons': [
                    [
                      [0.1, 0.2],
                      [0.3, 0.2],
                      [0.3, 0.4],
                      [0.1, 0.4],
                    ]
                  ],
                }
              ];
            default:
              return null;
          }
        },
      );
    });

    test('should detect objects in image', () async {
      final result = await platform.detectImage('test_image.jpg');

      expect(log, hasLength(1));
      expect(log.first.method, equals('detectImage'));
      expect(log.first.arguments, {
        'imagePath': 'test_image.jpg',
      });
      expect(result, isList);
      expect(result?[0]?.label, equals('person'));
      expect(result?[0]?.confidence, equals(0.95));
      expect(result?[0]?.index, equals(0));
    });

    test('should classify image', () async {
      final result = await platform.classifyImage('test_image.jpg');

      expect(log, hasLength(1));
      expect(log.first.method, equals('classifyImage'));
      expect(log.first.arguments, {
        'imagePath': 'test_image.jpg',
      });
      expect(result, isList);
      expect(result?[0]?.label, equals('person'));
      expect(result?[0]?.confidence, equals(0.95));
      expect(result?[0]?.index, equals(0));
    });

    test('should segment image', () async {
      final result = await platform.segmentImage('test_image.jpg');

      expect(log, hasLength(1));
      expect(log.first.method, equals('segmentImage'));
      expect(log.first.arguments, {
        'imagePath': 'test_image.jpg',
      });
      expect(result, isList);
      expect(result?[0]?.label, equals('person'));
      expect(result?[0]?.confidence, equals(0.95));
      expect(result?[0]?.index, equals(0));
    });
  });

  group('PlatformChannelUltralyticsYolo - Event Channels', () {
    late PlatformChannelUltralyticsYolo platform;
    late List<MethodCall> log;

    setUp(() {
      platform = PlatformChannelUltralyticsYolo();
      log = <MethodCall>[];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        platform.methodChannel,
        (MethodCall methodCall) async {
          log.add(methodCall);
          return null;
        },
      );
    });

    test('should handle detection result stream', () async {
      final stream = platform.detectionResultStream;
      expect(stream, isNotNull);
    });

    test('should handle segment result stream', () async {
      final stream = platform.segmentResultStream;
      expect(stream, isNotNull);
    });

    test('should handle classification result stream', () async {
      final stream = platform.classificationResultStream;
      expect(stream, isNotNull);
    });

    test('should handle inference time stream', () async {
      final stream = platform.inferenceTimeStream;
      expect(stream, isNotNull);
    });

    test('should handle FPS rate stream', () async {
      final stream = platform.fpsRateStream;
      expect(stream, isNotNull);
    });
  });
}
