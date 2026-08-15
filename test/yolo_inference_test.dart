// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/core/yolo_inference.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';

import 'utils/test_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOInference', () {
    late MethodChannel mockChannel;
    late List<MethodCall> log;

    setUp(() {
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      mockChannel = setup.$1;
      log = setup.$2;
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(mockChannel, null);
      log.clear();
    });

    test('constructor initializes correctly', () {
      final inference = YOLOInference(
        channel: mockChannel,
        instanceId: 'test_instance',
        task: YOLOTask.detect,
      );

      expect(inference, isNotNull);
    });

    test('predict with valid image data', () async {
      final inference = YOLOInference(
        channel: mockChannel,
        instanceId: 'test_instance',
        task: YOLOTask.detect,
      );

      final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      final result = await inference.predict(imageBytes);

      expect(result, isA<Map<String, dynamic>>());
      expect(result['detections'], [containsPair('className', 'person')]);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'predictSingleImage',
        arguments: {'image': imageBytes, 'instanceId': 'test_instance'},
      );
    });

    test('predict with confidence and IoU thresholds', () async {
      final inference = YOLOInference(
        channel: mockChannel,
        instanceId: 'test_instance',
        task: YOLOTask.detect,
      );

      final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      await inference.predict(
        imageBytes,
        confidenceThreshold: 0.7,
        iouThreshold: 0.5,
      );

      YOLOTestHelpers.assertMethodCalled(
        log,
        'predictSingleImage',
        arguments: {
          'image': imageBytes,
          'confidenceThreshold': 0.7,
          'iouThreshold': 0.5,
          'instanceId': 'test_instance',
        },
      );
    });

    test('predict throws InvalidInputException for empty image', () async {
      final inference = YOLOInference(
        channel: mockChannel,
        instanceId: 'test_instance',
        task: YOLOTask.detect,
      );

      expect(
        () => inference.predict(Uint8List(0)),
        throwsA(isA<InvalidInputException>()),
      );
    });

    test(
      'predict throws InvalidInputException for invalid confidence threshold',
      () async {
        final inference = YOLOInference(
          channel: mockChannel,
          instanceId: 'test_instance',
          task: YOLOTask.detect,
        );

        final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

        expect(
          () => inference.predict(imageBytes, confidenceThreshold: 1.5),
          throwsA(isA<InvalidInputException>()),
        );

        expect(
          () => inference.predict(imageBytes, confidenceThreshold: -0.1),
          throwsA(isA<InvalidInputException>()),
        );
      },
    );

    test(
      'predict throws InvalidInputException for invalid IoU threshold',
      () async {
        final inference = YOLOInference(
          channel: mockChannel,
          instanceId: 'test_instance',
          task: YOLOTask.detect,
        );

        final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

        expect(
          () => inference.predict(imageBytes, iouThreshold: 1.5),
          throwsA(isA<InvalidInputException>()),
        );

        expect(
          () => inference.predict(imageBytes, iouThreshold: -0.1),
          throwsA(isA<InvalidInputException>()),
        );
      },
    );

    test('predict handles platform exceptions', () async {
      final errorChannel = YOLOTestHelpers.setupMockChannel(
        customResponses: {
          'predictSingleImage': (_) => throw PlatformException(
            code: 'INFERENCE_ERROR',
            message: 'Inference failed',
          ),
        },
      );

      final inference = YOLOInference(
        channel: errorChannel,
        instanceId: 'test_instance',
        task: YOLOTask.detect,
      );

      final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);

      expect(
        () => inference.predict(imageBytes),
        throwsA(isA<YOLOException>()),
      );
    });

    test('predict processes task-specific results', () async {
      final box = {
        'class': 'person',
        'confidence': 0.9,
        'x1': 10.0,
        'y1': 20.0,
        'x2': 30.0,
        'y2': 40.0,
      };
      final responses = <YOLOTask, Map<String, dynamic>>{
        YOLOTask.segment: {
          'boxes': [box],
          'masks': [
            [
              [0.1, 0.2],
              [0.3, 0.4],
            ],
          ],
        },
        YOLOTask.semantic: {
          'semanticMask': {
            'classMap': [0, 1],
            'width': 2,
            'height': 1,
          },
        },
        YOLOTask.depth: {
          'depthMap': {
            'values': [1.0, 2.0],
            'width': 2,
            'height': 1,
            'minDepth': 1.0,
            'maxDepth': 2.0,
          },
        },
        YOLOTask.classify: {
          'classification': {'class': 2, 'name': 'bird', 'confidence': 0.8},
        },
        YOLOTask.pose: {
          'boxes': [box],
          'keypoints': [
            {
              'coordinates': [
                {'x': 1, 'y': 2, 'confidence': 0.7},
                {'x': 3, 'y': 4, 'confidence': 0.6},
              ],
            },
          ],
        },
      };
      final results = <YOLOTask, Map<String, dynamic>>{};

      for (final entry in responses.entries) {
        final inference = YOLOInference(
          channel: YOLOTestHelpers.setupMockChannel(
            customResponses: {'predictSingleImage': (_) => entry.value},
          ),
          instanceId: 'test_instance',
          task: entry.key,
        );

        results[entry.key] = await inference.predict(
          Uint8List.fromList([1, 2, 3]),
        );
      }

      expect(
        results[YOLOTask.segment]!['detections'],
        contains(
          containsPair('mask', [
            [0.1, 0.2],
            [0.3, 0.4],
          ]),
        ),
      );
      expect(
        results[YOLOTask.classify]!['detections'],
        contains(containsPair('className', 'bird')),
      );
      expect(
        results[YOLOTask.pose]!['detections'],
        contains(containsPair('keypoints', [1.0, 2.0, 0.7, 3.0, 4.0, 0.6])),
      );
      for (final task in [YOLOTask.semantic, YOLOTask.depth]) {
        final payloadKey = responses[task]!.keys.single;
        expect(results[task]!['detections'], isEmpty);
        expect(results[task]![payloadKey], responses[task]![payloadKey]);
      }
    });

    test('predict rejects invalid platform results', () {
      final inference = YOLOInference(
        channel: YOLOTestHelpers.setupMockChannel(
          customResponses: {'predictSingleImage': (_) => 'invalid'},
        ),
        instanceId: 'test_instance',
        task: YOLOTask.detect,
      );

      expect(
        () => inference.predict(Uint8List.fromList([1])),
        throwsA(isA<InferenceException>()),
      );
    });

    test('predict with default instance ID', () async {
      final inference = YOLOInference(
        channel: mockChannel,
        instanceId: 'default',
        task: YOLOTask.detect,
      );

      final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      await inference.predict(imageBytes);

      YOLOTestHelpers.assertMethodCalled(
        log,
        'predictSingleImage',
        arguments: {'image': imageBytes},
      );
    });

    test('predict includes OBB angle in detections', () async {
      final inference = YOLOInference(
        channel: YOLOTestHelpers.setupMockChannel(
          customResponses: {
            'predictSingleImage': (_) => {
              'obb': [
                {
                  'points': [
                    {'x': 0.1, 'y': 0.1},
                    {'x': 0.2, 'y': 0.1},
                    {'x': 0.2, 'y': 0.2},
                    {'x': 0.1, 'y': 0.2},
                  ],
                  'class': 'ship',
                  'confidence': 0.88,
                  'angle': 0.5235987756,
                },
              ],
            },
          },
        ),
        instanceId: 'test_instance',
        task: YOLOTask.obb,
      );

      final result = await inference.predict(Uint8List.fromList([1, 2, 3]));
      final first =
          (result['detections'] as List<dynamic>).first as Map<String, dynamic>;

      expect(first['className'], 'ship');
      expect(first['confidence'], 0.88);
      expect(first['angle'], closeTo(0.5235987756, 1e-9));
    });

    test('predict preserves OBB polygon points and class index', () async {
      final inference = YOLOInference(
        channel: YOLOTestHelpers.setupMockChannel(
          customResponses: {
            'predictSingleImage': (_) => {
              'obb': [
                {
                  'points': [
                    {'x': 0.1, 'y': 0.2},
                    {'x': 0.4, 'y': 0.2},
                    {'x': 0.4, 'y': 0.5},
                    {'x': 0.1, 'y': 0.5},
                  ],
                  'class': 'ship',
                  'classIndex': 3,
                  'confidence': 0.88,
                },
              ],
            },
          },
        ),
        instanceId: 'test_instance',
        task: YOLOTask.obb,
      );

      final result = await inference.predict(Uint8List.fromList([1, 2, 3]));
      final first =
          (result['detections'] as List<dynamic>).first as Map<String, dynamic>;

      expect(first['classIndex'], 3);
      expect(first['className'], 'ship');
      expect(first['polygon'], [
        {'x': 0.1, 'y': 0.2},
        {'x': 0.4, 'y': 0.2},
        {'x': 0.4, 'y': 0.5},
        {'x': 0.1, 'y': 0.5},
      ]);
      expect(first['boundingBox'], {
        'left': 0.1,
        'top': 0.2,
        'right': 0.4,
        'bottom': 0.5,
      });
    });
  });
}
