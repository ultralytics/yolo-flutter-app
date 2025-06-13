// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Task-Specific Features', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('predict handles mask data correctly for segmentation', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                'boxes': [
                  {
                    'class': 'person',
                    'confidence': 0.95,
                    'x1': 10.0,
                    'y1': 20.0,
                    'x2': 110.0,
                    'y2': 220.0,
                    'x1_norm': 0.1,
                    'y1_norm': 0.1,
                    'x2_norm': 0.9,
                    'y2_norm': 0.9,
                  },
                ],
                'masks': [
                  [
                    [0.0, 0.5, 1.0],
                    [0.2, 0.8, 0.3],
                    [1.0, 0.5, 0.0],
                  ],
                ],
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.segment);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      expect(result['boxes'], hasLength(1));
      expect(result['detections'], hasLength(1));
      final detection = result['detections'][0] as Map<String, dynamic>;
      expect(detection['mask'], isNotNull);
      expect(detection['mask'].length, 3);
      expect(detection['mask'][0].length, 3);
    });

    test(
      'predict handles keypoints data correctly for pose estimation',
      () async {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
              if (methodCall.method == 'loadModel') {
                return true;
              } else if (methodCall.method == 'predictSingleImage') {
                return {
                  'boxes': [
                    {
                      'class': 'person',
                      'confidence': 0.95,
                      'x1': 10.0,
                      'y1': 20.0,
                      'x2': 110.0,
                      'y2': 220.0,
                      'x1_norm': 0.1,
                      'y1_norm': 0.1,
                      'x2_norm': 0.9,
                      'y2_norm': 0.9,
                    },
                  ],
                  'keypoints': [
                    {
                      'coordinates': [
                        {'x': 100.0, 'y': 50.0, 'confidence': 0.9}, // nose
                        {'x': 95.0, 'y': 55.0, 'confidence': 0.85}, // left eye
                        {
                          'x': 105.0,
                          'y': 55.0,
                          'confidence': 0.87,
                        }, // right eye
                      ],
                    },
                  ],
                };
              }
              return null;
            });

        final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.pose);
        await yolo.loadModel();

        final image = Uint8List.fromList([1, 2, 3]);
        final result = await yolo.predict(image);

        expect(result['boxes'], hasLength(1));
        expect(result['detections'], hasLength(1));
        final detection = result['detections'][0] as Map<String, dynamic>;
        expect(detection['keypoints'], isNotNull);
        final keypoints = detection['keypoints'] as List<double>;
        expect(keypoints.length, 9); // 3 keypoints * 3 values each
      },
    );

    test('handleResult correctly processes null detections list', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                'boxes': [],
                // No detections key
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      expect(result['detections'], []);
    });

    test('loadModel handles MODEL_FILE_ERROR correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'MODEL_FILE_ERROR',
                message: 'Corrupted model file',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to load model: Corrupted model file'),
          ),
        ),
      );
    });

    test('predict handles image load error correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'IMAGE_LOAD_ERROR',
                message: 'Failed to decode image',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<InferenceException>().having(
            (e) => e.message,
            'message',
            contains('Platform error during inference: Failed to decode image'),
          ),
        ),
      );
    });

    test('predict handles inference error correctly', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              throw PlatformException(
                code: 'INFERENCE_ERROR',
                message: 'GPU out of memory',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);

      expect(
        () => yolo.predict(image),
        throwsA(
          isA<InferenceException>().having(
            (e) => e.message,
            'message',
            contains('Error during inference: GPU out of memory'),
          ),
        ),
      );
    });

    test('predict handles result with performance metrics', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                'boxes': [],
                'detections': [],
                'processingTimeMs': 33.5,
                'fps': 29.8,
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      expect(result['processingTimeMs'], 33.5);
      expect(result['fps'], 29.8);
    });

    test('switchModel without viewId throws exception', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);

      expect(
        () => yolo.switchModel('new.tflite', YOLOTask.detect),
        throwsStateError,
      );
    });
  });

  group('YOLOResult toString Coverage', () {
    test('YOLOResult toString formats correctly', () {
      final result = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.95,
        boundingBox: const Rect.fromLTWH(10, 20, 100, 200),
        normalizedBox: const Rect.fromLTWH(0.1, 0.1, 0.8, 0.8),
        keypoints: [Point(50, 60), Point(55, 65)],
        mask: [
          [0.0, 1.0],
          [1.0, 0.0],
        ],
      );

      final str = result.toString();
      expect(
        str,
        'YOLOResult{classIndex: 0, className: person, confidence: 0.95, boundingBox: Rect.fromLTRB(10.0, 20.0, 110.0, 220.0)}',
      );
    });

    test('Point toString formats correctly', () {
      final point = Point(123.45, 678.90);
      expect(point.toString(), 'Point(123.45, 678.9)');
    });

    test('YOLOResult toString with different values', () {
      final result = YOLOResult(
        classIndex: 2,
        className: 'car',
        confidence: 0.87,
        boundingBox: const Rect.fromLTWH(50, 75, 200, 150),
        normalizedBox: const Rect.fromLTWH(0.2, 0.3, 0.4, 0.5),
      );

      final str = result.toString();
      expect(
        str,
        'YOLOResult{classIndex: 2, className: car, confidence: 0.87, boundingBox: Rect.fromLTRB(50.0, 75.0, 250.0, 225.0)}',
      );
    });
  });
}
