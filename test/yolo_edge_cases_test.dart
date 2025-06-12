// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Edge Cases Coverage', () {
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
                    'classIndex': 0,
                    'className': 'person',
                    'confidence': 0.95,
                    'boundingBox': {
                      'left': 10.0,
                      'top': 20.0,
                      'right': 110.0,
                      'bottom': 220.0,
                    },
                    'normalizedBox': {
                      'left': 0.1,
                      'top': 0.1,
                      'right': 0.9,
                      'bottom': 0.9,
                    },
                    'mask': [
                      [0.0, 0.5, 1.0],
                      [0.2, 0.8, 0.3],
                      [1.0, 0.5, 0.0],
                    ],
                  },
                ],
                'detections': [],
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.segment);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      expect(result['boxes'], hasLength(1));
      final box = result['boxes'][0] as YOLOResult;
      expect(box.mask, isNotNull);
      expect(box.mask!.length, 3);
      expect(box.mask![0].length, 3);
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
                      'classIndex': 0,
                      'className': 'person',
                      'confidence': 0.95,
                      'boundingBox': {
                        'left': 10.0,
                        'top': 20.0,
                        'right': 110.0,
                        'bottom': 220.0,
                      },
                      'normalizedBox': {
                        'left': 0.1,
                        'top': 0.1,
                        'right': 0.9,
                        'bottom': 0.9,
                      },
                      'keypoints': [
                        100.0, 50.0, 0.9, // nose
                        95.0, 55.0, 0.85, // left eye
                        105.0, 55.0, 0.87, // right eye
                      ],
                    },
                  ],
                  'detections': [],
                };
              }
              return null;
            });

        final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.pose);
        await yolo.loadModel();

        final image = Uint8List.fromList([1, 2, 3]);
        final result = await yolo.predict(image);

        expect(result['boxes'], hasLength(1));
        final box = result['boxes'][0] as YOLOResult;
        expect(box.keypoints, isNotNull);
        expect(box.keypoints!.length, 3);
        expect(box.keypointConfidences, isNotNull);
        expect(box.keypointConfidences!.length, 3);
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
            contains('Error reading model file: Corrupted model file'),
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
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('Failed to process image: Failed to decode image'),
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
            contains('Inference failed: GPU out of memory'),
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
        throwsA(
          isA<YOLOException>().having(
            (e) => e.message,
            'message',
            contains(
              'ViewId not set. Make sure this YOLO instance is attached to a YOLOView',
            ),
          ),
        ),
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
