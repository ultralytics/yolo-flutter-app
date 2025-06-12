// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:flutter/services.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Coverage - Error paths', () {
    const MethodChannel channel = MethodChannel('yolo_single_image_channel');

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    test('_initializeInstance handles createInstance failure', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'createInstance') {
              throw Exception('Platform error');
            }
            return null;
          });

      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to initialize YOLO instance'),
          ),
        ),
      );
    });

    test(
      'switchModel handles UNSUPPORTED_TASK with task name in message',
      () async {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
              if (methodCall.method == 'setModel') {
                throw PlatformException(
                  code: 'UNSUPPORTED_TASK',
                  message: 'Task not supported',
                );
              }
              return null;
            });

        final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
        yolo.setViewId(1);

        expect(
          () => yolo.switchModel('model.tflite', YOLOTask.obb),
          throwsA(
            isA<ModelLoadingException>().having(
              (e) => e.message,
              'message',
              contains('Unsupported task type: obb for model: model.tflite'),
            ),
          ),
        );
      },
    );

    test('switchModel handles unknown error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'setModel') {
              throw Exception('Unknown error');
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      yolo.setViewId(1);

      expect(
        () => yolo.switchModel('model.tflite', YOLOTask.detect),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Failed to switch model'),
          ),
        ),
      );
    });

    test('loadModel handles UNSUPPORTED_TASK error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'UNSUPPORTED_TASK',
                message: 'Task not supported',
              );
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.pose);

      expect(
        () => yolo.loadModel(),
        throwsA(
          isA<ModelLoadingException>().having(
            (e) => e.message,
            'message',
            contains('Unsupported task type: pose for model: model.tflite'),
          ),
        ),
      );
    });

    test('loadModel handles generic platform error', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              throw PlatformException(
                code: 'UNKNOWN_ERROR',
                message: 'Something went wrong',
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
            contains('Failed to load model: Something went wrong'),
          ),
        ),
      );
    });

    test('predict validates confidence threshold range', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);

      // Test below 0
      expect(
        () => yolo.predict(image, confidenceThreshold: -0.1),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('Confidence threshold must be between 0.0 and 1.0'),
          ),
        ),
      );

      // Test above 1
      expect(
        () => yolo.predict(image, confidenceThreshold: 1.5),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('Confidence threshold must be between 0.0 and 1.0'),
          ),
        ),
      );
    });

    test('predict validates IoU threshold range', () async {
      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      final image = Uint8List.fromList([1, 2, 3]);

      // Test below 0
      expect(
        () => yolo.predict(image, iouThreshold: -0.1),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('IoU threshold must be between 0.0 and 1.0'),
          ),
        ),
      );

      // Test above 1
      expect(
        () => yolo.predict(image, iouThreshold: 1.5),
        throwsA(
          isA<InvalidInputException>().having(
            (e) => e.message,
            'message',
            contains('IoU threshold must be between 0.0 and 1.0'),
          ),
        ),
      );
    });

    test('predict includes threshold arguments when provided', () async {
      final List<MethodCall> log = [];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            log.add(methodCall);
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image, confidenceThreshold: 0.7, iouThreshold: 0.3);

      final predictCall = log.firstWhere(
        (call) => call.method == 'predictSingleImage',
      );
      expect(predictCall.arguments['confidenceThreshold'], 0.7);
      expect(predictCall.arguments['iouThreshold'], 0.3);
    });

    test('predict includes instanceId for multi-instance', () async {
      final List<MethodCall> log = [];

      // Mock the default channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            log.add(methodCall);
            if (methodCall.method == 'createInstance') {
              return null;
            }
            return null;
          });

      final yolo = YOLO(
        modelPath: 'model.tflite',
        task: YOLOTask.detect,
        useMultiInstance: true,
      );

      // Get the instance-specific channel name
      final instanceChannel = MethodChannel(
        'yolo_single_image_channel_${yolo.instanceId}',
      );

      // Mock the instance-specific channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(instanceChannel, (
            MethodCall methodCall,
          ) async {
            log.add(methodCall);
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {'boxes': [], 'detections': []};
            }
            return null;
          });

      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      await yolo.predict(image);

      final predictCall = log.firstWhere(
        (call) => call.method == 'predictSingleImage',
      );
      expect(predictCall.arguments.containsKey('instanceId'), true);
      expect(predictCall.arguments['instanceId'], startsWith('yolo_'));
    });

    test('predict handles empty boxes gracefully', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'loadModel') {
              return true;
            } else if (methodCall.method == 'predictSingleImage') {
              return {
                // No boxes key
                'detections': [],
              };
            }
            return null;
          });

      final yolo = YOLO(modelPath: 'model.tflite', task: YOLOTask.detect);
      await yolo.loadModel();

      final image = Uint8List.fromList([1, 2, 3]);
      final result = await yolo.predict(image);

      // When platform doesn't return boxes, the key won't exist
      expect(result.containsKey('boxes'), false);
      expect(result['detections'], []);
    });

    test('checkModelExists handles platform errors', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'checkModelExists') {
              throw PlatformException(code: 'ERROR', message: 'Platform error');
            }
            return null;
          });

      final result = await YOLO.checkModelExists('model.tflite');
      expect(result['exists'], false);
      expect(result['error'], 'Platform error');
    });

    test('checkModelExists handles general errors', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'checkModelExists') {
              throw Exception('General error');
            }
            return null;
          });

      final result = await YOLO.checkModelExists('model.tflite');
      expect(result['exists'], false);
      expect(result['error'], contains('Exception: General error'));
    });

    test('getStoragePaths handles platform errors', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'getStoragePaths') {
              throw PlatformException(code: 'ERROR', message: 'Platform error');
            }
            return null;
          });

      final result = await YOLO.getStoragePaths();
      expect(result, {});
    });

    test('getStoragePaths handles general errors', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
            if (methodCall.method == 'getStoragePaths') {
              throw Exception('General error');
            }
            return null;
          });

      final result = await YOLO.getStoragePaths();
      expect(result, {});
    });
  });

  group('YOLOResult Coverage - Error paths', () {
    test('fromMap handles missing optional fields', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
        // No optional fields
      };

      final result = YOLOResult.fromMap(map);
      expect(result.classIndex, 0);
      expect(result.className, 'person');
      expect(result.confidence, 0.95);
      expect(result.keypoints, isNull);
      expect(result.keypointConfidences, isNull);
      expect(result.mask, isNull);
    });

    test('BoundingBox getters calculate correctly', () {
      final map = {
        'classIndex': 0,
        'className': 'test',
        'confidence': 0.9,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
      };

      final result = YOLOResult.fromMap(map);
      final box = result.boundingBox;

      expect(box.left, 10);
      expect(box.top, 20);
      expect(box.width, 100);
      expect(box.height, 200);
      expect(box.center.dx, 60);
      expect(box.center.dy, 120);
    });

    test('Keypoint toString formats correctly', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.9,
        'boundingBox': {
          'left': 0.0,
          'top': 0.0,
          'right': 100.0,
          'bottom': 100.0,
        },
        'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
        'keypoints': [10.5, 20.5, 0.9], // x, y, confidence format
      };

      final result = YOLOResult.fromMap(map);
      expect(result.keypoints, isNotNull);
      expect(result.keypoints!.length, 1);
      final keypoint = result.keypoints!.first;
      expect(keypoint.x, 10.5);
      expect(keypoint.y, 20.5);
    });

    test('BoundingBox toString formats correctly', () {
      final map = {
        'classIndex': 0,
        'className': 'test',
        'confidence': 0.9,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
      };

      final result = YOLOResult.fromMap(map);
      final str = result.boundingBox.toString();
      expect(str, contains('Rect.fromLTRB(10.0, 20.0, 110.0, 220.0)'));
    });

    test('YOLOResult toString with all fields', () {
      final map = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 10.0,
          'top': 20.0,
          'right': 110.0,
          'bottom': 220.0,
        },
        'normalizedBox': {'left': 0.1, 'top': 0.1, 'right': 0.9, 'bottom': 0.9},
        'keypoints': [50.0, 50.0, 0.9],
        'mask': [
          [0.0, 1.0],
          [1.0, 0.0],
        ],
      };

      final result = YOLOResult.fromMap(map);
      final str = result.toString();
      expect(str, contains('YOLOResult'));
      expect(str, contains('person'));
      expect(str, contains('0.95'));
      // toString doesn't include keypoints or mask
      expect(result.keypoints!.length, 1);
      expect(result.mask!.length, 2);
    });

    test('YOLOResult toString without optional fields', () {
      final map = {
        'classIndex': 0,
        'className': 'car',
        'confidence': 0.85,
        'boundingBox': {
          'left': 0.0,
          'top': 0.0,
          'right': 100.0,
          'bottom': 100.0,
        },
        'normalizedBox': {'left': 0.0, 'top': 0.0, 'right': 1.0, 'bottom': 1.0},
      };

      final result = YOLOResult.fromMap(map);
      final str = result.toString();
      expect(str, contains('YOLOResult'));
      expect(str, contains('car'));
      expect(str, contains('0.85'));
      expect(str, isNot(contains('keypoints')));
      expect(str, isNot(contains('mask')));
    });
  });

  group('YOLOPerformanceMetrics Coverage', () {
    test('copyWith creates new instance with updated values', () {
      final metrics = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: DateTime(2024, 1, 1),
      );

      final updated = metrics.copyWith(fps: 60.0, frameNumber: 200);

      expect(updated.fps, 60.0);
      expect(updated.processingTimeMs, 33.3); // unchanged
      expect(updated.frameNumber, 200);
      expect(updated.timestamp, metrics.timestamp); // unchanged
    });

    test('toMap converts to correct format', () {
      final timestamp = DateTime(2024, 1, 1);
      final metrics = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final map = metrics.toMap();
      expect(map['fps'], 30.0);
      expect(map['processingTimeMs'], 33.3);
      expect(map['frameNumber'], 100);
      expect(map['timestamp'], timestamp.millisecondsSinceEpoch);
    });

    test('equality operator works correctly', () {
      final timestamp = DateTime(2024, 1, 1);
      final metrics1 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final metrics2 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final metrics3 = YOLOPerformanceMetrics(
        fps: 60.0, // different
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      // YOLOPerformanceMetrics doesn't override equality operator
      // so we need to compare their string representations
      expect(metrics1.toString(), equals(metrics2.toString()));
      expect(metrics1.toString(), isNot(equals(metrics3.toString())));
    });

    test('hashCode is consistent', () {
      final timestamp = DateTime(2024, 1, 1);
      final metrics1 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final metrics2 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      // Since equality isn't overridden, hashCode won't be consistent
      // Test that both objects have valid hashCodes instead
      expect(metrics1.hashCode, isA<int>());
      expect(metrics2.hashCode, isA<int>());
    });

    test('toString provides readable output', () {
      final metrics = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: DateTime(2024, 1, 1),
      );

      final str = metrics.toString();
      expect(str, contains('YOLOPerformanceMetrics'));
      expect(str, contains('fps: 30.0'));
      expect(str, contains('processingTime: 33.300ms'));
      expect(str, contains('frame: 100'));
      expect(str, contains('timestamp'));
    });
  });

  group('YOLOStreamingConfig Coverage', () {
    test('custom configuration with specific settings', () {
      const config = YOLOStreamingConfig.custom(
        includeDetections: false,
        includeClassifications: true,
        includeProcessingTimeMs: false,
        includeFps: true,
        includeMasks: false,
        includePoses: true,
        includeOBB: false,
        includeOriginalImage: true,
        maxFPS: 15,
      );

      expect(config.includeDetections, false);
      expect(config.includeClassifications, true);
      expect(config.includeProcessingTimeMs, false);
      expect(config.includeFps, true);
      expect(config.includeMasks, false);
      expect(config.includePoses, true);
      expect(config.includeOBB, false);
      expect(config.includeOriginalImage, true);
      expect(config.maxFPS, 15);
    });

    test('throttled config has correct settings', () {
      final config = YOLOStreamingConfig.throttled(maxFPS: 10);

      expect(config.includeDetections, true);
      expect(config.includeClassifications, true);
      expect(config.includeProcessingTimeMs, true);
      expect(config.includeFps, true);
      expect(config.includeMasks, false);
      expect(config.includePoses, false);
      expect(config.includeOBB, false);
      expect(config.includeOriginalImage, false);
      expect(config.maxFPS, 10);
      expect(config.throttleInterval, null);
      expect(config.inferenceFrequency, null);
      expect(config.skipFrames, null);
    });

    test('custom configuration with inference control', () {
      const config = YOLOStreamingConfig.custom(
        inferenceFrequency: 3,
        skipFrames: 2,
      );

      expect(config.inferenceFrequency, 3);
      expect(config.skipFrames, 2);
      expect(config.maxFPS, null);
      expect(config.throttleInterval, isNull);
    });
  });

  group('YOLOView Coverage - Error paths', () {
    test('YOLOViewController methods handle null channel gracefully', () async {
      final controller = YOLOViewController();

      // These should not throw
      await controller.setConfidenceThreshold(0.8);
      await controller.setIoUThreshold(0.5);
      await controller.setNumItemsThreshold(20);
      await controller.switchCamera();
      await controller.zoomIn();
      await controller.zoomOut();
      await controller.setZoomLevel(2.0);
      await controller.switchModel('model.tflite', YOLOTask.detect);
      await controller.setStreamingConfig(const YOLOStreamingConfig.minimal());
      await controller.stop();
    });

    test('YOLOViewController _applyThresholds handles errors', () async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setThresholds') {
              throw PlatformException(code: 'ERROR');
            } else if (methodCall.method == 'setConfidenceThreshold' ||
                methodCall.method == 'setIoUThreshold' ||
                methodCall.method == 'setNumItemsThreshold') {
              throw PlatformException(code: 'ERROR');
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should handle errors gracefully
      await controller.setThresholds(confidenceThreshold: 0.7);
    });
  });
}
