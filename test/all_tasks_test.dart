// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

// Store task types for each instance
final Map<String, String> _instanceTasks = {};

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Mock method channel
  const MethodChannel channel = MethodChannel('yolo_single_image_channel');
  final List<MethodCall> log = <MethodCall>[];

  setUp(() {
    log.clear();
    _instanceTasks.clear();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);

          switch (methodCall.method) {
            case 'loadModel':
              final args = methodCall.arguments as Map<dynamic, dynamic>;
              final instanceId = args['instanceId'] as String? ?? 'default';
              final task = args['task'] as String?;
              if (task != null) {
                _instanceTasks[instanceId] = task;
              }
              return true;
            case 'predictSingleImage':
              // Return mock data based on task type
              final args = methodCall.arguments as Map<dynamic, dynamic>;
              return _getMockResultForTask(args);
            default:
              return null;
          }
        });
  });

  group('All YOLO Tasks Tests', () {
    test('Pose estimation with keypoints', () async {
      final yolo = YOLO(modelPath: 'yolo11n-pose', task: YOLOTask.pose);

      await yolo.loadModel();
      final imageBytes = Uint8List(100); // Mock image data
      final results = await yolo.predict(imageBytes);

      // Check structure
      expect(results, isA<Map<String, dynamic>>());
      expect(results['boxes'], isNotNull);
      expect(results['detections'], isNotNull);
      expect(results['imageSize'], isNotNull);

      // Test YOLOResult parsing
      final detections = results['detections'] as List<dynamic>;
      expect(detections, isNotEmpty);

      final firstDetection = YOLOResult.fromMap(detections[0]);
      expect(firstDetection.className, equals('person'));
      expect(firstDetection.keypoints, isNotNull);
      expect(
        firstDetection.keypoints!.length,
        equals(17),
      ); // COCO pose keypoints
      expect(firstDetection.keypointConfidences!.length, equals(17));

      // Check normalized box
      expect(firstDetection.normalizedBox.left, greaterThanOrEqualTo(0));
      expect(firstDetection.normalizedBox.left, lessThanOrEqualTo(1));
    });

    test('Segmentation with masks', () async {
      final yolo = YOLO(modelPath: 'yolo11n-seg', task: YOLOTask.segment);

      await yolo.loadModel();
      final imageBytes = Uint8List(100);
      final results = await yolo.predict(imageBytes);

      expect(results['boxes'], isNotNull);
      expect(results['detections'], isNotNull);

      final detections = results['detections'] as List<dynamic>;
      expect(detections, isNotEmpty);

      final firstDetection = YOLOResult.fromMap(detections[0]);
      expect(firstDetection.mask, isNotNull);
      expect(firstDetection.mask!, isA<List<List<double>>>());
      expect(firstDetection.mask!.length, equals(160)); // Mock mask size
      expect(firstDetection.mask![0].length, equals(160));
    });

    test('Classification task', () async {
      final yolo = YOLO(modelPath: 'yolo11n-cls', task: YOLOTask.classify);

      await yolo.loadModel();
      final imageBytes = Uint8List(100);
      final results = await yolo.predict(imageBytes);

      expect(results['detections'], isNotNull);

      final detections = results['detections'] as List<dynamic>;
      expect(detections, isNotEmpty);

      final firstDetection = YOLOResult.fromMap(detections[0]);
      expect(firstDetection.className, equals('cat'));
      expect(firstDetection.confidence, equals(0.95));
    });

    test('OBB detection', () async {
      final yolo = YOLO(modelPath: 'yolo11n-obb', task: YOLOTask.obb);

      await yolo.loadModel();
      final imageBytes = Uint8List(100);
      final results = await yolo.predict(imageBytes);

      expect(results['detections'], isNotNull);

      final detections = results['detections'] as List<dynamic>;
      expect(detections, isNotEmpty);

      final firstDetection = YOLOResult.fromMap(detections[0]);
      expect(firstDetection.className, equals('vehicle'));
      expect(firstDetection.confidence, greaterThan(0));
    });

    test('Regular detection', () async {
      final yolo = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);

      await yolo.loadModel();
      final imageBytes = Uint8List(100);
      final results = await yolo.predict(imageBytes);

      expect(results['boxes'], isNotNull);
      expect(results['detections'], isNotNull);
      expect(results['imageSize'], isNotNull);

      final boxes = results['boxes'] as List<dynamic>;
      expect(boxes, isNotEmpty);

      // Check normalized coordinates
      final firstBox = boxes[0] as Map<dynamic, dynamic>;
      expect(firstBox['x1_norm'], isNotNull);
      expect(firstBox['y1_norm'], isNotNull);
      expect(firstBox['x2_norm'], isNotNull);
      expect(firstBox['y2_norm'], isNotNull);
    });
  });
}

// Mock data generator for different tasks
Map<String, dynamic> _getMockResultForTask(Map<dynamic, dynamic> args) {
  final base = {
    'boxes': [
      {
        'x1': 100.0,
        'y1': 100.0,
        'x2': 200.0,
        'y2': 200.0,
        'x1_norm': 0.15625,
        'y1_norm': 0.20833,
        'x2_norm': 0.3125,
        'y2_norm': 0.41667,
        'class': 'person',
        'confidence': 0.85,
      },
    ],
    'imageSize': {'width': 640, 'height': 480},
  };

  // Determine task from model path or instance ID
  String? taskType;
  final instanceId = args['instanceId'] as String?;

  // Check stored task from loadModel
  if (instanceId != null && _instanceTasks.containsKey(instanceId)) {
    taskType = _instanceTasks[instanceId];
  } else if (_instanceTasks.containsKey('default')) {
    taskType = _instanceTasks['default'];
  }

  // Add task-specific data
  switch (taskType) {
    case 'pose':
      base['keypoints'] = [
        {
          'coordinates': List.generate(
            17,
            (i) => {
              'x': 0.5 + i * 0.01,
              'y': 0.5 + i * 0.01,
              'confidence': 0.9 - i * 0.01,
            },
          ),
        },
      ];
      break;
    case 'segment':
      base['masks'] = [
        List.generate(160, (_) => List.generate(160, (_) => 0.0)),
      ];
      break;
    case 'classify':
      base['classification'] = {
        'topClass': 'cat',
        'topConfidence': 0.95,
        'top5Classes': ['cat', 'dog', 'bird', 'fish', 'horse'],
        'top5Confidences': [0.95, 0.03, 0.01, 0.005, 0.005],
      };
      break;
    case 'obb':
      base['obb'] = [
        {
          'points': [
            {'x': 100.0, 'y': 100.0},
            {'x': 200.0, 'y': 100.0},
            {'x': 200.0, 'y': 200.0},
            {'x': 100.0, 'y': 200.0},
          ],
          'class': 'vehicle',
          'confidence': 0.9,
        },
      ];
      break;
  }

  return base;
}
