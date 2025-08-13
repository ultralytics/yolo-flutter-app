// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

// Mock data functions
Map<String, dynamic> _getMockDetectionResult() {
  return {
    'boxes': [
      {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95,
        'boundingBox': {
          'left': 100.0,
          'top': 150.0,
          'right': 300.0,
          'bottom': 450.0,
        },
        'normalizedBox': {
          'left': 0.1,
          'top': 0.15,
          'right': 0.3,
          'bottom': 0.45,
        },
      },
      {
        'classIndex': 2,
        'className': 'car',
        'confidence': 0.87,
        'boundingBox': {
          'left': 400.0,
          'top': 200.0,
          'right': 600.0,
          'bottom': 350.0,
        },
        'normalizedBox': {
          'left': 0.4,
          'top': 0.2,
          'right': 0.6,
          'bottom': 0.35,
        },
      },
    ],
    'processingTimeMs': 25.5,
    'imageWidth': 640,
    'imageHeight': 480,
  };
}

Map<String, dynamic> _getMockStreamResult() {
  return {
    'detections': [
      {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.92,
        'boundingBox': {
          'left': 120.0,
          'top': 160.0,
          'right': 280.0,
          'bottom': 420.0,
        },
      },
    ],
    'fps': 30.5,
    'processingTimeMs': 22.3,
  };
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO E2E Functional Tests', () {
    late YOLO yoloInstance;
    late List<MethodCall> methodCallLog;

    setUp(() {
      methodCallLog = <MethodCall>[];

      // Mock the YOLO method channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('yolo_single_image_channel'),
            (MethodCall methodCall) async {
              methodCallLog.add(methodCall);

              switch (methodCall.method) {
                case 'loadModel':
                  return true;
                case 'predictSingleImage':
                  return _getMockDetectionResult();
                case 'predictStream':
                  return _getMockStreamResult();
                case 'dispose':
                  return true;
                default:
                  return null;
              }
            },
          );

      // Mock camera permissions
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/camera'),
            (MethodCall methodCall) async {
              if (methodCall.method == 'requestPermissions') {
                return {'camera': 'authorized', 'microphone': 'authorized'};
              }
              return null;
            },
          );
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('yolo_single_image_channel'),
            null,
          );
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/camera'),
            null,
          );
    });

    group('YOLO Model Loading Tests', () {
      testWidgets('YOLO model loads successfully with detect task', (
        WidgetTester tester,
      ) async {
        yoloInstance = YOLO(
          modelPath: 'assets/models/yolo11n.tflite',
          task: YOLOTask.detect,
        );

        final success = await yoloInstance.loadModel();

        expect(success, isTrue);
        expect(methodCallLog.length, 1);
        expect(methodCallLog[0].method, 'loadModel');
        expect(methodCallLog[0].arguments['task'], 'detect');
        expect(
          methodCallLog[0].arguments['modelPath'],
          'assets/models/yolo11n.tflite',
        );
      });

      testWidgets('YOLO model loads successfully with segment task', (
        WidgetTester tester,
      ) async {
        yoloInstance = YOLO(
          modelPath: 'assets/models/yolo11n-seg.mlpackage',
          task: YOLOTask.segment,
        );

        final success = await yoloInstance.loadModel();

        expect(success, isTrue);
        expect(methodCallLog[0].arguments['task'], 'segment');
      });

      testWidgets('YOLO model loads successfully with classify task', (
        WidgetTester tester,
      ) async {
        yoloInstance = YOLO(
          modelPath: 'assets/models/yolo11n-cls.tflite',
          task: YOLOTask.classify,
        );

        final success = await yoloInstance.loadModel();

        expect(success, isTrue);
        expect(methodCallLog[0].arguments['task'], 'classify');
      });

      testWidgets('YOLO model loads successfully with pose task', (
        WidgetTester tester,
      ) async {
        yoloInstance = YOLO(
          modelPath: 'assets/models/yolo11n-pose.tflite',
          task: YOLOTask.pose,
        );

        final success = await yoloInstance.loadModel();

        expect(success, isTrue);
        expect(methodCallLog[0].arguments['task'], 'pose');
      });
    });
  });
}
