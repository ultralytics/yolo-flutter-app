// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'utils/test_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOViewController', () {
    late YOLOViewController controller;
    late MethodChannel mockChannel;
    late List<MethodCall> log;

    setUp(() {
      controller = YOLOViewController();
      final setup = YOLOTestHelpers.createYOLOTestSetup();
      mockChannel = setup.$1;
      log = setup.$2;
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(mockChannel, null);
      log.clear();
    });

    test('default values and threshold clamping', () {
      expect(controller.confidenceThreshold, 0.25);
      expect(controller.iouThreshold, 0.7);
      expect(controller.numItemsThreshold, 30);

      // Test clamping
      controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);
      controller.setConfidenceThreshold(-0.5);
      expect(controller.confidenceThreshold, 0.0);

      controller.setIoUThreshold(1.2);
      expect(controller.iouThreshold, 1.0);
      controller.setIoUThreshold(-0.1);
      expect(controller.iouThreshold, 0.0);

      controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);
      controller.setNumItemsThreshold(0);
      expect(controller.numItemsThreshold, 1);
    });

    test('setThresholds updates values correctly', () async {
      await controller.setThresholds(
        confidenceThreshold: 0.8,
        iouThreshold: 0.6,
        numItemsThreshold: 50,
      );

      expect(controller.confidenceThreshold, 0.8);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 50);

      // Test partial updates
      await controller.setThresholds(confidenceThreshold: 0.7);
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.6); // unchanged
      expect(controller.numItemsThreshold, 50); // unchanged
    });

    test('platform methods work with initialized channel', () async {
      controller.init(mockChannel, 1);

      // Test threshold methods
      YOLOTestHelpers.validateThresholdBehavior(controller, log, mockChannel);

      // Test camera controls
      await controller.switchCamera();
      YOLOTestHelpers.assertMethodCalled(log, 'switchCamera');

      await controller.setTorchMode(true);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setTorchMode',
        arguments: {'enabled': true},
      );

      await controller.zoomIn();
      YOLOTestHelpers.assertMethodCalled(log, 'zoomIn');

      await controller.zoomOut();
      YOLOTestHelpers.assertMethodCalled(log, 'zoomOut');

      await controller.setZoomLevel(2.0);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setZoomLevel',
        arguments: {'zoomLevel': 2.0},
      );

      await controller.setShowOverlays(false);
      YOLOTestHelpers.assertMethodCalled(
        log,
        'setShowOverlays',
        arguments: {'visible': false},
      );

      // Test capture frame
      final result = await controller.captureFrame();
      expect(result, isA<Uint8List>());
      YOLOTestHelpers.assertMethodCalled(log, 'captureFrame');
    });

    test('lens, focus, and photo methods invoke platform channel', () async {
      final calls = <MethodCall>[];
      const channel = MethodChannel('yolo_controller_methods_test');
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (call) async {
            calls.add(call);
            switch (call.method) {
              case 'getAvailableLenses':
                return [
                  {'zoomFactor': 0.5, 'label': 'Ultra wide camera'},
                  {'zoomFactor': 1, 'label': 'Wide camera'},
                ];
              case 'capturePhoto':
                return Uint8List.fromList([1, 2, 3]);
              default:
                return true;
            }
          });
      addTearDown(() {
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
            .setMockMethodCallHandler(channel, null);
      });

      controller.init(channel, 7);
      final lenses = await controller.getAvailableLenses();
      await controller.setLens(0.5);
      await controller.tapToFocus(0.25, 0.75);
      final photo = await controller.capturePhoto(withOverlays: false);

      expect(lenses, hasLength(2));
      expect(lenses.first.zoomFactor, 0.5);
      expect(lenses.first.label, 'Ultra wide camera');
      expect(photo, orderedEquals([1, 2, 3]));
      YOLOTestHelpers.assertMethodCalled(calls, 'getAvailableLenses');
      YOLOTestHelpers.assertMethodCalled(
        calls,
        'setLens',
        arguments: {'zoomFactor': 0.5},
      );
      YOLOTestHelpers.assertMethodCalled(
        calls,
        'tapToFocus',
        arguments: {'x': 0.25, 'y': 0.75},
      );
      YOLOTestHelpers.assertMethodCalled(
        calls,
        'capturePhoto',
        arguments: {'withOverlays': false},
      );
    });

    test('native camera events are routed to streams', () async {
      final zoom = expectLater(controller.zoomEvents, emits(2.5));
      final lens = expectLater(
        controller.lensEvents,
        emits('Telephoto camera'),
      );
      final focus = expectLater(
        controller.focusEvents,
        emits(const Offset(0.2, 0.8)),
      );

      controller.onNativeEvent({'type': 'zoom', 'value': 2.5});
      controller.onNativeEvent({'type': 'lens', 'label': 'Telephoto camera'});
      controller.onNativeEvent({'type': 'focus', 'x': 0.2, 'y': 0.8});

      await Future.wait([zoom, lens, focus]);
    });

    test('methods handle uninitialized channel gracefully', () async {
      final uninitializedController = YOLOViewController();
      expect(
        () => uninitializedController.setConfidenceThreshold(0.8),
        returnsNormally,
      );
      expect(() => uninitializedController.switchCamera(), returnsNormally);
    });

    test('switchModel resolves metadata before updating the view', () async {
      controller.init(mockChannel, 1);

      await controller.switchModel('test_model.tflite');

      YOLOTestHelpers.assertMethodCalled(
        log,
        'setModel',
        arguments: {'modelPath': 'test_model.tflite', 'task': 'detect'},
      );
      YOLOTestHelpers.assertMethodCalled(log, 'inspectModel');
    });

    test('switchModel surfaces metadata mismatch', () async {
      final setup = YOLOTestHelpers.createYOLOTestSetup(
        customResponses: {
          'inspectModel': (_) => {
            'path': 'test_model.tflite',
            'task': 'segment',
            'labels': ['person'],
          },
          'setModel': (_) => true,
        },
      );
      mockChannel = setup.$1;
      log = setup.$2;
      controller.init(mockChannel, 1);

      await expectLater(
        controller.switchModel('test_model.tflite', YOLOTask.detect),
        throwsA(isA<ModelLoadingException>()),
      );
    });

    test('switchModel propagates native setModel failures', () async {
      // Unlike the fire-and-forget controls (which swallow native errors), a setModel failure must surface so
      // YOLOView can revert the switch target and route to onModelError instead of committing the new model.
      final setup = YOLOTestHelpers.createYOLOTestSetup(
        customResponses: {
          'setModel': (_) => throw PlatformException(
            code: 'ERROR',
            message: 'native setModel failed',
          ),
        },
      );
      mockChannel = setup.$1;
      log = setup.$2;
      controller.init(mockChannel, 1);

      await expectLater(
        controller.switchModel('test_model.tflite', YOLOTask.detect),
        throwsA(isA<PlatformException>()),
      );
    });
  });
}
