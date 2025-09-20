// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
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
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
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

      // Test capture frame
      final result = await controller.captureFrame();
      expect(result, isA<Uint8List>());
      YOLOTestHelpers.assertMethodCalled(log, 'captureFrame');
    });

    test('methods handle uninitialized channel gracefully', () async {
      final uninitializedController = YOLOViewController();
      expect(
        () => uninitializedController.setConfidenceThreshold(0.8),
        returnsNormally,
      );
      expect(() => uninitializedController.switchCamera(), returnsNormally);
    });
  });
}
