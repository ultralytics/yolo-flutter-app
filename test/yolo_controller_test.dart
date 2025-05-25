// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloViewController', () {
    late YoloViewController controller;
    late MethodChannel mockChannel;
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      controller = YoloViewController();
      mockChannel = const MethodChannel('test_channel');
      log.clear();

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(mockChannel, (MethodCall methodCall) async {
        log.add(methodCall);
        return null;
      });

      controller._init(mockChannel);
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(mockChannel, null);
    });

    test('default values are set correctly', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('setConfidenceThreshold clamps values and calls platform', () async {
      await controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);
      expect(log.last.method, 'setConfidenceThreshold');
      expect(log.last.arguments['threshold'], 1.0);

      await controller.setConfidenceThreshold(-0.5);
      expect(controller.confidenceThreshold, 0.0);
    });

    test('setIoUThreshold clamps values and calls platform', () async {
      await controller.setIoUThreshold(1.2);
      expect(controller.iouThreshold, 1.0);
      expect(log.last.method, 'setIoUThreshold');
      expect(log.last.arguments['threshold'], 1.0);

      await controller.setIoUThreshold(-0.1);
      expect(controller.iouThreshold, 0.0);
    });

    test('setNumItemsThreshold clamps values and calls platform', () async {
      await controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);
      expect(log.last.method, 'setNumItemsThreshold');
      expect(log.last.arguments['numItems'], 100);

      await controller.setNumItemsThreshold(0);
      expect(controller.numItemsThreshold, 1);
    });

    test('setThresholds updates multiple values at once', () async {
      await controller.setThresholds(
        confidenceThreshold: 0.8,
        iouThreshold: 0.6,
        numItemsThreshold: 50,
      );

      expect(controller.confidenceThreshold, 0.8);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 50);
      expect(log.last.method, 'setThresholds');
    });

    test('setThresholds updates only provided values', () async {
      await controller.setThresholds(confidenceThreshold: 0.7);

      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.45); // unchanged
      expect(controller.numItemsThreshold, 30); // unchanged
    });

    test('switchCamera calls platform method', () async {
      await controller.switchCamera();
      expect(log.last.method, 'switchCamera');
    });

    test('methods handle platform channel not initialized gracefully', () async {
      final uninitializedController = YoloViewController();
      
      // Should not throw, just log warning
      expect(() => uninitializedController.setConfidenceThreshold(0.8), returnsNormally);
      expect(() => uninitializedController.switchCamera(), returnsNormally);
    });
  });
}
