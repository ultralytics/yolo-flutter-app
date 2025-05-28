// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOViewController', () {
    late YOLOViewController controller;
    final List<MethodCall> log = <MethodCall>[];

    setUp(() {
      controller = YOLOViewController();
      log.clear();

      // Note: Cannot test _init directly as it's private
      // Controller methods will handle missing channel gracefully
    });

    tearDown(() {
      // No cleanup needed since we don't mock channels
    });

    test('default values are set correctly', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('setConfidenceThreshold clamps values', () async {
      await controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setConfidenceThreshold(-0.5);
      expect(controller.confidenceThreshold, 0.0);
    });

    test('setIoUThreshold clamps values', () async {
      await controller.setIoUThreshold(1.2);
      expect(controller.iouThreshold, 1.0);

      await controller.setIoUThreshold(-0.1);
      expect(controller.iouThreshold, 0.0);
    });

    test('setNumItemsThreshold clamps values', () async {
      await controller.setNumItemsThreshold(150);
      expect(controller.numItemsThreshold, 100);

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
    });

    test('setThresholds updates only provided values', () async {
      await controller.setThresholds(confidenceThreshold: 0.7);

      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.45); // unchanged
      expect(controller.numItemsThreshold, 30); // unchanged
    });

    test('switchCamera handles uninitialized channel gracefully', () async {
      // Should not throw when no method channel is set
      expect(() => controller.switchCamera(), returnsNormally);
    });

    test(
      'methods handle platform channel not initialized gracefully',
      () async {
        final uninitializedController = YOLOViewController();

        // Should not throw, just log warning
        expect(
          () => uninitializedController.setConfidenceThreshold(0.8),
          returnsNormally,
        );
        expect(() => uninitializedController.switchCamera(), returnsNormally);
      },
    );
  });
}
