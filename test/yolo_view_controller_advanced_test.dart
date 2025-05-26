// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloViewController Advanced Tests', () {
    late YoloViewController controller;
    List<MethodCall> methodCalls = [];

    setUp(() {
      controller = YoloViewController();
      methodCalls.clear();
    });

    test('methods handle null method channel gracefully', () async {
      // Before initialization, method channel is null
      // These should not throw but should print warnings
      await expectLater(controller.setConfidenceThreshold(0.7), completes);
      await expectLater(controller.setIoUThreshold(0.5), completes);
      await expectLater(controller.setNumItemsThreshold(25), completes);
      await expectLater(controller.switchCamera(), completes);
      await expectLater(
        controller.setThresholds(
          confidenceThreshold: 0.8,
          iouThreshold: 0.6,
          numItemsThreshold: 30,
        ),
        completes,
      );
    });

    test('threshold clamping works correctly in edge cases', () async {
      // Test double.infinity
      await controller.setConfidenceThreshold(double.infinity);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setIoUThreshold(double.infinity);
      expect(controller.iouThreshold, 1.0);

      // Test double.negativeInfinity
      await controller.setConfidenceThreshold(double.negativeInfinity);
      expect(controller.confidenceThreshold, 0.0);

      await controller.setIoUThreshold(double.negativeInfinity);
      expect(controller.iouThreshold, 0.0);

      // Test NaN - should clamp to 0
      await controller.setConfidenceThreshold(double.nan);
      expect(controller.confidenceThreshold, 0.0);

      await controller.setIoUThreshold(double.nan);
      expect(controller.iouThreshold, 0.0);

      // Test negative numbers for numItems
      await controller.setNumItemsThreshold(-100);
      expect(controller.numItemsThreshold, 1);

      // Test very large numbers for numItems
      await controller.setNumItemsThreshold(999999);
      expect(controller.numItemsThreshold, 100);
    });

    test('setThresholds with partial updates preserves other values', () async {
      // Set initial values
      await controller.setThresholds(
        confidenceThreshold: 0.5,
        iouThreshold: 0.4,
        numItemsThreshold: 20,
      );

      // Update only confidence
      await controller.setThresholds(confidenceThreshold: 0.7);
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.4);
      expect(controller.numItemsThreshold, 20);

      // Update only IoU
      await controller.setThresholds(iouThreshold: 0.6);
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 20);

      // Update only numItems
      await controller.setThresholds(numItemsThreshold: 30);
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 30);

      // Update all with null values (should not change anything)
      await controller.setThresholds(
        confidenceThreshold: null,
        iouThreshold: null,
        numItemsThreshold: null,
      );
      expect(controller.confidenceThreshold, 0.7);
      expect(controller.iouThreshold, 0.6);
      expect(controller.numItemsThreshold, 30);
    });

    test('multiple rapid threshold changes are handled correctly', () async {
      // Simulate rapid threshold changes
      final futures = <Future>[];

      for (int i = 0; i < 10; i++) {
        futures.add(controller.setConfidenceThreshold(i / 10));
        futures.add(controller.setIoUThreshold(i / 10));
        futures.add(controller.setNumItemsThreshold(i * 10));
      }

      await Future.wait(futures);

      // Final values should be from the last iteration
      expect(controller.confidenceThreshold, 0.9);
      expect(controller.iouThreshold, 0.9);
      expect(controller.numItemsThreshold, 90);
    });

    test('boundary values are handled correctly', () async {
      // Test exact boundary values
      await controller.setConfidenceThreshold(0.0);
      expect(controller.confidenceThreshold, 0.0);

      await controller.setConfidenceThreshold(1.0);
      expect(controller.confidenceThreshold, 1.0);

      await controller.setIoUThreshold(0.0);
      expect(controller.iouThreshold, 0.0);

      await controller.setIoUThreshold(1.0);
      expect(controller.iouThreshold, 1.0);

      await controller.setNumItemsThreshold(1);
      expect(controller.numItemsThreshold, 1);

      await controller.setNumItemsThreshold(100);
      expect(controller.numItemsThreshold, 100);
    });

    test('setThresholds with extreme values clamps correctly', () async {
      await controller.setThresholds(
        confidenceThreshold: -1000.0,
        iouThreshold: 1000.0,
        numItemsThreshold: -1000,
      );

      expect(controller.confidenceThreshold, 0.0);
      expect(controller.iouThreshold, 1.0);
      expect(controller.numItemsThreshold, 1);

      await controller.setThresholds(
        confidenceThreshold: double.maxFinite,
        iouThreshold: double.minPositive,
        numItemsThreshold: 9999999,
      );

      expect(controller.confidenceThreshold, 1.0);
      expect(controller.iouThreshold, closeTo(0.0, 0.01));
      expect(controller.numItemsThreshold, 100);
    });

    test('getters return correct values after initialization', () {
      final newController = YoloViewController();

      expect(newController.confidenceThreshold, 0.5);
      expect(newController.iouThreshold, 0.45);
      expect(newController.numItemsThreshold, 30);
    });
  });
}
