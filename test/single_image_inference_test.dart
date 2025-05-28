// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() {
  group('YOLOViewController', () {
    late YOLOViewController controller;

    setUp(() {
      controller = YOLOViewController();
    });

    test('default thresholds', () {
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });

    test('setConfidenceThreshold clamps value', () async {
      await controller.setConfidenceThreshold(1.5);
      expect(controller.confidenceThreshold, 1.0);
      await controller.setConfidenceThreshold(-0.5);
      expect(controller.confidenceThreshold, 0.0);
    });

    test('setIoUThreshold clamps value', () async {
      await controller.setIoUThreshold(2.0);
      expect(controller.iouThreshold, 1.0);
      await controller.setIoUThreshold(-1.0);
      expect(controller.iouThreshold, 0.0);
    });

    test('setNumItemsThreshold clamps value', () async {
      await controller.setNumItemsThreshold(200);
      expect(controller.numItemsThreshold, 100);
      await controller.setNumItemsThreshold(0);
      expect(controller.numItemsThreshold, 1);
    });
  });
}
