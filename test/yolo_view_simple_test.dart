// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloView', () {
    test('YoloView passes correct parameters to platform view', () {
      const view = YoloView(
        modelPath: 'test_model.tflite',
        task: YOLOTask.segment,
      );

      // Verify properties are correctly set
      expect(view.modelPath, equals('test_model.tflite'));
      expect(view.task, equals(YOLOTask.segment));
    });

    test('YoloViewController can be instantiated', () {
      final controller = YoloViewController();

      // Test that default values are set
      expect(controller.confidenceThreshold, 0.5);
      expect(controller.iouThreshold, 0.45);
      expect(controller.numItemsThreshold, 30);
    });
  });
}
