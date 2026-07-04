// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO constructor', () {
    test('creates an instance with an explicit model path and task', () {
      final yolo = YOLO(modelPath: 'test_model.tflite', task: YOLOTask.detect);

      expect(yolo.modelPath, 'test_model.tflite');
      expect(yolo.task, YOLOTask.detect);
    });

    test('preserves task selection for all supported task types', () {
      final detectYolo = YOLO(
        modelPath: 'detect_model.tflite',
        task: YOLOTask.detect,
      );
      final segmentYolo = YOLO(
        modelPath: 'segment_model.tflite',
        task: YOLOTask.segment,
      );
      final classifyYolo = YOLO(
        modelPath: 'classify_model.tflite',
        task: YOLOTask.classify,
      );
      final poseYolo = YOLO(
        modelPath: 'pose_model.tflite',
        task: YOLOTask.pose,
      );

      expect(detectYolo.task, YOLOTask.detect);
      expect(segmentYolo.task, YOLOTask.segment);
      expect(classifyYolo.task, YOLOTask.classify);
      expect(poseYolo.task, YOLOTask.pose);
    });

    test('preserves custom model paths verbatim', () {
      final assetPathYolo = YOLO(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
      );
      final unicodePathYolo = YOLO(
        modelPath: 'models/special_chars_émojï_🤖.tflite',
        task: YOLOTask.detect,
      );

      expect(assetPathYolo.modelPath, 'assets/models/yolo11n.tflite');
      expect(unicodePathYolo.modelPath, 'models/special_chars_émojï_🤖.tflite');
    });
  });
}
