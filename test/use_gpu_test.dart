import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  group('useGpu Feature Tests', () {
    test('YOLO constructor should accept useGpu parameter', () {
      // Test with GPU enabled (default)
      final yoloWithGpu = YOLO(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
        useGpu: true,
      );
      expect(yoloWithGpu.useGpu, true);

      // Test with GPU disabled
      final yoloWithoutGpu = YOLO(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
        useGpu: false,
      );
      expect(yoloWithoutGpu.useGpu, false);

      // Test default value
      final yoloDefault = YOLO(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
      );
      expect(yoloDefault.useGpu, true);
    });

    test('YOLOView constructor should accept useGpu parameter', () {
      // Test with GPU enabled (default)
      final viewWithGpu = YOLOView(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
        useGpu: true,
        onResult: (results) {},
      );
      expect(viewWithGpu.useGpu, true);

      // Test with GPU disabled
      final viewWithoutGpu = YOLOView(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
        useGpu: false,
        onResult: (results) {},
      );
      expect(viewWithoutGpu.useGpu, false);

      // Test default value
      final viewDefault = YOLOView(
        modelPath: 'assets/models/yolo11n.tflite',
        task: YOLOTask.detect,
        onResult: (results) {},
      );
      expect(viewDefault.useGpu, true);
    });

    test('YOLO withClassifierOptions should accept useGpu parameter', () {
      final classifierOptions = {
        'enable1ChannelSupport': true,
        'expectedChannels': 1,
      };

      // Test with GPU enabled
      final yoloWithGpu = YOLO.withClassifierOptions(
        modelPath: 'assets/models/classifier.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
        useGpu: true,
      );
      expect(yoloWithGpu.useGpu, true);

      // Test with GPU disabled
      final yoloWithoutGpu = YOLO.withClassifierOptions(
        modelPath: 'assets/models/classifier.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
        useGpu: false,
      );
      expect(yoloWithoutGpu.useGpu, false);

      // Test default value
      final yoloDefault = YOLO.withClassifierOptions(
        modelPath: 'assets/models/classifier.tflite',
        task: YOLOTask.classify,
        classifierOptions: classifierOptions,
      );
      expect(yoloDefault.useGpu, true);
    });
  });
}
