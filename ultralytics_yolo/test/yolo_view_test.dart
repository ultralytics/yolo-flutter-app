// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// These imports are needed for the widget implementation
// import 'package:flutter/foundation.dart';
// import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('YoloView passes correct parameters to platform view', () {
    const view = YoloView(
      modelPath: 'test_model.tflite',
      task: YOLOTask.segment,
    );

    // Verify properties are correctly set
    expect(view.modelPath, equals('test_model.tflite'));
    expect(view.task, equals(YOLOTask.segment));
  });

  // Platform-specific widget tests are skipped because they're difficult to test
  // in the CI environment due to platform detection complexities.
  // These would ideally be tested in an integration testing environment.
}
