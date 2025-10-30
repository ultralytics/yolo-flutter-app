// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// This is a basic Flutter integration test.
//
// Since integration tests run in a full Flutter application, they can interact
// with the host side of a plugin implementation, unlike Dart unit tests.
//
// For more information about Flutter integration tests, please see
// https://flutter.dev/to/integration-testing

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/platform/yolo_platform_interface.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Basic YOLO initialization test', (WidgetTester tester) async {
    // Create YOLO instance with required parameters
    final YOLO yolo = YOLO(
      modelPath: 'assets/models/yolo11n.tflite',
      task: YOLOTask.detect,
    );

    // Check that YOLO instance was created successfully
    expect(yolo, isNotNull);
    expect(yolo.modelPath, 'assets/models/yolo11n.tflite');
    expect(yolo.task, YOLOTask.detect);
  });

  testWidgets('getPlatformVersion test', (WidgetTester tester) async {
    final String? version = await YOLOPlatform.instance.getPlatformVersion();
    // The version string depends on the host platform running the test, so
    // just assert that some non-empty string is returned.
    expect(version?.isNotEmpty, true);
  });
}
