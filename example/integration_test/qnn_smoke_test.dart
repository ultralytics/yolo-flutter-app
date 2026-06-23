// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// On-device Qualcomm QNN NPU smoke test (requires a Snapdragon device + network).
//
// Downloads the QNN context-binary detect model from the release, loads it
// through the ONNX Runtime QNN Execution Provider (Hexagon NPU), and verifies
// real detections on bus.jpg.
//
// Run with (the QNN runtime is opt-in, off by default):
//   ENABLE_QNN=1 flutter test integration_test/qnn_smoke_test.dart -d <device>

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

Future<Uint8List> _download(String url) async {
  final client = HttpClient();
  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();
    final builder = BytesBuilder(copy: false);
    await for (final chunk in response) {
      builder.add(chunk);
    }
    return builder.takeBytes();
  } finally {
    client.close();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'QNN detect model runs on the Snapdragon NPU',
    (WidgetTester tester) async {
      await tester.runAsync(() async {
        final yolo = YOLO(
          modelPath:
              'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5/yolo26n_v73_qnn.onnx',
          task: YOLOTask.detect,
        );
        final loaded = await yolo.loadModel();
        expect(loaded, isTrue, reason: 'QNN model should load on the NPU');

        final image = await _download('https://ultralytics.com/images/bus.jpg');
        final results = await yolo.predict(image);
        final detections = (results['detections'] as List).cast<Map>();
        final classes = detections.map((d) => d['className']).toSet();
        // ignore: avoid_print
        print('QNN detections: ${detections.length} classes=$classes');

        expect(detections.length, greaterThanOrEqualTo(2));
        expect(classes, contains('bus'));
        expect(classes, contains('person'));

        await yolo.dispose();
      });
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
