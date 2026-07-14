// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Physical-device validation for every official YOLO26 depth model.
//
// Runs all sizes in one app session to avoid repeated Xiaomi install approvals:
//   flutter test integration_test/depth_benchmark_test.dart -d <device> \
//     --dart-define=RUN_DEPTH_BENCH=true

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const bool _runDepthBench = bool.fromEnvironment('RUN_DEPTH_BENCH');
const List<String> _depthModels = [
  'yolo26n-depth',
  'yolo26s-depth',
  'yolo26m-depth',
  'yolo26l-depth',
  'yolo26x-depth',
];

Future<Uint8List> _downloadImage() async {
  final client = HttpClient();
  try {
    final request = await client.getUrl(
      Uri.parse('https://ultralytics.com/images/bus.jpg'),
    );
    final response = await request.close();
    expect(response.statusCode, HttpStatus.ok);
    final bytes = BytesBuilder(copy: false);
    await for (final chunk in response) {
      bytes.add(chunk);
    }
    return bytes.takeBytes();
  } finally {
    client.close();
  }
}

void _validateDepth(String model, Map<String, dynamic> result) {
  if (Platform.isAndroid) {
    expect(
      result['accelerator'],
      'GPU',
      reason: '$model must run fully on GPU',
    );
  }
  final depth = result['depthMap'];
  expect(depth, isA<Map>(), reason: '$model must return a depth map');
  final map = depth! as Map<dynamic, dynamic>;
  final values = map['values'];
  expect(values, isA<Float32List>());
  final width = map['width'] as int;
  final height = map['height'] as int;
  final typedValues = values! as Float32List;
  expect(width, greaterThan(0));
  expect(height, greaterThan(0));
  expect(typedValues, hasLength(width * height));
  var expectedMin = double.infinity;
  var expectedMax = double.negativeInfinity;
  for (final value in typedValues) {
    if (value.isFinite && value > 0) {
      if (value < expectedMin) expectedMin = value;
      if (value > expectedMax) expectedMax = value;
    }
  }
  expect(expectedMin.isFinite, isTrue);
  expect((map['minDepth'] as num).toDouble(), expectedMin);
  expect((map['maxDepth'] as num).toDouble(), expectedMax);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'all depth models run on a physical mobile device',
    (WidgetTester tester) async {
      await tester.runAsync(() async {
        final image = await _downloadImage();
        for (final model in _depthModels) {
          final yolo = YOLO(modelPath: model, task: YOLOTask.depth);
          expect(await yolo.loadModel(), isTrue, reason: '$model should load');

          Map<String, dynamic>? result;
          for (var i = 0; i < 3; i++) {
            result = await yolo.predict(image);
          }
          _validateDepth(model, result!);

          var pre = 0.0;
          var inference = 0.0;
          var post = 0.0;
          var total = 0.0;
          const runs = 15;
          for (var i = 0; i < runs; i++) {
            result = await yolo.predict(image);
            pre += (result['preMs'] as num).toDouble();
            inference += (result['inferenceMs'] as num).toDouble();
            post += (result['postMs'] as num).toDouble();
            total += (result['speed'] as num).toDouble();
          }
          _validateDepth(model, result!);
          final depth = result['depthMap']! as Map<dynamic, dynamic>;
          // ignore: avoid_print
          print(
            'DEPTH|$model|${(pre / runs).toStringAsFixed(2)}|'
            '${(inference / runs).toStringAsFixed(2)}|'
            '${(post / runs).toStringAsFixed(2)}|'
            '${(total / runs).toStringAsFixed(2)}|'
            '${depth['width']}x${depth['height']}|'
            '${(depth['minDepth'] as num).toStringAsFixed(3)}|'
            '${(depth['maxDepth'] as num).toStringAsFixed(3)}',
          );
          await yolo.dispose();
        }
      });
    },
    skip: !_runDepthBench || !(Platform.isAndroid || Platform.isIOS),
    timeout: const Timeout(Duration(minutes: 30)),
  );
}
