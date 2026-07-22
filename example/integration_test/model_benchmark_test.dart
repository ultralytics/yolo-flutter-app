// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// On-device CPU/accelerator benchmark harness with optional QNN validation (device + network).
//
// Validation: loads the v73 QNN context-binary model for every task and checks real outputs on
// bus.jpg. Benchmark (enable with --dart-define=RUN_BENCH=true): times predict() per task per backend
// using the native speed (pre + inference + post, no plotting). CPU and GPU rows use the default
// official w8a32 LiteRT assets (what the app ships); QNN rows use the release context binaries.
//
// Run the shipped models on CPU and GPU (Android) or CPU and Neural Engine (iOS):
//   flutter test integration_test/model_benchmark_test.dart -d <device> --dart-define=RUN_BENCH=true
//
// Include QNN validation and benchmark rows on a supported Snapdragon device:
//   ENABLE_QNN=1 flutter test integration_test/model_benchmark_test.dart -d <device> \
//     --dart-define=RUN_BENCH=true --dart-define=RUN_QNN=true

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const String _releaseBase =
    'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.3.5';
const bool _runBench = bool.fromEnvironment('RUN_BENCH');
const bool _runQnn = bool.fromEnvironment('RUN_QNN');
const bool _runSoak = bool.fromEnvironment('RUN_SOAK');

const Map<String, (String, YOLOTask)> _tasks = {
  'detect': ('yolo26n', YOLOTask.detect),
  'segment': ('yolo26n-seg', YOLOTask.segment),
  'semantic': ('yolo26n-sem', YOLOTask.semantic),
  'depth': ('yolo26n-depth', YOLOTask.depth),
  'classify': ('yolo26n-cls', YOLOTask.classify),
  'pose': ('yolo26n-pose', YOLOTask.pose),
  'obb': ('yolo26n-obb', YOLOTask.obb),
};

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

Future<Map<String, dynamic>> _predictOnce(
  String modelPath,
  YOLOTask task,
  Uint8List image, {
  bool useGpu = true,
}) async {
  final yolo = YOLO(modelPath: modelPath, task: task, useGpu: useGpu);
  expect(await yolo.loadModel(), isTrue, reason: '$modelPath should load');
  final results = await yolo.predict(image);
  await yolo.dispose();
  return results;
}

Future<void> _bench(
  String label,
  String modelPath,
  YOLOTask task,
  Uint8List image, {
  bool useGpu = true,
  int warmup = 3,
  int runs = 15,
}) async {
  final yolo = YOLO(modelPath: modelPath, task: task, useGpu: useGpu);
  expect(await yolo.loadModel(), isTrue, reason: '$modelPath should load');
  for (var i = 0; i < warmup; i++) {
    await yolo.predict(image);
  }
  var pre = 0.0, infer = 0.0, post = 0.0, total = 0.0;
  for (var i = 0; i < runs; i++) {
    final r = await yolo.predict(image);
    pre += (r['preMs'] as num?)?.toDouble() ?? 0.0;
    infer += (r['inferenceMs'] as num?)?.toDouble() ?? 0.0;
    post += (r['postMs'] as num?)?.toDouble() ?? 0.0;
    total += (r['speed'] as num?)?.toDouble() ?? 0.0;
  }
  // ignore: avoid_print
  print(
    'BENCH|$label|${(pre / runs).toStringAsFixed(1)}|'
    '${(infer / runs).toStringAsFixed(1)}|${(post / runs).toStringAsFixed(1)}|'
    '${(total / runs).toStringAsFixed(1)}',
  );
  await yolo.dispose();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'QNN models run on the NPU for all six tasks',
    (WidgetTester tester) async {
      if (!_runQnn || !Platform.isAndroid) {
        return;
      }
      await tester.runAsync(() async {
        final image = await _download('https://ultralytics.com/images/bus.jpg');

        for (final entry in _tasks.entries.where(
          (entry) => entry.key != 'depth',
        )) {
          final (id, task) = entry.value;
          final results = await _predictOnce(
            '$_releaseBase/${id}_v73_qnn.onnx',
            task,
            image,
          );
          final detections =
              (results['detections'] as List?)?.cast<Map>() ?? [];
          final classes = detections.map((d) => d['className']).toSet();
          // ignore: avoid_print
          print(
            'QNN ${entry.key}: ${detections.length} detections classes=$classes '
            'keys=${results.keys.toList()} '
            'pre=${results['preMs']} infer=${results['inferenceMs']} post=${results['postMs']}',
          );
          switch (entry.key) {
            case 'detect' || 'segment':
              expect(classes, containsAll(['bus', 'person']));
            case 'pose':
              expect(detections, isNotEmpty);
            case 'classify':
              expect(results.containsKey('detections'), isTrue);
            case 'semantic':
              expect(results.containsKey('semanticMask'), isTrue);
            case 'obb':
              // DOTA aerial classes won't fire on bus.jpg; a clean run is the assertion
              expect(results, isA<Map<String, dynamic>>());
          }
        }
      });
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );

  testWidgets(
    'soak: sustained inference does not exhaust memory',
    (WidgetTester tester) async {
      if (!_runQnn || !_runSoak || !Platform.isAndroid) {
        return;
      }
      await tester.runAsync(() async {
        final image = await _download('https://ultralytics.com/images/bus.jpg');
        // Worst case: semantic logits are the largest output tensors in the model zoo.
        final yolo = YOLO(
          modelPath: '$_releaseBase/yolo26n-sem_v73_qnn.onnx',
          task: YOLOTask.semantic,
        );
        expect(await yolo.loadModel(), isTrue);
        for (var i = 0; i < 150; i++) {
          await yolo.predict(image);
          if (i % 25 == 0) {
            // ignore: avoid_print
            print('SOAK|$i');
          }
        }
        await yolo.dispose();
        // ignore: avoid_print
        print('SOAK|done');
      });
    },
    timeout: const Timeout(Duration(minutes: 30)),
  );

  testWidgets(
    'benchmark CPU vs preferred platform accelerator and optional QNN',
    (WidgetTester tester) async {
      if (!_runBench || !(Platform.isAndroid || Platform.isIOS)) {
        return;
      }
      await tester.runAsync(() async {
        final image = await _download('https://ultralytics.com/images/bus.jpg');
        for (final (index, entry) in _tasks.entries.indexed) {
          final (id, task) = entry.value;
          final accelerator = Platform.isAndroid
              ? 'gpu-preferred'
              : 'ane-preferred';
          final backends = <(String, String, bool)>[
            ('cpu', id, false),
            (accelerator, id, true),
            if (Platform.isAndroid && _runQnn && entry.key != 'depth')
              ('qnn', '$_releaseBase/${id}_v81_qnn.onnx', true),
          ];
          // Rotate backend order across tasks so no enabled backend is systematically measured last.
          for (var i = 0; i < backends.length; i++) {
            final (backend, modelPath, useGpu) =
                backends[(i + index) % backends.length];
            await _bench(
              '${entry.key}|$backend',
              modelPath,
              task,
              image,
              useGpu: useGpu,
            );
          }
        }
      });
    },
    timeout: const Timeout(Duration(minutes: 30)),
  );
}
