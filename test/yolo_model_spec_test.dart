// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_model_spec.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

void main() {
  group('YOLOModelSpec constructor and modelName', () {
    test('asserts when neither modelPath nor type is provided', () {
      expect(
        () => YOLOModelSpec(task: YOLOTask.detect),
        throwsA(isA<AssertionError>()),
      );
    });

    test('modelName uses type when provided', () {
      const spec = YOLOModelSpec(type: 'yolo11n', task: YOLOTask.detect);
      expect(spec.modelName, 'yolo11n');
      expect(spec.modelPath, isNull);
      expect(spec.task, YOLOTask.detect);
    });

    test('modelName derives from unix-style modelPath', () {
      const spec = YOLOModelSpec(
        modelPath: '/foo/bar/yolo11n.tflite',
        task: YOLOTask.segment,
      );
      expect(spec.modelName, 'yolo11n');
      expect(spec.modelPath, '/foo/bar/yolo11n.tflite');
      expect(spec.task, YOLOTask.segment);
    });

    test('modelName derives from windows-style modelPath', () {
      const spec = YOLOModelSpec(
        modelPath: r'C:\models\yolo11n-seg.onnx',
        task: YOLOTask.segment,
      );
      expect(spec.modelName, 'yolo11n-seg');
    });
  });

  group('YOLOModelSpec.toMap', () {
    test('includes modelName and task, omits modelPath when null/empty', () {
      const spec = YOLOModelSpec(type: 'yolo11n', task: YOLOTask.detect);
      final map = spec.toMap();
      expect(map['modelName'], 'yolo11n');
      expect(map['task'], YOLOTask.detect.name);
      expect(map.containsKey('modelPath'), isFalse);
    });

    test('includes modelPath when provided', () {
      const spec = YOLOModelSpec(
        modelPath: 'assets/yolo11n.tflite',
        task: YOLOTask.detect,
      );
      final map = spec.toMap();
      expect(map['modelName'], 'yolo11n');
      expect(map['task'], YOLOTask.detect.name);
      expect(map['modelPath'], 'assets/yolo11n.tflite');
    });
  });

  group('YOLOModelSpec.fromMap', () {
    test('prefers modelName field for type', () {
      final spec = YOLOModelSpec.fromMap({
        'modelName': 'custom-type',
        'task': 'segment',
      });
      expect(spec.modelName, 'custom-type');
      expect(spec.task, YOLOTask.segment);
      expect(spec.modelPath, isNull);
    });

    test('falls back to type field when modelName not provided', () {
      final spec = YOLOModelSpec.fromMap({
        'type': 'type-only',
        'task': 'classify',
      });
      expect(spec.modelName, 'type-only');
      expect(spec.task, YOLOTask.classify);
      expect(spec.modelPath, isNull);
    });

    test(
      'derives modelName from modelPath when no type/modelName provided',
      () {
        final spec = YOLOModelSpec.fromMap({
          'modelPath': '/a/b/c/yolo_pose.mlpackage',
          'task': 'pose',
        });
        expect(spec.modelPath, '/a/b/c/yolo_pose.mlpackage');
        expect(spec.modelName, 'yolo_pose');
        expect(spec.task, YOLOTask.pose);
      },
    );

    test('task parsing is case-insensitive', () {
      final s1 = YOLOModelSpec.fromMap({'type': 't', 'task': 'Detect'});
      final s2 = YOLOModelSpec.fromMap({'type': 't', 'task': 'SEGMENT'});
      final s3 = YOLOModelSpec.fromMap({'type': 't', 'task': 'cLaSsiFy'});
      expect(s1.task, YOLOTask.detect);
      expect(s2.task, YOLOTask.segment);
      expect(s3.task, YOLOTask.classify);
    });

    test('unknown task string defaults to detect', () {
      final spec = YOLOModelSpec.fromMap({'type': 't', 'task': 'unknown-task'});
      expect(spec.task, YOLOTask.detect);
    });
  });

  group('YOLOModelSpec.listFromDynamic', () {
    test('returns empty list for non-list input', () {
      expect(YOLOModelSpec.listFromDynamic(null), isEmpty);
      expect(YOLOModelSpec.listFromDynamic('not-a-list'), isEmpty);
      expect(YOLOModelSpec.listFromDynamic(123), isEmpty);
    });

    test('parses list of maps and filters invalid entries', () {
      final list = YOLOModelSpec.listFromDynamic([
        {'type': 'a', 'task': 'detect'},
        {'modelPath': '/m/b.tflite', 'task': 'segment'},
        'garbage',
        42,
        {'type': 'c', 'task': 'pose'},
      ]);
      expect(list.length, 3);
      expect(list[0].modelName, 'a');
      expect(list[0].task, YOLOTask.detect);
      expect(list[1].modelName, 'b');
      expect(list[1].task, YOLOTask.segment);
      expect(list[2].modelName, 'c');
      expect(list[2].task, YOLOTask.pose);
    });
  });

  group('YOLOModelSpec.copyWith & equality', () {
    test('copyWith updates specified fields', () {
      const spec = YOLOModelSpec(type: 'y', task: YOLOTask.obb);
      final updated = spec.copyWith(
        modelPath: '/x/y/z.tflite',
        task: YOLOTask.detect,
      );
      expect(updated.modelPath, '/x/y/z.tflite');
      expect(updated.task, YOLOTask.detect);
      // type remains the same
      expect(updated.modelName, 'y');
    });

    test('equality compares modelPath, type, and task', () {
      const a = YOLOModelSpec(type: 't', task: YOLOTask.detect);
      const b = YOLOModelSpec(type: 't', task: YOLOTask.detect);
      const c = YOLOModelSpec(type: 't', task: YOLOTask.segment);
      const d = YOLOModelSpec(
        modelPath: '/path/a.tflite',
        task: YOLOTask.detect,
      );

      expect(a == b, isTrue);
      expect(a.hashCode, b.hashCode);

      expect(a == c, isFalse);
      expect(a == d, isFalse);
    });
  });
}
