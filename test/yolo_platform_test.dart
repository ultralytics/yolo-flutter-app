// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';

class MockYOLOPlatform with MockPlatformInterfaceMixin implements YOLOPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<void> setModel(int viewId, String modelPath, String task) =>
      Future.value();
}

class _UnimplementedYOLOPlatform extends YOLOPlatform {
  Future<String?> callPlatformVersion() => super.getPlatformVersion();
  Future<void> callSetModel() => super.setModel(1, 'model.tflite', 'detect');
}

class _FakePlatform implements YOLOPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('fake');

  @override
  Future<void> setModel(int viewId, String modelPath, String task) async {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOPlatform', () {
    test('getPlatformVersion returns expected value from mock', () async {
      YOLOPlatform.instance = MockYOLOPlatform();
      expect(await YOLOPlatform.instance.getPlatformVersion(), '42');
    });

    test('default getPlatformVersion throws UnimplementedError', () {
      final platform = _UnimplementedYOLOPlatform();
      expect(
        () => platform.callPlatformVersion(),
        throwsA(isA<UnimplementedError>()),
      );
    });

    test('default setModel throws UnimplementedError', () {
      final platform = _UnimplementedYOLOPlatform();
      expect(() => platform.callSetModel(), throwsA(isA<UnimplementedError>()));
    });

    test('Cannot set instance with invalid token', () {
      expect(
        () => YOLOPlatform.instance = _FakePlatform(),
        throwsA(isA<AssertionError>()),
      );
    });
  });
}
