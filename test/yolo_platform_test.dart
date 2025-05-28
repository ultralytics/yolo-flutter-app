// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';

class MockYoloPlatform with MockPlatformInterfaceMixin implements YoloPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<void> setModel(int viewId, String modelPath, String task) =>
      Future.value();
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloPlatform', () {
    test('getPlatformVersion returns expected value from mock', () async {
      YoloPlatform.instance = MockYoloPlatform();
      expect(await YoloPlatform.instance.getPlatformVersion(), '42');
    });
  });
}
