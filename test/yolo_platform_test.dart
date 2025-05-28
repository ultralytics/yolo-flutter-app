// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';

class MockYOLOPlatform with MockPlatformInterfaceMixin implements YOLOPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOPlatform', () {
    test('getPlatformVersion returns expected value from mock', () async {
      YOLOPlatform.instance = MockYOLOPlatform();
      expect(await YOLOPlatform.instance.getPlatformVersion(), '42');
    });
  });
}
