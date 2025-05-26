// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_platform_interface.dart';
import 'package:ultralytics_yolo/yolo_method_channel.dart';

class MockYoloPlatform extends YoloPlatform {
  @override
  Future<String?> getPlatformVersion() async {
    return 'Mock Platform 1.0';
  }
}

class InvalidYoloPlatform implements YoloPlatform {
  @override
  Future<String?> getPlatformVersion() async {
    return 'Invalid Platform';
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class MinimalYoloPlatform extends YoloPlatform {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloPlatform', () {
    test('default instance is MethodChannelYolo', () {
      expect(YoloPlatform.instance, isA<MethodChannelYolo>());
    });

    test('can set and get valid instance', () {
      final mockPlatform = MockYoloPlatform();
      YoloPlatform.instance = mockPlatform;

      expect(YoloPlatform.instance, equals(mockPlatform));
      expect(YoloPlatform.instance, isA<MockYoloPlatform>());
    });

    test('getPlatformVersion throws UnimplementedError by default', () {
      final platform = MinimalYoloPlatform();

      expect(
        () => platform.getPlatformVersion(),
        throwsA(isA<UnimplementedError>()),
      );
    });

    test('getPlatformVersion returns value from mock implementation', () async {
      final mockPlatform = MockYoloPlatform();
      YoloPlatform.instance = mockPlatform;

      final version = await YoloPlatform.instance.getPlatformVersion();
      expect(version, 'Mock Platform 1.0');
    });

    test('verifyToken prevents invalid instance assignment', () {
      final invalidPlatform = InvalidYoloPlatform();

      expect(
        () => YoloPlatform.instance = invalidPlatform,
        throwsA(isA<AssertionError>()),
      );
    });

    test('instance property getter returns current instance', () {
      final mockPlatform = MockYoloPlatform();
      YoloPlatform.instance = mockPlatform;

      final instance1 = YoloPlatform.instance;
      final instance2 = YoloPlatform.instance;

      expect(identical(instance1, instance2), isTrue);
      expect(instance1, equals(mockPlatform));
    });

    test('token is properly initialized', () {
      // Create a new platform instance
      final platform = MockYoloPlatform();

      // Should be able to set it as instance without error
      expect(() => YoloPlatform.instance = platform, returnsNormally);
    });
  });
}
