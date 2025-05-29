// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_method_channel.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final YOLOMethodChannel platform = YOLOMethodChannel();
  const MethodChannel channel = MethodChannel('yolo_single_image_channel');

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          if (methodCall.method == 'getPlatformVersion') {
            return '42';
          }
          return null;
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('getPlatformVersion', () async {
    expect(await platform.getPlatformVersion(), '42');
  });

  test('setModel calls method channel with correct arguments', () async {
    var called = false;
    late MethodCall capturedCall;

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          if (methodCall.method == 'setModel') {
            called = true;
            capturedCall = methodCall;
          }
          return null;
        });

    await platform.setModel(1, 'model.tflite', 'detect');

    expect(called, isTrue);
    expect(capturedCall.method, 'setModel');
    expect(capturedCall.arguments, {
      'viewId': 1,
      'modelPath': 'model.tflite',
      'task': 'detect',
    });
  });
}
