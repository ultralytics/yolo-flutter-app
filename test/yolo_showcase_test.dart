// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/widgets/yolo_showcase.dart';

void main() {
  testWidgets('refreshes lens chips after switching camera', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final calls = <MethodCall>[];
    var lensRequestCount = 0;
    final testRoot = [
      Directory.systemTemp.path,
      'yolo_showcase_test_${DateTime.now().microsecondsSinceEpoch}',
    ].join(Platform.pathSeparator);
    final modelDir = [
      testRoot,
      'yolo26n.mlpackage',
    ].join(Platform.pathSeparator);
    Directory(testRoot).createSync(recursive: true);
    Directory(modelDir).createSync();
    File(
      [modelDir, 'Manifest.json'].join(Platform.pathSeparator),
    ).writeAsStringSync('{}');
    File(
      [testRoot, 'yolo26n_int8.tflite'].join(Platform.pathSeparator),
    ).writeAsBytesSync([0]);
    addTearDown(() {
      Directory(testRoot).deleteSync(recursive: true);
    });

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('plugins.flutter.io/path_provider'),
          (call) async {
            switch (call.method) {
              case 'getApplicationDocumentsDirectory':
                return testRoot;
              default:
                return null;
            }
          },
        );

    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    const codec = StandardMethodCodec();
    messenger.setMockMessageHandler(
      'dev.flutter.pigeon.wakelock_plus_platform_interface.WakelockPlusApi.toggle',
      (_) async => const StandardMessageCodec().encodeMessage(<Object?>[null]),
    );
    messenger.setMockStreamHandler(
      const EventChannel('${ChannelConfig.detectionResultsPrefix}[#1ad49]'),
      MockStreamHandler.inline(onListen: (_, _) {}),
    );
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(SystemChannels.platform_views, (call) async {
          switch (call.method) {
            case 'create':
              return 1;
            case 'resize':
              return {'width': 430.0, 'height': 900.0};
            case 'dispose':
            case 'touch':
            case 'setDirection':
            case 'clearFocus':
              return null;
            default:
              return null;
          }
        });

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel(ChannelConfig.singleImageChannel),
          (call) async {
            switch (call.method) {
              case 'checkModelExists':
                return {
                  'exists': true,
                  'path': call.arguments['modelPath'],
                  'location': 'assets',
                };
              case 'inspectModel':
                return {
                  'path': call.arguments['modelPath'],
                  'task': 'detect',
                  'labels': ['person'],
                };
              case 'getStoragePaths':
                return {
                  'internal': '/tmp/yolo_test',
                  'cache': '/tmp/yolo_test_cache',
                  'external': null,
                  'externalCache': null,
                };
              default:
                return null;
            }
          },
        );

    messenger.allMessagesHandler = (channel, handler, message) async {
      if (channel.startsWith(ChannelConfig.detectionResultsPrefix)) {
        return codec.encodeSuccessEnvelope(null);
      }
      if (!channel.startsWith(ChannelConfig.controlChannelPrefix)) {
        return handler?.call(message);
      }
      final call = codec.decodeMethodCall(message);
      try {
        final result = await (() async {
          calls.add(call);
          switch (call.method) {
            case 'setThresholds':
            case 'setShowOverlays':
            case 'switchCamera':
              return true;
            case 'getAvailableLenses':
              lensRequestCount += 1;
              if (lensRequestCount == 1) {
                return [
                  {'zoomFactor': 0.5, 'label': 'Ultra wide camera'},
                  {'zoomFactor': 1.0, 'label': 'Wide camera'},
                  {'zoomFactor': 3.0, 'label': 'Telephoto camera'},
                ];
              }
              return [
                {'zoomFactor': 1.0, 'label': 'Front camera'},
              ];
            default:
              return null;
          }
        })();
        return codec.encodeSuccessEnvelope(result);
      } on PlatformException catch (error) {
        return codec.encodeErrorEnvelope(
          code: error.code,
          message: error.message,
          details: error.details,
        );
      }
    };

    addTearDown(() {
      messenger.allMessagesHandler = null;
    });

    await tester.pumpWidget(
      const MaterialApp(
        home: SizedBox(width: 430, height: 900, child: YOLOShowcase()),
      ),
    );
    await tester.pump();
    await tester.pump();
    await tester.pump(const Duration(seconds: 2));

    expect(find.text('0.5'), findsOneWidget);
    expect(find.text('1'), findsOneWidget);
    expect(find.text('3'), findsOneWidget);

    await tester.tap(
      find.byIcon(CupertinoIcons.camera_rotate, skipOffstage: false),
    );
    for (var i = 0; i < 10 && lensRequestCount < 2; i++) {
      await tester.pump(const Duration(milliseconds: 100));
    }

    expect(find.text('0.5'), findsNothing);
    expect(find.text('1'), findsOneWidget);
    expect(find.text('3'), findsNothing);
    expect(
      calls.where((call) => call.method == 'getAvailableLenses'),
      hasLength(greaterThanOrEqualTo(2)),
    );
  });
}
