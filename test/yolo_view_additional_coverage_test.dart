// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOView Additional Coverage', () {
    testWidgets('YOLOViewController error paths in threshold methods', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      // Mock channel that throws on setConfidenceThreshold but succeeds on setThresholds
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setConfidenceThreshold') {
              throw PlatformException(code: 'ERROR');
            } else if (methodCall.method == 'setThresholds') {
              return null; // Success
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should fall back to _applyThresholds
      await controller.setConfidenceThreshold(0.7);
      expect(controller.confidenceThreshold, 0.7);
    });

    testWidgets('YOLOViewController error paths in IoU threshold', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setIoUThreshold') {
              throw PlatformException(code: 'ERROR');
            } else if (methodCall.method == 'setThresholds') {
              return null;
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.setIoUThreshold(0.3);
      expect(controller.iouThreshold, 0.3);
    });

    testWidgets('YOLOViewController error paths in numItems threshold', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'setNumItemsThreshold') {
              throw PlatformException(code: 'ERROR');
            } else if (methodCall.method == 'setThresholds') {
              return null;
            }
            return null;
          });

      controller.init(testChannel, 1);

      await controller.setNumItemsThreshold(20);
      expect(controller.numItemsThreshold, 20);
    });

    testWidgets('YOLOViewController handles errors in zoom methods', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'zoomIn' ||
                methodCall.method == 'zoomOut' ||
                methodCall.method == 'setZoomLevel') {
              throw PlatformException(code: 'ERROR');
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should not throw
      await controller.zoomIn();
      await controller.zoomOut();
      await controller.setZoomLevel(2.0);
    });

    testWidgets('YOLOViewController handles errors in other methods', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
            if (methodCall.method == 'switchCamera' ||
                methodCall.method == 'setStreamingConfig' ||
                methodCall.method == 'stop') {
              throw PlatformException(code: 'ERROR');
            }
            return null;
          });

      controller.init(testChannel, 1);

      // Should not throw
      await controller.switchCamera();
      await controller.setStreamingConfig(const YOLOStreamingConfig.minimal());
      await controller.stop();
    });

    testWidgets('YOLOViewController switchModel error rethrows', (
      tester,
    ) async {
      final controller = YOLOViewController();
      const testChannel = MethodChannel('test_channel');

      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(testChannel, (methodCall) async {
        if (methodCall.method == 'setModel') {
          throw PlatformException(code: 'ERROR', message: 'Test error');
        }
        return null;
      });

      controller.init(testChannel, 1);

      expect(
        () => controller.switchModel('model.tflite', YOLOTask.detect),
        throwsException,
      );
    });
  });
}
