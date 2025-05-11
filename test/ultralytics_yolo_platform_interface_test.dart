import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_interface.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('UltralyticsYoloPlatform Tests', () {
    late UltralyticsYoloPlatform platform;
    const channel = MethodChannel('ultralytics_yolo');

    setUp(() {
      debugPrint('Setting up test...');
      platform = UltralyticsYoloPlatform.instance;

      // Set up method channel mock
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        channel,
        (MethodCall methodCall) async {
          debugPrint('Method called: ${methodCall.method}');
          debugPrint('Arguments: ${methodCall.arguments}');

          if (methodCall.method == 'loadModel') {
            final args = methodCall.arguments as Map<dynamic, dynamic>;
            final modelData = args['model'] as Map<dynamic, dynamic>;

            // Return null for invalid model
            if (modelData.containsKey('invalid')) {
              debugPrint('Returning null for invalid model');
              return null;
            }

            // Return success for valid model
            debugPrint('Returning success for valid model');
            return 'success';
          }
          return null;
        },
      );
    });

    tearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
        channel,
        null,
      );
    });

    test('Platform instance is not null', () {
      debugPrint('Running: Platform instance is not null');
      expect(platform, isNotNull);
      debugPrint('✓ Platform instance is not null passed');
    });

    test('Platform instance is singleton', () {
      debugPrint('Running: Platform instance is singleton');
      final instance1 = UltralyticsYoloPlatform.instance;
      final instance2 = UltralyticsYoloPlatform.instance;
      expect(instance1, equals(instance2));
      debugPrint('✓ Platform instance is singleton passed');
    });

    test('Load model returns error for invalid model', () async {
      debugPrint('Running: Load model returns error for invalid model');
      try {
        final result = await platform.loadModel({'invalid': 'model'});
        debugPrint('Result: $result');
        expect(result, isNull);
        debugPrint('✓ Load model returns error for invalid model passed');
      } catch (e) {
        debugPrint('Error in test: $e');
        rethrow;
      }
    });
  });
}
