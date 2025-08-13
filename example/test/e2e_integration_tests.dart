// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo_example/main.dart';
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';
import 'package:ultralytics_yolo_example/presentation/screens/single_image_screen.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('YOLO Flutter App E2E Integration Tests', () {
    setUpAll(() async {
      // Set up any global test configuration
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/image_picker'),
            (MethodCall methodCall) async {
              if (methodCall.method == 'pickImage') {
                // Return a mock image for testing
                return 'mock_image_path';
              }
              return null;
            },
          );
    });

    tearDownAll(() async {
      // Clean up any global test resources
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(
            const MethodChannel('plugins.flutter.io/image_picker'),
            null,
          );
    });

    group('App Initialization and Navigation', () {
      testWidgets('App launches and shows camera inference screen', (
        WidgetTester tester,
      ) async {
        // Build the app
        await tester.pumpWidget(const MyApp());
        await tester.pumpAndSettle();

        // Verify the app launches successfully
        expect(find.byType(CameraInferenceScreen), findsOneWidget);
        expect(find.text('YOLO useGpu Example'), findsOneWidget);

        // Verify key UI elements are present
        expect(find.byType(YOLOView), findsOneWidget);
        expect(find.byType(AppBar), findsOneWidget);
      });

      testWidgets('App maintains state during hot reload', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(const MyApp());
        await tester.pumpAndSettle();

        // Verify initial state
        expect(find.byType(CameraInferenceScreen), findsOneWidget);

        // Simulate hot reload
        await tester.pumpWidget(const MyApp());
        await tester.pumpAndSettle();

        // Verify state is maintained
        expect(find.byType(CameraInferenceScreen), findsOneWidget);
      });
    });
  });
}
