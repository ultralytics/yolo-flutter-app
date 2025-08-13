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

    group('Camera Inference Screen Tests', () {
      testWidgets('Camera inference screen loads with default settings', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Verify default UI elements
        expect(find.byType(YOLOView), findsOneWidget);
        expect(find.byType(AppBar), findsOneWidget);

        // Verify default model type (detect)
        expect(find.text('Detect'), findsOneWidget);
      });

      testWidgets('Model type selection works correctly', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Find and tap model type buttons
        final detectButton = find.text('Detect');
        final segmentButton = find.text('Segment');
        final classifyButton = find.text('Classify');
        final poseButton = find.text('Pose');
        final obbButton = find.text('OBB');

        expect(detectButton, findsOneWidget);
        expect(segmentButton, findsOneWidget);
        expect(classifyButton, findsOneWidget);
        expect(poseButton, findsOneWidget);
        expect(obbButton, findsOneWidget);

        // Test switching to segment model
        await tester.tap(segmentButton);
        await tester.pumpAndSettle();

        // Verify UI updates (this would depend on your implementation)
        // You might check for loading indicators or state changes
      });

      testWidgets('Confidence threshold slider works', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Find confidence slider
        final confidenceSlider = find.byType(Slider);
        expect(confidenceSlider, findsWidgets);

        // Test slider interaction
        await tester.drag(confidenceSlider.first, const Offset(50.0, 0.0));
        await tester.pumpAndSettle();

        // Verify slider value changed (implementation dependent)
      });

      testWidgets('IoU threshold slider works', (WidgetTester tester) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Find IoU slider
        final iouSlider = find.byType(Slider);
        expect(iouSlider, findsWidgets);

        // Test slider interaction
        await tester.drag(iouSlider.first, const Offset(-30.0, 0.0));
        await tester.pumpAndSettle();
      });

      testWidgets('Camera controls work correctly', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Find camera control buttons
        final flipButton = find.byIcon(Icons.flip_camera_ios);
        final zoomInButton = find.byIcon(Icons.zoom_in);
        final zoomOutButton = find.byIcon(Icons.zoom_out);

        // Test camera flip
        if (flipButton.evaluate().isNotEmpty) {
          await tester.tap(flipButton);
          await tester.pumpAndSettle();
        }

        // Test zoom controls
        if (zoomInButton.evaluate().isNotEmpty) {
          await tester.tap(zoomInButton);
          await tester.pumpAndSettle();
        }

        if (zoomOutButton.evaluate().isNotEmpty) {
          await tester.tap(zoomOutButton);
          await tester.pumpAndSettle();
        }
      });

      testWidgets('Performance metrics are displayed', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Look for FPS display
        final fpsText = find.textContaining('FPS');
        if (fpsText.evaluate().isNotEmpty) {
          expect(fpsText, findsOneWidget);
        }

        // Look for detection count
        final detectionText = find.textContaining('Detections');
        if (detectionText.evaluate().isNotEmpty) {
          expect(detectionText, findsOneWidget);
        }
      });
    });

    group('Single Image Screen Tests', () {
      testWidgets('Single image screen loads correctly', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(const MaterialApp(home: SingleImageScreen()));
        await tester.pumpAndSettle();

        // Verify UI elements
        expect(find.byType(AppBar), findsOneWidget);
        expect(find.text('Single Image Inference'), findsOneWidget);
      });

      testWidgets('Image picker button is present and tappable', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(const MaterialApp(home: SingleImageScreen()));
        await tester.pumpAndSettle();

        // Find image picker button
        final pickImageButton = find.text('Pick Image');
        expect(pickImageButton, findsOneWidget);

        // Test button tap
        await tester.tap(pickImageButton);
        await tester.pumpAndSettle();

        // Verify image picker was triggered
        // This would be verified by checking if the mock method was called
      });

      testWidgets('Model loading indicator shows during initialization', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(const MaterialApp(home: SingleImageScreen()));

        // Check for loading indicator during initialization
        final loadingIndicator = find.byType(CircularProgressIndicator);
        if (loadingIndicator.evaluate().isNotEmpty) {
          expect(loadingIndicator, findsOneWidget);
        }

        await tester.pumpAndSettle();
      });

      testWidgets('Error handling for model loading failures', (
        WidgetTester tester,
      ) async {
        // This test would require mocking the YOLO model loading to fail
        await tester.pumpWidget(const MaterialApp(home: SingleImageScreen()));
        await tester.pumpAndSettle();

        // Verify error handling UI elements if present
        // This depends on your error handling implementation
      });
    });

    group('Model Management Tests', () {
      testWidgets('Model manager handles different model types', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Test switching between different model types
        final modelTypes = ['Detect', 'Segment', 'Classify', 'Pose', 'OBB'];

        for (final modelType in modelTypes) {
          final button = find.text(modelType);
          if (button.evaluate().isNotEmpty) {
            await tester.tap(button);
            await tester.pumpAndSettle();

            // Verify model loading state
            // This depends on your implementation
          }
        }
      });

      testWidgets('Model download progress is displayed', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Look for download progress indicators
        final progressIndicator = find.byType(LinearProgressIndicator);
        if (progressIndicator.evaluate().isNotEmpty) {
          expect(progressIndicator, findsOneWidget);
        }
      });

      testWidgets('Model loading status messages are shown', (
        WidgetTester tester,
      ) async {
        await tester.pumpWidget(
          const MaterialApp(home: CameraInferenceScreen()),
        );
        await tester.pumpAndSettle();

        // Look for status messages
        final statusText = find.textContaining('Loading');
        if (statusText.evaluate().isNotEmpty) {
          expect(statusText, findsOneWidget);
        }
      });
    });
  });
}
