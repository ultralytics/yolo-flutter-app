// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_result.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YoloView Simple Tests', () {
    testWidgets('creates widget with required parameters', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('creates widget with all parameters', (tester) async {
      void onResult(List<YOLOResult> results) {}
      void onMetrics(Map<String, double> metrics) {}
      final controller = YoloViewController();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.segment,
            controller: controller,
            cameraResolution: '1080p',
            onResult: onResult,
            onPerformanceMetrics: onMetrics,
            showNativeUI: true,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles Android platform', (tester) async {
      debugDefaultTargetPlatformOverride = TargetPlatform.android;

      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(find.byType(AndroidView), findsOneWidget);

      debugDefaultTargetPlatformOverride = null;
    });

    testWidgets('handles iOS platform', (tester) async {
      debugDefaultTargetPlatformOverride = TargetPlatform.iOS;

      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(find.byType(UiKitView), findsOneWidget);

      debugDefaultTargetPlatformOverride = null;
    });

    testWidgets('handles unsupported platform', (tester) async {
      debugDefaultTargetPlatformOverride = TargetPlatform.linux;

      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(find.text('Platform not supported for YoloView'), findsOneWidget);

      debugDefaultTargetPlatformOverride = null;
    });

    testWidgets('disposes properly', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      await tester.pumpWidget(const MaterialApp(home: SizedBox()));

      // Widget should be disposed without error
      expect(find.byType(YoloView), findsNothing);
    });

    testWidgets('state can be accessed via GlobalKey', (tester) async {
      final key = GlobalKey<YoloViewState>();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            key: key,
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(key.currentState, isNotNull);
      expect(key.currentState, isA<YoloViewState>());
    });

    testWidgets('handles different task types', (tester) async {
      for (final task in YOLOTask.values) {
        await tester.pumpWidget(
          MaterialApp(
            home: YoloView(
              modelPath: 'test.tflite',
              task: task,
            ),
          ),
        );

        expect(find.byType(YoloView), findsOneWidget);

        await tester.pumpWidget(const SizedBox());
      }
    });

    testWidgets('updates widget when properties change', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test1.tflite',
            task: YOLOTask.detect,
            showNativeUI: false,
          ),
        ),
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test2.tflite',
            task: YOLOTask.detect,
            showNativeUI: true,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles controller updates', (tester) async {
      final controller1 = YoloViewController();
      final controller2 = YoloViewController();

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            controller: controller1,
          ),
        ),
      );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            controller: controller2,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles callback updates', (tester) async {
      void onResult1(List<YOLOResult> results) {}
      void onResult2(List<YOLOResult> results) {}

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: onResult1,
          ),
        ),
      );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: onResult2,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles null callbacks to non-null', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: (results) {},
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });

    testWidgets('handles non-null callbacks to null', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
            onResult: (results) {},
            onPerformanceMetrics: (metrics) {},
          ),
        ),
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: YoloView(
            modelPath: 'test.tflite',
            task: YOLOTask.detect,
          ),
        ),
      );

      expect(find.byType(YoloView), findsOneWidget);
    });
  });
}
