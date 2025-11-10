import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_model_spec.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/widgets/yolo_overlay.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

class _FakeYOLOViewController extends YOLOViewController {
  bool calledSetConfidence = false;
  bool calledSetIoU = false;
  bool calledSetNumItems = false;
  bool calledSetThresholds = false;
  bool calledSwitchCamera = false;
  bool calledSetZoomLevel = false;
  bool calledSetShowOverlays = false;

  double? confidenceValue;
  double? iouValue;
  int? numItemsValue;
  double? zoomValue;
  bool? showOverlaysValue;

  double? thresholdsConf;
  double? thresholdsIou;
  int? thresholdsNumItems;

  @override
  Future<void> setConfidenceThreshold(double threshold) async {
    calledSetConfidence = true;
    confidenceValue = threshold;
  }

  @override
  Future<void> setIoUThreshold(double threshold) async {
    calledSetIoU = true;
    iouValue = threshold;
  }

  @override
  Future<void> setNumItemsThreshold(int numItems) async {
    calledSetNumItems = true;
    numItemsValue = numItems;
  }

  @override
  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
    int? numItemsThreshold,
  }) async {
    calledSetThresholds = true;
    thresholdsConf = confidenceThreshold;
    thresholdsIou = iouThreshold;
    thresholdsNumItems = numItemsThreshold;
  }

  @override
  Future<void> switchCamera() async {
    calledSwitchCamera = true;
  }

  @override
  Future<void> setZoomLevel(double zoomLevel) async {
    calledSetZoomLevel = true;
    zoomValue = zoomLevel;
  }

  @override
  Future<void> setShowOverlays(bool show) async {
    calledSetShowOverlays = true;
    showOverlaysValue = show;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('YOLOOverlay interactions', () {
    testWidgets('invokes onDetectionTap when tapped inside a detection box', (
      WidgetTester tester,
    ) async {
      // Arrange: one detection with a known bounding box in a 200x200 overlay
      final detection = YOLOResult(
        classIndex: 0,
        className: 'person',
        confidence: 0.9,
        boundingBox: const Rect.fromLTWH(20, 30, 80, 60), // x:20..100, y:30..90
        normalizedBox: const Rect.fromLTWH(0.1, 0.15, 0.4, 0.3),
        modelName: 'test',
      );

      YOLOResult? tapped;
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Center(
              child: SizedBox(
                width: 200,
                height: 200,
                child: YOLOOverlay(
                  detections: [detection],
                  onDetectionTap: (d) => tapped = d,
                  showClassName: true,
                  showConfidence: true,
                ),
              ),
            ),
          ),
        ),
      );

      // Act: tap near the center of the bounding box (x=60, y=60)
      final overlayFinder = find.byType(YOLOOverlay);
      final overlayBox = tester.renderObject<RenderBox>(overlayFinder);
      final overlayTopLeft = overlayBox.localToGlobal(Offset.zero);

      final tapPoint =
          overlayTopLeft + const Offset(20 + 40 /*half width*/, 30 + 30);
      await tester.tapAt(tapPoint);
      await tester.pumpAndSettle();

      // Assert: callback invoked with our detection
      expect(tapped, isNotNull);
      expect(tapped!.className, 'person');
    });

    testWidgets('does not invoke onDetectionTap when tapped outside', (
      WidgetTester tester,
    ) async {
      final detection = YOLOResult(
        classIndex: 0,
        className: 'car',
        confidence: 0.8,
        boundingBox: const Rect.fromLTWH(20, 30, 80, 60),
        normalizedBox: const Rect.fromLTWH(0.1, 0.15, 0.4, 0.3),
        modelName: 'test',
      );

      YOLOResult? tapped;
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Center(
              child: SizedBox(
                width: 200,
                height: 200,
                child: YOLOOverlay(
                  detections: [detection],
                  onDetectionTap: (d) => tapped = d,
                  showClassName: true,
                  showConfidence: true,
                ),
              ),
            ),
          ),
        ),
      );

      // Tap outside bounding box
      final overlayFinder = find.byType(YOLOOverlay);
      final overlayBox = tester.renderObject<RenderBox>(overlayFinder);
      final overlayTopLeft = overlayBox.localToGlobal(Offset.zero);

      final tapOutside = overlayTopLeft + const Offset(150, 150);
      await tester.tapAt(tapOutside);
      await tester.pumpAndSettle();

      expect(tapped, isNull);
    });
  });

  group('YOLOView public methods bridge to controller', () {
    testWidgets(
      'bridges thresholds, camera and overlay methods to controller',
      (WidgetTester tester) async {
        final fake = _FakeYOLOViewController();
        final key = GlobalKey();

        await tester.pumpWidget(
          MaterialApp(
            home: YOLOView(
              key: key,
              controller: fake,
              models: const [
                YOLOModelSpec(
                  modelPath: 'assets/models/yolo11n.tflite',
                  task: YOLOTask.detect,
                ),
              ],
            ),
          ),
        );

        // Access the private state via GlobalKey and dynamic
        final state = key.currentState as dynamic;

        // Call setConfidenceThreshold
        await state.setConfidenceThreshold(0.77);
        expect(fake.calledSetConfidence, isTrue);
        expect(fake.confidenceValue, 0.77);

        // Call setIoUThreshold
        await state.setIoUThreshold(0.66);
        expect(fake.calledSetIoU, isTrue);
        expect(fake.iouValue, 0.66);

        // Call setNumItemsThreshold
        await state.setNumItemsThreshold(23);
        expect(fake.calledSetNumItems, isTrue);
        expect(fake.numItemsValue, 23);

        // Call setThresholds
        await state.setThresholds(
          confidenceThreshold: 0.55,
          iouThreshold: 0.44,
          numItemsThreshold: 12,
        );
        expect(fake.calledSetThresholds, isTrue);
        expect(fake.thresholdsConf, 0.55);
        expect(fake.thresholdsIou, 0.44);
        expect(fake.thresholdsNumItems, 12);

        // Call switchCamera
        await state.switchCamera();
        expect(fake.calledSwitchCamera, isTrue);

        // Call setZoomLevel
        await state.setZoomLevel(2.25);
        expect(fake.calledSetZoomLevel, isTrue);
        expect(fake.zoomValue, 2.25);

        // Call setShowOverlays
        await state.setShowOverlays(false);
        expect(fake.calledSetShowOverlays, isTrue);
        expect(fake.showOverlaysValue, isFalse);
      },
    );
  });
}
