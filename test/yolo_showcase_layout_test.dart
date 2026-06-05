// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/widgets/camera_toolbar.dart';
import 'package:ultralytics_yolo/widgets/lens_picker.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import 'package:ultralytics_yolo/widgets/yolo_showcase.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_view.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('landscape toolbar ignores side safe-area insets', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues({});
    await tester.binding.setSurfaceSize(const Size(900, 430));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await tester.pumpWidget(
      const MaterialApp(
        home: MediaQuery(
          data: MediaQueryData(
            size: Size(900, 430),
            padding: EdgeInsets.only(right: 80, bottom: 24),
          ),
          child: YOLOShowcase(),
        ),
      ),
    );

    final toolbarRect = tester.getRect(find.byType(CameraToolbar));

    expect(toolbarRect.left, 0);
    expect(toolbarRect.width, 900);
  });

  testWidgets('showcase controls drive the camera controller', (tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.binding.setSurfaceSize(const Size(390, 780));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final calls = <MethodCall>[];
    const channel = MethodChannel('showcase_controls_test');
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
          calls.add(call);
          return switch (call.method) {
            'capturePhoto' => Uint8List.fromList([1, 2, 3]),
            'inspectModel' => {
              'path': call.arguments['modelPath'],
              'task': 'detect',
              'labels': ['person'],
            },
            'getAvailableLenses' => [
              {'zoomFactor': 0.5, 'label': 'Ultra wide'},
              {'zoomFactor': 1, 'label': 'Wide'},
            ],
            _ => true,
          };
        });
    addTearDown(() {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, null);
    });

    final captured = <Uint8List>[];
    final controller = YOLOViewController()..init(channel, 1);
    await tester.pumpWidget(
      MaterialApp(
        home: YOLOShowcase(
          controller: controller,
          showSemanticTask: false,
          versionLabel: 'v1.2.3',
          onCapture: captured.add,
        ),
      ),
    );
    await tester.pump();

    expect(find.text('Sem'), findsNothing);

    tester
        .widget<YOLOView>(find.byType(YOLOView))
        .onModelLoad
        ?.call('yolo26n', YOLOTask.detect);
    await tester.pump();

    await tester.tap(find.bySemanticsLabel('Pause'));
    await tester.pump();
    await tester.tap(find.bySemanticsLabel('Resume'));
    await tester.pump();
    await tester.tap(find.bySemanticsLabel('Switch camera'));
    await tester.pump(const Duration(milliseconds: 300));
    controller.onNativeEvent({'type': 'zoom', 'value': 1.5});
    controller.onNativeEvent({'type': 'lens', 'label': 'Wide'});
    await tester.pump();

    expect(find.text('1.50x'), findsOneWidget);
    expect(find.text('Wide'), findsOneWidget);

    await tester.tap(find.bySemanticsLabel('Turn torch on'));
    await tester.pump();
    await tester.tap(find.text('0.5'));
    final sliders = tester
        .widgetList<CupertinoSlider>(find.byType(CupertinoSlider))
        .toList();
    sliders[0].onChanged?.call(0.55);
    sliders[1].onChanged?.call(0.45);
    await tester.pump();

    await tester.tap(find.bySemanticsLabel('Share'));
    await tester.tap(find.byIcon(CupertinoIcons.info));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));

    expect(find.byIcon(Icons.menu_book_outlined), findsOneWidget);

    Navigator.of(tester.element(find.byIcon(Icons.menu_book_outlined))).pop();
    await tester.pump(const Duration(seconds: 1));
    await tester.tap(find.text('↓ YOLO26s'));
    await tester.pump();

    expect(find.textContaining('Downloading YOLO26s Detect'), findsOneWidget);

    await tester.tap(find.widgetWithText(TextButton, 'Cancel'));
    await tester.pump();
    await tester.tap(find.text('Seg'));
    await tester.pump();

    expect(captured.single, orderedEquals([1, 2, 3]));
    expect(
      calls.map((call) => call.method),
      containsAll([
        'setShowOverlays',
        'pause',
        'resume',
        'switchCamera',
        'setTorchMode',
        'setLens',
        'setConfidenceThreshold',
        'setIoUThreshold',
        'capturePhoto',
      ]),
    );
    expect(find.textContaining('Loading YOLO26n Segment'), findsOneWidget);
  });

  testWidgets('lens picker handles empty, single, and segmented lens sets', (
    tester,
  ) async {
    LensInfo? picked;

    Widget picker(List<LensInfo> lenses, {double zoom = 1}) {
      return MaterialApp(
        home: LensPicker(
          lenses: lenses,
          currentZoomFactor: zoom,
          onLensSelected: (lens) => picked = lens,
          trailing: const Text('Torch'),
        ),
      );
    }

    await tester.pumpWidget(picker(const []));
    expect(find.text('Torch'), findsOneWidget);

    await tester.pumpWidget(
      picker(const [LensInfo(zoomFactor: 0.5, label: 'Ultra wide')]),
    );
    await tester.tap(find.text('0.5'));
    expect(picked?.zoomFactor, 0.5);

    await tester.pumpWidget(
      picker(const [
        LensInfo(zoomFactor: 0.5, label: 'Ultra wide'),
        LensInfo(zoomFactor: 1, label: 'Wide'),
        LensInfo(zoomFactor: 2, label: 'Telephoto'),
      ], zoom: 1.5),
    );
    await tester.tap(find.text('2'));

    expect(picked?.zoomFactor, 2);
    expect(find.text('0.5'), findsOneWidget);
    expect(find.text('1'), findsOneWidget);
    expect(find.text('2'), findsOneWidget);
  });
}
