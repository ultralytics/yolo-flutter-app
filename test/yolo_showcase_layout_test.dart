// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:ultralytics_yolo/widgets/camera_toolbar.dart';
import 'package:ultralytics_yolo/widgets/yolo_showcase.dart';

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
}
