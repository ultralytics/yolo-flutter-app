// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// App launch / smoke test.
//
// Boots the real example application on a device or simulator and verifies it
// reaches its first frame without crashing. Because it runs the full app it
// exercises the native plugin registration and the YOLOView platform view —
// i.e. it catches launch-time native regressions (e.g. a stale storyboard
// module reference) that a host-only widget test cannot.
//
// Run with:  flutter test integration_test/app_launch_test.dart -d <device>

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:ultralytics_yolo_example/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('example app launches without crashing', (
    WidgetTester tester,
  ) async {
    app.main();

    // First frame.
    await tester.pump();
    // Let async launch work (platform view creation, permission requests, model
    // resolution) kick off. Deliberately not pumpAndSettle: the camera screen
    // drives a continuous frame/zoom stream that never settles.
    await tester.pump(const Duration(seconds: 1));
    await tester.pump(const Duration(seconds: 2));

    // The app shell mounted.
    expect(find.byType(MaterialApp), findsOneWidget);
    // The initial route (real-time camera screen) built its scaffold.
    expect(find.byType(Scaffold), findsWidgets);
    // Nothing threw during launch.
    expect(tester.takeException(), isNull);
  });
}
