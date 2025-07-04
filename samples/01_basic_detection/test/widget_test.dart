// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:basic_detection/main.dart';

void main() {
  testWidgets('App launches correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the app launches with expected title
    expect(find.text('YOLO Basic Detection'), findsOneWidget);

    // Verify that the pick image button exists
    expect(find.byIcon(Icons.photo_library), findsOneWidget);
  });
}
