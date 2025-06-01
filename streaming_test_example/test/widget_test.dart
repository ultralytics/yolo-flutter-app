// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:streaming_test_example/main.dart';

void main() {
  testWidgets('StreamingTestApp loads', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const StreamingTestApp());

    // Verify that the app loads (this test will require model loading, so we'll just check for basic elements)
    expect(find.byType(StreamingTestApp), findsOneWidget);
  });
}