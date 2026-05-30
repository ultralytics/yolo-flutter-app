// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/widgets/model_loading_status.dart';

void main() {
  testWidgets('shows download progress without blocking gestures', (
    tester,
  ) async {
    var tapped = false;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Stack(
            children: [
              TextButton(
                onPressed: () => tapped = true,
                child: const Text('Change model'),
              ),
              const ModelLoadingStatus(
                statusText: 'Downloading 42%',
                progress: 0.42,
              ),
            ],
          ),
        ),
      ),
    );

    expect(find.text('Downloading 42%'), findsOneWidget);
    final progress = tester.widget<LinearProgressIndicator>(
      find.byType(LinearProgressIndicator),
    );
    expect(progress.value, 0.42);

    await tester.tap(find.text('Change model'));
    expect(tapped, isTrue);
  });

  testWidgets('renders errors without a progress bar', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: ModelLoadingStatus(
            errorMessage: 'Model download failed',
            progress: 0.5,
          ),
        ),
      ),
    );

    expect(find.text('Model download failed'), findsOneWidget);
    expect(find.byType(LinearProgressIndicator), findsNothing);
  });
}
