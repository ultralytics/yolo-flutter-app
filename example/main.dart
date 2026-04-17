// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) =>
      const MaterialApp(title: 'YOLO Example', home: YOLOScreen());
}

class YOLOScreen extends StatelessWidget {
  const YOLOScreen({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('YOLO Detection')),
    body: YOLOView(
      modelPath: 'yolo26n',
      onResult: (results) {
        for (final r in results) {
          debugPrint('${r.className}: ${r.confidence}');
        }
      },
    ),
  );
}
