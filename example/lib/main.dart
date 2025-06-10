// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// example/lib/main.dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';

void main() {
  runApp(const YOLOExampleApp());
}

class YOLOExampleApp extends StatelessWidget {
  const YOLOExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'YOLO Plugin Example',
      home: CameraInferenceScreen(),
    );
  }
}
