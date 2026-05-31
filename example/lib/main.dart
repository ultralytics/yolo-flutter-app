// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';
import 'package:ultralytics_yolo_example/presentation/screens/single_image_screen.dart';

void main() {
  runApp(const YOLOExampleApp());
}

class YOLOExampleApp extends StatelessWidget {
  const YOLOExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ultralytics YOLO',
      themeMode: ThemeMode.dark,
      theme: ThemeData.dark(useMaterial3: true),
      initialRoute: '/',
      routes: {
        '/': (_) => const CameraInferenceScreen(),
        '/single': (_) => const SingleImageScreen(),
      },
    );
  }
}
