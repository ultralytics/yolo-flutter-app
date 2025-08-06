// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
<<<<<<< HEAD
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';
=======
import 'use_gpu_example.dart';
>>>>>>> cdd4d63d (fix: analyzer)

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'YOLO useGpu Example',
      home: CameraInferenceScreen(),
    );
  }
}
