// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'use_gpu_example.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO useGpu Example',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const UseGpuExample(),
    );
  }
}
