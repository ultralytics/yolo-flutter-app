// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Basic example of using ultralytics_yolo plugin for real-time object detection.
void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('YOLO Object Detection')),
        body: const ObjectDetectionView(),
      ),
    );
  }
}

class ObjectDetectionView extends StatelessWidget {
  const ObjectDetectionView({super.key});

  @override
  Widget build(BuildContext context) {
    return YOLOView(
      modelPath: 'yolo11n',  // Model file: assets/models/yolo11n.tflite (Android) or yolo11n.mlmodel (iOS)
      task: YOLOTask.detect,
      onResult: (List<YOLOResult> results) {
        // Process detection results
        for (final result in results) {
          debugPrint('Detected ${result.className} with ${(result.confidence * 100).toStringAsFixed(0)}% confidence');
        }
      },
    );
  }
}