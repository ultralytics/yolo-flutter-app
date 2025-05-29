import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

/// Minimal camera view example using YOLOView widget
class CameraViewExample extends StatefulWidget {
  const CameraViewExample({super.key});

  @override
  State<CameraViewExample> createState() => _CameraViewExampleState();
}

class _CameraViewExampleState extends State<CameraViewExample> {
  // Model path:
  // - iOS: Must be bundled in Xcode project (e.g., 'yolo11n')
  // - Android: Can use Flutter assets (e.g., 'assets/models/yolo11n.tflite')
  static const String modelPath = 'assets/models/yolo11n.tflite';
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Camera View Example')),
      body: YOLOView(
        modelPath: modelPath,
        task: YOLOTask.detect, // Object detection task
        onResult: (results) {
          // Results callback - called for each frame
          // results is List<YOLOResult>
          debugPrint('Detected ${results.length} objects');
        },
      ),
    );
  }
}