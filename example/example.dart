// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: 'YOLO Example', home: YOLOScreen());
  }
}

class YOLOScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('YOLO Detection')),
      body: YOLOView(
        modelPath: 'yolo11n',
        task: YOLOTask.detect,
        onResult: (results) {
          for (final result in results) {
            print('${result.className}: ${result.confidence}');
          }
        },
      ),
    );
  }
}

// Single image inference example
class SingleImageExample extends StatefulWidget {
  @override
  _SingleImageExampleState createState() => _SingleImageExampleState();
}

class _SingleImageExampleState extends State<SingleImageExample> {
  YOLO? _yolo;

  @override
  void initState() {
    super.initState();
    _initializeYOLO();
  }

  Future<void> _initializeYOLO() async {
    _yolo = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);
    await _yolo!.loadModel();
  }

  @override
  Widget build(BuildContext context) {
    return Container(); // Simplified for brevity
  }
}
