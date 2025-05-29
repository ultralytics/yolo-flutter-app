// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/yolo_task.dart';

/// Minimal single image inference example
class SingleImageExample extends StatefulWidget {
  const SingleImageExample({super.key});

  @override
  State<SingleImageExample> createState() => _SingleImageExampleState();
}

class _SingleImageExampleState extends State<SingleImageExample> {
  // Model path:
  // - iOS: Must be bundled in Xcode project (e.g., 'yolo11n')
  // - Android: Can use Flutter assets (e.g., 'assets/models/yolo11n.tflite')
  static const String modelPath = 'assets/models/yolo11n.tflite';

  final ImagePicker _picker = ImagePicker();
  late YOLO _yolo;
  Uint8List? _imageBytes;
  String _results = '';
  bool _isModelLoaded = false;

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    _yolo = YOLO(modelPath: modelPath, task: YOLOTask.detect);

    try {
      await _yolo.loadModel();
      setState(() => _isModelLoaded = true);
    } catch (e) {
      debugPrint('Error loading model: $e');
    }
  }

  Future<void> _pickAndAnalyzeImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    final bytes = await image.readAsBytes();
    setState(() => _imageBytes = bytes);

    // Run inference
    final result = await _yolo.predict(bytes);

    // Display results
    setState(() {
      if (result['boxes'] != null) {
        final boxes = result['boxes'] as List;
        _results = 'Found ${boxes.length} objects';
      } else {
        _results = 'No objects detected';
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Single Image Example')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (!_isModelLoaded)
              const CircularProgressIndicator()
            else
              ElevatedButton(
                onPressed: _pickAndAnalyzeImage,
                child: const Text('Pick Image'),
              ),
            const SizedBox(height: 20),
            if (_imageBytes != null) Image.memory(_imageBytes!, height: 300),
            const SizedBox(height: 20),
            Text(_results),
          ],
        ),
      ),
    );
  }
}
