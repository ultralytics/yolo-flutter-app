// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Demonstrates YOLO inference on a single gallery image.
class SingleImageScreen extends StatefulWidget {
  const SingleImageScreen({super.key});

  @override
  State<SingleImageScreen> createState() => _SingleImageScreenState();
}

class _SingleImageScreenState extends State<SingleImageScreen> {
  final _picker = ImagePicker();
  final _yolo = YOLO(modelPath: 'yolo26n');

  List<YOLOResult> _detections = const [];
  Uint8List? _imageBytes;
  Uint8List? _annotatedImage;
  bool _isModelReady = false;

  @override
  void initState() {
    super.initState();
    _initializeYOLO();
  }

  Future<void> _initializeYOLO() async {
    try {
      await _yolo.loadModel();
      if (mounted) setState(() => _isModelReady = true);
    } catch (e) {
      if (mounted) _showSnackBar('Error loading model: $e');
    }
  }

  Future<void> _pickAndPredict() async {
    if (!_isModelReady) {
      _showSnackBar('Model is loading, please wait...');
      return;
    }
    final file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    final result = await _yolo.predict(bytes);
    if (!mounted) return;
    final detections = (result['detections'] as List?)
        ?.whereType<Map>()
        .map(YOLOResult.fromMap)
        .toList(growable: false);
    setState(() {
      _detections = detections ?? const [];
      _annotatedImage = result['annotatedImage'] as Uint8List?;
      _imageBytes = bytes;
    });
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _yolo.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Single Image Inference')),
      body: Column(
        children: [
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _pickAndPredict,
            child: const Text('Pick Image & Run Inference'),
          ),
          const SizedBox(height: 10),
          if (!_isModelReady)
            const Padding(
              padding: EdgeInsets.all(8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 10),
                  Text('Preparing YOLO26 model...'),
                ],
              ),
            ),
          Expanded(
            child: ListView(
              children: [
                if (_annotatedImage != null || _imageBytes != null)
                  SizedBox(
                    height: 300,
                    width: double.infinity,
                    child: Image.memory(_annotatedImage ?? _imageBytes!),
                  ),
                const SizedBox(height: 10),
                if (_detections.isNotEmpty)
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 16),
                    child: Text(
                      'Detections',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                for (final d in _detections)
                  ListTile(
                    dense: true,
                    title: Text(d.className),
                    trailing: Text(
                      '${(d.confidence * 100).toStringAsFixed(1)}%',
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
