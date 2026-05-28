// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Mirrors the iOS `YOLOSingleImageUIKit` example: pick a gallery image, run
/// `yolo26n` detection, and list the resulting labels + confidences over the
/// annotated image returned by the native side.
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
  bool _isInferring = false;

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
    if (mounted) setState(() => _isInferring = true);
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
      _isInferring = false;
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
          Padding(
            padding: const EdgeInsets.all(12),
            child: FilledButton.icon(
              onPressed: (_isModelReady && !_isInferring)
                  ? _pickAndPredict
                  : null,
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('Pick image & run inference'),
            ),
          ),
          if (!_isModelReady)
            const Padding(
              padding: EdgeInsets.all(8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  SizedBox(width: 12),
                  Text('Preparing YOLO26 model...'),
                ],
              ),
            ),
          if (_isInferring) const LinearProgressIndicator(minHeight: 2),
          Expanded(
            child: ListView(
              children: [
                if (_annotatedImage != null || _imageBytes != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.memory(
                        _annotatedImage ?? _imageBytes!,
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                const SizedBox(height: 12),
                if (_detections.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Text(
                      'Detections (${_detections.length})',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                  ),
                for (final d in _detections)
                  ListTile(
                    dense: true,
                    leading: const Icon(Icons.crop_din),
                    title: Text(d.className),
                    trailing: Text(
                      '${(d.confidence * 100).toStringAsFixed(1)}%',
                      style: Theme.of(context).textTheme.bodyMedium,
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
