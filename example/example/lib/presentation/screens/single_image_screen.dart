// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import '../../services/model_manager.dart';
import '../../models/models.dart';

/// A screen that demonstrates YOLO inference on a single image.
///
/// This screen allows users to:
/// - Pick an image from the gallery
/// - Run YOLO inference on the selected image
/// - View detection results and annotated image
class SingleImageScreen extends StatefulWidget {
  const SingleImageScreen({super.key});

  @override
  State<SingleImageScreen> createState() => _SingleImageScreenState();
}

class _SingleImageScreenState extends State<SingleImageScreen> {
  final _picker = ImagePicker();
  List<Map<String, dynamic>> _detections = [];
  Uint8List? _imageBytes;
  Uint8List? _annotatedImage;
  late YOLO _yolo;
  String? _modelPath;
  bool _isModelReady = false;
  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();
    _modelManager = ModelManager();
    _initializeYOLO();
  }

  /// Initializes the YOLO model for inference
  Future<void> _initializeYOLO() async {
    _modelPath = await _modelManager.getModelPath(ModelType.segment);
    if (_modelPath == null) return;
    _yolo = YOLO(modelPath: _modelPath!, task: YOLOTask.segment);
    try {
      await _yolo.loadModel();
      if (mounted) setState(() => _isModelReady = true);
    } catch (e) {
      if (mounted) {
        final error = YOLOErrorHandler.handleError(
          e,
          'Failed to load model $_modelPath for task ${YOLOTask.segment.name}',
        );
        _showSnackBar('Error loading model: ${error.message}');
      }
    }
  }

  /// Picks an image from the gallery and runs inference
  Future<void> _pickAndPredict() async {
    if (!_isModelReady) {
      return _showSnackBar('Model is loading, please wait...');
    }
    final file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    final result = await _yolo.predict(bytes);
    if (mounted) {
      setState(() {
        _detections = result['boxes'] is List
            ? MapConverter.convertBoxesList(result['boxes'] as List)
            : [];
        _annotatedImage = result['annotatedImage'] as Uint8List?;
        _imageBytes = bytes;
      });
    }
  }

  void _showSnackBar(String msg) => mounted
      ? ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)))
      : null;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Single Image Inference'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Column(
        children: [
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _pickAndPredict,
            child: const Text('Pick Image & Run Inference'),
          ),
          const SizedBox(height: 10),
          if (!_isModelReady)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(width: 10),
                  Text(
                    Platform.isIOS
                        ? "Preparing local model..."
                        : "Model loading...",
                  ),
                ],
              ),
            ),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  if (_annotatedImage != null || _imageBytes != null)
                    SizedBox(
                      height: 300,
                      width: double.infinity,
                      child: Image.memory(_annotatedImage ?? _imageBytes!),
                    ),
                  const SizedBox(height: 10),
                  const Text('Detections:'),
                  Text(_detections.toString()),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
