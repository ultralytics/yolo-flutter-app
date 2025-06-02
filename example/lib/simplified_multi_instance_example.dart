// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'dart:io';

/// Simplified example demonstrating multiple YOLO instances
class SimplifiedMultiInstanceExample extends StatefulWidget {
  const SimplifiedMultiInstanceExample({Key? key}) : super(key: key);

  @override
  State<SimplifiedMultiInstanceExample> createState() =>
      _SimplifiedMultiInstanceExampleState();
}

class _SimplifiedMultiInstanceExampleState
    extends State<SimplifiedMultiInstanceExample> {
  // Simply create YOLO instances like regular objects
  late YOLO _detector;
  late YOLO _segmenter;
  late YOLO _classifier;

  bool _isLoading = true;
  String _statusMessage = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _initializeModels();
  }

  @override
  void dispose() {
    // Clean up is simple - just call dispose on each instance
    _detector.dispose();
    _segmenter.dispose();
    _classifier.dispose();
    super.dispose();
  }

  Future<void> _initializeModels() async {
    try {
      // Create instances - as simple as creating any object
      _detector = YOLO(
        modelPath: 'assets/models/yolov8n.tflite',
        task: YOLOTask.detect,
      );

      _segmenter = YOLO(
        modelPath: 'assets/models/yolov8n-seg.tflite',
        task: YOLOTask.segment,
      );

      _classifier = YOLO(
        modelPath: 'assets/models/yolov8n-cls.tflite',
        task: YOLOTask.classify,
      );

      setState(() {
        _statusMessage = 'Loading models...';
      });

      // Load models in parallel
      await Future.wait([
        _detector.loadModel(),
        _segmenter.loadModel(),
        _classifier.loadModel(),
      ]);

      setState(() {
        _statusMessage = 'All models loaded!';
        _isLoading = false;
      });

      // You can access instance IDs if needed
      print('Detector ID: ${_detector.instanceId}');
      print('Segmenter ID: ${_segmenter.instanceId}');
      print('Classifier ID: ${_classifier.instanceId}');
    } catch (e) {
      setState(() {
        _statusMessage = 'Error: ${e.toString()}';
        _isLoading = false;
      });
    }
  }

  Future<void> _runInference(String imagePath) async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Running inference...';
    });

    try {
      final imageFile = File(imagePath);
      final imageBytes = await imageFile.readAsBytes();

      // Run inference on all models in parallel
      final results = await Future.wait([
        _detector.predict(imageBytes),
        _segmenter.predict(imageBytes, confidenceThreshold: 0.3),
        _classifier.predict(imageBytes),
      ]);

      setState(() {
        _statusMessage =
            '''
Detection: ${results[0]['boxes']?.length ?? 0} objects found
Segmentation: ${results[1]['boxes']?.length ?? 0} segments found
Classification: Top class ${results[2]['classification']?['topClass'] ?? 'Unknown'}
        ''';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Inference error: ${e.toString()}';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Simplified Multi-Instance Example')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Status:', style: Theme.of(context).textTheme.titleMedium),
            Text(_statusMessage),
            const SizedBox(height: 20),

            if (_isLoading)
              const Center(child: CircularProgressIndicator())
            else
              ElevatedButton(
                onPressed: () => _runInference('path/to/image.jpg'),
                child: const Text('Run Inference'),
              ),
          ],
        ),
      ),
    );
  }
}

/// Even simpler example - functional style
Future<void> functionalExample() async {
  // Create instances inline
  final detector = YOLO(modelPath: 'yolov8n.tflite', task: YOLOTask.detect);
  final segmenter = YOLO(
    modelPath: 'yolov8n-seg.tflite',
    task: YOLOTask.segment,
  );

  // Load models
  await Future.wait([detector.loadModel(), segmenter.loadModel()]);

  // Use them
  final imageBytes = Uint8List.fromList([/* your image data */]);
  final detectResult = await detector.predict(imageBytes);
  final segmentResult = await segmenter.predict(imageBytes);

  print('Objects: ${detectResult['boxes']?.length}');
  print('Segments: ${segmentResult['boxes']?.length}');

  // Clean up
  await detector.dispose();
  await segmenter.dispose();
}

/// Example with dynamic model switching
class DynamicModelExample {
  final Map<String, YOLO> _models = {};

  Future<void> loadModel(String name, String path, YOLOTask task) async {
    // Create and load a new model
    final yolo = YOLO(modelPath: path, task: task);
    await yolo.loadModel();
    _models[name] = yolo;
  }

  Future<Map<String, dynamic>?> runInference(
    String modelName,
    Uint8List image,
  ) async {
    final model = _models[modelName];
    if (model == null) {
      throw Exception('Model $modelName not loaded');
    }
    return await model.predict(image);
  }

  Future<void> dispose() async {
    // Dispose all models
    for (final model in _models.values) {
      await model.dispose();
    }
    _models.clear();
  }
}

/// Example showing instance reuse in a list
class ModelListExample extends StatefulWidget {
  const ModelListExample({Key? key}) : super(key: key);

  @override
  State<ModelListExample> createState() => _ModelListExampleState();
}

class _ModelListExampleState extends State<ModelListExample> {
  final List<YOLO> _models = [];

  void _addModel(String path, YOLOTask task) async {
    final yolo = YOLO(modelPath: path, task: task);
    await yolo.loadModel();
    setState(() {
      _models.add(yolo);
    });
  }

  @override
  void dispose() {
    // Dispose all models
    for (final model in _models) {
      model.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: _models.length,
      itemBuilder: (context, index) {
        final model = _models[index];
        return ListTile(
          title: Text('Model ${index + 1}'),
          subtitle: Text('ID: ${model.instanceId}\nTask: ${model.task.name}'),
          trailing: IconButton(
            icon: const Icon(Icons.delete),
            onPressed: () async {
              await model.dispose();
              setState(() {
                _models.removeAt(index);
              });
            },
          ),
        );
      },
    );
  }
}
