import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Multi-Model',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        useMaterial3: true,
      ),
      home: const MultiModelScreen(),
    );
  }
}

class ModelConfig {
  final String name;
  final String modelPath;
  final YOLOTask task;
  final IconData icon;
  final Color color;

  const ModelConfig({
    required this.name,
    required this.modelPath,
    required this.task,
    required this.icon,
    required this.color,
  });
}

class MultiModelScreen extends StatefulWidget {
  const MultiModelScreen({super.key});

  @override
  State<MultiModelScreen> createState() => _MultiModelScreenState();
}

class _MultiModelScreenState extends State<MultiModelScreen> {
  final ImagePicker _picker = ImagePicker();
  
  // Available models
  final List<ModelConfig> _models = [
    ModelConfig(
      name: 'Object Detection',
      modelPath: 'yolo11n.tflite',
      task: YOLOTask.detect,
      icon: Icons.crop_square,
      color: Colors.blue,
    ),
    ModelConfig(
      name: 'Segmentation',
      modelPath: 'yolo11n-seg.tflite',
      task: YOLOTask.segment,
      icon: Icons.texture,
      color: Colors.green,
    ),
    ModelConfig(
      name: 'Classification',
      modelPath: 'yolo11n-cls.tflite',
      task: YOLOTask.classify,
      icon: Icons.category,
      color: Colors.orange,
    ),
    ModelConfig(
      name: 'Pose Estimation',
      modelPath: 'yolo11n-pose.tflite',
      task: YOLOTask.pose,
      icon: Icons.accessibility_new,
      color: Colors.red,
    ),
    ModelConfig(
      name: 'OBB Detection',
      modelPath: 'yolo11n-obb.tflite',
      task: YOLOTask.obb,
      icon: Icons.rotate_90_degrees_ccw,
      color: Colors.purple,
    ),
  ];

  // State
  ModelConfig? _selectedModel;
  YOLO? _currentYolo;
  File? _imageFile;
  Map<String, dynamic>? _results;
  bool _isLoading = false;
  String _statusMessage = 'Select a model and image to start';
  
  // Performance metrics
  double _loadTime = 0.0;
  double _inferenceTime = 0.0;
  int _modelSize = 0;

  @override
  void dispose() {
    _currentYolo?.dispose();
    super.dispose();
  }

  Future<void> _loadModel(ModelConfig model) async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Loading ${model.name} model...';
      _results = null;
    });

    final stopwatch = Stopwatch()..start();

    try {
      // Dispose previous model
      if (_currentYolo != null) {
        await _currentYolo!.dispose();
        _currentYolo = null;
      }

      // Load new model
      _currentYolo = YOLO(
        modelPath: model.modelPath,
        task: model.task,
      );

      final success = await _currentYolo!.loadModel();
      if (!success) {
        throw Exception('Failed to load model');
      }

      stopwatch.stop();
      _loadTime = stopwatch.elapsedMilliseconds.toDouble();

      setState(() {
        _selectedModel = model;
        _statusMessage = '${model.name} model loaded successfully';
        _modelSize = 0; // Would get from file system in real app
      });

      // Run inference if image is already selected
      if (_imageFile != null) {
        await _runInference();
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Error loading model: $e';
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _results = null;
      });
      
      if (_currentYolo != null) {
        await _runInference();
      }
    }
  }

  Future<void> _runInference() async {
    if (_imageFile == null || _currentYolo == null) return;

    setState(() {
      _isLoading = true;
      _statusMessage = 'Running inference...';
    });

    final stopwatch = Stopwatch()..start();

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final results = await _currentYolo!.predict(imageBytes);
      
      stopwatch.stop();
      _inferenceTime = stopwatch.elapsedMilliseconds.toDouble();

      setState(() {
        _results = results;
        _statusMessage = 'Inference completed';
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error during inference: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildModelSelector() {
    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Select Model',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _models.map((model) {
                final isSelected = _selectedModel == model;
                return ChoiceChip(
                  label: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        model.icon,
                        size: 18,
                        color: isSelected ? Colors.white : model.color,
                      ),
                      const SizedBox(width: 4),
                      Text(model.name),
                    ],
                  ),
                  selected: isSelected,
                  selectedColor: model.color,
                  onSelected: _isLoading ? null : (selected) {
                    if (selected) {
                      _loadModel(model);
                    }
                  },
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPerformanceMetrics() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Performance Metrics',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _MetricTile(
                  icon: Icons.download,
                  label: 'Load Time',
                  value: '${_loadTime.toStringAsFixed(0)}ms',
                  color: Colors.blue,
                ),
                _MetricTile(
                  icon: Icons.speed,
                  label: 'Inference',
                  value: '${_inferenceTime.toStringAsFixed(0)}ms',
                  color: Colors.green,
                ),
                _MetricTile(
                  icon: Icons.memory,
                  label: 'Model Size',
                  value: _modelSize > 0 ? '${(_modelSize / 1024 / 1024).toStringAsFixed(1)}MB' : 'N/A',
                  color: Colors.orange,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsView() {
    if (_results == null) return const SizedBox();

    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Results - ${_selectedModel?.name ?? 'Unknown'}',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            _buildTaskSpecificResults(),
          ],
        ),
      ),
    );
  }

  Widget _buildTaskSpecificResults() {
    if (_results == null || _selectedModel == null) return const SizedBox();

    switch (_selectedModel!.task) {
      case YOLOTask.detect:
      case YOLOTask.obb:
        final detections = _results!['detections'] as List<dynamic>? ?? [];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detected ${detections.length} objects'),
            const SizedBox(height: 8),
            ...detections.take(5).map((detection) {
              return ListTile(
                dense: true,
                leading: CircleAvatar(
                  backgroundColor: _selectedModel!.color,
                  radius: 16,
                  child: const Icon(Icons.check, color: Colors.white, size: 16),
                ),
                title: Text(detection['className'] ?? 'Unknown'),
                subtitle: Text(
                  'Confidence: ${((detection['confidence'] ?? 0) * 100).toStringAsFixed(1)}%',
                ),
              );
            }).toList(),
            if (detections.length > 5)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  '... and ${detections.length - 5} more',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
          ],
        );

      case YOLOTask.segment:
        final detections = _results!['detections'] as List<dynamic>? ?? [];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Found ${detections.length} instance masks'),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: detections.take(10).map((detection) {
                return Chip(
                  avatar: CircleAvatar(
                    backgroundColor: _selectedModel!.color,
                    child: const Icon(Icons.texture, color: Colors.white, size: 16),
                  ),
                  label: Text(detection['className'] ?? 'Unknown'),
                );
              }).toList(),
            ),
          ],
        );

      case YOLOTask.classify:
        final classification = _results!['classification'] as Map<dynamic, dynamic>?;
        if (classification == null) return const Text('No classification results');
        
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ListTile(
              dense: true,
              leading: CircleAvatar(
                backgroundColor: _selectedModel!.color,
                child: const Text('1', style: TextStyle(color: Colors.white)),
              ),
              title: Text(classification['topClass'] ?? 'Unknown'),
              subtitle: Text(
                'Confidence: ${((classification['topConfidence'] ?? 0) * 100).toStringAsFixed(1)}%',
              ),
            ),
            if (classification['top5Classes'] != null)
              ...List.generate(4, (i) {
                final classes = classification['top5Classes'] as List<dynamic>;
                final confidences = classification['top5Confidences'] as List<dynamic>;
                if (i + 1 >= classes.length) return const SizedBox();
                return ListTile(
                  dense: true,
                  leading: CircleAvatar(
                    backgroundColor: Colors.grey,
                    radius: 14,
                    child: Text('${i + 2}', style: const TextStyle(color: Colors.white, fontSize: 12)),
                  ),
                  title: Text(classes[i + 1]),
                  subtitle: Text('${(confidences[i + 1] * 100).toStringAsFixed(1)}%'),
                );
              }),
          ],
        );

      case YOLOTask.pose:
        final detections = _results!['detections'] as List<dynamic>? ?? [];
        final peopleWithPose = detections.where((d) => d['keypoints'] != null).length;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detected $peopleWithPose person(s) with pose'),
            const SizedBox(height: 8),
            if (peopleWithPose > 0)
              ...detections.take(3).where((d) => d['keypoints'] != null).map((detection) {
                final keypoints = detection['keypoints'] as List<dynamic>;
                return ListTile(
                  dense: true,
                  leading: CircleAvatar(
                    backgroundColor: _selectedModel!.color,
                    child: const Icon(Icons.accessibility_new, color: Colors.white, size: 16),
                  ),
                  title: const Text('Person'),
                  subtitle: Text('${keypoints.length} keypoints detected'),
                );
              }).toList(),
          ],
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Multi-Model Demo'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Status message
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: _isLoading ? Colors.orange.shade100 : Colors.green.shade100,
              child: Row(
                children: [
                  if (_isLoading)
                    const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  if (_isLoading) const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _statusMessage,
                      style: TextStyle(
                        color: _isLoading ? Colors.orange.shade900 : Colors.green.shade900,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Model selector
            _buildModelSelector(),

            // Performance metrics
            if (_selectedModel != null) _buildPerformanceMetrics(),

            // Image preview
            if (_imageFile != null)
              Container(
                margin: const EdgeInsets.all(8),
                height: 200,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.file(
                    _imageFile!,
                    fit: BoxFit.cover,
                    width: double.infinity,
                  ),
                ),
              ),

            // Results
            if (_results != null) _buildResultsView(),

            const SizedBox(height: 80),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isLoading ? null : _pickImage,
        icon: const Icon(Icons.image),
        label: const Text('Select Image'),
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _MetricTile({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: color, size: 24),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
        Text(
          value,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }
}