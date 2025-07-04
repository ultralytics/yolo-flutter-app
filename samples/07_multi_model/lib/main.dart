// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
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
      title: 'YOLO Multi-Instance',
      theme: ThemeData(primarySwatch: Colors.deepPurple, useMaterial3: true),
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

  // State - Multiple YOLO instances
  final Map<ModelConfig, YOLO> _loadedModels = {};
  final Set<ModelConfig> _activeModels = {};
  File? _imageFile;
  ui.Image? _uiImage;
  final Map<ModelConfig, Map<String, dynamic>> _allResults = {};
  final Map<ModelConfig, double> _loadTimes = {};
  final Map<ModelConfig, double> _inferenceTimes = {};
  bool _isLoading = false;
  String _statusMessage = 'Select models and an image to start';
  bool _showVisualization = true;
  bool _showLabels = true;
  bool _showConfidence = true;

  @override
  void dispose() {
    // Dispose all loaded models
    for (final yolo in _loadedModels.values) {
      yolo.dispose();
    }
    super.dispose();
  }

  Future<void> _toggleModel(ModelConfig model) async {
    if (_activeModels.contains(model)) {
      // Deactivate model
      setState(() {
        _activeModels.remove(model);
        _allResults.remove(model);
      });
    } else {
      // Activate model
      setState(() {
        _isLoading = true;
        _statusMessage = 'Loading ${model.name} model...';
      });

      try {
        // Load model if not already loaded
        if (!_loadedModels.containsKey(model)) {
          final stopwatch = Stopwatch()..start();

          final yolo = YOLO(
            modelPath: model.modelPath,
            task: model.task,
            useMultiInstance: true,
          );

          final success = await yolo.loadModel();
          if (!success) {
            throw Exception('Failed to load model');
          }

          stopwatch.stop();
          _loadedModels[model] = yolo;
          _loadTimes[model] = stopwatch.elapsedMilliseconds.toDouble();
        }

        setState(() {
          _activeModels.add(model);
          _statusMessage = '${model.name} model activated';
        });

        // Run inference if image is already selected
        if (_imageFile != null) {
          await _runInferenceForModel(model);
        }
      } catch (e) {
        setState(() {
          _statusMessage = 'Error loading model: $e';
        });
        if (mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('Error: $e')));
        }
      } finally {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _allResults.clear();
      });

      // Load image for visualization
      await _loadImageForVisualization();

      if (_activeModels.isNotEmpty) {
        await _runInferenceOnAllModels();
      }
    }
  }

  Future<void> _loadImageForVisualization() async {
    if (_imageFile == null) return;

    final imageBytes = await _imageFile!.readAsBytes();
    final codec = await ui.instantiateImageCodec(imageBytes);
    final frame = await codec.getNextFrame();
    setState(() {
      _uiImage = frame.image;
    });
  }

  Future<void> _runInferenceOnAllModels() async {
    if (_imageFile == null || _activeModels.isEmpty) return;

    setState(() {
      _isLoading = true;
      _statusMessage = 'Running inference on ${_activeModels.length} models...';
    });

    final imageBytes = await _imageFile!.readAsBytes();

    // Run inference on all active models
    for (final model in _activeModels) {
      await _runInferenceForModel(model, imageBytes: imageBytes);
    }

    setState(() {
      _isLoading = false;
      _statusMessage = 'All inferences completed';
    });
  }

  Future<void> _runInferenceForModel(
    ModelConfig model, {
    Uint8List? imageBytes,
  }) async {
    try {
      final bytes = imageBytes ?? await _imageFile!.readAsBytes();
      final yolo = _loadedModels[model];
      if (yolo == null) return;

      final stopwatch = Stopwatch()..start();
      final results = await yolo.predict(bytes);
      stopwatch.stop();

      setState(() {
        _allResults[model] = results;
        _inferenceTimes[model] = stopwatch.elapsedMilliseconds.toDouble();
      });
    } catch (e) {
      // Error during inference for ${model.name}: $e
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
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Active Models',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  '${_activeModels.length} / ${_models.length}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _models.map((model) {
                final isActive = _activeModels.contains(model);
                final isLoaded = _loadedModels.containsKey(model);
                return FilterChip(
                  label: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        model.icon,
                        size: 18,
                        color: isActive ? Colors.white : model.color,
                      ),
                      const SizedBox(width: 4),
                      Text(model.name),
                      if (isLoaded && !isActive)
                        const Icon(
                          Icons.check_circle,
                          size: 14,
                          color: Colors.green,
                        ),
                    ],
                  ),
                  selected: isActive,
                  selectedColor: model.color,
                  checkmarkColor: Colors.white,
                  onSelected: _isLoading
                      ? null
                      : (selected) {
                          _toggleModel(model);
                        },
                );
              }).toList(),
            ),
            if (_loadedModels.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  'Loaded models: ${_loadedModels.length}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildVisualizationControls() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 8),
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Row(
          children: [
            Text(
              'Visualization:',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(width: 8),
            ChoiceChip(
              label: const Text('Show'),
              selected: _showVisualization,
              onSelected: (value) => setState(() => _showVisualization = value),
              selectedColor: Colors.green,
            ),
            const SizedBox(width: 8),
            if (_showVisualization) ...[
              FilterChip(
                label: const Text('Labels'),
                selected: _showLabels,
                onSelected: (value) => setState(() => _showLabels = value),
              ),
              const SizedBox(width: 8),
              FilterChip(
                label: const Text('Confidence'),
                selected: _showConfidence,
                onSelected: (value) => setState(() => _showConfidence = value),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPerformanceComparison() {
    if (_activeModels.isEmpty) return const SizedBox();

    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Performance Comparison',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            ..._activeModels.map((model) {
              final loadTime = _loadTimes[model] ?? 0;
              final inferenceTime = _inferenceTimes[model] ?? 0;
              final hasResults = _allResults.containsKey(model);

              return Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: model.color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: model.color.withOpacity(0.3)),
                ),
                child: Row(
                  children: [
                    Icon(model.icon, color: model.color, size: 24),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            model.name,
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: model.color,
                            ),
                          ),
                          Text(
                            'Load: ${loadTime.toStringAsFixed(0)}ms | Inference: ${hasResults ? inferenceTime.toStringAsFixed(0) : "--"}ms',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                    if (hasResults)
                      Icon(Icons.check_circle, color: model.color, size: 20),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildImageWithVisualization() {
    if (_imageFile == null) return const SizedBox();

    return Container(
      margin: const EdgeInsets.all(8),
      height: 300,
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey),
        borderRadius: BorderRadius.circular(8),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.file(_imageFile!, fit: BoxFit.contain),
            if (_showVisualization &&
                _uiImage != null &&
                _allResults.isNotEmpty)
              LayoutBuilder(
                builder: (context, constraints) {
                  return CustomPaint(
                    size: Size(constraints.maxWidth, constraints.maxHeight),
                    painter: MultiModelPainter(
                      allResults: _allResults,
                      models: _activeModels.toList(),
                      imageSize: Size(
                        _uiImage!.width.toDouble(),
                        _uiImage!.height.toDouble(),
                      ),
                      showLabels: _showLabels,
                      showConfidence: _showConfidence,
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildAllResultsView() {
    if (_allResults.isEmpty) return const SizedBox();

    return Column(
      children: _activeModels
          .where((model) => _allResults.containsKey(model))
          .map((model) {
            return Card(
              margin: const EdgeInsets.all(8),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(model.icon, color: model.color, size: 20),
                        const SizedBox(width: 8),
                        Text(
                          model.name,
                          style: Theme.of(
                            context,
                          ).textTheme.titleMedium?.copyWith(color: model.color),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    _buildTaskSpecificResults(model, _allResults[model]!),
                  ],
                ),
              ),
            );
          })
          .toList(),
    );
  }

  Widget _buildTaskSpecificResults(
    ModelConfig model,
    Map<String, dynamic> results,
  ) {
    switch (model.task) {
      case YOLOTask.detect:
      case YOLOTask.obb:
        final detections = results['detections'] as List<dynamic>? ?? [];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detected ${detections.length} objects'),
            const SizedBox(height: 8),
            ...detections.take(5).map((detection) {
              return ListTile(
                dense: true,
                leading: CircleAvatar(
                  backgroundColor: model.color,
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
        final detections = results['detections'] as List<dynamic>? ?? [];
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
                    backgroundColor: model.color,
                    child: const Icon(
                      Icons.texture,
                      color: Colors.white,
                      size: 16,
                    ),
                  ),
                  label: Text(detection['className'] ?? 'Unknown'),
                );
              }).toList(),
            ),
          ],
        );

      case YOLOTask.classify:
        final classification =
            results['classification'] as Map<dynamic, dynamic>?;
        if (classification == null)
          return const Text('No classification results');

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ListTile(
              dense: true,
              leading: CircleAvatar(
                backgroundColor: model.color,
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
                final confidences =
                    classification['top5Confidences'] as List<dynamic>;
                if (i + 1 >= classes.length) return const SizedBox();
                return ListTile(
                  dense: true,
                  leading: CircleAvatar(
                    backgroundColor: Colors.grey,
                    radius: 14,
                    child: Text(
                      '${i + 2}',
                      style: const TextStyle(color: Colors.white, fontSize: 12),
                    ),
                  ),
                  title: Text(classes[i + 1]),
                  subtitle: Text(
                    '${(confidences[i + 1] * 100).toStringAsFixed(1)}%',
                  ),
                );
              }),
          ],
        );

      case YOLOTask.pose:
        final detections = results['detections'] as List<dynamic>? ?? [];
        final peopleWithPose = detections
            .where((d) => d['keypoints'] != null)
            .length;
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detected $peopleWithPose person(s) with pose'),
            const SizedBox(height: 8),
            if (peopleWithPose > 0)
              ...detections.take(3).where((d) => d['keypoints'] != null).map((
                detection,
              ) {
                final keypoints = detection['keypoints'] as List<dynamic>;
                return ListTile(
                  dense: true,
                  leading: CircleAvatar(
                    backgroundColor: model.color,
                    child: const Icon(
                      Icons.accessibility_new,
                      color: Colors.white,
                      size: 16,
                    ),
                  ),
                  title: const Text('Person'),
                  subtitle: Text('${keypoints.length ~/ 3} keypoints detected'),
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
        title: const Text('YOLO Multi-Instance Demo'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Status message
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: _isLoading
                  ? Colors.orange.shade100
                  : Colors.green.shade100,
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
                        color: _isLoading
                            ? Colors.orange.shade900
                            : Colors.green.shade900,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Model selector
            _buildModelSelector(),

            // Visualization controls
            if (_imageFile != null && _allResults.isNotEmpty)
              _buildVisualizationControls(),

            // Performance comparison
            _buildPerformanceComparison(),

            // Image with visualization
            _buildImageWithVisualization(),

            // Results for all active models
            _buildAllResultsView(),

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

// Custom painter for multi-model visualization
class MultiModelPainter extends CustomPainter {
  final Map<ModelConfig, Map<String, dynamic>> allResults;
  final List<ModelConfig> models;
  final Size imageSize;
  final bool showLabels;
  final bool showConfidence;

  // Official Ultralytics color palette
  static const List<Color> ultralyticsColors = [
    Color(0xFF042AFF), // Blue
    Color(0xFF0BDBEB), // Cyan
    Color(0xFFF3F3F3), // Light Gray
    Color(0xFF00DFB7), // Mint
    Color(0xFF111F68), // Dark Blue
    Color(0xFFFF6FDD), // Pink
    Color(0xFFFF444F), // Red
    Color(0xFFCCED00), // Lime
    Color(0xFF00F344), // Green
    Color(0xFFBD00FF), // Purple
    Color(0xFF00B4FF), // Light Blue
    Color(0xFFDD00BA), // Magenta
    Color(0xFF00FFFF), // Aqua
    Color(0xFF26C000), // Dark Green
    Color(0xFF01FFB3), // Light Mint
    Color(0xFF7D24FF), // Violet
    Color(0xFF7B0068), // Dark Purple
    Color(0xFFFF1B6C), // Hot Pink
    Color(0xFFFC6D2F), // Orange
    Color(0xFFA2FF0B), // Yellow-Green
  ];

  MultiModelPainter({
    required this.allResults,
    required this.models,
    required this.imageSize,
    required this.showLabels,
    required this.showConfidence,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale to fit image in canvas (BoxFit.contain)
    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;
    final double scale = scaleX < scaleY ? scaleX : scaleY;

    // Calculate offset to center image
    final double scaledWidth = imageSize.width * scale;
    final double scaledHeight = imageSize.height * scale;
    final double offsetX = (size.width - scaledWidth) / 2;
    final double offsetY = (size.height - scaledHeight) / 2;

    // Draw results for each active model
    int modelIndex = 0;
    for (final model in models) {
      if (!allResults.containsKey(model)) continue;

      final results = allResults[model]!;

      switch (model.task) {
        case YOLOTask.detect:
          _drawDetections(
            canvas,
            results['detections'] as List<dynamic>? ?? [],
            scale,
            offsetX,
            offsetY,
            modelIndex,
          );
          break;

        case YOLOTask.pose:
          _drawPoses(
            canvas,
            results['detections'] as List<dynamic>? ?? [],
            scale,
            offsetX,
            offsetY,
          );
          break;

        case YOLOTask.segment:
          _drawSegmentations(
            canvas,
            results['detections'] as List<dynamic>? ?? [],
            scale,
            offsetX,
            offsetY,
            size,
          );
          break;

        case YOLOTask.obb:
          _drawOBBDetections(canvas, results, scale, offsetX, offsetY);
          break;

        case YOLOTask.classify:
          _drawClassification(
            canvas,
            results['classification'] as Map<dynamic, dynamic>? ?? {},
            model.color, // Keep model color for classification
            size,
          );
          break;
      }

      modelIndex++;
    }
  }

  void _drawDetections(
    Canvas canvas,
    List<dynamic> detections,
    double scale,
    double offsetX,
    double offsetY,
    int modelIndex,
  ) {
    for (final detection in detections) {
      try {
        // Get class index for color
        final classIndex = detection['classIndex'] as int? ?? 0;
        final color = ultralyticsColors[classIndex % ultralyticsColors.length];

        // Use normalizedBox for coordinates (0-1 range)
        final box = detection['normalizedBox'] as Map<dynamic, dynamic>?;
        if (box == null) continue;

        // Convert normalized coordinates to canvas coordinates
        final left = (box['left'] as num).toDouble();
        final top = (box['top'] as num).toDouble();
        final right = (box['right'] as num).toDouble();
        final bottom = (box['bottom'] as num).toDouble();

        final x1 = left * imageSize.width * scale + offsetX;
        final y1 = top * imageSize.height * scale + offsetY;
        final x2 = right * imageSize.width * scale + offsetX;
        final y2 = bottom * imageSize.height * scale + offsetY;

        // Offset boxes slightly for multiple models
        final offset = modelIndex * 2.0;

        // Draw bounding box
        final paint = Paint()
          ..color = color.withOpacity(0.8)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 4.0;

        canvas.drawRect(
          Rect.fromLTRB(x1 + offset, y1 + offset, x2 + offset, y2 + offset),
          paint,
        );

        // Draw label
        if (showLabels) {
          final label = StringBuffer();
          label.write(detection['className'] ?? 'Unknown');
          if (showConfidence) {
            label.write(
              ' ${((detection['confidence'] ?? 0) * 100).toStringAsFixed(1)}%',
            );
          }

          _drawLabel(
            canvas,
            label.toString(),
            Offset(x1 + offset, y1 + offset - 4),
            color,
          );
        }
      } catch (e) {
        // Error drawing detection: $e
      }
    }
  }

  // Pose color palette matching native implementation
  static const List<List<double>> posePalette = [
    [255, 128, 0],
    [255, 153, 51],
    [255, 178, 102],
    [230, 230, 0],
    [255, 153, 255],
    [153, 204, 255],
    [255, 102, 255],
    [255, 51, 255],
    [102, 178, 255],
    [51, 153, 255],
    [255, 153, 153],
    [255, 102, 102],
    [255, 51, 51],
    [153, 255, 153],
    [102, 255, 102],
    [51, 255, 51],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 255],
  ];

  // Keypoint color indices (17 keypoints)
  static const List<int> kptColorIndices = [
    16,
    16,
    16,
    16,
    16,
    9,
    9,
    9,
    9,
    9,
    9,
    0,
    0,
    0,
    0,
    0,
    0,
  ];

  // Limb color indices for skeleton connections
  static const List<int> limbColorIndices = [
    0,
    0,
    0,
    0,
    7,
    7,
    7,
    9,
    9,
    9,
    9,
    9,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
  ];

  // Skeleton connections matching native implementation
  static const List<List<int>> skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
  ];

  void _drawPoses(
    Canvas canvas,
    List<dynamic> detections,
    double scale,
    double offsetX,
    double offsetY,
  ) {
    for (
      int detectionIndex = 0;
      detectionIndex < detections.length;
      detectionIndex++
    ) {
      final detection = detections[detectionIndex];
      try {
        final keypointsFlat = detection['keypoints'] as List<dynamic>? ?? [];
        if (keypointsFlat.isEmpty) continue;

        // Convert flat array to structured keypoints
        final keypoints = <Map<String, double>>[];
        for (int i = 0; i < keypointsFlat.length; i += 3) {
          if (i + 2 < keypointsFlat.length) {
            keypoints.add({
              'x': (keypointsFlat[i] as num).toDouble(),
              'y': (keypointsFlat[i + 1] as num).toDouble(),
              'confidence': (keypointsFlat[i + 2] as num).toDouble(),
            });
          }
        }

        // Draw skeleton connections
        for (int i = 0; i < skeleton.length; i++) {
          final connection = skeleton[i];
          final kp1Index = connection[0] - 1; // Convert to 0-based index
          final kp2Index = connection[1] - 1;

          if (kp1Index < keypoints.length && kp2Index < keypoints.length) {
            final kp1 = keypoints[kp1Index];
            final kp2 = keypoints[kp2Index];

            if (kp1['confidence']! > 0.3 && kp2['confidence']! > 0.3) {
              final p1 = Offset(
                kp1['x']! * imageSize.width * scale + offsetX,
                kp1['y']! * imageSize.height * scale + offsetY,
              );
              final p2 = Offset(
                kp2['x']! * imageSize.width * scale + offsetX,
                kp2['y']! * imageSize.height * scale + offsetY,
              );

              // Get limb color from palette
              final limbColor = posePalette[limbColorIndices[i]];
              final linePaint = Paint()
                ..color = Color.fromRGBO(
                  limbColor[0].toInt(),
                  limbColor[1].toInt(),
                  limbColor[2].toInt(),
                  0.6,
                )
                ..strokeWidth = 4.0;

              canvas.drawLine(p1, p2, linePaint);
            }
          }
        }

        // Draw keypoints
        for (
          int i = 0;
          i < keypoints.length && i < kptColorIndices.length;
          i++
        ) {
          final keypoint = keypoints[i];
          if (keypoint['confidence']! > 0.3) {
            final center = Offset(
              keypoint['x']! * imageSize.width * scale + offsetX,
              keypoint['y']! * imageSize.height * scale + offsetY,
            );

            // Get keypoint color from palette
            final kptColor = posePalette[kptColorIndices[i]];
            final pointPaint = Paint()
              ..color = Color.fromRGBO(
                kptColor[0].toInt(),
                kptColor[1].toInt(),
                kptColor[2].toInt(),
                1.0,
              )
              ..style = PaintingStyle.fill;

            canvas.drawCircle(center, 3, pointPaint);
          }
        }

        // Draw bounding box
        final box = detection['normalizedBox'] as Map<dynamic, dynamic>?;
        if (box != null && showLabels) {
          final left = (box['left'] as num).toDouble();
          final top = (box['top'] as num).toDouble();

          final x = left * imageSize.width * scale + offsetX;
          final y = top * imageSize.height * scale + offsetY;

          // For pose, all detections are "person" class, so use the same color (index 0)
          final labelColor =
              ultralyticsColors[0]; // Always use first color for person class
          _drawLabel(canvas, 'Person', Offset(x, y - 4), labelColor);
        }
      } catch (e) {
        // Error drawing pose: $e
      }
    }
  }

  void _drawSegmentations(
    Canvas canvas,
    List<dynamic> detections,
    double scale,
    double offsetX,
    double offsetY,
    Size canvasSize,
  ) {
    for (int i = 0; i < detections.length; i++) {
      try {
        final detection = detections[i];

        // Get class index for color
        final classIndex = detection['classIndex'] as int? ?? 0;
        final color = ultralyticsColors[classIndex % ultralyticsColors.length];

        final mask = detection['mask'] as List<dynamic>?;

        // Draw mask if available
        if (mask != null && mask.isNotEmpty) {
          final paint = Paint()
            ..color = color.withOpacity(0.4)
            ..style = PaintingStyle.fill;

          // Check if mask is properly formatted
          if (mask[0] is List) {
            final maskHeight = mask.length;
            final maskWidth = (mask[0] as List).length;

            // Calculate scale from mask size to image size
            final maskScaleX = imageSize.width / maskWidth;
            final maskScaleY = imageSize.height / maskHeight;

            // Create path for the mask
            final path = Path();

            // Draw mask pixels
            for (int y = 0; y < maskHeight; y += 2) {
              // Skip pixels for performance
              for (int x = 0; x < maskWidth; x += 2) {
                // Check if it's a valid 2D array access
                if (y < mask.length && x < (mask[y] as List).length) {
                  final pixelValue = mask[y][x];

                  // Check if pixel is part of mask (handle both int and double)
                  bool isInMask = false;
                  if (pixelValue is num) {
                    isInMask = pixelValue > 0.5;
                  } else if (pixelValue is bool) {
                    isInMask = pixelValue;
                  }

                  if (isInMask) {
                    // Convert mask coordinates to canvas coordinates
                    final pixelX = x * maskScaleX * scale + offsetX;
                    final pixelY = y * maskScaleY * scale + offsetY;
                    final pixelWidth = maskScaleX * scale * 2;
                    final pixelHeight = maskScaleY * scale * 2;

                    path.addRect(
                      Rect.fromLTWH(
                        pixelX,
                        pixelY,
                        pixelWidth + 0.5,
                        pixelHeight + 0.5,
                      ),
                    );
                  }
                }
              }
            }

            canvas.drawPath(path, paint);
          }
        }

        // Draw bounding box
        final box = detection['normalizedBox'] as Map<dynamic, dynamic>?;
        if (box != null) {
          final left = (box['left'] as num).toDouble();
          final top = (box['top'] as num).toDouble();
          final right = (box['right'] as num).toDouble();
          final bottom = (box['bottom'] as num).toDouble();

          final x1 = left * imageSize.width * scale + offsetX;
          final y1 = top * imageSize.height * scale + offsetY;
          final x2 = right * imageSize.width * scale + offsetX;
          final y2 = bottom * imageSize.height * scale + offsetY;

          // Draw border
          final borderPaint = Paint()
            ..color = color
            ..style = PaintingStyle.stroke
            ..strokeWidth = 4.0;

          canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), borderPaint);

          if (showLabels) {
            final label = StringBuffer();
            label.write(detection['className'] ?? 'Unknown');
            if (showConfidence) {
              label.write(
                ' ${((detection['confidence'] ?? 0) * 100).toStringAsFixed(1)}%',
              );
            }

            _drawLabel(canvas, label.toString(), Offset(x1, y1 - 4), color);
          }
        }
      } catch (e) {
        // Error drawing segmentation: $e
      }
    }
  }

  void _drawOBBDetections(
    Canvas canvas,
    Map<String, dynamic> results,
    double scale,
    double offsetX,
    double offsetY,
  ) {
    // For OBB, try both 'obb' array and 'detections' array
    final detections = results['detections'] as List<dynamic>? ?? [];

    for (final detection in detections) {
      try {
        // Get class index for color
        final classIndex = detection['classIndex'] as int? ?? 0;
        final color = ultralyticsColors[classIndex % ultralyticsColors.length];

        // Use normalizedBox for OBB as well (simplified visualization)
        final box = detection['normalizedBox'] as Map<dynamic, dynamic>?;
        if (box == null) continue;

        final left = (box['left'] as num).toDouble();
        final top = (box['top'] as num).toDouble();
        final right = (box['right'] as num).toDouble();
        final bottom = (box['bottom'] as num).toDouble();

        final x1 = left * imageSize.width * scale + offsetX;
        final y1 = top * imageSize.height * scale + offsetY;
        final x2 = right * imageSize.width * scale + offsetX;
        final y2 = bottom * imageSize.height * scale + offsetY;

        // Draw as rotated rectangle (simplified)
        final paint = Paint()
          ..color = color.withOpacity(0.2)
          ..style = PaintingStyle.fill;

        canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), paint);

        final borderPaint = Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 4.0;

        canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), borderPaint);

        if (showLabels) {
          final label = StringBuffer();
          label.write(detection['className'] ?? 'Unknown');
          if (showConfidence) {
            label.write(
              ' ${((detection['confidence'] ?? 0) * 100).toStringAsFixed(1)}%',
            );
          }

          _drawLabel(canvas, label.toString(), Offset(x1, y1 - 4), color);
        }
      } catch (e) {
        // Error drawing OBB: $e
      }
    }
  }

  void _drawClassification(
    Canvas canvas,
    Map<dynamic, dynamic> classification,
    Color color,
    Size canvasSize,
  ) {
    if (classification.isEmpty) return;

    // Draw classification result in top-left corner
    final bgPaint = Paint()
      ..color = Colors.black.withOpacity(0.7)
      ..style = PaintingStyle.fill;

    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    final label = StringBuffer();
    label.write('Class: ${classification['topClass'] ?? 'Unknown'}');
    if (showConfidence) {
      label.write(
        ' (${((classification['topConfidence'] ?? 0) * 100).toStringAsFixed(1)}%)',
      );
    }

    textPainter.text = TextSpan(
      text: label.toString(),
      style: TextStyle(color: color, fontSize: 16, fontWeight: FontWeight.bold),
    );
    textPainter.layout();

    final padding = 8.0;
    final bgRect = Rect.fromLTWH(
      padding,
      padding,
      textPainter.width + padding * 2,
      textPainter.height + padding,
    );

    canvas.drawRRect(
      RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
      bgPaint,
    );

    textPainter.paint(canvas, Offset(padding * 2, padding * 1.5));
  }

  void _drawLabel(Canvas canvas, String text, Offset position, Color color) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    // Draw background
    final bgPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final padding = 2.0;
    final bgRect = Rect.fromLTWH(
      position.dx,
      position.dy - textPainter.height - padding,
      textPainter.width + padding * 2,
      textPainter.height + padding,
    );

    canvas.drawRRect(
      RRect.fromRectAndRadius(bgRect, const Radius.circular(2)),
      bgPaint,
    );

    // Draw text
    textPainter.paint(
      canvas,
      Offset(position.dx + padding, position.dy - textPainter.height - padding),
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
