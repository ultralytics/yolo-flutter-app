// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
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
      title: 'YOLO Basic Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const DetectionScreen(),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  final ImagePicker _picker = ImagePicker();

  // State
  File? _imageFile;
  List<Map<String, dynamic>>? _detectionResults;
  bool _isProcessing = false;
  double _confidence = 0.45;
  double _iou = 0.45;
  String _processingTime = '';

  // YOLO instance
  late final YOLO _yolo;

  @override
  void initState() {
    super.initState();
    _yolo = YOLO(modelPath: 'yolo11n.tflite', task: YOLOTask.detect);
    _loadModel();
  }

  @override
  void dispose() {
    _yolo.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    setState(() => _isProcessing = true);
    try {
      final success = await _yolo.loadModel();
      if (!success) {
        throw Exception('Failed to load model');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading model: $e')));
      }
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _detectionResults = null;
        _processingTime = '';
      });
      _detectObjects();
    }
  }

  Future<void> _detectObjects() async {
    if (_imageFile == null) return;

    setState(() => _isProcessing = true);
    final stopwatch = Stopwatch()..start();

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final results = await _yolo.predict(
        imageBytes,
        confidenceThreshold: _confidence,
        iouThreshold: _iou,
      );

      stopwatch.stop();

      setState(() {
        _detectionResults = (results['detections'] as List<dynamic>?)
            ?.map((e) => Map<String, dynamic>.from(e))
            .toList();
        _processingTime = '${stopwatch.elapsedMilliseconds}ms';
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error during detection: $e')));
      }
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Basic Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Controls section
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey[100],
            child: Column(
              children: [
                // Stats row
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildStatCard(
                      'Detections',
                      '${_detectionResults?.length ?? 0}',
                      Icons.crop_square,
                    ),
                    if (_processingTime.isNotEmpty)
                      _buildStatCard(
                        'Processing',
                        _processingTime,
                        Icons.timer,
                      ),
                  ],
                ),
                const SizedBox(height: 16),
                // Confidence slider
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.tune),
                        const SizedBox(width: 8),
                        Text('Confidence: ${(_confidence * 100).toInt()}%'),
                      ],
                    ),
                    Slider(
                      value: _confidence,
                      min: 0.1,
                      max: 0.9,
                      divisions: 8,
                      label: '${(_confidence * 100).toInt()}%',
                      onChanged: (value) {
                        setState(() => _confidence = value);
                      },
                    ),
                  ],
                ),
                // Action buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _isProcessing
                          ? null
                          : () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                    ),
                    ElevatedButton.icon(
                      onPressed: _isProcessing
                          ? null
                          : () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                    ),
                  ],
                ),
              ],
            ),
          ),
          // Image and detections
          Expanded(
            child: _isProcessing
                ? const Center(child: CircularProgressIndicator())
                : _imageFile == null
                ? const Center(
                    child: Text(
                      'Select an image to detect objects',
                      style: TextStyle(fontSize: 18, color: Colors.grey),
                    ),
                  )
                : SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        // Image with detections
                        if (_imageFile != null)
                          FutureBuilder<ui.Image>(
                            future: _getImageInfo(),
                            builder: (context, snapshot) {
                              if (!snapshot.hasData) {
                                return Container(
                                  constraints: BoxConstraints(
                                    maxHeight:
                                        MediaQuery.of(context).size.height *
                                        0.5,
                                  ),
                                  child: Image.file(
                                    _imageFile!,
                                    fit: BoxFit.contain,
                                  ),
                                );
                              }

                              // Calculate scale to fit image within screen bounds
                              final screenWidth =
                                  MediaQuery.of(context).size.width -
                                  32; // Account for padding
                              final maxHeight =
                                  MediaQuery.of(context).size.height * 0.5;
                              final imageWidth = snapshot.data!.width
                                  .toDouble();
                              final imageHeight = snapshot.data!.height
                                  .toDouble();

                              double scale = 1.0;
                              if (imageWidth > screenWidth) {
                                scale = screenWidth / imageWidth;
                              }
                              if (imageHeight * scale > maxHeight) {
                                scale = maxHeight / imageHeight;
                              }

                              final scaledWidth = imageWidth * scale;
                              final scaledHeight = imageHeight * scale;

                              return Center(
                                child: CustomPaint(
                                  size: Size(scaledWidth, scaledHeight),
                                  painter: DetectionPainter(
                                    image: snapshot.data!,
                                    detections: _detectionResults ?? [],
                                    scale: scale,
                                  ),
                                ),
                              );
                            },
                          ),
                        // Detection list
                        if (_detectionResults != null &&
                            _detectionResults!.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.only(top: 16),
                            child: Card(
                              child: Padding(
                                padding: const EdgeInsets.all(16),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Detected Objects',
                                      style: Theme.of(
                                        context,
                                      ).textTheme.titleMedium,
                                    ),
                                    const SizedBox(height: 8),
                                    ..._buildDetectionList(),
                                  ],
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatCard(String label, String value, IconData icon) {
    return Column(
      children: [
        Icon(icon, size: 32, color: Theme.of(context).primaryColor),
        const SizedBox(height: 4),
        Text(label, style: Theme.of(context).textTheme.labelMedium),
        Text(value, style: Theme.of(context).textTheme.headlineSmall),
      ],
    );
  }

  List<Widget> _buildDetectionList() {
    final classCount = <String, int>{};
    for (final detection in _detectionResults!) {
      final className = detection['className'] ?? 'Unknown';
      classCount[className] = (classCount[className] ?? 0) + 1;
    }

    return classCount.entries.map((entry) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: _getColorForClass(entry.key),
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(child: Text(entry.key)),
            Text(
              '${entry.value}',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ],
        ),
      );
    }).toList();
  }

  Future<ui.Image> _getImageInfo() async {
    final bytes = await _imageFile!.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  Color _getColorForClass(String className) {
    // Ultralytics standard detection colors with 60% opacity
    final colors = [
      const Color.fromARGB(153, 4, 42, 255), // Blue
      const Color.fromARGB(153, 11, 219, 235), // Cyan
      const Color.fromARGB(153, 243, 243, 243), // Light Gray
      const Color.fromARGB(153, 0, 223, 183), // Turquoise
      const Color.fromARGB(153, 17, 31, 104), // Dark Blue
      const Color.fromARGB(153, 255, 111, 221), // Pink
      const Color.fromARGB(153, 255, 68, 79), // Red
      const Color.fromARGB(153, 204, 237, 0), // Yellow-Green
      const Color.fromARGB(153, 0, 243, 68), // Green
      const Color.fromARGB(153, 189, 0, 255), // Purple
      const Color.fromARGB(153, 0, 180, 255), // Light Blue
      const Color.fromARGB(153, 221, 0, 186), // Magenta
      const Color.fromARGB(153, 0, 255, 255), // Cyan
      const Color.fromARGB(153, 38, 192, 0), // Dark Green
      const Color.fromARGB(153, 1, 255, 179), // Mint
      const Color.fromARGB(153, 125, 36, 255), // Violet
      const Color.fromARGB(153, 123, 0, 104), // Dark Purple
      const Color.fromARGB(153, 255, 27, 108), // Hot Pink
      const Color.fromARGB(153, 252, 109, 47), // Orange
      const Color.fromARGB(153, 162, 255, 11), // Lime Green
    ];
    return colors[className.hashCode % colors.length];
  }
}

class DetectionPainter extends CustomPainter {
  final ui.Image image;
  final List<Map<String, dynamic>> detections;
  final double scale;

  DetectionPainter({
    required this.image,
    required this.detections,
    this.scale = 1.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Save canvas state
    canvas.save();

    // Scale canvas to fit the image
    canvas.scale(scale);

    // Draw the image
    final paint = Paint();
    canvas.drawImage(image, Offset.zero, paint);

    // Calculate scale factor for stroke widths based on image size
    final imageScale = (image.width + image.height) / 2000.0;
    final strokeScale = imageScale.clamp(0.5, 3.0); // Clamp between 0.5x and 3x

    // Draw detections
    for (final detection in detections) {
      final normalizedBox = detection['normalizedBox'] as Map<String, dynamic>?;
      if (normalizedBox == null) continue;

      // Convert normalized coordinates to pixel coordinates
      final rect = Rect.fromLTRB(
        normalizedBox['left'] * image.width.toDouble(),
        normalizedBox['top'] * image.height.toDouble(),
        normalizedBox['right'] * image.width.toDouble(),
        normalizedBox['bottom'] * image.height.toDouble(),
      );

      // Draw bounding box
      final boxPaint = Paint()
        ..color =
            _getColorForClass(
              detection['className'] ?? '',
            ) // Use color with original 60% opacity
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4.0 * strokeScale; // Scale stroke width with image size
      canvas.drawRect(rect, boxPaint);

      // Draw label background
      final className = detection['className'] ?? 'Unknown';
      final confidence = (detection['confidence'] ?? 0.0) * 100;
      final label = '$className ${confidence.toStringAsFixed(0)}%';

      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelBgRect = Rect.fromLTWH(
        rect.left,
        rect.top - 24,
        textPainter.width + 8,
        24,
      );

      final labelBgPaint = Paint()
        ..color = _getColorForClass(
          className,
        ); // Use color with original opacity
      canvas.drawRect(labelBgRect, labelBgPaint);

      // Draw label text
      textPainter.paint(canvas, Offset(rect.left + 4, rect.top - 22));
    }

    // Restore canvas state
    canvas.restore();
  }

  Color _getColorForClass(String className) {
    // Ultralytics standard detection colors with 60% opacity
    final colors = [
      const Color.fromARGB(153, 4, 42, 255), // Blue
      const Color.fromARGB(153, 11, 219, 235), // Cyan
      const Color.fromARGB(153, 243, 243, 243), // Light Gray
      const Color.fromARGB(153, 0, 223, 183), // Turquoise
      const Color.fromARGB(153, 17, 31, 104), // Dark Blue
      const Color.fromARGB(153, 255, 111, 221), // Pink
      const Color.fromARGB(153, 255, 68, 79), // Red
      const Color.fromARGB(153, 204, 237, 0), // Yellow-Green
      const Color.fromARGB(153, 0, 243, 68), // Green
      const Color.fromARGB(153, 189, 0, 255), // Purple
      const Color.fromARGB(153, 0, 180, 255), // Light Blue
      const Color.fromARGB(153, 221, 0, 186), // Magenta
      const Color.fromARGB(153, 0, 255, 255), // Cyan
      const Color.fromARGB(153, 38, 192, 0), // Dark Green
      const Color.fromARGB(153, 1, 255, 179), // Mint
      const Color.fromARGB(153, 125, 36, 255), // Violet
      const Color.fromARGB(153, 123, 0, 104), // Dark Purple
      const Color.fromARGB(153, 255, 27, 108), // Hot Pink
      const Color.fromARGB(153, 252, 109, 47), // Orange
      const Color.fromARGB(153, 162, 255, 11), // Lime Green
    ];
    return colors[className.hashCode % colors.length];
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return image != oldDelegate.image || detections != oldDelegate.detections;
  }
}
