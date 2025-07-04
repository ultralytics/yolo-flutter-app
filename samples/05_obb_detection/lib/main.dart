// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:ui' as ui;
import 'dart:math' as math;
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
      title: 'YOLO OBB Detection',
      theme: ThemeData(primarySwatch: Colors.purple, useMaterial3: true),
      home: const ObbDetectionScreen(),
    );
  }
}

class ObbDetectionScreen extends StatefulWidget {
  const ObbDetectionScreen({super.key});

  @override
  State<ObbDetectionScreen> createState() => _ObbDetectionScreenState();
}

class _ObbDetectionScreenState extends State<ObbDetectionScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  List<Map<String, dynamic>>? _detectionResults;
  Map<String, dynamic>? _rawResults;
  bool _isProcessing = false;
  bool _showLabels = true;
  bool _showConfidence = true;
  bool _showAngle = true;
  double _confidenceThreshold = 0.25;
  double _iouThreshold = 0.45;

  // YOLO instance for OBB detection
  late final YOLO _yolo;

  @override
  void initState() {
    super.initState();
    _yolo = YOLO(
      modelPath: 'yolo11n-obb.tflite', // OBB model
      task: YOLOTask.obb,
    );
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() => _isProcessing = true);
    try {
      final success = await _yolo.loadModel();
      if (!success) {
        throw Exception('Failed to load model');
      }
      // Model loaded successfully
    } catch (e) {
      // Error loading model: $e
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

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _detectionResults = null;
        _rawResults = null;
      });
      _detectObjects();
    }
  }

  Future<void> _detectObjects() async {
    if (_imageFile == null) return;

    setState(() => _isProcessing = true);

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final results = await _yolo.predict(
        imageBytes,
        confidenceThreshold: _confidenceThreshold,
        iouThreshold: _iouThreshold,
      );

      setState(() {
        _rawResults = results;
        _detectionResults = (results['detections'] as List<dynamic>?)
            ?.map((e) => Map<String, dynamic>.from(e))
            .toList();

        // Debug: Check what data we're receiving
        // Process detection results
        // _detectionResults contains ${_detectionResults?.length ?? 0} detections
      });
    } catch (e) {
      // Error during detection: $e
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
        title: const Text('YOLO OBB Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Controls
            Card(
              margin: const EdgeInsets.all(8),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Detection Settings',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 16),

                    // Confidence Threshold
                    Row(
                      children: [
                        const Icon(Icons.assessment, size: 20),
                        const SizedBox(width: 8),
                        const Text('Confidence:'),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Slider(
                            value: _confidenceThreshold,
                            min: 0.0,
                            max: 1.0,
                            divisions: 20,
                            label: _confidenceThreshold.toStringAsFixed(2),
                            onChanged: (value) {
                              setState(() => _confidenceThreshold = value);
                            },
                            onChangeEnd: (_) => _detectObjects(),
                          ),
                        ),
                        Text(_confidenceThreshold.toStringAsFixed(2)),
                      ],
                    ),

                    // IoU Threshold
                    Row(
                      children: [
                        const Icon(Icons.layers, size: 20),
                        const SizedBox(width: 8),
                        const Text('IoU:'),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Slider(
                            value: _iouThreshold,
                            min: 0.0,
                            max: 1.0,
                            divisions: 20,
                            label: _iouThreshold.toStringAsFixed(2),
                            onChanged: (value) {
                              setState(() => _iouThreshold = value);
                            },
                            onChangeEnd: (_) => _detectObjects(),
                          ),
                        ),
                        Text(_iouThreshold.toStringAsFixed(2)),
                      ],
                    ),

                    // Display options
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 16,
                      children: [
                        FilterChip(
                          label: const Text('Labels'),
                          selected: _showLabels,
                          onSelected: (value) =>
                              setState(() => _showLabels = value),
                        ),
                        FilterChip(
                          label: const Text('Confidence'),
                          selected: _showConfidence,
                          onSelected: (value) =>
                              setState(() => _showConfidence = value),
                        ),
                        FilterChip(
                          label: const Text('Angle'),
                          selected: _showAngle,
                          onSelected: (value) =>
                              setState(() => _showAngle = value),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            // Image and detections
            if (_imageFile != null)
              Container(
                margin: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Stack(
                  children: [
                    Image.file(_imageFile!),
                    if (_detectionResults != null && _rawResults != null)
                      Positioned.fill(
                        child: FutureBuilder<ui.Image>(
                          future: _getImageInfo(),
                          builder: (context, snapshot) {
                            if (!snapshot.hasData) return const SizedBox();
                            return CustomPaint(
                              painter: ObbPainter(
                                detections: _detectionResults!,
                                rawResults: _rawResults!,
                                imageSize: Size(
                                  snapshot.data!.width.toDouble(),
                                  snapshot.data!.height.toDouble(),
                                ),
                                showLabels: _showLabels,
                                showConfidence: _showConfidence,
                                showAngle: _showAngle,
                              ),
                            );
                          },
                        ),
                      ),
                  ],
                ),
              ),

            // Results summary
            if (_detectionResults != null)
              Card(
                margin: const EdgeInsets.all(8),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Detection Results',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      Text('Found ${_detectionResults!.length} objects'),
                      const SizedBox(height: 8),
                      // Show top detections
                      ..._detectionResults!.take(5).map((detection) {
                        final angle = _calculateAngleFromObb(detection);
                        return ListTile(
                          dense: true,
                          leading: CircleAvatar(
                            backgroundColor: _getColorForClass(
                              detection['className'] ?? '',
                            ),
                            child: Text(
                              '${detection['className']?.substring(0, 1).toUpperCase() ?? '?'}',
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                          title: Text(detection['className'] ?? 'Unknown'),
                          subtitle: Text(
                            'Confidence: ${((detection['confidence'] ?? 0) * 100).toStringAsFixed(1)}%' +
                                (angle != null
                                    ? ', Angle: ${angle.toStringAsFixed(1)}°'
                                    : ''),
                          ),
                        );
                      }).toList(),
                      if (_detectionResults!.length > 5)
                        Padding(
                          padding: const EdgeInsets.only(top: 8),
                          child: Text(
                            '... and ${_detectionResults!.length - 5} more',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ),
                    ],
                  ),
                ),
              ),

            // Processing indicator
            if (_isProcessing)
              const Padding(
                padding: EdgeInsets.all(16),
                child: CircularProgressIndicator(),
              ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isProcessing ? null : _pickImage,
        icon: const Icon(Icons.image),
        label: const Text('Select Image'),
      ),
    );
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

  double? _calculateAngleFromObb(Map<String, dynamic> detection) {
    // Try to get angle from the raw OBB data if available
    if (_rawResults != null && _rawResults!['obb'] != null) {
      final obbList = _rawResults!['obb'] as List<dynamic>;
      // Find matching OBB by class name
      for (final obb in obbList) {
        if (obb is Map && obb['class'] == detection['className']) {
          // Get angle from OBB data if available
          if (obb['angle'] != null) {
            // Convert from radians to degrees
            return (obb['angle'] as num).toDouble() * 180 / 3.14159;
          }
          // Otherwise calculate from points
          else if (obb['points'] != null) {
            final points = obb['points'] as List<dynamic>;
            if (points.length >= 2) {
              final p1 = points[0] as Map<dynamic, dynamic>;
              final p2 = points[1] as Map<dynamic, dynamic>;
              final dx =
                  (p2['x'] as num).toDouble() - (p1['x'] as num).toDouble();
              final dy =
                  (p2['y'] as num).toDouble() - (p1['y'] as num).toDouble();
              return math.atan2(dy, dx) * 180 / math.pi;
            }
          }
        }
      }
    }
    return null;
  }
}

class ObbPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  final Map<String, dynamic> rawResults;
  final Size imageSize;
  final bool showLabels;
  final bool showConfidence;
  final bool showAngle;

  ObbPainter({
    required this.detections,
    required this.rawResults,
    required this.imageSize,
    required this.showLabels,
    required this.showConfidence,
    required this.showAngle,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    // Debug
    // Canvas size: $size, Image size: $imageSize
    // Number of detections: ${detections.length}

    // Get OBB data if available
    final obbList = rawResults['obb'] as List<dynamic>? ?? [];
    // OBB list length: ${obbList.length}

    // For OBB task, we should primarily use the OBB data, not detections
    if (obbList.isNotEmpty) {
      // Draw from OBB data directly
      for (int i = 0; i < obbList.length; i++) {
        final obbData = obbList[i] as Map<dynamic, dynamic>;
        final className = obbData['class'] ?? 'Unknown';
        final confidence = obbData['confidence'] ?? 0.0;
        final color = _getColorForClass(className.toString());

        // Drawing OBB $i: class=$className, conf=$confidence

        if (obbData['points'] != null) {
          // Draw oriented bounding box using points
          final points = obbData['points'] as List<dynamic>;
          if (points.length >= 4) {
            final paint = Paint()
              ..color = color.withOpacity(0.3)
              ..style = PaintingStyle.fill;

            final borderPaint = Paint()
              ..color = color
              ..style = PaintingStyle.stroke
              ..strokeWidth = 4.0;

            // Convert points to Path
            final path = Path();
            for (int j = 0; j < points.length; j++) {
              final point = points[j] as Map<dynamic, dynamic>;
              // Points are already in normalized coordinates (0-1)
              final x = (point['x'] as num).toDouble() * size.width;
              final y = (point['y'] as num).toDouble() * size.height;

              if (j == 0) {
                path.moveTo(x, y);
              } else {
                path.lineTo(x, y);
              }
            }
            path.close();

            // Draw filled and border
            canvas.drawPath(path, paint);
            canvas.drawPath(path, borderPaint);

            // Draw corners
            for (final point in points) {
              // Points are already in normalized coordinates (0-1)
              final x = (point['x'] as num).toDouble() * size.width;
              final y = (point['y'] as num).toDouble() * size.height;
              canvas.drawCircle(Offset(x, y), 4, Paint()..color = color);
            }

            // Draw label
            if (showLabels || showConfidence || showAngle) {
              final textPainter = TextPainter(textDirection: TextDirection.ltr);

              String label = '';
              if (showLabels) label = className.toString();
              if (showConfidence) {
                if (label.isNotEmpty) label += ' ';
                label += '${(confidence * 100).toStringAsFixed(0)}%';
              }
              if (showAngle && obbData['angle'] != null) {
                final angleDeg =
                    (obbData['angle'] as num).toDouble() * 180 / 3.14159;
                if (label.isNotEmpty) label += ' ';
                label += '${angleDeg.toStringAsFixed(0)}°';
              }

              textPainter.text = TextSpan(
                text: label,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  backgroundColor: color.withOpacity(0.7),
                ),
              );
              textPainter.layout();

              // Position label at first point
              final firstPoint = points[0] as Map<dynamic, dynamic>;
              final labelX = (firstPoint['x'] as num).toDouble() * size.width;
              final labelY =
                  (firstPoint['y'] as num).toDouble() * size.height -
                  textPainter.height -
                  2;

              textPainter.paint(
                canvas,
                Offset(labelX, labelY > 0 ? labelY : 0),
              );
            }
          }
        }
      }
    } else if (detections.isNotEmpty) {
      // Fallback to regular detection boxes if no OBB data
      // No OBB data, falling back to regular boxes
      for (final detection in detections) {
        final color = _getColorForClass(detection['className'] ?? '');

        // Fallback to regular bounding box
        final normalizedBox =
            detection['normalizedBox'] as Map<String, dynamic>;
        final rect = Rect.fromLTRB(
          normalizedBox['left'] * size.width,
          normalizedBox['top'] * size.height,
          normalizedBox['right'] * size.width,
          normalizedBox['bottom'] * size.height,
        );

        final paint = Paint()
          ..color = color.withOpacity(0.3)
          ..style = PaintingStyle.fill;

        final borderPaint = Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 4.0;

        canvas.drawRect(rect, paint);
        canvas.drawRect(rect, borderPaint);

        // Draw label
        if (showLabels || showConfidence) {
          final textPainter = TextPainter(textDirection: TextDirection.ltr);

          String label = '';
          if (showLabels) label = detection['className'] ?? 'Unknown';
          if (showConfidence) {
            if (label.isNotEmpty) label += ' ';
            label +=
                '${((detection['confidence'] ?? 0) * 100).toStringAsFixed(0)}%';
          }

          textPainter.text = TextSpan(
            text: label,
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              backgroundColor: color.withOpacity(0.7),
            ),
          );
          textPainter.layout();

          // Position label at top-left of the box
          final labelX = normalizedBox['left'] * size.width;
          final labelY =
              normalizedBox['top'] * size.height - textPainter.height - 2;

          textPainter.paint(canvas, Offset(labelX, labelY > 0 ? labelY : 0));
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant ObbPainter oldDelegate) {
    return detections != oldDelegate.detections ||
        showLabels != oldDelegate.showLabels ||
        showConfidence != oldDelegate.showConfidence ||
        showAngle != oldDelegate.showAngle;
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
