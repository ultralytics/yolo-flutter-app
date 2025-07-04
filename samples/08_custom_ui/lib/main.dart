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
      title: 'YOLO Custom UI',
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        useMaterial3: true,
      ),
      home: const CustomUIScreen(),
    );
  }
}

class CustomUIScreen extends StatefulWidget {
  const CustomUIScreen({super.key});

  @override
  State<CustomUIScreen> createState() => _CustomUIScreenState();
}

class _CustomUIScreenState extends State<CustomUIScreen> 
    with TickerProviderStateMixin {
  final ImagePicker _picker = ImagePicker();
  
  // Animation controllers
  late AnimationController _pulseController;
  late AnimationController _slideController;
  late Animation<double> _pulseAnimation;
  late Animation<Offset> _slideAnimation;
  
  // State
  File? _imageFile;
  List<Map<String, dynamic>>? _detectionResults;
  bool _isProcessing = false;
  
  // UI customization options
  bool _showAnimations = true;
  bool _showConfidenceBar = true;
  bool _showDetectionGrid = false;
  bool _showHeatmap = false;
  double _confidenceThreshold = 0.45;
  
  // Visualization styles
  VisualizationStyle _currentStyle = VisualizationStyle.modern;
  
  // YOLO instance
  late final YOLO _yolo;

  @override
  void initState() {
    super.initState();
    
    // Initialize animations
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat();
    
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    
    _pulseAnimation = Tween<double>(
      begin: 1.0,
      end: 1.2,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 1),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _slideController,
      curve: Curves.easeOutCubic,
    ));
    
    // Initialize YOLO
    _yolo = YOLO(
      modelPath: 'yolo11n.tflite',
      task: YOLOTask.detect,
    );
    _loadModel();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _slideController.dispose();
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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading model: $e')),
        );
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
      );

      setState(() {
        _detectionResults = (results['detections'] as List<dynamic>?)
            ?.map((e) => Map<String, dynamic>.from(e))
            .toList();
      });
      
      // Trigger slide animation
      _slideController.forward();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error during detection: $e')),
        );
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
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: const Text('YOLO Custom UI'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        elevation: 0,
      ),
      body: Column(
        children: [
          // Style selector
          Container(
            height: 60,
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.inversePrimary,
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(24),
                bottomRight: Radius.circular(24),
              ),
            ),
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              itemCount: VisualizationStyle.values.length,
              itemBuilder: (context, index) {
                final style = VisualizationStyle.values[index];
                final isSelected = _currentStyle == style;
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: ChoiceChip(
                    label: Text(style.name.toUpperCase()),
                    selected: isSelected,
                    onSelected: (selected) {
                      setState(() => _currentStyle = style);
                    },
                  ),
                );
              },
            ),
          ),
          
          // Main content
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  // Visualization options
                  Card(
                    elevation: 8,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Visualization Options',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 16),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              FilterChip(
                                label: const Text('Animations'),
                                selected: _showAnimations,
                                onSelected: (value) => setState(() => _showAnimations = value),
                                avatar: const Icon(Icons.animation, size: 18),
                              ),
                              FilterChip(
                                label: const Text('Confidence Bar'),
                                selected: _showConfidenceBar,
                                onSelected: (value) => setState(() => _showConfidenceBar = value),
                                avatar: const Icon(Icons.bar_chart, size: 18),
                              ),
                              FilterChip(
                                label: const Text('Detection Grid'),
                                selected: _showDetectionGrid,
                                onSelected: (value) => setState(() => _showDetectionGrid = value),
                                avatar: const Icon(Icons.grid_on, size: 18),
                              ),
                              FilterChip(
                                label: const Text('Heatmap'),
                                selected: _showHeatmap,
                                onSelected: (value) => setState(() => _showHeatmap = value),
                                avatar: const Icon(Icons.gradient, size: 18),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // Image and detections
                  if (_imageFile != null)
                    SlideTransition(
                      position: _showAnimations ? _slideAnimation : AlwaysStoppedAnimation(Offset.zero),
                      child: Card(
                        elevation: 8,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: Stack(
                          children: [
                            Image.file(_imageFile!),
                            if (_detectionResults != null)
                              Positioned.fill(
                                child: FutureBuilder<ui.Image>(
                                  future: _getImageInfo(),
                                  builder: (context, snapshot) {
                                    if (!snapshot.hasData) return const SizedBox();
                                    return CustomPaint(
                                      painter: CustomVisualizationPainter(
                                        detections: _detectionResults!,
                                        imageSize: Size(
                                          snapshot.data!.width.toDouble(),
                                          snapshot.data!.height.toDouble(),
                                        ),
                                        style: _currentStyle,
                                        showConfidenceBar: _showConfidenceBar,
                                        showGrid: _showDetectionGrid,
                                        showHeatmap: _showHeatmap,
                                        animationValue: _showAnimations ? _pulseAnimation.value : 1.0,
                                      ),
                                    );
                                  },
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  
                  // Detection summary with animations
                  if (_detectionResults != null)
                    SlideTransition(
                      position: _showAnimations ? _slideAnimation : AlwaysStoppedAnimation(Offset.zero),
                      child: Card(
                        margin: const EdgeInsets.only(top: 16),
                        elevation: 8,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Detection Summary',
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                              const SizedBox(height: 8),
                              _buildAnimatedStats(),
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
      floatingActionButton: AnimatedBuilder(
        animation: _showAnimations ? _pulseAnimation : AlwaysStoppedAnimation(1.0),
        builder: (context, child) {
          return Transform.scale(
            scale: _showAnimations ? _pulseAnimation.value : 1.0,
            child: FloatingActionButton.extended(
              onPressed: _isProcessing ? null : _pickImage,
              icon: const Icon(Icons.image),
              label: const Text('Select Image'),
              backgroundColor: Theme.of(context).primaryColor,
            ),
          );
        },
      ),
    );
  }

  Widget _buildAnimatedStats() {
    if (_detectionResults == null) return const SizedBox();
    
    // Group by class
    final classCount = <String, int>{};
    for (final detection in _detectionResults!) {
      final className = detection['className'] ?? 'Unknown';
      classCount[className] = (classCount[className] ?? 0) + 1;
    }
    
    return Column(
      children: classCount.entries.map((entry) {
        return TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: entry.value.toDouble()),
          duration: const Duration(milliseconds: 800),
          curve: Curves.easeOutCubic,
          builder: (context, value, child) {
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                children: [
                  SizedBox(
                    width: 100,
                    child: Text(
                      entry.key,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Stack(
                      children: [
                        Container(
                          height: 24,
                          decoration: BoxDecoration(
                            color: Colors.grey[300],
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        FractionallySizedBox(
                          widthFactor: value / _detectionResults!.length,
                          child: Container(
                            height: 24,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Theme.of(context).primaryColor,
                                  Theme.of(context).primaryColor.withOpacity(0.7),
                                ],
                              ),
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    value.toInt().toString(),
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ],
              ),
            );
          },
        );
      }).toList(),
    );
  }

  Future<ui.Image> _getImageInfo() async {
    final bytes = await _imageFile!.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }
}

enum VisualizationStyle {
  modern,
  neon,
  minimal,
  glass,
}

class CustomVisualizationPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  final Size imageSize;
  final VisualizationStyle style;
  final bool showConfidenceBar;
  final bool showGrid;
  final bool showHeatmap;
  final double animationValue;

  CustomVisualizationPainter({
    required this.detections,
    required this.imageSize,
    required this.style,
    required this.showConfidenceBar,
    required this.showGrid,
    required this.showHeatmap,
    required this.animationValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    // Draw grid overlay
    if (showGrid) {
      _drawGrid(canvas, size);
    }

    // Draw heatmap
    if (showHeatmap) {
      _drawHeatmap(canvas, size);
    }

    // Draw detections
    for (final detection in detections) {
      _drawDetectionByStyle(canvas, detection, scaleX, scaleY);
    }
  }

  void _drawGrid(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.2)
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;

    const gridSize = 50.0;
    for (double x = 0; x < size.width; x += gridSize) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += gridSize) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  void _drawHeatmap(Canvas canvas, Size size) {
    for (final detection in detections) {
      final normalizedBox = detection['normalizedBox'] as Map<String, dynamic>;
      final centerX = (normalizedBox['left'] + normalizedBox['right']) / 2 * size.width;
      final centerY = (normalizedBox['top'] + normalizedBox['bottom']) / 2 * size.height;
      final confidence = detection['confidence'] ?? 0.5;
      
      final gradient = RadialGradient(
        colors: [
          Colors.red.withOpacity(confidence * 0.5),
          Colors.orange.withOpacity(confidence * 0.3),
          Colors.yellow.withOpacity(confidence * 0.1),
          Colors.transparent,
        ],
      );
      
      final paint = Paint()
        ..shader = gradient.createShader(
          Rect.fromCircle(center: Offset(centerX, centerY), radius: 100),
        );
      
      canvas.drawCircle(Offset(centerX, centerY), 100, paint);
    }
  }

  void _drawDetectionByStyle(
    Canvas canvas,
    Map<String, dynamic> detection,
    double scaleX,
    double scaleY,
  ) {
    final normalizedBox = detection['normalizedBox'] as Map<String, dynamic>;
    final rect = Rect.fromLTRB(
      normalizedBox['left'] * imageSize.width * scaleX,
      normalizedBox['top'] * imageSize.height * scaleY,
      normalizedBox['right'] * imageSize.width * scaleX,
      normalizedBox['bottom'] * imageSize.height * scaleY,
    );

    switch (style) {
      case VisualizationStyle.modern:
        _drawModernStyle(canvas, rect, detection);
        break;
      case VisualizationStyle.neon:
        _drawNeonStyle(canvas, rect, detection);
        break;
      case VisualizationStyle.minimal:
        _drawMinimalStyle(canvas, rect, detection);
        break;
      case VisualizationStyle.glass:
        _drawGlassStyle(canvas, rect, detection);
        break;
    }
  }

  void _drawModernStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection) {
    final color = _getColorForClass(detection['className'] ?? '');
    
    // Animated box
    final animatedRect = Rect.fromCenter(
      center: rect.center,
      width: rect.width * animationValue,
      height: rect.height * animationValue,
    );
    
    // Background
    final bgPaint = Paint()
      ..color = color.withOpacity(0.2)
      ..style = PaintingStyle.fill;
    canvas.drawRRect(
      RRect.fromRectAndRadius(animatedRect, const Radius.circular(8)),
      bgPaint,
    );
    
    // Border with gradient
    final borderPaint = Paint()
      ..shader = LinearGradient(
        colors: [color, color.withOpacity(0.5)],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ).createShader(rect)
      ..strokeWidth = 5
      ..style = PaintingStyle.stroke;
    canvas.drawRRect(
      RRect.fromRectAndRadius(animatedRect, const Radius.circular(8)),
      borderPaint,
    );
    
    // Corner accents
    _drawCornerAccents(canvas, animatedRect, color);
    
    // Label with confidence bar
    _drawModernLabel(canvas, rect, detection, color);
  }

  void _drawNeonStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection) {
    final color = _getColorForClass(detection['className'] ?? '');
    
    // Neon glow effect
    for (int i = 3; i > 0; i--) {
      final glowPaint = Paint()
        ..color = color.withOpacity(0.3 / i)
        ..strokeWidth = i * 3.0
        ..style = PaintingStyle.stroke
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, i * 2.0);
      canvas.drawRect(rect, glowPaint);
    }
    
    // Main border
    final borderPaint = Paint()
      ..color = color
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;
    canvas.drawRect(rect, borderPaint);
    
    // Neon label
    _drawNeonLabel(canvas, rect, detection, color);
  }

  void _drawMinimalStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection) {
    final color = Colors.black87;
    
    // Simple lines at corners
    final paint = Paint()
      ..color = color
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;
    
    const cornerLength = 20.0;
    
    // Top-left
    canvas.drawLine(Offset(rect.left, rect.top), Offset(rect.left + cornerLength, rect.top), paint);
    canvas.drawLine(Offset(rect.left, rect.top), Offset(rect.left, rect.top + cornerLength), paint);
    
    // Top-right
    canvas.drawLine(Offset(rect.right - cornerLength, rect.top), Offset(rect.right, rect.top), paint);
    canvas.drawLine(Offset(rect.right, rect.top), Offset(rect.right, rect.top + cornerLength), paint);
    
    // Bottom-left
    canvas.drawLine(Offset(rect.left, rect.bottom - cornerLength), Offset(rect.left, rect.bottom), paint);
    canvas.drawLine(Offset(rect.left, rect.bottom), Offset(rect.left + cornerLength, rect.bottom), paint);
    
    // Bottom-right
    canvas.drawLine(Offset(rect.right - cornerLength, rect.bottom), Offset(rect.right, rect.bottom), paint);
    canvas.drawLine(Offset(rect.right, rect.bottom - cornerLength), Offset(rect.right, rect.bottom), paint);
    
    // Minimal label
    _drawMinimalLabel(canvas, rect, detection);
  }

  void _drawGlassStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection) {
    final color = _getColorForClass(detection['className'] ?? '');
    
    // Glass effect background
    final glassPaint = Paint()
      ..color = Colors.white.withOpacity(0.1)
      ..style = PaintingStyle.fill
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(12)),
      glassPaint,
    );
    
    // Glass border
    final borderPaint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke;
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(12)),
      borderPaint,
    );
    
    // Glass label
    _drawGlassLabel(canvas, rect, detection, color);
  }

  void _drawCornerAccents(Canvas canvas, Rect rect, Color color) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 5
      ..style = PaintingStyle.stroke;
    
    const accentLength = 15.0;
    
    // Draw corner accents
    final corners = [
      rect.topLeft,
      rect.topRight,
      rect.bottomLeft,
      rect.bottomRight,
    ];
    
    for (final corner in corners) {
      canvas.drawCircle(corner, 4, Paint()..color = color);
    }
  }

  void _drawModernLabel(Canvas canvas, Rect rect, Map<String, dynamic> detection, Color color) {
    final className = detection['className'] ?? 'Unknown';
    final confidence = detection['confidence'] ?? 0.0;
    
    final textPainter = TextPainter(
      text: TextSpan(
        text: className,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    
    final labelRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(
        rect.left,
        rect.top - 30,
        textPainter.width + 16,
        26,
      ),
      const Radius.circular(13),
    );
    
    // Label background
    canvas.drawRRect(
      labelRect,
      Paint()..color = color,
    );
    
    // Text
    textPainter.paint(
      canvas,
      Offset(rect.left + 8, rect.top - 28),
    );
    
    // Confidence bar
    if (showConfidenceBar) {
      final barRect = Rect.fromLTWH(
        rect.left,
        rect.bottom + 4,
        rect.width * confidence,
        4,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(barRect, const Radius.circular(2)),
        Paint()..color = color,
      );
    }
  }

  void _drawNeonLabel(Canvas canvas, Rect rect, Map<String, dynamic> detection, Color color) {
    final text = '${detection['className'] ?? 'Unknown'} ${((detection['confidence'] ?? 0) * 100).toStringAsFixed(0)}%';
    
    // Glow text effect
    for (int i = 3; i > 0; i--) {
      final textPainter = TextPainter(
        text: TextSpan(
          text: text,
          style: TextStyle(
            color: color.withOpacity(0.5 / i),
            fontSize: 12,
            fontWeight: FontWeight.bold,
            shadows: [
              Shadow(
                color: color.withOpacity(0.8),
                blurRadius: i * 3.0,
              ),
            ],
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      
      textPainter.paint(
        canvas,
        Offset(rect.left + 4, rect.top - 20),
      );
    }
  }

  void _drawMinimalLabel(Canvas canvas, Rect rect, Map<String, dynamic> detection) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: detection['className'] ?? 'Unknown',
        style: const TextStyle(
          color: Colors.black87,
          fontSize: 11,
          fontWeight: FontWeight.normal,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    
    // White background for text
    canvas.drawRect(
      Rect.fromLTWH(
        rect.left,
        rect.bottom + 2,
        textPainter.width + 4,
        textPainter.height + 2,
      ),
      Paint()..color = Colors.white.withOpacity(0.8),
    );
    
    textPainter.paint(
      canvas,
      Offset(rect.left + 2, rect.bottom + 3),
    );
  }

  void _drawGlassLabel(Canvas canvas, Rect rect, Map<String, dynamic> detection, Color color) {
    final text = detection['className'] ?? 'Unknown';
    final confidence = detection['confidence'] ?? 0.0;
    
    final textPainter = TextPainter(
      text: TextSpan(
        text: '$text ${(confidence * 100).toStringAsFixed(0)}%',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w500,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    
    // Glass label background
    final labelRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(
        rect.left,
        rect.top - 28,
        textPainter.width + 16,
        24,
      ),
      const Radius.circular(12),
    );
    
    canvas.drawRRect(
      labelRect,
      Paint()
        ..color = Colors.black.withOpacity(0.3)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4),
    );
    
    canvas.drawRRect(
      labelRect,
      Paint()
        ..color = Colors.white.withOpacity(0.1)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
    
    textPainter.paint(
      canvas,
      Offset(rect.left + 8, rect.top - 26),
    );
  }

  Color _getColorForClass(String className) {
    final colors = [
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.red,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.indigo,
      Colors.amber,
      Colors.cyan,
    ];
    return colors[className.hashCode % colors.length];
  }

  @override
  bool shouldRepaint(covariant CustomVisualizationPainter oldDelegate) {
    return detections != oldDelegate.detections ||
           style != oldDelegate.style ||
           showConfidenceBar != oldDelegate.showConfidenceBar ||
           showGrid != oldDelegate.showGrid ||
           showHeatmap != oldDelegate.showHeatmap ||
           animationValue != oldDelegate.animationValue;
  }
}