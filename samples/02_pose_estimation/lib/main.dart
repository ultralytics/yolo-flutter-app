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
      title: 'YOLO Pose Estimation',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const PoseEstimationScreen(),
    );
  }
}

class PoseEstimationScreen extends StatefulWidget {
  const PoseEstimationScreen({super.key});

  @override
  State<PoseEstimationScreen> createState() => _PoseEstimationScreenState();
}

class _PoseEstimationScreenState extends State<PoseEstimationScreen> {
  final ImagePicker _picker = ImagePicker();
  
  // State
  File? _imageFile;
  List<YOLOResult>? _poseResults;
  bool _isProcessing = false;
  String _processingTime = '';
  
  // YOLO instance
  late final YOLO _yolo;
  
  // Pose keypoint connections for drawing skeleton
  static const List<List<int>> _skeleton = [
    [0, 1], [0, 2], [1, 3], [2, 4], // Head
    [5, 6], [5, 11], [6, 12], [11, 12], // Body
    [5, 7], [7, 9], [6, 8], [8, 10], // Arms
    [11, 13], [13, 15], [12, 14], [14, 16], // Legs
  ];
  
  // Keypoint names for display
  static const List<String> _keypointNames = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
  ];

  @override
  void initState() {
    super.initState();
    _yolo = YOLO(
      modelPath: 'yolo11n-pose.tflite',
      task: YOLOTask.pose,
    );
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

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _poseResults = null;
        _processingTime = '';
      });
      _detectPoses();
    }
  }

  Future<void> _detectPoses() async {
    if (_imageFile == null) return;

    setState(() => _isProcessing = true);
    final stopwatch = Stopwatch()..start();

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final result = await _yolo.predict(
        imageBytes,
        confidenceThreshold: 0.5,
        iouThreshold: 0.45,
      );

      stopwatch.stop();
      
      setState(() {
        _poseResults = _parseResults(result);
        _processingTime = '${stopwatch.elapsedMilliseconds}ms';
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error during pose detection: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  List<YOLOResult> _parseResults(Map<String, dynamic> result) {
    final detections = result['detections'] as List<dynamic>?;
    if (detections == null) return [];

    return detections.map((detection) {
      return YOLOResult.fromMap(detection as Map<dynamic, dynamic>);
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Pose Estimation'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Controls
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey[100],
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _isProcessing ? null : () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                    ),
                    ElevatedButton.icon(
                      onPressed: _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                    ),
                  ],
                ),
                if (_processingTime.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(
                      'Processing time: $_processingTime',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
              ],
            ),
          ),
          
          // Image and pose visualization
          Expanded(
            child: _isProcessing
                ? const Center(child: CircularProgressIndicator())
                : _imageFile == null
                    ? const Center(
                        child: Text(
                          'Select an image to detect poses',
                          style: TextStyle(fontSize: 18, color: Colors.grey),
                        ),
                      )
                    : SingleChildScrollView(
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          children: [
                            // Image with pose overlay
                            FutureBuilder<ui.Image>(
                              future: _getImageInfo(),
                              builder: (context, snapshot) {
                                if (!snapshot.hasData) {
                                  return Container(
                                    constraints: BoxConstraints(
                                      maxHeight: MediaQuery.of(context).size.height * 0.5,
                                    ),
                                    child: Image.file(
                                      _imageFile!,
                                      fit: BoxFit.contain,
                                    ),
                                  );
                                }
                                
                                // Calculate scale to fit image within screen bounds
                                final screenWidth = MediaQuery.of(context).size.width - 32; // Account for padding
                                final maxHeight = MediaQuery.of(context).size.height * 0.5;
                                final imageWidth = snapshot.data!.width.toDouble();
                                final imageHeight = snapshot.data!.height.toDouble();
                                
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
                                    painter: PosePainter(
                                      image: snapshot.data!,
                                      poseResults: _poseResults ?? [],
                                      skeleton: _skeleton,
                                      scale: scale,
                                    ),
                                  ),
                                );
                              },
                            ),
                            
                            // Detection info
                            if (_poseResults != null && _poseResults!.isNotEmpty)
                              Padding(
                                padding: const EdgeInsets.only(top: 16),
                                child: Card(
                                  child: Padding(
                                    padding: const EdgeInsets.all(16),
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'Detected ${_poseResults!.length} ${_poseResults!.length == 1 ? 'person' : 'people'}',
                                          style: Theme.of(context).textTheme.titleMedium,
                                        ),
                                        const SizedBox(height: 8),
                                        ..._poseResults!.asMap().entries.map((entry) {
                                          final index = entry.key;
                                          final pose = entry.value;
                                          return _buildPoseInfo(index + 1, pose);
                                        }),
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

  Widget _buildPoseInfo(int personNumber, YOLOResult pose) {
    final visibleKeypoints = <String>[];
    if (pose.keypoints != null && pose.keypointConfidences != null) {
      for (int i = 0; i < pose.keypoints!.length && i < _keypointNames.length; i++) {
        if (pose.keypointConfidences![i] > 0.5) {
          visibleKeypoints.add(_keypointNames[i]);
        }
      }
    }
    
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Person $personNumber (${(pose.confidence * 100).toStringAsFixed(1)}% confidence)',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          if (visibleKeypoints.isNotEmpty)
            Text(
              'Visible keypoints: ${visibleKeypoints.join(', ')}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
        ],
      ),
    );
  }

  Future<ui.Image> _getImageInfo() async {
    final bytes = await _imageFile!.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }
}

class PosePainter extends CustomPainter {
  final ui.Image image;
  final List<YOLOResult> poseResults;
  final List<List<int>> skeleton;
  final double scale;

  PosePainter({
    required this.image,
    required this.poseResults,
    required this.skeleton,
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
    
    // Draw poses
    for (final pose in poseResults) {
      if (pose.keypoints == null || pose.keypointConfidences == null) continue;
      
      // Draw skeleton connections
      final connectionPaint = Paint()
        ..color = Colors.green
        ..strokeWidth = 3
        ..style = PaintingStyle.stroke;
      
      for (final connection in skeleton) {
        final startIdx = connection[0];
        final endIdx = connection[1];
        
        if (startIdx < pose.keypoints!.length && 
            endIdx < pose.keypoints!.length &&
            pose.keypointConfidences![startIdx] > 0.5 &&
            pose.keypointConfidences![endIdx] > 0.5) {
          
          canvas.drawLine(
            Offset(pose.keypoints![startIdx].x, pose.keypoints![startIdx].y),
            Offset(pose.keypoints![endIdx].x, pose.keypoints![endIdx].y),
            connectionPaint,
          );
        }
      }
      
      // Draw keypoints
      for (int i = 0; i < pose.keypoints!.length; i++) {
        if (pose.keypointConfidences![i] > 0.5) {
          final keypoint = pose.keypoints![i];
          
          // Draw keypoint circle
          final keypointPaint = Paint()
            ..color = _getKeypointColor(i)
            ..style = PaintingStyle.fill;
          
          canvas.drawCircle(
            Offset(keypoint.x, keypoint.y),
            5,
            keypointPaint,
          );
          
          // Draw keypoint border
          final borderPaint = Paint()
            ..color = Colors.white
            ..strokeWidth = 2
            ..style = PaintingStyle.stroke;
          
          canvas.drawCircle(
            Offset(keypoint.x, keypoint.y),
            5,
            borderPaint,
          );
        }
      }
      
      // Draw bounding box
      final boxPaint = Paint()
        ..color = Colors.blue.withOpacity(0.3)
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke;
      
      canvas.drawRect(pose.boundingBox, boxPaint);
    }
    
    // Restore canvas state
    canvas.restore();
  }

  Color _getKeypointColor(int index) {
    // Color keypoints by body part
    if (index < 5) return Colors.red; // Head
    if (index < 11) return Colors.orange; // Arms
    return Colors.yellow; // Legs
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) {
    return image != oldDelegate.image || 
           poseResults != oldDelegate.poseResults;
  }
}