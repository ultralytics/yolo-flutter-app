import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Camera Detection',
      theme: ThemeData(
        primarySwatch: Colors.orange,
        useMaterial3: true,
      ),
      home: const CameraDetectionScreen(),
    );
  }
}

class CameraDetectionScreen extends StatefulWidget {
  const CameraDetectionScreen({super.key});

  @override
  State<CameraDetectionScreen> createState() => _CameraDetectionScreenState();
}

class _CameraDetectionScreenState extends State<CameraDetectionScreen> {
  // YOLO instance
  late final YOLOView _yoloView;
  
  // Settings
  double _confidenceThreshold = 0.45;
  double _iouThreshold = 0.5;
  bool _showLabels = true;
  bool _showConfidence = true;
  bool _showFPS = true;
  bool _isCameraActive = false;
  bool _modelLoaded = false;
  
  // Performance metrics
  double _currentFPS = 0.0;
  double _inferenceTime = 0.0;
  int _objectCount = 0;

  @override
  void initState() {
    super.initState();
    _initializeYOLO();
    _checkPermissions();
  }

  void _initializeYOLO() {
    _yoloView = YOLOView(
      modelPath: 'yolo11n.tflite',
      task: YOLOTask.detect,
      streamingConfig: YOLOStreamingConfig(
        confidenceThreshold: _confidenceThreshold,
        iouThreshold: _iouThreshold,
        showLabels: _showLabels,
        showConfidence: _showConfidence,
        showFPS: _showFPS,
      ),
      onLoad: () {
        setState(() {
          _modelLoaded = true;
        });
        print('YOLO model loaded successfully');
      },
      onError: (error) {
        print('Error loading model: $error');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error: $error')),
          );
        }
      },
      onFPSUpdate: (fps) {
        setState(() {
          _currentFPS = fps;
        });
      },
      onInferenceTimeUpdate: (time) {
        setState(() {
          _inferenceTime = time;
        });
      },
      onResultsUpdate: (results) {
        // Count detected objects
        if (results['detections'] != null) {
          setState(() {
            _objectCount = (results['detections'] as List).length;
          });
        }
      },
    );
  }

  Future<void> _checkPermissions() async {
    final status = await Permission.camera.status;
    if (!status.isGranted) {
      final result = await Permission.camera.request();
      if (!result.isGranted && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Camera permission is required for detection'),
          ),
        );
      }
    }
  }

  void _updateSettings() {
    _yoloView.updateStreamingConfig(
      YOLOStreamingConfig(
        confidenceThreshold: _confidenceThreshold,
        iouThreshold: _iouThreshold,
        showLabels: _showLabels,
        showConfidence: _showConfidence,
        showFPS: _showFPS,
      ),
    );
  }

  void _toggleCamera() {
    setState(() {
      _isCameraActive = !_isCameraActive;
    });
    
    if (_isCameraActive) {
      _yoloView.startCamera();
    } else {
      _yoloView.stopCamera();
    }
  }

  @override
  void dispose() {
    _yoloView.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO Camera Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Camera view
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey, width: 2),
              ),
              child: Stack(
                children: [
                  // YOLO camera view
                  _yoloView,
                  
                  // Performance overlay
                  if (_isCameraActive)
                    Positioned(
                      top: 8,
                      right: 8,
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              'FPS: ${_currentFPS.toStringAsFixed(1)}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 14,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              'Inference: ${_inferenceTime.toStringAsFixed(0)}ms',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                              ),
                            ),
                            Text(
                              'Objects: $_objectCount',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  
                  // Loading indicator
                  if (!_modelLoaded)
                    const Center(
                      child: CircularProgressIndicator(),
                    ),
                ],
              ),
            ),
          ),
          
          // Controls
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                // Detection settings
                Row(
                  children: [
                    const Icon(Icons.confidence, size: 20),
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
                          _updateSettings();
                        },
                      ),
                    ),
                    Text(_confidenceThreshold.toStringAsFixed(2)),
                  ],
                ),
                
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
                          _updateSettings();
                        },
                      ),
                    ),
                    Text(_iouThreshold.toStringAsFixed(2)),
                  ],
                ),
                
                // Display options
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: [
                    FilterChip(
                      label: const Text('Labels'),
                      selected: _showLabels,
                      onSelected: (value) {
                        setState(() => _showLabels = value);
                        _updateSettings();
                      },
                    ),
                    FilterChip(
                      label: const Text('Confidence'),
                      selected: _showConfidence,
                      onSelected: (value) {
                        setState(() => _showConfidence = value);
                        _updateSettings();
                      },
                    ),
                    FilterChip(
                      label: const Text('FPS'),
                      selected: _showFPS,
                      onSelected: (value) {
                        setState(() => _showFPS = value);
                        _updateSettings();
                      },
                    ),
                  ],
                ),
                
                // Camera control button
                const SizedBox(height: 16),
                ElevatedButton.icon(
                  onPressed: _modelLoaded ? _toggleCamera : null,
                  icon: Icon(_isCameraActive ? Icons.stop : Icons.camera_alt),
                  label: Text(_isCameraActive ? 'Stop Camera' : 'Start Camera'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isCameraActive ? Colors.red : Colors.green,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
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