// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'models/model_type.dart';
import 'services/model_manager.dart';

/// Dual model comparison screen showing annotated images from two YOLO models
class DualModelScreen extends StatefulWidget {
  const DualModelScreen({super.key});

  @override
  State<DualModelScreen> createState() => _DualModelScreenState();
}

class _DualModelScreenState extends State<DualModelScreen> {
  // Camera
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isCapturing = false;
  
  // YOLO instances - using new multi-instance feature
  YOLO? _yolo1;
  YOLO? _yolo2;
  bool _isModel1Loaded = false;
  bool _isModel2Loaded = false;
  
  // Model selection
  ModelType _model1Type = ModelType.detect;
  ModelType _model2Type = ModelType.segment;
  
  // Results
  Uint8List? _annotatedImage1;
  Uint8List? _annotatedImage2;
  int _detectionCount1 = 0;
  int _detectionCount2 = 0;
  
  // Performance
  double _processingTime1 = 0;
  double _processingTime2 = 0;
  double _fps = 0;
  
  // UI state
  bool _isLoading = false;
  String _statusMessage = '';
  Timer? _captureTimer;
  bool _isAutoCapture = true;
  
  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();
    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        setState(() {
          _statusMessage = 'Downloading models: ${(progress * 100).toInt()}%';
        });
      },
      onStatusUpdate: (message) {
        setState(() {
          _statusMessage = message;
        });
      },
    );
    _initializeCamera();
    _loadModels();
  }

  @override
  void dispose() {
    _captureTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _statusMessage = 'No cameras available';
        });
        return;
      }

      _cameraController = CameraController(
        cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      
      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
        
        // Start automatic capture
        _startAutomaticCapture();
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Camera initialization failed: $e';
      });
    }
  }

  Future<void> _loadModels() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Loading models...';
    });

    try {
      // Load model 1 (Detection)
      final model1Path = await _modelManager.getModelPath(_model1Type);
      _yolo1 = YOLO(
        modelPath: model1Path,
        task: _model1Type.task,
        useMultiInstance: true, // Enable multi-instance support
      );
      await _yolo1!.loadModel();
      setState(() {
        _isModel1Loaded = true;
      });
      
      // Load model 2 (Segmentation)
      final model2Path = await _modelManager.getModelPath(_model2Type);
      _yolo2 = YOLO(
        modelPath: model2Path,
        task: _model2Type.task,
        useMultiInstance: true, // Enable multi-instance support
      );
      await _yolo2!.loadModel();
      setState(() {
        _isModel2Loaded = true;
      });

      setState(() {
        _isLoading = false;
        _statusMessage = 'Models loaded successfully';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Failed to load models: $e';
      });
    }
  }

  void _startAutomaticCapture() {
    if (_isAutoCapture) {
      _captureTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
        if (_isCameraInitialized && !_isCapturing && _isModel1Loaded && _isModel2Loaded) {
          _captureAndProcess();
        }
      });
    }
  }

  void _stopAutomaticCapture() {
    _captureTimer?.cancel();
  }

  Future<void> _captureAndProcess() async {
    if (_isCapturing || _cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    final stopwatch = Stopwatch()..start();

    try {
      final image = await _cameraController!.takePicture();
      final imageBytes = await image.readAsBytes();

      // Process with both models simultaneously
      final futures = <Future<Map<String, dynamic>>>[];
      
      if (_yolo1 != null) {
        futures.add(_yolo1!.predict(imageBytes));
      }
      
      if (_yolo2 != null) {
        futures.add(_yolo2!.predict(imageBytes));
      }

      final results = await Future.wait(futures);
      
      stopwatch.stop();
      
      if (mounted && results.length >= 2) {
        setState(() {
          // Update FPS
          _fps = 1000 / stopwatch.elapsedMilliseconds;
          
          // Model 1 results
          final result1 = results[0];
          if (result1['annotatedImage'] != null) {
            _annotatedImage1 = (result1['annotatedImage'] as TypedData).buffer.asUint8List();
          }
          final boxes1 = result1['boxes'] as List<dynamic>? ?? [];
          _detectionCount1 = boxes1.length;
          _processingTime1 = stopwatch.elapsedMilliseconds / 2.0; // Approximate
          
          // Model 2 results
          final result2 = results[1];
          if (result2['annotatedImage'] != null) {
            _annotatedImage2 = (result2['annotatedImage'] as TypedData).buffer.asUint8List();
          }
          final boxes2 = result2['boxes'] as List<dynamic>? ?? [];
          _detectionCount2 = boxes2.length;
          _processingTime2 = stopwatch.elapsedMilliseconds / 2.0; // Approximate
        });
      }
    } catch (e) {
      debugPrint('Error during capture and processing: $e');
    } finally {
      setState(() {
        _isCapturing = false;
      });
    }
  }

  Future<void> _switchModel1(ModelType newType) async {
    if (newType == _model1Type) return;
    
    setState(() {
      _model1Type = newType;
      _isModel1Loaded = false;
      _annotatedImage1 = null;
      _statusMessage = 'Loading ${newType.displayName} for Model 1...';
    });

    try {
      final modelPath = await _modelManager.getModelPath(newType);
      _yolo1 = YOLO(
        modelPath: modelPath,
        task: newType.task,
        useMultiInstance: true,
      );
      await _yolo1!.loadModel();
      
      setState(() {
        _isModel1Loaded = true;
        _statusMessage = 'Model 1 updated to ${newType.displayName}';
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Failed to load ${newType.displayName}: $e';
      });
    }
  }

  Future<void> _switchModel2(ModelType newType) async {
    if (newType == _model2Type) return;
    
    setState(() {
      _model2Type = newType;
      _isModel2Loaded = false;
      _annotatedImage2 = null;
      _statusMessage = 'Loading ${newType.displayName} for Model 2...';
    });

    try {
      final modelPath = await _modelManager.getModelPath(newType);
      _yolo2 = YOLO(
        modelPath: modelPath,
        task: newType.task,
        useMultiInstance: true,
      );
      await _yolo2!.loadModel();
      
      setState(() {
        _isModel2Loaded = true;
        _statusMessage = 'Model 2 updated to ${newType.displayName}';
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Failed to load ${newType.displayName}: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Multi-Instance Dual Model Test'),
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
        actions: [
          // FPS and status display
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Center(
              child: Text(
                '${_fps.toStringAsFixed(1)} FPS',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Status bar
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(8),
            color: Colors.grey[900],
            child: Text(
              _statusMessage.isEmpty ? 'Ready for comparison' : _statusMessage,
              style: const TextStyle(color: Colors.white70, fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ),
          
          // Model selectors
          Container(
            padding: const EdgeInsets.all(8),
            color: Colors.grey[800],
            child: Row(
              children: [
                // Model 1 selector
                Expanded(
                  child: Column(
                    children: [
                      Text('Model 1: ${_model1Type.displayName}', 
                           style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      DropdownButton<ModelType>(
                        value: _model1Type,
                        isExpanded: true,
                        dropdownColor: Colors.grey[700],
                        style: const TextStyle(color: Colors.white),
                        items: ModelType.values.map((type) {
                          return DropdownMenuItem(
                            value: type,
                            child: Text(type.displayName),
                          );
                        }).toList(),
                        onChanged: (type) {
                          if (type != null) {
                            _switchModel1(type);
                          }
                        },
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                // Model 2 selector
                Expanded(
                  child: Column(
                    children: [
                      Text('Model 2: ${_model2Type.displayName}', 
                           style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      DropdownButton<ModelType>(
                        value: _model2Type,
                        isExpanded: true,
                        dropdownColor: Colors.grey[700],
                        style: const TextStyle(color: Colors.white),
                        items: ModelType.values.map((type) {
                          return DropdownMenuItem(
                            value: type,
                            child: Text(type.displayName),
                          );
                        }).toList(),
                        onChanged: (type) {
                          if (type != null) {
                            _switchModel2(type);
                          }
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          
          // Results display
          Expanded(
            child: _isLoading
                ? const Center(
                    child: CircularProgressIndicator(color: Colors.white)
                  )
                : Row(
                    children: [
                      // Model 1 results
                      Expanded(
                        child: Column(
                          children: [
                            // Header
                            Container(
                              width: double.infinity,
                              padding: const EdgeInsets.all(8),
                              color: Colors.blue[900],
                              child: Column(
                                children: [
                                  Text(
                                    _model1Type.displayName,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Text(
                                    'Detections: $_detectionCount1 | ${_processingTime1.toStringAsFixed(0)}ms',
                                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                                  ),
                                ],
                              ),
                            ),
                            // Annotated image
                            Expanded(
                              child: Container(
                                width: double.infinity,
                                color: Colors.black,
                                child: _annotatedImage1 != null
                                    ? Image.memory(
                                        _annotatedImage1!,
                                        fit: BoxFit.contain,
                                      )
                                    : Center(
                                        child: Column(
                                          mainAxisAlignment: MainAxisAlignment.center,
                                          children: [
                                            Icon(
                                              _isModel1Loaded ? Icons.camera_alt : Icons.download,
                                              color: Colors.white38,
                                              size: 48,
                                            ),
                                            const SizedBox(height: 8),
                                            Text(
                                              _isModel1Loaded ? 'No image' : 'Loading...',
                                              style: const TextStyle(color: Colors.white38),
                                            ),
                                          ],
                                        ),
                                      ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      
                      // Divider
                      Container(
                        width: 2,
                        color: Colors.grey[600],
                      ),
                      
                      // Model 2 results
                      Expanded(
                        child: Column(
                          children: [
                            // Header
                            Container(
                              width: double.infinity,
                              padding: const EdgeInsets.all(8),
                              color: Colors.green[900],
                              child: Column(
                                children: [
                                  Text(
                                    _model2Type.displayName,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Text(
                                    'Detections: $_detectionCount2 | ${_processingTime2.toStringAsFixed(0)}ms',
                                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                                  ),
                                ],
                              ),
                            ),
                            // Annotated image
                            Expanded(
                              child: Container(
                                width: double.infinity,
                                color: Colors.black,
                                child: _annotatedImage2 != null
                                    ? Image.memory(
                                        _annotatedImage2!,
                                        fit: BoxFit.contain,
                                      )
                                    : Center(
                                        child: Column(
                                          mainAxisAlignment: MainAxisAlignment.center,
                                          children: [
                                            Icon(
                                              _isModel2Loaded ? Icons.camera_alt : Icons.download,
                                              color: Colors.white38,
                                              size: 48,
                                            ),
                                            const SizedBox(height: 8),
                                            Text(
                                              _isModel2Loaded ? 'No image' : 'Loading...',
                                              style: const TextStyle(color: Colors.white38),
                                            ),
                                          ],
                                        ),
                                      ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
          ),
          
          // Control buttons
          Container(
            padding: const EdgeInsets.all(8),
            color: Colors.grey[900],
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: Icon(_isAutoCapture ? Icons.pause : Icons.play_arrow),
                  label: Text(_isAutoCapture ? 'Pause' : 'Resume'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isAutoCapture ? Colors.orange : Colors.green,
                  ),
                  onPressed: () {
                    setState(() {
                      _isAutoCapture = !_isAutoCapture;
                    });
                    if (_isAutoCapture) {
                      _startAutomaticCapture();
                    } else {
                      _stopAutomaticCapture();
                    }
                  },
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Capture'),
                  onPressed: _isCameraInitialized && _isModel1Loaded && _isModel2Loaded && !_isCapturing
                      ? _captureAndProcess
                      : null,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}