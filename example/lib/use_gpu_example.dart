// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Example demonstrating how to use the useGpu feature
///
/// This example shows how to:
/// 1. Create YOLO instances with GPU enabled/disabled
/// 2. Use YOLOView with GPU control
/// 3. Handle GPU-related crashes gracefully
class UseGpuExample extends StatefulWidget {
  const UseGpuExample({super.key});

  @override
  State<UseGpuExample> createState() => _UseGpuExampleState();
}

class _UseGpuExampleState extends State<UseGpuExample> {
  bool _useGpu = true;
  String _status = 'Ready';
  List<YOLOResult> _results = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('useGpu Feature Example'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // GPU Control Section
          Card(
            margin: const EdgeInsets.all(16),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'GPU Control',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Toggle GPU acceleration. Disable if you experience crashes on your device.',
                    style: TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      const Text('Use GPU: '),
                      Switch(
                        value: _useGpu,
                        onChanged: (value) {
                          setState(() {
                            _useGpu = value;
                          });
                        },
                      ),
                      Text(_useGpu ? 'Enabled' : 'Disabled'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Status: $_status',
                    style: TextStyle(
                      color: _status.contains('Error')
                          ? Colors.red
                          : Colors.green,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // YOLOView Section
          Expanded(
            child: Card(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                children: [
                  const Padding(
                    padding: EdgeInsets.all(16),
                    child: Text(
                      'Real-time Detection',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  Expanded(
                    child: YOLOView(
                      modelPath: 'yolo11n',
                      task: YOLOTask.detect,
                      useGpu: _useGpu, // Pass the useGpu parameter
                      onResult: (results) {
                        debugPrint(
                          'YOLO Detection Results: ${results.length} objects detected',
                        );
                        for (final result in results) {
                          debugPrint(
                            '  - ${result.className}: ${(result.confidence * 100).toStringAsFixed(1)}%',
                          );
                        }
                        setState(() {
                          _results = results;
                          _status = 'Detected ${results.length} objects';
                        });
                      },
                      onPerformanceMetrics: (metrics) {
                        // You can monitor performance here
                        debugPrint(
                          'FPS: ${metrics.fps}, Processing time: ${metrics.processingTimeMs}ms',
                        );
                      },
                      onStreamingData: (data) {
                        debugPrint('YOLO Streaming Data: $data');
                        // Check for any errors
                        if (data.containsKey('error')) {
                          debugPrint('YOLO Error: ${data['error']}');
                          setState(() {
                            _status = 'Error: ${data['error']}';
                          });
                        }
                        // Check for model loading status
                        if (data.containsKey('modelLoaded')) {
                          debugPrint('Model loaded: ${data['modelLoaded']}');
                          setState(() {
                            _status = 'Model loaded: ${data['modelLoaded']}';
                          });
                        }
                        // Check for camera status
                        if (data.containsKey('cameraStatus')) {
                          debugPrint('Camera status: ${data['cameraStatus']}');
                        }
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Results Section
          if (_results.isNotEmpty)
            Container(
              height: 100,
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
              ),
              child: ListView.builder(
                itemCount: _results.length,
                itemBuilder: (context, index) {
                  final result = _results[index];
                  return ListTile(
                    dense: true,
                    title: Text(result.className),
                    subtitle: Text(
                      'Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%',
                    ),
                    trailing: Text('GPU: ${_useGpu ? "On" : "Off"}'),
                  );
                },
              ),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _testSingleImageInference,
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  /// Test single image inference with GPU control
  Future<void> _testSingleImageInference() async {
    setState(() {
      _status = 'Loading model...';
    });

    try {
      // Create YOLO instance with GPU control
      final yolo = YOLO(
        modelPath: 'yolo11n',
        task: YOLOTask.detect,
        useGpu: _useGpu, // Pass the useGpu parameter
      );

      // Load the model
      final success = await yolo.loadModel();

      if (success) {
        setState(() {
          _status =
              'Model loaded successfully with GPU: ${_useGpu ? "Enabled" : "Disabled"}';
        });

        // Note: In a real app, you would load an actual image here
        // For this example, we just show that the model loaded successfully
        debugPrint('Model loaded with useGpu: $_useGpu');
      } else {
        setState(() {
          _status = 'Failed to load model';
        });
      }
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
      });
      debugPrint('Error loading model: $e');
    }
  }
}

/// Example showing how to handle GPU crashes gracefully
class GracefulGpuHandlingExample extends StatefulWidget {
  const GracefulGpuHandlingExample({super.key});

  @override
  State<GracefulGpuHandlingExample> createState() =>
      _GracefulGpuHandlingExampleState();
}

class _GracefulGpuHandlingExampleState
    extends State<GracefulGpuHandlingExample> {
  bool _useGpu = true;
  bool _hasGpuError = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Graceful GPU Handling'),
        backgroundColor: Colors.orange,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Instructions
          Card(
            margin: const EdgeInsets.all(16),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Graceful GPU Error Handling',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'This example shows how to automatically fall back to CPU if GPU causes crashes.',
                    style: TextStyle(fontSize: 14),
                  ),
                  if (_hasGpuError)
                    Container(
                      margin: const EdgeInsets.only(top: 8),
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.orange[100],
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: const Text(
                        '⚠️ GPU error detected, falling back to CPU',
                        style: TextStyle(color: Colors.orange),
                      ),
                    ),
                ],
              ),
            ),
          ),

          // YOLOView with error handling
          Expanded(
            child: YOLOView(
              modelPath: 'yolo11n',
              task: YOLOTask.detect,
              useGpu: _useGpu,
              onResult: (results) {
                // If we get results with GPU disabled, it means we successfully fell back
                if (!_useGpu && !_hasGpuError) {
                  setState(() {
                    _hasGpuError = true;
                  });
                }
              },
              onStreamingData: (data) {
                // Handle any errors in the streaming data
                if (data.containsKey('error')) {
                  final error = data['error'] as String?;
                  if (error != null && error.contains('GPU') && _useGpu) {
                    // Automatically fall back to CPU
                    setState(() {
                      _useGpu = false;
                      _hasGpuError = true;
                    });
                  }
                }
              },
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            _useGpu = !_useGpu;
            _hasGpuError = false;
          });
        },
        child: Icon(_useGpu ? Icons.settings : Icons.settings_power),
      ),
    );
  }
}
