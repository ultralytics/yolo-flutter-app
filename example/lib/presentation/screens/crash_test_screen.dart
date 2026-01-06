// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';
import '../../models/models.dart';
import '../../services/model_manager.dart';

/// Test screen to verify the fix for the crash when YOLOView is disposed
/// while TensorFlow Lite inference is running.
///
/// This screen tests:
/// - Navigating away while inference is active
/// - App lifecycle changes (pause/resume)
/// - Rapid navigation scenarios
class CrashTestScreen extends StatefulWidget {
  const CrashTestScreen({super.key});

  @override
  State<CrashTestScreen> createState() => _CrashTestScreenState();
}

class _CrashTestScreenState extends State<CrashTestScreen>
    with WidgetsBindingObserver {
  YOLOViewController? _controller;
  bool _isInitialized = false;
  int _detectionCount = 0;
  String? _lastError;
  String? _modelPath;
  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _modelManager = ModelManager();
    _controller = YOLOViewController();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      setState(() {
        _isInitialized = false;
        _lastError = null;
      });

      final modelPath = await _modelManager.getModelPath(ModelType.detect);

      if (mounted) {
        setState(() {
          _modelPath = modelPath;
          _isInitialized = modelPath != null;
          if (modelPath == null) {
            _lastError = 'Failed to load model';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _lastError = e.toString();
          _isInitialized = false;
        });
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Test: Stop controller before disposing
    _controller?.stop();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    if (state == AppLifecycleState.paused) {
      // Stop controller when app goes to background
      _controller?.stop();
    } else if (state == AppLifecycleState.resumed) {
      // Restart when app comes back to foreground
      setState(() {
        _controller = YOLOViewController();
      });
      _loadModel();
    }
  }

  void _onDetectionResult(List<YOLOResult> results) {
    if (!mounted) return;

    setState(() {
      _detectionCount += results.length;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crash Test Screen'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              // Test rapid restart scenario
              _controller?.stop();
              setState(() {
                _controller = YOLOViewController();
                _detectionCount = 0;
              });
              _loadModel();
            },
            tooltip: 'Restart (tests rapid disposal)',
          ),
        ],
      ),
      body: Column(
        children: [
          // Status info
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.blue.shade50,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Status: ${_isInitialized ? "Initialized" : "Not Initialized"}',
                ),
                Text('Detections: $_detectionCount'),
                if (_lastError != null)
                  Text(
                    'Error: $_lastError',
                    style: const TextStyle(color: Colors.red),
                  ),
              ],
            ),
          ),

          // YOLOView
          Expanded(
            child: _isInitialized && _controller != null && _modelPath != null
                ? YOLOView(
                    controller: _controller!,
                    task: YOLOTask.detect,
                    modelPath: _modelPath!,
                    useGpu: false,
                    showOverlays: true,
                    streamingConfig: YOLOStreamingConfig.custom(
                      includeDetections: true,
                      throttleInterval: Duration.zero,
                      maxFPS: 20,
                      inferenceFrequency: 10,
                    ),
                    onResult: _onDetectionResult,
                  )
                : Center(
                    child: _lastError != null
                        ? Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                'Error: $_lastError',
                                style: const TextStyle(color: Colors.red),
                              ),
                              const SizedBox(height: 16),
                              ElevatedButton(
                                onPressed: _loadModel,
                                child: const Text('Retry'),
                              ),
                            ],
                          )
                        : const CircularProgressIndicator(),
                  ),
          ),

          // Test buttons
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                ElevatedButton(
                  onPressed: () {
                    // Test navigating away while inference is active
                    Navigator.of(context).pop();
                  },
                  child: const Text('Navigate Back (Test Crash)'),
                ),
                const SizedBox(height: 8),
                ElevatedButton(
                  onPressed: () {
                    // Test showing dialog (causes widget rebuild)
                    showDialog(
                      context: context,
                      builder: (context) => AlertDialog(
                        title: const Text('Test Dialog'),
                        content: const Text(
                          'This dialog causes widget rebuild. '
                          'The app should not crash when dismissed.',
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text('OK'),
                          ),
                        ],
                      ),
                    );
                  },
                  child: const Text('Show Dialog (Test Rebuild)'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
