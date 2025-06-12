import 'package:flutter/material.dart';
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
  late YOLOController _controller;
  bool _modelLoaded = false;
  double _confidence = 0.25;
  double _iou = 0.45;
  int _detectionCount = 0;
  double _zoom = 1.0;
  bool _useGpu = false;

  @override
  void initState() {
    super.initState();
    _initializeController();
  }

  Future<void> _initializeController() async {
    // Create controller with model path
    // For demo purposes, using a non-existent model to show camera-only mode
    _controller = YOLOController(
      modelPath: 'assets/models/yolo11n.tflite',
      task: YOLOTask.detect,
    );

    // Set callbacks
    _controller.onCameraStreamChanged = (cameraInfo) {
      debugPrint('Camera info: ${cameraInfo.currentCameraType}');
    };

    _controller.onResultsChanged = (results) {
      setState(() {
        _detectionCount = results.length;
      });
    };
  }

  void _toggleCamera() {
    _controller.toggleCamera();
  }

  void _updateConfidence(double value) {
    setState(() {
      _confidence = value;
    });
    _controller.setConfidenceThreshold(value);
  }

  void _updateIoU(double value) {
    setState(() {
      _iou = value;
    });
    _controller.setIoUThreshold(value);
  }

  void _updateZoom(double value) {
    setState(() {
      _zoom = value;
    });
    _controller.setZoomRatio(value);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
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
          // Camera view
          Expanded(
            child: Container(
              color: Colors.black,
              child: YOLOView(
                controller: _controller,
                onModelLoaded: () {
                  setState(() {
                    _modelLoaded = true;
                  });
                },
              ),
            ),
          ),
          // Controls
          Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                // Detection info
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Column(
                        children: [
                          const Text('Detections'),
                          Text(
                            '$_detectionCount',
                            style: Theme.of(context).textTheme.headlineMedium,
                          ),
                        ],
                      ),
                      Column(
                        children: [
                          const Text('Model Status'),
                          Text(
                            _modelLoaded ? 'Loaded' : 'Not Loaded',
                            style: Theme.of(context).textTheme.bodyLarge,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const Divider(),
                // Confidence slider
                Row(
                  children: [
                    const SizedBox(width: 16),
                    const Icon(Icons.confidence),
                    const SizedBox(width: 8),
                    const Text('Confidence'),
                    Expanded(
                      child: Slider(
                        value: _confidence,
                        min: 0.0,
                        max: 1.0,
                        divisions: 20,
                        label: _confidence.toStringAsFixed(2),
                        onChanged: _updateConfidence,
                      ),
                    ),
                    SizedBox(
                      width: 50,
                      child: Text(
                        _confidence.toStringAsFixed(2),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
                // IoU slider
                Row(
                  children: [
                    const SizedBox(width: 16),
                    const Icon(Icons.layers),
                    const SizedBox(width: 8),
                    const Text('IoU'),
                    Expanded(
                      child: Slider(
                        value: _iou,
                        min: 0.0,
                        max: 1.0,
                        divisions: 20,
                        label: _iou.toStringAsFixed(2),
                        onChanged: _updateIoU,
                      ),
                    ),
                    SizedBox(
                      width: 50,
                      child: Text(
                        _iou.toStringAsFixed(2),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
                // Zoom slider
                Row(
                  children: [
                    const SizedBox(width: 16),
                    const Icon(Icons.zoom_in),
                    const SizedBox(width: 8),
                    const Text('Zoom'),
                    Expanded(
                      child: Slider(
                        value: _zoom,
                        min: 1.0,
                        max: 5.0,
                        divisions: 16,
                        label: '${_zoom.toStringAsFixed(1)}x',
                        onChanged: _updateZoom,
                      ),
                    ),
                    SizedBox(
                      width: 50,
                      child: Text(
                        '${_zoom.toStringAsFixed(1)}x',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
                const Divider(),
                // Action buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _toggleCamera,
                      icon: const Icon(Icons.cameraswitch),
                      label: const Text('Switch Camera'),
                    ),
                    ElevatedButton.icon(
                      onPressed: () {
                        setState(() {
                          _useGpu = !_useGpu;
                        });
                        _controller.useGpu(_useGpu);
                      },
                      icon: Icon(_useGpu ? Icons.gpu_on : Icons.gpu_off),
                      label: Text(_useGpu ? 'GPU On' : 'GPU Off'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}