# Front Camera Usage and Frame Rate Control Guide

This guide explains how to properly use the front camera with YOLO detection and control the frame rate for better performance.

## Issues and Solutions

### 1. Frame Rate Control (YOLOStreamingConfig not working)

**Problem**: Your `skipFrames: 149` setting had no effect.

**Solution**: Use the correct parameters in `YOLOStreamingConfig`:

#### Option A: Use `inferenceFrequency` (Recommended)
```dart
YOLOView(
  controller: _controller,
  streamingConfig: const YOLOStreamingConfig(
    inferenceFrequency: 5, // Process 5 frames per second
    // or use 10 for 10 FPS, 15 for 15 FPS, etc.
  ),
  modelPath: 'best_float32x.tflite',
  task: YOLOTask.detect,
  confidenceThreshold: 0.5,
  onResult: (results) {
    if (results.isNotEmpty) {
      print('Detected ${results.length} objects');
    }
  },
),
```

#### Option B: Use `maxFPS` for output throttling
```dart
YOLOView(
  controller: _controller,
  streamingConfig: const YOLOStreamingConfig(
    maxFPS: 5, // Limit output to 5 FPS
  ),
  modelPath: 'best_float32x.tflite',
  task: YOLOTask.detect,
  confidenceThreshold: 0.5,
  onResult: (results) {
    if (results.isNotEmpty) {
      print('Detected ${results.length} objects');
    }
  },
),
```

#### Option C: Use `skipFrames` with reasonable values
```dart
YOLOView(
  controller: _controller,
  streamingConfig: const YOLOStreamingConfig(
    skipFrames: 2, // Process every 3rd frame (skip 2 frames)
    // or skipFrames: 4 for every 5th frame
  ),
  modelPath: 'best_float32x.tflite',
  task: YOLOTask.detect,
  confidenceThreshold: 0.5,
  onResult: (results) {
    if (results.isNotEmpty) {
      print('Detected ${results.length} objects');
    }
  },
),
```

### 2. Front Camera Bounding Box Issues

**Problem**: Bounding boxes appear incorrectly positioned on front camera.

**Solution**: The native code already handles front camera coordinate transformation. The issue might be related to:

1. **Camera initialization**: Ensure the front camera is properly initialized
2. **Coordinate system**: The native implementation automatically flips coordinates for front camera
3. **Model compatibility**: Some models may need retraining for front camera usage

## Complete Working Example

Here's a complete example that addresses both issues:

```dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';

class FrontCameraExample extends StatefulWidget {
  const FrontCameraExample({super.key});

  @override
  State<FrontCameraExample> createState() => _FrontCameraExampleState();
}

class _FrontCameraExampleState extends State<FrontCameraExample> {
  final _controller = YOLOViewController();
  int _detectionCount = 0;
  double _currentFps = 0.0;
  int _targetFps = 5;

  // Streaming config for slower detection
  late YOLOStreamingConfig _streamingConfig;

  @override
  void initState() {
    super.initState();
    _updateStreamingConfig(_targetFps);
  }

  void _updateStreamingConfig(int fps) {
    _streamingConfig = YOLOStreamingConfig(
      inferenceFrequency: fps, // Control how often inference runs
      maxFPS: fps, // Control how often results are sent to Flutter
      includeDetections: true,
      includeClassifications: true,
      includeProcessingTimeMs: true,
      includeFps: true,
      includeMasks: false,
      includePoses: false,
      includeOBB: false,
      includeOriginalImage: false,
    );
    
    _controller.setStreamingConfig(_streamingConfig);
    setState(() {
      _targetFps = fps;
    });
  }

  void _onDetectionResults(List<YOLOResult> results) {
    setState(() {
      _detectionCount = results.length;
    });
    
    // Debug output
    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        'Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }
  }

  void _onPerformanceMetrics(YOLOPerformanceMetrics metrics) {
    setState(() {
      _currentFps = metrics.fps;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Front Camera Example'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        children: [
          // YOLO View with front camera configuration
          YOLOView(
            controller: _controller,
            streamingConfig: _streamingConfig,
            modelPath: 'assets/models/yolov8n.tflite', // Update with your model path
            task: YOLOTask.detect,
            confidenceThreshold: 0.5,
            onResult: _onDetectionResults,
            onPerformanceMetrics: _onPerformanceMetrics,
          ),

          // Top info overlay
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.7),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Detections: $_detectionCount',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    'FPS: ${_currentFps.toStringAsFixed(1)}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    'Target: $_targetFps',
                    style: const TextStyle(
                      color: Colors.yellow,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Control buttons
          Positioned(
            bottom: 32,
            right: 16,
            child: Column(
              children: [
                // Frame rate control
                FloatingActionButton(
                  heroTag: 'fps',
                  onPressed: () {
                    // Cycle through frame rates: 5 -> 10 -> 15 -> 30 -> 5
                    int nextFps;
                    if (_targetFps <= 5) {
                      nextFps = 10;
                    } else if (_targetFps <= 10) {
                      nextFps = 15;
                    } else if (_targetFps <= 15) {
                      nextFps = 30;
                    } else {
                      nextFps = 5;
                    }
                    _updateStreamingConfig(nextFps);
                  },
                  backgroundColor: Colors.blue,
                  child: const Icon(Icons.speed, color: Colors.white),
                ),
                const SizedBox(height: 16),
                
                // Camera switch
                FloatingActionButton(
                  heroTag: 'camera',
                  onPressed: () {
                    _controller.switchCamera();
                  },
                  backgroundColor: Colors.green,
                  child: const Icon(Icons.switch_camera, color: Colors.white),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
```

## Key Points

### Frame Rate Control
- **`inferenceFrequency`**: Controls how often the model runs inference (most important for performance)
- **`maxFPS`**: Controls how often results are sent to Flutter
- **`skipFrames`**: Alternative way to control inference frequency by skipping frames
- **Recommended values**: 5-15 FPS for battery saving, 30 FPS for smooth detection

### Front Camera Usage
- The native implementation automatically handles coordinate transformation
- Bounding boxes should appear correctly positioned
- If issues persist, try:
  1. Using a different model trained for front camera usage
  2. Adjusting confidence thresholds
  3. Checking device-specific camera capabilities

### Performance Optimization
- Use `YOLOStreamingConfig.minimal()` for maximum performance
- Disable unnecessary features (masks, poses, OBB) if not needed
- Monitor actual FPS vs target FPS to ensure settings are working

## Troubleshooting

1. **Frame rate not changing**: Ensure you're using `inferenceFrequency` or `maxFPS` instead of just `skipFrames`
2. **Front camera boxes wrong**: The coordinate transformation is handled automatically - if issues persist, it may be a model-specific problem
3. **High battery usage**: Lower the `inferenceFrequency` to 5-10 FPS
4. **Poor detection**: Increase `inferenceFrequency` or adjust confidence thresholds

## Available Frame Rate Options

- **5 FPS**: Very slow, battery saving, good for static scenes
- **10 FPS**: Balanced, moderate battery usage, good for most use cases
- **15 FPS**: Smooth, higher battery usage, good for moving objects
- **30 FPS**: Very smooth, high battery usage, best for fast-moving objects 