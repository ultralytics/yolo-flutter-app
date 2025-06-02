// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'dart:io';

/// Example demonstrating multiple YOLO instances with different models
class MultiInstanceExample extends StatefulWidget {
  const MultiInstanceExample({Key? key}) : super(key: key);

  @override
  State<MultiInstanceExample> createState() => _MultiInstanceExampleState();
}

class _MultiInstanceExampleState extends State<MultiInstanceExample> {
  String? _detectInstanceId;
  String? _segmentInstanceId;
  bool _isLoading = false;
  String _statusMessage = 'Not initialized';

  Map<String, dynamic>? _detectResults;
  Map<String, dynamic>? _segmentResults;

  @override
  void initState() {
    super.initState();
    _initializeInstances();
  }

  @override
  void dispose() {
    _disposeInstances();
    super.dispose();
  }

  Future<void> _initializeInstances() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Creating instances...';
    });

    try {
      // Create two instances
      _detectInstanceId = await YOLO.createInstance();
      _segmentInstanceId = await YOLO.createInstance();

      setState(() {
        _statusMessage = 'Loading models...';
      });

      // Load detection model on first instance
      final detectSuccess = await YOLO.loadModelWithInstance(
        instanceId: _detectInstanceId!,
        model: 'assets/models/yolov8n.tflite',
        task: YOLOTask.detect,
      );

      // Load segmentation model on second instance
      final segmentSuccess = await YOLO.loadModelWithInstance(
        instanceId: _segmentInstanceId!,
        model: 'assets/models/yolov8n-seg.tflite',
        task: YOLOTask.segment,
      );

      if (detectSuccess && segmentSuccess) {
        setState(() {
          _statusMessage = 'Both models loaded successfully!';
          _isLoading = false;
        });
      } else {
        throw Exception('Failed to load one or more models');
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Error: ${e.toString()}';
        _isLoading = false;
      });
    }
  }

  Future<void> _disposeInstances() async {
    if (_detectInstanceId != null) {
      await YOLO.disposeInstance(_detectInstanceId!);
    }
    if (_segmentInstanceId != null) {
      await YOLO.disposeInstance(_segmentInstanceId!);
    }
  }

  Future<void> _runInference(String imagePath) async {
    if (_detectInstanceId == null || _segmentInstanceId == null) {
      setState(() {
        _statusMessage = 'Instances not initialized';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _statusMessage = 'Running inference...';
    });

    try {
      // Load image
      final imageFile = File(imagePath);
      final imageBytes = await imageFile.readAsBytes();

      // Run detection on first instance
      final detectResult = await YOLO.detectImageWithInstance(
        instanceId: _detectInstanceId!,
        imageBytes: imageBytes,
        confidenceThreshold: 0.25,
        iouThreshold: 0.4,
      );

      // Run segmentation on second instance
      final segmentResult = await YOLO.detectImageWithInstance(
        instanceId: _segmentInstanceId!,
        imageBytes: imageBytes,
        confidenceThreshold: 0.3,
        iouThreshold: 0.5,
      );

      setState(() {
        _detectResults = detectResult;
        _segmentResults = segmentResult;
        _statusMessage = 'Inference completed successfully!';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Inference error: ${e.toString()}';
        _isLoading = false;
      });
    }
  }

  Widget _buildResultsView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (_detectResults != null) ...[
          Text(
            'Detection Results:',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          Text('Found ${_detectResults!['boxes']?.length ?? 0} objects'),
          const SizedBox(height: 16),
        ],
        if (_segmentResults != null) ...[
          Text(
            'Segmentation Results:',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          Text('Found ${_segmentResults!['boxes']?.length ?? 0} segments'),
          const SizedBox(height: 16),
        ],
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Multi-Instance YOLO Example')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Instance IDs:',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            Text('Detection: ${_detectInstanceId ?? "Not created"}'),
            Text('Segmentation: ${_segmentInstanceId ?? "Not created"}'),
            const SizedBox(height: 16),

            Text(
              'Status: $_statusMessage',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 16),

            if (_isLoading)
              const Center(child: CircularProgressIndicator())
            else ...[
              ElevatedButton(
                onPressed:
                    _detectInstanceId != null && _segmentInstanceId != null
                    ? () => _runInference('path/to/your/image.jpg')
                    : null,
                child: const Text('Run Inference'),
              ),
              const SizedBox(height: 16),
              _buildResultsView(),
            ],
          ],
        ),
      ),
    );
  }
}

/// Example using the backward compatible API
void backwardCompatibleExample() async {
  // Old API still works (uses default instance internally)
  final yolo = YOLO(
    modelPath: 'assets/models/yolov8n.tflite',
    task: YOLOTask.detect,
  );

  await yolo.loadModel();

  final imageBytes = Uint8List.fromList([/* your image data */]);
  final results = await yolo.predict(imageBytes);

  print('Detection results: ${results['boxes']?.length ?? 0} objects found');
}

/// Example showing concurrent inference with different models
Future<void> concurrentInferenceExample() async {
  // Create instances for different tasks
  final detectId = await YOLO.createInstance();
  final segmentId = await YOLO.createInstance();
  final poseId = await YOLO.createInstance();

  // Load different models
  await Future.wait([
    YOLO.loadModelWithInstance(
      instanceId: detectId,
      model: 'yolov8n.tflite',
      task: YOLOTask.detect,
    ),
    YOLO.loadModelWithInstance(
      instanceId: segmentId,
      model: 'yolov8n-seg.tflite',
      task: YOLOTask.segment,
    ),
    YOLO.loadModelWithInstance(
      instanceId: poseId,
      model: 'yolov8n-pose.tflite',
      task: YOLOTask.pose,
    ),
  ]);

  // Run inference concurrently on the same image
  final imageBytes = Uint8List.fromList([/* your image data */]);

  final results = await Future.wait([
    YOLO.detectImageWithInstance(instanceId: detectId, imageBytes: imageBytes),
    YOLO.detectImageWithInstance(instanceId: segmentId, imageBytes: imageBytes),
    YOLO.detectImageWithInstance(instanceId: poseId, imageBytes: imageBytes),
  ]);

  print('Detection: ${results[0]['boxes']?.length ?? 0} objects');
  print('Segmentation: ${results[1]['boxes']?.length ?? 0} segments');
  print('Pose: ${results[2]['keypoints']?.length ?? 0} poses');

  // Clean up
  await Future.wait([
    YOLO.disposeInstance(detectId),
    YOLO.disposeInstance(segmentId),
    YOLO.disposeInstance(poseId),
  ]);
}
