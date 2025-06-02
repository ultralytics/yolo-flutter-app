// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';

void main() => runApp(MultiInstanceTestApp());

class MultiInstanceTestApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Multi-Instance Test',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: MultiInstanceTestScreen(),
    );
  }
}

class MultiInstanceTestScreen extends StatefulWidget {
  @override
  _MultiInstanceTestScreenState createState() =>
      _MultiInstanceTestScreenState();
}

class _MultiInstanceTestScreenState extends State<MultiInstanceTestScreen> {
  // 複数のYOLOインスタンス
  YOLO? _detector;
  YOLO? _segmenter;

  // UI状態
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  bool _isLoading = false;
  String _statusMessage = 'Initializing...';

  // 結果
  Map<String, dynamic>? _detectionResults;
  Map<String, dynamic>? _segmentationResults;
  int _inferenceTimeDetection = 0;
  int _inferenceTimeSegmentation = 0;

  @override
  void initState() {
    super.initState();
    _initializeModels();
  }

  Future<void> _initializeModels() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Creating YOLO instances...';
    });

    try {
      // 複数のYOLOインスタンスを作成
      _detector = YOLO(modelPath: 'yolo11n', task: YOLOTask.detect);

      _segmenter = YOLO(modelPath: 'yolo11n-cls', task: YOLOTask.classify);

      setState(() {
        _statusMessage = 'Loading models...';
      });

      // モデルを並列でロード
      final loadResults = await Future.wait([
        _detector!.loadModel(),
        _segmenter!.loadModel(),
      ]);

      if (loadResults.every((result) => result)) {
        setState(() {
          _statusMessage =
              'Models loaded successfully!\n'
              'Detector ID: ${_detector!.instanceId}\n'
              'Segmenter ID: ${_segmenter!.instanceId}';
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

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image == null) return;

    setState(() {
      _selectedImage = File(image.path);
      _detectionResults = null;
      _segmentationResults = null;
    });

    await _runInference();
  }

  Future<void> _runInference() async {
    if (_selectedImage == null || _detector == null || _segmenter == null)
      return;

    setState(() {
      _isLoading = true;
      _statusMessage = 'Running inference...';
    });

    try {
      final imageBytes = await _selectedImage!.readAsBytes();

      // 物体検出の実行と時間計測
      final detectStopwatch = Stopwatch()..start();
      final detectionResult = await _detector!.predict(
        imageBytes,
        confidenceThreshold: 0.25,
        iouThreshold: 0.4,
      );
      detectStopwatch.stop();
      _inferenceTimeDetection = detectStopwatch.elapsedMilliseconds;

      // セグメンテーションの実行と時間計測
      final segmentStopwatch = Stopwatch()..start();
      final segmentationResult = await _segmenter!.predict(
        imageBytes,
        confidenceThreshold: 0.3,
        iouThreshold: 0.5,
      );
      segmentStopwatch.stop();
      _inferenceTimeSegmentation = segmentStopwatch.elapsedMilliseconds;

      setState(() {
        _detectionResults = detectionResult;
        _segmentationResults = segmentationResult;
        _statusMessage = 'Inference completed!';
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
    if (_detectionResults == null && _segmentationResults == null) {
      return Center(child: Text('No results yet. Select an image to start.'));
    }

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_detectionResults != null) ...[
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Detection Results',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    SizedBox(height: 8),
                    Text('Instance ID: ${_detector!.instanceId}'),
                    Text(
                      'Objects found: ${_detectionResults!['boxes']?.length ?? 0}',
                    ),
                    Text('Inference time: ${_inferenceTimeDetection}ms'),
                    if (_detectionResults!['boxes'] != null) ...[
                      SizedBox(height: 8),
                      Text('Detected classes:'),
                      ...(_detectionResults!['boxes'] as List)
                          .map((box) {
                            return Text(
                              '  - Class ${box['class']}: ${(box['confidence'] * 100).toStringAsFixed(1)}%',
                            );
                          })
                          .take(5),
                      if ((_detectionResults!['boxes'] as List).length > 5)
                        Text(
                          '  ... and ${(_detectionResults!['boxes'] as List).length - 5} more',
                        ),
                    ],
                  ],
                ),
              ),
            ),
            SizedBox(height: 16),
          ],

          if (_segmentationResults != null) ...[
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Segmentation Results',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    SizedBox(height: 8),
                    Text('Instance ID: ${_segmenter!.instanceId}'),
                    Text(
                      'Segments found: ${_segmentationResults!['boxes']?.length ?? 0}',
                    ),
                    Text('Inference time: ${_inferenceTimeSegmentation}ms'),
                    if (_segmentationResults!['boxes'] != null) ...[
                      SizedBox(height: 8),
                      Text('Segmented classes:'),
                      ...(_segmentationResults!['boxes'] as List)
                          .map((box) {
                            return Text(
                              '  - Class ${box['class']}: ${(box['confidence'] * 100).toStringAsFixed(1)}%',
                            );
                          })
                          .take(5),
                      if ((_segmentationResults!['boxes'] as List).length > 5)
                        Text(
                          '  ... and ${(_segmentationResults!['boxes'] as List).length - 5} more',
                        ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  @override
  void dispose() {
    _detector?.dispose();
    _segmenter?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLO Multi-Instance Test'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Status section
          Container(
            width: double.infinity,
            padding: EdgeInsets.all(16),
            color: Theme.of(context).colorScheme.surfaceVariant,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Status', style: Theme.of(context).textTheme.titleMedium),
                SizedBox(height: 4),
                Text(_statusMessage),
              ],
            ),
          ),

          // Image display
          if (_selectedImage != null)
            Container(
              height: 200,
              margin: EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(_selectedImage!, fit: BoxFit.contain),
              ),
            ),

          // Action buttons
          Padding(
            padding: EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed:
                      _isLoading || _detector == null || _segmenter == null
                      ? null
                      : () => _pickImage(ImageSource.camera),
                  icon: Icon(Icons.camera_alt),
                  label: Text('Camera'),
                ),
                ElevatedButton.icon(
                  onPressed:
                      _isLoading || _detector == null || _segmenter == null
                      ? null
                      : () => _pickImage(ImageSource.gallery),
                  icon: Icon(Icons.photo_library),
                  label: Text('Gallery'),
                ),
              ],
            ),
          ),

          // Results or loading indicator
          Expanded(
            child: Padding(
              padding: EdgeInsets.all(16),
              child: _isLoading
                  ? Center(child: CircularProgressIndicator())
                  : _buildResultsView(),
            ),
          ),
        ],
      ),
      // Floating action button for testing
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: Text('Instance Information'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Active YOLO Instances:'),
                  SizedBox(height: 8),
                  if (_detector != null)
                    Text('Detector: ${_detector!.instanceId}'),
                  if (_segmenter != null)
                    Text('Segmenter: ${_segmenter!.instanceId}'),
                  SizedBox(height: 16),
                  Text(
                    'Total instances: ${YOLOInstanceManager.getActiveInstanceIds().length}',
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text('Close'),
                ),
              ],
            ),
          );
        },
        child: Icon(Icons.info),
        tooltip: 'Show instance info',
      ),
    );
  }
}
