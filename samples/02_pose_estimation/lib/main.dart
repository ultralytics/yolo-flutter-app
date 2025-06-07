// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Human Pose Estimation Sample
///
/// This sample demonstrates how to use YOLO for human pose estimation:
/// 1. Load a YOLO pose model
/// 2. Select an image from gallery
/// 3. Detect human keypoints (17 points for COCO format)
/// 4. Display results with skeleton visualization
///
/// 人体姿勢推定のサンプル
/// YOLOを使った人体姿勢推定の実装例：
/// 1. YOLO poseモデルの読み込み
/// 2. ギャラリーから画像を選択
/// 3. 人体のキーポイント（COCO形式で17点）を検出
/// 4. スケルトン表示で結果を可視化

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Pose Estimation',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
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
  File? _imageFile;
  List<YOLOResult>? _results;
  bool _isLoading = false;
  String? _error;
  bool _showKeypoints = true;
  bool _showSkeleton = true;

  /// COCO keypoint names for display
  /// COCOキーポイント名（表示用）
  static const List<String> keypointNames = [
    'nose', // 0
    'left_eye', // 1
    'right_eye', // 2
    'left_ear', // 3
    'right_ear', // 4
    'left_shoulder', // 5
    'right_shoulder', // 6
    'left_elbow', // 7
    'right_elbow', // 8
    'left_wrist', // 9
    'right_wrist', // 10
    'left_hip', // 11
    'right_hip', // 12
    'left_knee', // 13
    'right_knee', // 14
    'left_ankle', // 15
    'right_ankle', // 16
  ];

  /// Skeleton connections (pairs of keypoint indices)
  /// スケルトン接続（キーポイントインデックスのペア）
  static const List<List<int>> skeleton = [
    // Head
    [0, 1], [0, 2], [1, 3], [2, 4],
    // Arms
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    // Body
    [5, 11], [6, 12], [11, 12],
    // Legs
    [11, 13], [13, 15], [12, 14], [14, 16],
  ];

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _results = null;
        _error = null;
      });
    }
  }

  Future<void> _runPoseEstimation() async {
    if (_imageFile == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // Create YOLO pose instance
      // YOLO poseインスタンスを作成
      final yolo = YOLO(
        modelPath: 'yolo11n-pose.tflite', // Use yolo11n-pose.mlmodel for iOS
        task: YOLOTask.pose,
      );

      // Load the model
      final success = await yolo.loadModel();
      if (!success) {
        throw Exception('Failed to load pose model');
      }

      // Read image bytes
      final imageBytes = await _imageFile!.readAsBytes();

      // Run inference
      final response = await yolo.predict(imageBytes);

      // Parse results
      final detections = response['detections'] as List<dynamic>;
      final results = detections
          .map((detection) => YOLOResult.fromMap(detection))
          .toList();

      setState(() {
        _results = results;
        _isLoading = false;
      });

      // Clean up
      await yolo.dispose();
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Human Pose Estimation'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Image display area
          Expanded(
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: _buildImageView(),
            ),
          ),

          // Controls
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                const Text('Show:'),
                const SizedBox(width: 8),
                FilterChip(
                  label: const Text('Keypoints'),
                  selected: _showKeypoints,
                  onSelected: (value) => setState(() => _showKeypoints = value),
                ),
                const SizedBox(width: 8),
                FilterChip(
                  label: const Text('Skeleton'),
                  selected: _showSkeleton,
                  onSelected: (value) => setState(() => _showSkeleton = value),
                ),
              ],
            ),
          ),

          // Results display
          if (_results != null)
            Container(
              height: 120,
              padding: const EdgeInsets.all(16),
              child: _buildResultsList(),
            ),

          // Error display
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.red),
                textAlign: TextAlign.center,
              ),
            ),

          // Action buttons
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _pickImage,
                  icon: const Icon(Icons.image),
                  label: const Text('Select Image'),
                ),
                ElevatedButton.icon(
                  onPressed: _imageFile == null || _isLoading
                      ? null
                      : _runPoseEstimation,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.accessibility_new),
                  label: Text(_isLoading ? 'Processing...' : 'Detect Poses'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageView() {
    if (_imageFile == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.image, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text('Select an image to detect human poses'),
          ],
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        return FutureBuilder<ImageInfo>(
          future: _getImageInfo(),
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              return Center(
                child: Image.file(_imageFile!, fit: BoxFit.contain),
              );
            }

            final imageInfo = snapshot.data!;
            final imageSize = Size(
              imageInfo.image.width.toDouble(),
              imageInfo.image.height.toDouble(),
            );

            // Calculate the actual display size of the image
            final displaySize = _calculateFittedImageSize(
              imageSize,
              Size(constraints.maxWidth, constraints.maxHeight),
            );

            return Stack(
              alignment: Alignment.center,
              children: [
                // Original image
                Image.file(_imageFile!, fit: BoxFit.contain),

                // Pose overlay with proper size
                if (_results != null)
                  CustomPaint(
                    size: displaySize,
                    painter: PosePainter(
                      results: _results!,
                      showKeypoints: _showKeypoints,
                      showSkeleton: _showSkeleton,
                    ),
                  ),
              ],
            );
          },
        );
      },
    );
  }

  Future<ImageInfo> _getImageInfo() async {
    final image = FileImage(_imageFile!);
    final completer = Completer<ImageInfo>();
    final stream = image.resolve(const ImageConfiguration());

    stream.addListener(
      ImageStreamListener((info, _) {
        completer.complete(info);
      }),
    );

    return completer.future;
  }

  Size _calculateFittedImageSize(Size imageSize, Size containerSize) {
    final aspectRatio = imageSize.width / imageSize.height;
    final containerAspectRatio = containerSize.width / containerSize.height;

    if (aspectRatio > containerAspectRatio) {
      // Image is wider than container
      final width = containerSize.width;
      final height = width / aspectRatio;
      return Size(width, height);
    } else {
      // Image is taller than container
      final height = containerSize.height;
      final width = height * aspectRatio;
      return Size(width, height);
    }
  }

  Widget _buildResultsList() {
    return ListView.builder(
      itemCount: _results!.length,
      itemBuilder: (context, index) {
        final result = _results![index];
        final keypointCount = result.keypoints?.length ?? 0;
        return ListTile(
          leading: CircleAvatar(
            backgroundColor: Colors.green,
            child: Text('${index + 1}'),
          ),
          title: Text('Person ${index + 1}'),
          subtitle: Text('$keypointCount keypoints detected'),
          trailing: Text('${(result.confidence * 100).toStringAsFixed(1)}%'),
        );
      },
    );
  }
}

/// Custom painter for drawing pose keypoints and skeleton
/// 姿勢のキーポイントとスケルトンを描画するカスタムペインター
class PosePainter extends CustomPainter {
  final List<YOLOResult> results;
  final bool showKeypoints;
  final bool showSkeleton;

  PosePainter({
    required this.results,
    required this.showKeypoints,
    required this.showSkeleton,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Colors for different people
    final colors = [
      Colors.red,
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.purple,
      Colors.cyan,
      Colors.pink,
      Colors.teal,
    ];

    for (int personIdx = 0; personIdx < results.length; personIdx++) {
      final result = results[personIdx];
      final keypoints = result.keypoints;
      final confidences = result.keypointConfidences;

      if (keypoints == null || confidences == null) continue;

      final color = colors[personIdx % colors.length];

      // Draw skeleton connections first (behind keypoints)
      if (showSkeleton) {
        final skeletonPaint = Paint()
          ..color = color.withOpacity(0.8)
          ..strokeWidth = 3
          ..style = PaintingStyle.stroke;

        for (final connection in _PoseEstimationScreenState.skeleton) {
          final startIdx = connection[0];
          final endIdx = connection[1];

          // Check if both keypoints are visible (confidence > 0.5)
          if (startIdx < keypoints.length &&
              endIdx < keypoints.length &&
              confidences[startIdx] > 0.5 &&
              confidences[endIdx] > 0.5) {
            final startPoint = keypoints[startIdx];
            final endPoint = keypoints[endIdx];

            canvas.drawLine(
              Offset(startPoint.x * size.width, startPoint.y * size.height),
              Offset(endPoint.x * size.width, endPoint.y * size.height),
              skeletonPaint,
            );
          }
        }
      }

      // Draw keypoints
      if (showKeypoints) {
        final keypointPaint = Paint()..style = PaintingStyle.fill;

        for (int i = 0; i < keypoints.length && i < confidences.length; i++) {
          if (confidences[i] > 0.5) {
            // Only draw visible keypoints
            final keypoint = keypoints[i];
            final x = keypoint.x * size.width;
            final y = keypoint.y * size.height;

            // Outer circle (color)
            keypointPaint.color = color;
            canvas.drawCircle(Offset(x, y), 6, keypointPaint);

            // Inner circle (white)
            keypointPaint.color = Colors.white;
            canvas.drawCircle(Offset(x, y), 3, keypointPaint);
          }
        }
      }

      // Draw bounding box
      final boxPaint = Paint()
        ..color = color.withOpacity(0.5)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      final box = result.normalizedBox;
      canvas.drawRect(
        Rect.fromLTRB(
          box.left * size.width,
          box.top * size.height,
          box.right * size.width,
          box.bottom * size.height,
        ),
        boxPaint,
      );

      // Draw label
      final textPainter = TextPainter(
        text: TextSpan(
          text: 'Person ${personIdx + 1}',
          style: TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            backgroundColor: color,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
          box.left * size.width,
          box.top * size.height - textPainter.height - 2,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) => true;
}
