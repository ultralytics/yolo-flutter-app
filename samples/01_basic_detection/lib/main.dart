import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Basic Object Detection Sample
/// 
/// This sample demonstrates the simplest way to use YOLO for object detection:
/// 1. Load a YOLO model
/// 2. Select an image from gallery
/// 3. Run inference
/// 4. Display results with bounding boxes
/// 
/// 基本的な物体検出のサンプル
/// YOLOを使った最もシンプルな物体検出の実装例：
/// 1. YOLOモデルの読み込み
/// 2. ギャラリーから画像を選択
/// 3. 推論の実行
/// 4. バウンディングボックス付きで結果を表示

void main() => runApp(const MyApp());

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
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  List<YOLOResult>? _results;
  bool _isLoading = false;
  String? _error;

  /// Pick an image from gallery
  /// ギャラリーから画像を選択
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

  /// Run YOLO detection on the selected image
  /// 選択した画像でYOLO検出を実行
  Future<void> _runDetection() async {
    if (_imageFile == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // 1. Create YOLO instance
      // YOLOインスタンスを作成
      final yolo = YOLO(
        modelPath: 'yolo11n.tflite', // Use yolo11n.mlmodel for iOS
        task: YOLOTask.detect,
      );

      // 2. Load the model
      // モデルを読み込む
      final success = await yolo.loadModel();
      if (!success) {
        throw Exception('Failed to load model');
      }

      // 3. Read image bytes
      // 画像データを読み込む
      final imageBytes = await _imageFile!.readAsBytes();

      // 4. Run inference
      // 推論を実行
      final response = await yolo.predict(imageBytes);

      // 5. Parse results using YOLOResult
      // YOLOResultを使って結果を解析
      final detections = response['detections'] as List<dynamic>;
      final results = detections
          .map((detection) => YOLOResult.fromMap(detection))
          .toList();

      setState(() {
        _results = results;
        _isLoading = false;
      });

      // 6. Clean up
      // クリーンアップ
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
        title: const Text('Basic Object Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Image display area / 画像表示エリア
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

          // Results display / 結果表示
          if (_results != null)
            Container(
              height: 120,
              padding: const EdgeInsets.all(16),
              child: _buildResultsList(),
            ),

          // Error display / エラー表示
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.red),
                textAlign: TextAlign.center,
              ),
            ),

          // Action buttons / アクションボタン
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
                  onPressed: _imageFile == null || _isLoading ? null : _runDetection,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.play_arrow),
                  label: Text(_isLoading ? 'Processing...' : 'Detect Objects'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build the image view with bounding boxes
  /// バウンディングボックス付きの画像ビューを構築
  Widget _buildImageView() {
    if (_imageFile == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.image, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text('Select an image to detect objects'),
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // Original image / 元画像
        Image.file(
          _imageFile!,
          fit: BoxFit.contain,
        ),

        // Bounding boxes overlay / バウンディングボックスのオーバーレイ
        if (_results != null)
          CustomPaint(
            painter: BoundingBoxPainter(
              results: _results!,
              imageFile: _imageFile!,
            ),
          ),
      ],
    );
  }

  /// Build the results list
  /// 結果リストを構築
  Widget _buildResultsList() {
    return ListView.builder(
      itemCount: _results!.length,
      itemBuilder: (context, index) {
        final result = _results![index];
        return ListTile(
          leading: CircleAvatar(
            child: Text('${index + 1}'),
          ),
          title: Text(result.className),
          subtitle: Text('Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%'),
        );
      },
    );
  }
}

/// Custom painter for drawing bounding boxes
/// バウンディングボックスを描画するカスタムペインター
class BoundingBoxPainter extends CustomPainter {
  final List<YOLOResult> results;
  final File imageFile;

  BoundingBoxPainter({
    required this.results,
    required this.imageFile,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    // Colors for different objects / オブジェクトごとの色
    final colors = [
      Colors.red,
      Colors.green,
      Colors.blue,
      Colors.orange,
      Colors.purple,
      Colors.cyan,
      Colors.pink,
      Colors.teal,
    ];

    for (int i = 0; i < results.length; i++) {
      final result = results[i];
      final color = colors[i % colors.length];
      paint.color = color;

      // Convert normalized coordinates to canvas coordinates
      // 正規化座標をキャンバス座標に変換
      final left = result.normalizedBox.left * size.width;
      final top = result.normalizedBox.top * size.height;
      final right = result.normalizedBox.right * size.width;
      final bottom = result.normalizedBox.bottom * size.height;

      final rect = Rect.fromLTRB(left, top, right, bottom);

      // Draw bounding box / バウンディングボックスを描画
      canvas.drawRect(rect, paint);

      // Draw label background / ラベル背景を描画
      final label = '${result.className} ${(result.confidence * 100).toStringAsFixed(0)}%';
      textPainter.text = TextSpan(
        text: label,
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );
      textPainter.layout();

      final labelBgRect = Rect.fromLTWH(
        left,
        top - textPainter.height - 4,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      paint.style = PaintingStyle.fill;
      canvas.drawRect(labelBgRect, paint);

      // Draw label text / ラベルテキストを描画
      textPainter.paint(
        canvas,
        Offset(left + 4, top - textPainter.height - 2),
      );

      paint.style = PaintingStyle.stroke;
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) => true;
}