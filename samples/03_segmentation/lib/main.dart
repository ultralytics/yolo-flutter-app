import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Instance Segmentation Sample
/// 
/// This sample demonstrates how to use YOLO for instance segmentation:
/// 1. Load a YOLO segmentation model
/// 2. Select an image from gallery
/// 3. Detect objects and their pixel-perfect masks
/// 4. Display results with colored masks overlay
/// 
/// インスタンスセグメンテーションのサンプル
/// YOLOを使ったインスタンスセグメンテーションの実装例：
/// 1. YOLOセグメンテーションモデルの読み込み
/// 2. ギャラリーから画像を選択
/// 3. オブジェクトとそのピクセル単位のマスクを検出
/// 4. カラーマスクのオーバーレイで結果を表示

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Segmentation',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.purple),
        useMaterial3: true,
      ),
      home: const SegmentationScreen(),
    );
  }
}

class SegmentationScreen extends StatefulWidget {
  const SegmentationScreen({super.key});

  @override
  State<SegmentationScreen> createState() => _SegmentationScreenState();
}

class _SegmentationScreenState extends State<SegmentationScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  List<YOLOResult>? _results;
  bool _isLoading = false;
  String? _error;
  double _maskOpacity = 0.5;
  bool _showBoundingBoxes = true;
  bool _showMasks = true;

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

  /// Run YOLO segmentation on the selected image
  /// 選択した画像でYOLOセグメンテーションを実行
  Future<void> _runSegmentation() async {
    if (_imageFile == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // 1. Create YOLO instance
      // YOLOインスタンスを作成
      final yolo = YOLO(
        modelPath: 'yolo11n-seg.tflite', // Use yolo11n-seg.mlmodel for iOS
        task: YOLOTask.segment,
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
        title: const Text('Instance Segmentation'),
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

          // Controls / コントロール
          if (_results != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Column(
                children: [
                  // Mask opacity slider / マスク透明度スライダー
                  Row(
                    children: [
                      const Text('Mask Opacity:'),
                      Expanded(
                        child: Slider(
                          value: _maskOpacity,
                          min: 0.0,
                          max: 1.0,
                          onChanged: (value) {
                            setState(() {
                              _maskOpacity = value;
                            });
                          },
                        ),
                      ),
                      Text('${(_maskOpacity * 100).toStringAsFixed(0)}%'),
                    ],
                  ),
                  
                  // Toggle switches / トグルスイッチ
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      FilterChip(
                        label: const Text('Masks'),
                        selected: _showMasks,
                        onSelected: (value) {
                          setState(() {
                            _showMasks = value;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Bounding Boxes'),
                        selected: _showBoundingBoxes,
                        onSelected: (value) {
                          setState(() {
                            _showBoundingBoxes = value;
                          });
                        },
                      ),
                    ],
                  ),
                ],
              ),
            ),

          // Results summary / 結果サマリー
          if (_results != null)
            Container(
              height: 80,
              padding: const EdgeInsets.all(16),
              child: _buildResultsSummary(),
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
                  onPressed: _imageFile == null || _isLoading ? null : _runSegmentation,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.play_arrow),
                  label: Text(_isLoading ? 'Processing...' : 'Run Segmentation'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build the image view with segmentation masks and bounding boxes
  /// セグメンテーションマスクとバウンディングボックス付きの画像ビューを構築
  Widget _buildImageView() {
    if (_imageFile == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.image, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text('Select an image for segmentation'),
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

        // Segmentation masks and bounding boxes overlay
        // セグメンテーションマスクとバウンディングボックスのオーバーレイ
        if (_results != null)
          FutureBuilder<ui.Image>(
            future: _getImageInfo(),
            builder: (context, snapshot) {
              if (!snapshot.hasData) return const SizedBox();
              return CustomPaint(
                painter: SegmentationPainter(
                  results: _results!,
                  imageFile: _imageFile!,
                  imageSize: Size(
                    snapshot.data!.width.toDouble(),
                    snapshot.data!.height.toDouble(),
                  ),
                  maskOpacity: _maskOpacity,
                  showMasks: _showMasks,
                  showBoundingBoxes: _showBoundingBoxes,
                ),
              );
            },
          ),
      ],
    );
  }

  /// Build the results summary
  /// 結果サマリーを構築
  Widget _buildResultsSummary() {
    final uniqueClasses = _results!.map((r) => r.className).toSet();
    final classCounts = <String, int>{};
    
    for (final result in _results!) {
      classCounts[result.className] = (classCounts[result.className] ?? 0) + 1;
    }

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Theme.of(context).primaryColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              'Total: ${_results!.length} objects',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(width: 8),
          ...classCounts.entries.map((entry) => Container(
            margin: const EdgeInsets.only(right: 8),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: _getColorForIndex(
                uniqueClasses.toList().indexOf(entry.key)
              ).withOpacity(0.2),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text('${entry.key}: ${entry.value}'),
          )),
        ],
      ),
    );
  }

  /// Get color for object index
  /// オブジェクトインデックスの色を取得
  Color _getColorForIndex(int index) {
    final colors = [
      Colors.red,
      Colors.green,
      Colors.blue,
      Colors.orange,
      Colors.purple,
      Colors.cyan,
      Colors.pink,
      Colors.teal,
      Colors.amber,
      Colors.lime,
    ];
    return colors[index % colors.length];
  }

  /// Get image information for proper scaling
  /// 適切なスケーリングのための画像情報を取得
  Future<ui.Image> _getImageInfo() async {
    final bytes = await _imageFile!.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }
}

/// Custom painter for drawing segmentation masks and bounding boxes
/// セグメンテーションマスクとバウンディングボックスを描画するカスタムペインター
class SegmentationPainter extends CustomPainter {
  final List<YOLOResult> results;
  final File imageFile;
  final Size imageSize;
  final double maskOpacity;
  final bool showMasks;
  final bool showBoundingBoxes;

  SegmentationPainter({
    required this.results,
    required this.imageFile,
    required this.imageSize,
    required this.maskOpacity,
    required this.showMasks,
    required this.showBoundingBoxes,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.fill;

    final boxPaint = Paint()
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
      Colors.amber,
      Colors.lime,
    ];

    // Group results by class for consistent colors
    // クラスごとに結果をグループ化して一貫した色を使用
    final uniqueClasses = results.map((r) => r.className).toSet().toList();

    for (int i = 0; i < results.length; i++) {
      final result = results[i];
      final classIndex = uniqueClasses.indexOf(result.className);
      final color = colors[classIndex % colors.length];

      // Draw segmentation mask / セグメンテーションマスクを描画
      if (showMasks && result.mask != null) {
        paint.color = color.withOpacity(maskOpacity);
        
        // The mask is a 2D array representing a binary mask
        // マスクは2次元配列のバイナリマスク
        final maskData = result.mask!;
        
        if (maskData.isNotEmpty && maskData[0].isNotEmpty) {
          final maskHeight = maskData.length;
          final maskWidth = maskData[0].length;
          
          // Calculate the actual displayed image area considering BoxFit.contain
          // BoxFit.containを考慮して実際に表示される画像領域を計算
          final double imageAspectRatio = imageSize.width / imageSize.height;
          final double canvasAspectRatio = size.width / size.height;
          
          double displayWidth, displayHeight;
          double offsetX = 0, offsetY = 0;
          
          if (imageAspectRatio > canvasAspectRatio) {
            // Image is wider than canvas
            displayWidth = size.width;
            displayHeight = size.width / imageAspectRatio;
            offsetY = (size.height - displayHeight) / 2;
          } else {
            // Image is taller than canvas
            displayHeight = size.height;
            displayWidth = size.height * imageAspectRatio;
            offsetX = (size.width - displayWidth) / 2;
          }
          
          // Calculate scale factors from mask size to displayed image size
          // マスクサイズから表示画像サイズへのスケールファクターを計算
          final scaleX = displayWidth / maskWidth;
          final scaleY = displayHeight / maskHeight;
          
          // Create a path for the mask
          // マスクのパスを作成
          final path = Path();
          
          // Draw mask pixels
          // マスクのピクセルを描画
          for (int y = 0; y < maskHeight; y++) {
            for (int x = 0; x < maskWidth; x++) {
              if (maskData[y][x] > 0.5) {
                // This pixel is part of the mask
                // このピクセルはマスクの一部
                final rect = Rect.fromLTWH(
                  offsetX + x * scaleX,
                  offsetY + y * scaleY,
                  scaleX + 0.5, // Add small overlap to avoid gaps
                  scaleY + 0.5, // Add small overlap to avoid gaps
                );
                path.addRect(rect);
              }
            }
          }
          
          canvas.drawPath(path, paint);
        }
      }

      // Draw bounding box / バウンディングボックスを描画
      if (showBoundingBoxes) {
        boxPaint.color = color;

        // Calculate the actual displayed image area considering BoxFit.contain
        // BoxFit.containを考慮して実際に表示される画像領域を計算
        final double imageAspectRatio = imageSize.width / imageSize.height;
        final double canvasAspectRatio = size.width / size.height;
        
        double displayWidth, displayHeight;
        double offsetX = 0, offsetY = 0;
        
        if (imageAspectRatio > canvasAspectRatio) {
          // Image is wider than canvas
          displayWidth = size.width;
          displayHeight = size.width / imageAspectRatio;
          offsetY = (size.height - displayHeight) / 2;
        } else {
          // Image is taller than canvas
          displayHeight = size.height;
          displayWidth = size.height * imageAspectRatio;
          offsetX = (size.width - displayWidth) / 2;
        }

        // Convert normalized coordinates to canvas coordinates
        // 正規化座標をキャンバス座標に変換
        final left = offsetX + result.normalizedBox.left * displayWidth;
        final top = offsetY + result.normalizedBox.top * displayHeight;
        final right = offsetX + result.normalizedBox.right * displayWidth;
        final bottom = offsetY + result.normalizedBox.bottom * displayHeight;

        final rect = Rect.fromLTRB(left, top, right, bottom);
        canvas.drawRect(rect, boxPaint);

        // Draw label / ラベルを描画
        final label = '${result.className} ${(result.confidence * 100).toStringAsFixed(0)}%';
        textPainter.text = TextSpan(
          text: label,
          style: const TextStyle(
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

        paint.color = color;
        paint.style = PaintingStyle.fill;
        canvas.drawRect(labelBgRect, paint);

        textPainter.paint(
          canvas,
          Offset(left + 4, top - textPainter.height - 2),
        );
      }
    }
  }

  @override
  bool shouldRepaint(SegmentationPainter oldDelegate) => true;
}