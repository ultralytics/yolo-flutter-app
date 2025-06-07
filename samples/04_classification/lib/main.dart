// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Image Classification Sample
///
/// This sample demonstrates how to use YOLO for image classification:
/// 1. Load a YOLO classification model
/// 2. Select an image from gallery
/// 3. Classify the entire image into categories
/// 4. Display results with confidence scores
///
/// 画像分類のサンプル
/// YOLOを使った画像分類の実装例：
/// 1. YOLO分類モデルの読み込み
/// 2. ギャラリーから画像を選択
/// 3. 画像全体をカテゴリに分類
/// 4. 信頼度スコア付きで結果を表示

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Classification',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.amber),
        useMaterial3: true,
      ),
      home: const ClassificationScreen(),
    );
  }
}

class ClassificationScreen extends StatefulWidget {
  const ClassificationScreen({super.key});

  @override
  State<ClassificationScreen> createState() => _ClassificationScreenState();
}

class _ClassificationScreenState extends State<ClassificationScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  Map<String, dynamic>? _classificationResult;
  bool _isLoading = false;
  String? _error;
  int _topK = 5; // Show top 5 predictions

  /// Pick an image from gallery
  /// ギャラリーから画像を選択
  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imageFile = File(image.path);
        _classificationResult = null;
        _error = null;
      });
    }
  }

  /// Run YOLO classification on the selected image
  /// 選択した画像でYOLO分類を実行
  Future<void> _runClassification() async {
    if (_imageFile == null) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // 1. Create YOLO instance
      // YOLOインスタンスを作成
      final yolo = YOLO(
        modelPath: 'yolo11n-cls.tflite', // Use yolo11n-cls.mlmodel for iOS
        task: YOLOTask.classify,
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

      // 5. Extract classification results
      // 分類結果を抽出
      setState(() {
        _classificationResult = response;
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

  /// Get top predictions from classification result
  /// 分類結果からトップ予測を取得
  List<MapEntry<String, double>> _getTopPredictions() {
    if (_classificationResult == null) return [];

    final probs = _classificationResult!['probs'] as Map<String, dynamic>;

    // Convert to list of entries and sort by confidence
    var entries = probs.entries
        .map((e) => MapEntry(e.key, (e.value as num).toDouble()))
        .toList();

    entries.sort((a, b) => b.value.compareTo(a.value));

    // Return top K predictions
    return entries.take(_topK).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Classification'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Image display area / 画像表示エリア
          Expanded(
            flex: 2,
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
          Expanded(
            flex: 2,
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              child: _buildResultsView(),
            ),
          ),

          // Controls / コントロール
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                const Text('Show top:'),
                const SizedBox(width: 8),
                SegmentedButton<int>(
                  segments: const [
                    ButtonSegment(value: 3, label: Text('3')),
                    ButtonSegment(value: 5, label: Text('5')),
                    ButtonSegment(value: 10, label: Text('10')),
                  ],
                  selected: {_topK},
                  onSelectionChanged: (Set<int> newSelection) {
                    setState(() {
                      _topK = newSelection.first;
                    });
                  },
                ),
              ],
            ),
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
                  onPressed: _imageFile == null || _isLoading
                      ? null
                      : _runClassification,
                  icon: _isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.play_arrow),
                  label: Text(_isLoading ? 'Processing...' : 'Classify Image'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build the image view
  /// 画像ビューを構築
  Widget _buildImageView() {
    if (_imageFile == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.image, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text('Select an image to classify'),
          ],
        ),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.file(_imageFile!, fit: BoxFit.contain),
    );
  }

  /// Build the results view
  /// 結果ビューを構築
  Widget _buildResultsView() {
    if (_classificationResult == null) {
      return const Center(
        child: Text('No results yet', style: TextStyle(color: Colors.grey)),
      );
    }

    final topPredictions = _getTopPredictions();

    if (topPredictions.isEmpty) {
      return const Center(child: Text('No predictions available'));
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(
            'Top $_topK Predictions',
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: topPredictions.length,
            itemBuilder: (context, index) {
              final prediction = topPredictions[index];
              final confidence = prediction.value;
              final isTopPrediction = index == 0;

              return Card(
                elevation: isTopPrediction ? 4 : 1,
                color: isTopPrediction
                    ? Theme.of(context).primaryColor.withOpacity(0.1)
                    : null,
                margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: ListTile(
                  leading: CircleAvatar(
                    backgroundColor: isTopPrediction
                        ? Theme.of(context).primaryColor
                        : Colors.grey,
                    child: Text(
                      '${index + 1}',
                      style: TextStyle(
                        color: isTopPrediction ? Colors.white : Colors.white70,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  title: Text(
                    prediction.key,
                    style: TextStyle(
                      fontWeight: isTopPrediction
                          ? FontWeight.bold
                          : FontWeight.normal,
                    ),
                  ),
                  subtitle: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const SizedBox(height: 4),
                      LinearProgressIndicator(
                        value: confidence,
                        backgroundColor: Colors.grey[300],
                        valueColor: AlwaysStoppedAnimation<Color>(
                          isTopPrediction
                              ? Theme.of(context).primaryColor
                              : Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Confidence: ${(confidence * 100).toStringAsFixed(2)}%',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ],
                  ),
                  trailing: isTopPrediction
                      ? const Icon(Icons.star, color: Colors.amber)
                      : null,
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
