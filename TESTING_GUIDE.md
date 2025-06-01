# YOLO Flutter 実機テストガイド

## 1. 事前準備

### 必要なもの
- iOS: iPhone/iPad実機、Apple Developer Account（無料でOK）
- Android: Android実機、開発者モードを有効化
- テスト用のYOLOモデルファイル（.tflite形式）

### モデルファイルの準備
```bash
# exampleアプリのassetsフォルダにモデルを配置
example/assets/models/
├── yolov8n.tflite      # 物体検出用
├── yolov8n-seg.tflite  # セグメンテーション用
└── yolov8n-cls.tflite  # 分類用
```

## 2. iOS実機テスト

### 2.1 Xcodeでの設定
```bash
# プロジェクトディレクトリから
cd example/ios
pod install
open Runner.xcworkspace
```

### 2.2 実機での実行
1. XcodeでRunnerプロジェクトを開く
2. Signing & Capabilitiesタブで開発チームを選択
3. 実機をMacに接続
4. スキーム選択で実機を選択
5. Runボタンをクリック

### 2.3 権限の設定（Info.plist）
```xml
<key>NSCameraUsageDescription</key>
<string>カメラを使用してリアルタイム推論を行います</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>画像を選択して推論を行います</string>
```

## 3. Android実機テスト

### 3.1 開発者モードの有効化
1. 設定 → デバイス情報
2. ビルド番号を7回タップ
3. 開発者向けオプションでUSBデバッグを有効化

### 3.2 実機での実行
```bash
# プロジェクトディレクトリから
cd example

# 接続されているデバイスを確認
flutter devices

# 実機で実行
flutter run -d <device_id>
```

### 3.3 権限の設定（AndroidManifest.xml）
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

## 4. テストコードの作成

### 4.1 基本的なテストアプリ
```dart
// example/lib/test_multi_instance.dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class TestMultiInstanceApp extends StatefulWidget {
  @override
  _TestMultiInstanceAppState createState() => _TestMultiInstanceAppState();
}

class _TestMultiInstanceAppState extends State<TestMultiInstanceApp> {
  YOLO? _detector;
  YOLO? _segmenter;
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  String _results = 'No results yet';
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initializeModels();
  }

  Future<void> _initializeModels() async {
    setState(() => _isLoading = true);
    
    try {
      // 複数インスタンスを作成
      _detector = YOLO(
        modelPath: 'assets/models/yolov8n.tflite',
        task: YOLOTask.detect,
      );
      
      _segmenter = YOLO(
        modelPath: 'assets/models/yolov8n-seg.tflite',
        task: YOLOTask.segment,
      );
      
      // モデルをロード
      await Future.wait([
        _detector!.loadModel(),
        _segmenter!.loadModel(),
      ]);
      
      setState(() {
        _results = 'Models loaded successfully!\n';
        _results += 'Detector ID: ${_detector!.instanceId}\n';
        _results += 'Segmenter ID: ${_segmenter!.instanceId}';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _results = 'Error loading models: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _pickAndProcessImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    
    setState(() {
      _selectedImage = File(image.path);
      _isLoading = true;
    });
    
    try {
      final imageBytes = await _selectedImage!.readAsBytes();
      
      // 両方のモデルで推論
      final results = await Future.wait([
        _detector!.predict(imageBytes),
        _segmenter!.predict(imageBytes),
      ]);
      
      setState(() {
        _results = 'Detection Results:\n';
        _results += '- Found ${results[0]['boxes']?.length ?? 0} objects\n\n';
        _results += 'Segmentation Results:\n';
        _results += '- Found ${results[1]['boxes']?.length ?? 0} segments\n';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _results = 'Error during inference: $e';
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    _segmenter?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('YOLO Multi-Instance Test'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_selectedImage != null)
                Container(
                  height: 200,
                  child: Image.file(_selectedImage!),
                ),
              SizedBox(height: 20),
              if (_isLoading)
                CircularProgressIndicator()
              else
                ElevatedButton(
                  onPressed: _pickAndProcessImage,
                  child: Text('Select Image'),
                ),
              SizedBox(height: 20),
              Padding(
                padding: EdgeInsets.all(16),
                child: Text(_results),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

void main() => runApp(TestMultiInstanceApp());
```

### 4.2 pubspec.yamlの更新
```yaml
# example/pubspec.yaml
dependencies:
  flutter:
    sdk: flutter
  ultralytics_yolo:
    path: ../
  image_picker: ^1.0.0

flutter:
  assets:
    - assets/models/
```

## 5. テスト実行手順

### 5.1 基本的なテスト
1. アプリを起動
2. モデルのロードが成功することを確認
3. インスタンスIDが表示されることを確認
4. 画像を選択して推論を実行
5. 両方のモデルから結果が返ることを確認

### 5.2 パフォーマンステスト
```dart
// 推論時間を測定
final stopwatch = Stopwatch()..start();
final result = await detector.predict(imageBytes);
stopwatch.stop();
print('Inference time: ${stopwatch.elapsedMilliseconds}ms');
```

### 5.3 メモリリークテスト
```dart
// 複数回インスタンスを作成・破棄
for (int i = 0; i < 10; i++) {
  final yolo = YOLO(
    modelPath: 'yolov8n.tflite',
    task: YOLOTask.detect,
  );
  await yolo.loadModel();
  final result = await yolo.predict(imageBytes);
  await yolo.dispose();
  print('Iteration $i completed');
}
```

## 6. トラブルシューティング

### iOS関連
- **コード署名エラー**: Xcodeで適切な開発チームを選択
- **モデルが見つからない**: Build Phasesでアセットが含まれているか確認
- **メモリ不足**: モデルサイズを確認、より小さいモデルを使用

### Android関連
- **権限エラー**: 実行時権限のリクエストを実装
- **モデルロードエラー**: assetsフォルダの構造を確認
- **デバイスが認識されない**: adb devicesでデバイスを確認

## 7. デバッグ方法

### ログの確認
```dart
// Flutterログ
print('Creating YOLO instance: ${yolo.instanceId}');

// ネイティブログ
// iOS: Xcodeのコンソール
// Android: flutter logs または adb logcat
```

### ブレークポイント
- Flutter: VS CodeまたはAndroid Studioでブレークポイント設定
- iOS: Xcodeでネイティブコードにブレークポイント
- Android: Android Studioでネイティブコードにブレークポイント

## 8. リリース前のチェックリスト

- [ ] すべてのモデルタイプ（detect, segment, classify, pose, obb）でテスト
- [ ] 異なるモデルサイズでテスト（n, s, m, l, x）
- [ ] メモリリークがないことを確認
- [ ] 複数インスタンスの同時実行をテスト
- [ ] インスタンスの作成・破棄を繰り返してテスト
- [ ] 実機での推論速度を確認
- [ ] エラーハンドリングが適切に動作することを確認