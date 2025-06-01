# YOLO Multi-Instance Test App

独立したマルチインスタンステスト用アプリケーションです。

## 実行方法

### 方法1: スクリプトを使用
```bash
cd example
./lib/run_multi_instance_test.sh
```

### 方法2: 直接実行
```bash
cd example
flutter run lib/multi_instance_test_main.dart
```

### 方法3: 特定のデバイスで実行
```bash
# デバイス一覧を確認
flutter devices

# iOS実機で実行
flutter run -d <ios-device-id> lib/multi_instance_test_main.dart

# Android実機で実行
flutter run -d <android-device-id> lib/multi_instance_test_main.dart
```

## 必要なファイル

以下のモデルファイルを配置してください：
```
example/assets/models/
├── yolov8n.tflite      # 物体検出用
└── yolov8n-seg.tflite  # セグメンテーション用
```

モデルファイルは[Ultralytics](https://docs.ultralytics.com/modes/export/)からダウンロードできます。

## テスト内容

このアプリは以下をテストします：

1. **複数インスタンスの作成**
   - 物体検出用インスタンス
   - セグメンテーション用インスタンス

2. **並列モデルロード**
   - 両方のモデルを同時にロード

3. **推論実行**
   - 同じ画像で両方のモデルを実行
   - 推論時間の計測

4. **インスタンスID確認**
   - 各インスタンスの一意なIDを表示
   - アクティブなインスタンス数の確認

## 主な機能

- **カメラ撮影**: カメラから画像を撮影して推論
- **ギャラリー選択**: 保存済み画像を選択して推論
- **結果表示**: 検出結果とセグメンテーション結果を同時表示
- **パフォーマンス計測**: 各モデルの推論時間を表示
- **インスタンス情報**: フローティングボタンでインスタンス情報を確認

## トラブルシューティング

### モデルが見つからない
```bash
# モデルファイルの存在を確認
ls -la example/assets/models/
```

### ビルドエラー
```bash
# クリーンビルド
cd example
flutter clean
flutter pub get
flutter run lib/multi_instance_test_main.dart
```

### Android Gradleエラー
既にsettings.gradleでAndroid Gradle Plugin 8.3.0に更新済みです。

## デバッグ方法

1. **ログの確認**
   ```bash
   flutter logs
   ```

2. **インスタンスIDの確認**
   アプリ内の情報ボタン（右下のフローティングボタン）をタップ

3. **メモリ使用量の監視**
   - iOS: Xcode > Debug Navigator > Memory
   - Android: Android Studio > Profiler