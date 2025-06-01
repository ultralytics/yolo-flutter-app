#!/bin/bash

# マルチインスタンステストアプリを実行するスクリプト

echo "🚀 Starting YOLO Multi-Instance Test App..."
echo ""

# プロジェクトディレクトリに移動
cd "$(dirname "$0")/.."

# Flutterのパスを確認
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter not found in PATH"
    exit 1
fi

# デバイスを確認
echo "📱 Available devices:"
flutter devices
echo ""

# アセットファイルの確認
echo "📁 Checking for model files..."
if [ ! -d "assets/models" ]; then
    echo "❌ assets/models directory not found!"
    echo "Please create the directory and add YOLO model files:"
    echo "  - assets/models/yolov8n.tflite"
    echo "  - assets/models/yolov8n-seg.tflite"
    exit 1
fi

# クリーンビルド
echo "🧹 Cleaning previous build..."
flutter clean

# パッケージの取得
echo "📦 Getting packages..."
flutter pub get

# マルチインスタンステストアプリを実行
echo ""
echo "🚀 Running Multi-Instance Test App..."
echo "This will launch a separate test app for multi-instance functionality"
echo ""

flutter run lib/multi_instance_test_main.dart

# エラーコードを返す
exit $?