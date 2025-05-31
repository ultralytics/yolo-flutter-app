# Stack Overflow Bug Fix Summary

## 問題の発見
- **日時**: 2025年5月31日（土）
- **問題**: example appを実行すると即座にStack Overflowエラーで落ちる
- **影響**: mainブランチが壊れており、全ユーザーがexampleを実行できない

## エラーの詳細
```
[ERROR:flutter/runtime/dart_vm_initializer.cc(40)] Unhandled Exception: Stack Overflow
#0      main (package:ultralytics_yolo_example/main.dart:7:1)
#1      main (package:ultralytics_yolo_example/main.dart:7:24)
...（無限に続く）
```

## 原因分析
1. **commit 7a6ede0** (feat: improve score) でバグが導入された
2. `example/lib/main.dart` と `example/main.dart` が自分自身を呼び出す無限再帰
   ```dart
   void main() => main();  // 自分自身を呼び出している！
   ```
3. pub.devスコア改善のためにshared_main.dartパターンを導入しようとしたが、実装ミス

## 解決プロセス

### 1. 最初の修正案（shared_mainを維持）
- `shared_main.dart` をインポートするよう修正
- しかしパスの問題で動作せず

### 2. 最終的な解決策（シンプル化）
- `shared_main.dart` パターンを削除
- `example/lib/main.dart` に直接アプリコードを配置（Flutter標準）
- 不要な `example/main.dart` を削除

### 修正内容
```diff
// example/lib/main.dart
- void main() => main();
+ import 'package:flutter/material.dart';
+ import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';
+ 
+ void main() {
+   runApp(const YOLOExampleApp());
+ }
+ // ... 実際のアプリコード
```

## ブランチ構成

### 1. `fix-stack-overflow-urgent` (緊急修正用)
- 最小限の変更でバグ修正のみ
- 2ファイルの変更（lib/main.dart修正、main.dart削除）
- **これをPRとして提出**

### 2. `ideal-example-structure` (将来の理想形)
- シンプルな39行の `example/example.dart` を追加
- pub.dev表示の最適化
- shared_mainパターンを完全に削除
- **将来の改善提案用に保存**

## 対応状況
1. ✅ 開発者にDMで通知 → 「明日まで不在」との返答
2. ✅ 緊急修正PRを作成（fix-stack-overflow-urgent）
3. ✅ 社長にレビュー依頼
4. ✅ 公開Slackで状況を共有

## 今後の推奨事項

### 短期
- 緊急修正PRのマージ（mainブランチの復旧）
- CIにexample実行テストを追加

### 中期
- `example/example.dart` の追加（pub.dev表示の最適化）
- より充実したテストカバレッジ

### 教訓
- pub.devスコアの改善は重要だが、動作確認は必須
- 金曜日の駆け込みPRは要注意
- レビュー時にローカルでの動作確認が重要

## 関連情報
- **影響を受けるパッケージ**: ultralytics_yolo (1000+ downloads)
- **企業**: Ultralytics（有名なYOLO開発企業）
- **緊急度**: 高（mainブランチが壊れている）

## Git設定メモ
このリポジトリでは以下の設定が必要：
```bash
git config user.name "john-rocky"
git config user.email "rockyshikoku@gmail.com"
```