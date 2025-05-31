# YOLO Streaming Enhancement Plan

## 概要

YOLOViewのリアルタイム結果ストリームに、mask、pose、OBB、annotatedImage、originalImageを含める機能拡張の実装方針。
fps/msは常時送信し、後方互換性を保ちながら段階的に実装する。

## 現在の問題

- YOLOViewのリアルタイムストリームにmask、pose、OBBが含まれていない
- 全フレームでこれらを取得すると処理が重くなる
- detect/classifyもオプション化を検討
- annotatedImageやoriginalImageの需要もある
- ストリーミング頻度の制御が必要

## 提案ソリューション

### 1. YOLOStreamingConfig設計

```dart
class YOLOStreamingConfig {
  // 常時含まれるデータ（fps/msを含む）
  final bool includeDetections;           // デフォルト: true
  final bool includeClassifications;      // デフォルト: true  
  final bool includeProcessingTimeMs;     // デフォルト: true（常時送信）
  final bool includeFps;                  // デフォルト: true（常時送信）
  
  // オプショナルデータ（重い処理）
  final bool includeMasks;                // デフォルト: タスクに応じて
  final bool includePoses;                // デフォルト: タスクに応じて
  final bool includeOBB;                  // デフォルト: タスクに応じて
  final bool includeAnnotatedImage;       // デフォルト: true（iOS既存動作）
  final bool includeOriginalImage;        // デフォルト: false（新機能）
  
  // ストリーミング制御
  final int? maxFPS;                      // フレーム制限
  final Duration? throttleInterval;       // 最小送信間隔
  
  // プリセット用コンストラクタ
  const YOLOStreamingConfig.standard();   // バランス型（現在の動作）
  const YOLOStreamingConfig.lightweight();// 軽量（基本検出のみ）
  const YOLOStreamingConfig.detailed();   // 全データ含む
  const YOLOStreamingConfig.custom({...});// カスタム設定
}
```

### 2. 後方互換性戦略

#### 既存コードの動作保証
```dart
// 現在のコード（変更不要）
YOLOView(
  onResult: (results) => print('${results.length} objects'),
  onPerformanceMetrics: (metrics) => print('FPS: ${metrics['fps']}'),
)
// → 内部的にYOLOStreamingConfig.standard()が適用される
```

#### 新機能の段階的利用
```dart
// レベル1: プリセット使用
YOLOView(
  streamingConfig: YOLOStreamingConfig.lightweight(),
  onResult: (results) => processResults(results),
)

// レベル2: カスタム設定
YOLOView(
  streamingConfig: YOLOStreamingConfig.custom(
    includeOriginalImage: true,
    maxFPS: 20,
    includeMasks: false,
  ),
  onResult: (results) => processResults(results),
)
```

## 実装計画

### Phase 1: コア実装 (1-2週間)

#### 実装項目
1. **YOLOStreamingConfigクラス作成**
   - ファイル: `lib/yolo_streaming_config.dart`
   - プリセット設定の実装
   - バリデーション機能

2. **YOLOView API拡張**
   - `streamingConfig`パラメータ追加
   - 既存パラメータとの統合
   - デフォルト動作の保持

3. **基本フィルタリング実装**
   - Platform側でのデータ選択制御
   - 設定に基づく結果生成

#### 変更ファイル
- `lib/yolo_view.dart`: streamingConfig追加
- `lib/yolo_streaming_config.dart`: 新規作成
- `ios/Classes/SwiftYOLOPlatformView.swift`: 設定反映
- `android/.../YOLOPlatformView.kt`: 設定反映

### Phase 2: パフォーマンス最適化 (1週間)

#### 実装項目
1. **フレーム制限機能**
   - maxFPS設定の実装
   - Platform側でのフレームスキップ

2. **Throttle機能**
   - 最小送信間隔制御
   - バッファリング機能

3. **適応品質機能**
   - FPS低下検知
   - 自動的な重いデータの除外

### Phase 3: 高度な機能 (1週間)

#### 実装項目
1. **OriginalImage機能**
   - カメラフレームの保存
   - メモリ効率的な実装

2. **オンデマンド取得**
   - トリガーベースの詳細データ取得
   - 期間限定の高詳細モード

3. **プリセット最適化**
   - デバイス性能に基づく自動調整
   - 用途別最適化設定

## Platform実装詳細

### iOS実装変更点

#### SwiftYOLOPlatformView.swift
```swift
// 設定に基づくデータフィルタリング
private func filterResultData(_ result: YOLOResult, config: StreamingConfig) -> [String: Any] {
    var data: [String: Any] = [:]
    
    // 常時含む
    if config.includeFps { data["fps"] = currentFPS }
    if config.includeProcessingTimeMs { data["processingTimeMs"] = processingTime }
    
    // 条件付き
    if config.includeDetections { data["detections"] = result.detections }
    if config.includeMasks && result.masks != nil { data["masks"] = result.masks }
    if config.includeOriginalImage { data["originalImage"] = captureOriginalFrame() }
    
    return data
}
```

### Android実装変更点

#### YOLOPlatformView.kt
```kotlin
// 設定に基づく結果生成
private fun filterResultData(result: YOLOResult, config: StreamingConfig): Map<String, Any> {
    val data = mutableMapOf<String, Any>()
    
    // 常時含む
    if (config.includeFps) data["fps"] = currentFPS
    if (config.includeProcessingTimeMs) data["processingTimeMs"] = processingTime
    
    // 条件付き
    if (config.includeDetections) data["detections"] = result.detections
    if (config.includeMasks && result.masks != null) data["masks"] = result.masks
    if (config.includeOriginalImage) data["originalImage"] = captureOriginalFrame()
    
    return data
}
```

## テスト戦略

### 単体テスト
```dart
// YOLOStreamingConfigのテスト
testWidgets('YOLOStreamingConfig default values', (tester) async {
  final config = YOLOStreamingConfig.standard();
  expect(config.includeFps, true);
  expect(config.includeProcessingTimeMs, true);
  expect(config.includeOriginalImage, false);
});

// フィルタリング機能のテスト
testWidgets('YOLOView with lightweight config', (tester) async {
  final config = YOLOStreamingConfig.lightweight();
  final results = <YOLOResult>[];
  
  await tester.pumpWidget(YOLOView(
    streamingConfig: config,
    onResult: (r) => results.addAll(r),
  ));
  
  // 軽量設定での動作確認
});
```

### 統合テスト
- 既存example appでの動作確認
- パフォーマンス測定（before/after比較）
- 各プリセット設定での動作検証
- メモリ使用量の監視

### パフォーマンステスト
```dart
// パフォーマンス測定用
void measurePerformance() {
  final stopwatch = Stopwatch()..start();
  
  // 各設定でのFPS測定
  final standardFPS = measureFPS(YOLOStreamingConfig.standard());
  final lightweightFPS = measureFPS(YOLOStreamingConfig.lightweight());
  final detailedFPS = measureFPS(YOLOStreamingConfig.detailed());
  
  print('Standard: ${standardFPS}fps');
  print('Lightweight: ${lightweightFPS}fps');
  print('Detailed: ${detailedFPS}fps');
}
```

## 実装優先度

### 高優先度 ⭐⭐⭐
1. **後方互換性確保**: 既存コードが変更なしで動作
2. **基本フラグ制御**: include/excludeの基本機能
3. **fps/ms常時送信**: パフォーマンス監視の継続

### 中優先度 ⭐⭐
1. **originalImage機能**: デバッグ・分析用途
2. **maxFPS制限**: パフォーマンス調整
3. **プリセット設定**: 初心者向け簡単設定

### 低優先度 ⭐
1. **オンデマンド取得**: 高度なユースケース
2. **適応品質**: 自動最適化機能
3. **詳細統計**: 高度な分析機能

## リスク管理

### 技術的リスク
- **メモリ使用量増加**: originalImage機能による影響
- **Platform間の非互換性**: iOS/Androidでの動作差異
- **パフォーマンス劣化**: 過度な設定による性能低下

### 対策
- メモリ監視機能の実装
- Platform固有の最適化
- 自動的な設定調整機能

## 成功指標

### 機能指標
- [ ] 既存コードの100%動作保証
- [ ] 新設定でのパフォーマンス向上（軽量設定で20%以上のFPS向上）
- [ ] originalImage機能の正常動作

### パフォーマンス指標
- [ ] 軽量設定: 30fps以上維持
- [ ] 標準設定: 現在と同等のパフォーマンス
- [ ] 詳細設定: 15fps以上維持

### ユーザビリティ指標
- [ ] プリセット設定での簡単な利用
- [ ] ドキュメントの完全性
- [ ] サンプルコードの提供

## 今後の拡張可能性

### 将来的な機能
- **リアルタイム品質調整**: 自動的なパフォーマンス最適化
- **カスタムフィルター**: ユーザー定義のデータ処理
- **統計機能**: 長期間のパフォーマンス分析
- **クラウド連携**: リアルタイムデータのクラウド送信

### アーキテクチャ拡張
- **プラグイン型設計**: サードパーティ拡張への対応
- **設定プロファイル**: 複数設定の管理機能
- **パフォーマンス最適化エンジン**: AI による自動調整

---

## 更新履歴

- 2025/05/31: 初版作成
- 今後の実装進捗に応じて更新予定