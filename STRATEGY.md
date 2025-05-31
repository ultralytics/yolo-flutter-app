# YOLOView ストリーム機能追加戦略

## 現状分析

### 問題の経緯
1. **元々**: mainブランチは専用クラス（ObjectDetector, Segmenter, PoseEstimator, ObbDetector）で描画・モデル切り替えが正常動作
2. **変更**: YOLOPlatformViewでストリーム機能を実装、originalImage対応のためYOLO統一クラスに変更
3. **結果**: 描画ずれ、モデル切り替え黒画面問題が発生

### 現在の状態 (問題あり)
- ✅ ストリーム機能: YOLOPlatformView.ktで実装済み
- ✅ originalImageサポート: YOLO統一クラスで対応
- ❌ **描画ずれ**: YOLO統一クラスで座標系が変わった
- ❌ **モデル切り替え黒画面**: 複雑な制御ロジックが原因
- ❌ **YOLOViewでストリーム機能なし**: PlatformViewでのみ実装

### オリジナル状態 (動作良好)
- ✅ **描画正常**: 専用クラス使用で座標系正確
- ✅ **モデル切り替え正常**: シンプルな実装
- ❌ ストリーム機能なし
- ❌ originalImageサポートなし

## 理想の最終状態

### 目標
- ✅ **描画正常**: 専用クラスを維持
- ✅ **YOLOViewでストリーム機能**: 新規追加
- ✅ **originalImageサポート**: YOLOView内部のbitmapを再利用
- ✅ **モデル切り替え正常**: シンプル実装維持

## 戦略: オリジナルから始めるアプローチ

### なぜオリジナル状態から始めるか

#### 時間比較
- **オリジナルから**: 約1-2時間（機能追加のみ）
- **現状修正**: 約3-5時間 + デバッグ時間（複数問題の修正）

#### リスク比較
- **オリジナルから**: 低リスク（既存機能は動作保証済み）
- **現状修正**: 高リスク（何が壊れているか完全把握困難）

### originalImageの革新的アプローチ

#### 従来の複雑な方法（不要）
```kotlin
// YOLO統一クラスを使ってoriginalImageを取得
val yoloResult = YOLO.predictWithBitmap(bitmap, w, h, includeOriginalImage = true)
```

#### 新しいシンプルな方法（採用）
```kotlin
// YOLOView内でImageProxyから作ったbitmapをそのまま使用
private fun onFrame(imageProxy: ImageProxy) {
    val bitmap = ImageUtils.toBitmap(imageProxy) // 既にある！
    
    // 推論（専用クラス使用）
    val result = predictor.predict(bitmap, h, w, rotateForCamera = true)
    
    // ストリーム設定でoriginalImageが必要なら
    if (streamConfig?.includeOriginalImage == true) {
        result.originalImage = bitmap  // そのまま設定
    }
    
    // ストリームに送信
    streamCallback?.invoke(convertResultToStreamData(result))
}
```

## ストリーム実装の詳細戦略

### 既存実装の活用判断 🤔

#### 記憶すべき重要な実装
YOLOPlatformView.ktの以下の実装は**極めて価値が高く、必ず保持すべき**：

#### 🚨 **重要な除外決定**: annotatedImage
**YOLOViewではannotatedImageを生成しない**：
- **技術的理由**: YOLOViewはCanvas描画（リアルタイム）、bitmap生成（ファイル）は別物
- **アーキテクチャ理由**: overlayViewで直接描画、画像として保存しない
- **パフォーマンス理由**: 不要な画像生成処理を避ける
- **実装理由**: YOLOView実装には存在しない機能

→ **ストリーム設定からannotatedImageは完全除外**

##### 1. Mask データ変換
```kotlin
// detectionIndex重要！（class indexではない）
if (includeMasks && result.masks != null && detectionIndex < result.masks!!.masks.size) {
    val maskData = result.masks!!.masks[detectionIndex] // detection順序でアクセス
    val maskDataDouble = maskData.map { row ->
        row.map { it.toDouble() } // Flutter互換性のためDouble変換
    }
    detection["mask"] = maskDataDouble
}
```

##### 2. Keypoints データ変換
```kotlin
if (includePoses && detectionIndex < result.keypointsList.size) {
    val keypoints = result.keypointsList[detectionIndex]
    // フラット配列 [x1, y1, conf1, x2, y2, conf2, ...]
    val keypointsFlat = mutableListOf<Double>()
    for (i in keypoints.xy.indices) {
        keypointsFlat.add(keypoints.xy[i].first.toDouble())
        keypointsFlat.add(keypoints.xy[i].second.toDouble())
        if (i < keypoints.conf.size) {
            keypointsFlat.add(keypoints.conf[i].toDouble())
        } else {
            keypointsFlat.add(0.0) // デフォルト信頼度
        }
    }
    detection["keypoints"] = keypointsFlat
}
```

##### 3. OBB データ変換（最も複雑）
```kotlin
if (includeOBB && detectionIndex < result.obb.size) {
    val obbResult = result.obb[detectionIndex]
    val obbBox = obbResult.box
    
    // 4角形への変換
    val polygon = obbBox.toPolygon()
    val points = polygon.map { point ->
        mapOf("x" to point.x.toDouble(), "y" to point.y.toDouble())
    }
    
    // 包括的なOBBデータ
    val obbDataMap = mapOf(
        "centerX" to obbBox.cx.toDouble(),
        "centerY" to obbBox.cy.toDouble(),
        "width" to obbBox.w.toDouble(),
        "height" to obbBox.h.toDouble(),
        "angle" to obbBox.angle.toDouble(), // radians
        "angleDegrees" to (obbBox.angle * 180.0 / Math.PI), // degrees
        "area" to obbBox.area.toDouble(),
        "points" to points, // 4 corner points
        "confidence" to obbResult.confidence.toDouble(),
        "className" to obbResult.cls,
        "classIndex" to obbResult.index
    )
    detection["obb"] = obbDataMap
}
```

##### 4. パフォーマンス制御（maxFPS/throttling）
```kotlin
// 設定解析（型安全）
private fun setupThrottlingFromConfig() {
    val maxFPS = when (maxFPSValue) {
        is Int -> maxFPSValue
        is Double -> maxFPSValue.toInt()
        is String -> maxFPSValue.toIntOrNull()
        else -> null
    }
    if (maxFPS != null && maxFPS > 0) {
        targetFrameInterval = (1_000_000_000L / maxFPS) // FPS→ナノ秒
    }
    
    val throttleMs = when (throttleMsValue) {
        is Int -> throttleMsValue
        is Double -> throttleMsValue.toInt() 
        is String -> throttleMsValue.toIntOrNull()
        else -> null
    }
    if (throttleMs != null && throttleMs > 0) {
        throttleInterval = throttleMs * 1_000_000L // ms→ナノ秒
    }
}

// フレームスキップ判定
private fun shouldProcessFrame(): Boolean {
    val now = System.nanoTime()
    
    // maxFPS制御
    targetFrameInterval?.let { interval ->
        if (now - lastInferenceTime < interval) return false
    }
    
    // throttleInterval制御  
    throttleInterval?.let { interval ->
        if (now - lastInferenceTime < interval) return false
    }
    return true
}
```

##### 5. Detection Index問題の解決
```kotlin
// 🚨 重要：detectionIndexを使う（class indexではない）
for ((detectionIndex, box) in result.boxes.withIndex()) {
    // detectionIndexでmasks, keypoints, obbにアクセス
    // box.indexはclass indexなので配列アクセスに使ってはダメ
}
```

#### 戦略修正：既存実装を活用

**結論: 既存のストリーム実装は絶対に活用すべき**

理由：
1. **複雑なデータ変換**: 特にOBBの4角形変換は一から書くのは困難
2. **Index問題の解決済み**: detectionIndex vs classIndexの混同を解決済み
3. **Flutter互換性**: Double変換など細かい配慮済み
4. **テスト済み**: 既に動作確認されている

## 実装計画

### Phase 1: オリジナル状態に復元
```bash
# 現在の変更を退避
git stash

# オリジナル状態に戻る
git reset --hard HEAD~5  # 適切なコミットを指定
```

### Phase 2: YOLOViewにストリーム機能追加

#### 必要な追加要素
```kotlin
class YOLOView {
    // ストリーム設定
    private var streamConfig: YOLOStreamConfig? = null
    private var streamCallback: ((Map<String, Any>) -> Unit)? = null
    
    // 設定メソッド
    fun setStreamConfig(config: YOLOStreamConfig?) {
        this.streamConfig = config
    }
    
    fun setStreamCallback(callback: ((Map<String, Any>) -> Unit)?) {
        this.streamCallback = callback
    }
    
    // 拡張されたonFrame
    private fun onFrame(imageProxy: ImageProxy) {
        val bitmap = ImageUtils.toBitmap(imageProxy) ?: return
        
        // 既存の推論処理（専用クラス使用）
        val result = predictor?.predict(bitmap, h, w, rotateForCamera = true)
        
        // originalImage設定（必要時のみ）
        if (streamConfig?.includeOriginalImage == true) {
            result?.originalImage = bitmap
        }
        
        // 既存の描画処理
        inferenceResult = result
        inferenceCallback?.invoke(result)
        
        // 新しいストリーム処理
        result?.let { 
            streamCallback?.invoke(convertResultToStreamData(it))
        }
    }
    
    // YOLOPlatformViewから移植
    private fun convertResultToStreamData(result: YOLOResult): Map<String, Any> {
        // 既存のYOLOPlatformView.convertResultToMap()の実装をコピー
    }
}
```

#### ストリーム設定クラス
```kotlin
data class YOLOStreamConfig(
    val includeDetections: Boolean = true,
    val includeClassifications: Boolean = true,
    val includeProcessingTimeMs: Boolean = true,
    val includeFps: Boolean = true,
    val includeMasks: Boolean = true,
    val includePoses: Boolean = true,
    val includeOBB: Boolean = true,
    val includeOriginalImage: Boolean = false,  // ImageProxyのbitmapを再利用
    val maxFPS: Int? = null,
    val throttleIntervalMs: Int? = null
    // annotatedImageは除外：YOLOViewはCanvas描画でbitmap生成しない
)
```

### Phase 3: YOLOPlatformViewでYOLOView機能を活用
```kotlin
class YOLOPlatformView {
    init {
        // YOLOViewのストリーム機能を設定
        yoloView.setStreamConfig(streamingConfig)
        yoloView.setStreamCallback { streamData ->
            // 既存のevent channel送信ロジック
            sendToFlutter(streamData)
        }
    }
}
```

## 技術的利点

### 1. 描画安定性
- 専用クラス使用で座標系の問題なし
- 既存の動作保証されたコード

### 2. パフォーマンス最適化
- bitmap再利用でメモリ効率向上
- 不要な変換処理なし

### 3. 保守性
- シンプルなアーキテクチャ
- 機能分離による理解しやすさ

### 4. 拡張性
- YOLOViewで直接ストリーム機能使用可能
- PlatformView以外でも利用可能

## リスク管理

### 低リスク要因
- 既存動作コードをベースとする
- 段階的実装
- 機能追加のみ（既存機能変更最小限）

### 注意点
- YOLOResultにoriginalImageフィールドが存在するか確認
- streamConfigの設定タイミング
- メモリリーク防止（bitmapの適切な管理）

## 成功指標

1. ✅ 描画が正常（ボックス、マスク、キーポイント、OBB）
2. ✅ モデル切り替えが正常
3. ✅ YOLOViewでストリーム機能動作
4. ✅ originalImage取得可能
5. ✅ パフォーマンス劣化なし

この戦略により、最小リスクで最大効果を得られる実装が可能になります。