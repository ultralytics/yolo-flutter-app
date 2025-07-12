# YOLO Flutter Plugin - Pose/OBB座標問題の最終分析

## 現状
- **動作している**: detect, segment, classify（リアルタイム・単一画像とも）
- **問題がある**: pose, obb（640x640以外のモデルで座標がずれる）

## 問題の核心

### 1. 座標処理の違い

#### Segmenter（正常動作）
```swift
// postProcessSegment: 生の座標を取得
let x = pointerWrapper.pointer[j]
let y = pointerWrapper.pointer[numAnchors + j]

// 後で正規化（修正済み）
let rect = CGRect(
  x: box.minX / CGFloat(self.modelInputSize.width),
  y: box.minY / CGFloat(self.modelInputSize.height), ...)
```

#### ObbDetector（問題あり）
```swift
// postProcessOBB: すぐに正規化
let cx = pointer[i] / inputW
let cy = pointer[numAnchors + i] / inputH

// OBBクラスには正規化座標（0-1）が格納される
let obb = OBB(cx: cx, cy: cy, w: w, h: h, angle: angle)
```

#### PoseEstimater（問題あり）
```swift
// PostProcessPose: 生の座標を取得
let x = featurePointer[j]
let y = featurePointer[numAnchors + j]

// 後で正規化
let Nx = box.origin.x / CGFloat(modelInputSize.width)
let Ny = box.origin.y / CGFloat(modelInputSize.height)
```

### 2. 問題の原因

1. **モデル出力の解釈**
   - モデルの生の出力がピクセル座標なのか、既に正規化されているのかが不明確
   - タスクごとに異なる前提で実装されている

2. **BasePredictor.swift**のモデル入力サイズ取得（修正済み）
   - NCHW形式の正しい解釈が必要
   - shape[2]がheight、shape[3]がwidth

3. **imageCropAndScaleOption**
   - `.scaleFill`が正しい（`.scaleFit`ではさらに悪化）
   - 問題は前処理ではなく、座標の解釈にある

### 3. なぜ640x640では動作するのか

考えられる理由：
1. 640x640が最も一般的なYOLOモデルサイズ
2. モデルの出力形式が640x640を前提に最適化されている
3. 座標の誤差が目立ちにくい

### 4. 解決の方向性

1. **モデル出力の単位を確認**
   - デバッグログで生の座標値を確認
   - 640x640と他のサイズで比較

2. **座標処理の統一**
   - すべてのタスクで同じ座標解釈を使用
   - Segmenterの成功パターンに合わせる

3. **テスト**
   - 異なるモデルサイズ（320x320, 384x640等）
   - 異なるアスペクト比の画像