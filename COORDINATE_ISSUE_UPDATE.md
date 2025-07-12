# YOLO Flutter Plugin - Coordinate Issue Root Cause

## 問題の真の原因

BasePredictor.swiftの`getModelInputSize`関数に問題があります。

### 元のコード（問題あり）：
```swift
if shape.count >= 2 {
  let height = shape[0].intValue
  let width = shape[1].intValue
  return (width: width, height: height)
}
```

### 問題の詳細：
CoreMLモデルの`multiArrayConstraint`のshapeは**NCHW形式**（batch, channels, height, width）です：
- 640x640モデル: shape = [1, 3, 640, 640]
- 320x320モデル: shape = [1, 3, 320, 320]

元のコードは：
- shape[0] = 1 (batch size) → heightと誤解釈
- shape[1] = 3 (channels) → widthと誤解釈
- 結果：modelInputSize = (width: 3, height: 1)

### なぜ「惜しい位置」なのか：

考えられる理由：
1. **2つの入力形式**：
   - 一部のモデルは`imageConstraint`を使用（正しくwidth/heightを取得）
   - 他のモデルは`multiArrayConstraint`を使用（間違った解釈）

2. **どこかに補正ロジックがある可能性**：
   - Vision frameworkが内部で補正している
   - または別の場所でフォールバック処理がある

### 修正案：
```swift
if let multiArrayConstraint = inputDescription.multiArrayConstraint {
  let shape = multiArrayConstraint.shape
  if shape.count == 4 {
    // NCHW形式: [batch, channels, height, width]
    let height = shape[2].intValue
    let width = shape[3].intValue
    return (width: width, height: height)
  } else if shape.count >= 2 {
    // フォールバック
    let height = shape[0].intValue
    let width = shape[1].intValue
    return (width: width, height: height)
  }
}
```

### テスト方法：
1. print文を追加してmodelInputSizeの値を確認
2. 640x640モデルと他のサイズのモデルでshape配列の内容を比較
3. imageConstraintとmultiArrayConstraintのどちらが使われているか確認