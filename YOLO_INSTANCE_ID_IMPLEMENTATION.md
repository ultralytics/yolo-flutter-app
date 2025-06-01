# YOLO インスタンスID実装方針

## 概要
現在のYOLOクラスのシングルトン実装（iOS側）をインスタンスID方式に変更し、複数のYOLOインスタンスを同時に管理できるようにする。

## 現状分析

### 現在の実装状況
- **Flutter/Dart側**: 複数インスタンス対応済み
- **iOS側**: SingleImageYOLO.sharedによるシングルトン実装
- **Android側**: インスタンスベースだが、プラグインごとに1インスタンス
- **問題点**: iOS側のシングルトンにより、同時に複数のモデルを使用できない

## 実装戦略

### Phase 1: インスタンスID管理システムの設計と実装

#### 1.1 Flutter側のインスタンスマネージャー
```dart
// lib/yolo_instance_manager.dart
class YOLOInstanceManager {
  static final Map<String, YOLO> _instances = {};
  static final Map<String, String> _instanceChannels = {};
  static int _instanceCounter = 0;
  
  static String createInstance() {
    final instanceId = 'yolo_${_instanceCounter++}';
    final channelName = 'yolo_single_image_channel_$instanceId';
    _instances[instanceId] = YOLO._withChannel(channelName, instanceId);
    _instanceChannels[instanceId] = channelName;
    return instanceId;
  }
  
  static YOLO? getInstance(String instanceId) => _instances[instanceId];
  
  static void disposeInstance(String instanceId) {
    _instances[instanceId]?.dispose();
    _instances.remove(instanceId);
    _instanceChannels.remove(instanceId);
  }
}
```

#### 1.2 メソッドチャンネルの変更
- 静的チャンネルからインスタンスベースのチャンネルへ
- 各インスタンスが独自のチャンネルを持つ

### Phase 2: プラットフォーム側の実装

#### 2.1 iOS実装 (Swift)
```swift
// YOLOInstanceManager.swift
@MainActor
class YOLOInstanceManager {
    static let shared = YOLOInstanceManager()
    private var instances: [String: YOLO] = [:]
    private var loadingStates: [String: Bool] = [:]
    
    private init() {}
    
    func createInstance(instanceId: String) throws -> YOLO {
        let yolo = YOLO()
        instances[instanceId] = yolo
        return yolo
    }
    
    func getInstance(instanceId: String) -> YOLO? {
        return instances[instanceId]
    }
    
    func removeInstance(instanceId: String) {
        instances.removeValue(forKey: instanceId)
        loadingStates.removeValue(forKey: instanceId)
    }
}
```

#### 2.2 Android実装 (Kotlin)
```kotlin
// YOLOInstanceManager.kt
class YOLOInstanceManager {
    private val instances = mutableMapOf<String, YOLO>()
    
    fun createInstance(instanceId: String): YOLO {
        val yolo = YOLO()
        instances[instanceId] = yolo
        return yolo
    }
    
    fun getInstance(instanceId: String): YOLO? {
        return instances[instanceId]
    }
    
    fun removeInstance(instanceId: String) {
        instances[instanceId]?.close()
        instances.remove(instanceId)
    }
}
```

### Phase 3: API設計 (簡素化版)

#### 3.1 新しいシンプルなAPI
```dart
// インスタンスの作成 - 通常のオブジェクトのように
final detector = YOLO(
  modelPath: 'yolov8n.tflite',
  task: YOLOTask.detect,
);

final segmenter = YOLO(
  modelPath: 'yolov8n-seg.tflite', 
  task: YOLOTask.segment,
);

// モデルのロード
await detector.loadModel();
await segmenter.loadModel();

// 推論の実行
final detectResult = await detector.predict(imageBytes);
final segmentResult = await segmenter.predict(imageBytes);

// インスタンスIDへのアクセス（必要な場合）
print('Detector ID: ${detector.instanceId}');
print('Segmenter ID: ${segmenter.instanceId}');

// インスタンスの破棄
await detector.dispose();
await segmenter.dispose();
```

#### 3.2 内部実装
```dart
class YOLO {
  late final String _instanceId;
  late final MethodChannel _channel;
  
  // プロパティとしてインスタンスIDを公開
  String get instanceId => _instanceId;
  
  YOLO({required this.modelPath, required this.task}) {
    // インスタンスIDを自動生成
    _instanceId = 'yolo_${DateTime.now().millisecondsSinceEpoch}_${this.hashCode}';
    
    // インスタンス専用のチャンネルを作成
    final channelName = 'yolo_single_image_channel_$_instanceId';
    _channel = MethodChannel(channelName);
    
    // プラットフォーム側で初期化
    _initializeInstance();
  }
}
```

### Phase 4: メソッドチャンネルの実装詳細

#### 4.1 Flutter側
```dart
// メソッド呼び出しにinstanceIdを含める
Future<void> loadModelWithInstance({
  required String instanceId,
  required String model,
  required Task task,
}) async {
  final channel = MethodChannel(_instanceChannels[instanceId]!);
  await channel.invokeMethod('loadModel', {
    'instanceId': instanceId,
    'model': model,
    'task': task.name,
  });
}
```

#### 4.2 iOS側
```swift
// チャンネルごとにインスタンスを管理
public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let instanceId = args["instanceId"] as? String else {
        result(FlutterError(...))
        return
    }
    
    switch call.method {
    case "createInstance":
        createInstance(instanceId: instanceId, result: result)
    case "loadModel":
        loadModel(instanceId: instanceId, args: args, result: result)
    // ... 他のメソッド
    }
}
```

## 実装手順

### Step 1: Flutter側の基盤実装
1. YOLOInstanceManagerクラスの作成
2. YOLOクラスへのinstanceId対応追加
3. メソッドチャンネルの拡張

### Step 2: iOS側の実装
1. SingleImageYOLOシングルトンの除去
2. YOLOInstanceManagerの実装
3. メソッドハンドラーの更新

### Step 3: Android側の実装
1. YOLOInstanceManagerの実装
2. メソッドハンドラーの更新

### Step 4: テストとドキュメント
1. ユニットテストの作成
2. 統合テストの作成
3. サンプルコードの更新

## テスト計画

### ユニットテスト
- 複数インスタンスの作成・破棄
- 異なるモデルの同時ロード
- インスタンスごとの独立した推論

### 統合テスト
- iOS/Androidでの動作確認
- メモリリークのチェック
- パフォーマンステスト

## 注意事項

1. **メモリ管理**: 複数インスタンスによるメモリ使用量の増加に注意
2. **スレッドセーフティ**: 並行アクセス時の安全性を確保
3. **後方互換性**: 既存のAPIを壊さないよう注意
4. **エラーハンドリング**: 無効なinstanceIdへの適切な対応

## タイムライン

- **Week 1**: Flutter側の実装とテスト
- **Week 2**: iOS側の実装とテスト
- **Week 3**: Android側の実装とテスト
- **Week 4**: 統合テストとドキュメント作成