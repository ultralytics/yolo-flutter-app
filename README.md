# 🚀 Auto.js YOLO 插件 (Ultralytics 定制版)

本项目是基于 Ultralytics YOLO 和 TensorFlow Lite 的 **Auto.js Pro 原生插件**。

---

## ✨ 核心特性

- **纯原生体验**: 模块轻量化，专为 Auto.js 自动化设计。
- **模型热加载**: **仅支持加载外部绝对路径模型**，无需重新打包插件即可更换模型。
- **高性能**: 针对 `arm64-v8a` 优化，支持 **GPU 加速**。
- **专一性**: 专注于目标检测 (Object Detection)，支持 YOLOv8/v11/v26。

## 🛠️ 构建与部署

### 构建 APK
```bash
./gradlew assembleDebug
```
构建产物: `build/outputs/apk/debug/*.apk`

### 模型准备
插件不内置模型文件以保持轻量。请准备好 `.tflite` 模型（建议 INT8 量化）并放置在设备存储中。
例如放置在：`/sdcard/Documents/yolo_model.tflite`

---

## 💻 使用指南 (Auto.js)

### 代码示例
```javascript
// 1. 加载插件
var yolo = $plugins.load("com.ultralytics.yolo.plugin");

// 2. 加载模型 (必须使用绝对路径)
// 找不到文件会直接报错
var modelPath = "/sdcard/Documents/yolo26n_int8.tflite";
yolo.loadModel(modelPath, true); // true 为开启 GPU

// 3. 配置参数
yolo.setConfidence(0.35); // 过滤置信度低于 0.35 的结果

// 4. 截图并检测
if (requestScreenCapture()) {
    var img = captureScreen();
    var result = yolo.detect(img);
    
    if (result && result.boxes.length > 0) {
        log("检测到 " + result.boxes.length + " 个目标");
        result.boxes.forEach(box => {
            log("标签: " + box.cls + " 置信度: " + box.conf.toFixed(2));
            log("位置: " + box.xywh.toString());
        });
    }
}
```

## � 重要变更 (与 Flutter 版不同)
- **不再支持 Assets 加载模型**：出于灵活性考虑，模型加载必须显式传入包含 `/sdcard/` 等前缀的绝对路径。
- **简化后处理**：内置高效 Kotlin NMS。
- **单架构支持**：默认仅生成 `arm64-v8a` 代码。

## 📄 开源协议
[AGPL-3.0 License](LICENSE)
