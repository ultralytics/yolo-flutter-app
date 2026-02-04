# 🚀 Auto.js YOLO 插件 (Ultralytics 定制版)

本项目是基于 Ultralytics YOLO 和 TensorFlow Lite 的 **Auto.js 原生插件**。
专为 Auto.js 自动化脚本设计，提供极速、轻量级的本地物体检测能力。

## ✨ 核心特性

- **纯原生体验**: 模块轻量化，专为 Auto.js 自动化设计
- **模型热加载**: 支持外部绝对路径模型，无需重新打包
- **高性能**: 针对 arm64-v8a 优化，支持 GPU 加速
- **专一性**: 专注于目标检测，支持 YOLOv8/v11/v26

## 🛠️ 构建要求

- **JDK**: 21+
- **Gradle**: 8.11.1+
- **Kotlin**: 2.1.0
- **Android Gradle Plugin**: 8.8.0
- **Compile SDK**: 35
- **Target SDK**: 35
- **Min SDK**: 24 (Android 7.0+)

## 📦 快速构建

```bash
# 清理并构建 Debug 版本
./gradlew clean assembleDebug

# 构建 Release 版本
./gradlew assembleRelease
```

构建产物: `build/outputs/apk/debug/yolo-debug.apk`

### 模型准备

插件不内置模型文件以保持轻量。请准备好 `.tflite` 模型（建议 INT8 量化）并放置在设备存储中。
例如放置在：`/sdcard/Documents/yolo_model.tflite`

---

## 💻 使用指南 (Auto.js)

### 1. 安装插件

将构建好的 APK 安装到手机上，Auto.js 会自动识别并加载插件。

### 2. 编写脚本

```javascript
// 1. 加载插件
var yolo = plugins.load("com.ultralytics.yolo.plugin");

// 2. 加载模型 (必须使用绝对路径)
// 找不到文件会直接报错
var modelPath = "./modules/yolo26n_int8.tflite";
yolo.loadModel(modelPath, true); // true 为开启 GPU

// 3. 配置参数
yolo.setConfidence(0.35); // 过滤置信度低于 0.35 的结果

// 4. 截图并检测
if (requestScreenCapture()) {
  var img = captureScreen();

  // 返回结果对象包含: boxes (列表), speed (耗时), fps, origShape 等
  var result = yolo.detect(img);

  if (result && result.boxes.length > 0) {
    log("检测到 " + result.boxes.length + " 个目标");
    result.boxes.forEach((box) => {
      // box 结构: { cls: "person", conf: 0.85, xywh: RectF(...) }
      log(`类别: ${box.cls}, 置信度: ${box.conf}, 位置: ${box.xywh}`);

      // 可视化绘制 (示例)
      // canvas.drawRect(...)
    });
  }
}
```

## 📂 项目结构

```text
├── src/main/
│   ├── assets/
│   │   └── plugin-yolo/
│   │       └── index.js            # JS 接口定义 (Rhino 模块)
│   ├── kotlin/com/ultralytics/yolo/
│   │   ├── YOLOPlugin.kt           # 插件入口，JS 桥接
│   │   ├── YOLOPluginRegistry.kt   # 插件注册
│   │   ├── YOLO.kt                 # 统一 API 入口
│   │   ├── ObjectDetector.kt       # 核心引擎，TFLite 推理与后处理
│   │   ├── YOLOResult.kt           # 结果数据结构 (Box, Size)
│   │   ├── YOLOConstants.kt        # 常量定义
│   │   └── Utils.kt                # 文件加载通用工具
│   └── AndroidManifest.xml         # 插件声明
└── build.gradle                    # 构建配置
```

## 📝 模型导出要求

为了获得最佳性能，请使用 **YOLOv26** 导出 **TFLite INT8** 量化模型：

```bash
yolo export model=yolo26n.pt format=tflite int8
```

- **不再支持 Assets 加载模型**：出于灵活性考虑，模型加载必须显式传入包含 `/sdcard/` 等前缀的绝对路径。
- **简化后处理**：内置高效 Kotlin NMS。
- **单架构支持**：默认仅生成 `arm64-v8a` 代码。

## 📄 开源协议

[AGPL-3.0 License](LICENSE)
