<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 YOLO Flutter - Ultralytics 官方插件

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://app.codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

Ultralytics YOLO Flutter 是官方 Flutter 插件，用于在 iOS 和 Android 应用中运行 YOLO 模型。它支持[目标检测](https://docs.ultralytics.com/tasks/detect)、[实例分割](https://docs.ultralytics.com/tasks/segment)、[语义分割](https://docs.ultralytics.com/tasks/semantic)、[图像分类](https://docs.ultralytics.com/tasks/classify)、[姿态估计](https://docs.ultralytics.com/tasks/pose)和[旋转框检测](https://docs.ultralytics.com/tasks/obb)，并提供两种核心用法：

- `YOLO`：单张图片推理
- `YOLOView`：实时相机推理

这个插件的目标很直接：要么使用官方模型 ID，要么把你自己的导出模型丢进应用里，让插件自动解析任务元数据。

<div align="center">
  <br>
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" target="_blank"><img width="100%" src="https://github.com/user-attachments/assets/d5dab2e7-f473-47ce-bc63-69bef89ba52a" alt="Ultralytics YOLO iOS App previews"></a>
  <br>
  <br>
  <a href="https://github.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://x.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://space.bilibili.com/3546646073837954"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://apps.apple.com/us/app/idetection/id1452689527" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/app/app-store.svg" width="15%" alt="Apple App store"></a>
</div>

## ✨ 为什么用这个插件

- Ultralytics 官方插件
- 一套 Dart API 同时覆盖 Android 和 iOS
- 基于元数据加载模型，并内置官方模型下载与缓存
- 同时支持实时相机推理和单图推理
- 提供阈值、GPU、流式数据等生产环境控制项
- 支持 YOLO26 和 YOLO11 系列模型

| 功能              | Android | iOS | 说明                                                          |
| ----------------- | ------- | --- | ------------------------------------------------------------- |
| 目标检测          | ✅      | ✅  | 边界框、类别和置信度                                          |
| 实例分割          | ✅      | ✅  | 实例掩膜、边界框和类别                                        |
| 语义分割          | ✅      | ✅  | 每个像素的密集类别掩膜                                        |
| 图像分类          | ✅      | ✅  | Top-1 类别预测和分数                                          |
| 姿态估计          | ✅      | ✅  | 关键点、边界框和置信度                                        |
| 旋转框（OBB）检测 | ✅      | ✅  | 旋转框和多边形角点                                            |
| 实时相机推理      | ✅      | ✅  | 使用 `YOLOView` 构建实时相机场景                              |
| 单图推理          | ✅      | ✅  | 使用 `YOLO` 处理图片字节                                      |
| 官方模型          | ✅      | ✅  | 内置模型 ID 发现、下载和缓存                                  |
| 自定义模型        | ✅      | ✅  | Android 用 LiteRT（TFLite），iOS 用 Core ML，并优先读取元数据 |

## ⚡ 快速开始

安装插件：

插件地址：https://pub.dev/packages/ultralytics_yolo

```yaml
dependencies:
  ultralytics_yolo: ^0.5.1
```

```bash
flutter pub get
```

先用默认官方模型跑起来：

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

final modelId = YOLO.defaultOfficialModel() ?? 'yolo26n';

YOLOView(
  modelPath: modelId,
  onResult: (results) {
    for (final r in results) {
      debugPrint('${r.className}: ${r.confidence}');
    }
  },
)
```

单图推理示例：

```dart
final yolo = YOLO(modelPath: 'yolo26n');
await yolo.loadModel();
final results = await yolo.predict(imageBytes);
```

**[▶️ 示例应用](./example)** | **[📖 安装指南](doc/install.md)** | **[⚡ 快速开始文档](doc/quickstart.md)**

## 📦 模型加载方式

插件支持三种模型来源。

### 1. 官方模型 ID

可直接使用默认官方模型，或传入指定官方 ID，例如 `yolo26n`：

```dart
final yolo = YOLO(modelPath: YOLO.defaultOfficialModel() ?? 'yolo26n');
```

插件会自动下载并缓存当前平台对应的官方产物。可通过 `YOLO.officialModels()` 查看当前平台可用的全部官方 ID。Android TFLite 统一从 Flutter `v0.3.5` release 下载，iOS Core ML 统一从 YOLO iOS `v8.3.0` release 下载，后续包版本发布不会改变模型 URL。

### 2. 你自己的导出模型

传入你自己的导出 YOLO 模型本地路径或 Flutter 资源路径：

```dart
final yolo = YOLO(modelPath: 'assets/models/my-finetuned-model.tflite');
```

如果模型导出时带有元数据，插件会自动推断 `task`。如果没有，就显式传入 `task`。

```dart
final yolo = YOLO(
  modelPath: 'assets/models/my-finetuned-model.tflite',
  task: YOLOTask.detect,
);
```

### 3. 远程模型 URL

传入 `http` 或 `https` URL，插件会先下载到应用存储，再完成加载。

## 🧭 官方模型还是自定义模型

| 场景                       | 推荐方式                        |
| -------------------------- | ------------------------------- |
| 想最快跑通接入             | 使用官方模型 ID，例如 `yolo26n` |
| 你训练或导出了自己的模型   | 使用自定义资源或本地文件        |
| 不同客户或环境需要不同模型 | 使用远程 URL                    |
| 希望插件自动推断 `task`    | 使用带元数据的导出模型          |
| 你的导出模型没有元数据     | 自定义模型并显式传入 `task`     |

官方模型可直接从 `YOLO.defaultOfficialModel()` 或 `YOLO.officialModels()` 开始；自定义模型则直接从你准备实际交付的导出文件开始。 Android TFLite 示例资产来自 [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5)， iOS Core ML 示例资产来自 [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)。

## 📥 把你自己的模型放进应用

对于自定义模型，应用侧配置应尽量简单：

- Android 原生资源：把 `.tflite` 放到 `android/app/src/main/assets`
- Android Flutter 资源：把 `.tflite` 放到 `assets/models/`
- iOS 工程资源：把 `.mlpackage` 或 `.mlmodel` 拖进 `ios/Runner.xcworkspace`
- iOS Flutter 资源：把 `.mlpackage.zip` 放到 `assets/models/`

然后把对应路径传给 `modelPath` 即可。

### iOS 导出注意事项

YOLO26 模型是 NMS-free 的，Core ML 导出请使用 `nms=False` 和 `end2end=True`（Swift 解码器消费的就是 YOLO26 端到端输出格式）：

```python
from ultralytics import YOLO

# Square [640, 640] works best when one model must run in both portrait and landscape.
# Ultralytics imgsz order is [height, width]; use [640, 384] for portrait-only or [384, 640] for landscape-only.
YOLO("yolo26n.pt").export(format="coreml", nms=False, end2end=True, imgsz=[640, 640])
```

其他任务可以使用默认导出参数，`imgsz` 同样优先使用方形尺寸。 固定方向场景再使用对应的长宽比。

### Android 导出注意事项

Android 推理通过 `CompiledModel` API 运行在 [LiteRT](https://ai.google.dev/edge/litert) 2.x （Google 对 TensorFlow Lite 的重新命名）之上，并带有自动的 GPU → CPU 加速器降级链： 当模型支持时，插件会把整张计算图编译到 GPU 上运行，否则回退到 CPU 上的 XNNPACK。

要获得最快的 GPU 路径，请导出 fp16、非端到端（non-end-to-end）的 TFLite 模型：

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
```

在三星 Galaxy S26 上，官方 `yolo26n_int8.tflite` 通过 LiteRT OpenCL GPU delegate 完整编译，在实时相机示例中约为 15 FPS / 每帧约 32 毫秒。数值为近似值，会随设备而变化。

int8 资产是官方下载产物（体积更小），但 int8 的 GPU 覆盖取决于设备驱动和计算图；不支持的图或算子会回退到 CPU，而不会报错。

## 🎯 该用哪个 API

如果你已经拿到了图片字节并且只想做单次推理，用 `YOLO`：

```dart
final yolo = YOLO(modelPath: 'yolo26n');
await yolo.loadModel();
final results = await yolo.predict(imageBytes);
```

如果你要做实时相机推理，用 `YOLOView`：

```dart
final controller = YOLOViewController();

YOLOView(
  modelPath: 'yolo26n',
  controller: controller,
  onResult: (results) {},
)

await controller.switchModel('assets/models/custom.tflite', YOLOTask.detect);
```

如果你想要完整的 Ultralytics 相机 UI，用 `YOLOShowcase`：

```dart
YOLOShowcase(
  initialTask: YOLOTask.detect,
  initialModelSize: 'n',
  onCapture: (bytes) {},
)
```

## 🔄 从 0.3.x UI API 迁移

0.4.0 版本移除了旧的 Dart 侧叠层/控件层。相机检测结果现在由 `YOLOView` 原生渲染；Flutter 仅负责周围的应用控件。

| 已移除的 0.3.x API                               | 0.4.0 替代方案                                                                           |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `YOLOOverlay`, `YOLOOverlayTheme`                | 删除这些 widget。使用原生 `YOLOView` 叠层，或通过 `onResult`/`YOLO.predict()` 自行渲染。 |
| `YOLOControls`                                   | 使用 `YOLOShowcase` 获取完整 UI，或直接组合导出的 Material widget。                      |
| `YOLOView.showNativeUI`                          | 使用 `YOLOShowcase` 获取内置控件；使用裸 `YOLOView` 自行构建 UI。                        |
| `YOLOView.showOverlays`, `YOLOView.overlayTheme` | 无构造参数替代。相机叠层绘制为原生实现，不再从 Dart 侧控制主题。                         |
| `YOLOViewController.setShowUIControls()`         | 在 `YOLOView` 外侧自行显示/隐藏 Flutter 控件。                                           |
| `YOLOViewController.setShowOverlays()`           | 无控制器替代。`capturePhoto(withOverlays: false)` 仅影响捕获的 JPEG 输出。               |

## 🧩 推荐接入模式

| 应用类型                                | 推荐模型加载方式                                 |
| --------------------------------------- | ------------------------------------------------ |
| 实时相机场景                            | `YOLOView(modelPath: 'yolo26n')`                 |
| 图库或单图推理流程                      | `YOLO(modelPath: 'yolo26n')`                     |
| 应用内置自定义模型                      | `YOLO(modelPath: 'assets/models/custom.tflite')` |
| 同时支持 Core ML 与 TFLite 的跨平台应用 | 使用各平台对应导出文件，并让元数据决定 `task`    |
| 运行时动态切换模型                      | `YOLOViewController.switchModel(...)`            |

## 📚 文档

| 指南                                   | 说明                           |
| -------------------------------------- | ------------------------------ |
| **[安装指南](doc/install.md)**         | 环境要求与平台配置             |
| **[快速开始](doc/quickstart.md)**      | 最短路径跑通第一个示例         |
| **[模型指南](doc/models.md)**          | 官方模型、自定义模型、导出流程 |
| **[使用指南](doc/usage.md)**           | 常见应用模式与示例             |
| **[API 参考](doc/api.md)**             | 完整 API 文档                  |
| **[性能优化](doc/performance.md)**     | 性能调优与控制项               |
| **[故障排查](doc/troubleshooting.md)** | 常见问题与修复方法             |

## 🤝 社区与支持

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics) [![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/) [![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

- **💬 有问题？** [Discord](https://discord.com/invite/ultralytics) | [Forums](https://community.ultralytics.com/) | [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues)
- **🐛 发现 bug？** [在这里提交](https://github.com/ultralytics/yolo-flutter-app/issues/new)
- **💡 想提功能建议？** [欢迎告诉我们](https://github.com/ultralytics/yolo-flutter-app/issues/new)

## 💡 参与贡献

Ultralytics 的成长离不开社区协作，我们非常重视你的贡献。无论是修复 bug、增强功能还是改进文档，你的参与都非常重要。请查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing)以了解如何参与，也欢迎通过[问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享你的反馈。衷心感谢所有贡献者的支持！🙏

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 许可证

Ultralytics 提供两种许可证，以适应不同需求：

- **AGPL-3.0 License**：适合学生、研究人员以及热衷于开源协作的开发者。这是一个经 [OSI 批准](https://opensource.org/license/agpl-3.0)的许可证，鼓励知识共享和开放贡献。详情请参阅 [LICENSE](https://github.com/ultralytics/yolo-flutter-app/blob/main/LICENSE) 文件。
- **Enterprise License**：适用于商业应用，允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品与服务中，而无需遵守 AGPL-3.0 的开源要求。如有商业使用需求，请了解[企业许可证](https://www.ultralytics.com/license)。

## 🔗 相关资源

### 原生 iOS 开发

如果你希望在 iOS 应用中直接使用 YOLO 模型与 Swift 集成，而不是通过 Flutter，可以查看我们的专用 iOS 仓库：

👉 **[Ultralytics YOLO iOS App](https://github.com/ultralytics/yolo-ios-app)** - 一个原生 iOS 应用，演示如何使用 Ultralytics YOLO 模型进行实时目标检测、分割、分类、姿态估计和旋转框检测。

该仓库提供：

- 面向 iOS 的纯 Swift 实现
- 直接的 Core ML 集成
- 原生 iOS UI 组件
- 多种 YOLO 任务的示例代码
- 针对 iOS 性能的优化

## 📮 联系方式

如果你在使用 Ultralytics YOLO 时遇到问题，或有功能建议，请通过 [GitHub Issues](https://github.com/ultralytics/yolo-flutter-app/issues) 提交。若想参与更广泛的讨论、提问或获取社区支持，欢迎加入我们的 [Discord](https://discord.com/invite/ultralytics) 服务器。

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://x.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://space.bilibili.com/3546646073837954"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
