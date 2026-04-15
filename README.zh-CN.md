<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 YOLO Flutter - Ultralytics 官方插件

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

Ultralytics YOLO Flutter 是官方 Flutter 插件，用于在 iOS 和 Android 应用中运行 YOLO 模型。它支持[目标检测](https://docs.ultralytics.com/tasks/detect/)、[实例分割](https://docs.ultralytics.com/tasks/segment/)、[图像分类](https://docs.ultralytics.com/tasks/classify/)、[姿态估计](https://docs.ultralytics.com/tasks/pose/)和[旋转框检测](https://docs.ultralytics.com/tasks/obb/)，并提供两种核心用法：

- `YOLO`：单张图片推理
- `YOLOView`：实时相机推理

这个插件的目标很直接：要么使用官方模型 ID，要么把你自己的导出模型丢进应用里，让插件自动解析任务元数据。

## ✨ 为什么用这个插件

| 功能     | Android | iOS |
| -------- | ------- | --- |
| 检测     | ✅      | ✅  |
| 分类     | ✅      | ✅  |
| 分割     | ✅      | ✅  |
| 姿态估计 | ✅      | ✅  |
| OBB 检测 | ✅      | ✅  |

- Ultralytics 官方插件
- 一套 Flutter API 同时覆盖 Android 和 iOS
- 基于模型元数据的加载流程
- 内置官方模型下载与缓存
- 同时支持实时相机推理和单图推理
- 提供阈值、GPU、流式数据等生产环境能力

## ⚡ 快速开始

安装插件：

```yaml
dependencies:
  ultralytics_yolo: ^0.2.0
```

```bash
flutter pub get
```

先用默认官方模型跑起来：

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

YOLOView(
  modelPath: 'yolo26n',
  onResult: (results) {
    for (final result in results) {
      print('${result.className}: ${result.confidence}');
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

直接使用官方 ID，例如 `yolo26n`：

```dart
final yolo = YOLO(modelPath: 'yolo26n');
```

插件会自动下载并缓存当前平台对应的官方产物。可通过 `YOLO.officialModels()` 查看当前平台可用的官方 ID。

### 2. 你自己的导出模型

传入本地路径或 Flutter 资源路径：

```dart
final yolo = YOLO(modelPath: 'assets/models/custom.tflite');
```

如果模型导出时带有元数据，插件会自动推断 `task`。如果没有，就显式传入 `task`。

### 3. 远程模型 URL

传入 `http` 或 `https` URL，插件会先下载到应用存储，再完成加载。

## 📥 把你自己的模型放进应用

对于自定义模型，应用侧配置应尽量简单：

- Android 原生资源：把 `.tflite` 放到 `android/app/src/main/assets`
- Android Flutter 资源：把 `.tflite` 放到 `assets/models/`
- iOS 工程资源：把 `.mlpackage` 或 `.mlmodel` 拖进 `ios/Runner.xcworkspace`
- iOS Flutter 资源：把 `.mlpackage.zip` 放到 `assets/models/`

然后把对应路径传给 `modelPath` 即可。

### iOS 导出注意事项

导出 CoreML 检测模型时必须使用 `nms=True`：

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="coreml", nms=True)
```

其他任务可以使用默认导出参数。

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
- **💡 想提功能建议？** [欢迎告诉我们](https://github.com/ultralytics/yolo-flutter-app/discussions)

## 💡 参与贡献

Ultralytics 的成长离不开社区协作，我们非常重视你的贡献。无论是修复 bug、增强功能还是改进文档，你的参与都非常重要。请查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)以了解如何参与，也欢迎通过[问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享你的反馈。衷心感谢所有贡献者的支持！🙏

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 许可证

Ultralytics 提供两种许可证，以适应不同需求：

- **AGPL-3.0 License**：适合学生、研究人员以及热衷于开源协作的开发者。这是一个经 [OSI 批准](https://opensource.org/license/agpl-v3)的许可证，鼓励知识共享和开放贡献。详情请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
- **Enterprise License**：适用于商业应用，允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品与服务中，而无需遵守 AGPL-3.0 的开源要求。如有商业使用需求，请了解[企业许可证](https://www.ultralytics.com/license)。

## 🔗 相关资源

### 原生 iOS 开发

如果你希望在 iOS 应用中直接使用 YOLO 模型与 Swift 集成，而不是通过 Flutter，可以查看我们的专用 iOS 仓库：

👉 **[Ultralytics YOLO iOS App](https://github.com/ultralytics/yolo-ios-app)** - 一个原生 iOS 应用，演示如何使用 Ultralytics YOLO 模型进行实时目标检测、分割、分类与姿态估计。

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
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
