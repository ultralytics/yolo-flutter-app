<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 YOLO Flutter - Ultralytics 官方插件

[![Ultralytics Actions](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/format.yml)
[![.github/workflows/ci.yml](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/yolo-flutter-app/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ultralytics/yolo-flutter-app/graph/badge.svg?token=8lpScd9O2a)](https://codecov.io/gh/ultralytics/yolo-flutter-app)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/ultralytics/)

欢迎使用 Ultralytics YOLO Flutter 插件！你可以将先进的 [Ultralytics YOLO](https://docs.ultralytics.com/) [计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型无缝集成到 Flutter 移动应用中。该插件发布于 https://pub.dev/packages/ultralytics_yolo ，同时支持 Android 和 iOS 平台，并提供[目标检测](https://docs.ultralytics.com/tasks/detect/)、[图像分类](https://docs.ultralytics.com/tasks/classify/)、[实例分割](https://docs.ultralytics.com/tasks/segment/)、[姿态估计](https://docs.ultralytics.com/tasks/pose/)和[旋转框检测](https://docs.ultralytics.com/tasks/obb/) API。

**✨ 为什么选择 YOLO Flutter？**

| 功能     | Android | iOS |
| -------- | ------- | --- |
| 检测     | ✅      | ✅  |
| 分类     | ✅      | ✅  |
| 分割     | ✅      | ✅  |
| 姿态估计 | ✅      | ✅  |
| OBB 检测 | ✅      | ✅  |

- **Ultralytics 官方插件** - 直接来自 YOLO 创建团队
- **实时性能** - 在现代移动设备上可达 30 FPS
- **5 类 AI 任务** - 检测、分割、分类、姿态、OBB
- **跨平台** - 单一代码库同时支持 iOS 与 Android
- **可用于生产环境** - 内置性能控制与优化能力
- **动态模型加载** - 无需重启相机即可切换模型
- **帧捕获** - 可捕获带检测叠加层的画面用于分享或保存

## ⚡ 快速开始（2 分钟）

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

// 添加这个 widget 即可开始目标检测
YOLOView(
  modelPath: 'yolo11n',
  task: YOLOTask.detect,
  onResult: (results) {
    print('Found ${results.length} objects!');
    for (final result in results) {
      print('${result.className}: ${result.confidence}');
    }
  },
)
```

**[▶️ 体验在线示例](./example)** | **[📖 完整安装指南](doc/install.md)**

## 🎯 你可以构建什么

| 任务         | 描述             | 使用场景           | 性能      |
| ------------ | ---------------- | ------------------ | --------- |
| **检测**     | 发现目标及其位置 | 安防、库存、购物   | 25-30 FPS |
| **分割**     | 像素级目标掩码   | 图片编辑           | 15-25 FPS |
| **分类**     | 识别图像类别     | 内容审核、自动标注 | 30+ FPS   |
| **姿态估计** | 人体姿态与关键点 | 健身应用、动作捕捉 | 20-30 FPS |
| **OBB 检测** | 旋转边界框检测   | 航拍图像           | 20-25 FPS |

**[📱 查看示例 →](doc/usage.md)** | **[⚡ 性能指南 →](doc/performance.md)**

## 🚀 安装

### 1. 添加到 `pubspec.yaml`

```yaml
dependencies:
  ultralytics_yolo: ^0.2.0
```

### 2. 安装依赖

```bash
flutter pub get
```

### 3. 添加模型

你可以通过以下任一方式获取模型：

1. 从本仓库的 [release assets](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.2.0) 下载
2. 从 [Ultralytics HUB](https://www.ultralytics.com/hub) 获取
3. 通过 [Ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 导出（[CoreML](https://docs.ultralytics.com/integrations/coreml/)/[TFLite](https://docs.ultralytics.com/integrations/tflite/)）

对于 YOLO26，使用相同步骤并从 `v0.2.0` release 中获取 `yolo26*` 产物即可，例如 `yolo26n.tflite` 或 `yolo26n.mlpackage`。

### 为 iOS 导出模型

```python
# 检测任务必须使用 nms=True
YOLO("yolo11n.pt").export(format="coreml", nms=True)

# 其他任务使用 nms=False（默认值）
YOLO("yolo11n-seg.pt").export(format="coreml")
```

**[📥 下载模型](doc/models.md)**

请按以下方式将模型随应用一同打包。

对于 iOS：将 `mlpackage` 或 `mlmodel` 直接拖入 **ios/Runner.xcworkspace**，并将 target 设置为 `Runner`。

对于 Android：创建 **android/app/src/main/assets** 文件夹，并将 `tflite` 文件放入其中。

### 4. 平台相关配置

**[🔧 安装配置指南](doc/install.md)**

## 🏆 受到开发者信赖

- ✅ **Ultralytics 官方插件** - 由 YOLO 创建团队维护
- ✅ **已在生产环境验证** - 已被多款应用实际使用
- ✅ **持续活跃开发** - 定期更新与新增功能
- ✅ **社区驱动** - 开源且支持响应及时

**性能**：现代设备上最高可达 30 FPS | **模型大小**：已优化至 6MB 起 | **平台支持**：iOS 13.0+ 与 Android API 21+

## 📚 文档

| 指南                                   | 说明                 | 适合对象 |
| -------------------------------------- | -------------------- | -------- |
| **[安装指南](doc/install.md)**         | 安装、配置、环境要求 | 新用户   |
| **[快速开始](doc/quickstart.md)**      | 2 分钟上手指南       | 新用户   |
| **[使用指南](doc/usage.md)**           | 常见用例与代码示例   | 所有用户 |
| **[性能优化](doc/performance.md)**     | 推理控制与调优       | 生产应用 |
| **[API 参考](doc/api.md)**             | 完整技术参考         | 开发者   |
| **[故障排查](doc/troubleshooting.md)** | 常见问题与解决方案   | 所有用户 |

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
