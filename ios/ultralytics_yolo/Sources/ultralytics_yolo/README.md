<a href="https://www.ultralytics.com"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# iOS Native Layer

This directory contains the **Flutter-specific** iOS layer of the plugin. The shared YOLO inference core (Core AI and Core ML model loading, metadata parsing, and the task-specific predictors) lives in the [`UltralyticsYOLO` Swift package](https://github.com/ultralytics/yolo-ios-app) and is consumed here via `import UltralyticsYOLO`, so the plugin and the native iOS app share a single source of truth.

These sources are compiled by **both** build systems from this same tree: the Swift Package manifest at `ios/ultralytics_yolo/Package.swift` (Swift Package Manager) and `ios/ultralytics_yolo.podspec` (CocoaPods). Both pin the same `UltralyticsYOLO` version range — bump them together.

## What Lives Here

- the platform view + method-channel bridge to the Dart API (`YOLOPlugin`, `YOLOInstanceManager`, `SwiftYOLOPlatformView`)
- the real-time camera/view layer (`YOLOView`, `VideoCapture`, `YOLOCamera`, `BoundingBoxView`)
- live-overlay drawing (`YOLOOverlayStyle`) and stream configuration (`YOLOStreamConfig`)

Core AI and Core ML loading and the per-task predictors come from the `UltralyticsYOLO` package — not this directory.

## Current Model Flow

The Flutter package resolves model source and task first, then the native iOS layer loads the resolved Core AI or Core ML model. Core AI (`.aimodel`) is the default on iOS 27 and later; Core ML (`.mlpackage`) remains the fallback for earlier iOS versions and for the iOS Simulator, which does not ship Core AI. The layer reports Core AI availability to Dart through the `isCoreAIAvailable` channel method.

That means the iOS native code is responsible for:

- reading model metadata (Core ML `creatorDefinedKey`, or the same keys from `metadata.json` inside an `.aimodel`)
- creating the correct predictor for the resolved task
- running inference and returning normalized results

It is not responsible for maintaining a separate example-only model catalog.

## Supported Inputs

The Flutter side can hand the iOS layer:

- an official model ID resolved into a cached Core AI asset (iOS 27+) or Core ML package
- a bundled `.mlpackage` or `.mlmodel`
- an extracted `.mlpackage` originating from a Flutter asset `.mlpackage.zip`
- an extracted `.aimodel` originating from a Flutter asset `.aimodel.zip` (requires `UltralyticsYOLO >= 8.9.15` on an iOS 27+ device)

## Export Reminder

With `ultralytics[export-coreml]>=8.4.142`, this detect export uses `nms=False` to select the NMS-free head, matching
the shipped Core ML assets and the Swift object detector:

```python
from ultralytics import YOLO

# Use 224 for classification and 640 for every other mobile task.
# Square [640, 640] works best when one model must run in both portrait and landscape.
# Ultralytics imgsz order is [height, width]; use [640, 384] for portrait-only or [384, 640] for landscape-only.
YOLO("yolo26n.pt").export(format="coreml", nms=False, imgsz=[640, 640])
```

Other tasks use the same square-orientation guidance and `nms=False`. Classification, semantic, and depth retain
their native outputs. Use `nms=None` for raw one-to-many outputs with Swift-side NMS, or `nms=True` for embedded NMS
on supported tasks. The `end2end` metadata field continues to describe the actual exported graph.

The Core AI export uses FP16 and the raw one-to-many head for detect, segment, pose, and OBB, which the `UltralyticsYOLO`
SDK decodes with Swift NMS; do not pass `nms=False`. It requires `ultralytics>=8.4.155`, macOS 26 or later on Apple
silicon, `torch>=2.8`, and `coreai-torch>=0.4.2`. Core AI has no NMS operator, so `nms=True` is not available:

```python
from ultralytics import YOLO

# Use 224 for classification and 640 for every other mobile task.
YOLO("yolo26n.pt").export(format="coreai", quantize=16, imgsz=640)
```

Zip the resulting `yolo26n.aimodel` directory (keeping it as the top-level entry) to ship it as a Flutter asset.
