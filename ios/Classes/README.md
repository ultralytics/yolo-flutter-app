<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# iOS Native Layer

This directory contains the **Flutter-specific** iOS layer of the plugin. The shared YOLO inference core (Core ML model loading, metadata parsing, and the task-specific predictors) lives in the [`UltralyticsYOLO` Swift package](https://github.com/ultralytics/yolo-ios-app) and is consumed here via `import UltralyticsYOLO`, so the plugin and the native iOS app share a single source of truth.

## What Lives Here

- the platform view + method-channel bridge to the Dart API (`YOLOPlugin`, `YOLOInstanceManager`, `SwiftYOLOPlatformView`)
- the real-time camera/view layer (`YOLOView`, `VideoCapture`, `YOLOCamera`, `BoundingBoxView`)
- live-overlay drawing (`YOLOOverlayStyle`) and stream configuration (`YOLOStreamConfig`)

Core ML loading, metadata inspection, and the per-task predictors come from the `UltralyticsYOLO` package — not this directory.

## Current Model Flow

The Flutter package resolves model source and task first, then the native iOS layer loads the resolved Core ML model.

That means the iOS native code is responsible for:

- reading Core ML metadata
- creating the correct predictor for the resolved task
- running inference and returning normalized results

It is not responsible for maintaining a separate example-only model catalog.

## Supported Inputs

The Flutter side can hand the iOS layer:

- an official model ID resolved into a cached Core ML package
- a bundled `.mlpackage` or `.mlmodel`
- an extracted `.mlpackage` originating from a Flutter asset `.mlpackage.zip`

## Export Reminder

For detection models on iOS, Core ML exports must use `nms=True`:

```python
from ultralytics import YOLO

# Square [640, 640] works best when one model must run in both portrait and landscape.
# Ultralytics imgsz order is [height, width]; use [640, 384] for portrait-only or [384, 640] for landscape-only.
YOLO("yolo26n.pt").export(format="coreml", nms=True, imgsz=[640, 640])
```

Other tasks can use the default export settings, with the same square-orientation guidance for `imgsz`.
