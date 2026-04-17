<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# iOS Native Layer

This directory contains the iOS implementation used by the Flutter plugin.

## What Lives Here

- Core ML model loading
- exported metadata inspection
- camera inference for `YOLOView`
- task-specific predictor implementations
- method-channel hooks used by the Dart API

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

YOLO("yolo26n.pt").export(format="coreml", nms=True)
```

Other tasks can use the default export settings.
