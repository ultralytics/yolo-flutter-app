# `useGpu` Feature

## Overview

`useGpu` lets you choose whether inference should prefer GPU-backed execution when the platform supports it.

The default is still:

```dart
useGpu: true
```

That is usually the best choice for performance: on Android it engages the LiteRT 2.x GPU accelerator when the model is GPU-compatible (a fp16 non-end-to-end export) and otherwise falls back to CPU. Disabling GPU is useful when a device-specific GPU path is unstable.

## Basic Usage

```dart
final yolo = YOLO(
  modelPath: 'yolo26n',
  useGpu: false,
);
```

For `YOLOView`:

```dart
YOLOView(
  modelPath: 'yolo26n',
  useGpu: false,
  onResult: (results) {},
)
```

## When To Disable GPU

Disable GPU when:

- model initialization crashes on specific devices
- you are debugging load failures
- reliability matters more than peak throughput

## Platform Behavior

### Android

Android inference runs on LiteRT 2.x (Google's rebrand of TensorFlow Lite) via the `CompiledModel` API, with an automatic **GPU → CPU accelerator ladder**:

- `useGpu: true` requests the GPU. The plugin compiles the whole graph for the GPU when the model is compatible, and otherwise falls back to XNNPACK CPU.
- `useGpu: false` runs on XNNPACK CPU.

NNAPI is no longer used (it is deprecated and slower).

Actual GPU acceleration requires a **fp16, non-end-to-end** model:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, imgsz=640)
```

Here `half=True` produces fp16 weights the GPU can run, and `nms=False` keeps the raw (non-end2end) head while the plugin runs NMS on CPU sub-millisecond. int8 and end-to-end (`nms=True`) models use INT64 ops and int8 quantization the GPU cannot compile, so even with `useGpu: true` they silently fall back to CPU. They still load and run correctly.

On a Galaxy S26 (Adreno) a fp16 non-end2end YOLO26n detect model runs at roughly **7 ms/inference on the GPU** versus about **30 ms on CPU** (approximate, device-dependent).

### iOS

- `useGpu: true` uses broader Core ML compute units
- `useGpu: false` avoids the GPU path

## Recommendation

Start with the default GPU setting.

Only force CPU mode when a real device shows instability:

```dart
final yolo = YOLO(
  modelPath: 'assets/models/custom.tflite',
  task: YOLOTask.detect,
  useGpu: false,
);
```
