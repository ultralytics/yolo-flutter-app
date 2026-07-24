# `useGpu` Feature

## Overview

`useGpu` lets you choose whether inference should prefer GPU-backed execution when the platform supports it.

The default is still:

```dart
useGpu: true
```

That is usually the best choice for performance: on Android it engages the LiteRT 2.x GPU accelerator when the model is GPU-compatible and otherwise falls back to CPU. Disabling GPU is useful when a device-specific GPU path is unstable.

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

- `useGpu: true` requests the GPU. The plugin compiles the whole model for the GPU when it can; models the GPU cannot compile fall back to XNNPACK CPU.
- `useGpu: false` runs on XNNPACK CPU.

NNAPI is no longer used (it is deprecated and slower).

Official int8 YOLO26 LiteRT assets can compile on the LiteRT GPU path on supported devices, but int8 GPU coverage depends on the device driver and graph. A non-end-to-end LiteRT export is still useful for GPU benchmarking (the GPU delegate runs the FP32 graph in FP16):

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="litert", nms=False, end2end=False, imgsz=640)
# Classification models use imgsz=224.
```

Here the FP32 model runs in FP16 on the LiteRT GPU delegate, `nms=False` leaves NMS to the plugin, and `end2end=False` keeps the YOLO26 raw head for the Android LiteRT conversion path. Keep `useGpu: true` and verify the actual delegate from LiteRT logs.

On a Galaxy S26, the official `yolo26n_int8.tflite` compiled fully with the LiteRT OpenCL GPU delegate (`Replacing 395 out of 395 node(s) with delegate (LITERT_CL)`) and ran around **15 FPS / 32 ms** in the live camera example.

### iOS

- `useGpu: true` enables hardware-accelerated Core ML inference: `.cpuAndNeuralEngine` on iOS 16+ (the GPU is deliberately excluded to avoid contention with preview/overlay compositing), `.all` on older iOS
- `useGpu: false` pins inference to `.cpuOnly` (no Neural Engine or GPU)

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
