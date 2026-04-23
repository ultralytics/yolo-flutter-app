# `useGpu` Feature

## Overview

`useGpu` lets you choose whether inference should prefer GPU-backed execution when the platform supports it.

The default is still:

```dart
useGpu: true
```

That is usually the best choice for performance, but disabling GPU is useful when a device-specific GPU path is unstable.

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

- `useGpu: true` enables the GPU delegate when available
- `useGpu: false` falls back to CPU or NNAPI paths

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
