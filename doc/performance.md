---
title: Performance Guide
description: Practical performance tuning for YOLO Flutter on Android and iOS
path: /integrations/flutter/performance/
---

# Performance Guide

Good performance comes from choosing the right model, limiting unnecessary work, and testing on real devices.

## Depth input geometry (0.6.16)

Depth stretches the full image to the model input on Android and iOS, matching Ultralytics validation and calibration.
The full model-resolution map covers the original image; official models return 640×640 maps without cropped padding.
The iOS implementation requires UltralyticsYOLO 8.9.16 through both CocoaPods and Swift Package Manager.

Validated on Xiaomi 17 with the official YOLO26n depth model: CPU and GPU inference for 1280×720, 720×1280 and
1242×375 images, plus live front and rear cameras in all four portrait/landscape orientations. Each result returned a
640×640 map and a positive, finite depth range. iPhone 17 Pro / iOS 27 validation covered Core ML and Core AI depth,
classification and detection camera buffers; see the [iOS profiling record](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md).

## 📦 Choose the Right Model

Start with the smallest model that meets your accuracy needs.

```dart
final fast = YOLO(modelPath: 'yolo26n');
```

If you need a larger custom model, load that model directly:

```dart
final accurate = YOLO(
  modelPath: 'assets/models/custom-large.tflite',
  task: YOLOTask.detect,
);
```

Use a simple rule:

- default app flow: start with `yolo26n`
- custom production flow: benchmark your exact export on target devices

## 📊 Measured Backend Performance

### Xiaomi 17 LiteRT and QNN

**Hardware:** [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) with 12 GB LPDDR5X memory and Android 17 /
API 37. Its 3 nm [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5)
(SM8850) has an 8-core Qualcomm Oryon CPU (2 Prime cores up to 4.6 GHz and 6 Performance cores up to 3.62 GHz),
Adreno GPU, and Hexagon NPU.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | NPU<br><sup>QNN W8A16<br>(ms)</sup>    | NPU<br><sup>LiteRT FP32<br>(ms)</sup>  |
| ------------- | -------- | --------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| YOLO26n       | Detect   | 640                         | 36.9<br><sup>1.9 / 32.6 / 2.3</sup>    | 10.6<br><sup>1.8 / 6.2 / 2.6</sup>     | 10.2<br><sup>1.9 / 6.0 / 2.3</sup>     | **10.1**<br><sup>3.5 / 4.0 / 2.5</sup> |
| YOLO26n-seg   | Segment  | 640                         | 50.5<br><sup>1.9 / 42.9 / 5.8</sup>    | 18.0<br><sup>1.9 / 9.5 / 6.6</sup>     | **16.2**<br><sup>2.1 / 8.4 / 5.7</sup> | 19.2<br><sup>3.6 / 5.9 / 9.7</sup>     |
| YOLO26n-sem   | Semantic | 640                         | 45.7<br><sup>1.9 / 35.7 / 8.2</sup>    | 30.6<br><sup>1.9 / 20.5 / 8.3</sup>    | **10.8**<br><sup>2.0 / 6.2 / 2.6</sup> | 25.8<br><sup>3.6 / 11.0 / 11.3</sup>   |
| YOLO26n-depth | Depth    | 640                         | 66.8<br><sup>2.2 / 55.0 / 9.7</sup>    | 21.9<br><sup>2.1 / 10.1 / 9.8</sup>    | **19.3**<br><sup>2.2 / 7.4 / 9.7</sup> | 24.8<br><sup>4.3 / 6.0 / 14.5</sup>    |
| YOLO26n-cls   | Classify | 224                         | 3.6<br><sup>0.3 / 3.3 / 0.0</sup>      | 2.0<br><sup>0.4 / 1.4 / 0.2</sup>      | **0.8**<br><sup>0.3 / 0.5 / 0.0</sup>  | 1.5<br><sup>0.6 / 0.7 / 0.3</sup>      |
| YOLO26n-pose  | Pose     | 640                         | 39.9<br><sup>2.0 / 36.5 / 1.4</sup>    | 10.0<br><sup>1.8 / 6.7 / 1.4</sup>     | **9.7**<br><sup>1.9 / 5.9 / 1.8</sup>  | 11.5<br><sup>3.5 / 4.2 / 3.8</sup>     |
| YOLO26n-obb   | OBB      | 640                         | 36.3<br><sup>2.0 / 33.2 / 1.2</sup>    | 10.1<br><sup>2.6 / 5.9 / 1.6</sup>     | **8.1**<br><sup>1.9 / 5.1 / 1.0</sup>  | 8.2<br><sup>3.5 / 3.7 / 0.9</sup>      |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets (Ultralytics 8.4.188 exports). Backend order rotates between tasks in one sequential sweep.
Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK, every GPU row delegated the complete graph to LiteRT
OpenCL (`LITERT_CL`), and every QNN row used the Hexagon HTP backend. The LiteRT NPU column runs an FP32 export
(`format=litert` without `quantize`) through `litert-npu-runtime-qualcomm` 2.3.0, which compiles it on the device; the
first load includes that compile (about 0.3 s for `n`, up to 6 s for `x`). It was measured with a prototype NPU build of
the plugin, which this release does not include. LiteRT reports an NPU compile even when it silently runs the graph on
the GPU, so every NPU row was checked against the GPU time.

### Pixel 10 w8a32 LiteRT

**Hardware:** [Google Pixel 10](https://store.google.com/product/pixel_10_specs) with 12 GB memory and Android 17 /
API 37. Its 3 nm [Google Tensor G5](https://blog.google/products-and-platforms/devices/pixel/tensor-g5-pixel-10/) has
an 8-core CPU (1 Prime core up to 3.78 GHz, 5 Performance cores up to 3.05 GHz, and 2 Efficiency cores up to
2.25 GHz), PowerVR D-Series GPU, and Google TPU. Core clocks and the GPU driver name were read from the benchmark
device because Google does not publish them in the linked specifications.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup>  | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   |
| ------------- | -------- | --------------------------- | --------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 36.6<br><sup>1.5 / 33.5 / 1.6</sup>     | **36.5**<br><sup>3.5 / 28.4 / 4.6</sup>  |
| YOLO26n-seg   | Segment  | 640                         | 53.1<br><sup>1.5 / 45.8 / 5.7</sup>     | **51.5**<br><sup>1.6 / 40.1 / 9.9</sup>  |
| YOLO26n-sem   | Semantic | 640                         | **57.2**<br><sup>1.5 / 47.7 / 8.0</sup> | 75.9<br><sup>1.8 / 63.4 / 10.8</sup>     |
| YOLO26n-depth | Depth    | 640                         | 75.0<br><sup>1.6 / 65.1 / 8.3</sup>     | **50.1**<br><sup>1.7 / 33.5 / 14.9</sup> |
| YOLO26n-cls   | Classify | 224                         | **6.1**<br><sup>0.3 / 5.5 / 0.3</sup>   | 12.1<br><sup>0.9 / 11.0 / 0.1</sup>      |
| YOLO26n-pose  | Pose     | 640                         | **35.2**<br><sup>1.5 / 31.9 / 1.8</sup> | 38.2<br><sup>3.5 / 29.3 / 5.4</sup>      |
| YOLO26n-obb   | OBB      | 640                         | **32.7**<br><sup>1.5 / 29.3 / 1.9</sup> | 38.8<br><sup>3.8 / 29.6 / 5.4</sup>      |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets (Ultralytics 8.4.188 exports). CPU/GPU order alternates between tasks in one sequential sweep.
Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT
OpenCL (`LITERT_CL`).
The Tensor NPU has no YOLO LiteRT column: on-device compilation fails because Google's EdgeTPU service rejects apps
that are not on its allowlist, and LiteRT then falls back to the GPU. Google supports Tensor only through
[ahead-of-time compilation](https://developers.google.com/edge/litert/next/npu) with the
[Tensor SDK](https://developers.google.com/edge/litert/next/tensor-sdk) beta, which requires sign-up. A third-party
model precompiled for Tensor G5 ran fully on the NPU from the same non-allowlisted app, but we have not yet compiled or
measured YOLO this way.

### Galaxy S26 Exynos w8a32 LiteRT

**Hardware:** [Samsung Galaxy S26](https://www.samsung.com/uk/smartphones/galaxy-s26/) (SM-S942B) with 12 GB memory
and Android 17 / API 37. Its 2 nm [Exynos 2600](https://semiconductor.samsung.com/processor/mobile-processor/exynos-2600/)
has a 10-core Armv9.3 CPU (1 C1-Ultra core up to 3.8 GHz, 3 performance C1-Pro cores up to 3.26 GHz, and 6 efficiency
C1-Pro cores up to 2.76 GHz), Xclipse 960 GPU, and Samsung NPU.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>  | NPU<br><sup>LiteRT w8a32 AOT<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | -------------------------------------- | --------------------------------------- | ------------------------------------------ |
| YOLO26n       | Detect   | 640                         | 30.2<br><sup>1.3 / 27.3 / 1.6</sup>    | 15.9<br><sup>1.4 / 11.5 / 2.9</sup>     | **12.2**<br><sup>1.3 / 8.6 / 2.2</sup>     |
| YOLO26n-seg   | Segment  | 640                         | 45.6<br><sup>1.3 / 39.1 / 5.2</sup>    | 30.5<br><sup>1.3 / 23.6 / 5.6</sup>     | **17.0**<br><sup>1.4 / 9.7 / 5.8</sup>     |
| YOLO26n-sem   | Semantic | 640                         | 38.1<br><sup>1.3 / 28.6 / 8.2</sup>    | **26.1**<br><sup>1.3 / 16.8 / 7.9</sup> | 28.3<br><sup>1.7 / 15.1 / 11.5</sup>       |
| YOLO26n-depth | Depth    | 640                         | 63.5<br><sup>1.4 / 53.6 / 8.4</sup>    | 37.3<br><sup>1.4 / 24.6 / 11.3</sup>    | **28.4**<br><sup>2.1 / 12.9 / 13.4</sup>   |
| YOLO26n-cls   | Classify | 224                         | 3.8<br><sup>0.2 / 3.4 / 0.2</sup>      | **2.8**<br><sup>0.2 / 2.6 / 0.0</sup>   | 6.1<br><sup>0.5 / 5.2 / 0.4</sup>          |
| YOLO26n-pose  | Pose     | 640                         | 31.1<br><sup>1.3 / 28.4 / 1.4</sup>    | 18.4<br><sup>1.6 / 13.0 / 3.8</sup>     | **13.5**<br><sup>1.8 / 9.5 / 2.2</sup>     |
| YOLO26n-obb   | OBB      | 640                         | 27.9<br><sup>1.3 / 25.7 / 0.9</sup>    | 15.3<br><sup>1.9 / 11.4 / 2.1</sup>     | **11.4**<br><sup>1.7 / 8.3 / 1.4</sup>     |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets (Ultralytics 8.4.188 exports). CPU/GPU order alternates between tasks in one sequential sweep.
Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT
OpenCL (`LITERT_CL`).
The NPU column uses the `w8a32` assets compiled ahead of time for the Exynos 2600 (`E9965`) with LiteRT's Samsung
compiler plugin, run through the prototype NPU build. Samsung has no on-device compiler in LiteRT 2.3, and Google does
not yet publish a Samsung NPU runtime, so these rows used a dispatch library built from the LiteRT 2.3 sources.

### Xiaomi 17T Pro w8a32 LiteRT

**Hardware:** [Xiaomi 17T Pro](https://www.mi.com/global/product/xiaomi-17t-pro/specs/) (2602EPTC0G) with 12 GB
LPDDR5X memory and Android 16 / API 36. Its 3 nm
[MediaTek Dimensity 9500](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9500) (MT6993) has an
8-core Armv9.3 CPU (1 C1-Ultra core up to 4.21 GHz, 3 C1-Premium cores up to 3.5 GHz, and 4 C1-Pro cores up to
2.7 GHz), Mali-G1 Ultra MC12 GPU, and MediaTek NPU 990.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   | NPU<br><sup>LiteRT FP32<br>(ms)</sup>    |
| ------------- | -------- | --------------------------- | -------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 31.4<br><sup>1.5 / 28.0 / 1.9</sup>    | 25.6<br><sup>1.4 / 22.2 / 2.0</sup>      | **11.0**<br><sup>1.3 / 7.8 / 1.8</sup>   |
| YOLO26n-seg   | Segment  | 640                         | 77.8<br><sup>2.6 / 65.3 / 9.9</sup>    | 47.4<br><sup>2.6 / 33.6 / 11.3</sup>     | **36.5**<br><sup>2.6 / 23.4 / 10.4</sup> |
| YOLO26n-sem   | Semantic | 640                         | 81.3<br><sup>2.6 / 62.2 / 16.5</sup>   | **79.3**<br><sup>2.6 / 59.5 / 17.1</sup> | 87.6<br><sup>2.6 / 68.3 / 16.7</sup>     |
| YOLO26n-depth | Depth    | 640                         | 111.3<br><sup>3.4 / 91.1 / 16.8</sup>  | **55.3**<br><sup>2.8 / 36.3 / 16.1</sup> | —                                        |
| YOLO26n-cls   | Classify | 224                         | 6.3<br><sup>0.5 / 5.4 / 0.4</sup>      | 8.1<br><sup>0.5 / 7.5 / 0.1</sup>        | **3.2**<br><sup>0.5 / 2.2 / 0.5</sup>    |
| YOLO26n-pose  | Pose     | 640                         | 53.9<br><sup>2.6 / 49.5 / 1.8</sup>    | 32.8<br><sup>2.7 / 26.7 / 3.4</sup>      | **16.2**<br><sup>2.9 / 11.1 / 2.3</sup>  |
| YOLO26n-obb   | OBB      | 640                         | 49.5<br><sup>2.7 / 44.4 / 2.4</sup>    | **29.4**<br><sup>2.8 / 22.7 / 3.9</sup>  | —                                        |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets (Ultralytics 8.4.188 exports). CPU/GPU order alternates between tasks in one sequential sweep.
Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT
OpenCL (`LITERT_CL`).
The NPU column uses the same FP32 exports and prototype build as the Xiaomi 17, through
`litert-npu-runtime-mediatek` 2.3.0. Depth and OBB have no NPU result: the MediaTek NPU compiler rejects their
exponential and trigonometric ops, and LiteRT runs those graphs on the GPU.

### iPhone 17 Pro Core ML

**Hardware:** [Apple iPhone 17 Pro](https://support.apple.com/en-us/125090) with 12 GB memory and iOS 27.0.1. Its
A19 Pro has a 6-core CPU (2 Performance and 4 Efficiency cores), 6-core GPU with Neural Accelerators, and 16-core
Neural Engine.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>Core ML `.cpuOnly`<br>(ms)</sup> | CPU + ANE preferred<br><sup>Core ML `.cpuAndNeuralEngine`<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| YOLO26n       | Detect   | 640                         | 8.8<br><sup>0.0 / 8.8 / 0.0</sup>            | **3.7**<br><sup>0.0 / 3.7 / 0.0</sup>                                   |
| YOLO26n-seg   | Segment  | 640                         | 12.3<br><sup>0.0 / 11.7 / 0.6</sup>          | **4.9**<br><sup>0.0 / 4.3 / 0.6</sup>                                   |
| YOLO26n-sem   | Semantic | 640                         | 9.3<br><sup>0.0 / 9.0 / 0.3</sup>            | **4.4**<br><sup>0.0 / 4.1 / 0.3</sup>                                   |
| YOLO26n-depth | Depth    | 640                         | 26.0<br><sup>0.0 / 25.1 / 1.0</sup>          | **5.5**<br><sup>0.0 / 4.7 / 0.9</sup>                                   |
| YOLO26n-cls   | Classify | 224                         | 4.1<br><sup>0.0 / 4.1 / 0.0</sup>            | **2.0**<br><sup>0.0 / 1.9 / 0.0</sup>                                   |
| YOLO26n-pose  | Pose     | 640                         | 12.0<br><sup>0.0 / 12.0 / 0.0</sup>          | **4.0**<br><sup>0.0 / 4.0 / 0.0</sup>                                   |
| YOLO26n-obb   | OBB      | 640                         | 10.7<br><sup>0.0 / 10.7 / 0.0</sup>          | **3.9**<br><sup>0.0 / 3.9 / 0.0</sup>                                   |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v8.3.0` assets. CPU/accelerator order alternates between tasks in one sequential sweep. The CPU rows
request Core ML `.cpuOnly`; the CPU + ANE preferred rows request `.cpuAndNeuralEngine`, with final operation placement
controlled by Core ML. Native model descriptions confirmed 224 × 224 input for classification and 640 × 640 for
every other task. Core ML models re-exported with Ultralytics 8.4.188 measured within 0.3 ms of every row on the same
phone. On iOS 27.0.1, `.cpuOnly` inference of the `l` and `x` depth, pose, and OBB models crashes inside Core ML's CPU
kernels (`EXC_BAD_ACCESS` in `libBNNS`); the same models run with `.cpuAndNeuralEngine`, and every other task and size
runs on the CPU.

Reproduce the Android sweep with:

```bash
cd example && flutter test integration_test/qnn_benchmark_test.dart -d DEVICE_ID --dart-define=RUN_BENCH=true
```

Run the complete task/backend matrix in one invocation, especially on Xiaomi devices that require confirmation for
every app install. Do not invoke the benchmark separately per task.

Use a profile build on iOS so Swift postprocessing is optimized:

```bash
cd example && flutter drive --profile -d DEVICE_ID --driver=test_driver/integration_test.dart \
  --target=integration_test/qnn_benchmark_test.dart --dart-define=RUN_BENCH=true
```

Generic output labels the requested automatic paths `gpu-preferred` on Android and `ane-preferred` on iOS because
LiteRT and Core ML may fall back. Verify native device logs before recording either path as an actual GPU or Neural
Engine result; the Pixel and Galaxy tables above record GPU only because every model logged full GPU compilation.

### iPhone 17 Pro Core AI

Core ML remains the default, and Core AI (`.aimodel`) is an opt-in for iOS 27 and later devices (see the
[model guide](models.md)). Measured in the native iOS app on an iPhone 17 Pro with iOS 27.0, not through this Flutter
plugin, which runs the same `UltralyticsYOLO` predictors: end to end (preprocessing + inference + postprocessing) Core AI
is level with Core ML rather than faster, it is slower for semantic, depth, and CPU-only inference, its FP16 assets are
twice the download, and the iOS 27.0 runtime still returns wrong results for some graphs. The
[yolo-ios-app performance record](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md) owns the
full tables, findings, and known runtime issues.

## 🔭 Optimization Findings and Future Exploration

The current benchmark tables include results from the Android LiteRT optimization pass, including the
segment/semantic postprocess work from #549 and #550. What was tried, what worked, and what's left to explore:

**Shipped:**

- **Flat-output decode** for detect/pose/OBB: postprocess dropped from ~12 ms to 0.7-2.4 ms on every backend by
  reading the model output directly (no reshape copies, no JNI nested-array marshaling, confidence checked before
  box reads) — the same pattern as MediaPipe's decode and the iOS SDK's raw-pointer reads.
- **Channel-last (NHWC) QNN exports** (ultralytics#24790): removes the app's CPU transpose and the NPU's boundary
  transpose simultaneously; detect inference 7.4 → 5.8 ms.
- **In-graph ArgMax semantic QNN exports** (ultralytics#24790): a uint8 class map replaces ~80 MB of float logits;
  stable ~50 ms vs 123-1065 ms (erratic) before.
- **GPU program cache** (`GpuOptions.serializationDir`): model re-opens skip OpenCL compilation.
- **Cache-local segment decode** (#549): the class-argmax over the `[1, 4+nc+32, 8400]` head read each class
  ~33 KB apart — a cache miss per class per anchor. Reorganized to class-major (each class plane streamed
  sequentially into a per-anchor running best) and dropped a redundant early-reject scan. Segment postprocess
  ~9.6 → 7.3 ms (**−24%**) on the 80-class model, output bit-identical (matching detection counts before/after). The
  win scales with class count: an isolated 20-class decode dropped −57%, 1-class is within noise (so custom
  few-class models never regress).
- **Cache-local segment mask generation** (#549): the NCHW mask `Σ coeff·proto` gathered 32 proto values
  `planeSize` (~102 KB) apart per pixel. Reorganized to plane-major accumulation — stream each proto plane over the
  detection box into a reused per-pixel accumulator. A further **−1.2 ms / −13%** off segment postprocess
  (thermally matched at ~22 ms inference), bit-identical. NHWC (legacy onnx2tf) proto is already pixel-contiguous
  and was left unchanged.
- **High-resolution segment overlays** (#549): the visible combined mask now scales logits to model-input resolution
  before thresholding, matching the iOS SDK's sharp segment overlay path instead of upscaling a 160x160 RGBA mask.
  A naive Kotlin bilinear paint measured **15.5 ms postprocess** and was rejected; caching per-column interpolation
  reduced the [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) GPU path to **6.6 ms postprocess** for
  `yolo26n-seg`, trading a small amount of time for model-resolution masks while keeping the cache-local
  decode/mask-gen wins above. Raw mask payloads remain at the existing cropped proto resolution.
- **Preprocessing cleanup**: the Android path writes NCHW inputs directly for NCHW LiteRT/QNN models, eliminating the
  extra HWC→CHW float transpose, and clears only real letterbox padding instead of repainting the full target bitmap
  every frame. On the [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) w8a32 GPU sweep, 640px preprocessing is
  now ~1.8-1.9 ms across detect/segment/semantic/pose and ~2.6 ms for OBB, down from the old 6-9 ms range for
  single-image `predict()` benchmarks. The live camera HUD can still show ~5-10 ms `pre` in portrait because that path
  also rotates and letterboxes each CameraX frame before packing RGB.
- **Rejected live-camera preprocessing prototypes**: fusing CameraX RGBA `ImageProxy` → rotated/letterboxed CHW floats
  in Kotlin was slower than the current Bitmap/Canvas path on the [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/)
  `yolo26n-seg` live HUD. Direct `ByteBuffer` bilinear sampling measured **55.6 ms pre**, direct nearest-neighbor
  measured **61.1 ms pre**, and copying the RGBA plane into a reusable `ByteArray` before nearest-neighbor sampling
  measured **10.5 ms pre**. The restored Bitmap/Canvas path measured **~7.4 ms pre** in the same validation session.
  A 320x240 CameraX analysis stream spot-check was also neutral (**7.1 ms pre**) while reducing input detail before
  upscaling to the 640px model input, so lower camera resolution is not a default speed fix.
- **Postprocess overhead cleanup** (#549): classifier top-5 now uses fixed-size linear insertion instead of sorting
  every score; semantic keeps dense class maps as `IntArray` until Flutter serialization; segment NMS buckets
  detections once by class instead of filtering the full candidate list per class; native detect NMS stops once
  `numItemsThreshold` boxes are kept. Outputs and Flutter payloads are unchanged.
- **Postprocess follow-ups** (#550): the model-resolution mask skips its per-instance probability-mask loop entirely
  on the default path (raw masks not requested) instead of scanning each detection box for a no-op; the semantic
  class map is sent to Flutter as a compact `Int32List` rather than a boxed `List<Int>` copy, dropping a per-frame
  copy of the dense map (decoded identically on the Dart side). Outputs unchanged.
- **Native depth painting**: ported the iOS Accelerate depth-colorization pattern to the existing Android C++ decode
  library, replacing bounds-checked Kotlin min/max and color sweeps while preserving the metric `Float32List`, invalid
  pixel transparency, and near-to-far gradient. In its A/B test on the Xiaomi 17, postprocess fell from **50.2-54.1 ms**
  to **8.2-9.7 ms** across all five official 640px depth models (**-82.8% mean**), with exact native min/max equality
  against every positive finite output pixel. Current Xiaomi 17 depth timings with the `v0.7.0` assets and full
  640×640 maps:

  | Depth model | GPU preprocess | GPU inference | GPU post | GPU total   | CPU total |
  | ----------- | -------------- | ------------- | -------- | ----------- | --------- |
  | YOLO26n     | 2.1 ms         | 10.1 ms       | 9.8 ms   | **21.9 ms** | 66.8 ms   |
  | YOLO26s     | 3.4 ms         | 20.5 ms       | 13.7 ms  | **37.6 ms** | 104.1 ms  |
  | YOLO26m     | 3.6 ms         | 33.2 ms       | 14.2 ms  | **51.0 ms** | 155.5 ms  |
  | YOLO26l     | 2.9 ms         | 38.4 ms       | 12.7 ms  | **54.1 ms** | 186.5 ms  |
  | YOLO26x     | 2.8 ms         | 62.6 ms       | 12.0 ms  | **77.3 ms** | 287.7 ms  |

  Each row is the mean of 15 `predict()` calls after 3 warmups on `bus.jpg`. Every GPU row compiled fully on the
  LiteRT GPU, so Depth should use the GPU on this device.

- **Core ML Depth backend sweep**: all five official INT8 Core ML Depth models on the
  [iPhone 17 Pro hardware described above](#iphone-17-pro-core-ml) (iOS 27.0.1). CPU + ANE preferred values request
  `.cpuAndNeuralEngine`; CPU values request `.cpuOnly`. Each row is 15 `predict()` calls after 3 warmups on `bus.jpg`:

  | Depth model | CPU inference | CPU post | CPU total | CPU + ANE preferred inference | Preferred post | Preferred total |
  | ----------- | ------------- | -------- | --------- | ----------------------------- | -------------- | --------------- |
  | YOLO26n     | 25.1 ms       | 1.0 ms   | 26.0 ms   | 4.7 ms                        | 0.9 ms         | **5.5 ms**      |
  | YOLO26s     | 34.9 ms       | 1.0 ms   | 35.9 ms   | 6.2 ms                        | 0.8 ms         | **7.1 ms**      |
  | YOLO26m     | 58.4 ms       | 1.0 ms   | 59.4 ms   | 9.7 ms                        | 0.9 ms         | **10.5 ms**     |
  | YOLO26l     | crashes       | —        | —         | 11.3 ms                       | 0.9 ms         | **12.2 ms**     |
  | YOLO26x     | crashes       | —        | —         | 19.1 ms                       | 0.9 ms         | **20.1 ms**     |

  Vision performs scaling inside the request, so `preMs` is 0 and preprocessing is included in inference. The `l` and
  `x` CPU runs crash inside Core ML's CPU kernels on iOS 27.0.1 (see the iPhone table above). These are single-image
  burst measurements; the iOS app's sustained 720p camera path measures 16.5 ms/frame for YOLO26n Depth.

**Tested and intentionally NOT changed (don't re-litigate without new evidence):**

- `htp_performance_mode`: burst/sustained/high_performance are identical for our use — ORT votes the same max DCVS
  corner for burst and sustained; the default stays `burst`.
- `offload_graph_io_quantization=0` is the Ultralytics export default from 8.4.188: with the offload on,
  `onnxruntime-qnn` 2.4+ writes an EPContext I/O-name map that this plugin's ONNX Runtime cannot read. Keeping I/O
  quantization on the HTP measured no slower than the July QAIRT 2.46 assets.
- **Naive A8W8 quantization**: 33% faster inference but zero detections — a shared uint8 scale on the concatenated
  output destroys scores. W8A16 stays the export default.
- **fp16 GPU variants**: identical inference time to INT8 on the LiteRT GPU accelerator (it computes in fp16
  internally either way) — no reason to ship larger fp16 assets.
- **In-graph ArgMax for semantic TFLite**: the GPU delegate cannot compile `ARG_MAX` with int64 indices (what
  onnx2tf emits; its argmax-replacement flags no longer exist), so the whole graph falls back to CPU — 137 ms vs
  37.6 ms for GPU logits + the app's NHWC argmax. The class-map export stays QNN/Core ML-only.
- **int32 class maps**: uint8 quarters the NPU→CPU output transfer and every consumer reads it (Core ML promotes
  it to int32 in-spec); int32 indices are reserved for >256-class models. uint8 stays the class-map dtype.
- **Class-major argmax for detect / OBB decode**: the cache reorganization that won on segment (above) showed no
  benefit on detect — already native C++, where the JNI `GetFloatArrayElements` copy of the ~705k-float output
  dominates, not the argmax — and a slight regression on OBB, which has only ~15 classes so the per-frame buffer
  clear plus extra pass outweighs the small cache gain. Both reverted: the win needs _both_ many classes and the
  Kotlin path. Semantic argmax was already class-major (NCHW logits), so it was already optimal.

**Future exploration (in expected-value order):**

1. **A8W8 with mixed precision**: Qualcomm's "LiteMP" recipe (per-channel weights + ~10% of layers promoted to
   16-bit via `init_overrides`) recovers full accuracy at near-A8 speed — potential further ~30% NPU inference cut,
   needs an mAP validation loop in the exporter.
2. **Zero-copy I/O**: LiteRT supports AHardwareBuffer/GL/CL tensor interop and ORT QNN has a shared-memory
   allocator (`QnnHtpShared`), but both are C/C++-API-only today — revisit when the Kotlin/Java surfaces catch up,
   or via a small JNI shim. Attacks the remaining ~2 ms preprocess copy.
3. **RGBA camera input**: feeding 4-channel frames avoids the RGB repack (the GPU's native layout is 4-channel).
4. **Per-run HTP power votes** (`qnn.htp_perf_mode` run options): vote high during camera sessions, low after —
   battery/thermal optimization for sustained use rather than a latency win.

## 🎚️ Tune Thresholds

Higher confidence thresholds reduce post-processing work and visual noise:

```dart
await controller.setThresholds(
  confidenceThreshold: 0.6,
  iouThreshold: 0.3,
  numItemsThreshold: 10,
);
```

Lower thresholds increase recall, but also increase work and on-screen clutter.

## 📡 Limit Streaming Work

When using `YOLOView`, cap frame rate and disable data you do not need:

```dart
final config = YOLOStreamingConfig.throttled(
  maxFPS: 15,
  includeMasks: false,
  includeOriginalImage: false,
);
```

Good defaults:

- use `maxFPS` to cap UI churn
- disable masks unless you actually render them
- disable `includeOriginalImage` unless you consume full frame bytes

## 🖥️ GPU vs CPU

`useGpu` defaults to `true`, but CPU can be the better choice on unstable devices:

```dart
final yolo = YOLO(
  modelPath: 'yolo26n',
  useGpu: false,
);
```

On Android, inference runs on LiteRT 2.x with an automatic **GPU → CPU accelerator ladder**: with `useGpu: true` the plugin compiles the whole model for the GPU when it can; models the GPU cannot compile fall back to XNNPACK CPU. (iOS uses Core ML.)

The official YOLO26 Android assets (w8a32 LiteRT) compile on the LiteRT GPU path on supported devices, though GPU coverage still depends on the device driver and graph. For example, a Galaxy S26 compiles the official `yolo26n_w8a32.tflite` fully with the OpenCL delegate (`Replacing 414 out of 414 node(s) with delegate (LITERT_CL)`) and takes 15.9 ms per image (11.5 ms inference). Always confirm delegate placement with device logs instead of assuming a quantization format implies CPU or GPU.

non-end-to-end LiteRT exports are still useful for GPU benchmarking (the GPU delegate runs them in FP16):

```python
# Requires ultralytics>=8.4.142
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="litert", nms=None, imgsz=640)
# Classification models use imgsz=224.
```

Use CPU when:

- device-specific GPU paths are unstable
- startup reliability matters more than peak throughput
- you are debugging model-load failures

## 🧠 Manage Memory

Large models and multiple active instances increase memory use quickly.

Prefer:

- one active `YOLO` instance unless you truly need more
- explicit `dispose()` for single-image instances you no longer use
- one active `YOLOView` per camera screen

```dart
await yolo.dispose();
```

## 🔄 Multi-Instance Tips

Only enable multi-instance mode when you need parallel models:

```dart
final detector = YOLO(
  modelPath: 'yolo26n',
  useMultiInstance: true,
);
```

If you are comparing models, keep the set small and benchmark only what you need.

## 📱 Test on Real Devices

Do not treat simulator or emulator performance as representative.

Measure:

- model load time
- inference latency
- battery drain
- frame rate under real camera usage

## ✅ Quick Recommendations

- Start with `yolo26n`.
- Prefer metadata-carrying official models or exports.
- Cap streaming frame rate before trying to micro-optimize code.
- On Android, keep `useGpu: true` for the automatic LiteRT GPU -> CPU ladder, and verify actual delegate placement on target devices.
- Benchmark the actual export you plan to ship.

## Performance Record

This is the canonical record of the on-device profiling behind the Ultralytics YOLO Flutter example app and Android LiteRT camera pipeline. Each section captures the question, the empirical result, and the current conclusion. Use this as the baseline for future Flutter performance work, and append new profiling entries here as more devices, models, and runtime paths are validated.

> [!IMPORTANT]
> Host, emulator, and simulator numbers are only screening signals. Always confirm on the target device with the same exported asset, camera path, and runtime delegate that the app will ship.

### Test Setup

- **Device (ground truth):** Samsung Galaxy S26 (`SM S9420`), Android 16 / API 36.
- **Build:** Flutter example debug APK, package `com.ultralytics.yolo`.
- **Model:** the legacy `yolo26n_int8.tflite` detect asset (640x640, `quantize=8`) these camera experiments used; the official assets are now `w8a32` (see the tables above).
- **Runtime:** Android LiteRT 2.x `CompiledModel` with `useGpu: true`.
- **UI:** `YOLOShowcase` real-time camera view with the default task/size controls and threshold sliders.
- **Numbers:** EMA-smoothed app metrics after the camera and model are warm.

### Methodology

| Tool                                                           | What it measures                                    | Notes                                                                                                                           |
| -------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `YOLOView.onPerformanceMetrics` / on-screen FPS label          | Native inference stream rate and `processingTimeMs` | Best quick app-level signal. Includes the native predictor's per-frame processing time, not Flutter build/layout time.          |
| `adb logcat`                                                   | Runtime delegate selection and camera/device errors | Confirm LiteRT accelerator placement from logs such as `Replacing ... node(s) with delegate (LITERT_CL)` and `compiled on GPU`. |
| `adb exec-out screencap`                                       | Actual rendered UI and camera state                 | Catches startup, black-screen, overlay, and control-layout failures that tests and logs can miss.                               |
| `flutter analyze`, `flutter test`, `flutter build apk --debug` | Static, widget, and build health                    | Required before trusting a device-only performance result.                                                                      |

### What the App's "Inference Time" Measures

The on-screen `ms` value comes from native `YOLOResult.speed`, forwarded through `processingTimeMs`. It represents the native predictor's per-frame model processing path and result conversion. It excludes Flutter widget build time, UI composition, Android camera exposure time, and screenshot capture.

`FPS` is the actual inference stream rate emitted by the plugin. It is not simply `1000 / processingTimeMs` because the camera, analyzer backpressure, delegate scheduling, thermal state, and stream throttling can all cap the frame cadence.

### Experiment: Android Startup Path

**Q:** Why did the app get stuck on startup on the Galaxy S26?

**A:** There were two separate blockers.

1. Startup visibility was tied to first inference. If model resolution, camera binding, or layout did not reach first inference, the user could wait indefinitely without usable UI.
2. The Flutter route then failed its first layout because `CupertinoSlidingSegmentedControl` received a zero-width warm-up pass and computed a negative segment width.

**Shipped:** The example app now enters Flutter immediately. The size segmented control tolerates zero-width and narrow warm-up constraints before the real viewport metrics arrive, and model changes use a single modal loading overlay.

### Experiment: Segment and Semantic Post-Processing

**Q:** Beyond the GPU forward pass, where does per-frame time go for the segment and semantic tasks?

**A:** Both predictors reshaped the entire flat model output into jagged nested arrays every frame, then re-read it during post-processing (segment: the ~1M-float detection head plus the ~0.8M-float mask proto; semantic: the whole `[1, H, W, C]` map, re-read three levels deep during per-pixel argmax). The reshape ran before any useful work.

**Shipped:** Both predictors index the flat `run()` output in place (no per-frame reshape allocation or copy) and walk contiguous memory: the segment proto mask matmul iterates channels contiguously, and the semantic argmax reads one cache line per pixel for the NHWC layout. Output is bit-identical (verified `diff=0` against the old kernels on a standalone benchmark).

**Result:** A host-JVM microbenchmark of the changed loops showed ~1.8-2.1x on the semantic argmax and ~1.55x on the segment proto matmul plus detection indexing; the removed per-frame reshape copy is additional savings on top. On the Galaxy S26 both tasks rendered correct mask overlays without crashes; the tables above have their current timings.

**Conclusion:** Keep model outputs flat and index by stride, and confirm output equivalence with an explicit diff when refactoring post-processing. This brings Android to parity with the iOS SDK, whose Swift decode reads `MLMultiArray` through a raw `Float` pointer and bulk-copies mask rows ([yolo-ios-app](https://github.com/ultralytics/yolo-ios-app) PR #246).

### Experiment: Bundling Nano Models at Build Time

**Q:** Can the app ship with the nano models so common tasks work offline on first launch (and so segment/semantic can be profiled on a device with no network)?

**A:** Yes. `scripts/fetch_bundled_models.sh` downloads the six nano YOLO26 models into `example/assets/models/` at build time, wired into the Android Gradle `preBuild` and an iOS run-script build phase. The files stay gitignored and are never committed. `YOLOModelResolver` already checks `assets/models/` before a network download, so a bundled model means no first-run fetch. The download is best-effort (always exits `0`) so offline builds still succeed, and it is **skipped under CI** (`CI` / `GITHUB_ACTIONS`) so GitHub builds stay fast and off the network - CI exercises the runtime-download fallback instead.

**Shipped:** Local and release builds bundle `yolo26n` for all seven tasks; larger sizes still download on demand. This supersedes the earlier temporary "bundle for local validation" workaround.

**Conclusion:** Build-time bundling removes first-run download latency for the default models and makes on-device profiling reproducible without network access, while CI keeps using the runtime path.

### Experiment: Release Download vs. Local Validation

**Q:** Did first-use model download fail because of the Flutter resolver?

**A:** No. The attached S26 could not resolve `github.com`:

```text
ping: unknown host github.com
SocketException: Failed host lookup: 'github.com'
```

The app UI correctly showed the resolver failure. To validate the camera/inference path independent of device DNS, the same `yolo26n_int8.tflite` release asset was temporarily bundled into the ignored example asset directory for a local debug build. With that asset present, the app loaded the model, started CameraX, compiled LiteRT on GPU, and displayed live inference.

**Conclusion:** Autodownload requires normal device DNS/network access to GitHub release assets. The nano models are now bundled into the example app at build time for local/release builds (see "Bundling Nano Models at Build Time"); larger sizes remain release-hosted and downloaded/cached on first use.

### Experiment: Model Availability UI

**Q:** Why did the active `YOLO26n` chip still show a download arrow after the bundled/cache load succeeded?

**A:** The startup availability scan can complete before the model resolver copies a bundled official asset into app storage. The control state then stayed stale even though `YOLOView` had loaded the model.

**Shipped:** `YOLOShowcase._onModelLoaded` now marks the loaded size available. The selected chip clears its download glyph as soon as a model load succeeds, regardless of whether the file came from cache, bundled assets, or first-use download.

### Current Shipped Configuration

- Android official assets: YOLO26 w8a32 `.tflite`, `n/s/m/l/x`, detect/segment/semantic/depth/classify/pose/OBB, hosted on `ultralytics/yolo-flutter-app` release `v0.7.0`.
- Android export settings: `quantize=w8a32` (int8 weights, FP32 activations — dynamic-range, no calibration), `nms=False, end2end=False` (equivalent to `nms=None` with Ultralytics >=8.4.142); classify `imgsz=224`, all other tasks `imgsz=640`.
- Android runtime: LiteRT 2.x with GPU -> CPU accelerator fallback.
- Example UI: controls expose all seven tasks and all five model sizes; model changes use one modal loading overlay for downloads and native model reloads.
- Bundled models: local/release builds fetch the seven `yolo26n` nano models into `example/assets/models/` at build time (gitignored, not committed; skipped under CI), so nano tasks work offline with no first-run download; larger sizes download on demand.
- iOS runtime: with `useGpu: true` (the default) on iOS 16+, Core ML is pinned to `.cpuAndNeuralEngine` (Neural Engine + CPU), not `.all` - avoids GPU contention with the live preview/overlay compositing. iOS 15 and earlier use `.all`; `useGpu: false` pins to `.cpuOnly`.

### Open Levers

- **LiteRT NPU tier:** the measured LiteRT NPU path (FP32 exports compiled on the device on Qualcomm and MediaTek, AOT on
  Exynos) needs a reliable placement signal before it can ship: LiteRT 2.3's Kotlin API reports an NPU compile even when
  the graph silently runs on the GPU.
- **Camera preset tuning:** Measure CameraX target resolution and analyzer throughput separately from model processing.
- **Frame-rate policy:** Explore explicit camera frame duration / analyzer throttling for latency vs. thermal tradeoffs.
- **iOS post-processing port:** The iOS SDK further reduced segment/semantic decode with raw `MLMultiArray` pointer reads and `[Float]` mask coefficients (yolo-ios-app PR #246); mirror any remaining hot-loop `NSNumber` reads in the plugin's `ios/ultralytics_yolo/Sources/ultralytics_yolo/*.swift` if profiling shows them.
