---
title: Performance Guide
description: Practical performance tuning for YOLO Flutter on Android and iOS
path: /integrations/flutter/performance/
---

# Performance Guide

Good performance comes from choosing the right model, limiting unnecessary work, and testing on real devices.

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

End-to-end `predict()` speeds for the official YOLO26n models on a [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) phone powered by the
Qualcomm [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones) (SM8850), which pairs
a Qualcomm Oryon CPU with an Adreno GPU and the Hexagon NPU (HTP architecture v81). Each cell shows the **total time**
with the preprocess / inference / postprocess split beneath it.

| Model        | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>INT8 TFLite<br>(ms)</sup> | GPU Adreno<br><sup>INT8 TFLite<br>(ms)</sup> | NPU Hexagon<br><sup>QNN W8A16<br>(ms)</sup>      |
| ------------ | -------- | --------------------------- | ------------------------------------- | -------------------------------------------- | ------------------------------------------------ |
| YOLO26n      | Detect   | 640                         | 53.3<br><sup>3.6 / 47.4 / 2.4</sup>   | 17.2<br><sup>3.6 / 9.1 / 4.5</sup>           | **11.3**<br><sup>3.5 / 5.6 / 2.2</sup>           |
| YOLO26n-seg  | Segment  | 640                         | 76.0<br><sup>3.6 / 64.7 / 7.7</sup>   | 23.9<br><sup>3.6 / 11.8 / 8.6</sup>          | **21.3**<br><sup>3.5 / 7.9 / 10.0</sup>          |
| YOLO26n-sem  | Semantic | 1024                        | 66.6<br><sup>3.6 / 46.3 / 16.8</sup>  | **37.7**<br><sup>3.6 / 17.4 / 16.7</sup>     | 49.1<sup>1</sup><br><sup>8.8 / 20.8 / 19.5</sup> |
| YOLO26n-cls  | Classify | 224                         | 5.2<br><sup>0.8 / 4.0 / 0.5</sup>     | 4.5<br><sup>1.6 / 2.2 / 0.7</sup>            | **2.4**<br><sup>1.1 / 0.6 / 0.7</sup>            |
| YOLO26n-pose | Pose     | 640                         | 57.7<br><sup>3.5 / 52.4 / 1.8</sup>   | 15.2<br><sup>3.6 / 9.7 / 1.9</sup>           | **10.8**<br><sup>3.5 / 5.6 / 1.8</sup>           |
| YOLO26n-obb  | OBB      | 1024                        | 50.3<br><sup>3.6 / 45.4 / 1.3</sup>   | **13.9**<br><sup>3.8 / 8.2 / 1.8</sup>       | 21.0<br><sup>8.8 / 10.9 / 1.3</sup>              |

- **Speed** values are the full `predict()` time — preprocessing + inference + postprocessing, excluding annotation
  drawing — as the mean of 15 runs after 3 warmup runs on [bus.jpg](https://ultralytics.com/images/bus.jpg).
  <br>Reproduce the QNN validation and rows with `cd example && ENABLE_QNN=1 flutter test integration_test/model_benchmark_test.dart -d <device> --dart-define=RUN_BENCH=true --dart-define=RUN_QNN=true` (the example app's QNN runtime is opt-in). The historical CPU/GPU rows use retired `v0.3.5` INT8 assets and are not reproduced by the current-model harness.
- Benchmarks were run with the `ultralytics_yolo` Flutter plugin `0.6.8`; official Android LiteRT assets are hosted
  on the [yolo-flutter-app `v0.6.6` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6).
- **CPU** and **GPU** run the legacy official INT8 TFLite assets (the prior `v0.3.5` default; the current default is
  w8a32 LiteRT — see the migration table below), on LiteRT with `useGpu: false` / `true`. **NPU** runs the
  `*_v81_qnn.onnx` context binaries (INT8 weights, 16-bit activations) from the same release via the ONNX Runtime QNN
  Execution Provider.
- This mixed CPU/GPU/NPU snapshot is kept for QNN comparison. For the current Android LiteRT CPU/GPU numbers after
  the latest preprocessing and segment/semantic postprocess cleanup, use the migration and LiteRT tables below.
- <sup>1</sup> Semantic QNN uses the in-graph ArgMax class-map exports (ultralytics#24790), which replaced erratic
  123-1065 ms logits decoding with a stable ~49 ms; the GPU remains slightly faster for semantic at 1024px. The
  official `v0.3.5` QNN release assets ship in this channel-last class-map format, exported with ultralytics
  8.4.65.
- **These are single-image burst latencies**, not sustained camera frame times: one photo through `predict()` on a
  thermally rested device. Real-time camera operation runs higher — camera frames are letterboxed to the model
  input every frame and the silicon thermally settles under load (on an iPhone 17 Pro, YOLO26n detect measures
  ~3.8 ms burst and 11.3 ms/frame sustained with the shipped model-sized inference stream). On Android, portrait live-camera preprocessing still
  pays for CameraX frame rotation/letterbox every frame, so the in-app `pre` HUD can remain several milliseconds
  higher than the single-image rows even after the RGB packing cleanup. Watch the in-app pre/inference/post HUD line
  for your device's steady-state numbers, and benchmark your exact models on your target hardware.

### Reproducing QNN release assets

The original six v81 assets were exported with `ultralytics` `8.4.65`, verified in an x86-64 HTP simulator, and then
validated on the Xiaomi 17. The depth asset was exported from `ultralytics` `8.4.104` at commit `6729f6fe8`, quantized
to W8A16 with `depth8.yaml`, finalized for HTP v81 with QAIRT `2.46.0`, and validated on the same physical device.

The supported prebuilt-wheel route is Windows x64. Run the following in PowerShell; the Linux-style paths used for
the CPU quantization stage cannot finalize a QNN context on their own. Install CPU-only PyTorch before Ultralytics so
the export host does not download CUDA packages:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu `
  torch==2.13.0 torchvision==0.28.0
uv pip install --python .venv\Scripts\python.exe `
  "ultralytics @ git+https://github.com/ultralytics/ultralytics.git@6729f6fe84ec218e68c7506016c76781f5af8447"
uv pip install --python .venv\Scripts\python.exe `
  onnx==1.21.0 onnxslim==0.1.94 onnxruntime-qnn==2.2.0
.venv\Scripts\yolo.exe export model=yolo26n-depth.pt format=qnn name=81 imgsz=640 data=depth8.yaml batch=1
Rename-Item yolo26n-depth_qnn.onnx yolo26n-depth_v81_qnn.onnx
```

QNN context finalization requires a supported QNN export host or device; macOS cannot create the context binary. Do
not upload until the standalone asset loads through ONNX Runtime QNN on its target HTP architecture and returns a
valid depth map. The published asset can be downloaded and verified against the bytes validated on the device:

```bash
gh release download v0.3.5 --repo ultralytics/yolo-flutter-app \
  --pattern yolo26n-depth_v81_qnn.onnx --dir verify
printf '%s  %s\n' \
  7de30462abae8ab66ccad180eeced3176028456af04dd64be0e7b3d26d3e36bc \
  verify/yolo26n-depth_v81_qnn.onnx | shasum -a 256 -c -
```

### Format migration: legacy INT8 TFLite → w8a32 LiteRT

The official Android assets moved from the legacy onnx2tf **INT8 TFLite** format (NHWC, `v0.3.5`) to **w8a32 LiteRT**
(int8 weights + FP32 activations, NCHW, `v0.6.6`) — the smallest GPU-compatible litert format, which needs no
calibration data. The CPU/GPU columns above are the legacy INT8 baseline; the table below compares the four litert
quantization formats against it on the same [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/).

Same-device yolo26n detect, Adreno GPU, measured in one sustained sweep — so **inference** (the format-dependent stage)
is the comparable metric; preprocessing reflects the warmed-up thermal state of the back-to-back run:

| Android format                        | size (MB) | GPU inference (ms) | GPU-compiles |
| ------------------------------------- | --------- | ------------------ | ------------ |
| onnx2tf INT8 (legacy, `v0.3.5`)       | 2.9       | 8.6                | yes          |
| **w8a32 LiteRT (official, `v0.6.6`)** | **2.9**   | **8.4**            | **yes**      |
| INT8 LiteRT                           | 2.9       | 11.0               | yes          |
| FP32 LiteRT                           | 10.0      | 8.8                | yes          |
| w8a16 LiteRT                          | 3.0       | (CPU fallback)     | no — fails   |

- **w8a32 lands within noise of the retired onnx2tf INT8 model on the GPU** and is the smallest download, so the
  migration does not regress GPU inference. Unlike the old onnx2tf INT8 (which fell back to CPU on many devices), all
  three GPU-capable litert formats compile fully on the Adreno GPU here.
- INT8 LiteRT is the slowest litert format and still needs calibration; FP32 ties w8a32 on speed but is ~3.4× the
  download; **w8a16 fails to compile on the GPU delegate** and runs ~40× slower on CPU, so it is not used.

Per-task before/after on the Adreno GPU — the legacy onnx2tf **INT8 TFLite** assets vs the new **w8a32 LiteRT** assets,
both measured in the same run on the [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) at the shipped Android
`imgsz` (224 classify, 640 others). Each cell is the **total** with the preprocess / inference / postprocess split
beneath it. The runtime was `ultralytics_yolo` `0.6.10`; the `w8a32` assets come from the
[yolo-flutter-app `v0.6.6` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6).

| Model         | Task     | size<br><sup>(pixels)</sup> | Before<br><sup>onnx2tf INT8 TFLite<br>(ms)</sup> | After<br><sup>w8a32 LiteRT<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | ------------------------------------------------ | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 14.0<br><sup>1.8 / 8.1 / 4.2</sup>               | **13.5**<br><sup>1.9 / 8.1 / 3.5</sup>   |
| YOLO26n-seg   | Segment  | 640                         | 30.1<br><sup>1.9 / 20.3 / 8.0</sup>              | **28.6**<br><sup>1.8 / 20.1 / 6.7</sup>  |
| YOLO26n-sem   | Semantic | 640                         | **26.4**<br><sup>1.9 / 16.4 / 8.1</sup>          | 32.9<br><sup>1.8 / 23.0 / 8.2</sup>      |
| YOLO26n-depth | Depth    | 640                         | n/a                                              | **23.0**<br><sup>2.0 / 12.9 / 8.2</sup>  |
| YOLO26n-cls   | Classify | 224                         | 3.5<br><sup>0.9 / 2.2 / 0.4</sup>                | **3.2**<br><sup>1.0 / 2.2 / 0.1</sup>    |
| YOLO26n-pose  | Pose     | 640                         | 17.4<br><sup>2.4 / 9.9 / 5.1</sup>               | **14.0**<br><sup>1.9 / 9.3 / 2.8</sup>   |
| YOLO26n-obb   | OBB      | 640                         | 13.9<br><sup>3.0 / 8.3 / 2.7</sup>               | **13.0**<br><sup>2.9 / 7.9 / 2.3</sup>   |

w8a32 matches or beats the legacy onnx2tf INT8 format on five of the six tasks that have a legacy counterpart, and
adds the official Depth path. **Semantic remains the format regression** because the w8a32 NCHW logits cost more
inference time than the legacy NHWC logits, even after preprocessing cleanup. The legacy onnx2tf models run unchanged
on LiteRT 2.x alongside the new NCHW exports, confirming the runtime's layout-adaptive path.

### Xiaomi 17 current LiteRT and QNN

Current-model CPU, GPU, and NPU results from one Xiaomi 17 sweep. LiteRT uses the shipped `v0.6.6` w8a32 assets;
QNN uses the v81 W8A16 context binaries from `v0.3.5`, including the depth asset added on July 22, 2026. Semantic
and OBB retain their historical 1024px QNN exports, so their QNN timings are not same-resolution LiteRT comparisons.

| Model         | Task     | LiteRT / QNN<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   | NPU<br><sup>QNN W8A16<br>(ms)</sup>     |
| ------------- | -------- | ----------------------------------- | -------------------------------------- | ---------------------------------------- | --------------------------------------- |
| YOLO26n       | Detect   | 640 / 640                           | 52.5<br><sup>1.8 / 48.4 / 2.4</sup>    | 16.3<br><sup>2.2 / 9.0 / 5.0</sup>       | **10.8**<br><sup>1.8 / 6.7 / 2.3</sup>  |
| YOLO26n-seg   | Segment  | 640 / 640                           | 84.7<br><sup>2.2 / 75.7 / 6.7</sup>    | 30.6<br><sup>2.0 / 21.8 / 6.8</sup>      | **17.4**<br><sup>1.9 / 9.9 / 5.6</sup>  |
| YOLO26n-sem   | Semantic | 640 / 1024                          | 66.1<br><sup>2.0 / 55.2 / 9.0</sup>    | 38.2<br><sup>1.9 / 27.1 / 9.2</sup>      | **30.4**<br><sup>4.9 / 18.3 / 7.1</sup> |
| YOLO26n-depth | Depth    | 640 / 640                           | 135.4<br><sup>2.0 / 125.4 / 8.0</sup>  | **28.7**<br><sup>2.5 / 14.1 / 12.0</sup> | 38.1<br><sup>2.0 / 26.6 / 9.4</sup>     |
| YOLO26n-cls   | Classify | 224 / 224                           | 5.1<br><sup>0.4 / 4.6 / 0.0</sup>      | 3.0<br><sup>0.6 / 2.1 / 0.3</sup>        | **0.9**<br><sup>0.3 / 0.6 / 0.0</sup>   |
| YOLO26n-pose  | Pose     | 640 / 640                           | 76.9<br><sup>2.7 / 71.8 / 2.3</sup>    | 13.5<br><sup>2.3 / 9.1 / 2.0</sup>       | **11.2**<br><sup>2.0 / 7.1 / 2.2</sup>  |
| YOLO26n-obb   | OBB      | 640 / 1024                          | 55.4<br><sup>2.1 / 51.8 / 1.4</sup>    | **13.2**<br><sup>3.4 / 7.8 / 2.1</sup>   | 20.4<br><sup>4.9 / 14.0 / 1.6</sup>     |

These are means of 15 runs after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10`. Backend order rotates
between tasks, and this remains one sequential sweep rather than thermally isolated runs. Native logs confirmed every
LiteRT model on CPU and GPU and every QNN model on the Hexagon NPU, including depth input/output at 640 × 640.

### Pixel 10 w8a32 LiteRT

The same seven shipped YOLO26n models on a Google Pixel 10 (Tensor G5, Android 16 / API 36). Each cell is the total
`predict()` time with the preprocess / inference / postprocess split beneath it.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   |
| ------------- | -------- | --------------------------- | -------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 53.6<br><sup>1.5 / 50.5 / 1.6</sup>    | **43.9**<br><sup>3.9 / 36.2 / 3.8</sup>  |
| YOLO26n-seg   | Segment  | 640                         | 90.0<br><sup>2.0 / 80.8 / 7.2</sup>    | **48.5**<br><sup>2.4 / 35.4 / 10.7</sup> |
| YOLO26n-sem   | Semantic | 640                         | 71.0<br><sup>1.5 / 61.6 / 8.0</sup>    | **69.9**<br><sup>1.5 / 58.0 / 10.5</sup> |
| YOLO26n-depth | Depth    | 640                         | 124.9<br><sup>1.5 / 117.2 / 6.3</sup>  | **50.0**<br><sup>1.8 / 36.5 / 11.7</sup> |
| YOLO26n-cls   | Classify | 224                         | **4.0**<br><sup>0.3 / 3.4 / 0.3</sup>  | 17.6<br><sup>0.9 / 16.6 / 0.1</sup>      |
| YOLO26n-pose  | Pose     | 640                         | 59.5<br><sup>1.5 / 56.8 / 1.2</sup>    | **45.9**<br><sup>3.9 / 38.4 / 3.7</sup>  |
| YOLO26n-obb   | OBB      | 640                         | 52.1<br><sup>1.5 / 49.1 / 1.6</sup>    | **45.6**<br><sup>4.1 / 38.8 / 2.7</sup>  |

These are means of 15 runs after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10` and the official `v0.6.6`
assets. CPU/GPU order alternates between task rows to avoid systematically measuring one backend second; this remains
a single sequential sweep rather than thermally isolated runs. Device logs confirmed that every model compiled fully
on the requested LiteRT CPU and GPU backends.

### Galaxy S26 Exynos w8a32 LiteRT

The same seven shipped YOLO26n models on a Samsung Galaxy S26 (SM-S942B, Exynos 2600 with Xclipse 960 GPU,
Android 16 / API 36).

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   |
| ------------- | -------- | --------------------------- | -------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 36.7<br><sup>1.2 / 33.9 / 1.7</sup>    | **17.3**<br><sup>2.5 / 12.3 / 2.5</sup>  |
| YOLO26n-seg   | Segment  | 640                         | 55.4<br><sup>1.2 / 48.9 / 5.3</sup>    | **30.8**<br><sup>1.2 / 23.3 / 6.2</sup>  |
| YOLO26n-sem   | Semantic | 640                         | 48.7<br><sup>1.2 / 39.4 / 8.0</sup>    | **36.2**<br><sup>1.2 / 27.2 / 7.8</sup>  |
| YOLO26n-depth | Depth    | 640                         | 93.4<br><sup>1.2 / 85.2 / 7.0</sup>    | **33.7**<br><sup>1.3 / 22.4 / 10.0</sup> |
| YOLO26n-cls   | Classify | 224                         | **2.8**<br><sup>0.2 / 2.4 / 0.3</sup>  | 3.2<br><sup>0.2 / 3.0 / 0.0</sup>        |
| YOLO26n-pose  | Pose     | 640                         | 42.9<br><sup>1.3 / 40.5 / 1.1</sup>    | **19.3**<br><sup>1.4 / 14.7 / 3.2</sup>  |
| YOLO26n-obb   | OBB      | 640                         | 37.3<br><sup>1.3 / 34.8 / 1.2</sup>    | **17.8**<br><sup>1.7 / 14.4 / 1.7</sup>  |

These are means of 15 runs after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10` and the official `v0.6.6`
assets. CPU/GPU order alternates between task rows, and the results are one sequential sweep rather than thermally
isolated runs. Device logs confirmed that every model compiled fully on the requested LiteRT CPU and GPU backends.

### iPhone 17 Pro Core ML

The same sweep on an Apple iPhone 17 Pro (iOS 26.5.2) using the shipped Core ML models. The preferred path
requests `.cpuAndNeuralEngine`; Core ML controls final operation placement.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>Core ML<br>(ms)</sup>    | CPU + ANE preferred<br><sup>Core ML<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | ------------------------------------ | ------------------------------------------------- |
| YOLO26n       | Detect   | 640                         | 9.1<br><sup>0.0 / 9.0 / 0.0</sup>    | **3.6**<br><sup>0.0 / 3.6 / 0.0</sup>             |
| YOLO26n-seg   | Segment  | 640                         | 22.2<br><sup>0.0 / 12.3 / 9.9</sup>  | **13.4**<br><sup>0.0 / 4.2 / 9.2</sup>            |
| YOLO26n-sem   | Semantic | 640                         | 24.0<br><sup>0.0 / 22.1 / 2.0</sup>  | **14.9**<br><sup>0.0 / 12.9 / 1.9</sup>           |
| YOLO26n-depth | Depth    | 640                         | 49.3<br><sup>0.0 / 24.5 / 24.8</sup> | **28.9**<br><sup>0.0 / 4.8 / 24.1</sup>           |
| YOLO26n-cls   | Classify | 224                         | 2.3<br><sup>0.0 / 2.3 / 0.1</sup>    | **2.0**<br><sup>0.0 / 2.0 / 0.0</sup>             |
| YOLO26n-pose  | Pose     | 640                         | 12.2<br><sup>0.0 / 12.2 / 0.1</sup>  | **3.9**<br><sup>0.0 / 3.9 / 0.1</sup>             |
| YOLO26n-obb   | OBB      | 640                         | 22.8<br><sup>0.0 / 22.8 / 0.0</sup>  | **7.4**<br><sup>0.0 / 7.4 / 0.0</sup>             |

These are means of 15 runs after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10`. CPU/accelerator order
alternates between task rows, and the results are one sequential sweep rather than thermally isolated runs. The
shipped Android and iOS depth artifacts both declare fixed 640 × 640 inputs.

Reproduce either seven-model sweep on Android or iOS with:

```bash
cd example && flutter test integration_test/model_benchmark_test.dart -d DEVICE_ID --dart-define=RUN_BENCH=true
```

Generic output labels the requested automatic paths `gpu-preferred` on Android and `ane-preferred` on iOS because
LiteRT and Core ML may fall back. Verify native device logs before recording either path as an actual GPU or Neural
Engine result; the Pixel and Galaxy tables above record GPU only because every model logged full GPU compilation.

## 🔭 Optimization Findings and Future Exploration

The migration table above reflects the current Android LiteRT optimization pass on the Snapdragon 8 Elite Gen 5,
including the segment/semantic postprocess work from #549 and #550. What was tried, what worked, and what's left on
the table:

**Shipped (in the table):**

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
  reduced the [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) GPU path to **6.7-8.0 ms postprocess** for
  `yolo26n-seg`, trading a small amount of time for model-resolution masks while keeping the cache-local
  decode/mask-gen wins above. Raw mask payloads remain at the existing cropped proto resolution.
- **Preprocessing cleanup**: the Android path writes NCHW inputs directly for NCHW LiteRT/QNN models, eliminating the
  extra HWC→CHW float transpose, and clears only real letterbox padding instead of repainting the full target bitmap
  every frame. On the [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) w8a32 GPU sweep, 640px preprocessing is
  now ~1.8-1.9 ms across detect/segment/semantic/pose and ~2.9 ms for OBB, down from the old 6-9 ms range for
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
  pixel transparency, and near-to-far gradient. On the Xiaomi 17, postprocess fell from **50.2-54.1 ms** to
  **8.2-9.7 ms** across all five official 640px depth models (**-82.8% mean**). The grouped physical-device test
  verified exact native min/max equality against every positive finite output pixel and confirmed that all five models
  compiled fully on the LiteRT GPU:

  | Depth model | Preprocess | GPU inference | Post before | Post after | Optimized total |
  | ----------- | ---------- | ------------- | ----------- | ---------- | --------------- |
  | YOLO26n     | 2.0 ms     | 12.9 ms       | 54.1 ms     | **8.2 ms** | **23.0 ms**     |
  | YOLO26s     | 2.2 ms     | 20.5 ms       | 50.2 ms     | **9.1 ms** | **31.8 ms**     |
  | YOLO26m     | 2.8 ms     | 35.2 ms       | 51.0 ms     | **9.7 ms** | **47.7 ms**     |
  | YOLO26l     | 1.9 ms     | 45.2 ms       | 51.5 ms     | **8.8 ms** | **56.0 ms**     |
  | YOLO26x     | 2.1 ms     | 79.6 ms       | 51.9 ms     | **8.6 ms** | **90.3 ms**     |

  Each row is the mean of 15 `predict()` calls after 3 warmups on `bus.jpg`; maps were 480x640 after letterbox crop.
  Inference varied with GPU/thermal state between the before/after sweeps, so the attributable A/B result is the
  instrumented postprocess column rather than the total-time difference.

  The matching LiteRT XNNPACK CPU sweep confirms that Depth should use the GPU on this device:

  | Depth model | CPU preprocess | CPU inference | CPU post | CPU total  |
  | ----------- | -------------- | ------------- | -------- | ---------- |
  | YOLO26n     | 5.07 ms        | 300.91 ms     | 19.15 ms | 325.13 ms  |
  | YOLO26s     | 4.60 ms        | 413.98 ms     | 19.17 ms | 437.75 ms  |
  | YOLO26m     | 4.75 ms        | 725.81 ms     | 19.21 ms | 749.77 ms  |
  | YOLO26l     | 4.68 ms        | 894.44 ms     | 19.19 ms | 918.31 ms  |
  | YOLO26x     | 4.65 ms        | 1481.00 ms    | 19.82 ms | 1505.47 ms |

- **Core ML Depth backend sweep**: the same grouped physical-device harness ran all five official INT8 Core ML Depth
  models on an iPhone 17 Pro (A19, iOS 26.5.2). Neural Engine values use `.cpuAndNeuralEngine`; CPU values use
  `.cpuOnly`. Each row is 15 `predict()` calls after 3 warmups on `bus.jpg`, with a 480x640 typed metric map:

  | Depth model | CPU inference | CPU post | CPU total | Neural Engine inference | NE post | NE total     |
  | ----------- | ------------- | -------- | --------- | ----------------------- | ------- | ------------ |
  | YOLO26n     | 23.90 ms      | 0.85 ms  | 24.75 ms  | 4.67 ms                 | 0.87 ms | **5.54 ms**  |
  | YOLO26s     | 33.67 ms      | 0.90 ms  | 34.57 ms  | 6.15 ms                 | 0.85 ms | **7.01 ms**  |
  | YOLO26m     | 55.27 ms      | 0.93 ms  | 56.21 ms  | 9.64 ms                 | 0.89 ms | **10.54 ms** |
  | YOLO26l     | 67.32 ms      | 0.93 ms  | 68.25 ms  | 10.87 ms                | 0.94 ms | **11.80 ms** |
  | YOLO26x     | 116.77 ms     | 0.94 ms  | 117.71 ms | 19.38 ms                | 0.93 ms | **20.30 ms** |

  Vision performs scaling inside the request, so `preMs` is 0 and preprocessing is included in inference. These are
  single-image burst measurements; the iOS app's sustained 720p camera path measures 16.5 ms/frame for YOLO26n Depth.

**Tested and intentionally NOT changed (don't re-litigate without new evidence):**

- `htp_performance_mode`: burst/sustained/high_performance are identical for our use — ORT votes the same max DCVS
  corner for burst and sustained; the default stays `burst`.
- `offload_graph_io_quantization=0`: no measurable effect on normal-sized outputs.
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
   or via a small JNI shim. Attacks the remaining ~3.5 ms preprocess copy.
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

The official YOLO26 Android assets (w8a32 LiteRT) compile on the LiteRT GPU path on supported devices, though GPU coverage still depends on the device driver and graph. For example, a Galaxy S26 compiled the legacy `yolo26n_int8.tflite` fully with the OpenCL delegate (`Replacing 395 out of 395 node(s) with delegate (LITERT_CL)`) and ran at about **15 FPS / 32 ms** in the live camera example app. Always confirm delegate placement with device logs instead of assuming a quantization format implies CPU or GPU.

non-end-to-end LiteRT exports are still useful for GPU benchmarking (the GPU delegate runs them in FP16):

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="litert", nms=False, end2end=False, imgsz=640)
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
- **Model:** `yolo26n_int8.tflite`, official YOLO26 detect asset, 640x640 input, `quantize=8`, `nms=False`, `end2end=False`.
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

### Experiment: Official INT8 TFLite on GPU

**Q:** Can the official Android `yolo26n_int8.tflite` asset run on GPU?

**A:** Yes on the Galaxy S26. LiteRT loaded the GPU accelerator and compiled the full graph with the OpenCL delegate:

```text
RegisterAccelerator: name=LiteRT GPU
Replacing 395 out of 395 node(s) with delegate (LITERT_CL)
ObjectDetector: LiteRT compiled on GPU; inputDims=[1, 640, 640, 3] outputDims=[[1, 84, 8400]]
```

Observed app-level result:

| Device     | Model                 | Runtime                  | On-screen FPS | On-screen ms |
| ---------- | --------------------- | ------------------------ | ------------- | ------------ |
| Galaxy S26 | `yolo26n_int8.tflite` | LiteRT GPU (`LITERT_CL`) | ~15.2 FPS     | ~32 ms       |

**Conclusion:** Keep official Android assets as int8 TFLite for download size and confirm GPU placement per device from LiteRT logs. Do not assume int8 means CPU-only on current LiteRT, and do not assume all int8 graphs receive the same GPU coverage: driver support is device-dependent, and graphs the GPU cannot compile fall back to CPU.

### Experiment: Segment and Semantic Post-Processing

**Q:** Beyond the GPU forward pass, where does per-frame time go for the segment and semantic tasks?

**A:** Both predictors reshaped the entire flat model output into jagged nested arrays every frame, then re-read it during post-processing (segment: the ~1M-float detection head plus the ~0.8M-float mask proto; semantic: the whole `[1, H, W, C]` map, re-read three levels deep during per-pixel argmax). The reshape ran before any useful work.

**Shipped:** Both predictors index the flat `run()` output in place (no per-frame reshape allocation or copy) and walk contiguous memory: the segment proto mask matmul iterates channels contiguously, and the semantic argmax reads one cache line per pixel for the NHWC layout. Output is bit-identical (verified `diff=0` against the old kernels on a standalone benchmark).

**Result:** A host-JVM microbenchmark of the changed loops showed ~1.8-2.1x on the semantic argmax and ~1.55x on the segment proto matmul plus detection indexing; the removed per-frame reshape copy is additional savings on top. On the Galaxy S26 (GPU, bundled nano models) segment `yolo26n-seg` ran ~25 FPS / ~30 ms and semantic `yolo26n-sem` ~20-22 FPS / ~43 ms, both with correct mask overlays and no crashes.

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

- Android official assets: YOLO26 w8a32 `.tflite`, `n/s/m/l/x`, detect/segment/semantic/depth/classify/pose/OBB, hosted on `ultralytics/yolo-flutter-app` release `v0.6.6`.
- Android export settings: `quantize=w8a32` (int8 weights, FP32 activations — dynamic-range, no calibration), `nms=False`, `end2end=False`; classify `imgsz=224`, all other tasks `imgsz=640`.
- Android runtime: LiteRT 2.x with GPU -> CPU accelerator fallback.
- Example UI: controls expose all seven tasks and all five model sizes; model changes use one modal loading overlay for downloads and native model reloads.
- Bundled models: local/release builds fetch the seven `yolo26n` nano models into `example/assets/models/` at build time (gitignored, not committed; skipped under CI), so nano tasks work offline with no first-run download; larger sizes download on demand.
- iOS runtime: with `useGpu: true` (the default) on iOS 16+, Core ML is pinned to `.cpuAndNeuralEngine` (Neural Engine + CPU), not `.all` - avoids GPU contention with the live preview/overlay compositing. iOS 15 and earlier use `.all`; `useGpu: false` pins to `.cpuOnly`.

### Open Levers

- **Release-device matrix:** Repeat the S26 test on Pixel, older Snapdragon, Tensor, and Exynos devices to record GPU delegate coverage and fallback behavior.
- **CPU baseline:** Force `useGpu: false` on the same official int8 assets and record FPS/ms/power deltas.
- **FP16 benchmark:** Compare non-end-to-end fp16 TFLite exports against the official int8 assets on devices where both compile on GPU.
- **Camera preset tuning:** Measure CameraX target resolution and analyzer throughput separately from model processing.
- **Frame-rate policy:** Explore explicit camera frame duration / analyzer throttling for latency vs. thermal tradeoffs.
- **iOS post-processing port:** The iOS SDK further reduced segment/semantic decode with raw `MLMultiArray` pointer reads and `[Float]` mask coefficients (yolo-ios-app PR #246); mirror any remaining hot-loop `NSNumber` reads in the plugin's `ios/ultralytics_yolo/Sources/ultralytics_yolo/*.swift` if profiling shows them.
