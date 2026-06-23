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

End-to-end `predict()` speeds for the official YOLO26n models on a [Xiaomi 17](https://www.mi.com/) phone powered by the
Qualcomm [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones) (SM8850), which pairs
a Qualcomm Oryon CPU with an Adreno GPU and the Hexagon NPU (HTP architecture v81). Each cell shows the **total time**
with the preprocess / inference / postprocess split beneath it.

| Model        | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>INT8 TFLite<br>(ms)</sup> | GPU Adreno<br><sup>INT8 TFLite<br>(ms)</sup> | NPU Hexagon<br><sup>QNN A16W8<br>(ms)</sup>      |
| ------------ | -------- | --------------------------- | ------------------------------------- | -------------------------------------------- | ------------------------------------------------ |
| YOLO26n      | Detect   | 640                         | 53.3<br><sup>3.6 / 47.4 / 2.4</sup>   | 17.2<br><sup>3.6 / 9.1 / 4.5</sup>           | **11.3**<br><sup>3.5 / 5.6 / 2.2</sup>           |
| YOLO26n-seg  | Segment  | 640                         | 76.0<br><sup>3.6 / 64.7 / 7.7</sup>   | 23.9<br><sup>3.6 / 11.8 / 8.6</sup>          | **21.3**<br><sup>3.5 / 7.9 / 10.0</sup>          |
| YOLO26n-sem  | Semantic | 1024                        | 66.6<br><sup>3.6 / 46.3 / 16.8</sup>  | **37.7**<br><sup>3.6 / 17.4 / 16.7</sup>     | 49.1<sup>1</sup><br><sup>8.8 / 20.8 / 19.5</sup> |
| YOLO26n-cls  | Classify | 224                         | 5.2<br><sup>0.8 / 4.0 / 0.5</sup>     | 4.5<br><sup>1.6 / 2.2 / 0.7</sup>            | **2.4**<br><sup>1.1 / 0.6 / 0.7</sup>            |
| YOLO26n-pose | Pose     | 640                         | 57.7<br><sup>3.5 / 52.4 / 1.8</sup>   | 15.2<br><sup>3.6 / 9.7 / 1.9</sup>           | **10.8**<br><sup>3.5 / 5.6 / 1.8</sup>           |
| YOLO26n-obb  | OBB      | 1024                        | 50.3<br><sup>3.6 / 45.4 / 1.3</sup>   | **13.9**<br><sup>3.8 / 8.2 / 1.8</sup>       | 21.0<br><sup>8.8 / 10.9 / 1.3</sup>              |

- **Speed** values are the full `predict()` time — preprocessing + inference + postprocessing, excluding annotation
  drawing — as the mean of 15 runs after 3 warmup runs on [bus.jpg](https://ultralytics.com/images/bus.jpg).
  <br>Reproduce with `ENABLE_QNN=1 flutter test integration_test/qnn_benchmark_test.dart -d <device> --dart-define=RUN_BENCH=true` (the example app's QNN runtime is opt-in)
- **CPU** and **GPU** run the default official INT8 TFLite assets the plugin auto-downloads, on LiteRT with
  `useGpu: false` / `true`. **NPU** runs the `*_v81_qnn.onnx` context binaries (INT8 weights, 16-bit activations) from
  the same release via the ONNX Runtime QNN Execution Provider.
- <sup>1</sup> Semantic QNN uses the in-graph ArgMax class-map exports (ultralytics#24790), which replaced erratic
  123-1065 ms logits decoding with a stable ~49 ms; the GPU remains slightly faster for semantic at 1024px. The
  official `v0.3.5` QNN release assets ship in this channel-last class-map format, exported with ultralytics
  8.4.65.
- **These are single-image burst latencies**, not sustained camera frame times: one photo through `predict()` on a
  thermally rested device. Real-time camera operation runs higher — full-sensor frames are letterboxed to the model
  input every frame and the silicon thermally settles under load (on an iPhone 17 Pro, YOLO26n detect measures
  ~3.8 ms burst but ~16 ms/frame sustained in the live camera). Watch the in-app pre/inference/post HUD line for
  your device's steady-state numbers, and benchmark your exact models on your target hardware.

## 🔭 Optimization Findings and Future Exploration

The table above reflects a device-validated optimization pass (Snapdragon 8 Elite Gen 5, June 2026). What was tried,
what worked, and what's left on the table:

**Shipped (in the table):**

- **Flat-output decode** for detect/pose/OBB: postprocess dropped from ~12 ms to 0.7-2.4 ms on every backend by
  reading the model output directly (no reshape copies, no JNI nested-array marshaling, confidence checked before
  box reads) — the same pattern as MediaPipe's decode and the iOS SDK's raw-pointer reads.
- **Channel-last (NHWC) QNN exports** (ultralytics#24790): removes the app's CPU transpose and the NPU's boundary
  transpose simultaneously; detect inference 7.4 → 5.8 ms.
- **In-graph ArgMax semantic QNN exports** (ultralytics#24790): a uint8 class map replaces ~80 MB of float logits;
  stable ~50 ms vs 123-1065 ms (erratic) before.
- **GPU program cache** (`GpuOptions.serializationDir`): model re-opens skip OpenCL compilation.

**Tested and intentionally NOT changed (don't re-litigate without new evidence):**

- `htp_performance_mode`: burst/sustained/high_performance are identical for our use — ORT votes the same max DCVS
  corner for burst and sustained; the default stays `burst`.
- `offload_graph_io_quantization=0`: no measurable effect on normal-sized outputs.
- **Naive A8W8 quantization**: 33% faster inference but zero detections — a shared uint8 scale on the concatenated
  output destroys scores. A16W8 stays the export default.
- **fp16 GPU variants**: identical inference time to INT8 on the LiteRT GPU accelerator (it computes in fp16
  internally either way) — no reason to ship larger fp16 assets.
- **In-graph ArgMax for semantic TFLite**: the GPU delegate cannot compile `ARG_MAX` with int64 indices (what
  onnx2tf emits; its argmax-replacement flags no longer exist), so the whole graph falls back to CPU — 137 ms vs
  37.6 ms for GPU logits + the app's NHWC argmax. The class-map export stays QNN/Core ML-only.
- **int32 class maps**: uint8 quarters the NPU→CPU output transfer and every consumer reads it (Core ML promotes
  it to int32 in-spec); int32 indices are reserved for >256-class models. uint8 stays the class-map dtype.

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

The official YOLO26 int8 TFLite assets can compile on the LiteRT GPU path on supported devices, but int8 GPU coverage depends on the device driver and graph. For example, a Galaxy S26 compiled `yolo26n_int8.tflite` fully with the OpenCL delegate (`Replacing 395 out of 395 node(s) with delegate (LITERT_CL)`) and ran at about **15 FPS / 32 ms** in the live camera example app. Always confirm delegate placement with device logs instead of assuming a quantization format implies CPU or GPU.

fp16 non-end-to-end TFLite exports are still useful for GPU benchmarking:

```python
from ultralytics import YOLO

YOLO("yolo26n.pt").export(format="tflite", half=True, nms=False, end2end=False, imgsz=640)
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
- **Model:** `yolo26n_int8.tflite`, official YOLO26 detect asset, 640x640 input, `int8=True`, `nms=False`, `end2end=False`.
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

**Shipped:** Local and release builds bundle `yolo26n` for all six tasks; larger sizes still download on demand. This supersedes the earlier temporary "bundle for local validation" workaround.

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

- Android official assets: YOLO26 int8 `.tflite`, `n/s/m/l/x`, detect/segment/semantic/classify/pose/OBB, hosted on `ultralytics/yolo-flutter-app` release `v0.3.5`.
- Android export settings: `int8=True`, `nms=False`, `end2end=False`; classify `imgsz=224`, all other tasks `imgsz=640`; calibration from `ultralytics.cfg.TASK2CALIBRATIONDATA`.
- Android runtime: LiteRT 2.x with GPU -> CPU accelerator fallback.
- Example UI: controls expose all six tasks and all five model sizes; model changes use one modal loading overlay for downloads and native model reloads.
- Bundled models: local/release builds fetch the six `yolo26n` nano models into `example/assets/models/` at build time (gitignored, not committed; skipped under CI), so nano tasks work offline with no first-run download; larger sizes download on demand.
- iOS runtime: with `useGpu: true` (the default) on iOS 16+, Core ML is pinned to `.cpuAndNeuralEngine` (Neural Engine + CPU), not `.all` - avoids GPU contention with the live preview/overlay compositing. iOS 15 and earlier use `.all`; `useGpu: false` pins to `.cpuOnly`.

### Open Levers

- **Release-device matrix:** Repeat the S26 test on Pixel, older Snapdragon, Tensor, and Exynos devices to record GPU delegate coverage and fallback behavior.
- **CPU baseline:** Force `useGpu: false` on the same official int8 assets and record FPS/ms/power deltas.
- **FP16 benchmark:** Compare non-end-to-end fp16 TFLite exports against the official int8 assets on devices where both compile on GPU.
- **Camera preset tuning:** Measure CameraX target resolution and analyzer throughput separately from model processing.
- **Frame-rate policy:** Explore explicit camera frame duration / analyzer throttling for latency vs. thermal tradeoffs.
- **iOS post-processing port:** The iOS SDK further reduced segment/semantic decode with raw `MLMultiArray` pointer reads and `[Float]` mask coefficients (yolo-ios-app PR #246); mirror any remaining hot-loop `NSNumber` reads in the plugin's `ios/ultralytics_yolo/Sources/ultralytics_yolo/*.swift` if profiling shows them.
