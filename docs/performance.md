# Real-Time Performance

Canonical record of the on-device profiling behind the Ultralytics YOLO Flutter example app and Android LiteRT camera pipeline. Each section captures the question, the empirical result, and the current conclusion. Use this as the baseline for future Flutter performance work.

> [!IMPORTANT]
> Host, emulator, and simulator numbers are only screening signals. Always confirm on the target device with the same exported asset, camera path, and runtime delegate that the app will ship.

## Test Setup

- **Device (ground truth):** Samsung Galaxy S26 (`SM S9420`), Android 16 / API 36.
- **Build:** Flutter example debug APK, package `com.ultralytics.yolo_example`.
- **Model:** `yolo26n_int8.tflite`, official YOLO26 detect asset, 640x640 input, `int8=True`, `nms=False`, `end2end=False`.
- **Runtime:** Android LiteRT 2.x `CompiledModel` with `useGpu: true`.
- **UI:** `YOLOShowcase` real-time camera view with the default task/size controls and threshold sliders.
- **Numbers:** EMA-smoothed app metrics after the camera and model are warm.

## Methodology

| Tool                                                           | What it measures                                    | Notes                                                                                                                           |
| -------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `YOLOView.onPerformanceMetrics` / on-screen FPS label          | Native inference stream rate and `processingTimeMs` | Best quick app-level signal. Includes the native predictor's per-frame processing time, not Flutter build/layout time.          |
| `adb logcat`                                                   | Runtime delegate selection and camera/device errors | Confirm LiteRT accelerator placement from logs such as `Replacing ... node(s) with delegate (LITERT_CL)` and `compiled on GPU`. |
| `adb exec-out screencap`                                       | Actual rendered UI and camera state                 | Catches splash, black-screen, overlay, and control-layout failures that tests and logs can miss.                                |
| `flutter analyze`, `flutter test`, `flutter build apk --debug` | Static, widget, and build health                    | Required before trusting a device-only performance result.                                                                      |

## What the App's "Inference Time" Measures

The on-screen `ms` value comes from native `YOLOResult.speed`, forwarded through `processingTimeMs`. It represents the native predictor's per-frame model processing path and result conversion. It excludes Flutter widget build time, UI composition, Android camera exposure time, and screenshot capture.

`FPS` is the actual inference stream rate emitted by the plugin. It is not simply `1000 / processingTimeMs` because the camera, analyzer backpressure, delegate scheduling, thermal state, and stream throttling can all cap the frame cadence.

## Experiment: Android Startup Path

**Q:** Why did the app get stuck after the splash screen on the Galaxy S26?

**A:** There were two separate blockers.

1. The native splash was preserved until first inference. If model resolution, camera binding, or layout did not reach first inference, the Android launch splash could hide the Flutter UI indefinitely.
2. After removing the preserved native splash, the Flutter route still failed its first layout because `CupertinoSlidingSegmentedControl` received a zero-width warm-up pass and computed a negative segment width.

**Shipped:** The example app no longer preserves the native splash until first inference. The Flutter showcase owns the in-app startup/loading state. The size segmented control now tolerates zero-width and narrow warm-up constraints before the real viewport metrics arrive.

## Experiment: Official INT8 TFLite on GPU

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

**Conclusion:** Keep official Android assets as int8 TFLite for download size and confirm GPU placement per device from LiteRT logs. Do not assume int8 means CPU-only on current LiteRT, and do not assume all int8 graphs receive the same GPU coverage: driver support is device-dependent, and unsupported graphs or ops may fall back to CPU.

## Experiment: Release Download vs. Local Validation

**Q:** Did first-use model download fail because of the Flutter resolver?

**A:** No. The attached S26 could not resolve `github.com`:

```text
ping: unknown host github.com
SocketException: Failed host lookup: 'github.com'
```

The app UI correctly showed the resolver failure. To validate the camera/inference path independent of device DNS, the same `yolo26n_int8.tflite` release asset was temporarily bundled into the ignored example asset directory for a local debug build. With that asset present, the app loaded the model, started CameraX, compiled LiteRT on GPU, and displayed live inference.

**Conclusion:** Autodownload requires normal device DNS/network access to GitHub release assets. Local bundled assets are useful for validation, but official models should remain release-hosted and downloaded/cached on first use.

## Experiment: Model Availability UI

**Q:** Why did the active `YOLO26n` chip still show a download arrow after the bundled/cache load succeeded?

**A:** The startup availability scan can complete before the model resolver copies a bundled official asset into app storage. The control state then stayed stale even though `YOLOView` had loaded the model.

**Shipped:** `YOLOShowcase._onModelLoaded` now marks the loaded size available. The selected chip clears its download glyph as soon as a model load succeeds, regardless of whether the file came from cache, bundled assets, or first-use download.

## Current Shipped Configuration

- Android official assets: YOLO26 int8 `.tflite`, `n/s/m/l/x`, detect/segment/semantic/classify/pose/OBB, hosted on `ultralytics/yolo-flutter-app` release `v0.3.5`.
- Android export settings: `int8=True`, `nms=False`, `end2end=False`; classify `imgsz=224`, all other tasks `imgsz=640`; calibration from `ultralytics.cfg.TASK2CALIBRATIONDATA`.
- Android runtime: LiteRT 2.x with GPU -> CPU accelerator fallback.
- Example UI: Flutter overlay owns loading state after native splash removal; controls expose all six tasks and all five model sizes.

## Open Levers

- **Release-device matrix:** Repeat the S26 test on Pixel, older Snapdragon, Tensor, and Exynos devices to record GPU delegate coverage and fallback behavior.
- **CPU baseline:** Force `useGpu: false` on the same official int8 assets and record FPS/ms/power deltas.
- **FP16 benchmark:** Compare non-end-to-end fp16 TFLite exports against the official int8 assets on devices where both compile on GPU.
- **Camera preset tuning:** Measure CameraX target resolution and analyzer throughput separately from model processing.
- **Frame-rate policy:** Explore explicit camera frame duration / analyzer throttling for latency vs. thermal tradeoffs.
