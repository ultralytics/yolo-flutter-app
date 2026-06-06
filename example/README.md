<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Flutter Example App

This example app shows the two core flows in the plugin:

- camera inference with `YOLOShowcase`
- single-image inference with `YOLO`

It now uses the same package-level model resolver as real apps instead of keeping separate example-only model logic.

## 🚀 Run It

```bash
cd example
flutter pub get
flutter run
```

## 📱 What It Demonstrates

- official model IDs such as `yolo26n`
- automatic official model download and caching
- custom model loading through the plugin API
- metadata-based task resolution
- full `YOLOShowcase` camera UI and single-image inference in one simple app

Example assets come from the same canonical release locations as the package resolver. They autodownload on first use and cache in app storage:

- Android TFLite int8: [yolo-flutter-app `v0.3.5`](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.3.5)
- iOS Core ML int8: [yolo-ios-app `v8.3.0`](https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.3.0)

## 🧠 Why The Example Is Small

The example is intentionally thin.

It does not maintain its own model taxonomy or download system anymore. That logic now lives in the package so users can drop the same flow into their own apps directly.

## 🔄 0.4.0 UI Migration

The camera screen is now a thin wrapper around `YOLOShowcase`, not the removed 0.3.x Dart overlay/control APIs.

| Removed 0.3.x API                                | What this example uses now                                                                      |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `YOLOOverlay`, `YOLOOverlayTheme`                | Native overlays from `YOLOView` inside `YOLOShowcase`.                                          |
| `YOLOControls`, `YOLOView.showNativeUI`          | `YOLOShowcase` for the full camera UI.                                                          |
| `YOLOView.showOverlays`, `YOLOView.overlayTheme` | No Dart-side overlay theme/toggle; consume `onResult` or `YOLO.predict()` for custom rendering. |
| `YOLOViewController.setShowUIControls()`         | Own any custom Flutter controls around a bare `YOLOView`.                                       |

For a custom camera layout, use `YOLOView` as the native camera surface and compose exported widgets such as `TaskSegmentedControl`, `ModelSizeSegmentedControl`, `ThresholdSliderRow`, `LensPicker`, `CameraToolbar`, and `PerformanceLabel` around it.

## 🔧 Customizing The Example

To try your own model:

- Android Flutter assets: add a `.tflite` file under `assets/models/`
- iOS Flutter assets: add a `.mlpackage.zip` file under `assets/models/`
- or replace the official model ID with your own local path or URL

If the export lacks metadata, pass `task` explicitly.
