<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Flutter Example App

This example app shows the two core flows in the plugin:

- camera inference with `YOLOView`
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
- camera inference and single-image inference in one simple app

## 🧠 Why The Example Is Small

The example is intentionally thin.

It does not maintain its own model taxonomy or download system anymore. That logic now lives in the package so users can drop the same flow into their own apps directly.

## 🔧 Customizing The Example

To try your own model:

- Android Flutter assets: add a `.tflite` file under `assets/models/`
- iOS Flutter assets: add a `.mlpackage.zip` file under `assets/models/`
- or replace the official model ID with your own local path or URL

If the export lacks metadata, pass `task` explicitly.
