# Getting Started with YOLO Flutter

This guide will walk you through setting up YOLO Flutter in your app, from installation to your first working detection.

## Prerequisites

- Flutter 3.0+ installed
- Android Studio / Xcode for platform setup
- Physical device recommended (camera access)

## Installation

### 1. Add Dependency

Add to your `pubspec.yaml`:

```yaml
dependencies:
    ultralytics_yolo: ^0.1.5
```

Install:

```bash
flutter pub get
```

### 2. Platform Setup

#### Android Setup

Add to `android/app/src/main/AndroidManifest.xml`:

```xml
<!-- Camera permission -->
<uses-permission android:name="android.permission.CAMERA" />

<!-- Optional: Internet for model downloads -->
<uses-permission android:name="android.permission.INTERNET" />
```

Update `android/app/build.gradle`:

```gradle
android {
    compileSdk 35  // Required for ultralytics_yolo

    defaultConfig {
        minSdkVersion 21  // Required minimum
    }
}
```

#### iOS Setup

Add to `ios/Runner/Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access for object detection</string>
```

Update `ios/Podfile`:

```ruby
platform :ios, '13.0'  # Required minimum
```

## Models

### Download Pre-trained Models

YOLO Flutter uses different model formats for each platform:

- **Android**: `.tflite` (TensorFlow Lite)
- **iOS**: `.mlmodel` or `.mlpackage` (Core ML)

#### Option 1: Download Official Models

```bash
# Detection models
curl -L -o assets/yolo11n.tflite https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.tflite
curl -L -o assets/yolo11n.mlpackage https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.mlpackage

# Segmentation models
curl -L -o assets/yolo11n-seg.tflite https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.tflite
curl -L -o assets/yolo11n-seg.mlpackage https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.mlpackage
```

#### Option 2: Use Model Manager (Recommended)

The plugin includes a model manager that downloads models automatically:

```dart
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

final modelManager = ModelManager();

// Downloads and returns local path
final modelPath = await modelManager.getModelPath(ModelType.detect);
```

### Add Models to Assets

Update `pubspec.yaml`:

```yaml
flutter:
    assets:
        - assets/
        - assets/models/ # If using custom models
```

## Your First Detection App

Create a simple detection app:

```dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Flutter Demo',
      home: DetectionScreen(),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  @override
  _DetectionScreenState createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  final YOLOViewController _controller = YOLOViewController();
  String? _modelPath;
  bool _isLoading = true;
  int _detectionCount = 0;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      final modelManager = ModelManager();
      final modelPath = await modelManager.getModelPath(ModelType.detect);

      setState(() {
        _modelPath = modelPath;
        _isLoading = false;
      });
    } catch (e) {
      print('Error loading model: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLO Detection'),
        backgroundColor: Colors.blue,
      ),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : _modelPath == null
              ? Center(child: Text('Failed to load model'))
              : Column(
                  children: [
                    // Detection count display
                    Container(
                      padding: EdgeInsets.all(16),
                      color: Colors.black87,
                      width: double.infinity,
                      child: Text(
                        'Objects detected: $_detectionCount',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    // YOLO camera view
                    Expanded(
                      child: YOLOView(
                        controller: _controller,
                        modelPath: _modelPath!,
                        task: YOLOTask.detect,
                        onResult: (results) {
                          setState(() {
                            _detectionCount = results.length;
                          });

                          // Print detection details
                          for (final result in results) {
                            print(
                              'Detected: ${result.className} '
                              '(${(result.confidence * 100).toStringAsFixed(1)}%)'
                            );
                          }
                        },
                      ),
                    ),
                  ],
                ),
    );
  }
}
```

## Test Your App

1. **Run the app**:

    ```bash
    flutter run
    ```

2. **Point camera at objects** - You should see:

    - Bounding boxes around detected objects
    - Object names and confidence scores
    - Detection count updating in real-time

3. **Check console** - You'll see detection logs:
    ```
    Detected: person (95.2%)
    Detected: chair (87.3%)
    Detected: laptop (92.1%)
    ```

## Common First Steps

### Adjust Detection Sensitivity

```dart
YOLOView(
  // ... other parameters
  confidenceThreshold: 0.3,  // Lower = more detections
  iouThreshold: 0.4,         // Lower = more overlapping boxes
)
```

### Switch Tasks

```dart
// Change task for different AI capabilities
YOLOView(
  modelPath: 'assets/yolo11n-seg.tflite',  // Segmentation model
  task: YOLOTask.segment,                   // Segmentation task
  // ...
)
```

### Handle Performance

```dart
YOLOView(
  // ... other parameters
  streamingConfig: YOLOStreamingConfig.minimal(), // Better performance
  // OR
  streamingConfig: YOLOStreamingConfig.custom(
    maxFPS: 15,              // Limit frame rate
    inferenceFrequency: 10,  // Reduce inference frequency
  ),
)
```

## Next Steps

- **[Examples](./examples.md)** - See common use cases and patterns
- **[Performance](./performance.md)** - Optimize for production apps
- **[Streaming](./streaming.md)** - Advanced real-time features
- **[Troubleshooting](./troubleshooting.md)** - Fix common issues

## Quick Links

- [Full API Reference](./api-reference.md)
- [Example Apps](../example/)
- [Model Downloads](https://github.com/ultralytics/assets/releases)
- [Community Support](https://discord.com/invite/ultralytics)
