# YOLO Multi-Instance Test App

A dedicated application for testing multi-instance YOLO functionality.

## How to Run

### Method 1: Using Script

```bash
cd example
./lib/run_multi_instance_test.sh
```

### Method 2: Direct Execution

```bash
cd example
flutter run lib/multi_instance_test_main.dart
```

### Method 3: Run on Specific Device

```bash
# Check available devices
flutter devices

# Run on iOS device
flutter run -d < ios-device-id > lib/multi_instance_test_main.dart

# Run on Android device
flutter run -d < android-device-id > lib/multi_instance_test_main.dart
```

## Required Files

Please place the following model files:

```
example/assets/models/
├── yolov8n.tflite      # For object detection
└── yolov8n-seg.tflite  # For segmentation
```

Model files can be downloaded from [Ultralytics](https://docs.ultralytics.com/modes/export/).

## Test Features

This app tests the following:

1. **Multiple Instance Creation**

   - Object detection instance
   - Segmentation instance

2. **Parallel Model Loading**

   - Load both models simultaneously

3. **Inference Execution**

   - Run both models on the same image
   - Measure inference time

4. **Instance ID Verification**
   - Display unique ID for each instance
   - Check number of active instances

## Main Features

- **Camera Capture**: Take photos with camera for inference
- **Gallery Selection**: Select saved images for inference
- **Result Display**: Show detection and segmentation results simultaneously
- **Performance Measurement**: Display inference time for each model
- **Instance Information**: Check instance info via floating button

## Troubleshooting

### Model Not Found

```bash
# Check if model files exist
ls -la example/assets/models/
```

### Build Errors

```bash
# Clean build
cd example
flutter clean
flutter pub get
flutter run lib/multi_instance_test_main.dart
```

### Android Gradle Errors

Android Gradle Plugin has already been updated to 8.3.0 in settings.gradle.

## Debugging

1. **Check Logs**

   ```bash
   flutter logs
   ```

2. **Verify Instance IDs**
   Tap the info button (floating action button in bottom right)

3. **Monitor Memory Usage**
   - iOS: Xcode > Debug Navigator > Memory
   - Android: Android Studio > Profiler
