<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Tests for YOLO Flutter Plugin

This directory contains comprehensive tests for the YOLO Flutter plugin and example applications, ensuring reliability and functionality across different scenarios.

## 🧪 Test Structure

The test suite is organized to validate various aspects of the plugin and example applications:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality validation
- **Widget Tests**: UI component behavior verification
- **Performance Tests**: Application performance benchmarks
- **Multi-Instance Tests**: Multiple YOLO instance functionality

## 🔄 Current Status

Tests are currently being refactored to improve coverage and maintainability. The updated test suite will include:

- Enhanced test coverage for [YOLO model integration](https://docs.ultralytics.com/models/)
- Improved [object detection](https://docs.ultralytics.com/tasks/detect/) test scenarios
- Better error handling validation
- Performance optimization tests
- Multi-instance YOLO functionality testing

## 🚀 Running Tests

### Standard Tests

Run tests using standard Flutter testing commands:

```bash
# Run all tests
flutter test

# Run specific test files
flutter test test/widget_test.dart

# Run tests with coverage
flutter test --coverage
```

### Multi-Instance Tests

For testing multiple YOLO instance functionality:

#### Method 1: Using Script

```bash
cd example
./lib/run_multi_instance_test.sh
```

#### Method 2: Direct Execution

```bash
cd example
flutter run lib/multi_instance_test_main.dart
```

#### Method 3: Run on Specific Device

```bash
# Check available devices
flutter devices

# Run on iOS device
flutter run -d < ios-device-id > lib/multi_instance_test_main.dart

# Run on Android device
flutter run -d < android-device-id > lib/multi_instance_test_main.dart
```

## 📋 Test Categories

### Core Plugin Tests

The test suite covers:

- **Model Loading**: Verification of [YOLO model](https://docs.ultralytics.com/models/yolo11/) initialization
- **Image Processing**: Input validation and preprocessing tests
- **Detection Accuracy**: Output validation for various object types
- **UI Responsiveness**: User interface interaction tests
- **Error Scenarios**: Edge case and error handling validation

### Multi-Instance Test Features

The multi-instance test app validates:

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

#### Multi-Instance Test App Features

- **Camera Capture**: Take photos with camera for inference
- **Gallery Selection**: Select saved images for inference
- **Result Display**: Show detection and segmentation results simultaneously
- **Performance Measurement**: Display inference time for each model
- **Instance Information**: Check instance info via floating button

## 📁 Required Files for Multi-Instance Tests

Place the following model files in the appropriate directory:

```
example/assets/models/
├── yolov8n.tflite      # For object detection
└── yolov8n-seg.tflite  # For segmentation
```

Model files can be downloaded from [Ultralytics](https://docs.ultralytics.com/modes/export/).

## 🔧 Troubleshooting

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

### Debugging

1. **Check Logs**

   ```bash
   flutter logs
   ```

2. **Verify Instance IDs**
   Tap the info button (floating action button in bottom right)

3. **Monitor Memory Usage**
   - iOS: Xcode > Debug Navigator > Memory
   - Android: Android Studio > Profiler

## 🛠️ Contributing

We welcome contributions to improve our test coverage! When contributing tests, please:

1. Follow Flutter testing best practices
2. Include both positive and negative test cases
3. Document test scenarios clearly
4. Ensure tests are deterministic and reliable
5. Add multi-instance tests when applicable

For more information about contributing to Ultralytics projects, visit our [contributing guidelines](https://docs.ultralytics.com/help/contributing/).

## 📚 Resources

- [Flutter Testing Documentation](https://docs.flutter.dev/testing)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Computer Vision Testing Best Practices](https://www.ultralytics.com/blog/computer-vision-models-in-finance)
- [YOLO Model Export Guide](https://docs.ultralytics.com/modes/export/)

Stay tuned for updates as we enhance the testing framework to provide better validation and reliability for the plugin and example applications!
