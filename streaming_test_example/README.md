# YOLO Streaming Test Example

This is a dedicated test application for YOLO streaming functionality with inference frequency control.

## 🎯 Purpose

This app focuses specifically on testing:
- **Inference Frequency Control** - Control how often YOLO inference runs (5-30 FPS or Auto)
- **Streaming Performance** - Real-time FPS and processing time metrics
- **Data Type Switching** - Toggle between Minimal and Full streaming modes
- **Real-time Indicators** - Visual feedback for available data types

## 🚀 Quick Start

```bash
cd streaming_test_example/
flutter pub get
flutter run
```

## 📱 Features

### **Inference Frequency Control**
- **Auto**: Maximum inference frequency (device-dependent, usually 30-60 FPS)
- **30 FPS**: High frequency - smooth tracking, high resource usage
- **20 FPS**: Balanced - good tracking with moderate resource usage  
- **15 FPS**: Medium frequency - adequate detection, balanced power usage
- **10 FPS**: Low frequency - basic detection, low resource usage
- **5 FPS**: Very low - minimal detection, battery saving mode

### **Streaming Modes**
- **Minimal Mode**: Basic detection only (~1-2KB/frame)
- **Full Mode**: All data types (~100KB-1MB/frame)

### **Real-time Metrics**
- Live FPS display
- Processing time in milliseconds
- Detection count
- Data availability indicators (DETECT, MASKS, POSES, OBB, IMAGE)

## 🔧 Usage

1. **Select Inference Frequency**: Tap "Inference Frequency" to expand options
2. **Choose Mode**: Switch between Minimal/Full streaming modes
3. **Monitor Performance**: Watch real-time metrics in the top status bar
4. **Observe Indicators**: Check which data types are available in real-time

## 🎛️ Configuration Examples

The app demonstrates various `YOLOStreamingConfig` configurations:

```dart
// Minimal mode with 15 FPS inference
YOLOStreamingConfig(
  includeDetections: true,
  includeClassifications: true,
  includeProcessingTimeMs: true,
  includeFps: true,
  includeMasks: false,
  includePoses: false,
  includeOBB: false,
  includeOriginalImage: false,
  inferenceFrequency: 15,
)

// Full mode with auto inference frequency
YOLOStreamingConfig(
  includeDetections: true,
  includeClassifications: true,
  includeProcessingTimeMs: true,
  includeFps: true,
  includeMasks: true,
  includePoses: true,
  includeOBB: true,
  includeOriginalImage: false,
  inferenceFrequency: null, // Auto
)
```

## 📊 Performance Testing

Use this app to test different configurations and measure:
- Impact of inference frequency on FPS
- Resource usage at different frequency settings
- Data transfer overhead in Full vs Minimal mode
- Real-time responsiveness

## 🔗 Related

- **Full Example**: See `/example/` for complete YOLO functionality demo
- **Simple Example**: See `/simple_example/` for basic YOLO usage
- **Main Plugin**: See `/lib/` for plugin source code