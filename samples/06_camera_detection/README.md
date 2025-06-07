# 06 Camera Detection Sample

This sample demonstrates real-time object detection using the device camera with YOLO Flutter plugin. It showcases the YOLOView widget for live camera feed processing with adjustable detection parameters.

## Features

- ✅ Real-time camera object detection
- ✅ Live FPS counter
- ✅ Inference time display
- ✅ Object count tracking
- ✅ Adjustable confidence threshold
- ✅ Adjustable IoU threshold
- ✅ Toggle labels and confidence display
- ✅ Start/stop camera control
- ✅ Permission handling

## Key Components

### YOLOView Widget

The `YOLOView` widget provides a complete camera preview with integrated YOLO detection:

```dart
YOLOView(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
  streamingConfig: YOLOStreamingConfig(
    confidenceThreshold: 0.45,
    iouThreshold: 0.5,
    showLabels: true,
    showConfidence: true,
    showFPS: true,
  ),
  onLoad: () => print('Model loaded'),
  onError: (error) => print('Error: $error'),
  onFPSUpdate: (fps) => updateFPS(fps),
  onInferenceTimeUpdate: (time) => updateTime(time),
  onResultsUpdate: (results) => processResults(results),
)
```

### Performance Metrics

The sample displays real-time performance information:
- **FPS**: Frames processed per second
- **Inference Time**: Time taken for each detection (ms)
- **Object Count**: Number of detected objects

### Camera Control

- Start/stop camera with a single button
- Automatic permission handling
- Graceful error handling

## Code Structure

```dart
// Initialize YOLOView
final yoloView = YOLOView(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
);

// Update settings dynamically
yoloView.updateStreamingConfig(
  YOLOStreamingConfig(
    confidenceThreshold: newThreshold,
  ),
);

// Control camera
yoloView.startCamera();
yoloView.stopCamera();
```

## Permissions

### Android
Add to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

### iOS
Add to `Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is required for object detection</string>
```

## Performance Tips

1. **Model Selection**:
   - YOLOv11n: Best for real-time (25-30 FPS)
   - YOLOv11s: More accurate but slower (18-22 FPS)

2. **Threshold Tuning**:
   - Higher confidence = fewer false positives
   - Lower IoU = fewer duplicate detections

3. **Display Options**:
   - Disable labels/confidence for better performance
   - FPS display has minimal impact

## Common Use Cases

- 🛒 Retail: Product recognition
- 🚗 Traffic: Vehicle counting
- 🏭 Manufacturing: Quality control
- 🏠 Security: Person detection
- 📱 AR: Object tracking

## Troubleshooting

### Camera not starting
- Check camera permissions
- Ensure model is loaded
- Verify device has camera

### Low FPS
- Use lighter model (YOLOv11n)
- Reduce input resolution
- Disable visual overlays

### No detections
- Lower confidence threshold
- Ensure good lighting
- Check model compatibility

## Screenshot

[Add screenshot showing camera detection in action]