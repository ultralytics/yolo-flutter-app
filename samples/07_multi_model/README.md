# 07 Multi-Model Sample

This sample demonstrates how to dynamically switch between different YOLO models and tasks within a single application. It showcases model loading, disposal, and performance comparison across all supported YOLO tasks.

## Features

- ✅ Dynamic model loading and switching
- ✅ Support for all YOLO tasks (detect, segment, classify, pose, obb)
- ✅ Performance metrics (load time, inference time)
- ✅ Memory management with proper model disposal
- ✅ Task-specific result visualization
- ✅ Model comparison capabilities

## Supported Models

1. **Object Detection** (`yolo11n.tflite`)
   - Standard bounding box detection
   - 80 COCO classes

2. **Segmentation** (`yolo11n-seg.tflite`)
   - Instance segmentation masks
   - Pixel-level object detection

3. **Classification** (`yolo11n-cls.tflite`)
   - Whole image classification
   - Top-5 predictions

4. **Pose Estimation** (`yolo11n-pose.tflite`)
   - Human keypoint detection
   - 17 COCO keypoints

5. **OBB Detection** (`yolo11n-obb.tflite`)
   - Oriented bounding boxes
   - Rotated object detection

## Code Structure

```dart
// Model configuration
class ModelConfig {
  final String name;
  final String modelPath;
  final YOLOTask task;
  final IconData icon;
  final Color color;
}

// Load a model
await _currentYolo?.dispose(); // Dispose previous
_currentYolo = YOLO(
  modelPath: model.modelPath,
  task: model.task,
);
await _currentYolo.loadModel();

// Switch models dynamically
void _loadModel(ModelConfig model) async {
  // Properly dispose previous model
  // Load new model
  // Update UI
}
```

## Memory Management

The sample demonstrates proper memory management:

```dart
// Always dispose previous model before loading new one
if (_currentYolo != null) {
  await _currentYolo!.dispose();
  _currentYolo = null;
}

// Clean up in dispose
@override
void dispose() {
  _currentYolo?.dispose();
  super.dispose();
}
```

## Performance Metrics

The app tracks and displays:
- **Load Time**: Time to load the model into memory
- **Inference Time**: Time for single image prediction
- **Model Size**: Size of the model file (when available)

## UI Components

### Model Selector
Choice chips for easy model switching with visual indicators:
- Icon representing task type
- Color coding for different models
- Selected state indication

### Performance Dashboard
Real-time metrics display:
- Load time in milliseconds
- Inference time in milliseconds
- Model size in MB

### Results View
Task-specific result presentation:
- Detection: List of objects with confidence
- Segmentation: Chip view of detected instances
- Classification: Top-5 predictions
- Pose: Number of people and keypoints
- OBB: Oriented detection results

## Best Practices

1. **Model Loading**:
   - Show loading indicators
   - Handle errors gracefully
   - Dispose previous models

2. **Performance**:
   - Use Stopwatch for accurate timing
   - Display metrics to users
   - Compare model performance

3. **Memory**:
   - Dispose models when switching
   - Clear results between runs
   - Monitor memory usage

## Use Cases

- 🔄 Model A/B testing
- 📊 Performance benchmarking
- 🎯 Task selection based on use case
- 🧪 Experimentation with different models
- 📱 Adaptive model selection

## Tips

- Start with lighter models (nano variants)
- Monitor memory usage when switching models
- Consider caching frequently used models
- Implement model preloading for better UX

## Screenshot

[Add screenshot showing multi-model interface]