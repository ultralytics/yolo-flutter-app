# 07 Multi-Instance Sample

This sample demonstrates how to load and maintain multiple YOLO models simultaneously and run them on the same image. It showcases true multi-instance capabilities where multiple models are kept in memory and can be applied to images in sequence.

## Features

- ✅ **Multiple models loaded simultaneously** - Keep multiple YOLO instances in memory
- ✅ **Sequential inference** - Apply all active models to the same image
- ✅ **Model persistence** - Models stay loaded even when deactivated
- ✅ **Performance comparison** - Compare inference times across models
- ✅ **Memory efficient** - Activate/deactivate models without reloading
- ✅ **Side-by-side results** - See results from all models at once

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

## Key Differences from Single Model Approach

```dart
// Multi-instance approach - Multiple models in memory
final Map<ModelConfig, YOLO> _loadedModels = {};
final Set<ModelConfig> _activeModels = {};

// Load model once, keep in memory
if (!_loadedModels.containsKey(model)) {
  final yolo = YOLO(
    modelPath: model.modelPath,
    task: model.task,
  );
  await yolo.loadModel();
  _loadedModels[model] = yolo;
}

// Run inference on all active models
for (final model in _activeModels) {
  final results = await _loadedModels[model]!.predict(imageBytes);
  // Process results...
}
```

## Memory Management

The sample demonstrates multi-instance memory management:

```dart
// Models persist in memory until app is closed
final Map<ModelConfig, YOLO> _loadedModels = {};

// Toggle model activation without reloading
if (_activeModels.contains(model)) {
  _activeModels.remove(model);  // Deactivate but keep loaded
} else {
  _activeModels.add(model);     // Reactivate instantly
}

// Clean up ALL models on dispose
@override
void dispose() {
  for (final yolo in _loadedModels.values) {
    yolo.dispose();
  }
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

Filter chips showing model states:

- **Active** (selected) - Model is loaded and will process images
- **Loaded** (green check) - Model is in memory but not active
- **Not loaded** - Model needs to be loaded first

### Performance Comparison

Shows all active models with:

- Individual load times
- Per-model inference times
- Visual completion indicators

### Results View

All active models' results displayed simultaneously:

- Each model gets its own result card
- Results appear in the order models were activated
- Task-specific visualization for each model type

## Best Practices

1. **Multi-Instance Management**:
   - Load models on-demand
   - Keep frequently used models in memory
   - Monitor total memory usage

2. **Performance**:
   - Run inference sequentially to avoid memory spikes
   - Compare inference times across models
   - Consider model size vs accuracy tradeoffs

3. **User Experience**:
   - Show clear model states (loaded/active)
   - Allow quick toggling without reload delays
   - Display all results together for comparison

## Use Cases

- 🔄 **Ensemble predictions** - Combine results from multiple models
- 📊 **Model comparison** - See which model performs best on your data
- 🎯 **Multi-task analysis** - Apply detection + classification on same image
- 🧪 **Accuracy vs Speed testing** - Compare nano vs small vs medium models
- 📱 **Redundancy** - Use multiple models for critical detections

## Tips

- Start with 2-3 models to test memory limits
- Activate models before selecting an image for faster results
- Use the same image to compare model outputs fairly
- Consider device memory when loading multiple large models
- Deactivate unused models to free up processing power

## Screenshot

[Add screenshot showing multi-model interface]
