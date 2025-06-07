# 05 OBB Detection Sample

This sample demonstrates Oriented Bounding Box (OBB) detection using YOLO Flutter plugin. OBB detection is useful for detecting rotated objects in images, particularly in aerial/satellite imagery, document analysis, and other scenarios where objects are not axis-aligned.

## Features

- ✅ Load YOLO OBB model
- ✅ Select image from gallery
- ✅ Detect objects with rotation angles
- ✅ Draw oriented bounding boxes
- ✅ Display rotation angles
- ✅ Adjustable confidence and IoU thresholds
- ✅ Toggle labels, confidence scores, and angles

## What is OBB?

Oriented Bounding Box (OBB) detection differs from regular object detection:
- **Regular detection**: Axis-aligned bounding boxes (rectangles parallel to image edges)
- **OBB detection**: Rotated bounding boxes that better fit object orientation

This is particularly useful for:
- 🛩️ Aerial/satellite imagery (vehicles, buildings, ships)
- 📄 Document analysis (rotated text, tables)
- 🏭 Industrial inspection (parts on conveyor belts)
- 🚗 Parking lot analysis (angled parking spaces)

## Code Structure

```dart
// Initialize YOLO with OBB task
final yolo = YOLO(
  modelPath: 'yolo11n-obb.tflite',
  task: YOLOTask.obb,
);

// Get OBB results
final results = await yolo.predict(imageBytes);
final obbData = results['obb'] as List<dynamic>;

// Each OBB contains:
// - points: 4 corner points of the rotated box
// - class: Object class name
// - confidence: Detection confidence
```

## Visualization

The sample includes a custom painter that:
1. Draws oriented bounding boxes using the 4 corner points
2. Fills boxes with semi-transparent colors
3. Highlights corners with circular markers
4. Calculates and displays rotation angles
5. Falls back to regular boxes if OBB data unavailable

## Requirements

- Flutter SDK
- YOLO OBB model file (`.tflite` for Android, `.mlmodel` for iOS)
- Sample works best with aerial/satellite imagery

## Model Training

To train an OBB model:
```python
from ultralytics import YOLO

# Train OBB model
model = YOLO('yolo11n-obb.yaml')
model.train(data='path/to/obb-dataset.yaml')

# Export for mobile
model.export(format='tflite')  # Android
model.export(format='coreml')  # iOS
```

## Dataset Format

OBB datasets use rotated bounding boxes with format:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
Where (x1,y1) to (x4,y4) are the four corners of the oriented box.

## Tips

- OBB models are particularly effective for aerial/drone imagery
- Adjust confidence threshold based on your use case
- Higher IoU threshold helps reduce overlapping detections
- The angle display shows object orientation relative to horizontal

## Screenshot

[Add screenshot showing OBB detection on aerial imagery]