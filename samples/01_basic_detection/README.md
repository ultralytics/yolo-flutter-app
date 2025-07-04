# YOLO Basic Object Detection Sample

This sample demonstrates the fundamental usage of the Ultralytics YOLO Flutter plugin for object detection.

## Features

- Single image object detection from camera or gallery
- Real-time visualization of detected objects with bounding boxes
- Display of class names and confidence scores
- Support for 80 object classes (COCO dataset)

## Setup

1. Add the detection model to your assets:
   - Android: Place `yolo11n.tflite` in `android/app/src/main/assets/`
   - iOS: Add `yolo11n.mlmodel` to the Xcode project

2. Run the app:
   ```bash
   flutter run
   ```

## Key Concepts

### Basic YOLO Usage

```dart
// 1. Create YOLO instance
final yolo = YOLO(
  modelPath: 'yolo11n.tflite',
  task: YOLOTask.detect,
);

// 2. Load the model
await yolo.loadModel();

// 3. Perform detection
final results = await yolo.predict(
  imageBytes,
  confidenceThreshold: 0.45,
  iouThreshold: 0.45,
);

// 4. Process results
final detections = results['detections'] as List<dynamic>;
for (final detection in detections) {
  print('${detection['className']}: ${detection['confidence']}');
}
```

### Understanding Detection Results

Each detection contains:

- `className`: The name of the detected object (e.g., "person", "car")
- `confidence`: Detection confidence score (0.0 to 1.0)
- `boundingBox`: Pixel coordinates of the object
- `normalizedBox`: Normalized coordinates (0.0 to 1.0)

### Visualization

The sample includes a custom painter that:

- Draws bounding boxes around detected objects
- Displays class names and confidence scores
- Uses different colors for different object classes

## Configuration Options

- **Confidence Threshold**: Minimum confidence for detections (default: 0.45)
- **IoU Threshold**: Non-maximum suppression threshold (default: 0.45)
- **Number of Items**: Maximum detections per image (default: 30)

## Model Information

The sample uses `yolo11n.tflite`, which can detect 80 object classes including:

- People and animals
- Vehicles (cars, trucks, buses, etc.)
- Everyday objects (chairs, tables, bottles, etc.)
- Sports equipment
- Food items

For different performance/accuracy trade-offs, you can use:

- `yolo11n`: Fastest, lowest accuracy
- `yolo11s`: Balanced speed and accuracy
- `yolo11m`: Good accuracy, moderate speed
- `yolo11l`: High accuracy, slower
- `yolo11x`: Best accuracy, slowest
