# Instance Segmentation Sample

This sample demonstrates how to use the Ultralytics YOLO Flutter plugin for instance segmentation tasks.

## Features

- Load YOLO segmentation models (`.tflite` for Android, `.mlmodel` for iOS)
- Select images from gallery
- Detect objects with pixel-perfect masks
- Interactive UI controls:
  - Adjustable mask opacity
  - Toggle masks on/off
  - Toggle bounding boxes on/off
- Color-coded results by object class
- Real-time performance metrics

## Getting Started

1. **Add a segmentation model**:

   - For Android: Place your `yolo11n-seg.tflite` file in `android/app/src/main/assets/`
   - For iOS: Add your `yolo11n-seg.mlmodel` to the Xcode project

2. **Run the app**:

   ```bash
   flutter pub get
   flutter run
   ```

3. **Usage**:
   - Tap "Select Image" to choose a photo from your gallery
   - Tap "Run Segmentation" to process the image
   - Use the controls to adjust visualization:
     - Drag the slider to change mask transparency
     - Toggle masks and bounding boxes on/off

## Model Information

This sample is configured to use `yolo11n-seg`, which provides a good balance between performance and accuracy:

- Model size: ~7MB
- Speed: 15-25 FPS on modern devices
- 80 object classes (COCO dataset)

## Customization

To use a different model, update the model path in `lib/main.dart`:

```dart
final yolo = YOLO(
  modelPath: 'your-model-name.tflite', // Android
  // modelPath: 'your-model-name.mlmodel', // iOS
  task: YOLOTask.segment,
);
```

## Technical Details

### Segmentation Mask Format

The plugin returns segmentation masks as a list of normalized coordinates (0-1 range) that form a polygon around each detected object. The sample app converts these to pixel coordinates for visualization.

### Performance Tips

- Use smaller models (nano/small) for better performance
- Consider reducing input image size for faster processing
- Enable GPU acceleration (automatically handled by the plugin)

## Troubleshooting

1. **Model not loading**: Ensure the model file is in the correct location and the filename matches
2. **Poor performance**: Try using a smaller model variant (e.g., yolo11n instead of yolo11s)
3. **Masks not showing**: Check that your model is a segmentation model (includes `-seg` in the name)

## Learn More

- [Ultralytics Segmentation Models](https://docs.ultralytics.com/tasks/segment/)
- [YOLO Flutter Plugin Documentation](https://pub.dev/packages/ultralytics_yolo)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)
