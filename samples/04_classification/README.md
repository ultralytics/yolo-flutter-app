# Image Classification Sample

This sample demonstrates how to use the Ultralytics YOLO Flutter plugin for image classification tasks.

## Features

- Load YOLO classification models (`.tflite` for Android, `.mlmodel` for iOS)
- Select images from gallery
- Classify images into 1000+ categories
- Interactive UI controls:
  - Show top 3, 5, or 10 predictions
  - Visual confidence indicators
  - Highlighted top prediction
- Beautiful results display with progress bars

## Getting Started

1. **Add a classification model**:
   - For Android: Place your `yolo11n-cls.tflite` file in `android/app/src/main/assets/`
   - For iOS: Add your `yolo11n-cls.mlmodel` to the Xcode project

2. **Run the app**:

   ```bash
   flutter pub get
   flutter run
   ```

3. **Usage**:
   - Tap "Select Image" to choose a photo from your gallery
   - Tap "Classify Image" to process the image
   - Use the segmented button to show more or fewer predictions
   - The top prediction is highlighted with a star

## Model Information

This sample is configured to use `yolo11n-cls`, which provides excellent classification performance:

- Model size: ~5MB
- Speed: 30+ FPS on modern devices
- 1000 classes (ImageNet dataset)
- Top-1 accuracy: ~70%
- Top-5 accuracy: ~90%

## Customization

To use a different model, update the model path in `lib/main.dart`:

```dart
final yolo = YOLO(
  modelPath: 'your-model-name.tflite', // Android
  // modelPath: 'your-model-name.mlmodel', // iOS
  task: YOLOTask.classify,
);
```

## Technical Details

### Classification Output Format

The plugin returns a probability map with class names as keys and confidence scores as values (0-1 range). The sample app sorts these by confidence and displays the top predictions.

### Performance Tips

- Use smaller models (nano) for fastest performance
- The classification task is typically faster than detection
- Results are cached until a new image is selected

## Common Use Cases

- **Content Moderation**: Identify inappropriate content
- **Product Recognition**: Classify products in e-commerce apps
- **Scene Understanding**: Determine the context of images
- **Photo Organization**: Auto-tag images by content

## Troubleshooting

1. **Model not loading**: Ensure the model file is in the correct location and the filename matches
2. **Low confidence scores**: This is normal for complex images; the model shows uncertainty
3. **Unexpected classes**: Make sure you're using the correct model for your use case

## Learn More

- [Ultralytics Classification Models](https://docs.ultralytics.com/tasks/classify/)
- [YOLO Flutter Plugin Documentation](https://pub.dev/packages/ultralytics_yolo)
- [ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
