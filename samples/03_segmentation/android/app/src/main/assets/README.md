# YOLO Model Assets

This directory is for YOLO model files (.tflite format) used by the Android application.

## Model Placement

Place your TensorFlow Lite (.tflite) model files in this directory. Models should be downloaded separately as they are not included in the repository to keep it lightweight.

## Example Model Names

For segmentation tasks, you can use models such as:
- `yolov11n-seg.tflite` - Nano segmentation model (lightweight, faster)
- `yolov11s-seg.tflite` - Small segmentation model (balanced)
- `yolov11m-seg.tflite` - Medium segmentation model (more accurate)
- `yolov11l-seg.tflite` - Large segmentation model (higher accuracy)
- `yolov11x-seg.tflite` - Extra large segmentation model (best accuracy)

## Model Download

You can download pre-trained models from the Ultralytics repository or convert your own models to TensorFlow Lite format using the Ultralytics tools.

## Note

The .gitkeep file in this directory ensures the assets folder is tracked by Git even when empty. You can safely delete it once you add your model files.