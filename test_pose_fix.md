# Testing Pose Estimation Fix for Non-Square Models

## Issue Summary

640x320 pose models have misaligned keypoints in single image inference due to incorrect handling of aspect ratio distortion when using `.scaleFill` with non-square models.

## Fix Applied

Modified `PoseEstimater.swift` to:

1. Calculate the actual scale factor used by `.scaleFill` (which takes the larger of scaleX or scaleY)
2. Compute the offsets introduced when the scaled image is centered in the model input
3. Adjust both bounding box and keypoint coordinates to account for these transformations

## Testing Steps

### 1. Build the iOS Plugin

```bash
cd example
flutter clean
flutter pub get
cd ios
pod install
flutter build ios
```

### 2. Test with 640x320 Model

- Load a 640x320 pose model
- Process a single image with people
- Check console output for debug messages showing:
  - Input size vs model size
  - Aspect ratios
  - Scale factors and offsets
  - Raw vs adjusted keypoint coordinates

### 3. Expected Results

- Keypoints should now correctly overlap with the person's joints
- Bounding boxes should properly enclose the detected persons
- Debug output should show non-zero offsets for non-square aspect ratio mismatches

### 4. Verify 640x640 Models Still Work

- Test with a 640x640 model to ensure the fix doesn't break square models
- For square models with square inputs, offsets should be zero

## Debug Output to Look For

```
DEBUG PostProcessPose: inputSize: (1920.0, 1080.0), modelInputSize: (width: 640, height: 320)
DEBUG PostProcessPose: inputAspectRatio: 1.777..., modelAspectRatio: 2.0
DEBUG PostProcessPose: scaleFactor: 0.333..., scaledDims: (640.0, 360.0)
DEBUG PostProcessPose: offsets: (0.0, -20.0)
```

This shows the image was scaled to 640x360 (maintaining aspect ratio) and then cropped to 640x320, with 20 pixels cut from top and bottom.
