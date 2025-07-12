# YOLO Flutter Plugin - Coordinate Space Issue Documentation

## Problem Summary
All YOLO tasks (detect, segment, pose, obb) show incorrect bounding box positions when the model input size is not 640x640. The issue affects single image inference but not real-time camera inference.

## Root Cause Analysis

### 1. Coordinate Space Mismatch
The Vision framework internally resizes input images to match the model's expected input size (e.g., 640x640, 320x320, etc.), but the coordinate transformation logic doesn't account for this resize operation.

### 2. Current Implementation Issues

#### All Predictors (ObjectDetector, PoseEstimater, etc.)
```swift
// In predictOnImage methods:
let imageWidth = image.extent.width
let imageHeight = image.extent.height
self.inputSize = CGSize(width: imageWidth, height: imageHeight)  // Sets to actual image size
```

The problem: `inputSize` is set to the actual image dimensions, but the model outputs coordinates in the model input space (e.g., 640x640).

#### Segmenter.swift (Additional hardcoding issue)
```swift
// Lines 52, 145:
let rect = CGRect(x: box.minX / 640, y: box.minY / 640, ...)
// Lines 294-295:
let xScale = containerSize.width / 640.0
let yScale = containerSize.height / 640.0
```

### 3. Why 640x640 Models Work
When the model input size is 640x640:
- Model outputs coordinates in 0-640 range
- Division by 640 correctly normalizes to 0-1 range
- Multiplication by image size correctly scales to image coordinates

When the model input size is different (e.g., 320x320):
- Model outputs coordinates in 0-320 range
- But code assumes 0-640 range (in Segmenter) or uses wrong size for normalization
- Results in incorrect positioning and scaling

### 4. Real-time vs Single Image Difference

#### Real-time (works correctly):
- Uses normalized coordinates (0-1) throughout
- Transforms directly to view coordinates
- No assumption about model input size

#### Single Image (has issues):
- Attempts to work in pixel coordinates
- Mixes coordinate spaces (model input space vs image space)
- Makes assumptions about model input size

## Solution Approach

### For ObjectDetector and similar:
Instead of:
```swift
self.inputSize = CGSize(width: imageWidth, height: imageHeight)
```

Should maintain awareness of model input size and handle coordinate transformation properly.

### For Segmenter:
Replace hardcoded 640 with dynamic `modelInputSize`.

### General Fix:
Ensure consistent coordinate space transformations:
1. Model outputs → Normalized (0-1) using actual model input size
2. Normalized → Target image/view coordinates

## Test Case
1. Load a model with non-640x640 input size (e.g., 320x320, 384x640)
2. Run single image inference
3. Check if bounding boxes align with detected objects
4. Compare with 640x640 model results