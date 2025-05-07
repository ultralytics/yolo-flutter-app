## 0.0.7

- Fix Android implementation for inference results not displaying or updating
- Fix "Unresolved reference: setIoUThreshold" error by fixing method name casing
- Add support for both "setIoUThreshold" and "setIouThreshold" method names for robustness
- Enhance error handling and logging for event channel communication
- Improve StreamHandler implementation for more reliable event dispatching
- Add fallback mechanisms for when direct method calls fail
- Fix reflection-based sink access for CustomStreamHandler
- Add test message mechanism to verify event channel connection
- Significantly increase logging for easier troubleshooting
- Update documentation with clear guidance on model placement and path resolution
- Recommend using model name only (without extension) for best cross-platform compatibility

## 0.0.8

- Fix iOS implementation for loading .mlmodel files from Flutter assets
- Significantly improve model path resolution for different path formats
- Add extensive logging to help debug model loading issues
- Fix Flutter asset bundle path issues with nested directories

## 0.0.7

- Fix iOS implementation to properly load models from Flutter assets
- Improve asset path resolution for paths like 'assets/models/yolo11n.mlmodel'
- Fix syntax errors in YoloPlugin.swift

## 0.0.6

- Add iOS implementation for checkModelExists method
- Add iOS implementation for getStoragePaths method
- Fix cross-platform consistency for model path resolution

## 0.0.5

- Update README to match current implementation of YOLO class constructor
- Fix documentation for threshold management in the API reference
- Add optional controller-based approach for managing YoloView settings
- Make onResult callback truly optional
- Improve threshold controls with IoU threshold support
- Update code documentation with detailed examples
- Add support for direct YoloView state access via GlobalKey
- Enhance error handling and debug logging
- Translate Japanese comments to English

## 0.0.4

- Initial release
- Object detection with YOLOv8 models
- Segmentation support
- Image classification support
- Pose estimation support
- Oriented Bounding Box (OBB) detection support
- Android/iOS platform support
- Real-time detection with camera feed
- Customizable confidence threshold
- YoloView Flutter widget implementation
