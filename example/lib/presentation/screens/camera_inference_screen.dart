// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../controllers/camera_inference_controller.dart';
import '../widgets/camera_inference_content.dart';
import '../widgets/camera_inference_overlay.dart';
import '../widgets/camera_logo_overlay.dart';
import '../widgets/camera_controls.dart';
import '../widgets/threshold_slider.dart';

/// A screen that demonstrates real-time YOLO inference using the device camera.
///
/// This screen provides:
/// - Live camera feed with YOLO object detection
/// - Model selection (detect, segment, classify, pose, obb)
/// - Adjustable thresholds (confidence, IoU, max detections)
/// - Camera controls (flip, zoom)
/// - Performance metrics (FPS)
class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  late final CameraInferenceController _controller;
  int _rebuildKey = 0;

  @override
  void initState() {
    super.initState();
    _controller = CameraInferenceController();
    _controller.initialize().catchError((error) {
      if (mounted) {
        _showError('Model Loading Error', error.toString());
      }
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Check if route is current (we've navigated back to this screen)
    final route = ModalRoute.of(context);
    if (route?.isCurrent == true) {
      // Force rebuild when navigating back to ensure camera restarts
      // The rebuild will create a new YOLOView which will automatically start the camera
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          setState(() {
            _rebuildKey++;
          });
        }
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isLandscape =
        MediaQuery.of(context).orientation == Orientation.landscape;

    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Camera Inference')),
      body: ListenableBuilder(
        listenable: _controller,
        builder: (context, child) {
          return Stack(
            children: [
              CameraInferenceContent(
                key: ValueKey('camera_content_$_rebuildKey'),
                controller: _controller,
                rebuildKey: _rebuildKey,
              ),
              CameraInferenceOverlay(
                controller: _controller,
                isLandscape: isLandscape,
              ),
              CameraLogoOverlay(
                controller: _controller,
                isLandscape: isLandscape,
              ),
              CameraControls(
                currentZoomLevel: _controller.currentZoomLevel,
                isFrontCamera: _controller.isFrontCamera,
                activeSlider: _controller.activeSlider,
                onZoomChanged: _controller.setZoomLevel,
                onSliderToggled: _controller.toggleSlider,
                onCameraFlipped: _controller.flipCamera,
                isLandscape: isLandscape,
              ),
              ThresholdSlider(
                activeSlider: _controller.activeSlider,
                confidenceThreshold: _controller.confidenceThreshold,
                iouThreshold: _controller.iouThreshold,
                numItemsThreshold: _controller.numItemsThreshold,
                onValueChanged: _controller.updateSliderValue,
                isLandscape: isLandscape,
              ),
            ],
          );
        },
      ),
    );
  }

  void _showError(String title, String message) => showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('OK'),
        ),
      ],
    ),
  );
}
