import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../controllers/camera_inference_controller.dart';
import 'model_loading_overlay.dart';

/// Main content widget that handles the camera view and loading states.
/// Uses the controller-provided models list and does not directly handle model paths.
class CameraInferenceContent extends StatelessWidget {
  const CameraInferenceContent({super.key, required this.controller});

  final CameraInferenceController controller;

  @override
  Widget build(BuildContext context) {
    // When models are ready and not loading, show the YOLOView with controller-provided models
    if (!controller.isModelLoading && controller.modelsForView.isNotEmpty) {
      return YOLOView(
        key: const ValueKey('yolo_view_static'),
        controller: controller.yoloController,
        models: controller.modelsForView,
        streamingConfig: const YOLOStreamingConfig.full(),
        onResult: controller.onDetectionResults,
        onPerformanceMetrics: (metrics) =>
            controller.onPerformanceMetrics(metrics.fps),
        onZoomChanged: controller.onZoomChanged,
      );
    }

    // Show loading overlay while models are being prepared
    if (controller.isModelLoading) {
      return ModelLoadingOverlay(
        loadingMessage: controller.loadingMessage,
        downloadProgress: controller.downloadProgress,
      );
    }

    // Fallback state when no models have been loaded yet
    return const Center(
      child: Text('No model loaded', style: TextStyle(color: Colors.white)),
    );
  }
}
