// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import '../controllers/camera_inference_controller.dart';

/// Main content widget that handles the camera view and loading states
class CameraInferenceContent extends StatelessWidget {
  const CameraInferenceContent({
    super.key,
    required this.controller,
    this.rebuildKey = 0,
  });

  final CameraInferenceController controller;
  final int rebuildKey;

  @override
  Widget build(BuildContext context) {
    if (controller.modelPath.isNotEmpty) {
      return YOLOView(
        key: ValueKey(
          'yolo_view_${controller.modelPath}_${controller.selectedTask.name}_$rebuildKey',
        ),
        controller: controller.yoloController,
        modelPath: controller.modelPath,
        task: controller.selectedTask,
        streamingConfig: const YOLOStreamingConfig.minimal(),
        onResult: controller.onDetectionResults,
        onPerformanceMetrics: (metrics) =>
            controller.onPerformanceMetrics(metrics.fps),
        onZoomChanged: controller.onZoomChanged,
        lensFacing: controller.lensFacing,
      );
    } else {
      return const Center(
        child: Text('No model loaded', style: TextStyle(color: Colors.white)),
      );
    }
  }
}
