// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../../models/models.dart';
import '../controllers/camera_inference_controller.dart';
import 'detection_stats_display.dart';
import 'model_selector.dart';
import 'threshold_pill.dart';

/// Top overlay widget containing model selector, stats, and threshold pills
class CameraInferenceOverlay extends StatelessWidget {
  const CameraInferenceOverlay({
    super.key,
    required this.controller,
    required this.isLandscape,
  });

  final CameraInferenceController controller;
  final bool isLandscape;

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16),
      left: isLandscape ? 8 : 16,
      right: isLandscape ? 8 : 16,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          ModelSelector(
            selectedModel: controller.selectedModel,
            isModelLoading: controller.isModelLoading,
            onModelChanged: controller.changeModel,
          ),
          SizedBox(height: isLandscape ? 8 : 12),
          DetectionStatsDisplay(
            detectionCount: controller.detectionCount,
            currentFps: controller.currentFps,
          ),
          const SizedBox(height: 8),
          _buildThresholdPills(),
        ],
      ),
    );
  }

  Widget _buildThresholdPills() {
    if (controller.activeSlider == SliderType.confidence) {
      return ThresholdPill(
        label:
            'CONFIDENCE THRESHOLD: ${controller.confidenceThreshold.toStringAsFixed(2)}',
      );
    } else if (controller.activeSlider == SliderType.iou) {
      return ThresholdPill(
        label: 'IOU THRESHOLD: ${controller.iouThreshold.toStringAsFixed(2)}',
      );
    } else if (controller.activeSlider == SliderType.numItems) {
      return ThresholdPill(label: 'ITEMS MAX: ${controller.numItemsThreshold}');
    }
    return const SizedBox.shrink();
  }
}
