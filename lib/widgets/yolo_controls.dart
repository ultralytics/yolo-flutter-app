// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

/// A widget that provides UI controls for YOLO detection settings.
class YOLOControls extends StatelessWidget {
  final YOLOViewController controller;
  final bool showAdvanced;
  final VoidCallback? onControlsChanged;

  const YOLOControls({
    super.key,
    required this.controller,
    this.showAdvanced = false,
    this.onControlsChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(8.0),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detection Controls',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            _buildConfidenceSlider(),
            const SizedBox(height: 16),
            _buildIoUSlider(),
            if (showAdvanced) ...[
              const SizedBox(height: 16),
              _buildNumItemsSlider(),
            ],
            const SizedBox(height: 16),
            _buildCameraControls(),
          ],
        ),
      ),
    );
  }

  Widget _buildConfidenceSlider() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Confidence: ${(controller.confidenceThreshold * 100).toStringAsFixed(0)}%',
        ),
        Slider(
          value: controller.confidenceThreshold,
          min: 0.0,
          max: 1.0,
          divisions: 20,
          onChanged: (value) {
            controller.setConfidenceThreshold(value);
            onControlsChanged?.call();
          },
        ),
      ],
    );
  }

  Widget _buildIoUSlider() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'IoU Threshold: ${(controller.iouThreshold * 100).toStringAsFixed(0)}%',
        ),
        Slider(
          value: controller.iouThreshold,
          min: 0.0,
          max: 1.0,
          divisions: 20,
          onChanged: (value) {
            controller.setIoUThreshold(value);
            onControlsChanged?.call();
          },
        ),
      ],
    );
  }

  Widget _buildNumItemsSlider() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Max Items: ${controller.numItemsThreshold}'),
        Slider(
          value: controller.numItemsThreshold.toDouble(),
          min: 1.0,
          max: 100.0,
          divisions: 99,
          onChanged: (value) {
            controller.setNumItemsThreshold(value.round());
            onControlsChanged?.call();
          },
        ),
      ],
    );
  }

  Widget _buildCameraControls() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        ElevatedButton.icon(
          onPressed: controller.isInitialized ? controller.switchCamera : null,
          icon: const Icon(Icons.switch_camera),
          label: const Text('Switch Camera'),
        ),
        ElevatedButton.icon(
          onPressed: controller.isInitialized ? controller.zoomIn : null,
          icon: const Icon(Icons.zoom_in),
          label: const Text('Zoom In'),
        ),
        ElevatedButton.icon(
          onPressed: controller.isInitialized ? controller.zoomOut : null,
          icon: const Icon(Icons.zoom_out),
          label: const Text('Zoom Out'),
        ),
      ],
    );
  }
}

class YOLOControlsCompact extends StatelessWidget {
  final YOLOViewController controller;

  const YOLOControlsCompact({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8.0),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8.0),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            onPressed: controller.isInitialized
                ? controller.switchCamera
                : null,
            icon: const Icon(Icons.switch_camera, color: Colors.white),
            tooltip: 'Switch Camera',
          ),
          IconButton(
            onPressed: controller.isInitialized ? controller.zoomIn : null,
            icon: const Icon(Icons.zoom_in, color: Colors.white),
            tooltip: 'Zoom In',
          ),
          IconButton(
            onPressed: controller.isInitialized ? controller.zoomOut : null,
            icon: const Icon(Icons.zoom_out, color: Colors.white),
            tooltip: 'Zoom Out',
          ),
        ],
      ),
    );
  }
}
