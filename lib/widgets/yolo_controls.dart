// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

/// Backward-compatible control widgets retained as deprecated shims.
@Deprecated(
  'Build controls in your app with YOLOViewController directly. This wrapper '
  'will be removed in a future release.',
)
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
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detection Controls',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            _ThresholdSlider(
              label:
                  'Confidence: ${(controller.confidenceThreshold * 100).toStringAsFixed(0)}%',
              value: controller.confidenceThreshold,
              onChanged: (value) {
                controller.setConfidenceThreshold(value);
                onControlsChanged?.call();
              },
            ),
            const SizedBox(height: 16),
            _ThresholdSlider(
              label:
                  'IoU Threshold: ${(controller.iouThreshold * 100).toStringAsFixed(0)}%',
              value: controller.iouThreshold,
              onChanged: (value) {
                controller.setIoUThreshold(value);
                onControlsChanged?.call();
              },
            ),
            if (showAdvanced) ...[
              const SizedBox(height: 16),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Max Items: ${controller.numItemsThreshold}'),
                  Slider(
                    value: controller.numItemsThreshold.toDouble(),
                    min: 1,
                    max: 100,
                    divisions: 99,
                    onChanged: (value) {
                      controller.setNumItemsThreshold(value.round());
                      onControlsChanged?.call();
                    },
                  ),
                ],
              ),
            ],
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed:
                      controller.isInitialized ? controller.switchCamera : null,
                  icon: const Icon(Icons.switch_camera),
                  label: const Text('Switch Camera'),
                ),
                ElevatedButton.icon(
                  onPressed: controller.isInitialized ? controller.zoomIn : null,
                  icon: const Icon(Icons.zoom_in),
                  label: const Text('Zoom In'),
                ),
                ElevatedButton.icon(
                  onPressed:
                      controller.isInitialized ? controller.zoomOut : null,
                  icon: const Icon(Icons.zoom_out),
                  label: const Text('Zoom Out'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

@Deprecated(
  'Build controls in your app with YOLOViewController directly. This wrapper '
  'will be removed in a future release.',
)
class YOLOControlsCompact extends StatelessWidget {
  final YOLOViewController controller;

  const YOLOControlsCompact({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            onPressed: controller.isInitialized ? controller.switchCamera : null,
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

class _ThresholdSlider extends StatelessWidget {
  final String label;
  final double value;
  final ValueChanged<double> onChanged;

  const _ThresholdSlider({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label),
        Slider(
          value: value,
          min: 0,
          max: 1,
          divisions: 20,
          onChanged: onChanged,
        ),
      ],
    );
  }
}
