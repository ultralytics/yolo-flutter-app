// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

/// A widget for selecting the active YOLO task and official model.
class ModelSelector extends StatelessWidget {
  const ModelSelector({
    super.key,
    required this.selectedTask,
    required this.selectedModel,
    required this.availableTasks,
    required this.availableModels,
    required this.onTaskChanged,
    required this.onModelChanged,
  });

  final YOLOTask selectedTask;
  final String selectedModel;
  final List<YOLOTask> availableTasks;
  final List<String> availableModels;
  final ValueChanged<YOLOTask> onTaskChanged;
  final ValueChanged<String> onModelChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildModelPicker(),
        const SizedBox(height: 8),
        _buildTaskSelector(),
      ],
    );
  }

  Widget _buildModelPicker() {
    return Container(
      height: 36,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            (selectedModel.isEmpty ? 'NO MODEL' : selectedModel).toUpperCase(),
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(width: 6),
          PopupMenuButton<String>(
            padding: EdgeInsets.zero,
            elevation: 2,
            enabled: availableModels.isNotEmpty,
            onSelected: onModelChanged,
            itemBuilder: (_) => availableModels
                .map(
                  (model) => PopupMenuItem<String>(
                    value: model,
                    child: Text(model.toUpperCase()),
                  ),
                )
                .toList(),
            child: const Icon(Icons.keyboard_arrow_down, color: Colors.white),
          ),
        ],
      ),
    );
  }

  Widget _buildTaskSelector() {
    return Container(
      height: 36,
      padding: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(8),
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: availableTasks.map((task) {
            final isSelected = selectedTask == task;
            return GestureDetector(
              onTap: () {
                if (task != selectedTask) {
                  onTaskChanged(task);
                }
              },
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 4,
                ),
                decoration: BoxDecoration(
                  color: isSelected ? Colors.white : Colors.transparent,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  task.name.toUpperCase(),
                  style: TextStyle(
                    color: isSelected ? Colors.black : Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }
}
