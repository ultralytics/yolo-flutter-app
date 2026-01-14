// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import '../../models/models.dart';

/// A widget for selecting YOLO family and task.
class ModelSelector extends StatelessWidget {
  const ModelSelector({
    super.key,
    required this.selectedFamily,
    required this.selectedTask,
    required this.isModelLoading,
    required this.onFamilyChanged,
    required this.onTaskChanged,
  });

  final ModelFamily selectedFamily;
  final ModelTask selectedTask;
  final bool isModelLoading;
  final ValueChanged<ModelFamily> onFamilyChanged;
  final ValueChanged<ModelTask> onTaskChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildFamilySelector(context),
        const SizedBox(height: 8),
        _buildTaskSelector(context),
      ],
    );
  }

  Widget _buildFamilySelector(BuildContext context) {
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
            selectedFamily.label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(width: 6),
          PopupMenuButton<ModelFamily>(
            padding: EdgeInsets.zero,
            elevation: 2,
            onSelected: (family) {
              if (!isModelLoading) {
                onFamilyChanged(family);
              }
            },
            itemBuilder: (_) => ModelFamily.values
                .map(
                  (family) => PopupMenuItem<ModelFamily>(
                    value: family,
                    child: Text(family.label),
                  ),
                )
                .toList(),
            child: const Icon(Icons.keyboard_arrow_down, color: Colors.white),
          ),
        ],
      ),
    );
  }

  Widget _buildTaskSelector(BuildContext context) {
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
          children: ModelTask.values.map((task) {
            final isSelected = selectedTask == task;
            return GestureDetector(
              onTap: () {
                if (!isModelLoading && task != selectedTask) {
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
                  task.label,
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
