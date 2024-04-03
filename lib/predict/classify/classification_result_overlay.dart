import 'dart:math';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/predict/classify/classification_result.dart';

/// Base class for classification overlays.
abstract class BaseClassificationOverlay extends StatelessWidget {
  /// Constructs a [BaseClassificationOverlay].
  const BaseClassificationOverlay({
    required this.classificationResults,
    super.key,
  });

  /// The classification results to display.
  final List<ClassificationResult?> classificationResults;
}

/// An overlay that displays the top three classification results.
class ClassificationResultOverlay extends BaseClassificationOverlay {
  /// Constructs a [ClassificationResultOverlay].
  const ClassificationResultOverlay({
    required super.classificationResults,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    if (classificationResults.isEmpty) return Container();

    final objects = classificationResults
        .getRange(0, min(classificationResults.length, 3))
        .toList();

    return SafeArea(
      child: Align(
        child: Container(
          margin: const EdgeInsets.all(20),
          padding: const EdgeInsets.all(20),
          decoration: const BoxDecoration(
            borderRadius: BorderRadius.all(Radius.circular(10)),
            color: Colors.black54,
          ),
          child: ListView.builder(
            padding: EdgeInsets.zero,
            shrinkWrap: true,
            itemCount: objects.length,
            itemBuilder: (context, index) {
              final object = objects[index];

              if (object == null) return Container();

              return Text(
                '${object.label} - '
                '${(object.confidence * 100).toStringAsPrecision(2)}%',
                style: const TextStyle(color: Colors.white70),
              );
            },
          ),
        ),
      ),
    );
  }
}
