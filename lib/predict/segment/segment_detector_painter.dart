// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:math';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/predict/segment/detected_segment.dart';

/// A painter used to draw the detected segments on the screen.

class SegmentDetectorPainter extends CustomPainter {
  /// Creates a [SegmentDetectorPainter].
  SegmentDetectorPainter({
    required this.results,
    required this.imageSize,
    this.maskColor, // Make maskColor optional
    this.displayWidth,
    this.showBoundingBoxes = true,
    this.showSegments = true,
  });

  /// List of detected segments to be painted.
  final List<DetectedSegment> results;

  /// The actual size of the image being processed.
  final Size imageSize; // Actual image size

  /// Optional color for the segmentation masks.
  final Color? maskColor; // Nullable maskColor

  /// Optional width for displaying the results.
  final double? displayWidth;

  /// Whether to show bounding boxes around detected segments.
  final bool showBoundingBoxes;

  /// Whether to show segmentation masks.
  final bool showSegments;

  // Helper function to generate a random color
  Color _generateRandomColor() {
    final random = Random();
    return Color.fromRGBO(
      random.nextInt(256),
      random.nextInt(256),
      random.nextInt(256),
      0.5, // Keep opacity consistent
    );
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (displayWidth == null) {
      return; // Cannot paint without displayWidth
    }

    final boundingBoxPaint = Paint()
      ..color = Colors.blue.withValues(alpha: 128)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    final scaleX =
        displayWidth! / 80.0; // Assuming segment polygons are based on 80x80
    final displayHeight = (imageSize.height / imageSize.width) * displayWidth!;
    final scaleY = displayHeight / 80.0;

    for (final result in results) {
      // Draw segmentation masks
      if (showSegments && result.polygons.isNotEmpty) {
        final segmentPaint = Paint()
          ..color = maskColor?.withValues(alpha: 128) ?? _generateRandomColor()
          ..style = PaintingStyle.fill;

        for (final polygon in result.polygons) {
          if (polygon.isNotEmpty) {
            final path = Path();

            final scaledPoints = polygon.map((point) {
              return Offset(point.dx * scaleX, point.dy * scaleY);
            }).toList();

            path.addPolygon(scaledPoints, true);
            canvas.drawPath(path, segmentPaint);
          }
        }
      }

      // Draw bounding boxes and labels
      if (showBoundingBoxes) {
        canvas.drawRect(
          Rect.fromLTWH(
            result.boundingBox.left * displayWidth!,
            result.boundingBox.top * displayHeight,
            result.boundingBox.width * displayWidth!,
            result.boundingBox.height * displayHeight,
          ),
          boundingBoxPaint,
        );

        if (result.label.isNotEmpty) {
          final textSpan = TextSpan(
            text: result.label,
            style: const TextStyle(
              color: Colors.black,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          );

          final textPainter = TextPainter(
            text: textSpan,
            textAlign: TextAlign.left,
            textDirection: TextDirection.ltr,
          )..layout(maxWidth: size.width);

          final offset = Offset(
            result.boundingBox.left * displayWidth!,
            result.boundingBox.top * displayHeight,
          );

          textPainter.paint(canvas, offset);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant SegmentDetectorPainter oldDelegate) {
    final should = oldDelegate.results != results;

    return should;
  }
}
