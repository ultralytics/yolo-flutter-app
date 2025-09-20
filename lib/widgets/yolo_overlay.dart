// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// A widget that displays detection overlays on top of the camera view.
class YOLOOverlay extends StatelessWidget {
  final List<YOLOResult> detections;
  final bool showConfidence;
  final bool showClassName;
  final YOLOOverlayTheme theme;
  final void Function(YOLOResult detection)? onDetectionTap;

  const YOLOOverlay({
    super.key,
    required this.detections,
    this.showConfidence = true,
    this.showClassName = true,
    this.theme = const YOLOOverlayTheme(),
    this.onDetectionTap,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: YOLODetectionPainter(
        detections: detections,
        showConfidence: showConfidence,
        showClassName: showClassName,
        theme: theme,
      ),
      child: GestureDetector(
        onTapDown: (details) => _handleTap(details, context),
        child: Container(),
      ),
    );
  }

  void _handleTap(TapDownDetails details, BuildContext context) {
    if (onDetectionTap == null) return;

    final RenderBox renderBox = context.findRenderObject() as RenderBox;
    final localPosition = renderBox.globalToLocal(details.globalPosition);

    for (final detection in detections) {
      if (_isPointInBoundingBox(localPosition, detection)) {
        onDetectionTap!(detection);
        break;
      }
    }
  }

  bool _isPointInBoundingBox(Offset point, YOLOResult detection) {
    return point.dx >= detection.boundingBox.left &&
        point.dx <= detection.boundingBox.right &&
        point.dy >= detection.boundingBox.top &&
        point.dy <= detection.boundingBox.bottom;
  }
}

/// Custom painter for drawing detection overlays.
class YOLODetectionPainter extends CustomPainter {
  final List<YOLOResult> detections;
  final bool showConfidence;
  final bool showClassName;
  final YOLOOverlayTheme theme;

  YOLODetectionPainter({
    required this.detections,
    required this.showConfidence,
    required this.showClassName,
    required this.theme,
  });

  @override
  void paint(Canvas canvas, Size size) {
    for (final detection in detections) {
      _drawBoundingBox(canvas, detection);
      if (showClassName || showConfidence) {
        _drawLabel(canvas, detection);
      }
    }
  }

  void _drawBoundingBox(Canvas canvas, YOLOResult detection) {
    final paint = Paint()
      ..color = theme.boundingBoxColor
      ..strokeWidth = theme.boundingBoxWidth
      ..style = PaintingStyle.stroke;

    final rect = Rect.fromLTRB(
      detection.boundingBox.left,
      detection.boundingBox.top,
      detection.boundingBox.right,
      detection.boundingBox.bottom,
    );

    canvas.drawRect(rect, paint);
  }

  void _drawLabel(Canvas canvas, YOLOResult detection) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: _buildLabelText(detection),
        style: TextStyle(
          color: theme.textColor,
          fontSize: theme.textSize,
          fontWeight: theme.textWeight,
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();

    final labelRect = Rect.fromLTRB(
      detection.boundingBox.left,
      detection.boundingBox.top - textPainter.height - 4,
      detection.boundingBox.left + textPainter.width + 8,
      detection.boundingBox.top,
    );

    // Draw background
    final backgroundPaint = Paint()..color = theme.labelBackgroundColor;
    canvas.drawRect(labelRect, backgroundPaint);

    // Draw text
    textPainter.paint(
      canvas,
      Offset(
        detection.boundingBox.left + 4,
        detection.boundingBox.top - textPainter.height,
      ),
    );
  }

  String _buildLabelText(YOLOResult detection) {
    final parts = <String>[];
    if (showClassName) parts.add(detection.className);
    if (showConfidence) {
      parts.add('${(detection.confidence * 100).toStringAsFixed(1)}%');
    }
    return parts.join(' ');
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return oldDelegate is YOLODetectionPainter &&
        (oldDelegate.detections != detections ||
            oldDelegate.showConfidence != showConfidence ||
            oldDelegate.showClassName != showClassName ||
            oldDelegate.theme != theme);
  }
}

/// Theme configuration for YOLO overlays.
class YOLOOverlayTheme {
  final Color boundingBoxColor;
  final double boundingBoxWidth;
  final Color textColor;
  final double textSize;
  final FontWeight textWeight;
  final Color labelBackgroundColor;

  const YOLOOverlayTheme({
    this.boundingBoxColor = Colors.red,
    this.boundingBoxWidth = 2.0,
    this.textColor = Colors.white,
    this.textSize = 12.0,
    this.textWeight = FontWeight.bold,
    this.labelBackgroundColor = Colors.black54,
  });

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is YOLOOverlayTheme &&
        other.boundingBoxColor == boundingBoxColor &&
        other.boundingBoxWidth == boundingBoxWidth &&
        other.textColor == textColor &&
        other.textSize == textSize &&
        other.textWeight == textWeight &&
        other.labelBackgroundColor == labelBackgroundColor;
  }

  @override
  int get hashCode {
    return Object.hash(
      boundingBoxColor,
      boundingBoxWidth,
      textColor,
      textSize,
      textWeight,
      labelBackgroundColor,
    );
  }
}
