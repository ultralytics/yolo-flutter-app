// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// A widget that displays detection overlays on top of the camera view or custom images.
///
/// When used with custom images (e.g., from HTTP/RTSP sources), provide [imageRect]
/// and [imageSize] to properly position and scale the detection overlays.
///
/// Example usage with custom image:
/// ```dart
/// Stack(
///   children: [
///     Image.network('http://camera-url/image.jpg'),
///     YOLOOverlay(
///       detections: yoloResults,
///       imageRect: Rect.fromLTWH(x, y, width, height), // Image position/size on screen
///       imageSize: Size(originalWidth, originalHeight), // Original image dimensions
///     ),
///   ],
/// )
/// ```
class YOLOOverlay extends StatelessWidget {
  final List<YOLOResult> detections;
  final bool showConfidence;
  final bool showClassName;
  final YOLOOverlayTheme theme;
  final void Function(YOLOResult detection)? onDetectionTap;
  final Rect? imageRect;
  final Size? imageSize;

  const YOLOOverlay({
    super.key,
    required this.detections,
    this.showConfidence = true,
    this.showClassName = true,
    this.theme = const YOLOOverlayTheme(),
    this.onDetectionTap,
    this.imageRect,
    this.imageSize,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: YOLODetectionPainter(
        detections: detections,
        showConfidence: showConfidence,
        showClassName: showClassName,
        theme: theme,
        imageRect: imageRect,
        imageSize: imageSize,
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
    Rect detectionRect = detection.boundingBox;

    if (imageRect != null && imageSize != null) {
      detectionRect = _transformRect(detection.boundingBox);
    }

    return point.dx >= detectionRect.left &&
        point.dx <= detectionRect.right &&
        point.dy >= detectionRect.top &&
        point.dy <= detectionRect.bottom;
  }

  Rect _transformRect(Rect originalRect) {
    if (imageRect == null || imageSize == null) {
      return originalRect;
    }

    final scaleX = imageRect!.width / imageSize!.width;
    final scaleY = imageRect!.height / imageSize!.height;

    return Rect.fromLTRB(
      imageRect!.left + originalRect.left * scaleX,
      imageRect!.top + originalRect.top * scaleY,
      imageRect!.left + originalRect.right * scaleX,
      imageRect!.top + originalRect.bottom * scaleY,
    );
  }
}

/// Custom painter for drawing detection overlays.
class YOLODetectionPainter extends CustomPainter {
  final List<YOLOResult> detections;
  final bool showConfidence;
  final bool showClassName;
  final YOLOOverlayTheme theme;
  final Rect? imageRect;
  final Size? imageSize;

  YOLODetectionPainter({
    required this.detections,
    required this.showConfidence,
    required this.showClassName,
    required this.theme,
    this.imageRect,
    this.imageSize,
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

    Rect rect = Rect.fromLTRB(
      detection.boundingBox.left,
      detection.boundingBox.top,
      detection.boundingBox.right,
      detection.boundingBox.bottom,
    );

    if (imageRect != null && imageSize != null) {
      rect = _transformRect(rect);
    }

    canvas.drawRect(rect, paint);
  }

  Rect _transformRect(Rect originalRect) {
    if (imageRect == null || imageSize == null) {
      return originalRect;
    }

    final scaleX = imageRect!.width / imageSize!.width;
    final scaleY = imageRect!.height / imageSize!.height;

    return Rect.fromLTRB(
      imageRect!.left + originalRect.left * scaleX,
      imageRect!.top + originalRect.top * scaleY,
      imageRect!.left + originalRect.right * scaleX,
      imageRect!.top + originalRect.bottom * scaleY,
    );
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

    Rect detectionRect = detection.boundingBox;
    if (imageRect != null && imageSize != null) {
      detectionRect = _transformRect(detection.boundingBox);
    }

    final labelRect = Rect.fromLTRB(
      detectionRect.left,
      detectionRect.top - textPainter.height - 4,
      detectionRect.left + textPainter.width + 8,
      detectionRect.top,
    );

    // Draw background
    final backgroundPaint = Paint()..color = theme.labelBackgroundColor;
    canvas.drawRect(labelRect, backgroundPaint);

    // Draw text
    textPainter.paint(
      canvas,
      Offset(detectionRect.left + 4, detectionRect.top - textPainter.height),
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
            oldDelegate.theme != theme ||
            oldDelegate.imageRect != imageRect ||
            oldDelegate.imageSize != imageSize);
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
