import 'dart:ui';

/// A detected object with segmentation mask polygons.
class DetectedSegment {
  /// Creates a [DetectedSegment].
  DetectedSegment({
    required this.confidence,
    required this.boundingBox,
    required this.index,
    required this.label,
    required this.polygons, // Add this property
  });

  /// Creates a [DetectedSegment] from a [json] object.
  factory DetectedSegment.fromJson(Map<dynamic, dynamic> json) {
    final polygonListDynamic = json['polygons'] as List<dynamic>?;
    final polygonsList = <List<Offset>>[];

    if (polygonListDynamic != null) {
      for (var polygonDynamic in polygonListDynamic) {
        if (polygonDynamic is List) {
          final polygon = <Offset>[];
          for (var pointDynamic in polygonDynamic) {
            if (pointDynamic is Map) {
              final x = pointDynamic['x'];
              final y = pointDynamic['y'];
              if (x is num && y is num) {
                polygon.add(Offset(x.toDouble(), y.toDouble()));
              } else {
                print(
                    'Warning: Invalid point format in polygon: $pointDynamic');
              }
            } else {
              print('Warning: Point is not a Map: $pointDynamic');
            }
          }
          polygonsList.add(polygon);
        } else {
          print('Warning: Polygon is not a List: $polygonDynamic');
        }
      }
    }

    return DetectedSegment(
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      boundingBox: Rect.fromLTWH(
        (json['x'] as num?)?.toDouble() ?? 0.0,
        (json['y'] as num?)?.toDouble() ?? 0.0,
        (json['width'] as num?)?.toDouble() ?? 0.0,
        (json['height'] as num?)?.toDouble() ?? 0.0,
      ),
      index: (json['index'] as num?)?.toInt() ?? 0,
      label: json['label'] as String? ?? '',
      polygons: polygonsList,
    );
  }

  /// The confidence of the detection.
  final double confidence;

  /// The bounding box of the detection.
  final Rect boundingBox;

  /// The index of the label.
  final int index;

  /// The label of the detection.
  final String label;

  /// The segmentation mask polygons. Each list of Offset represents a single polygon.
  final List<List<Offset>> polygons;
}
