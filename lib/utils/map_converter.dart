// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:ui';
import '../models/yolo_result.dart';

/// Helpers for adapting loosely-typed platform-channel data
/// (`Map<dynamic,dynamic>`, `List<dynamic>`) to strongly-typed Dart values.
class MapConverter {
  /// Converts a dynamic map to `Map<String, dynamic>`.
  static Map<String, dynamic> convertToTypedMap(Map<dynamic, dynamic> map) {
    return Map<String, dynamic>.fromEntries(
      map.entries.map((e) => MapEntry(e.key.toString(), e.value)),
    );
  }

  /// Null-safe variant of [convertToTypedMap].
  static Map<String, dynamic>? convertToTypedMapSafe(
    Map<dynamic, dynamic>? map,
  ) => map == null ? null : convertToTypedMap(map);

  /// Converts a list of dynamic maps to `List<Map<String, dynamic>>`, skipping
  /// non-map entries.
  static List<Map<String, dynamic>> convertMapsList(List<dynamic> maps) {
    return maps.whereType<Map>().map(convertToTypedMap).toList();
  }

  /// Alias of [convertMapsList] for detection-box lists.
  static List<Map<String, dynamic>> convertBoxesList(List<dynamic> boxes) =>
      convertMapsList(boxes);

  /// Reads `map[key]` as double; returns [fallback] if absent or non-numeric.
  static double safeGetDouble(
    Map<dynamic, dynamic> map,
    String key, {
    double fallback = 0.0,
  }) {
    final value = map[key];
    return value is num ? value.toDouble() : fallback;
  }

  /// Reads `map[key]` as int; returns [fallback] if absent or non-numeric.
  static int safeGetInt(
    Map<dynamic, dynamic> map,
    String key, {
    int fallback = 0,
  }) {
    final value = map[key];
    return value is num ? value.toInt() : fallback;
  }

  /// Reads `map[key]` as String; returns [fallback] if absent or wrong type.
  static String safeGetString(
    Map<dynamic, dynamic> map,
    String key, {
    String fallback = '',
  }) {
    final value = map[key];
    return value is String ? value : fallback;
  }

  /// Reads `map[key]` as [Uint8List] or returns null.
  static Uint8List? safeGetUint8List(Map<dynamic, dynamic> map, String key) {
    final value = map[key];
    return value is Uint8List ? value : null;
  }

  /// Converts a `{left, top, right, bottom}` map to a [Rect].
  static Rect convertBoundingBox(Map<dynamic, dynamic> boxMap) => Rect.fromLTRB(
    safeGetDouble(boxMap, 'left'),
    safeGetDouble(boxMap, 'top'),
    safeGetDouble(boxMap, 'right'),
    safeGetDouble(boxMap, 'bottom'),
  );

  /// Unpacks a flat `[x0, y0, c0, x1, y1, c1, ...]` keypoint list into
  /// parallel lists of points and confidences.
  static ({List<Point> keypoints, List<double> confidences}) convertKeypoints(
    List<dynamic> keypointsData,
  ) {
    final count = keypointsData.length ~/ 3;
    final keypoints = <Point>[];
    final confidences = <double>[];
    for (var i = 0; i < count; i++) {
      final base = i * 3;
      final x = keypointsData[base];
      final y = keypointsData[base + 1];
      final c = keypointsData[base + 2];
      keypoints.add(
        Point(x is num ? x.toDouble() : 0.0, y is num ? y.toDouble() : 0.0),
      );
      confidences.add(c is num ? c.toDouble() : 0.0);
    }
    return (keypoints: keypoints, confidences: confidences);
  }

  /// Converts a list of rows of numbers into a typed `List<List<double>>`
  /// mask. Non-numeric values become `0.0`.
  static List<List<double>> convertMaskData(List<dynamic> maskData) {
    return maskData.map((row) {
      if (row is! List) return const <double>[];
      return row.map((v) => v is num ? v.toDouble() : 0.0).toList();
    }).toList();
  }
}
