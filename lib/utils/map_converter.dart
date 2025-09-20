// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';
import 'dart:ui';
import '../models/yolo_result.dart';

/// Centralized map conversion utilities for YOLO operations.
///
/// This class provides a unified way to handle map conversions and data parsing
/// that was previously duplicated across multiple files in the codebase.
class MapConverter {
  /// Converts a dynamic map to a typed String-dynamic map.
  ///
  /// This method centralizes the map conversion pattern that was repeated
  /// throughout the codebase, particularly in result processing.
  ///
  /// [map] The dynamic map to convert
  /// Returns a properly typed `Map<String, dynamic>`
  static Map<String, dynamic> convertToTypedMap(Map<dynamic, dynamic> map) {
    return Map<String, dynamic>.fromEntries(
      map.entries.map((e) => MapEntry(e.key.toString(), e.value)),
    );
  }

  /// Converts a list of dynamic maps to a list of typed maps.
  ///
  /// This method handles the common pattern of converting lists of maps
  /// that was repeated in detection result processing.
  ///
  /// [maps] The list of dynamic maps to convert
  /// Returns a list of properly typed `Map<String, dynamic>`
  static List<Map<String, dynamic>> convertMapsList(List<dynamic> maps) {
    return maps.whereType<Map>().map((item) {
      return convertToTypedMap(item);
    }).toList();
  }

  /// Converts detection boxes from dynamic format to typed format.
  ///
  /// This method centralizes the box conversion logic that was duplicated
  /// across multiple files, particularly in result processing.
  ///
  /// [boxes] The list of dynamic box data
  /// Returns a list of properly typed box maps
  static List<Map<String, dynamic>> convertBoxesList(List<dynamic> boxes) {
    return boxes.whereType<Map>().map((item) {
      return convertToTypedMap(item);
    }).toList();
  }

  /// Safely extracts a double value from a map with fallback.
  ///
  /// This method centralizes the safe double extraction pattern that was
  /// repeated throughout the codebase for coordinate and confidence values.
  ///
  /// [map] The map to extract from
  /// [key] The key to extract
  /// [fallback] The fallback value if extraction fails
  /// Returns the extracted double value or fallback
  static double safeGetDouble(
    Map<dynamic, dynamic> map,
    String key, {
    double fallback = 0.0,
  }) {
    final value = map[key];
    if (value is num) {
      return value.toDouble();
    }
    return fallback;
  }

  /// Safely extracts an int value from a map with fallback.
  ///
  /// This method centralizes the safe int extraction pattern that was
  /// repeated throughout the codebase for index and count values.
  ///
  /// [map] The map to extract from
  /// [key] The key to extract
  /// [fallback] The fallback value if extraction fails
  /// Returns the extracted int value or fallback
  static int safeGetInt(
    Map<dynamic, dynamic> map,
    String key, {
    int fallback = 0,
  }) {
    final value = map[key];
    if (value is num) {
      return value.toInt();
    }
    return fallback;
  }

  /// Safely extracts a string value from a map with fallback.
  ///
  /// This method centralizes the safe string extraction pattern that was
  /// repeated throughout the codebase for class names and labels.
  ///
  /// [map] The map to extract from
  /// [key] The key to extract
  /// [fallback] The fallback value if extraction fails
  /// Returns the extracted string value or fallback
  static String safeGetString(
    Map<dynamic, dynamic> map,
    String key, {
    String fallback = '',
  }) {
    final value = map[key];
    if (value is String) {
      return value;
    }
    return fallback;
  }

  /// Converts a bounding box map to a Rect object.
  ///
  /// This method centralizes the bounding box conversion logic that was
  /// duplicated across multiple files in result processing.
  ///
  /// [boxMap] The map containing bounding box coordinates
  /// Returns a Rect object with the bounding box coordinates
  static Rect convertBoundingBox(Map<dynamic, dynamic> boxMap) {
    return Rect.fromLTRB(
      safeGetDouble(boxMap, 'left'),
      safeGetDouble(boxMap, 'top'),
      safeGetDouble(boxMap, 'right'),
      safeGetDouble(boxMap, 'bottom'),
    );
  }

  /// Converts a list of keypoint data to Point objects.
  ///
  /// This method centralizes the keypoint conversion logic that was
  /// duplicated in pose estimation result processing.
  ///
  /// [keypointsData] The list of keypoint data (x, y, confidence triplets)
  /// Returns a list of Point objects and their confidence values
  static ({List<Point> keypoints, List<double> confidences}) convertKeypoints(
    List<dynamic> keypointsData,
  ) {
    final keypoints = <Point>[];
    final confidences = <double>[];

    for (var i = 0; i < keypointsData.length; i += 3) {
      if (i + 2 < keypointsData.length) {
        final x = keypointsData[i] is num
            ? (keypointsData[i] as num).toDouble()
            : 0.0;
        final y = keypointsData[i + 1] is num
            ? (keypointsData[i + 1] as num).toDouble()
            : 0.0;
        final confidence = keypointsData[i + 2] is num
            ? (keypointsData[i + 2] as num).toDouble()
            : 0.0;

        keypoints.add(Point(x, y));
        confidences.add(confidence);
      }
    }

    return (keypoints: keypoints, confidences: confidences);
  }

  /// Converts mask data from dynamic format to typed format.
  ///
  /// This method centralizes the mask conversion logic that was
  /// duplicated in segmentation result processing.
  ///
  /// [maskData] The dynamic mask data
  /// Returns a properly typed mask as `List<List<double>>`
  static List<List<double>> convertMaskData(List<dynamic> maskData) {
    return maskData.map((row) {
      if (row is List) {
        return row.map((val) {
          if (val is num) {
            return val.toDouble();
          }
          return 0.0;
        }).toList();
      }
      return <double>[];
    }).toList();
  }

  /// Safely extracts a Uint8List from a map.
  ///
  /// This method centralizes the Uint8List extraction pattern that was
  /// repeated in image data processing.
  ///
  /// [map] The map to extract from
  /// [key] The key to extract
  /// Returns the Uint8List or null if not found
  static Uint8List? safeGetUint8List(Map<dynamic, dynamic> map, String key) {
    final value = map[key];
    if (value is Uint8List) {
      return value;
    }
    return null;
  }

  /// Converts a map to a typed map with null safety.
  ///
  /// This method provides a safe way to convert dynamic maps to typed maps
  /// with proper null handling that was inconsistent across the codebase.
  ///
  /// [map] The dynamic map to convert
  /// Returns a properly typed map with null safety
  static Map<String, dynamic>? convertToTypedMapSafe(
    Map<dynamic, dynamic>? map,
  ) {
    if (map == null) return null;
    return convertToTypedMap(map);
  }
}
