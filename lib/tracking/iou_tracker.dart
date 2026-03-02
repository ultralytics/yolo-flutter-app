// Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import 'dart:ui';
import 'package:ultralytics_yolo/models/yolo_result.dart';

/// Internal representation of a tracked object.
class _Track {
  final int id;
  Rect normalizedBox;
  int classIndex;
  String className;
  int age = 0;

  _Track({
    required this.id,
    required this.normalizedBox,
    required this.classIndex,
    required this.className,
  });
}

/// Greedy IoU-based object tracker that assigns persistent track IDs
/// to [YOLOResult] detections across frames.
///
/// Designed for mostly-stationary objects (e.g. cards on a table) where
/// a full multi-object tracker (Kalman + Hungarian) is overkill.
///
/// Usage:
/// ```dart
/// final tracker = IoUTracker();
/// final tracked = tracker.update(detections);
/// for (final r in tracked) {
///   print('Track #${r.trackId}: ${r.className}');
/// }
/// ```
class IoUTracker {
  /// Minimum IoU overlap required to match a detection to an existing track.
  final double iouThreshold;

  /// Number of consecutive frames a track can remain unmatched before removal.
  final int maxAge;

  /// When true, only detections with the same class as the track can match.
  final bool matchByClass;

  int _nextTrackId = 1;
  List<_Track> _tracks = [];

  IoUTracker({
    this.iouThreshold = 0.3,
    this.maxAge = 15,
    this.matchByClass = true,
  });

  /// Runs one tracking step: matches [detections] against existing tracks
  /// using greedy IoU assignment, and returns a new list of [YOLOResult]
  /// with [YOLOResult.trackId] populated.
  List<YOLOResult> update(List<YOLOResult> detections) {
    if (detections.isEmpty) {
      // Age out all tracks
      _tracks = _tracks.where((t) {
        t.age++;
        return t.age <= maxAge;
      }).toList();
      return detections;
    }

    if (_tracks.isEmpty) {
      // First frame or after reset — assign new IDs to everything
      final results = <YOLOResult>[];
      for (final det in detections) {
        final id = _nextTrackId++;
        _tracks.add(_Track(
          id: id,
          normalizedBox: det.normalizedBox,
          classIndex: det.classIndex,
          className: det.className,
        ));
        results.add(det.copyWith(trackId: id));
      }
      return results;
    }

    // Build IoU pairs: (trackIdx, detIdx, iou)
    final pairs = <_IoUPair>[];
    for (var ti = 0; ti < _tracks.length; ti++) {
      for (var di = 0; di < detections.length; di++) {
        if (matchByClass &&
            _tracks[ti].classIndex != detections[di].classIndex) {
          continue;
        }
        final iou = _computeIoU(
          _tracks[ti].normalizedBox,
          detections[di].normalizedBox,
        );
        if (iou >= iouThreshold) {
          pairs.add(_IoUPair(ti, di, iou));
        }
      }
    }

    // Sort descending by IoU for greedy assignment
    pairs.sort((a, b) => b.iou.compareTo(a.iou));

    final matchedTracks = <int>{};
    final matchedDets = <int>{};
    final detToTrackId = <int, int>{}; // detIdx → trackId

    for (final pair in pairs) {
      if (matchedTracks.contains(pair.trackIdx) ||
          matchedDets.contains(pair.detIdx)) {
        continue;
      }
      matchedTracks.add(pair.trackIdx);
      matchedDets.add(pair.detIdx);

      final track = _tracks[pair.trackIdx];
      final det = detections[pair.detIdx];
      track.normalizedBox = det.normalizedBox;
      track.classIndex = det.classIndex;
      track.className = det.className;
      track.age = 0;
      detToTrackId[pair.detIdx] = track.id;
    }

    // Age and prune unmatched existing tracks (before appending new ones)
    final originalTrackCount = _tracks.length;
    for (var ti = 0; ti < originalTrackCount; ti++) {
      if (!matchedTracks.contains(ti)) {
        _tracks[ti].age++;
      }
    }
    _tracks.removeWhere((t) => t.age > maxAge);

    // Create new tracks for unmatched detections
    for (var di = 0; di < detections.length; di++) {
      if (matchedDets.contains(di)) continue;
      final det = detections[di];
      final id = _nextTrackId++;
      _tracks.add(_Track(
        id: id,
        normalizedBox: det.normalizedBox,
        classIndex: det.classIndex,
        className: det.className,
      ));
      detToTrackId[di] = id;
    }

    // Build output
    final results = <YOLOResult>[];
    for (var di = 0; di < detections.length; di++) {
      results.add(detections[di].copyWith(trackId: detToTrackId[di]!));
    }
    return results;
  }

  /// Clears all tracks and resets the ID counter.
  void reset() {
    _tracks.clear();
    _nextTrackId = 1;
  }

  /// Computes Intersection-over-Union between two [Rect]s.
  static double _computeIoU(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.isEmpty || intersection.width <= 0 || intersection.height <= 0) {
      return 0.0;
    }
    final intersectionArea = intersection.width * intersection.height;
    final unionArea =
        a.width * a.height + b.width * b.height - intersectionArea;
    if (unionArea <= 0) return 0.0;
    return intersectionArea / unionArea;
  }
}

class _IoUPair {
  final int trackIdx;
  final int detIdx;
  final double iou;
  _IoUPair(this.trackIdx, this.detIdx, this.iou);
}
