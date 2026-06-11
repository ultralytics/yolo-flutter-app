// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Real-time performance data for a YOLO inference: frame rate, processing
/// time, frame counter, and timestamp. Delivered via
/// [YOLOView.onPerformanceMetrics].
class YOLOPerformanceMetrics {
  /// Current frames per second.
  ///
  /// Represents the actual inference rate, not the camera capture rate.
  /// Higher values indicate better performance.
  final double fps;

  /// Processing time for the current frame in milliseconds.
  ///
  /// This includes the time for model inference and result processing,
  /// but excludes camera capture and UI rendering time.
  final double processingTimeMs;

  /// Preprocessing time (letterbox + normalization) in milliseconds.
  final double preMs;

  /// Model inference time in milliseconds.
  final double inferenceMs;

  /// Postprocessing time (decode + NMS/masks) in milliseconds.
  final double postMs;

  /// Sequential frame number since detection started.
  ///
  /// Useful for tracking dropped frames or debugging timing issues.
  final int frameNumber;

  /// Timestamp when these metrics were captured.
  ///
  /// Uses system time for correlation with other app events.
  final DateTime timestamp;

  /// Creates performance metrics with the specified values.
  ///
  /// All parameters are required to ensure complete performance data.
  const YOLOPerformanceMetrics({
    required this.fps,
    required this.processingTimeMs,
    required this.frameNumber,
    required this.timestamp,
    this.preMs = 0.0,
    this.inferenceMs = 0.0,
    this.postMs = 0.0,
  });

  /// Creates performance metrics from a raw data map.
  ///
  /// This factory constructor is used internally to parse performance
  /// data from native platform implementations.
  ///
  /// Returns metrics with default values if data is missing or invalid.
  factory YOLOPerformanceMetrics.fromMap(Map<String, dynamic> data) {
    return YOLOPerformanceMetrics(
      fps: (data['fps'] as num?)?.toDouble() ?? 0.0,
      processingTimeMs: (data['processingTimeMs'] as num?)?.toDouble() ?? 0.0,
      frameNumber: (data['frameNumber'] as num?)?.toInt() ?? 0,
      timestamp: DateTime.now(), // Use current time as fallback
      preMs: (data['preMs'] as num?)?.toDouble() ?? 0.0,
      inferenceMs: (data['inferenceMs'] as num?)?.toDouble() ?? 0.0,
      postMs: (data['postMs'] as num?)?.toDouble() ?? 0.0,
    );
  }

  /// Converts metrics to a map representation.
  ///
  /// Useful for serialization or debugging purposes.
  Map<String, dynamic> toMap() {
    return {
      'fps': fps,
      'processingTimeMs': processingTimeMs,
      'frameNumber': frameNumber,
      'timestamp': timestamp.millisecondsSinceEpoch,
      'preMs': preMs,
      'inferenceMs': inferenceMs,
      'postMs': postMs,
    };
  }

  /// Returns a string representation of the performance metrics.
  ///
  /// Formatted for easy reading in logs or debug output.
  @override
  String toString() {
    return 'YOLOPerformanceMetrics('
        'fps: ${fps.toStringAsFixed(1)}, '
        'processingTime: ${processingTimeMs.toStringAsFixed(3)}ms, '
        'frame: $frameNumber, '
        'timestamp: ${timestamp.toIso8601String()})';
  }

  /// Creates a copy with modified values.
  ///
  /// Any parameter not specified will retain its current value.
  YOLOPerformanceMetrics copyWith({
    double? fps,
    double? processingTimeMs,
    int? frameNumber,
    DateTime? timestamp,
    double? preMs,
    double? inferenceMs,
    double? postMs,
  }) {
    return YOLOPerformanceMetrics(
      fps: fps ?? this.fps,
      processingTimeMs: processingTimeMs ?? this.processingTimeMs,
      frameNumber: frameNumber ?? this.frameNumber,
      timestamp: timestamp ?? this.timestamp,
      preMs: preMs ?? this.preMs,
      inferenceMs: inferenceMs ?? this.inferenceMs,
      postMs: postMs ?? this.postMs,
    );
  }

  /// Checks if the current performance indicates good real-time performance.
  ///
  /// Returns true if FPS is above 15 and processing time is reasonable.
  bool get isGoodPerformance => fps >= 15.0 && processingTimeMs <= 100.0;

  /// Checks if performance indicates potential issues.
  ///
  /// Returns true if FPS is very low or processing time is very high.
  bool get hasPerformanceIssues => fps < 10.0 || processingTimeMs > 200.0;

  /// Gets a human-readable performance rating.
  ///
  /// Returns a string describing the current performance level.
  String get performanceRating {
    if (fps >= 25.0 && processingTimeMs <= 50.0) return 'Excellent';
    if (fps >= 15.0 && processingTimeMs <= 100.0) return 'Good';
    if (fps >= 10.0 && processingTimeMs <= 150.0) return 'Fair';
    return 'Poor';
  }
}
