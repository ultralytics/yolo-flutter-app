// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Configuration class for customizing YOLO streaming behavior.
///
/// This class allows fine-grained control over what data is included in
/// real-time YOLO detection streams. It provides options to balance
/// performance with data richness based on application needs.
///
/// Example usage:
/// ```dart
/// // Lightweight configuration for high FPS
/// final lightConfig = YOLOStreamingConfig.lightweight();
///
/// // Detailed configuration with all data
/// final detailedConfig = YOLOStreamingConfig.detailed();
///
/// // Custom configuration
/// final customConfig = YOLOStreamingConfig.custom(
///   includeOriginalImage: true,
///   maxFPS: 20,
///   includeMasks: false,
/// );
/// ```
class YOLOStreamingConfig {
  /// Whether to include basic detection results (bounding boxes, confidence, class).
  /// 
  /// This is the core YOLO output and is typically always enabled.
  /// Disabling this will result in no detection data being sent.
  final bool includeDetections;

  /// Whether to include classification results.
  ///
  /// Relevant for classify tasks. When enabled, classification scores
  /// are included in the streaming results.
  final bool includeClassifications;

  /// Whether to include processing time metrics in milliseconds.
  ///
  /// This provides timing information for performance monitoring.
  /// Default is true to maintain compatibility with existing apps.
  final bool includeProcessingTimeMs;

  /// Whether to include frames per second (FPS) metrics.
  ///
  /// This provides real-time FPS information for performance monitoring.
  /// Default is true to maintain compatibility with existing apps.
  final bool includeFps;

  /// Whether to include segmentation masks in the results.
  ///
  /// Only relevant for segmentation tasks. Masks can be memory-intensive
  /// and may impact performance on lower-end devices.
  final bool includeMasks;

  /// Whether to include pose estimation keypoints.
  ///
  /// Only relevant for pose estimation tasks. Includes keypoint coordinates
  /// and confidence scores for detected poses.
  final bool includePoses;

  /// Whether to include oriented bounding box (OBB) data.
  ///
  /// Only relevant for OBB detection tasks. Includes rotated bounding box
  /// coordinates and angles.
  final bool includeOBB;

  /// Whether to include annotated images with drawn detections.
  ///
  /// Annotated images show bounding boxes, labels, and confidence scores
  /// drawn on the original image. This can be memory-intensive.
  final bool includeAnnotatedImage;

  /// Whether to include original camera frames without annotations.
  ///
  /// Original images are useful for custom post-processing or debugging.
  /// This significantly increases memory usage and should be used carefully.
  final bool includeOriginalImage;

  /// Maximum frames per second for streaming.
  ///
  /// When set, limits the rate at which results are sent to improve
  /// performance. Null means no limit (device-dependent maximum).
  final int? maxFPS;

  /// Minimum interval between result transmissions.
  ///
  /// When set, ensures a minimum time gap between consecutive results.
  /// Useful for throttling high-frequency updates.
  final Duration? throttleInterval;

  /// Creates a YOLOStreamingConfig with custom settings.
  ///
  /// This constructor allows full customization of streaming behavior.
  /// All parameters have sensible defaults that maintain backward compatibility.
  const YOLOStreamingConfig({
    this.includeDetections = true,
    this.includeClassifications = true,
    this.includeProcessingTimeMs = true,
    this.includeFps = true,
    this.includeMasks = true,
    this.includePoses = true,
    this.includeOBB = true,
    this.includeAnnotatedImage = true,
    this.includeOriginalImage = false,
    this.maxFPS,
    this.throttleInterval,
  });

  /// Creates a standard configuration that matches current default behavior.
  ///
  /// This configuration includes all detection data (masks, poses, OBB when applicable),
  /// performance metrics (FPS, processing time), and annotated images.
  /// Original images are excluded to manage memory usage.
  ///
  /// This is the default configuration used when no streamingConfig is specified,
  /// ensuring backward compatibility with existing applications.
  ///
  /// Typical performance: 15-25 FPS depending on device and model.
  const YOLOStreamingConfig.standard()
      : includeDetections = true,
        includeClassifications = true,
        includeProcessingTimeMs = true,
        includeFps = true,
        includeMasks = true,
        includePoses = true,
        includeOBB = true,
        includeAnnotatedImage = true,
        includeOriginalImage = false,
        maxFPS = null,
        throttleInterval = null;

  /// Creates a lightweight configuration optimized for high performance.
  ///
  /// This configuration includes only essential detection data and performance metrics.
  /// Heavy data like masks, poses, OBB, and images are excluded to maximize FPS.
  ///
  /// Ideal for:
  /// - Real-time applications requiring high frame rates
  /// - Resource-constrained devices
  /// - Applications that only need basic bounding box detection
  ///
  /// Typical performance: 25-35 FPS depending on device and model.
  const YOLOStreamingConfig.lightweight()
      : includeDetections = true,
        includeClassifications = true,
        includeProcessingTimeMs = true,
        includeFps = true,
        includeMasks = false,
        includePoses = false,
        includeOBB = false,
        includeAnnotatedImage = false,
        includeOriginalImage = false,
        maxFPS = null,
        throttleInterval = null;

  /// Creates a detailed configuration that includes all available data.
  ///
  /// This configuration includes every type of data: detection results,
  /// task-specific data (masks, poses, OBB), performance metrics,
  /// and both annotated and original images.
  ///
  /// Ideal for:
  /// - Development and debugging
  /// - Data collection and analysis
  /// - Applications that need comprehensive detection information
  ///
  /// Note: This configuration is memory-intensive and may impact performance.
  /// Typical performance: 10-20 FPS depending on device and model.
  const YOLOStreamingConfig.detailed()
      : includeDetections = true,
        includeClassifications = true,
        includeProcessingTimeMs = true,
        includeFps = true,
        includeMasks = true,
        includePoses = true,
        includeOBB = true,
        includeAnnotatedImage = true,
        includeOriginalImage = true,
        maxFPS = null,
        throttleInterval = null;

  /// Creates a custom configuration with specified parameters.
  ///
  /// This named constructor provides a convenient way to create custom
  /// configurations while maintaining default values for unspecified parameters.
  ///
  /// Example:
  /// ```dart
  /// final config = YOLOStreamingConfig.custom(
  ///   includeOriginalImage: true,
  ///   maxFPS: 20,
  ///   includeMasks: false,
  /// );
  /// ```
  const YOLOStreamingConfig.custom({
    bool? includeDetections,
    bool? includeClassifications,
    bool? includeProcessingTimeMs,
    bool? includeFps,
    bool? includeMasks,
    bool? includePoses,
    bool? includeOBB,
    bool? includeAnnotatedImage,
    bool? includeOriginalImage,
    int? maxFPS,
    Duration? throttleInterval,
  })  : includeDetections = includeDetections ?? true,
        includeClassifications = includeClassifications ?? true,
        includeProcessingTimeMs = includeProcessingTimeMs ?? true,
        includeFps = includeFps ?? true,
        includeMasks = includeMasks ?? true,
        includePoses = includePoses ?? true,
        includeOBB = includeOBB ?? true,
        includeAnnotatedImage = includeAnnotatedImage ?? true,
        includeOriginalImage = includeOriginalImage ?? false,
        maxFPS = maxFPS,
        throttleInterval = throttleInterval;

  /// Creates a copy of this configuration with modified parameters.
  ///
  /// This method allows creating variations of existing configurations
  /// by changing only specific parameters.
  ///
  /// Example:
  /// ```dart
  /// final baseConfig = YOLOStreamingConfig.standard();
  /// final modifiedConfig = baseConfig.copyWith(
  ///   includeOriginalImage: true,
  ///   maxFPS: 15,
  /// );
  /// ```
  YOLOStreamingConfig copyWith({
    bool? includeDetections,
    bool? includeClassifications,
    bool? includeProcessingTimeMs,
    bool? includeFps,
    bool? includeMasks,
    bool? includePoses,
    bool? includeOBB,
    bool? includeAnnotatedImage,
    bool? includeOriginalImage,
    int? maxFPS,
    Duration? throttleInterval,
  }) {
    return YOLOStreamingConfig(
      includeDetections: includeDetections ?? this.includeDetections,
      includeClassifications: includeClassifications ?? this.includeClassifications,
      includeProcessingTimeMs: includeProcessingTimeMs ?? this.includeProcessingTimeMs,
      includeFps: includeFps ?? this.includeFps,
      includeMasks: includeMasks ?? this.includeMasks,
      includePoses: includePoses ?? this.includePoses,
      includeOBB: includeOBB ?? this.includeOBB,
      includeAnnotatedImage: includeAnnotatedImage ?? this.includeAnnotatedImage,
      includeOriginalImage: includeOriginalImage ?? this.includeOriginalImage,
      maxFPS: maxFPS ?? this.maxFPS,
      throttleInterval: throttleInterval ?? this.throttleInterval,
    );
  }

  /// Validates the configuration and returns any warnings or errors.
  ///
  /// This method checks for potentially problematic combinations of settings
  /// and returns a list of warning messages. An empty list indicates no issues.
  ///
  /// Example warnings:
  /// - Including original images without FPS limit may cause memory issues
  /// - Disabling all detection data will result in empty streams
  /// - Very high FPS limits may not be achievable on all devices
  List<String> validate() {
    final warnings = <String>[];

    // Check if all detection data is disabled
    if (!includeDetections && !includeClassifications) {
      warnings.add('All detection data is disabled. Results will be empty.');
    }

    // Check for memory-intensive settings
    if (includeOriginalImage && maxFPS == null) {
      warnings.add(
        'Including original images without FPS limit may cause memory issues. Consider setting maxFPS.',
      );
    }

    // Check for very high FPS limits
    if (maxFPS != null && maxFPS! > 60) {
      warnings.add(
        'FPS limit of $maxFPS is very high and may not be achievable on all devices.',
      );
    }

    // Check for very restrictive throttling
    if (throttleInterval != null && throttleInterval!.inMilliseconds > 1000) {
      warnings.add(
        'Throttle interval of ${throttleInterval!.inMilliseconds}ms is quite restrictive and may result in very low update rates.',
      );
    }

    return warnings;
  }

  /// Returns a Map representation of this configuration.
  ///
  /// This method is useful for serialization and platform communication.
  /// The returned map contains all configuration parameters as key-value pairs.
  Map<String, dynamic> toMap() {
    return {
      'includeDetections': includeDetections,
      'includeClassifications': includeClassifications,
      'includeProcessingTimeMs': includeProcessingTimeMs,
      'includeFps': includeFps,
      'includeMasks': includeMasks,
      'includePoses': includePoses,
      'includeOBB': includeOBB,
      'includeAnnotatedImage': includeAnnotatedImage,
      'includeOriginalImage': includeOriginalImage,
      'maxFPS': maxFPS,
      'throttleIntervalMs': throttleInterval?.inMilliseconds,
    };
  }

  /// Creates a YOLOStreamingConfig from a Map.
  ///
  /// This method is useful for deserialization and platform communication.
  /// Missing keys will use default values.
  factory YOLOStreamingConfig.fromMap(Map<String, dynamic> map) {
    return YOLOStreamingConfig(
      includeDetections: map['includeDetections'] as bool? ?? true,
      includeClassifications: map['includeClassifications'] as bool? ?? true,
      includeProcessingTimeMs: map['includeProcessingTimeMs'] as bool? ?? true,
      includeFps: map['includeFps'] as bool? ?? true,
      includeMasks: map['includeMasks'] as bool? ?? true,
      includePoses: map['includePoses'] as bool? ?? true,
      includeOBB: map['includeOBB'] as bool? ?? true,
      includeAnnotatedImage: map['includeAnnotatedImage'] as bool? ?? true,
      includeOriginalImage: map['includeOriginalImage'] as bool? ?? false,
      maxFPS: map['maxFPS'] as int?,
      throttleInterval: map['throttleIntervalMs'] != null
          ? Duration(milliseconds: map['throttleIntervalMs'] as int)
          : null,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    
    return other is YOLOStreamingConfig &&
        other.includeDetections == includeDetections &&
        other.includeClassifications == includeClassifications &&
        other.includeProcessingTimeMs == includeProcessingTimeMs &&
        other.includeFps == includeFps &&
        other.includeMasks == includeMasks &&
        other.includePoses == includePoses &&
        other.includeOBB == includeOBB &&
        other.includeAnnotatedImage == includeAnnotatedImage &&
        other.includeOriginalImage == includeOriginalImage &&
        other.maxFPS == maxFPS &&
        other.throttleInterval == throttleInterval;
  }

  @override
  int get hashCode {
    return Object.hash(
      includeDetections,
      includeClassifications,
      includeProcessingTimeMs,
      includeFps,
      includeMasks,
      includePoses,
      includeOBB,
      includeAnnotatedImage,
      includeOriginalImage,
      maxFPS,
      throttleInterval,
    );
  }

  @override
  String toString() {
    return 'YOLOStreamingConfig('
        'includeDetections: $includeDetections, '
        'includeClassifications: $includeClassifications, '
        'includeProcessingTimeMs: $includeProcessingTimeMs, '
        'includeFps: $includeFps, '
        'includeMasks: $includeMasks, '
        'includePoses: $includePoses, '
        'includeOBB: $includeOBB, '
        'includeAnnotatedImage: $includeAnnotatedImage, '
        'includeOriginalImage: $includeOriginalImage, '
        'maxFPS: $maxFPS, '
        'throttleInterval: $throttleInterval'
        ')';
  }
}