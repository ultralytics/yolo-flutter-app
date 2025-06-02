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

  /// Whether to include original camera frames without annotations.
  ///
  /// Original images are useful for custom post-processing or debugging.
  /// This significantly increases memory usage and should be used carefully.
  final bool includeOriginalImage;

  /// Maximum frames per second for streaming output.
  ///
  /// This controls how often results are sent to Flutter, not inference frequency.
  /// When set, limits the rate at which results are sent to improve
  /// performance. Null means no limit (device-dependent maximum).
  final int? maxFPS;

  /// Minimum interval between result transmissions.
  ///
  /// When set, ensures a minimum time gap between consecutive results.
  /// Useful for throttling high-frequency updates.
  final Duration? throttleInterval;

  /// Target inference frequency in frames per second.
  ///
  /// This controls how often YOLO inference is actually performed on camera frames.
  /// Lower values reduce CPU/GPU usage and heat generation but may miss fast objects.
  /// Higher values provide smoother tracking but consume more resources.
  ///
  /// Examples:
  /// - `30`: High frequency - smooth tracking, high resource usage
  /// - `15`: Balanced - good tracking with moderate resource usage
  /// - `10`: Low frequency - basic detection, low resource usage
  /// - `5`: Very low - minimal detection, battery saving
  /// - `null`: Maximum frequency (device-dependent, usually 30-60 FPS)
  final int? inferenceFrequency;

  /// Skip frames between inferences for power saving.
  ///
  /// This is an alternative way to control inference frequency by specifying
  /// how many camera frames to skip between inferences.
  ///
  /// Examples:
  /// - `0`: Process every frame (maximum frequency)
  /// - `1`: Process every 2nd frame (half frequency)
  /// - `2`: Process every 3rd frame (1/3 frequency)
  /// - `4`: Process every 5th frame (1/5 frequency)
  ///
  /// Note: If both `inferenceFrequency` and `skipFrames` are set,
  /// `inferenceFrequency` takes precedence.
  final int? skipFrames;

  /// Creates a YOLOStreamingConfig with custom settings.
  ///
  /// This constructor allows full customization of streaming behavior.
  /// Defaults are optimized for high-speed operation with minimal data.
  const YOLOStreamingConfig({
    this.includeDetections = true,
    this.includeClassifications = true,
    this.includeProcessingTimeMs = true,
    this.includeFps = true,
    this.includeMasks = false, // Changed to false for performance
    this.includePoses = false, // Changed to false for performance
    this.includeOBB = false, // Changed to false for performance
    this.includeOriginalImage = false,
    this.maxFPS,
    this.throttleInterval,
    this.inferenceFrequency,
    this.skipFrames,
  });

  /// Creates a minimal configuration optimized for maximum performance.
  ///
  /// This is the default configuration for YOLOView, providing only essential
  /// detection data and performance metrics. Heavy data like masks, poses,
  /// OBB, and images are excluded to maximize FPS.
  ///
  /// Ideal for:
  /// - Real-time applications requiring high frame rates
  /// - Resource-constrained devices
  /// - Applications that only need basic bounding box detection
  ///
  /// Typical performance: 25-35+ FPS depending on device and model.
  const YOLOStreamingConfig.minimal()
    : includeDetections = true,
      includeClassifications = true,
      includeProcessingTimeMs = true,
      includeFps = true,
      includeMasks = false,
      includePoses = false,
      includeOBB = false,
      includeOriginalImage = false,
      maxFPS = null,
      throttleInterval = null,
      inferenceFrequency = null,
      skipFrames = null;

  /// Creates a custom configuration with specified parameters.
  ///
  /// Any unspecified parameters default to false (except detections,
  /// classifications, and performance metrics which default to true).
  ///
  /// Example:
  /// ```dart
  /// // Only include masks, no other extra data
  /// final config = YOLOStreamingConfig.custom(
  ///   includeMasks: true,
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
    bool? includeOriginalImage,
    this.maxFPS,
    this.throttleInterval,
    this.inferenceFrequency,
    this.skipFrames,
  }) : includeDetections = includeDetections ?? true,
       includeClassifications = includeClassifications ?? true,
       includeProcessingTimeMs = includeProcessingTimeMs ?? true,
       includeFps = includeFps ?? true,
       includeMasks = includeMasks ?? false,
       includePoses = includePoses ?? false,
       includeOBB = includeOBB ?? false,
       includeOriginalImage = includeOriginalImage ?? false;

  /// Creates a configuration with segmentation masks.
  ///
  /// Suitable for segmentation models where you need pixel-level masks.
  /// May impact performance due to additional data transfer.
  ///
  /// Typical performance: 15-25 FPS depending on device and mask resolution.
  const YOLOStreamingConfig.withMasks()
    : includeDetections = true,
      includeClassifications = true,
      includeProcessingTimeMs = true,
      includeFps = true,
      includeMasks = true,
      includePoses = false,
      includeOBB = false,
      includeOriginalImage = false,
      maxFPS = null,
      throttleInterval = null,
      inferenceFrequency = null,
      skipFrames = null;

  /// Creates a configuration with pose keypoints.
  ///
  /// Suitable for pose estimation models where you need keypoint data.
  /// Includes skeleton information for detected human poses.
  ///
  /// Typical performance: 20-30 FPS depending on device and number of people.
  const YOLOStreamingConfig.withPoses()
    : includeDetections = true,
      includeClassifications = true,
      includeProcessingTimeMs = true,
      includeFps = true,
      includeMasks = false,
      includePoses = true,
      includeOBB = false,
      includeOriginalImage = false,
      maxFPS = null,
      throttleInterval = null,
      inferenceFrequency = null,
      skipFrames = null;

  /// Creates a full configuration with all data included.
  ///
  /// This includes all possible data: detections, masks, poses, OBB,
  /// and original images. Use with caution as it significantly impacts
  /// performance and memory usage.
  ///
  /// Ideal for:
  /// - Debugging and development
  /// - Non-real-time processing
  /// - When all detection data is needed
  ///
  /// Typical performance: 5-15 FPS depending on device and model complexity.
  const YOLOStreamingConfig.full()
    : includeDetections = true,
      includeClassifications = true,
      includeProcessingTimeMs = true,
      includeFps = true,
      includeMasks = true,
      includePoses = true,
      includeOBB = true,
      includeOriginalImage = false,
      maxFPS = null,
      throttleInterval = null,
      inferenceFrequency = null,
      skipFrames = null;

  /// Creates a debug configuration with all data and images.
  ///
  /// This includes everything from full() plus original camera frames.
  /// Extremely resource-intensive and should only be used for debugging.
  ///
  /// WARNING: This configuration will significantly impact performance
  /// and may cause memory issues on lower-end devices.
  ///
  /// Typical performance: 2-10 FPS depending on device and image resolution.
  const YOLOStreamingConfig.debug()
    : includeDetections = true,
      includeClassifications = true,
      includeProcessingTimeMs = true,
      includeFps = true,
      includeMasks = true,
      includePoses = true,
      includeOBB = true,
      includeOriginalImage = true,
      maxFPS = null,
      throttleInterval = null,
      inferenceFrequency = null,
      skipFrames = null;

  /// Creates a throttled configuration with specified FPS limit.
  ///
  /// This is useful when you want to limit the processing rate to save
  /// battery or reduce system load.
  ///
  /// Example:
  /// ```dart
  /// // Limit to 10 FPS for battery saving
  /// final config = YOLOStreamingConfig.throttled(maxFPS: 10);
  /// ```
  factory YOLOStreamingConfig.throttled({
    required int maxFPS,
    bool includeDetections = true,
    bool includeClassifications = true,
    bool includeProcessingTimeMs = true,
    bool includeFps = true,
    bool includeMasks = false,
    bool includePoses = false,
    bool includeOBB = false,
    bool includeOriginalImage = false,
    int? inferenceFrequency,
    int? skipFrames,
  }) {
    return YOLOStreamingConfig(
      includeDetections: includeDetections,
      includeClassifications: includeClassifications,
      includeProcessingTimeMs: includeProcessingTimeMs,
      includeFps: includeFps,
      includeMasks: includeMasks,
      includePoses: includePoses,
      includeOBB: includeOBB,
      includeOriginalImage: includeOriginalImage,
      maxFPS: maxFPS,
      inferenceFrequency: inferenceFrequency,
      skipFrames: skipFrames,
    );
  }

  /// Creates a power-saving configuration with reduced inference frequency.
  ///
  /// This configuration reduces both output FPS and inference frequency
  /// to minimize battery drain and heat generation.
  ///
  /// Example:
  /// ```dart
  /// // Low power mode: 10 inference per second, 15 max output FPS
  /// final config = YOLOStreamingConfig.powerSaving();
  ///
  /// // Custom power saving with 5 inferences per second
  /// final config = YOLOStreamingConfig.powerSaving(inferenceFrequency: 5);
  /// ```
  factory YOLOStreamingConfig.powerSaving({
    int inferenceFrequency = 10,
    int maxFPS = 15,
  }) {
    return YOLOStreamingConfig(
      includeDetections: true,
      includeClassifications: true,
      includeProcessingTimeMs: true,
      includeFps: true,
      includeMasks: false,
      includePoses: false,
      includeOBB: false,
      includeOriginalImage: false,
      maxFPS: maxFPS,
      inferenceFrequency: inferenceFrequency,
    );
  }

  /// Creates a performance configuration optimized for high frame rates.
  ///
  /// This configuration maximizes inference frequency while keeping
  /// data transfer minimal for the best possible performance.
  ///
  /// Example:
  /// ```dart
  /// // High performance: 30 inferences per second
  /// final config = YOLOStreamingConfig.highPerformance();
  /// ```
  factory YOLOStreamingConfig.highPerformance({int inferenceFrequency = 30}) {
    return YOLOStreamingConfig(
      includeDetections: true,
      includeClassifications: true,
      includeProcessingTimeMs: true,
      includeFps: true,
      includeMasks: false,
      includePoses: false,
      includeOBB: false,
      includeOriginalImage: false,
      inferenceFrequency: inferenceFrequency,
    );
  }

  @override
  String toString() {
    return 'YOLOStreamingConfig('
        'detections: $includeDetections, '
        'classifications: $includeClassifications, '
        'processingTime: $includeProcessingTimeMs, '
        'fps: $includeFps, '
        'masks: $includeMasks, '
        'poses: $includePoses, '
        'obb: $includeOBB, '
        'originalImage: $includeOriginalImage, '
        'maxFPS: $maxFPS, '
        'throttleInterval: ${throttleInterval?.inMilliseconds}ms, '
        'inferenceFrequency: $inferenceFrequency, '
        'skipFrames: $skipFrames)';
  }
}
