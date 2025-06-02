// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation

/// Configuration class for YOLOView streaming functionality
/// Controls what data is included in real-time streaming and performance settings
public struct YOLOStreamConfig {
  // Basic inference data (always useful)
  public let includeDetections: Bool
  public let includeClassifications: Bool
  public let includeProcessingTimeMs: Bool
  public let includeFps: Bool

  // Task-specific advanced data
  public let includeMasks: Bool
  public let includePoses: Bool
  public let includeOBB: Bool

  // Original image data (uses CVPixelBuffer efficiently - no additional conversion needed)
  public let includeOriginalImage: Bool

  // Performance controls
  public let maxFPS: Int?  // Limit inference to max FPS (e.g., 15, 30)
  public let throttleIntervalMs: Int?  // Minimum interval between inferences in milliseconds

  // Inference frequency controls
  public let inferenceFrequency: Int?  // Target inference frequency in FPS (e.g., 5, 10, 15, 30)
  public let skipFrames: Int?  // Skip frames between inferences (alternative to inferenceFrequency)

  // Note: annotatedImage is intentionally excluded for YOLOView
  // YOLOView uses CALayer drawing (real-time overlay), not UIImage generation

  public init(
    includeDetections: Bool = true,
    includeClassifications: Bool = true,
    includeProcessingTimeMs: Bool = true,
    includeFps: Bool = true,
    includeMasks: Bool = false,
    includePoses: Bool = false,
    includeOBB: Bool = false,
    includeOriginalImage: Bool = false,
    maxFPS: Int? = nil,
    throttleIntervalMs: Int? = nil,
    inferenceFrequency: Int? = nil,
    skipFrames: Int? = nil
  ) {
    self.includeDetections = includeDetections
    self.includeClassifications = includeClassifications
    self.includeProcessingTimeMs = includeProcessingTimeMs
    self.includeFps = includeFps
    self.includeMasks = includeMasks
    self.includePoses = includePoses
    self.includeOBB = includeOBB
    self.includeOriginalImage = includeOriginalImage
    self.maxFPS = maxFPS
    self.throttleIntervalMs = throttleIntervalMs
    self.inferenceFrequency = inferenceFrequency
    self.skipFrames = skipFrames
  }

  /// Default minimal configuration - optimized for maximum performance
  public static let DEFAULT = YOLOStreamConfig()  // Uses all default values (minimal data)

  /// Full features configuration - includes all detection features
  public static let FULL = YOLOStreamConfig(
    includeDetections: true,
    includeClassifications: true,
    includeProcessingTimeMs: true,
    includeFps: true,
    includeMasks: true,
    includePoses: true,
    includeOBB: true,
    includeOriginalImage: false,
    maxFPS: nil  // No limit, but will be slower due to data processing
  )

  /// Debug configuration - includes everything for development
  public static let DEBUG = YOLOStreamConfig(
    includeDetections: true,
    includeClassifications: true,
    includeProcessingTimeMs: true,
    includeFps: true,
    includeMasks: true,
    includePoses: true,
    includeOBB: true,
    includeOriginalImage: true,
    maxFPS: 10  // Limited FPS due to heavy data
  )

  /// Custom builder for specific needs
  public static func custom(
    includeMasks: Bool = false,
    includePoses: Bool = false,
    includeOBB: Bool = false,
    includeOriginalImage: Bool = false,
    maxFPS: Int? = nil,
    throttleIntervalMs: Int? = nil,
    inferenceFrequency: Int? = nil,
    skipFrames: Int? = nil
  ) -> YOLOStreamConfig {
    return YOLOStreamConfig(
      includeMasks: includeMasks,
      includePoses: includePoses,
      includeOBB: includeOBB,
      includeOriginalImage: includeOriginalImage,
      maxFPS: maxFPS,
      throttleIntervalMs: throttleIntervalMs,
      inferenceFrequency: inferenceFrequency,
      skipFrames: skipFrames
    )
  }
}

/// Extension to create YOLOStreamConfig from Dictionary (for Flutter integration)
extension YOLOStreamConfig {
  public static func from(dict: [String: Any]) -> YOLOStreamConfig {
    return YOLOStreamConfig(
      includeDetections: dict["includeDetections"] as? Bool ?? true,
      includeClassifications: dict["includeClassifications"] as? Bool ?? true,
      includeProcessingTimeMs: dict["includeProcessingTimeMs"] as? Bool ?? true,
      includeFps: dict["includeFps"] as? Bool ?? true,
      includeMasks: dict["includeMasks"] as? Bool ?? false,
      includePoses: dict["includePoses"] as? Bool ?? false,
      includeOBB: dict["includeOBB"] as? Bool ?? false,
      includeOriginalImage: dict["includeOriginalImage"] as? Bool ?? false,
      maxFPS: {
        if let maxFPS = dict["maxFPS"] as? Int { return maxFPS }
        if let maxFPS = dict["maxFPS"] as? Double { return Int(maxFPS) }
        if let maxFPS = dict["maxFPS"] as? String { return Int(maxFPS) }
        return nil
      }(),
      throttleIntervalMs: {
        if let throttleMs = dict["throttleIntervalMs"] as? Int { return throttleMs }
        if let throttleMs = dict["throttleIntervalMs"] as? Double { return Int(throttleMs) }
        if let throttleMs = dict["throttleIntervalMs"] as? String { return Int(throttleMs) }
        return nil
      }(),
      inferenceFrequency: {
        if let freq = dict["inferenceFrequency"] as? Int { return freq }
        if let freq = dict["inferenceFrequency"] as? Double { return Int(freq) }
        if let freq = dict["inferenceFrequency"] as? String { return Int(freq) }
        return nil
      }(),
      skipFrames: {
        if let skip = dict["skipFrames"] as? Int { return skip }
        if let skip = dict["skipFrames"] as? Double { return Int(skip) }
        if let skip = dict["skipFrames"] as? String { return Int(skip) }
        return nil
      }()
    )
  }
}
