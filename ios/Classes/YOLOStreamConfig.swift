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
    public let maxFPS: Int?              // Limit inference to max FPS (e.g., 15, 30)
    public let throttleIntervalMs: Int?   // Minimum interval between inferences in milliseconds
    
    // Note: annotatedImage is intentionally excluded for YOLOView
    // YOLOView uses CALayer drawing (real-time overlay), not UIImage generation
    
    public init(
        includeDetections: Bool = true,
        includeClassifications: Bool = true,
        includeProcessingTimeMs: Bool = true,
        includeFps: Bool = true,
        includeMasks: Bool = true,
        includePoses: Bool = true,
        includeOBB: Bool = true,
        includeOriginalImage: Bool = false,
        maxFPS: Int? = nil,
        throttleIntervalMs: Int? = nil
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
    }
    
    /// Preset configurations for common use cases
    public static let MINIMAL = YOLOStreamConfig(
        includeDetections: true,
        includeClassifications: true,
        includeProcessingTimeMs: true,
        includeFps: true,
        includeMasks: false,
        includePoses: false,
        includeOBB: false,
        includeOriginalImage: false,
        maxFPS: 30
    )
    
    /// Balanced configuration for most applications
    public static let BALANCED = YOLOStreamConfig(
        includeDetections: true,
        includeClassifications: true,
        includeProcessingTimeMs: true,
        includeFps: true,
        includeMasks: true,
        includePoses: true,
        includeOBB: true,
        includeOriginalImage: false,
        maxFPS: 15
    )
    
    /// Maximum data including original images - lowest performance
    public static let FULL = YOLOStreamConfig(
        includeDetections: true,
        includeClassifications: true,
        includeProcessingTimeMs: true,
        includeFps: true,
        includeMasks: true,
        includePoses: true,
        includeOBB: true,
        includeOriginalImage: true,
        maxFPS: 10
    )
    
    /// Performance optimized for low-end devices
    public static let PERFORMANCE = YOLOStreamConfig(
        includeDetections: true,
        includeClassifications: false,
        includeProcessingTimeMs: true,
        includeFps: true,
        includeMasks: false,
        includePoses: false,
        includeOBB: false,
        includeOriginalImage: false,
        maxFPS: 15,
        throttleIntervalMs: 100
    )
}

/// Extension to create YOLOStreamConfig from Dictionary (for Flutter integration)
extension YOLOStreamConfig {
    public static func from(dict: [String: Any]) -> YOLOStreamConfig {
        return YOLOStreamConfig(
            includeDetections: dict["includeDetections"] as? Bool ?? true,
            includeClassifications: dict["includeClassifications"] as? Bool ?? true,
            includeProcessingTimeMs: dict["includeProcessingTimeMs"] as? Bool ?? true,
            includeFps: dict["includeFps"] as? Bool ?? true,
            includeMasks: dict["includeMasks"] as? Bool ?? true,
            includePoses: dict["includePoses"] as? Bool ?? true,
            includeOBB: dict["includeOBB"] as? Bool ?? true,
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
            }()
        )
    }
}