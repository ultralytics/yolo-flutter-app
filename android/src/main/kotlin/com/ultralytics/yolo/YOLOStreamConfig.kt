// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

/**
 * Configuration class for YOLOView streaming functionality
 * Controls what data is included in real-time streaming and performance settings
 */
data class YOLOStreamConfig(
    // Basic inference data (always useful)
    val includeDetections: Boolean = true,
    val includeClassifications: Boolean = true,
    val includeProcessingTimeMs: Boolean = true,
    val includeFps: Boolean = true,
    
    // Task-specific advanced data
    val includeMasks: Boolean = true,
    val includePoses: Boolean = true,
    val includeOBB: Boolean = true,
    
    // Original image data (uses ImageProxy bitmap reuse - no additional conversion needed)
    val includeOriginalImage: Boolean = false,
    
    // Performance controls
    val maxFPS: Int? = null,              // Limit inference to max FPS (e.g., 15, 30)
    val throttleIntervalMs: Int? = null   // Minimum interval between inferences in milliseconds
    
    // Note: annotatedImage is intentionally excluded for YOLOView
    // YOLOView uses Canvas drawing (real-time overlay), not bitmap generation
) {
    companion object {
        /**
         * Preset configurations for common use cases
         */
        
        /** Minimal data for basic object detection - highest performance */
        val MINIMAL = YOLOStreamConfig(
            includeDetections = true,
            includeClassifications = true,
            includeProcessingTimeMs = true,
            includeFps = true,
            includeMasks = false,
            includePoses = false,
            includeOBB = false,
            includeOriginalImage = false,
            maxFPS = 30
        )
        
        /** Balanced configuration for most applications */
        val BALANCED = YOLOStreamConfig(
            includeDetections = true,
            includeClassifications = true,
            includeProcessingTimeMs = true,
            includeFps = true,
            includeMasks = true,
            includePoses = true,
            includeOBB = true,
            includeOriginalImage = false,
            maxFPS = 15
        )
        
        /** Maximum data including original images - lowest performance */
        val FULL = YOLOStreamConfig(
            includeDetections = true,
            includeClassifications = true,
            includeProcessingTimeMs = true,
            includeFps = true,
            includeMasks = true,
            includePoses = true,
            includeOBB = true,
            includeOriginalImage = true,
            maxFPS = 10
        )
        
        /** Performance optimized for low-end devices */
        val PERFORMANCE = YOLOStreamConfig(
            includeDetections = true,
            includeClassifications = false,
            includeProcessingTimeMs = true,
            includeFps = true,
            includeMasks = false,
            includePoses = false,
            includeOBB = false,
            includeOriginalImage = false,
            maxFPS = 15,
            throttleIntervalMs = 100
        )
    }
}