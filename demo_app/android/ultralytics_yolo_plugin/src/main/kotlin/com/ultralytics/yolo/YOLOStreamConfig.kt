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
    
    // Task-specific advanced data - default to false for performance
    val includeMasks: Boolean = false,
    val includePoses: Boolean = false,
    val includeOBB: Boolean = false,
    
    // Original image data (uses ImageProxy bitmap reuse - no additional conversion needed)
    val includeOriginalImage: Boolean = false,
    
    // Performance controls
    val maxFPS: Int? = null,              // Limit inference to max FPS (e.g., 15, 30)
    val throttleIntervalMs: Int? = null,  // Minimum interval between inferences in milliseconds
    
    // Inference frequency controls
    val inferenceFrequency: Int? = null,  // Target inference frequency in FPS (e.g., 5, 10, 15, 30)
    val skipFrames: Int? = null           // Skip frames between inferences (alternative to inferenceFrequency)
    
    // Note: annotatedImage is intentionally excluded for YOLOView
    // YOLOView uses Canvas drawing (real-time overlay), not bitmap generation
) {
    companion object {
        /**
         * Preset configurations for common use cases
         */
        
        /** Default minimal configuration - optimized for maximum performance */
        val DEFAULT = YOLOStreamConfig()  // Uses all default values (minimal data)
        
        /** Full features configuration - includes all detection features */
        val FULL = YOLOStreamConfig(
            includeDetections = true,
            includeClassifications = true,
            includeProcessingTimeMs = true,
            includeFps = true,
            includeMasks = true,
            includePoses = true,
            includeOBB = true,
            includeOriginalImage = false,
            maxFPS = null  // No limit, but will be slower due to data processing
        )
        
        /** Debug configuration - includes everything for development */
        val DEBUG = YOLOStreamConfig(
            includeDetections = true,
            includeClassifications = true,
            includeProcessingTimeMs = true,
            includeFps = true,
            includeMasks = true,
            includePoses = true,
            includeOBB = true,
            includeOriginalImage = true,
            maxFPS = 10  // Limited FPS due to heavy data
        )
        
        /** Custom builder for specific needs */
        fun custom(
            includeMasks: Boolean = false,
            includePoses: Boolean = false,
            includeOBB: Boolean = false,
            includeOriginalImage: Boolean = false,
            maxFPS: Int? = null,
            throttleIntervalMs: Int? = null,
            inferenceFrequency: Int? = null,
            skipFrames: Int? = null
        ) = YOLOStreamConfig(
            includeMasks = includeMasks,
            includePoses = includePoses,
            includeOBB = includeOBB,
            includeOriginalImage = includeOriginalImage,
            maxFPS = maxFPS,
            throttleIntervalMs = throttleIntervalMs,
            inferenceFrequency = inferenceFrequency,
            skipFrames = skipFrames
        )
    }
}