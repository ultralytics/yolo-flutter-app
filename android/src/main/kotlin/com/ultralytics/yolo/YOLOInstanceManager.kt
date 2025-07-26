// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Manages multiple YOLO instances with unique IDs
 */
class YOLOInstanceManager {
    companion object {
        private const val TAG = "YOLOInstanceManager"
        
        @JvmStatic
        val shared = YOLOInstanceManager()
    }
    
    private val instances = mutableMapOf<String, YOLO>()
    private val loadingStates = mutableMapOf<String, Boolean>()
    private val instanceOptions = mutableMapOf<String, Map<String, Any>>()
    
    init {
        // Initialize default instance for backward compatibility
        createInstance("default")
    }
    
    /**
     * Creates a new instance placeholder with the given ID
     */
    fun createInstance(instanceId: String) {
        loadingStates[instanceId] = false
        Log.d(TAG, "Created instance placeholder: $instanceId")
    }
    
    /**
     * Gets a YOLO instance by ID
     */
    fun getInstance(instanceId: String): YOLO? {
        return instances[instanceId]
    }
    
    /**
     * Loads a model for a specific instance
     */
    fun loadModel(
        instanceId: String,
        context: Context,
        modelPath: String,
        task: YOLOTask,
        callback: (Result<Unit>) -> Unit
    ) {
        loadModel(instanceId, context, modelPath, task, null, callback)
    }
    
    /**
     * Loads a model for a specific instance with classifier options
     */
    fun loadModel(
        instanceId: String,
        context: Context,
        modelPath: String,
        task: YOLOTask,
        classifierOptions: Map<String, Any>?,
        callback: (Result<Unit>) -> Unit
    ) {
        // Check if already loaded
        if (instances[instanceId] != null) {
            callback(Result.success(Unit))
            return
        }
        
        // Check if loading
        if (loadingStates[instanceId] == true) {
            Log.w(TAG, "Model is already loading for instance: $instanceId")
            callback(Result.failure(Exception("Model is already loading")))
            return
        }
        
        // Start loading
        loadingStates[instanceId] = true
        
        try {
            // Store classifier options if provided
            classifierOptions?.let { options ->
                instanceOptions[instanceId] = options
                Log.d(TAG, "Stored classifier options for instance $instanceId: $options")
            }
            
            // Get labels from model metadata or use default
            val yolo = YOLO(context, modelPath, task, emptyList(), true, classifierOptions)
            instances[instanceId] = yolo
            loadingStates[instanceId] = false
            Log.d(TAG, "Model loaded successfully for instance: $instanceId ${if (classifierOptions != null) "with classifier options" else ""}")
            callback(Result.success(Unit))
        } catch (e: Exception) {
            loadingStates[instanceId] = false
            instanceOptions.remove(instanceId) // Clean up options on failure
            Log.e(TAG, "Failed to load model for instance $instanceId: ${e.message}")
            callback(Result.failure(e))
        }
    }
    
    /**
     * Runs inference on a specific instance
     */
    fun predict(
        instanceId: String,
        bitmap: Bitmap,
        confidenceThreshold: Float? = null,
        iouThreshold: Float? = null
    ): YOLOResult? {
        val yolo = instances[instanceId] ?: return null
        
        // Store original thresholds
        val originalConfThreshold = yolo.getConfidenceThreshold()
        val originalIouThreshold = yolo.getIouThreshold()
        
        // Apply custom thresholds if provided
        confidenceThreshold?.let { yolo.setConfidenceThreshold(it) }
        iouThreshold?.let { yolo.setIouThreshold(it) }
        
        // Run prediction
        val result = yolo.predict(bitmap)
        
        // Restore original thresholds
        yolo.setConfidenceThreshold(originalConfThreshold)
        yolo.setIouThreshold(originalIouThreshold)
        
        return result
    }
    
    /**
     * Removes an instance
     */
    fun removeInstance(instanceId: String) {
        instances.remove(instanceId)
        loadingStates.remove(instanceId)
        instanceOptions.remove(instanceId)
        Log.d(TAG, "Removed instance: $instanceId")
    }
    
    /**
     * Gets all active instance IDs
     */
    fun getActiveInstanceIds(): List<String> {
        return instances.keys.toList()
    }
    
    /**
     * Checks if an instance exists
     */
    fun hasInstance(instanceId: String): Boolean {
        return instances.containsKey(instanceId)
    }
    
    /**
     * Gets classifier options for a specific instance
     */
    fun getClassifierOptions(instanceId: String): Map<String, Any>? {
        return instanceOptions[instanceId]
    }
    
    /**
     * Clears all instances
     */
    fun clearAll() {
        instances.clear()
        loadingStates.clear()
        instanceOptions.clear()
        Log.d(TAG, "Cleared all instances")
    }
}