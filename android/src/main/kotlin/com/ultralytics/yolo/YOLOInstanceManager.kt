// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Manages multiple YOLO instances with unique IDs
 */
object YOLOInstanceManager {
    private const val TAG = "YOLOInstanceManager"
    
    // Singleton access
    val shared: YOLOInstanceManager = this
    
    // Store YOLO instances by their ID
    private val instances = mutableMapOf<String, YOLO>()
    
    // Store loading states to prevent multiple concurrent loads
    private val loadingStates = mutableMapOf<String, Boolean>()
    
    // Store classifier options per instance
    private val instanceOptions = mutableMapOf<String, Map<String, Any>>()
    
    init {
        // Initialize default instance for backward compatibility
        createInstance("default")
    }
    
    /**
     * Creates a new instance placeholder
     */
    fun createInstance(instanceId: String) {
        // Just register the ID, actual YOLO instance created on load
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
     * Loads a model for a specific instance (overload without useGpu for backward compatibility)
     */
    fun loadModel(
        instanceId: String,
        context: Context,
        modelPath: String,
        task: YOLOTask,
        callback: (Result<Unit>) -> Unit
    ) {
        // Call the main implementation with default useGpu = true
        loadModel(
            instanceId = instanceId,
            context = context,
            modelPath = modelPath,
            task = task,
            useGpu = true,
            classifierOptions = null,
            callback = callback
        )
    }
    
    /**
     * Loads a model for a specific instance with GPU control and classifier options
     */
    fun loadModel(
        instanceId: String,
        context: Context,
        modelPath: String,
        task: YOLOTask,
        useGpu: Boolean = true,
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
            
            // Create YOLO instance with the specified parameters
            val yolo = YOLO(context, modelPath, task, emptyList(), useGpu, classifierOptions)
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
        val yolo = instances[instanceId] ?: run {
            Log.e(TAG, "No model loaded for instance: $instanceId")
            return null
        }
        
        // Store original thresholds
        val originalConfThreshold = yolo.getConfidenceThreshold()
        val originalIouThreshold = yolo.getIouThreshold()
        
        // Apply custom thresholds if provided
        confidenceThreshold?.let { yolo.setConfidenceThreshold(it) }
        iouThreshold?.let { yolo.setIouThreshold(it) }
        
        return try {
            val result = yolo.predict(bitmap)
            
            // Restore original thresholds
            yolo.setConfidenceThreshold(originalConfThreshold)
            yolo.setIouThreshold(originalIouThreshold)
            
            result
        } catch (e: Exception) {
            Log.e(TAG, "Prediction failed for instance $instanceId: ${e.message}")
            
            // Restore thresholds even on error
            yolo.setConfidenceThreshold(originalConfThreshold)
            yolo.setIouThreshold(originalIouThreshold)
            
            null
        }
    }
    
    /**
     * Disposes a specific instance
     */
    fun dispose(instanceId: String) {
        instances[instanceId]?.let { yolo ->
            try {
                // YOLO class doesn't have a close() method, just remove from map
                Log.d(TAG, "Disposing instance: $instanceId")
            } catch (e: Exception) {
                Log.e(TAG, "Error disposing instance $instanceId: ${e.message}")
            }
        }
        instances.remove(instanceId)
        loadingStates.remove(instanceId)
        instanceOptions.remove(instanceId)
    }
    
    /**
     * Removes an instance (alias for dispose for compatibility)
     */
    fun removeInstance(instanceId: String) {
        dispose(instanceId)
    }
    
    /**
     * Disposes all instances
     */
    fun disposeAll() {
        val allIds = instances.keys.toList()
        allIds.forEach { dispose(it) }
        Log.d(TAG, "Disposed all ${allIds.size} instances")
    }
    
    /**
     * Checks if an instance exists
     */
    fun hasInstance(instanceId: String): Boolean {
        return instances.containsKey(instanceId)
    }
    
    /**
     * Gets all active instance IDs
     */
    fun getActiveInstanceIds(): List<String> {
        return instances.keys.toList()
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
        disposeAll()
    }
}