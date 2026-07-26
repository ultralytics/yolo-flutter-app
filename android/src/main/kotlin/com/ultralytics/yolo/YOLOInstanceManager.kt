// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages multiple YOLO instances with unique IDs
 */
object YOLOInstanceManager {
    private const val TAG = "YOLOInstanceManager"

    // Singleton access
    val shared: YOLOInstanceManager = this

    // Store YOLO instances by their ID
    private val instances = ConcurrentHashMap<String, YOLO>()

    // Store classifier options per instance
    private val instanceOptions = ConcurrentHashMap<String, Map<String, Any>>()

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
        numItemsThreshold: Int = 30,
        classifierOptions: Map<String, Any>?,
        callback: (Result<Unit>) -> Unit
    ) {
        try {
            instances.computeIfAbsent(instanceId) {
                classifierOptions?.let { options -> instanceOptions[instanceId] = options }
                YOLO(context, modelPath, task, emptyList(), useGpu, numItemsThreshold, classifierOptions)
            }
            callback(Result.success(Unit))
        } catch (e: Exception) {
            instanceOptions.remove(instanceId)
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

        return synchronized(yolo) {
            val originalConfThreshold = yolo.getConfidenceThreshold()
            val originalIouThreshold = yolo.getIouThreshold()
            confidenceThreshold?.let { yolo.setConfidenceThreshold(it) }
            iouThreshold?.let { yolo.setIouThreshold(it) }
            try {
                yolo.predict(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Prediction failed for instance $instanceId: ${e.message}")
                null
            } finally {
                yolo.setConfidenceThreshold(originalConfThreshold)
                yolo.setIouThreshold(originalIouThreshold)
            }
        }
    }

    fun predictorInstance(instanceId: String){
        instances[instanceId]?.let { yolo ->
            yolo.predictorInstance();
        }
    }

    /**
     * Disposes a specific instance
     */
    fun dispose(instanceId: String) {
        instances.computeIfPresent(instanceId) { _, yolo ->
            yolo.close()
            null
        }
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
