// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.util.concurrent.ConcurrentHashMap
import kotlin.coroutines.EmptyCoroutineContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Manages multiple YOLO instances with unique IDs
 */
class YOLOInstanceManager {
    private val tag = "YOLOInstanceManager"

    // Store YOLO instances by their ID
    private val instances = ConcurrentHashMap<String, YOLO>()

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
            instances.putIfAbsent(
                instanceId,
                YOLO(context, modelPath, task, emptyList(), useGpu, numItemsThreshold, classifierOptions)
            )
            callback(Result.success(Unit))
        } catch (e: Exception) {
            Log.e(tag, "Failed to load model for instance $instanceId: ${e.message}")
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
            Log.e(tag, "No model loaded for instance: $instanceId")
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
                Log.e(tag, "Prediction failed for instance $instanceId: ${e.message}")
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
    suspend fun dispose(instanceId: String) {
        val yolo = instances.remove(instanceId) ?: return
        withContext(Dispatchers.IO) { yolo.close() }
    }

    /**
     * Disposes all instances
     */
    fun disposeAll() {
        val all = instances.values.toList()
        instances.clear()
        if (all.isNotEmpty()) Dispatchers.IO.dispatch(EmptyCoroutineContext) { all.forEach(YOLO::close) }
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
     * Clears all instances
     */
    fun clearAll() {
        disposeAll()
    }
}
