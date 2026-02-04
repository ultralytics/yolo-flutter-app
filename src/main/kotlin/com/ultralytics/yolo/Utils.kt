// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.support.common.FileUtil

val ultralyticsColors: List<Int> = listOf(
    Color.argb(153, 4, 42, 255),
    Color.argb(153, 11, 219, 235),
    Color.argb(153, 243, 243, 243),
    Color.argb(153, 0, 223, 183),
    Color.argb(153, 17, 31, 104),
    Color.argb(153, 255, 111, 221),
    Color.argb(153, 255, 68, 79),
    Color.argb(153, 204, 237, 0),
    Color.argb(153, 0, 243, 68),
    Color.argb(153, 189, 0, 255),
    Color.argb(153, 0, 180, 255),
    Color.argb(153, 221, 0, 186),
    Color.argb(153, 0, 255, 255),
    Color.argb(153, 38, 192, 0),
    Color.argb(153, 1, 255, 179),
    Color.argb(153, 125, 36, 255),
    Color.argb(153, 123, 0, 104),
    Color.argb(153, 255, 27, 108),
    Color.argb(153, 252, 109, 47),
    Color.argb(153, 162, 255, 11)
)

/**
 * Utility functions for YOLO operations
 */
object YOLOUtils {
    private const val TAG = "YOLOUtils"
    
    /**
     * Checks if the provided path is an absolute file path
     */
    fun isAbsolutePath(path: String): Boolean {
        return path.startsWith("/")
    }

    /**
     * Checks if a file exists at the specified absolute path
     */
    fun fileExistsAtPath(path: String): Boolean {
        val file = java.io.File(path)
        return file.exists() && file.isFile
    }
    
    /**
     * Loads a model file from either assets or the file system.
     * Supports both asset paths and absolute file system paths.
     * If the provided model path doesn't include an extension, ".tflite" will be appended.
     * 
     * @param context The application context
     * @param modelPath The model path (can be an asset path or absolute filesystem path)
     * @return ByteBuffer containing the model data
     */
    fun loadModelFile(context: Context, modelPath: String): java.nio.MappedByteBuffer {
        val finalModelPath = ensureTFLiteExtension(modelPath)
        Log.d(TAG, "Loading model from path: $finalModelPath")
        
        try {
            // Check if it's an absolute path and the file exists
            if (isAbsolutePath(finalModelPath) && fileExistsAtPath(finalModelPath)) {
                Log.d(TAG, "Loading model from absolute path: $finalModelPath")
                return loadModelFromFilesystem(finalModelPath)
            } else {
                // Try loading from assets
                Log.d(TAG, "Loading model from assets: $finalModelPath")
                return FileUtil.loadMappedFile(context, finalModelPath)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model with path: $finalModelPath, error: ${e.message}")
            
            // If the model with extension can't be found, try the original path as a fallback
            try {
                if (isAbsolutePath(modelPath) && fileExistsAtPath(modelPath)) {
                    Log.d(TAG, "Loading model from absolute path (fallback): $modelPath")
                    return loadModelFromFilesystem(modelPath)
                } else {
                    Log.d(TAG, "Loading model from assets (fallback): $modelPath")
                    return FileUtil.loadMappedFile(context, modelPath)
                }
            } catch (e2: Exception) {
                Log.e(TAG, "Failed to load model with both paths. Original error: ${e.message}, Fallback error: ${e2.message}")
                throw e2
            }
        }
    }
    
    /**
     * Loads a model file from the filesystem
     * @param filePath Absolute path to the model file
     * @return ByteBuffer containing the model data
     */
    private fun loadModelFromFilesystem(filePath: String): java.nio.MappedByteBuffer {
        val file = java.io.File(filePath)
        val fileChannel = java.io.RandomAccessFile(file, "r").channel
        val startOffset = 0L
        val declaredLength = file.length()
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Ensure the model path has a .tflite extension
     */
    fun ensureTFLiteExtension(modelPath: String): String {
        return if (!modelPath.lowercase().endsWith(".tflite")) {
            "$modelPath.tflite"
        } else {
            modelPath
        }
    }
    
    /**
     * Checks all possible paths where a model could be found
     * @param context Application context
     * @param modelPath Path to check (could be asset or absolute path)
     * @return Map containing status and resolved path
     */
    fun checkModelExistence(context: Context, modelPath: String): Map<String, Any> {
        // Try with .tflite extension
        val withExtension = ensureTFLiteExtension(modelPath)
        
        // Check absolute paths first
        if (isAbsolutePath(withExtension) && fileExistsAtPath(withExtension)) {
            return mapOf("exists" to true, "path" to withExtension, "location" to "filesystem")
        }
        
        if (isAbsolutePath(modelPath) && fileExistsAtPath(modelPath)) {
            return mapOf("exists" to true, "path" to modelPath, "location" to "filesystem")
        }
        
        // Then check assets
        try {
            // This will throw an exception if the asset doesn't exist
            context.assets.openFd(withExtension).close()
            return mapOf("exists" to true, "path" to withExtension, "location" to "assets")
        } catch (e: Exception) {
            // Asset with extension doesn't exist, try without extension
            try {
                context.assets.openFd(modelPath).close()
                return mapOf("exists" to true, "path" to modelPath, "location" to "assets")
            } catch (e2: Exception) {
                // Neither exists
                return mapOf("exists" to false, "path" to modelPath, "location" to "unknown")
            }
        }
    }
}
