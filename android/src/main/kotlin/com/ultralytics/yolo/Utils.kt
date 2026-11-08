// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import kotlin.math.abs
import kotlin.math.roundToInt

// Annotation colors, mirroring ultralytics/utils/plotting.py Colors.
internal const val OPAQUE = 0xFF000000.toInt()

/** Default class palette (opaque ARGB), indexed by class id % 20. */
private val ultralyticsColors = intArrayOf(
    0x042AFF, 0x0BDBEB, 0xF3F3F3, 0x00DFB7, 0x111F68, 0xFF6FDD, 0xFF444F, 0xCCED00, 0x00F344, 0xBD00FF,
    0x00B4FF, 0xDD00BA, 0x00FFFF, 0x26C000, 0x01FFB3, 0x7D24FF, 0x7B0068, 0xFF1B6C, 0xFC6D2F, 0xA2FF0B,
).map { it or OPAQUE }.toIntArray()

private val posePalette = intArrayOf(
    0xFF8000, 0xFF9933, 0xFFB266, 0xE6E600, 0xFF99FF, 0x99CCFF, 0xFF66FF, 0xFF33FF, 0x66B2FF, 0x3399FF,
    0xFF9999, 0xFF6666, 0xFF3333, 0x99FF99, 0x66FF66, 0x33FF33, 0x00FF00, 0x0000FF, 0xFF0000, 0xFFFFFF,
).map { it or OPAQUE }.toIntArray()

// COCO-17 keypoint and limb indices into posePalette, and the 0-indexed skeleton as (start, end) pairs.
private val kptColorIndices = intArrayOf(16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0)
private val limbColorIndices = intArrayOf(0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16)
private val skeleton = intArrayOf(
    15, 13, 13, 11, 16, 14, 14, 12, 11, 12, 5, 11, 6, 12, 5, 6, 5, 7, 6, 8,
    7, 9, 8, 10, 1, 2, 0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6,
)

/** Jet depth colormap (opaque ARGB, low = dark blue, high = dark red): `clamp(1.5 - |4t - center|, 0, 1)`. */
val depthColors = IntArray(256) { i ->
    fun channel(center: Int) = ((1.5f - abs(4f * i / 255f - center)).coerceIn(0f, 1f) * 255f).roundToInt()
    OPAQUE or (channel(3) shl 16) or (channel(2) shl 8) or channel(1)
}

/** Class color: the model's `colors` metadata entry when present, else the default palette (no boxing). */
fun classColor(index: Int, colors: List<Int>): Int =
    if (index < colors.size) colors[index] else ultralyticsColors[index % ultralyticsColors.size]

/** Label font owner: system default typeface (Roboto, regular) at this px size, the floor for scaled static labels. */
const val LABEL_TEXT_SIZE = 40f

/** Label text color on a class-colored background: #111F68 when its luma exceeds 150, else white. */
fun labelTextColor(background: Int): Int {
    val luma = 0.299 * (background shr 16 and 0xFF) + 0.587 * (background shr 8 and 0xFF) + 0.114 * (background and 0xFF)
    return if (luma > 150) 0xFF111F68.toInt() else Color.WHITE
}

/**
 * Draw one pose of [count] keypoints from [xy] (x0, y0, x1, y1, ...; x is NaN below the confidence threshold) without
 * allocating. COCO-17 layouts use the pose palette and skeleton; other layouts color keypoint k with the class palette
 * and draw no limbs.
 */
fun drawPose(canvas: Canvas, xy: FloatArray, count: Int, paint: Paint, radius: Float, lineWidth: Float) {
    val coco = count == kptColorIndices.size
    paint.style = Paint.Style.FILL
    for (k in 0 until count) {
        if (xy[2 * k].isNaN()) continue
        paint.color = if (coco) posePalette[kptColorIndices[k]] else ultralyticsColors[k % ultralyticsColors.size]
        canvas.drawCircle(xy[2 * k], xy[2 * k + 1], radius, paint)
    }
    if (!coco) return
    paint.style = Paint.Style.STROKE
    paint.strokeWidth = lineWidth
    for (j in limbColorIndices.indices) {
        val a = 2 * skeleton[2 * j]
        val b = 2 * skeleton[2 * j + 1]
        if (xy[a].isNaN() || xy[b].isNaN()) continue
        paint.color = posePalette[limbColorIndices[j]]
        canvas.drawLine(xy[a], xy[a + 1], xy[b], xy[b + 1], paint)
    }
}

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
    // Memory-maps a model file bundled in assets (in place of litert-support's FileUtil.loadMappedFile).
    private fun loadMappedAsset(context: Context, assetPath: String): java.nio.MappedByteBuffer {
        context.assets.openFd(assetPath).use { afd ->
            java.io.FileInputStream(afd.fileDescriptor).use { fis ->
                return fis.channel.map(
                    java.nio.channels.FileChannel.MapMode.READ_ONLY,
                    afd.startOffset,
                    afd.declaredLength
                )
            }
        }
    }

    fun loadModelFile(context: Context, modelPath: String): java.nio.MappedByteBuffer {
        val finalModelPath = ensureTFLiteExtension(modelPath)

        try {
            // Check if it's an absolute path and the file exists
            if (isAbsolutePath(finalModelPath) && fileExistsAtPath(finalModelPath)) {
                return loadModelFromFilesystem(finalModelPath)
            } else {
                // Try loading from assets
                return loadMappedAsset(context, finalModelPath)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model with path: $finalModelPath, error: ${e.message}")

            // If the model with extension can't be found, try the original path as a fallback
            try {
                if (isAbsolutePath(modelPath) && fileExistsAtPath(modelPath)) {
                    return loadModelFromFilesystem(modelPath)
                } else {
                    return loadMappedAsset(context, modelPath)
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
     * Ensure the model path has a model extension, appending .tflite (the default runtime) when missing.
     * QNN context-binary ONNX paths (.onnx) pass through unchanged.
     */
    fun ensureTFLiteExtension(modelPath: String): String {
        val lower = modelPath.lowercase()
        return if (!lower.endsWith(".tflite") && !lower.endsWith(".onnx")) {
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
