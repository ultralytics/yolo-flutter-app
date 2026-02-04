// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.net.URL
import kotlin.math.max

/**
 * Unified YOLO class that can handle different tasks (detection)
 */
class YOLO(
    private val context: Context,
    private val modelPath: String,
    val task: YOLOTask,
    private val labels: List<String> = emptyList(),
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30,
    private val classifierOptions: Map<String, Any>? = null
) {
    private val TAG = "YOLO"

    // The underlying predictor that will be initialized based on the task
    /**
     * Create optimized TFLite Interpreter options
     */
    /**
     * Create custom options for TFLite interpreters
     * Note: each predictor will handle these options differently
     */
    private fun createCustomOptions(): Interpreter.Options? {
        return try {
            Interpreter.Options().apply {
                // Use all available CPU cores for maximum parallelism
                setNumThreads(Runtime.getRuntime().availableProcessors())
                
                // Allow FP16 precision for faster computation
                setAllowFp16PrecisionForFp32(true)
                
                // Log configuration
                Log.d(TAG, "Interpreter options: threads=${Runtime.getRuntime().availableProcessors()}, FP16 enabled")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error creating interpreter options: ${e.message}")
            null
        }
    }

    private val predictor: Predictor by lazy {
        val options = createCustomOptions()
        when (task) {
            YOLOTask.DETECT -> ObjectDetector(context, modelPath, labels, useGpu, numItemsThreshold = numItemsThreshold, customOptions = options)
        }
    }

    /**
     * This method is used to directly instantiate the predictor to avoid lazy invocation.
     */
    fun predictorInstance(): Predictor {
        return predictor
    }

    /**
     * Operator function that enables calling the YOLO instance like a function (bitmap input)
     * Example: val result = model(bitmap)
     * @param bitmap The bitmap to process
     * @param rotateForCamera Set to true if this is a camera feed bitmap that needs rotation
     */
    operator fun invoke(bitmap: Bitmap, rotateForCamera: Boolean = false): YOLOResult {
        return predict(bitmap, rotateForCamera)
    }

    /**
     * Predict using a bitmap input
     * @param bitmap The bitmap to process
     * @param rotateForCamera Whether to rotate the image for camera processing, defaults to false for standard bitmap inference
     */
    fun predict(bitmap: Bitmap, rotateForCamera: Boolean = false): YOLOResult {
        val result = predictor.predict(bitmap, bitmap.width, bitmap.height, rotateForCamera, isLandscape = false)
        
        val annotatedImage = drawAnnotations(bitmap, result, rotateForCamera)
        
        return result.copy(
            originalImage = bitmap,
            annotatedImage = annotatedImage
        )
    }

    /**
     * Predict using a local image Uri
     * No rotation is applied (single image processing)
     */
    fun predict(imageUri: Uri): YOLOResult? {
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
            val result = predictor.predict(bitmap, bitmap.width, bitmap.height, rotateForCamera = false, isLandscape = false)
            return result.copy(
                originalImage = bitmap,
                annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = false)
            )
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load image from Uri: ${e.message}")
            return null
        }
    }

    /**
     * Predict using a remote image URL (suspending function for network operations)
     * No rotation is applied (single image processing)
     */
    suspend fun predict(imageUrl: String): YOLOResult? = withContext(Dispatchers.IO) {
        try {
            val bitmap = BitmapFactory.decodeStream(URL(imageUrl).openStream())
            val result = predictor.predict(bitmap, bitmap.width, bitmap.height, rotateForCamera = false, isLandscape = false)
            return@withContext result.copy(
                originalImage = bitmap,
                annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = false)
            )
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load image from URL: ${e.message}")
            return@withContext null
        }
    }

    /**
     * Calculate smart label position that ensures the label stays within screen bounds
     * @param boxRect The bounding box rectangle
     * @param labelWidth The width of the label
     * @param labelHeight The height of the label
     * @param viewWidth The width of the view/canvas
     * @param viewHeight The height of the view/canvas
     * @return The adjusted rectangle for the label
     */
    private fun calculateSmartLabelRect(
        boxRect: RectF,
        labelWidth: Float,
        labelHeight: Float,
        viewWidth: Float,
        viewHeight: Float
    ): RectF {
        // Initial position: above the box
        var labelLeft = boxRect.left
        var labelTop = boxRect.top - labelHeight
        var labelRight = labelLeft + labelWidth
        var labelBottom = boxRect.top
        
        // Check top boundary
        if (labelTop < 0) {
            // Place inside top of box
            labelTop = boxRect.top
            labelBottom = labelTop + labelHeight
        }
        
        // Check left boundary
        if (labelLeft < 0) {
            labelLeft = 0f
            labelRight = labelWidth
        }
        
        // Check right boundary
        if (labelRight > viewWidth) {
            labelRight = viewWidth
            labelLeft = labelRight - labelWidth
            // If still too wide, align with box's right edge
            if (labelLeft < 0) {
                labelLeft = maxOf(0f, boxRect.right - labelWidth)
            }
        }
        
        // Check bottom boundary
        if (labelBottom > viewHeight) {
            labelBottom = viewHeight
            labelTop = labelBottom - labelHeight
        }
        
        return RectF(labelLeft, labelTop, labelRight, labelBottom)
    }

    /**
     * Draw annotations on the input image based on the result type
     * @param bitmap Input bitmap to annotate
     * @param result YOLOResult containing detection results
     * @param rotateForCamera If true, rotate the image 90 degrees (for camera feed), otherwise don't rotate
     */
    private fun drawAnnotations(bitmap: Bitmap, result: YOLOResult, rotateForCamera: Boolean = true): Bitmap {
        // If the result already contains an annotated image, return it
        if (result.annotatedImage != null) {
            return result.annotatedImage
        }

        val output: Bitmap
        val canvas: Canvas

        if (rotateForCamera) {
            // Camera feed: rotate 90 degrees as before
            val matrix = Matrix().apply {
                // Rotate 90 degrees clockwise
                postRotate(90f)
            }
            val rotatedBitmap = Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.width,
                bitmap.height,
                matrix,
                true
            )

            // Draw on rotated bitmap
            output = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)
            canvas = Canvas(output)
        } else {
            // Single image: no rotation needed
            output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            canvas = Canvas(output)
        }

        // Calculate appropriate line thickness and text size based on image dimensions
        val maxDimension = max(output.width, output.height)
        val lineThickness = maxDimension / 200f  // Adaptive line thickness
        val calculatedTextSize = maxDimension / 40f  // Adaptive text size

        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = lineThickness.coerceAtLeast(3f) // Minimum 3f
            textSize = calculatedTextSize.coerceAtLeast(40f) // Minimum 40f
        }

        // Helper function for coordinate transformation
        // Transform from original coordinates to rotated coordinates (only when needed)
        fun transformRect(rect: RectF): RectF {
            if (!rotateForCamera) {
                // Return as-is if no rotation
                return rect
            }
            
            // Get dimensions of original and rotated images
            val originalWidth = bitmap.width.toFloat()
            val originalHeight = bitmap.height.toFloat()

            // Coordinate transformation after 90-degree rotation
            // x' = y, y' = width - x
            return RectF(
                rect.top,                    // new left = original top
                originalWidth - rect.right,  // new top = distance from original right edge
                rect.bottom,                 // new right = original bottom
                originalWidth - rect.left    // new bottom = distance from original left edge
            )
        }

        when (task) {
            YOLOTask.DETECT -> {
                // Draw bounding boxes
                for ((i, box) in result.boxes.withIndex()) {
                    paint.color = ultralyticsColors[box.index % ultralyticsColors.size]

                    // Transform coordinates
                    val transformedRect = transformRect(box.xywh)
                    // Draw rounded rectangle with corner radius
                    val cornerRadius = 12f
                    canvas.drawRoundRect(transformedRect, cornerRadius, cornerRadius, paint)

                    // Draw label with background
                    val labelText = "${box.cls} ${(box.conf * 100).toInt()}%"
                    val labelPadding = 8f
                    
                    // Measure text
                    val textBounds = Rect()
                    paint.getTextBounds(labelText, 0, labelText.length, textBounds)
                    
                    // Calculate label size
                    val labelWidth = textBounds.width() + labelPadding * 2
                    val labelHeight = textBounds.height() + labelPadding * 2
                    
                    // Calculate smart label position
                    val labelRect = calculateSmartLabelRect(
                        transformedRect,
                        labelWidth,
                        labelHeight,
                        output.width.toFloat(),
                        output.height.toFloat()
                    )
                    
                    // Draw label background
                    paint.style = Paint.Style.FILL
                    canvas.drawRoundRect(labelRect, cornerRadius, cornerRadius, paint)
                    
                    // Draw label text in white
                    paint.color = Color.WHITE
                    canvas.drawText(
                        labelText,
                        labelRect.left + labelPadding,
                        labelRect.bottom - labelPadding,
                        paint
                    )
                    
                    // Reset paint for next box
                    paint.style = Paint.Style.STROKE
                }
            }
        }

        return output
    }

    /**
     * Set confidence threshold for detection
     */
    fun setConfidenceThreshold(threshold: Double) {
        predictor.setConfidenceThreshold(threshold)
    }
    
    /**
     * Set confidence threshold for detection (Float overload)
     */
    fun setConfidenceThreshold(threshold: Float) {
        predictor.setConfidenceThreshold(threshold.toDouble())
    }
    
    /**
     * Get current confidence threshold
     */
    fun getConfidenceThreshold(): Float {
        return predictor.getConfidenceThreshold().toFloat()
    }

    /**
     * Set IoU threshold for non-maximum suppression
     */
    fun setIouThreshold(threshold: Double) {
        predictor.setIouThreshold(threshold)
    }
    
    /**
     * Set IoU threshold for non-maximum suppression (Float overload)
     */
    fun setIouThreshold(threshold: Float) {
        predictor.setIouThreshold(threshold.toDouble())
    }
    
    /**
     * Get current IoU threshold
     */
    fun getIouThreshold(): Float {
        return predictor.getIouThreshold().toFloat()
    }

    /**
     * Set maximum number of detections
     */
    fun setNumItemsThreshold(max: Int) {
        predictor.setNumItemsThreshold(max)
    }
}