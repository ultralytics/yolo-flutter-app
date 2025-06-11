// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.net.URL
import kotlin.math.max

/**
 * Unified YOLO class that can handle different tasks (detection, segmentation, classification, pose estimation, OBB detection)
 */
class YOLO(
    private val context: Context,
    private val modelPath: String,
    val task: YOLOTask,
    private val labels: List<String> = emptyList(),
    private val useGpu: Boolean = true
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
                
                // useGpu is handled in individual predictors
                
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
            YOLOTask.DETECT -> ObjectDetector(context, modelPath, labels, useGpu, options)
            YOLOTask.SEGMENT -> Segmenter(context, modelPath, labels, useGpu, options)
            YOLOTask.CLASSIFY -> Classifier(context, modelPath, labels, useGpu) // Classifier doesn't accept options
            YOLOTask.POSE -> PoseEstimator(context, modelPath, labels, useGpu, customOptions = options)
            YOLOTask.OBB -> ObbDetector(context, modelPath, labels, useGpu, options)
        }
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
        return result.copy(
            originalImage = bitmap,
            annotatedImage = drawAnnotations(bitmap, result, rotateForCamera)
        )
    }

    /**
     * Predict using an ImageProxy (from CameraX)
     * Always applies rotation for camera feed
     */
    fun predict(imageProxy: ImageProxy): YOLOResult? {
        val bitmap = ImageUtils.toBitmap(imageProxy) ?: return null
        val result = predictor.predict(bitmap, imageProxy.width, imageProxy.height, rotateForCamera = true, isLandscape = false)
        return result.copy(
            originalImage = bitmap,
            annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = true)
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
                    canvas.drawRoundRect(box.xywh, cornerRadius, cornerRadius, paint)

                    // Draw label with background
                    val labelText = "${box.cls} ${(box.conf * 100).toInt()}%"
                    val labelPadding = 8f
                    
                    // Measure text
                    val textBounds = Rect()
                    paint.getTextBounds(labelText, 0, labelText.length, textBounds)
                    
                    // Calculate label background position
                    val labelLeft = transformedRect.left
                    val labelTop = transformedRect.top - textBounds.height() - labelPadding * 2
                    val labelRight = labelLeft + textBounds.width() + labelPadding * 2
                    val labelBottom = transformedRect.top
                    
                    // Draw label background
                    paint.style = Paint.Style.FILL
                    val labelRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
                    canvas.drawRoundRect(labelRect, cornerRadius, cornerRadius, paint)
                    
                    // Draw label text in white
                    paint.color = Color.WHITE
                    canvas.drawText(
                        labelText,
                        labelLeft + labelPadding,
                        labelBottom - labelPadding,
                        paint
                    )
                    
                    // Reset paint for next box
                    paint.style = Paint.Style.STROKE
                }
            }
            YOLOTask.SEGMENT -> {
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
                    
                    // Calculate label background position
                    val labelLeft = transformedRect.left
                    val labelTop = transformedRect.top - textBounds.height() - labelPadding * 2
                    val labelRight = labelLeft + textBounds.width() + labelPadding * 2
                    val labelBottom = transformedRect.top
                    
                    // Draw label background
                    paint.style = Paint.Style.FILL
                    val labelRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
                    canvas.drawRoundRect(labelRect, cornerRadius, cornerRadius, paint)
                    
                    // Draw label text in white
                    paint.color = Color.WHITE
                    canvas.drawText(
                        labelText,
                        labelLeft + labelPadding,
                        labelBottom - labelPadding,
                        paint
                    )
                    
                    // Reset paint style
                    paint.style = Paint.Style.STROKE
                }

                // Overlay segmentation mask if available
                result.masks?.combinedMask?.let { mask ->
                    val maskToUse: Bitmap
                    
                    if (rotateForCamera) {
                        // Mask also needs to be rotated (camera feed)
                        val maskMatrix = Matrix().apply {
                            postRotate(90f)
                        }
                        maskToUse = Bitmap.createBitmap(
                            mask,
                            0,
                            0,
                            mask.width,
                            mask.height,
                            maskMatrix,
                            true
                        )
                    } else {
                        // No rotation for single image
                        maskToUse = mask
                    }

                    val maskScaled = Bitmap.createScaledBitmap(
                        maskToUse,
                        output.width,
                        output.height,
                        true
                    )
                    paint.style = Paint.Style.FILL
                    paint.alpha = 128
                    canvas.drawBitmap(maskScaled, 0f, 0f, paint)
                }
            }
            YOLOTask.CLASSIFY -> {
                // Draw classification result at the top
                result.probs?.let { probs ->
                    paint.style = Paint.Style.FILL
                    paint.color = Color.WHITE
                    paint.alpha = 180
                    canvas.drawRect(0f, 0f, output.width.toFloat(), 160f, paint)

                    paint.alpha = 255
                    paint.color = Color.BLACK
                    paint.textSize = 60f
                    canvas.drawText(
                        "${probs.top1} ${"%.2f".format(probs.top1Conf * 100)}%",
                        20f,
                        80f,
                        paint
                    )

                    // Draw top-5 classes
                    paint.textSize = 30f
                    for ((i, cls) in probs.top5.withIndex()) {
                        if (i == 0) continue // Skip top-1 which is already shown
                        canvas.drawText(
                            "$cls ${"%.2f".format(probs.top5Confs[i] * 100)}%",
                            20f,
                            120f + i * 40f,
                            paint
                        )
                    }
                }
            }
            YOLOTask.POSE -> {
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
                    
                    // Calculate label background position
                    val labelLeft = transformedRect.left
                    val labelTop = transformedRect.top - textBounds.height() - labelPadding * 2
                    val labelRight = labelLeft + textBounds.width() + labelPadding * 2
                    val labelBottom = transformedRect.top
                    
                    // Draw label background
                    paint.style = Paint.Style.FILL
                    val labelRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
                    canvas.drawRoundRect(labelRect, cornerRadius, cornerRadius, paint)
                    
                    // Draw label text in white
                    paint.color = Color.WHITE
                    canvas.drawText(
                        labelText,
                        labelLeft + labelPadding,
                        labelBottom - labelPadding,
                        paint
                    )
                    
                    // Reset paint style
                    paint.style = Paint.Style.STROKE
                }

                // Draw keypoints
                for (keypoints in result.keypointsList) {
                    paint.style = Paint.Style.FILL

                    // Transform and draw keypoint coordinates (only when needed)
                    val transformedPoints = keypoints.xy.map { (x, y) ->
                        if (rotateForCamera) {
                            // x' = y, y' = width - x (camera feed rotation)
                            val originalWidth = bitmap.width.toFloat()
                            Pair(y, originalWidth - x)
                        } else {
                            // No rotation for single image
                            Pair(x, y)
                        }
                    }

                    // Define keypoint color indices (same as YoloView)
                    val kptColorIndices = intArrayOf(
                        16, 16, 16, 16, 16,
                        9, 9, 9, 9, 9, 9,
                        0, 0, 0, 0, 0, 0
                    )
                    
                    // Define pose palette for coloring
                    val posePalette = arrayOf(
                        floatArrayOf(255f, 128f, 0f),
                        floatArrayOf(255f, 153f, 51f),
                        floatArrayOf(255f, 178f, 102f),
                        floatArrayOf(230f, 230f, 0f),
                        floatArrayOf(255f, 153f, 255f),
                        floatArrayOf(153f, 204f, 255f),
                        floatArrayOf(255f, 102f, 255f),
                        floatArrayOf(255f, 51f, 255f),
                        floatArrayOf(102f, 178f, 255f),
                        floatArrayOf(51f, 153f, 255f),
                        floatArrayOf(255f, 153f, 153f),
                        floatArrayOf(255f, 102f, 102f),
                        floatArrayOf(255f, 51f, 51f),
                        floatArrayOf(153f, 255f, 153f),
                        floatArrayOf(102f, 255f, 102f),
                        floatArrayOf(51f, 255f, 51f),
                        floatArrayOf(0f, 255f, 0f),
                        floatArrayOf(0f, 0f, 255f),
                        floatArrayOf(255f, 0f, 0f),
                        floatArrayOf(255f, 255f, 255f)
                    )
                    
                    // Draw keypoints with proper coloring
                    for ((i, point) in transformedPoints.withIndex()) {
                        val confidence = keypoints.conf[i]
                        if (confidence > 0.25f) {
                            // Use same color scheme as YoloView
                            val colorIdx = if (i < kptColorIndices.size) kptColorIndices[i] else 0
                            val rgbArray = posePalette[colorIdx % posePalette.size]
                            paint.color = Color.argb(
                                255,
                                rgbArray[0].toInt().coerceIn(0, 255),
                                rgbArray[1].toInt().coerceIn(0, 255),
                                rgbArray[2].toInt().coerceIn(0, 255)
                            )
                            canvas.drawCircle(point.first, point.second, 8f, paint)
                        }
                    }

                    // Draw skeleton lines using proper skeleton structure
                    paint.strokeWidth = 2f
                    
                    // Convert transformedPoints to PointF array for skeleton drawing
                    val points = Array<PointF?>(transformedPoints.size) { null }
                    for (i in transformedPoints.indices) {
                        if (keypoints.conf[i] > 0.25f) {
                            points[i] = PointF(transformedPoints[i].first, transformedPoints[i].second)
                        }
                    }
                    
                    // Define skeleton connections (same as YoloView)
                    val skeleton = arrayOf(
                        intArrayOf(16, 14),
                        intArrayOf(14, 12),
                        intArrayOf(17, 15),
                        intArrayOf(15, 13),
                        intArrayOf(12, 13),
                        intArrayOf(6, 12),
                        intArrayOf(7, 13),
                        intArrayOf(6, 7),
                        intArrayOf(6, 8),
                        intArrayOf(7, 9),
                        intArrayOf(8, 10),
                        intArrayOf(9, 11),
                        intArrayOf(2, 3),
                        intArrayOf(1, 2),
                        intArrayOf(1, 3),
                        intArrayOf(2, 4),
                        intArrayOf(3, 5),
                        intArrayOf(4, 6),
                        intArrayOf(5, 7)
                    )
                    
                    // Define color indices for limbs
                    val limbColorIndices = intArrayOf(
                        0, 0, 0, 0,
                        7, 7, 7,
                        9, 9, 9, 9, 9,
                        16, 16, 16, 16, 16, 16, 16
                    )
                    
                    // Draw skeleton connections
                    paint.style = Paint.Style.STROKE
                    for ((idx, bone) in skeleton.withIndex()) {
                        val i1 = bone[0] - 1  // 1-indexed to 0-indexed
                        val i2 = bone[1] - 1
                        val p1 = points.getOrNull(i1)
                        val p2 = points.getOrNull(i2)
                        
                        if (p1 != null && p2 != null) {
                            // Use same color scheme as YoloView
                            val limbColorIdx = if (idx < limbColorIndices.size) limbColorIndices[idx] else 0
                            val rgbArray = posePalette[limbColorIdx % posePalette.size]
                            paint.color = Color.argb(
                                255,
                                rgbArray[0].toInt().coerceIn(0, 255),
                                rgbArray[1].toInt().coerceIn(0, 255),
                                rgbArray[2].toInt().coerceIn(0, 255)
                            )
                            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, paint)
                        }
                    }
                    paint.strokeWidth = 3f
                }
            }
            YOLOTask.OBB -> {
                // Draw oriented bounding boxes
                for (obbResult in result.obb) {
                    paint.color = ultralyticsColors[obbResult.index % ultralyticsColors.size]

                    // Transform OBB polygon vertices (only when needed)
                    val poly = obbResult.box.toPolygon().map {
                        // Scale original coordinates to image size
                        val x = it.x * bitmap.width
                        val y = it.y * bitmap.height

                        if (rotateForCamera) {
                            // Rotation transformation (camera feed)
                            val originalWidth = bitmap.width.toFloat()
                            PointF(y, originalWidth - x)
                        } else {
                            // No rotation for single image
                            PointF(x, y)
                        }
                    }

                    if (poly.size >= 4) {
                        val path = Path().apply {
                            moveTo(poly[0].x, poly[0].y)
                            for (p in poly.drop(1)) {
                                lineTo(p.x, p.y)
                            }
                            close()
                        }
                        canvas.drawPath(path, paint)

                        paint.style = Paint.Style.FILL
                        canvas.drawText(
                            "${obbResult.cls} ${"%.2f".format(obbResult.confidence * 100)}%",
                            poly[0].x,
                            poly[0].y - 10,
                            paint
                        )
                        paint.style = Paint.Style.STROKE
                    }
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