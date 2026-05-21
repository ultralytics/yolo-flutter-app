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
            YOLOTask.SEGMENT -> Segmenter(context, modelPath, labels, useGpu, numItemsThreshold = numItemsThreshold, customOptions = options)
            YOLOTask.SEMANTIC -> SemanticSegmenter(context, modelPath, labels, useGpu, customOptions = options)
            YOLOTask.CLASSIFY -> Classifier(context, modelPath, labels, useGpu, options, classifierOptions)
            YOLOTask.POSE -> PoseEstimator(context, modelPath, labels, useGpu, numItemsThreshold = numItemsThreshold, customOptions = options)
            YOLOTask.OBB -> ObbDetector(context, modelPath, labels, useGpu, numItemsThreshold = numItemsThreshold, customOptions = options)
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
        
        // Don't create annotated image for classification tasks to save memory and processing time
        val annotatedImage = if (task == YOLOTask.CLASSIFY) {
            null
        } else {
            drawAnnotations(bitmap, result, rotateForCamera)
        }
        
        return result.copy(
            originalImage = bitmap,
            annotatedImage = annotatedImage
        )
    }

    /**
     * Predict using an ImageProxy (from CameraX)
     * Always applies rotation for camera feed
     */
    fun predict(imageProxy: ImageProxy): YOLOResult? {
        val bitmap = ImageUtils.toBitmap(imageProxy) ?: return null
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val isRotated = rotationDegrees % 180 != 0
        val orientedWidth = if (isRotated) imageProxy.height else imageProxy.width
        val orientedHeight = if (isRotated) imageProxy.width else imageProxy.height
        (predictor as? BasePredictor)?.cameraRotationDegrees = rotationDegrees
        val result = predictor.predict(
            bitmap,
            orientedWidth,
            orientedHeight,
            rotateForCamera = true,
            isLandscape = false
        )
        return result.copy(
            originalImage = bitmap,
            annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = true, rotationDegrees)
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
    private fun drawAnnotations(
        bitmap: Bitmap,
        result: YOLOResult,
        rotateForCamera: Boolean = true,
        rotationDegrees: Int? = null
    ): Bitmap {
        // Static-image predictors can return a pre-rendered annotation.
        if (!rotateForCamera && result.annotatedImage != null) {
            return result.annotatedImage
        }

        val output: Bitmap
        val canvas: Canvas

        if (rotateForCamera) {
            val degrees = rotationDegrees ?: 90
            val isRotated = degrees % 180 != 0
            output = Bitmap.createBitmap(
                if (isRotated) bitmap.height else bitmap.width,
                if (isRotated) bitmap.width else bitmap.height,
                Bitmap.Config.ARGB_8888
            )
            canvas = Canvas(output)
            // Draw the camera frame in the same orientation used for inference.
            canvas.save()
            canvas.translate(output.width / 2f, output.height / 2f)
            canvas.rotate(degrees.toFloat())
            canvas.drawBitmap(bitmap, -bitmap.width / 2f, -bitmap.height / 2f, null)
            canvas.restore()
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

        val resultWidth = result.origShape.width.toFloat().takeIf { it > 0f } ?: output.width.toFloat()
        val resultHeight = result.origShape.height.toFloat().takeIf { it > 0f } ?: output.height.toFloat()
        val scaleX = output.width / resultWidth
        val scaleY = output.height / resultHeight

        fun transformPoint(x: Float, y: Float): PointF {
            return PointF(x * scaleX, y * scaleY)
        }

        // Result coordinates are already mapped into the image orientation used for inference.
        fun transformRect(rect: RectF): RectF {
            return RectF(
                rect.left * scaleX,
                rect.top * scaleY,
                rect.right * scaleX,
                rect.bottom * scaleY
            )
        }

        when (task) {
            YOLOTask.DETECT -> {
                // Draw bounding boxes
                for (box in result.boxes) {
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
            YOLOTask.SEGMENT -> {
                // Draw bounding boxes
                for (box in result.boxes) {
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
                    
                    // Reset paint style
                    paint.style = Paint.Style.STROKE
                }

                // Overlay segmentation mask if available
                result.masks?.combinedMask?.let { mask ->
                    val maskScaled = Bitmap.createScaledBitmap(
                        mask,
                        output.width,
                        output.height,
                        true
                    )
                    paint.style = Paint.Style.FILL
                    paint.alpha = 128
                    paint.isFilterBitmap = true
                    canvas.drawBitmap(maskScaled, 0f, 0f, paint)
                    if (maskScaled !== mask) maskScaled.recycle()
                }
            }
            YOLOTask.SEMANTIC -> {
                result.semanticMask?.maskImage?.let { mask ->
                    val maskScaled = Bitmap.createScaledBitmap(mask, output.width, output.height, true)
                    paint.style = Paint.Style.FILL
                    paint.alpha = 128
                    paint.isFilterBitmap = true
                    canvas.drawBitmap(maskScaled, 0f, 0f, paint)
                    paint.alpha = 255
                    if (maskScaled !== mask) maskScaled.recycle()
                }
            }
            YOLOTask.CLASSIFY -> {
                result.probs?.let { probs ->
                    paint.textSize = 40f
                    val labelPadding = 8f
                    var labelTop = 20f
                    for ((i, cls) in probs.top5Labels.withIndex()) {
                        val labelText = "$cls ${"%.1f".format(probs.top5Confs[i] * 100)}"
                        val colorIndex = probs.top5Indices.getOrNull(i) ?: probs.top1Index
                        paint.color = ultralyticsColors[colorIndex % ultralyticsColors.size]
                        val textBounds = Rect()
                        paint.getTextBounds(labelText, 0, labelText.length, textBounds)
                        val labelRect = RectF(
                            20f,
                            labelTop,
                            20f + textBounds.width() + labelPadding * 2,
                            labelTop + textBounds.height() + labelPadding * 2
                        )
                        paint.style = Paint.Style.FILL
                        canvas.drawRoundRect(labelRect, 12f, 12f, paint)
                        paint.color = Color.WHITE
                        canvas.drawText(labelText, labelRect.left + labelPadding, labelRect.bottom - labelPadding, paint)
                        labelTop = labelRect.bottom + 10f
                    }
                }
            }
            YOLOTask.POSE -> {
                // Draw bounding boxes
                for (box in result.boxes) {
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
                    
                    // Reset paint style
                    paint.style = Paint.Style.STROKE
                }

                // Draw keypoints
                for (keypoints in result.keypointsList) {
                    paint.style = Paint.Style.FILL

                    val transformedPoints = keypoints.xy.map { (x, y) ->
                        val point = transformPoint(x, y)
                        Pair(point.x, point.y)
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

                    val poly = obbResult.box.toPolygon(resultWidth, resultHeight).map {
                        transformPoint(it.x * resultWidth, it.y * resultHeight)
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

                        // Draw label with background
                        val labelText = "${obbResult.cls} ${"%.2f".format(obbResult.confidence * 100)}%"
                        val labelPadding = 8f
                        val cornerRadius = 12f
                        
                        // Measure text
                        val textBounds = Rect()
                        paint.getTextBounds(labelText, 0, labelText.length, textBounds)
                        
                        // Find bounding box of the OBB polygon
                        val minX = poly.map { it.x }.minOrNull() ?: 0f
                        val maxX = poly.map { it.x }.maxOrNull() ?: 0f
                        val minY = poly.map { it.y }.minOrNull() ?: 0f
                        val maxY = poly.map { it.y }.maxOrNull() ?: 0f
                        val obbBounds = RectF(minX, minY, maxX, maxY)
                        
                        // Calculate label size
                        val labelWidth = textBounds.width() + labelPadding * 2
                        val labelHeight = textBounds.height() + labelPadding * 2
                        
                        // Calculate smart label position
                        val labelRect = calculateSmartLabelRect(
                            obbBounds,
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
                        
                        // Reset paint
                        paint.color = ultralyticsColors[obbResult.index % ultralyticsColors.size]
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
