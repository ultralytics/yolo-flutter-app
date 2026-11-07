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
import java.io.IOException
import java.net.URL
import kotlin.math.max

/**
 * Unified YOLO class that can handle detection, segmentation, depth, classification, pose, and OBB tasks.
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

    // The underlying predictor that will be initialized based on the task. LiteRT 2.x CompiledModel handles thread/
    // accelerator configuration internally (see LiteRtModel), so there are no per-interpreter options to pass.
    private var closed = false
    private val predictorDelegate = lazy(LazyThreadSafetyMode.NONE) {
        Predictor.create(context, modelPath, task, labels, useGpu, numItemsThreshold, classifierOptions) as BasePredictor
    }
    private val predictor: BasePredictor
        get() {
            check(!closed) { "YOLO instance is closed" }
            return predictorDelegate.value
        }

    /**
     * This method is used to directly instantiate the predictor to avoid lazy invocation.
     */
    @Synchronized
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
    @Synchronized
    fun predict(bitmap: Bitmap, rotateForCamera: Boolean = false): YOLOResult {
        predictor.includeRawMaskData = true
        val result = predictor.predict(bitmap, bitmap.width, bitmap.height, rotateForCamera, isLandscape = false)
        
        // Don't create annotated image for classification tasks to save memory and processing time
        val annotatedImage = if (task == YOLOTask.CLASSIFY) {
            null
        } else {
            drawAnnotations(bitmap, result, rotateForCamera)
        }
        
        return result.copy(
            originalImage = bitmap,
            annotatedImage = annotatedImage,
            accelerator = predictor.accelerator
        )
    }

    /**
     * Predict using an ImageProxy (from CameraX)
     * Always applies rotation for camera feed
     */
    @Synchronized
    fun predict(imageProxy: ImageProxy): YOLOResult? {
        val bitmap = ImageUtils.toBitmap(imageProxy) ?: return null
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val isRotated = rotationDegrees % 180 != 0
        val orientedWidth = if (isRotated) imageProxy.height else imageProxy.width
        val orientedHeight = if (isRotated) imageProxy.width else imageProxy.height
        predictor.cameraRotationDegrees = rotationDegrees
        predictor.includeRawMaskData = false
        val result = predictor.predict(
            bitmap,
            orientedWidth,
            orientedHeight,
            rotateForCamera = true,
            isLandscape = false
        )
        return result.copy(
            originalImage = bitmap,
            annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = true, rotationDegrees),
            accelerator = predictor.accelerator
        )
    }

    /**
     * Predict using a local image Uri
     * No rotation is applied (single image processing)
     */
    @Synchronized
    fun predict(imageUri: Uri): YOLOResult? {
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri)
            predictor.includeRawMaskData = true
            val result = predictor.predict(bitmap, bitmap.width, bitmap.height, rotateForCamera = false, isLandscape = false)
            return result.copy(
                originalImage = bitmap,
                annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = false),
                accelerator = predictor.accelerator
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
            return@withContext synchronized(this@YOLO) {
                predictor.includeRawMaskData = true
                val result = predictor.predict(bitmap, bitmap.width, bitmap.height, rotateForCamera = false, isLandscape = false)
                result.copy(
                    originalImage = bitmap,
                    annotatedImage = drawAnnotations(bitmap, result, rotateForCamera = false),
                    accelerator = predictor.accelerator
                )
            }
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
            textSize = calculatedTextSize.coerceAtLeast(LABEL_TEXT_SIZE)
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

        val colors = predictor.colors
        val cornerRadius = 12f
        val labelPadding = 8f
        val textBounds = Rect()

        // Box and label backgrounds keep 0.6 opacity; the label text color follows the class color's luma.
        fun drawLabel(labelText: String, bounds: RectF, color: Int) {
            paint.getTextBounds(labelText, 0, labelText.length, textBounds)
            val labelRect = calculateSmartLabelRect(
                bounds,
                textBounds.width() + labelPadding * 2,
                textBounds.height() + labelPadding * 2,
                output.width.toFloat(),
                output.height.toFloat()
            )
            paint.style = Paint.Style.FILL
            paint.color = color
            paint.alpha = 153
            canvas.drawRoundRect(labelRect, cornerRadius, cornerRadius, paint)
            paint.color = labelTextColor(color)
            canvas.drawText(labelText, labelRect.left + labelPadding, labelRect.bottom - labelPadding, paint)
            paint.style = Paint.Style.STROKE
        }

        fun drawBoxes() {
            for (box in result.boxes) {
                val color = classColor(box.index, colors)
                val rect = transformRect(box.xywh)
                paint.color = color
                paint.alpha = 153
                canvas.drawRoundRect(rect, cornerRadius, cornerRadius, paint)
                drawLabel("${box.cls} ${(box.conf * 100).toInt()}%", rect, color)
            }
        }

        when (task) {
            YOLOTask.DETECT -> drawBoxes()
            YOLOTask.SEGMENT -> {
                drawBoxes()
                // Overlay segmentation mask if available
                result.masks?.combinedMask?.let { mask ->
                    val maskScaled = Bitmap.createScaledBitmap(
                        mask,
                        output.width,
                        output.height,
                        false
                    )
                    paint.style = Paint.Style.FILL
                    paint.alpha = 128
                    paint.isFilterBitmap = false
                    canvas.drawBitmap(maskScaled, 0f, 0f, paint)
                    if (maskScaled !== mask) maskScaled.recycle()
                }
            }
            YOLOTask.SEMANTIC, YOLOTask.DEPTH -> {
                (result.semanticMask?.maskImage ?: result.depthMap?.image)?.let { mask ->
                    val maskScaled = Bitmap.createScaledBitmap(mask, output.width, output.height, task == YOLOTask.DEPTH)
                    paint.style = Paint.Style.FILL
                    paint.alpha = if (task == YOLOTask.DEPTH) 153 else 128
                    paint.isFilterBitmap = task == YOLOTask.DEPTH
                    canvas.drawBitmap(maskScaled, 0f, 0f, paint)
                    paint.alpha = 255
                    if (maskScaled !== mask) maskScaled.recycle()
                }
            }
            YOLOTask.CLASSIFY -> {
                result.probs?.let { probs ->
                    paint.textSize = LABEL_TEXT_SIZE
                    var labelTop = 20f
                    for ((i, cls) in probs.top5Labels.withIndex()) {
                        val labelText = "$cls ${"%.1f".format(probs.top5Confs[i] * 100)}"
                        val color = classColor(probs.top5Indices.getOrNull(i) ?: probs.top1Index, colors)
                        paint.getTextBounds(labelText, 0, labelText.length, textBounds)
                        val labelRect = RectF(
                            20f,
                            labelTop,
                            20f + textBounds.width() + labelPadding * 2,
                            labelTop + textBounds.height() + labelPadding * 2
                        )
                        paint.style = Paint.Style.FILL
                        paint.color = color
                        paint.alpha = 153
                        canvas.drawRoundRect(labelRect, cornerRadius, cornerRadius, paint)
                        paint.color = labelTextColor(color)
                        canvas.drawText(labelText, labelRect.left + labelPadding, labelRect.bottom - labelPadding, paint)
                        labelTop = labelRect.bottom + 10f
                    }
                }
            }
            YOLOTask.POSE -> {
                drawBoxes()
                for (keypoints in result.keypointsList) {
                    val points = keypoints.xy.mapIndexed { i, (x, y) ->
                        if (keypoints.conf[i] > 0.25f) transformPoint(x, y) else null
                    }
                    drawPose(canvas, points, paint, radius = 8f, lineWidth = 2f)
                }
            }
            YOLOTask.OBB -> {
                for (obbResult in result.obb) {
                    val color = classColor(obbResult.index, colors)
                    paint.color = color
                    paint.alpha = 153
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
                        val bounds = RectF(poly.minOf { it.x }, poly.minOf { it.y }, poly.maxOf { it.x }, poly.maxOf { it.y })
                        drawLabel("${obbResult.cls} ${"%.2f".format(obbResult.confidence * 100)}%", bounds, color)
                    }
                }
            }
        }

        return output
    }

    /**
     * Set confidence threshold for detection
     */
    @Synchronized
    fun setConfidenceThreshold(threshold: Double) {
        predictor.setConfidenceThreshold(threshold)
    }
    
    /**
     * Set confidence threshold for detection (Float overload)
     */
    @Synchronized
    fun setConfidenceThreshold(threshold: Float) {
        predictor.setConfidenceThreshold(threshold.toDouble())
    }
    
    /**
     * Get current confidence threshold
     */
    @Synchronized
    fun getConfidenceThreshold(): Float {
        return predictor.getConfidenceThreshold().toFloat()
    }

    /**
     * Set IoU threshold for non-maximum suppression
     */
    @Synchronized
    fun setIouThreshold(threshold: Double) {
        predictor.setIouThreshold(threshold)
    }
    
    /**
     * Set IoU threshold for non-maximum suppression (Float overload)
     */
    @Synchronized
    fun setIouThreshold(threshold: Float) {
        predictor.setIouThreshold(threshold.toDouble())
    }
    
    /**
     * Get current IoU threshold
     */
    @Synchronized
    fun getIouThreshold(): Float {
        return predictor.getIouThreshold().toFloat()
    }

    /**
     * Set maximum number of detections
     */
    @Synchronized
    fun setNumItemsThreshold(max: Int) {
        predictor.setNumItemsThreshold(max)
    }

    @Synchronized
    fun close() {
        closed = true
        if (predictorDelegate.isInitialized()) predictorDelegate.value.close()
    }
}
