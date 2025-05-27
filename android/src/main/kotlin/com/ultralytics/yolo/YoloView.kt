// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.*
import android.widget.FrameLayout
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.Executors
import kotlin.math.max

class YoloView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : FrameLayout(context, attrs), DefaultLifecycleObserver {

    // Lifecycle owner for camera
    private var lifecycleOwner: LifecycleOwner? = null

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        private const val TAG = "YoloView"

        // Line thickness and corner radius
        private const val BOX_LINE_WIDTH = 8f
        private const val BOX_CORNER_RADIUS = 12f
        private const val KEYPOINT_LINE_WIDTH = 6f

        // Colors derived from Ultralytics
        private val ultralyticsColors = arrayOf(
            Color.argb(153, 4,   42,  255),
            Color.argb(153, 11,  219, 235),
            Color.argb(153, 243, 243, 243),
            Color.argb(153, 0,   223, 183),
            Color.argb(153, 17,  31,  104),
            Color.argb(153, 255, 111, 221),
            Color.argb(153, 255, 68,  79),
            Color.argb(153, 204, 237, 0),
            Color.argb(153, 0,   243, 68),
            Color.argb(153, 189, 0,   255),
            Color.argb(153, 0,   180, 255),
            Color.argb(153, 221, 0,   186),
            Color.argb(153, 0,   255, 255),
            Color.argb(153, 38,  192, 0),
            Color.argb(153, 1,   255, 179),
            Color.argb(153, 125, 36,  255),
            Color.argb(153, 123, 0,   104),
            Color.argb(153, 255, 27,  108),
            Color.argb(153, 252, 109, 47),
            Color.argb(153, 162, 255, 11)
        )

        // Pose
        private val posePalette = arrayOf(
            floatArrayOf(255f, 128f,  0f),
            floatArrayOf(255f, 153f,  51f),
            floatArrayOf(255f, 178f, 102f),
            floatArrayOf(230f, 230f,   0f),
            floatArrayOf(255f, 153f, 255f),
            floatArrayOf(153f, 204f, 255f),
            floatArrayOf(255f, 102f, 255f),
            floatArrayOf(255f,  51f, 255f),
            floatArrayOf(102f, 178f, 255f),
            floatArrayOf( 51f, 153f, 255f),
            floatArrayOf(255f, 153f, 153f),
            floatArrayOf(255f, 102f, 102f),
            floatArrayOf(255f,  51f,  51f),
            floatArrayOf(153f, 255f, 153f),
            floatArrayOf(102f, 255f, 102f),
            floatArrayOf( 51f, 255f,  51f),
            floatArrayOf(  0f, 255f,   0f),
            floatArrayOf(  0f,   0f, 255f),
            floatArrayOf(255f,   0f,   0f),
            floatArrayOf(255f, 255f, 255f),
        )

        private val kptColorIndices = intArrayOf(
            16,16,16,16,16,
            9, 9, 9, 9, 9, 9,
            0, 0, 0, 0, 0, 0
        )

        private val limbColorIndices = intArrayOf(
            0, 0, 0, 0,
            7, 7, 7,
            9, 9, 9, 9, 9,
            16,16,16,16,16,16,16
        )

        private val skeleton = arrayOf(
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
    }

    // Callback to notify inference results externally
    private var inferenceCallback: ((YOLOResult) -> Unit)? = null

    /** Set the callback */
    fun setOnInferenceCallback(callback: (YOLOResult) -> Unit) {
        this.inferenceCallback = callback
    }

    // Callback to notify model load completion
    private var modelLoadCallback: ((Boolean) -> Unit)? = null

    /** Set model load completion callback (true: success) */
    fun setOnModelLoadCallback(callback: (Boolean) -> Unit) {
        this.modelLoadCallback = callback
    }

    // Use a PreviewView, forcing a TextureView under the hood
    private val previewView: PreviewView = PreviewView(context).apply {
        // Force TextureView usage so the overlay can be on top
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        scaleType = PreviewView.ScaleType.FILL_CENTER
    }

    // The overlay for bounding boxes
    private val overlayView: OverlayView = OverlayView(context)

    private var inferenceResult: YOLOResult? = null
    private var predictor: Predictor? = null
    private var task: YOLOTask = YOLOTask.DETECT
    private var modelName: String = "Model"

    // Camera config
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    // detection thresholds (can be changed externally via setters)
    private var confidenceThreshold = 0.25  // initial value
    private var iouThreshold = 0.45
    private var numItemsThreshold = 30

    init {
        // Clear any existing children
        removeAllViews()

        // 1) A container for the camera preview
        val previewContainer = FrameLayout(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.MATCH_PARENT,
                LayoutParams.MATCH_PARENT
            )
        }

        // 2) Add the previewView to that container
        previewContainer.addView(previewView, LayoutParams(
            LayoutParams.MATCH_PARENT,
            LayoutParams.MATCH_PARENT
        ))

        // 3) Add that container
        addView(previewContainer)

        // 4) Add the overlay on top
        addView(overlayView, LayoutParams(
            LayoutParams.MATCH_PARENT,
            LayoutParams.MATCH_PARENT
        ))

        // Ensure overlay is visually above the preview container
        overlayView.elevation = 100f
        overlayView.translationZ = 100f
        previewContainer.elevation = 1f

        Log.d(TAG, "YoloView init: forced TextureView usage for camera preview + overlay on top.")
    }

    // region threshold setters

    fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf
        (predictor as? ObjectDetector)?.setConfidenceThreshold(conf.toFloat())
    }

    fun setIouThreshold(iou: Double) {
        iouThreshold = iou
        (predictor as? ObjectDetector)?.setIouThreshold(iou.toFloat())
    }

    fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        (predictor as? ObjectDetector)?.setNumItemsThreshold(n)
    }

    // endregion

    // region Model / Task

    fun setModel(modelPath: String, task: YOLOTask, context: Context) {
        Executors.newSingleThreadExecutor().execute {
            try {
                val newPredictor = when (task) {
                    YOLOTask.DETECT -> ObjectDetector(context, modelPath, loadLabels(modelPath), useGpu = true).apply {
                        setConfidenceThreshold(confidenceThreshold.toFloat())
                        setIouThreshold(iouThreshold.toFloat())
                        setNumItemsThreshold(numItemsThreshold)
                    }
                    YOLOTask.SEGMENT -> Segmenter(context, modelPath, loadLabels(modelPath), useGpu = true)
                    YOLOTask.CLASSIFY -> Classifier(context, modelPath, loadLabels(modelPath), useGpu = true)
                    YOLOTask.POSE -> PoseEstimator(context, modelPath, loadLabels(modelPath), useGpu = true)
                    YOLOTask.OBB -> ObbDetector(context, modelPath, loadLabels(modelPath), useGpu = true)
                }

                post {
                    this.task = task
                    this.predictor = newPredictor
                    this.modelName = modelPath.substringAfterLast("/")
                    modelLoadCallback?.invoke(true)
                    Log.d(TAG, "Model loaded successfully: $modelPath")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: $modelPath", e)
                post {
                    modelLoadCallback?.invoke(false)
                }
            }
        }
    }

    private fun loadLabels(modelPath: String): List<String> {
        // Try to load labels from model metadata first
        val loadedLabels = YoloFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        if (loadedLabels != null) {
            Log.d(TAG, "Labels loaded from model metadata: ${loadedLabels.size} classes")
            return loadedLabels
        }
        
        // Return COCO dataset's 80 classes as a fallback
        // This is much more complete than the previous 7-class hardcoded list
        Log.d(TAG, "Using COCO classes as fallback")
        return listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )
    }

    // endregion

    /**
     * Called when a LifecycleOwner is available for camera operations
     */
    fun onLifecycleOwnerAvailable(owner: LifecycleOwner) {
        this.lifecycleOwner = owner
        // Register as a lifecycle observer to handle lifecycle events
        owner.lifecycle.addObserver(this)
        
        // If camera was requested but couldn't start due to missing lifecycle owner, try again
        if (allPermissionsGranted()) {
            startCamera()
        }
        Log.d(TAG, "LifecycleOwner set: ${owner.javaClass.simpleName}")
    }
    
    // region camera init

    fun initCamera() {
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            val activity = context as? Activity ?: return
            ActivityCompat.requestPermissions(
                activity,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(context, "Camera permission not granted.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    fun startCamera() {
        Log.d(TAG, "Starting camera...")

        try {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    Log.d(TAG, "Camera provider obtained")

                    val preview = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()

                    val executor = Executors.newSingleThreadExecutor()
                    imageAnalysis.setAnalyzer(executor) { imageProxy ->
                        onFrame(imageProxy)
                    }

                    val cameraSelector = CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build()

                    Log.d(TAG, "Unbinding all camera use cases")
                    cameraProvider.unbindAll()

                    try {
                        val owner = lifecycleOwner
                        if (owner == null) {
                            Log.e(TAG, "No LifecycleOwner available. Call onLifecycleOwnerAvailable() first.")
                            return@addListener
                        }

                        Log.d(TAG, "Binding camera use cases to lifecycle")
                        cameraProvider.bindToLifecycle(
                            owner,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )

                        Log.d(TAG, "Setting surface provider to previewView")
                        preview.setSurfaceProvider(previewView.surfaceProvider)
                        Log.d(TAG, "Camera setup completed successfully")
                    } catch (e: Exception) {
                        Log.e(TAG, "Use case binding failed", e)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error getting camera provider", e)
                }
            }, ContextCompat.getMainExecutor(context))
        } catch (e: Exception) {
            Log.e(TAG, "Error starting camera", e)
        }
    }

    fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    // endregion
    
    // Lifecycle methods from DefaultLifecycleObserver
    override fun onStart(owner: LifecycleOwner) {
        Log.d(TAG, "Lifecycle onStart")
        if (allPermissionsGranted()) {
            startCamera()
        }
    }

    override fun onStop(owner: LifecycleOwner) {
        Log.d(TAG, "Lifecycle onStop")
        // Camera will be automatically stopped by CameraX when lifecycle stops
    }

    // region onFrame (per frame inference)

    private fun onFrame(imageProxy: ImageProxy) {
        val w = imageProxy.width
        val h = imageProxy.height
        Log.d(TAG, "Processing frame: ${w}x${h}")

        val bitmap = ImageUtils.toBitmap(imageProxy) ?: run {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap")
            imageProxy.close()
            return
        }

        predictor?.let { p ->
            try {
                // For camera feed, we typically rotate the bitmap
                val result = p.predict(bitmap, h, w, rotateForCamera = true)
                inferenceResult = result

                // Log
                Log.d(TAG, "Inference complete: ${result.boxes.size} boxes detected")

                // Callback
                inferenceCallback?.invoke(result)

                // Update overlay
                post {
                    overlayView.invalidate()
                    Log.d(TAG, "Overlay invalidated for redraw")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during prediction", e)
            }
        }
        imageProxy.close()
    }

    // endregion

    // region OverlayView

    private inner class OverlayView(context: Context) : View(context) {
        private val paint = Paint().apply { isAntiAlias = true }

        init {
            // Make background transparent
            setBackgroundColor(Color.TRANSPARENT)
            // Use hardware layer for better z-order 
            setLayerType(LAYER_TYPE_HARDWARE, null)

            // Raise overlay
            elevation = 1000f
            translationZ = 1000f

            setWillNotDraw(false)

            Log.d(TAG, "OverlayView initialized with enhanced Z-order + hardware acceleration")
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val result = inferenceResult ?: return
            
            Log.d(TAG, "OverlayView onDraw: Drawing result with ${result.boxes.size} boxes")

            val iw = result.origShape.width.toFloat()
            val ih = result.origShape.height.toFloat()

            val vw = width.toFloat()
            val vh = height.toFloat()
            
            Log.d(TAG, "OverlayView dimensions: View(${vw}x${vh}), Image(${iw}x${ih})")

            // Scale factor from camera image to view
            val scaleX = vw / iw
            val scaleY = vh / ih
            val scale = max(scaleX, scaleY)

            val scaledW = iw * scale
            val scaledH = ih * scale

            val dx = (vw - scaledW) / 2f
            val dy = (vh - scaledH) / 2f
            
            // Check if using front camera
            val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
            
            Log.d(TAG, "OverlayView scaling: scale=${scale}, dx=${dx}, dy=${dy}, frontCamera=${isFrontCamera}")

            when (task) {
                // ----------------------------------------
                // DETECT
                // ----------------------------------------
                YOLOTask.DETECT -> {
                    Log.d(TAG, "Drawing DETECT boxes: ${result.boxes.size}")
                    for (box in result.boxes) {
                        // Adjust alpha based on confidence
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        // Log the original box.xywh values
                        Log.d(TAG, "Box raw coords: L=${box.xywh.left}, T=${box.xywh.top}, R=${box.xywh.right}, B=${box.xywh.bottom}, cls=${box.cls}, conf=${box.conf}")
                        
                        // Draw bounding box like in the original code
                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // Flip vertically for front camera
                        if (isFrontCamera) {
                            val flippedTop = vh - bottom
                            val flippedBottom = vh - top
                            top = flippedTop
                            bottom = flippedBottom
                        }
                        
                        Log.d(TAG, "Drawing box for ${box.cls}: L=$left, T=$top, R=$right, B=$bottom, conf=${box.conf}")

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        // Label text
                        val labelText = "${box.cls} ${"%.1f".format(box.conf * 100)}%"
                        paint.textSize = 40f
                        val fm = paint.fontMetrics
                        val textWidth = paint.measureText(labelText)
                        val textHeight = fm.bottom - fm.top
                        val pad = 8f

                        // Label background height is (text height + 2*padding)
                        val labelBoxHeight = textHeight + 2 * pad
                        // Place label on top of the box's upper edge
                        val labelBottom = top
                        val labelTop = labelBottom - labelBoxHeight

                        // Rectangle for label background
                        val labelLeft = left
                        val labelRight = left + textWidth + 2 * pad
                        val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)

                        // Draw background
                        paint.style = Paint.Style.FILL
                        paint.color = newColor
                        canvas.drawRoundRect(bgRect, BOX_CORNER_RADIUS, BOX_CORNER_RADIUS, paint)

                        // Center text vertically within the rectangle
                        paint.color = Color.WHITE
                        // Center position = (bgRect.top + bgRect.bottom)/2
                        val centerY = (labelTop + labelBottom) / 2
                        // Baseline = centerY - (fm.descent + fm.ascent)/2
                        val baseline = centerY - (fm.descent + fm.ascent) / 2
                        // X coordinate is left-aligned plus padding
                        val textX = labelLeft + pad

                        canvas.drawText(labelText, textX, baseline, paint)
                    }
                }
                // ----------------------------------------
                // SEGMENT
                // ----------------------------------------
                YOLOTask.SEGMENT -> {
                    // Bounding boxes & labels
                    for (box in result.boxes) {
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        // Draw bounding box
                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // Flip vertically for front camera
                        if (isFrontCamera) {
                            val flippedTop = vh - bottom
                            val flippedBottom = vh - top
                            top = flippedTop
                            bottom = flippedBottom
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        // Label background + text (vertically centered)
                        val labelText = "${box.cls} ${"%.1f".format(box.conf * 100)}%"
                        paint.textSize = 40f
                        val fm = paint.fontMetrics
                        val textWidth = paint.measureText(labelText)
                        val textHeight = fm.bottom - fm.top
                        val pad = 8f

                        val labelBoxHeight = textHeight + 2 * pad
                        val labelBottom = top
                        val labelTop = labelBottom - labelBoxHeight
                        val labelLeft = left
                        val labelRight = left + textWidth + 2 * pad
                        val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)

                        paint.style = Paint.Style.FILL
                        paint.color = newColor
                        canvas.drawRoundRect(bgRect, BOX_CORNER_RADIUS, BOX_CORNER_RADIUS, paint)

                        paint.color = Color.WHITE
                        val centerY = (labelTop + labelBottom) / 2
                        val baseline = centerY - (fm.descent + fm.ascent) / 2
                        canvas.drawText(labelText, labelLeft + pad, baseline, paint)
                    }

                    // Segmentation mask
                    result.masks?.combinedMask?.let { maskBitmap ->
                        val src = Rect(0, 0, maskBitmap.width, maskBitmap.height)
                        val dst = RectF(dx, dy, dx + scaledW, dy + scaledH)
                        val maskPaint = Paint().apply { alpha = 128 }
                        
                        if (isFrontCamera) {
                            // For front camera, flip the mask vertically
                            canvas.save()
                            // Translate to center, flip vertically, translate back
                            canvas.translate(0f, vh / 2f)
                            canvas.scale(1f, -1f)
                            canvas.translate(0f, -vh / 2f)
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                            canvas.restore()
                        } else {
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                        }
                    }
                }
                // ----------------------------------------
                // CLASSIFY (display large in center)
                // ----------------------------------------
                YOLOTask.CLASSIFY -> {
                    result.probs?.let { probs ->
                        val alpha = (probs.top1Conf * 255).toInt().coerceIn(0, 255)
                        // Select color based on top1Index
                        val baseColor = ultralyticsColors[probs.top1Index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        val labelText = "${probs.top1} ${"%.1f".format(probs.top1Conf * 100)}%"
                        paint.textSize = 60f
                        val textWidth = paint.measureText(labelText)
                        val fm = paint.fontMetrics
                        val textHeight = fm.bottom - fm.top
                        val pad = 16f

                        // Screen center
                        val centerX = vw / 2f
                        val centerY = vh / 2f

                        val bgLeft   = centerX - (textWidth / 2) - pad
                        val bgTop    = centerY - (textHeight / 2) - pad
                        val bgRight  = centerX + (textWidth / 2) + pad
                        val bgBottom = centerY + (textHeight / 2) + pad

                        paint.color = newColor
                        paint.style = Paint.Style.FILL
                        val bgRect = RectF(bgLeft, bgTop, bgRight, bgBottom)
                        canvas.drawRoundRect(bgRect, 20f, 20f, paint)

                        paint.color = Color.WHITE
                        val baseline = centerY - (fm.descent + fm.ascent)/2
                        canvas.drawText(labelText, centerX - (textWidth / 2), baseline, paint)
                    }
                }
                // ----------------------------------------
                // POSE
                // ----------------------------------------
                YOLOTask.POSE -> {
                    // Bounding boxes
                    for (box in result.boxes) {
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // Flip vertically for front camera
                        if (isFrontCamera) {
                            val flippedTop = vh - bottom
                            val flippedBottom = vh - top
                            top = flippedTop
                            bottom = flippedBottom
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )
                    }

                    // Keypoints & skeleton
                    Log.d(TAG, "🎨 Drawing keypoints - keypointsList size: ${result.keypointsList.size}")
                    for ((personIdx, person) in result.keypointsList.withIndex()) {
                        Log.d(TAG, "👤 Person $personIdx - keypoints: ${person.xyn.size}, conf array size: ${person.conf.size}")
                        val points = arrayOfNulls<PointF>(person.xyn.size)
                        var drawnKps = 0
                        for (i in person.xyn.indices) {
                            val kp = person.xyn[i]
                            val conf = person.conf[i]
                            if (conf > 0.25f) {
                                drawnKps++
                                val pxCam = kp.first * iw
                                val pyCam = kp.second * ih
                                val px = pxCam * scale + dx
                                var py = pyCam * scale + dy
                                
                                // Flip vertically for front camera
                                if (isFrontCamera) {
                                    py = vh - py
                                }

                                val colorIdx = if (i < kptColorIndices.size) kptColorIndices[i] else 0
                                val rgbArray = posePalette[colorIdx % posePalette.size]
                                paint.color = Color.argb(
                                    255,
                                    rgbArray[0].toInt().coerceIn(0,255),
                                    rgbArray[1].toInt().coerceIn(0,255),
                                    rgbArray[2].toInt().coerceIn(0,255)
                                )
                                paint.style = Paint.Style.FILL
                                canvas.drawCircle(px, py, 8f, paint)

                                points[i] = PointF(px, py)
                                
                                // Debug first few keypoints
                                if (i < 3) {
                                    Log.d(TAG, "🔵 Person $personIdx KP[$i] - normalized(${kp.first},${kp.second}) pixel($px,$py) conf=$conf")
                                }
                            }
                        }
                        Log.d(TAG, "✅ Person $personIdx - Drew $drawnKps/${person.xyn.size} keypoints (conf>0.25)")

                        // Skeleton connection
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = KEYPOINT_LINE_WIDTH
                        for ((idx, bone) in skeleton.withIndex()) {
                            val i1 = bone[0] - 1  // 1-indexed to 0-indexed
                            val i2 = bone[1] - 1
                            val p1 = points.getOrNull(i1)
                            val p2 = points.getOrNull(i2)
                            if (p1 != null && p2 != null) {
                                val limbColorIdx = if (idx < limbColorIndices.size) limbColorIndices[idx] else 0
                                val rgbArray = posePalette[limbColorIdx % posePalette.size]
                                paint.color = Color.argb(
                                    255,
                                    rgbArray[0].toInt().coerceIn(0,255),
                                    rgbArray[1].toInt().coerceIn(0,255),
                                    rgbArray[2].toInt().coerceIn(0,255)
                                )
                                canvas.drawLine(p1.x, p1.y, p2.x, p2.y, paint)
                            }
                        }
                    }
                }
                // ----------------------------------------
                // OBB
                // ----------------------------------------
                YOLOTask.OBB -> {
                    for (obbRes in result.obb) {
                        val alpha = (obbRes.confidence * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[obbRes.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH

                        // Draw rotated rectangle (polygon) using path
                        val polygon = obbRes.box.toPolygon().map { pt ->
                            val x = pt.x * scaledW + dx
                            var y = pt.y * scaledH + dy
                            
                            // Flip vertically for front camera
                            if (isFrontCamera) {
                                y = vh - y
                            }
                            
                            PointF(x, y)
                        }
                        if (polygon.size >= 4) {
                            val path = Path().apply {
                                moveTo(polygon[0].x, polygon[0].y)
                                for (p in polygon.drop(1)) {
                                    lineTo(p.x, p.y)
                                }
                                close()
                            }
                            canvas.drawPath(path, paint)

                            // Label text
                            val labelText = "${obbRes.cls} ${"%.1f".format(obbRes.confidence * 100)}%"
                            paint.textSize = 40f
                            paint.typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)

                            val fm = paint.fontMetrics
                            val textWidth = paint.measureText(labelText)
                            val textHeight = fm.bottom - fm.top
                            val padding = 10f
                            val cornerRadius = 8f

                            // Display background rectangle near polygon[0]
                            val labelBoxHeight = textHeight + 2 * padding
                            val labelBottom = polygon[0].y
                            val labelTop = labelBottom - labelBoxHeight
                            val labelLeft = polygon[0].x
                            val labelRight = labelLeft + textWidth + 2 * padding

                            val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
                            paint.style = Paint.Style.FILL
                            paint.color = newColor
                            canvas.drawRoundRect(bgRect, cornerRadius, cornerRadius, paint)

                            // Center text vertically
                            paint.color = Color.WHITE
                            val centerY = (labelTop + labelBottom) / 2
                            val baseline = centerY - (fm.descent + fm.ascent) / 2
                            val textX = labelLeft + padding
                            canvas.drawText(labelText, textX, baseline, paint)
                        }
                    }
                }
            }
        }
    }
}