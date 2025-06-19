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
import android.view.ScaleGestureDetector
import androidx.camera.core.*
import androidx.camera.core.Camera
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
import kotlin.math.min
import android.widget.TextView
import android.view.Gravity
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import android.content.res.Configuration

class YOLOView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : FrameLayout(context, attrs), DefaultLifecycleObserver {

    // Lifecycle owner for camera
    private var lifecycleOwner: LifecycleOwner? = null

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private var previewUseCase: Preview? = null

        private const val TAG = "YOLOView"

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
    
    // Streaming functionality
    private var streamConfig: YOLOStreamConfig? = null
    private var streamCallback: ((Map<String, Any>) -> Unit)? = null
    
    // Frame counter for streaming
    private var frameNumberCounter: Long = 0
    
    // Throttling variables for performance control
    private var lastInferenceTime: Long = 0
    private var targetFrameInterval: Long? = null // in nanoseconds
    private var throttleInterval: Long? = null // in nanoseconds
    
    // Inference frequency control variables
    private var inferenceFrameInterval: Long? = null // Target inference interval in nanoseconds
    private var frameSkipCount: Int = 0 // Current frame skip counter
    private var targetSkipFrames: Int = 0 // Number of frames to skip between inferences

    /** Set the callback */
    fun setOnInferenceCallback(callback: (YOLOResult) -> Unit) {
        this.inferenceCallback = callback
    }
    
    /** Set streaming configuration */
    fun setStreamConfig(config: YOLOStreamConfig?) {
        Log.d(TAG, "🔄 Setting new streaming config")
        Log.d(TAG, "📋 Previous config: $streamConfig")
        this.streamConfig = config
        setupThrottlingFromConfig()
        Log.d(TAG, "✅ New streaming config set: $config")
        Log.d(TAG, "🎯 Key settings - includeMasks: ${config?.includeMasks}, includeProcessingTimeMs: ${config?.includeProcessingTimeMs}, inferenceFrequency: ${config?.inferenceFrequency}")
    }
    
    /** Set streaming callback */
    fun setStreamCallback(callback: ((Map<String, Any>) -> Unit)?) {
        this.streamCallback = callback
        Log.d(TAG, "Streaming callback set: ${callback != null}")
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
    private var camera: Camera? = null

    // New fields for proper teardown:
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalysisUseCase: ImageAnalysis? = null    

    // Zoom related
    private var currentZoomRatio = 1.0f
    private var minZoomRatio = 1.0f
    private var maxZoomRatio = 10.0f
    private lateinit var scaleGestureDetector: ScaleGestureDetector
    var onZoomChanged: ((Float) -> Unit)? = null

    // detection thresholds (can be changed externally via setters)
    private var confidenceThreshold = 0.25  // initial value
    private var iouThreshold = 0.45
    private var numItemsThreshold = 30
    private lateinit var zoomLabel: TextView

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
        
        // Add zoom label
        zoomLabel = TextView(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.CENTER
            }
            text = "1.0x"
            textSize = 24f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(128, 0, 0, 0))
            setPadding(16, 8, 16, 8)
            visibility = View.GONE
        }
        addView(zoomLabel)
        
        // Initialize scale gesture detector for pinch-to-zoom
        scaleGestureDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val scale = detector.scaleFactor
                val newZoomRatio = currentZoomRatio * scale
                
                // Clamp zoom ratio between min and max
                val clampedZoomRatio = newZoomRatio.coerceIn(minZoomRatio, camera?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: maxZoomRatio)
                
                camera?.cameraControl?.setZoomRatio(clampedZoomRatio)
                currentZoomRatio = clampedZoomRatio
                
                // Notify zoom change
                onZoomChanged?.invoke(currentZoomRatio)
                
                return true
            }
        })

        Log.d(TAG, "YoloView init: forced TextureView usage for camera preview + overlay on top.")
    }

    // region threshold setters

    fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf
        (predictor as? ObjectDetector)?.setConfidenceThreshold(conf)
    }

    fun setIouThreshold(iou: Double) {
        iouThreshold = iou
        (predictor as? ObjectDetector)?.setIouThreshold(iou)
    }

    fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        (predictor as? ObjectDetector)?.setNumItemsThreshold(n)
    }
    
    fun setZoomLevel(zoomLevel: Float) {
        camera?.let { cam: Camera ->
            // Clamp zoom level between min and max
            val clampedZoomRatio = zoomLevel.coerceIn(minZoomRatio, cam.cameraInfo.zoomState.value?.maxZoomRatio ?: maxZoomRatio)
            
            cam.cameraControl.setZoomRatio(clampedZoomRatio)
            currentZoomRatio = clampedZoomRatio
            
            // Notify zoom change
            onZoomChanged?.invoke(currentZoomRatio)
        }
    }

    // endregion

    // region Model / Task

    fun setModel(modelPath: String, task: YOLOTask, callback: ((Boolean) -> Unit)? = null) {
        Executors.newSingleThreadExecutor().execute {
            try {
                val newPredictor = when (task) {
                    YOLOTask.DETECT -> ObjectDetector(context, modelPath, loadLabels(modelPath), useGpu = true).apply {
                        setConfidenceThreshold(confidenceThreshold)
                        setIouThreshold(iouThreshold)
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
                    callback?.invoke(true)
                    Log.d(TAG, "Model loaded successfully: $modelPath")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load model: $modelPath. Camera will run without inference.", e)
                post {
                    // Set predictor to null to ensure camera-only mode
                    this.predictor = null
                    this.modelName = "No Model"
                    modelLoadCallback?.invoke(false)
                    callback?.invoke(false)
                }
            }
        }
    }

    private fun loadLabels(modelPath: String): List<String> {
        // Try to load labels from model metadata first
        val loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
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
            cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    Log.d(TAG, "Camera provider obtained")

                    previewUseCase = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()

                    imageAnalysisUseCase = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()

                    cameraExecutor = Executors.newSingleThreadExecutor()
                    imageAnalysisUseCase!!.setAnalyzer(cameraExecutor!!) { imageProxy ->
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
                        camera = cameraProvider.bindToLifecycle(
                            owner,
                            cameraSelector,
                            previewUseCase,
                            imageAnalysisUseCase  // the field, not a local val
                        )
                        
                        // Reset zoom to 1.0x when camera starts
                        currentZoomRatio = 1.0f
                        onZoomChanged?.invoke(currentZoomRatio)

                        Log.d(TAG, "Setting surface provider to previewView")
                        previewUseCase?.setSurfaceProvider(previewView.surfaceProvider)
                        
                        // Initialize zoom
                        camera?.let { cam: Camera ->
                            val cameraInfo = cam.cameraInfo
                            minZoomRatio = cameraInfo.zoomState.value?.minZoomRatio ?: 1.0f
                            maxZoomRatio = cameraInfo.zoomState.value?.maxZoomRatio ?: 1.0f
                            currentZoomRatio = cameraInfo.zoomState.value?.zoomRatio ?: 1.0f
                            Log.d(TAG, "Zoom initialized - min: $minZoomRatio, max: $maxZoomRatio, current: $currentZoomRatio")
                        }
                        
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
        val orientation = context.resources.configuration.orientation
        val isLandscapeDevice = orientation == Configuration.ORIENTATION_LANDSCAPE

        val bitmap = ImageUtils.toBitmap(imageProxy) ?: run {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap")
            imageProxy.close()
            return
        }

        predictor?.let { p ->
            // Check if we should run inference on this frame
            if (!shouldRunInference()) {
                Log.d(TAG, "Skipping inference due to frequency control")
                imageProxy.close()
                return
            }
            
            try {
                // Get device orientation
                val orientation = context.resources.configuration.orientation
                val isLandscape = orientation == Configuration.ORIENTATION_LANDSCAPE
                
                // For camera feed, we typically rotate the bitmap
                // In landscape mode, we don't rotate, so width/height should match actual bitmap dimensions
                val result = if (isLandscape) {
                    p.predict(bitmap, w, h, rotateForCamera = true, isLandscape = isLandscape)
                } else {
                    // In portrait mode, keep the original behavior (h, w)
                    p.predict(bitmap, h, w, rotateForCamera = true, isLandscape = isLandscape)
                }
                
                // Apply originalImage if streaming config requires it
                val resultWithOriginalImage = if (streamConfig?.includeOriginalImage == true) {
                    result.copy(originalImage = bitmap)  // Reuse bitmap from ImageProxy conversion
                } else {
                    result
                }
                
                inferenceResult = resultWithOriginalImage

                // Log
                
                // Callback
                inferenceCallback?.invoke(resultWithOriginalImage)
                
                // Streaming callback (with output throttling)
                streamCallback?.let { callback ->
                    if (shouldProcessFrame()) {
                        updateLastInferenceTime()
                        
                        // Convert to stream data and send
                        val streamData = convertResultToStreamData(resultWithOriginalImage)
                        // Add timestamp and frame info
                        val enhancedStreamData = HashMap<String, Any>(streamData)
                        enhancedStreamData["timestamp"] = System.currentTimeMillis()
                        enhancedStreamData["frameNumber"] = frameNumberCounter++
                        
                        callback.invoke(enhancedStreamData)
                    } else {
                        Log.d(TAG, "Skipping frame output due to throttling")
                    }
                }

                // Update overlay
                post {
                    overlayView.invalidate()
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

            // Make overlay not intercept touch events
            isClickable = false
            isFocusable = false

            Log.d(TAG, "OverlayView initialized with enhanced Z-order + hardware acceleration")
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val result = inferenceResult ?: return
            

            val iw = result.origShape.width.toFloat()
            val ih = result.origShape.height.toFloat()

            val vw = width.toFloat()
            val vh = height.toFloat()
            
            // Get device orientation for debugging
            val orientation = context.resources.configuration.orientation
            val isLandscape = orientation == Configuration.ORIENTATION_LANDSCAPE
            

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
            

            when (task) {
                // ----------------------------------------
                // DETECT
                // ----------------------------------------
                YOLOTask.DETECT -> {
                    Log.d(TAG, "Drawing DETECT boxes: ${result.boxes.size}")
                    
                    // Debug first box coordinates
                    if (result.boxes.isNotEmpty()) {
                        val firstBox = result.boxes[0]
                        Log.d(TAG, "=== First Box Debug ===")
                        Log.d(TAG, "Box normalized coords: (${firstBox.xywhn.left}, ${firstBox.xywhn.top}, ${firstBox.xywhn.right}, ${firstBox.xywhn.bottom})")
                        Log.d(TAG, "Box pixel coords: (${firstBox.xywh.left}, ${firstBox.xywh.top}, ${firstBox.xywh.right}, ${firstBox.xywh.bottom})")
                    }
                    
                    for (box in result.boxes) {
                        val alpha = (box.conf * 255).toInt().coerceIn(0, 255)
                        val baseColor = ultralyticsColors[box.index % ultralyticsColors.size]
                        val newColor = Color.argb(
                            alpha,
                            Color.red(baseColor),
                            Color.green(baseColor),
                            Color.blue(baseColor)
                        )

                        // Use same coordinate calculation for all orientations
                        // since the image is now correctly oriented before inference
                        var left = box.xywh.left * scale + dx
                        var top = box.xywh.top * scale + dy
                        var right = box.xywh.right * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // Ensure coordinates are within view bounds and maintain aspect ratio
                        val boxWidth = right - left
                        val boxHeight = bottom - top
                        
                        // Adjust coordinates to maintain aspect ratio and stay within bounds
                        if (left < 0) {
                            left = 0f
                            right = left + boxWidth
                        }
                        if (right > vw) {
                            right = vw.toFloat()
                            left = right - boxWidth
                        }
                        if (top < 0) {
                            top = 0f
                            bottom = top + boxHeight
                        }
                        if (bottom > vh) {
                            bottom = vh.toFloat()
                            top = bottom - boxHeight
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
                        canvas.drawText(labelText, labelLeft + pad, baseline, paint)
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
                    for (person in result.keypointsList) {
                        val points = arrayOfNulls<PointF>(person.xyn.size)
                        for (i in person.xyn.indices) {
                            val kp = person.xyn[i]
                            val conf = person.conf[i]
                            if (conf > 0.25f) {
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
                            }
                        }

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
        
        override fun onTouchEvent(event: MotionEvent?): Boolean {
            // Pass through all touch events
            return false
        }
    }
    
    // Scale listener for pinch-to-zoom
    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
            // Show zoom label when pinch starts
            zoomLabel.visibility = View.VISIBLE
            return true
        }
        
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val scaleFactor = detector.scaleFactor
            val newZoomRatio = currentZoomRatio * scaleFactor
            
            // Clamp zoom within min/max bounds
            val clampedZoom = newZoomRatio.coerceIn(minZoomRatio, maxZoomRatio)
            
            // Apply zoom to camera
            camera?.cameraControl?.setZoomRatio(clampedZoom)
            currentZoomRatio = clampedZoom
            
            // Update zoom label
            zoomLabel.text = String.format("%.1fx", currentZoomRatio)
            
            return true
        }
        
        override fun onScaleEnd(detector: ScaleGestureDetector) {
            // Hide zoom label after 2 seconds
            zoomLabel.postDelayed({
                zoomLabel.visibility = View.GONE
            }, 2000)
        }
    }
    
    // Touch event handling for pinch-to-zoom
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleGestureDetector.onTouchEvent(event)
        return true
    }
    
    // region Streaming functionality
    
    /**
     * Setup throttling parameters from streaming configuration
     */
    private fun setupThrottlingFromConfig() {
        streamConfig?.let { config ->
            // Setup maxFPS throttling (for result output)
            config.maxFPS?.let { maxFPS ->
                if (maxFPS > 0) {
                    targetFrameInterval = (1_000_000_000L / maxFPS) // Convert to nanoseconds
                    Log.d(TAG, "maxFPS throttling enabled - target FPS: $maxFPS, interval: ${targetFrameInterval!! / 1_000_000}ms")
                }
            } ?: run {
                targetFrameInterval = null
                Log.d(TAG, "maxFPS throttling disabled")
            }
            
            // Setup throttleInterval (for result output)
            config.throttleIntervalMs?.let { throttleMs ->
                if (throttleMs > 0) {
                    throttleInterval = throttleMs * 1_000_000L // Convert ms to nanoseconds
                    Log.d(TAG, "throttleInterval enabled - interval: ${throttleMs}ms")
                }
            } ?: run {
                throttleInterval = null
                Log.d(TAG, "throttleInterval disabled")
            }
            
            // Setup inference frequency control
            config.inferenceFrequency?.let { inferenceFreq ->
                if (inferenceFreq > 0) {
                    inferenceFrameInterval = (1_000_000_000L / inferenceFreq) // Convert to nanoseconds
                    Log.d(TAG, "Inference frequency control enabled - target inference FPS: $inferenceFreq, interval: ${inferenceFrameInterval!! / 1_000_000}ms")
                }
            } ?: run {
                inferenceFrameInterval = null
                Log.d(TAG, "Inference frequency control disabled")
            }
            
            // Setup frame skipping
            config.skipFrames?.let { skipFrames ->
                if (skipFrames > 0) {
                    targetSkipFrames = skipFrames
                    frameSkipCount = 0 // Reset counter
                    Log.d(TAG, "Frame skipping enabled - skip $skipFrames frames between inferences")
                }
            } ?: run {
                targetSkipFrames = 0
                frameSkipCount = 0
                Log.d(TAG, "Frame skipping disabled")
            }
            
            // Initialize timing
            lastInferenceTime = System.nanoTime()
        }
    }
    
    /**
     * Check if we should run inference on this frame based on inference frequency control
     */
    private fun shouldRunInference(): Boolean {
        val now = System.nanoTime()
        
        // Check frame skipping control first (simpler, more deterministic)
        if (targetSkipFrames > 0) {
            frameSkipCount++
            if (frameSkipCount <= targetSkipFrames) {
                // Still skipping frames
                return false
            } else {
                // Reset counter and allow inference
                frameSkipCount = 0
                return true
            }
        }
        
        // Check inference frequency control (time-based)
        inferenceFrameInterval?.let { interval ->
            if (now - lastInferenceTime < interval) {
                return false
            }
        }
        
        return true
    }
    
    /**
     * Check if we should send results to Flutter based on output throttling settings
     */
    private fun shouldProcessFrame(): Boolean {
        val now = System.nanoTime()
        
        // Check maxFPS throttling
        targetFrameInterval?.let { interval ->
            if (now - lastInferenceTime < interval) {
                return false
            }
        }
        
        // Check throttleInterval
        throttleInterval?.let { interval ->
            if (now - lastInferenceTime < interval) {
                return false
            }
        }
        
        return true
    }
    
    /**
     * Update the last inference time (call this when actually processing)
     */
    private fun updateLastInferenceTime() {
        lastInferenceTime = System.nanoTime()
    }
    
    /**
     * Convert YOLOResult to a Map for streaming (ported from archived YOLOPlatformView)
     * Uses detection index correctly to avoid class index confusion
     */
    private fun convertResultToStreamData(result: YOLOResult): Map<String, Any> {
        val map = HashMap<String, Any>()
        val config = streamConfig ?: return emptyMap()
        
        // Convert detection results (if enabled)
        if (config.includeDetections) {
            val detections = ArrayList<Map<String, Any>>()
            
            // Convert detection boxes - CRITICAL: use detectionIndex, not class index
            for ((detectionIndex, box) in result.boxes.withIndex()) {
                val detection = HashMap<String, Any>()
                detection["classIndex"] = box.index
                detection["className"] = box.cls
                detection["confidence"] = box.conf.toDouble()
                
                // Bounding box in original coordinates
                val boundingBox = HashMap<String, Any>()
                boundingBox["left"] = box.xywh.left.toDouble()
                boundingBox["top"] = box.xywh.top.toDouble()
                boundingBox["right"] = box.xywh.right.toDouble()
                boundingBox["bottom"] = box.xywh.bottom.toDouble()
                detection["boundingBox"] = boundingBox
                
                // Normalized bounding box (0-1)
                val normalizedBox = HashMap<String, Any>()
                normalizedBox["left"] = box.xywhn.left.toDouble()
                normalizedBox["top"] = box.xywhn.top.toDouble()
                normalizedBox["right"] = box.xywhn.right.toDouble()
                normalizedBox["bottom"] = box.xywhn.bottom.toDouble()
                detection["normalizedBox"] = normalizedBox
                
                // Add mask data for segmentation (if available and enabled)
                if (config.includeMasks && result.masks != null && detectionIndex < result.masks!!.masks.size) {
                    val maskData = result.masks!!.masks[detectionIndex] // Get mask for this detection
                    // Convert List<List<Float>> to List<List<Double>> for Flutter compatibility
                    val maskDataDouble = maskData.map { row ->
                        row.map { it.toDouble() }
                    }
                    detection["mask"] = maskDataDouble
                    Log.d(TAG, "✅ Added mask data (${maskData.size}x${maskData.firstOrNull()?.size ?: 0}) for detection $detectionIndex")
                }
                
                // Add pose keypoints (if available and enabled)
                if (config.includePoses && detectionIndex < result.keypointsList.size) {
                    val keypoints = result.keypointsList[detectionIndex]
                    // Convert to flat array [x1, y1, conf1, x2, y2, conf2, ...]
                    val keypointsFlat = mutableListOf<Double>()
                    for (i in keypoints.xy.indices) {
                        keypointsFlat.add(keypoints.xy[i].first.toDouble())
                        keypointsFlat.add(keypoints.xy[i].second.toDouble())
                        if (i < keypoints.conf.size) {
                            keypointsFlat.add(keypoints.conf[i].toDouble())
                        } else {
                            keypointsFlat.add(0.0) // Default confidence if missing
                        }
                    }
                    detection["keypoints"] = keypointsFlat
                    Log.d(TAG, "Added keypoints data (${keypoints.xy.size} points) for detection $detectionIndex")
                }
                
                detections.add(detection)
            }
            
            // Handle OBB results directly (same pattern as overlay: for obbRes in result.obb)
            for (obbRes in result.obb) {
                val detection = HashMap<String, Any>()
                detection["classIndex"] = obbRes.index
                detection["className"] = obbRes.cls
                detection["confidence"] = obbRes.confidence.toDouble()
                
                // Get OBB polygon points (4 corners of rotated rectangle)
                val polygon = obbRes.box.toPolygon()
                val imgWidth = result.origShape.width.toFloat()
                val imgHeight = result.origShape.height.toFloat()
                
                // Convert polygon points to pixel coordinates  
                val polygonPixels = polygon.map { point ->
                    mapOf(
                        "x" to (point.x * imgWidth).toDouble(),
                        "y" to (point.y * imgHeight).toDouble()
                    )
                }
                
                // Store polygon points directly for precise OBB cropping
                detection["polygon"] = polygonPixels
                
                // Also calculate AABB as fallback for compatibility (but Flutter should use polygon)
                var minX = Float.MAX_VALUE
                var maxX = Float.MIN_VALUE  
                var minY = Float.MAX_VALUE
                var maxY = Float.MIN_VALUE
                
                for (point in polygon) {
                    if (point.x < minX) minX = point.x
                    if (point.x > maxX) maxX = point.x
                    if (point.y < minY) minY = point.y
                    if (point.y > maxY) maxY = point.y
                }
                
                // Fallback bounding box (enlarged) - only use if polygon cropping fails
                val boundingBox = HashMap<String, Any>()
                boundingBox["left"] = (minX * imgWidth).toDouble()
                boundingBox["top"] = (minY * imgHeight).toDouble()
                boundingBox["right"] = (maxX * imgWidth).toDouble()
                boundingBox["bottom"] = (maxY * imgHeight).toDouble()
                detection["boundingBox"] = boundingBox
                
                // Normalized bounding box (0-1) - fallback
                val normalizedBox = HashMap<String, Any>()
                normalizedBox["left"] = minX.toDouble()
                normalizedBox["top"] = minY.toDouble()
                normalizedBox["right"] = maxX.toDouble()
                normalizedBox["bottom"] = maxY.toDouble()
                detection["normalizedBox"] = normalizedBox
                
                // Add OBB-specific data
                if (config.includeOBB) {
                    val points = polygon.map { point ->
                        mapOf(
                            "x" to point.x.toDouble(),
                            "y" to point.y.toDouble()
                        )
                    }
                    
                    val obbDataMap = mapOf(
                        "centerX" to obbRes.box.cx.toDouble(),
                        "centerY" to obbRes.box.cy.toDouble(),
                        "width" to obbRes.box.w.toDouble(),
                        "height" to obbRes.box.h.toDouble(),
                        "angle" to obbRes.box.angle.toDouble(),
                        "angleDegrees" to (obbRes.box.angle * 180.0 / Math.PI),
                        "area" to obbRes.box.area.toDouble(),
                        "points" to points,
                        "confidence" to obbRes.confidence.toDouble(),
                        "className" to obbRes.cls,
                        "classIndex" to obbRes.index
                    )
                    
                    detection["obb"] = obbDataMap
                    Log.d(TAG, "✅ Added OBB data: ${obbRes.cls} (${String.format("%.1f", obbRes.box.angle * 180.0 / Math.PI)}° rotation)")
                }
                
                detections.add(detection)
            }
            
            map["detections"] = detections
            Log.d(TAG, "✅ Total detections in stream: ${detections.size} (boxes: ${result.boxes.size}, obb: ${result.obb.size})")
        }
        
        // Add performance metrics (if enabled)
        if (config.includeProcessingTimeMs) {
            val processingTimeMs = result.speed.toDouble()
            map["processingTimeMs"] = processingTimeMs
        } else {
            Log.d(TAG, "⚠️ Skipping processingTimeMs (includeProcessingTimeMs=${config.includeProcessingTimeMs})")
        }
        
        if (config.includeFps) {
            map["fps"] = result.fps?.toDouble() ?: 0.0
        }
        
        // Add original image (if available and enabled)
        if (config.includeOriginalImage) {
            result.originalImage?.let { bitmap ->
                val outputStream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                val imageData = outputStream.toByteArray()
                map["originalImage"] = imageData
                Log.d(TAG, "✅ Added original image data (${imageData.size} bytes)")
            }
        }
        
        return map
    }
    
    // endregion
    
    /**
     * Capture current camera frame with detection overlays
     * Returns the captured image as a ByteArray (JPEG format)
     */
    fun captureFrame(): ByteArray? {
        try {
            // Create bitmap to hold the captured frame
            val width = width
            val height = height
            if (width <= 0 || height <= 0) {
                Log.e(TAG, "Invalid view dimensions for capture: ${width}x${height}")
                return null
            }
            
            // Create bitmap and canvas
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            
            // Method 1: Try to get bitmap from PreviewView directly
            var cameraFrameCaptured = false
            previewView.bitmap?.let { cameraBitmap ->
                Log.d(TAG, "Got camera bitmap from PreviewView: ${cameraBitmap.width}x${cameraBitmap.height}")
                // Draw the camera bitmap scaled to fit
                val matrix = Matrix()
                val scaleX = width.toFloat() / cameraBitmap.width
                val scaleY = height.toFloat() / cameraBitmap.height
                matrix.setScale(scaleX, scaleY)
                canvas.drawBitmap(cameraBitmap, matrix, null)
                cameraFrameCaptured = true
            }
            
            if (!cameraFrameCaptured) {
                // Method 2: Use hardware acceleration to capture the view
                Log.w(TAG, "PreviewView.bitmap is null, trying hardware capture")
                
                // Enable drawing cache temporarily
                isDrawingCacheEnabled = true
                buildDrawingCache()
                drawingCache?.let { cache ->
                    canvas.drawBitmap(cache, 0f, 0f, null)
                    cameraFrameCaptured = true
                }
                isDrawingCacheEnabled = false
                
                if (!cameraFrameCaptured) {
                    // Method 3: Last resort - draw the entire view hierarchy
                    Log.w(TAG, "Drawing cache failed, using draw method")
                    // Draw PreviewView first
                    previewView.draw(canvas)
                }
            }
            
            // Always draw the overlay on top
            overlayView.draw(canvas)
            
            // Convert bitmap to JPEG byte array
            val outputStream = java.io.ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            val imageData = outputStream.toByteArray()
            
            // Clean up
            outputStream.close()
            bitmap.recycle()
            
            Log.d(TAG, "Frame captured successfully: ${imageData.size} bytes, camera captured: $cameraFrameCaptured")
            return imageData
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing frame", e)
            return null
        }
    }

    /**
     * Stop camera and inference (can be restarted later)
     */
    fun stop() {
        Log.d(TAG, "YOLOView.stop() called - tearing down camera")

        try {
            // 1) Unbind all use-cases
            if (::cameraProviderFuture.isInitialized) {
                val cameraProvider = cameraProviderFuture.get()
                Log.d(TAG, "Unbinding all camera use cases")
                cameraProvider.unbindAll()
            }

            // 2) Clear the analyzer so no threads keep the camera alive
            imageAnalysisUseCase?.clearAnalyzer()
            imageAnalysisUseCase = null

            // 3) Detach the PreviewView surface
            previewUseCase?.setSurfaceProvider(null)

            // 4) Shutdown the executor
            cameraExecutor?.let { exec ->
                Log.d(TAG, "Shutting down camera executor")
                exec.shutdown()
                if (!exec.awaitTermination(1, TimeUnit.SECONDS)) {
                    Log.w(TAG, "Executor didn't shut down in time; forcing shutdown")
                    exec.shutdownNow()
                }
            }
            cameraExecutor = null

            // 5) Null out camera and inference machinery
            camera = null
            predictor = null
            inferenceCallback = null
            streamCallback = null
            inferenceResult = null

            Log.d(TAG, "YOLOView stop completed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during YOLOView stop", e)
        }
    }

}
