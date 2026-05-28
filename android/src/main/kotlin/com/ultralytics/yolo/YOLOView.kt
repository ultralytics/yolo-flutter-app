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
import android.hardware.camera2.CameraCharacteristics
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import android.util.SizeF
import kotlin.math.abs
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

/**
 * Describes a back-camera lens with its equivalent zoom factor relative to the main
 * wide-angle lens (1.0x). Used by `getAvailableLenses` to populate the Dart lens picker.
 */
data class LensInfo(
    val zoomFactor: Double,
    val label: String,
    val cameraInfo: CameraInfo? = null
)

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
        this.streamConfig = config
        setupThrottlingFromConfig()
    }

    /** Set streaming callback */
    fun setStreamCallback(callback: ((Map<String, Any>) -> Unit)?) {
        this.streamCallback = callback
    }

    // Generic event callback used to forward {type:"zoom"|"lens"|"focus", ...} maps
    // to the Flutter event sink without coupling YOLOView to a Flutter type.
    private var eventCallback: ((Map<String, Any>) -> Unit)? = null

    /** Set a callback that receives typed events (zoom/lens/focus) for the Flutter event sink. */
    fun setEventCallback(callback: ((Map<String, Any>) -> Unit)?) {
        this.eventCallback = callback
    }

    private fun emitEvent(event: Map<String, Any>) {
        try {
            eventCallback?.invoke(event)
        } catch (e: Exception) {
            Log.e(TAG, "Error emitting event", e)
        }
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
    private var preferWideBackCamera = false
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var camera: Camera? = null

    // New fields for proper teardown:
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalysisUseCase: ImageAnalysis? = null
    
    // Flag to track if the view is stopped/disposed to prevent race conditions
    @Volatile
    private var isStopped = false    

    // Zoom related
    private var currentZoomRatio = 1.0f
    private var minZoomRatio = 1.0f
    private var maxZoomRatio = 10.0f
    var onZoomChanged: ((Float) -> Unit)? = null

    // Multi-lens enumeration / selection
    private var cachedLenses: List<LensInfo> = emptyList()
    private var selectedLensZoomFactor: Double? = null
    private var selectedLensCameraInfo: CameraInfo? = null
    private var selectedLensLabel: String? = null

    // Optional ImageCapture use-case (bound alongside Preview+Analysis when supported)
    private var imageCaptureUseCase: ImageCapture? = null

    // detection thresholds (can be changed externally via setters)
    private var confidenceThreshold = 0.25  // initial value
    private var iouThreshold = 0.7
    private var numItemsThreshold = 30
    private var showOverlays = true
    private lateinit var zoomLabel: TextView
    private lateinit var cameraButton: TextView
    private lateinit var confidenceLabel: TextView
    private var showUIControls = false

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
            text = "ZOOM: 1.0x"
            textSize = 28f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 255, 0, 0))
            setPadding(20, 15, 20, 15)
            visibility = View.GONE
        }
        addView(zoomLabel)
        zoomLabel.elevation = 1000f
        
        // Add camera switch button
        cameraButton = TextView(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.TOP or Gravity.END
                topMargin = 100
                rightMargin = 50
            }
            text = "📷 CAMERA"
            textSize = 24f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 0, 100, 200))
            setPadding(20, 15, 20, 15)
            visibility = View.GONE
            
            setOnClickListener {
                switchCamera()
            }
        }
        addView(cameraButton)
        cameraButton.elevation = 1000f
        
        // Add confidence threshold label
        confidenceLabel = TextView(context).apply {
            layoutParams = LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.START
                bottomMargin = 100
                leftMargin = 50
            }
            text = "Confidence: 0.50"
            textSize = 20f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 200, 100, 0))
            setPadding(15, 10, 15, 10)
            visibility = View.GONE
        }
        addView(confidenceLabel)
        confidenceLabel.elevation = 1000f
        // Dart owns gestures (pinch + tap) via Flutter GestureDetector in YOLOShowcase;
        // native is setter-only. Do not attach ScaleGestureDetector here.
    }

    // region threshold setters

    fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf
        predictor?.setConfidenceThreshold(conf)
        // Update the confidence label if UI controls are shown
        if (showUIControls) {
            post {
                confidenceLabel.text = "Confidence: ${String.format("%.2f", conf)}"
            }
        }
    }

    fun setIouThreshold(iou: Double) {
        iouThreshold = iou
        predictor?.setIouThreshold(iou)
    }

    fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        predictor?.setNumItemsThreshold(n)
    }
    
    fun setShowOverlays(show: Boolean) {
        showOverlays = show
    }
    
    fun setShowUIControls(show: Boolean) {
        showUIControls = show
        // Show/hide all UI controls
        val visibility = if (show) View.VISIBLE else View.GONE
        zoomLabel.visibility = visibility
        cameraButton.visibility = visibility
        confidenceLabel.visibility = visibility
    }
    
    fun setZoomLevel(zoomLevel: Float) {
        camera?.let { cam: Camera ->
            // Clamp zoom level between min and max
            val clampedZoomRatio = zoomLevel.coerceIn(minZoomRatio, cam.cameraInfo.zoomState.value?.maxZoomRatio ?: maxZoomRatio)

            cam.cameraControl.setZoomRatio(clampedZoomRatio)
            currentZoomRatio = clampedZoomRatio

            // Notify zoom change
            onZoomChanged?.invoke(currentZoomRatio)

            // Emit zoom event so Dart-side ZoomIndicator stays in sync.
            emitEvent(mapOf("type" to "zoom", "value" to clampedZoomRatio.toDouble()))

            // Mirror iOS upstream updateSelectedLens: if the new zoom crosses a lens
            // boundary, swap CameraSelector to the matching physical lens.
            maybeSnapLensForZoom(clampedZoomRatio.toDouble())
        }
    }

    /**
     * If the requested zoom factor maps onto a different physical back-camera lens than
     * the currently selected one, switch CameraSelector and emit a `lens` event. Same
     * thresholds as iOS upstream `updateSelectedLens` (largest lens whose zoomFactor is
     * <= requested wins; ties broken by the smallest lens).
     */
    private fun maybeSnapLensForZoom(zoomFactor: Double) {
        if (lensFacing != CameraSelector.LENS_FACING_BACK) return
        val lenses = cachedLenses.filter { it.cameraInfo != null }
        if (lenses.size < 2) return

        val sorted = lenses.sortedBy { it.zoomFactor }
        val target = sorted.lastOrNull { zoomFactor >= it.zoomFactor - 0.01 } ?: sorted.first()

        val currentLens = selectedLensCameraInfo
        if (currentLens == target.cameraInfo) return

        try {
            switchToLens(target)
            emitEvent(mapOf("type" to "lens", "label" to target.label))
        } catch (e: Exception) {
            Log.w(TAG, "Lens snap to ${target.label} failed", e)
        }
    }

    fun setTorchMode(enabled: Boolean) {
        camera?.let { cam ->
            if (cam.cameraInfo.hasFlashUnit()) {
                cam.cameraControl.enableTorch(enabled)
            }
        }
    }

    // endregion

    // region Model / Task

    fun setModel(modelPath: String, task: YOLOTask, useGpu: Boolean = true, callback: ((Boolean) -> Unit)? = null) {
        Executors.newSingleThreadExecutor().execute {
            try {
                val newPredictor = when (task) {
                    YOLOTask.DETECT -> ObjectDetector(context = context, modelPath = modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.SEGMENT -> Segmenter(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.SEMANTIC -> SemanticSegmenter(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.CLASSIFY -> Classifier(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.POSE -> PoseEstimator(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                    YOLOTask.OBB -> ObbDetector(context, modelPath, labels = loadLabels(modelPath), useGpu = useGpu)
                }

                // Apply thresholds to all predictor types
                newPredictor.apply {
                    setConfidenceThreshold(confidenceThreshold)
                    setIouThreshold(iouThreshold)
                    setNumItemsThreshold(numItemsThreshold)
                }

                post {
                    this.task = task
                    this.predictor = newPredictor
                    this.modelName = modelPath.substringAfterLast("/")
                    modelLoadCallback?.invoke(true)
                    callback?.invoke(true)
                    // Ensure camera starts after model loads if it's not already running
                    if (allPermissionsGranted() && lifecycleOwner != null && (camera == null || isStopped)) {
                        startCamera()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load model: $modelPath. Camera will run without inference.", e)
                post {
                    // Clear predictor so the camera can keep running until a valid model is set
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
            return loadedLabels
        }

        // Return COCO dataset's 80 classes as a fallback
        // This is much more complete than the previous 7-class hardcoded list
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
        owner.lifecycle.addObserver(this)
        
        if (allPermissionsGranted() && (camera == null || isStopped)) {
            startCamera()
        }
    }
    
    // region camera init

    fun initCamera() {
        if (allPermissionsGranted()) {
            if (lifecycleOwner != null && (camera == null || isStopped)) {
                startCamera()
            }
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
        isStopped = false

        try {
            cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()

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

                    val cameraSelector = buildCameraSelector(cameraProvider)

                    cameraProvider.unbindAll()

                    try {
                        val owner = lifecycleOwner
                        if (owner == null) {
                            Log.e(TAG, "No LifecycleOwner available. Call onLifecycleOwnerAvailable() first.")
                            return@addListener
                        }

                        // Refresh lens enumeration once we have a camera provider.
                        cachedLenses = computeLensInfos(cameraProvider)

                        // Preferred path: bind Preview + ImageAnalysis + ImageCapture so
                        // capturePhoto() can grab a full-resolution still. Some low-tier
                        // devices cannot bind three use-cases simultaneously; in that
                        // case fall back to Preview + ImageAnalysis only and rely on
                        // captureFrame() for snapshots.
                        imageCaptureUseCase = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            .build()

                        camera = try {
                            cameraProvider.bindToLifecycle(
                                owner,
                                cameraSelector,
                                previewUseCase,
                                imageAnalysisUseCase,
                                imageCaptureUseCase
                            )
                        } catch (e: IllegalArgumentException) {
                            Log.w(TAG, "Three-use-case binding failed, falling back without ImageCapture", e)
                            imageCaptureUseCase = null
                            cameraProvider.bindToLifecycle(
                                owner,
                                cameraSelector,
                                previewUseCase,
                                imageAnalysisUseCase
                            )
                        }

                        // Reset zoom to 1.0x when camera starts
                        currentZoomRatio = 1.0f
                        onZoomChanged?.invoke(currentZoomRatio)

                        previewUseCase?.setSurfaceProvider(previewView.surfaceProvider)

                        // Initialize zoom
                        camera?.let { cam: Camera ->
                            val cameraInfo = cam.cameraInfo
                            minZoomRatio = cameraInfo.zoomState.value?.minZoomRatio ?: 1.0f
                            maxZoomRatio = cameraInfo.zoomState.value?.maxZoomRatio ?: 1.0f
                            currentZoomRatio = cameraInfo.zoomState.value?.zoomRatio ?: 1.0f
                        }
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

    private fun buildCameraSelector(cameraProvider: ProcessCameraProvider): CameraSelector {
        // If the caller explicitly picked a lens via setLens(), honor it (back-camera only).
        if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            selectedLensCameraInfo?.let { target ->
                if (cameraProvider.availableCameraInfos.contains(target)) {
                    return CameraSelector.Builder()
                        .addCameraFilter { infos -> infos.filter { it == target } }
                        .build()
                }
            }
        }

        if (lensFacing != CameraSelector.LENS_FACING_BACK || !preferWideBackCamera) {
            return CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()
        }

        return selectWidestBackCamera(cameraProvider)?.let { wideCamera ->
            CameraSelector.Builder()
                .addCameraFilter { cameraInfos ->
                    cameraInfos.filter { it == wideCamera }
                }
                .build()
        } ?: CameraSelector.DEFAULT_BACK_CAMERA
    }

    private fun selectWidestBackCamera(cameraProvider: ProcessCameraProvider): CameraInfo? {
        return cameraProvider.availableCameraInfos
            .mapNotNull { cameraInfo ->
                try {
                    val camera2Info = Camera2CameraInfo.from(cameraInfo)
                    val facing = camera2Info.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                    if (facing != CameraCharacteristics.LENS_FACING_BACK) return@mapNotNull null

                    val focalLength = camera2Info
                        .getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.minOrNull() ?: return@mapNotNull null
                    cameraInfo to focalLength
                } catch (e: Exception) {
                    Log.w(TAG, "Skipping camera with unreadable metadata", e)
                    null
                }
            }
            .minByOrNull { it.second }
            ?.first
    }

    fun setLensFacing(facing: Int, preferWideBackCamera: Boolean = false) {
        lensFacing = facing
        this.preferWideBackCamera = preferWideBackCamera && facing == CameraSelector.LENS_FACING_BACK
        // Restart camera if already started
        if (::cameraProviderFuture.isInitialized) {
            startCamera()
        }
    }

    fun switchCamera() {
        preferWideBackCamera = false
        // Clear any sticky lens selection when the user explicitly flips cameras.
        selectedLensCameraInfo = null
        selectedLensZoomFactor = null
        selectedLensLabel = null
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    // endregion

    // region multi-lens / focus / capture (Dart-driven setters)

    /**
     * Enumerate back-facing physical lenses with an equivalent zoom factor relative to
     * the main wide-angle (1.0x). Mirrors the iOS app's `AVCaptureDevice.DiscoverySession`
     * output. If the device exposes only one back camera, returns a single "Default" entry.
     */
    fun enumerateLenses(): List<LensInfo> {
        if (cachedLenses.isNotEmpty()) return cachedLenses
        return try {
            val provider = ProcessCameraProvider.getInstance(context).get(1, TimeUnit.SECONDS)
            computeLensInfos(provider).also { cachedLenses = it }
        } catch (e: Exception) {
            Log.w(TAG, "enumerateLenses: cameraProvider unavailable", e)
            emptyList()
        }
    }

    private fun computeLensInfos(cameraProvider: ProcessCameraProvider): List<LensInfo> {
        // Read focal lengths for each back-facing camera and compute an equivalent zoom
        // factor relative to the widest standard lens (which we treat as the 1.0x
        // reference, matching how iOS scales relative to the main wide-angle).
        // SENSOR_INFO_PHYSICAL_SIZE is also queried so we degrade gracefully on devices
        // that omit one or the other characteristic.
        data class Raw(val info: CameraInfo, val focalLength: Float, val sensorWidth: Float)

        val raws = cameraProvider.availableCameraInfos.mapNotNull { info ->
            try {
                val c2 = Camera2CameraInfo.from(info)
                val facing = c2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) return@mapNotNull null
                val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.minOrNull() ?: return@mapNotNull null
                val sensor: SizeF? = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                Raw(info, focal, sensor?.width ?: 0f)
            } catch (e: Exception) {
                Log.w(TAG, "computeLensInfos: skipping camera with unreadable metadata", e)
                null
            }
        }

        if (raws.isEmpty()) return emptyList()
        if (raws.size == 1) {
            return listOf(LensInfo(zoomFactor = 1.0, label = "Default", cameraInfo = raws[0].info))
        }

        // The "main" lens is the widest non-ultrawide lens. Heuristic: among lenses whose
        // 35mm-equivalent focal length is around 24-35mm, pick the one with the smallest
        // focal-length value that is NOT classified as ultra-wide. Practically we treat
        // the lens with the median focal length as main; lenses with shorter focal length
        // are ultra-wide (~0.5x), longer focal length lenses are telephoto.
        val sorted = raws.sortedBy { it.focalLength }
        val mainIdx = if (sorted.size >= 3) 1 else 0  // ultrawide, wide, tele -> pick wide
        val mainFocal = sorted[mainIdx].focalLength

        return sorted.map { raw ->
            val zoom: Double
            val label: String
            when {
                raw.focalLength < mainFocal - 0.01f -> {
                    // Ultra-wide: iOS exposes these as 0.5x relative to the main lens.
                    zoom = (raw.focalLength.toDouble() / mainFocal.toDouble()).coerceAtLeast(0.1)
                    // Snap to 0.5x if the math lands close (Android focal-length ratios
                    // typically come out near 0.5 for true ultra-wide modules).
                    val rounded = if (abs(zoom - 0.5) < 0.15) 0.5 else zoom
                    label = "Ultra wide camera"
                    LensInfo(zoomFactor = rounded, label = label, cameraInfo = raw.info)
                }
                raw.focalLength > mainFocal + 0.01f -> {
                    zoom = raw.focalLength.toDouble() / mainFocal.toDouble()
                    label = "Telephoto camera"
                    LensInfo(zoomFactor = zoom, label = label, cameraInfo = raw.info)
                }
                else -> {
                    LensInfo(zoomFactor = 1.0, label = "Wide camera", cameraInfo = raw.info)
                }
            }
        }
    }

    /**
     * Switch the active back-camera lens to the one whose computed zoom factor is closest
     * to [zoomFactor]. Emits a `{type:"lens",label}` event on the existing event sink.
     */
    fun setLens(zoomFactor: Double) {
        val lenses = if (cachedLenses.isEmpty()) enumerateLenses() else cachedLenses
        if (lenses.isEmpty()) return
        val target = lenses.minByOrNull { abs(it.zoomFactor - zoomFactor) } ?: return
        switchToLens(target)
        emitEvent(mapOf("type" to "lens", "label" to target.label))
    }

    private fun switchToLens(target: LensInfo) {
        selectedLensCameraInfo = target.cameraInfo
        selectedLensZoomFactor = target.zoomFactor
        selectedLensLabel = target.label
        // Switching lenses always means we're staying on the back side.
        lensFacing = CameraSelector.LENS_FACING_BACK
        preferWideBackCamera = false
        // Rebind so the new CameraSelector is honored.
        startCamera()
    }

    /**
     * Tap-to-focus. [x] and [y] are normalized view-relative coordinates in 0..1.
     * Builds a FocusMeteringAction via the PreviewView's MeteringPointFactory and
     * triggers AF/AE. Emits `{type:"focus",x,y}` when the future completes successfully
     * so the Dart `FocusReticle` can animate.
     */
    fun tapToFocus(x: Double, y: Double) {
        val cam = camera ?: return
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return
        val nx = x.toFloat().coerceIn(0f, 1f)
        val ny = y.toFloat().coerceIn(0f, 1f)
        try {
            val factory = previewView.meteringPointFactory
            val point = factory.createPoint(nx * w, ny * h)
            val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(3, TimeUnit.SECONDS)
                .build()
            val future = cam.cameraControl.startFocusAndMetering(action)
            future.addListener({
                try {
                    val result = future.get()
                    if (result.isFocusSuccessful) {
                        post {
                            emitEvent(mapOf("type" to "focus", "x" to nx.toDouble(), "y" to ny.toDouble()))
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "tapToFocus: focus future failed", e)
                }
            }, ContextCompat.getMainExecutor(context))
        } catch (e: Exception) {
            Log.e(TAG, "tapToFocus failed", e)
        }
    }

    /**
     * Capture a still photo. Preferred path uses the bound ImageCapture use-case so we
     * get a full-resolution JPEG; if [withOverlays] is true the current overlay bitmap
     * is composited on top of the still before re-encoding. If ImageCapture binding
     * isn't available (e.g. three-use-case bind failed), falls back to [captureFrame]
     * which snapshots the preview + overlay composite.
     */
    fun capturePhoto(withOverlays: Boolean = true, callback: (ByteArray?) -> Unit) {
        val ic = imageCaptureUseCase
        if (ic == null) {
            callback(captureFrame())
            return
        }
        try {
            ic.takePicture(
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            val jpegBytes = imageProxyToJpegBytes(image)
                            if (jpegBytes == null) {
                                callback(captureFrame())
                                return
                            }
                            if (!withOverlays) {
                                callback(jpegBytes)
                                return
                            }
                            // Composite the current overlay bitmap on top of the still.
                            val composed = compositeOverlayOnJpeg(jpegBytes)
                            callback(composed ?: jpegBytes)
                        } catch (e: Exception) {
                            Log.e(TAG, "capturePhoto: error processing capture", e)
                            callback(captureFrame())
                        } finally {
                            image.close()
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.w(TAG, "capturePhoto: ImageCapture failed, falling back to captureFrame", exception)
                        callback(captureFrame())
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "capturePhoto: takePicture threw, falling back", e)
            callback(captureFrame())
        }
    }

    private fun imageProxyToJpegBytes(image: ImageProxy): ByteArray? {
        return try {
            val plane = image.planes[0]
            val buffer = plane.buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            // ImageCapture (JPEG format) hands us a JPEG buffer directly.
            if (image.format == ImageFormat.JPEG || image.format == 256 /* JPEG */) {
                bytes
            } else {
                // Fallback: convert via Bitmap (rare path).
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
                val out = java.io.ByteArrayOutputStream()
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
                out.toByteArray()
            }
        } catch (e: Exception) {
            Log.e(TAG, "imageProxyToJpegBytes failed", e)
            null
        }
    }

    private fun compositeOverlayOnJpeg(jpegBytes: ByteArray): ByteArray? {
        return try {
            val still = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
            // Render overlay onto a bitmap sized to match the still.
            val composite = Bitmap.createBitmap(still.width, still.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(composite)
            canvas.drawBitmap(still, 0f, 0f, null)
            // Capture the overlay at its current view size and scale it to the still.
            val overlayBitmap = Bitmap.createBitmap(overlayView.width.coerceAtLeast(1), overlayView.height.coerceAtLeast(1), Bitmap.Config.ARGB_8888)
            overlayView.draw(Canvas(overlayBitmap))
            val matrix = Matrix().apply {
                setScale(still.width.toFloat() / overlayBitmap.width, still.height.toFloat() / overlayBitmap.height)
            }
            canvas.drawBitmap(overlayBitmap, matrix, null)
            val out = java.io.ByteArrayOutputStream()
            composite.compress(Bitmap.CompressFormat.JPEG, 90, out)
            still.recycle()
            overlayBitmap.recycle()
            composite.recycle()
            out.toByteArray()
        } catch (e: Exception) {
            Log.e(TAG, "compositeOverlayOnJpeg failed", e)
            null
        }
    }

    // endregion
    
    // Lifecycle methods from DefaultLifecycleObserver
    override fun onStart(owner: LifecycleOwner) {
        if (allPermissionsGranted()) {
            // Always restart camera on start if it's stopped or null
            // This ensures camera resumes when navigating back
            if (isStopped || camera == null) {
                startCamera()
            }
        }
    }

    override fun onResume(owner: LifecycleOwner) {
        if (allPermissionsGranted()) {
            // Double-check camera is running on resume
            if (isStopped || camera == null) {
                startCamera()
            }
        }
    }

    override fun onStop(owner: LifecycleOwner) {
        // Camera will be automatically stopped by CameraX when lifecycle stops
    }

    // region onFrame (per frame inference)

    private fun onFrame(imageProxy: ImageProxy) {
        // Early return if view is stopped to prevent accessing closed resources
        if (isStopped) {
            imageProxy.close()
            return
        }

        val w = imageProxy.width
        val h = imageProxy.height

        val bitmap = ImageUtils.toBitmap(imageProxy) ?: run {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap")
            imageProxy.close()
            return
        }

        // Check again after bitmap conversion (in case stop() was called during conversion)
        if (isStopped) {
            imageProxy.close()
            return
        }

        predictor?.let { p ->
            // Double-check stopped flag before inference (predictor might be closed)
            if (isStopped) {
                imageProxy.close()
                return
            }

            // Check if we should run inference on this frame
            if (!shouldRunInference()) {
                imageProxy.close()
                return
            }
            
            try {
                // Get device orientation
                val orientation = context.resources.configuration.orientation
                val isLandscape = orientation == Configuration.ORIENTATION_LANDSCAPE
                
                // Check if using front camera
                val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
                val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                val isRotated = rotationDegrees % 180 != 0
                val orientedWidth = if (isRotated) h else w
                val orientedHeight = if (isRotated) w else h
                
                // Set camera facing information in predictor
                (p as? BasePredictor)?.let { basePredictor ->
                    basePredictor.isFrontCamera = isFrontCamera
                    basePredictor.cameraRotationDegrees = rotationDegrees
                }
                
                val result = p.predict(
                    bitmap,
                    orientedWidth,
                    orientedHeight,
                    rotateForCamera = true,
                    isLandscape = isLandscape
                )

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

        private fun labelText(name: String, confidence: Float) =
            "$name ${"%.1f".format(confidence * 100)}"

        private fun colorFor(index: Int, confidence: Float): Int {
            val alpha = (confidence * 255).toInt().coerceIn((0.6f * 255).toInt(), 255)
            val baseColor = ultralyticsColors[index % ultralyticsColors.size]
            return Color.argb(
                alpha,
                Color.red(baseColor),
                Color.green(baseColor),
                Color.blue(baseColor)
            )
        }

        private fun drawLabel(
            canvas: Canvas,
            text: String,
            color: Int,
            anchorLeft: Float,
            anchorTop: Float,
            anchorRight: Float,
            viewWidth: Float,
            viewHeight: Float,
            centered: Boolean = false
        ) {
            paint.textSize = 40f
            val fm = paint.fontMetrics
            val textWidth = paint.measureText(text)
            val textHeight = fm.bottom - fm.top
            val pad = 8f
            val labelWidth = textWidth + 2 * pad
            val labelHeight = textHeight + 2 * pad
            var labelLeft = if (centered) (viewWidth - labelWidth) / 2 else anchorLeft
            var labelTop = if (centered) (viewHeight - labelHeight) / 2 else anchorTop - labelHeight
            var labelRight = labelLeft + labelWidth
            var labelBottom = labelTop + labelHeight

            if (labelTop < 0) {
                labelTop = anchorTop
                labelBottom = labelTop + labelHeight
            }
            if (labelLeft < 0) {
                labelLeft = 0f
                labelRight = labelWidth
            }
            if (labelRight > viewWidth) {
                labelRight = viewWidth
                labelLeft = maxOf(0f, anchorRight - labelWidth)
            }
            if (labelBottom > viewHeight) {
                labelBottom = viewHeight
                labelTop = labelBottom - labelHeight
            }

            val bgRect = RectF(labelLeft, labelTop, labelRight, labelBottom)
            paint.style = Paint.Style.FILL
            paint.color = color
            canvas.drawRoundRect(bgRect, BOX_CORNER_RADIUS, BOX_CORNER_RADIUS, paint)

            paint.color = Color.WHITE
            val centerY = (bgRect.top + bgRect.bottom) / 2
            val baseline = centerY - (fm.descent + fm.ascent) / 2
            canvas.drawText(text, bgRect.left + pad, baseline, paint)
        }

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
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val result = inferenceResult ?: return
            
            // Only draw overlays if showOverlays is true
            if (!showOverlays) {
                return
            }

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
                    for (box in result.boxes) {
                        val newColor = colorFor(box.index, box.conf)

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
                        
                        // Flip horizontally for front camera (DETECT task)
                        if (isFrontCamera) {
                            val flippedLeft = vw - right
                            val flippedRight = vw - left
                            left = flippedLeft
                            right = flippedRight
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        drawLabel(canvas, labelText(box.cls, box.conf), newColor, left, top, right, vw, vh)
                    }
                }
                // ----------------------------------------
                // SEGMENT
                // ----------------------------------------
                YOLOTask.SEGMENT -> {
                    // Bounding boxes & labels
                    for (box in result.boxes) {
                        val newColor = colorFor(box.index, box.conf)

                        // Draw bounding box
                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // For front camera POSE, apply horizontal flip
                        if (isFrontCamera) {
                            // Flip horizontally
                            val flippedLeft = vw - right
                            val flippedRight = vw - left
                            left = flippedLeft
                            right = flippedRight
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        drawLabel(canvas, labelText(box.cls, box.conf), newColor, left, top, right, vw, vh)
                    }

                    // Segmentation mask
                    result.masks?.combinedMask?.let { maskBitmap ->
                        val src = Rect(0, 0, maskBitmap.width, maskBitmap.height)
                        val dst = RectF(dx, dy, dx + scaledW, dy + scaledH)
                        val maskPaint = Paint().apply {
                            alpha = 128
                            isFilterBitmap = true
                        }
                        
                        if (isFrontCamera) {
                            // For front camera, flip the mask horizontally
                            canvas.save()
                            // Translate to center, flip horizontally, translate back
                            canvas.translate(vw / 2f, 0f)
                            canvas.scale(-1f, 1f)
                            canvas.translate(-vw / 2f, 0f)
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                            canvas.restore()
                        } else {
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                        }
                    }
                }
                // ----------------------------------------
                // SEMANTIC
                // ----------------------------------------
                YOLOTask.SEMANTIC -> {
                    result.semanticMask?.maskImage?.let { maskBitmap ->
                        val src = Rect(0, 0, maskBitmap.width, maskBitmap.height)
                        val dst = RectF(dx, dy, dx + scaledW, dy + scaledH)
                        val maskPaint = Paint().apply {
                            alpha = 128
                            isFilterBitmap = true
                        }

                        if (isFrontCamera) {
                            canvas.save()
                            canvas.translate(vw / 2f, 0f)
                            canvas.scale(-1f, 1f)
                            canvas.translate(-vw / 2f, 0f)
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                            canvas.restore()
                        } else {
                            canvas.drawBitmap(maskBitmap, src, dst, maskPaint)
                        }
                    }
                }
                // ----------------------------------------
                // CLASSIFY
                // ----------------------------------------
                YOLOTask.CLASSIFY -> {
                    result.probs?.let { probs ->
                        val newColor = colorFor(probs.top1Index, probs.top1Conf)

                        drawLabel(
                            canvas,
                            labelText(probs.top1Label, probs.top1Conf),
                            newColor,
                            16f,
                            16f,
                            16f,
                            vw,
                            vh,
                            centered = true
                        )
                    }
                }
                // ----------------------------------------
                // POSE
                // ----------------------------------------
                YOLOTask.POSE -> {
                    // Bounding boxes
                    for (box in result.boxes) {
                        val newColor = colorFor(box.index, box.conf)

                        var left   = box.xywh.left   * scale + dx
                        var top    = box.xywh.top    * scale + dy
                        var right  = box.xywh.right  * scale + dx
                        var bottom = box.xywh.bottom * scale + dy
                        
                        // For front camera POSE, apply horizontal flip
                        if (isFrontCamera) {
                            // Flip horizontally
                            val flippedLeft = vw - right
                            val flippedRight = vw - left
                            left = flippedLeft
                            right = flippedRight
                        }

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            left, top, right, bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )
                        
                        drawLabel(canvas, labelText(box.cls, box.conf), newColor, left, top, right, vw, vh)
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
                                var px = pxCam * scale + dx
                                var py = pyCam * scale + dy
                                
                                // For front camera POSE, apply horizontal flip
                                if (isFrontCamera) {
                                    px = vw - px  // Flip horizontally
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
                        val newColor = colorFor(obbRes.index, obbRes.confidence)

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH

                        // Draw rotated rectangle (polygon) using path
                        val polygon = obbRes.box.toPolygon(iw, ih).map { pt ->
                            var x = pt.x * scaledW + dx
                            val y = pt.y * scaledH + dy
                            
                            // Flip horizontally for front camera
                            if (isFrontCamera) {
                                x = vw - x
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

                            // Find bounding box of the OBB polygon
                            val minX = polygon.map { it.x }.minOrNull() ?: 0f
                            val maxX = polygon.map { it.x }.maxOrNull() ?: 0f
                            val minY = polygon.map { it.y }.minOrNull() ?: 0f
                            drawLabel(
                                canvas,
                                labelText(obbRes.cls, obbRes.confidence),
                                newColor,
                                minX,
                                minY,
                                maxX,
                                vw,
                                vh
                            )
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
                }
            } ?: run {
                targetFrameInterval = null
            }

            // Setup throttleInterval (for result output)
            config.throttleIntervalMs?.let { throttleMs ->
                if (throttleMs > 0) {
                    throttleInterval = throttleMs * 1_000_000L // Convert ms to nanoseconds
                }
            } ?: run {
                throttleInterval = null
            }

            // Setup inference frequency control
            config.inferenceFrequency?.let { inferenceFreq ->
                if (inferenceFreq > 0) {
                    inferenceFrameInterval = (1_000_000_000L / inferenceFreq) // Convert to nanoseconds
                }
            } ?: run {
                inferenceFrameInterval = null
            }

            // Setup frame skipping
            config.skipFrames?.let { skipFrames ->
                if (skipFrames > 0) {
                    targetSkipFrames = skipFrames
                    frameSkipCount = 0 // Reset counter
                }
            } ?: run {
                targetSkipFrames = 0
                frameSkipCount = 0
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
     * Flattens keypoints data into a single array format: [x1, y1, conf1, x2, y2, conf2, ...]
     */
    private fun flattenKeypoints(keypoints: Keypoints): List<Double> {
        val flattened = mutableListOf<Double>()
        for (i in keypoints.xy.indices) {
            flattened.add(keypoints.xy[i].first.toDouble())
            flattened.add(keypoints.xy[i].second.toDouble())
            val confidence = if (i < keypoints.conf.size) {
                keypoints.conf[i].toDouble()
            } else {
                0.0
            }
            flattened.add(confidence)
        }
        return flattened
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

            if (config.includePoses && result.keypointsList.isNotEmpty() && result.boxes.isEmpty()) {
                for ((poseIndex, keypoints) in result.keypointsList.withIndex()) {
                    val detection = HashMap<String, Any>()
                    detection["classIndex"] = 0
                    detection["className"] = "person"
                    detection["confidence"] = 1.0
                    var minX = Float.MAX_VALUE
                    var minY = Float.MAX_VALUE
                    var maxX = Float.MIN_VALUE
                    var maxY = Float.MIN_VALUE
                    
                    for (kp in keypoints.xy) {
                        if (kp.first > 0 && kp.second > 0) {
                            minX = minOf(minX, kp.first)
                            minY = minOf(minY, kp.second)
                            maxX = maxOf(maxX, kp.first)
                            maxY = maxOf(maxY, kp.second)
                        }
                    }
                    val boundingBox = HashMap<String, Any>()
                    boundingBox["left"] = minX.toDouble()
                    boundingBox["top"] = minY.toDouble()
                    boundingBox["right"] = maxX.toDouble()
                    boundingBox["bottom"] = maxY.toDouble()
                    detection["boundingBox"] = boundingBox
                    
                    // Normalized bounding box
                    val normalizedBox = HashMap<String, Any>()
                    normalizedBox["left"] = (minX / result.origShape.width).toDouble()
                    normalizedBox["top"] = (minY / result.origShape.height).toDouble()
                    normalizedBox["right"] = (maxX / result.origShape.width).toDouble()
                    normalizedBox["bottom"] = (maxY / result.origShape.height).toDouble()
                    detection["normalizedBox"] = normalizedBox
                    
                    val keypointsFlat = flattenKeypoints(keypoints)
                    detection["keypoints"] = keypointsFlat

                    detections.add(detection)
                }
            }
            
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
                }
                
                // Add pose keypoints (if available and enabled)
                if (config.includePoses && result.keypointsList.isNotEmpty()) {
                    if (detectionIndex < result.keypointsList.size) {
                        val keypoints = result.keypointsList[detectionIndex]
                        val keypointsFlat = flattenKeypoints(keypoints)
                        detection["keypoints"] = keypointsFlat
                    }
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
                val imgWidth = result.origShape.width.toFloat()
                val imgHeight = result.origShape.height.toFloat()
                val polygon = obbRes.box.toPolygon(imgWidth, imgHeight)
                
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
                }
                
                detections.add(detection)
            }
            
            map["detections"] = detections
        }

        if (config.includeMasks) {
            result.semanticMask?.let { semanticMask ->
                map["semanticMask"] = mapOf(
                    "classMap" to semanticMask.classMap,
                    "width" to semanticMask.width,
                    "height" to semanticMask.height
                )
            }
        }

        // Add classification results (if available and enabled for CLASSIFY task)
        if (config.includeClassifications && result.probs != null && result.boxes.isEmpty()) {
            val probs = result.probs!!

            val top5Count = minOf(
                probs.top5Indices.size,
                probs.top5Labels.size,
                probs.top5Confs.size
            )
            val top5List = (0 until top5Count).map { index ->
                val classIdx = probs.top5Indices[index]
                val name = probs.top5Labels[index]
                val conf = probs.top5Confs[index]
                mapOf(
                    "class" to classIdx,
                    "name" to name,
                    "confidence" to conf.toDouble()
                )
            }

            // Add classification result to detections array (for compatibility with YOLOResult.fromMap)
            val detections = (map["detections"] as? List<Map<String, Any>>)?.toMutableList() ?: ArrayList()

            val classificationDetection = HashMap<String, Any>()
            classificationDetection["class"] = probs.top1Index
            classificationDetection["name"] = probs.top1Label
            classificationDetection["confidence"] = probs.top1Conf.toDouble()
            classificationDetection["top5"] = top5List

            // Full image bounding box for classification
            val boundingBox = HashMap<String, Any>()
            boundingBox["left"] = 0.0
            boundingBox["top"] = 0.0
            boundingBox["right"] = result.origShape.width.toDouble()
            boundingBox["bottom"] = result.origShape.height.toDouble()
            classificationDetection["boundingBox"] = boundingBox

            // Normalized bounding box (full image)
            val normalizedBox = HashMap<String, Any>()
            normalizedBox["left"] = 0.0
            normalizedBox["top"] = 0.0
            normalizedBox["right"] = 1.0
            normalizedBox["bottom"] = 1.0
            classificationDetection["normalizedBox"] = normalizedBox

            detections.add(classificationDetection)
            map["detections"] = detections
        }
        
        // Add performance metrics (if enabled)
        if (config.includeProcessingTimeMs) {
            val processingTimeMs = result.speed.toDouble()
            map["processingTimeMs"] = processingTimeMs
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
        // Set stopped flag first to prevent new frames from being processed
        isStopped = true

        try {
            imageAnalysisUseCase?.clearAnalyzer()
            if (::cameraProviderFuture.isInitialized) {
                try {
                    val cameraProvider = cameraProviderFuture.get(1, TimeUnit.SECONDS)
                    cameraProvider.unbindAll()
                } catch (e: Exception) {
                    Log.e(TAG, "Error getting camera provider for unbind", e)
                }
            }

            imageAnalysisUseCase = null
            imageCaptureUseCase = null

            previewUseCase?.setSurfaceProvider(null)
            previewUseCase = null

            cameraExecutor?.let { exec ->
                exec.shutdown()
                try {
                    if (!exec.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                        Log.w(TAG, "Executor didn't shut down in time; forcing shutdown")
                        exec.shutdownNow()
                        if (!exec.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                            Log.e(TAG, "Executor failed to terminate after forced shutdown")
                        }
                    }
                } catch (e: InterruptedException) {
                    Log.e(TAG, "Interrupted while waiting for executor shutdown", e)
                    exec.shutdownNow()
                    Thread.currentThread().interrupt()
                }
            }
            cameraExecutor = null

            camera = null
            
            // Close predictor safely - ensure no inference is running
            try {
                (predictor as? BasePredictor)?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing predictor", e)
            }
            predictor = null
            inferenceCallback = null
            streamCallback = null
            inferenceResult = null
        } catch (e: Exception) {
            Log.e(TAG, "Error during YOLOView stop", e)
        }
    }

}
