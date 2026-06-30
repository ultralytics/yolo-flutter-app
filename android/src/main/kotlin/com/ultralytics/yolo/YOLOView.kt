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
import android.hardware.camera2.CameraManager
import android.os.Build
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
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
import java.util.concurrent.atomic.AtomicInteger
import android.content.res.Configuration

/**
 * Describes a back-camera lens with its equivalent zoom factor relative to the main wide-angle lens (1.0x). Used by
 * `getAvailableLenses` to populate the Dart lens picker.
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
        val resolutionChanged = config?.analysisWidth != streamConfig?.analysisWidth ||
            config?.analysisHeight != streamConfig?.analysisHeight
        this.streamConfig = config
        setupThrottlingFromConfig()
        if (resolutionChanged && camera != null) {
            startCamera() // rebind so the new analysis resolution takes effect
        }
    }

    /** Set streaming callback */
    fun setStreamCallback(callback: ((Map<String, Any>) -> Unit)?) {
        this.streamCallback = callback
    }

    // Generic event callback used to forward {type:"zoom"|"lens"|"focus", ...} maps to the Flutter event sink without
    // coupling YOLOView to a Flutter type.
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
    private var targetRotation = Surface.ROTATION_0

    // Bumped on every camera (re)bind. A rebind path (lens snap / camera flip / resume) shuts down the old executor
    // without awaiting it, so an in-flight analyzer frame on the old thread can still be running when the new analyzer
    // goes live; stale frames check this and bail so two predict() calls never overlap on the non-thread-safe predictor.
    private val cameraGeneration = AtomicInteger(0)
    
    // Flag to track if the view is stopped/disposed to prevent race conditions
    @Volatile
    private var isStopped = false

    // Distinguishes an intentional Dart-driven pause (pauseCamera) from a lifecycle stop. Both set isStopped, but a
    // lifecycle onStart/onResume must NOT auto-restart the camera while the app explicitly paused it.
    @Volatile
    private var intentionallyPaused = false

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
    // Effective zoom (relative to the wide-camera 1.0x reference) to emit after the next `startCamera()` rebind.
    // `startCamera()` resets physical zoom to 1.0x, but on a non-wide lens the user-facing effective zoom equals the
    // lens reference factor (e.g. 0.5x on ultra-wide, 2.0x on telephoto). Without this the ZoomIndicator/LensPicker
    // would snap back to 1.0x after every lens change.
    private var pendingEffectiveZoomToEmit: Double? = null
    private var pendingZoomRatioToApply: Float? = null

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
            text = "Confidence: 0.25"
            textSize = 20f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(200, 200, 100, 0))
            setPadding(15, 10, 15, 10)
            visibility = View.GONE
        }
        addView(confidenceLabel)
        confidenceLabel.elevation = 1000f
        // Dart owns gestures (pinch + tap) via Flutter GestureDetector in YOLOShowcase; native is setter-only. Do not
        // attach ScaleGestureDetector here.
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
        if (!show) {
            inferenceResult = null
        }
        post {
            overlayView.invalidate()
        }
    }
    
    fun setShowUIControls(show: Boolean) {
        showUIControls = show
        // Show/hide all UI controls
        val visibility = if (show) View.VISIBLE else View.GONE
        zoomLabel.visibility = visibility
        cameraButton.visibility = visibility
        confidenceLabel.visibility = visibility
    }
    
    /**
     * Apply an *effective* zoom (relative to the wide-camera 1.0x reference). The physical setZoomRatio applied to the
     * underlying camera is `effective / selectedLensZoomFactor`, so on telephoto the same effective 2.0x produces
     * physical 1.0x (the tele's native FOV). Auto lens snap happens here before the digital zoom is applied.
     */
    fun setZoomLevel(zoomLevel: Float) {
        // First check whether the effective zoom should switch us to a different physical lens. `maybeSnapLensForZoom`
        // will rebind and emit the post-rebind zoom event itself; bail out so we don't apply digital zoom to the old
        // lens that's about to die.
        if (maybeSnapLensForZoom(zoomLevel.toDouble())) return

        camera?.let { cam: Camera ->
            val lensFactor = (selectedLensZoomFactor ?: 1.0).toFloat()
            val physical = (zoomLevel / lensFactor).coerceIn(
                minZoomRatio,
                cam.cameraInfo.zoomState.value?.maxZoomRatio ?: maxZoomRatio
            )
            cam.cameraControl.setZoomRatio(physical)
            currentZoomRatio = physical

            val effective = (physical * lensFactor).toDouble()

            // Notify zoom change (legacy callback uses physical ratio).
            onZoomChanged?.invoke(physical)

            // Dart-side ZoomIndicator consumes effective zoom so the value is consistent across lens switches.
            emitEvent(mapOf("type" to "zoom", "value" to effective))
        }
    }

    /**
     * If the requested effective zoom maps onto a different physical back-camera lens than the currently selected one,
     * switch CameraSelector and emit a `lens` event. Returns `true` when a snap was triggered (callers should not also
     * apply digital zoom on the about-to-rebind lens). Same thresholds as iOS upstream `updateSelectedLens` (largest
     * lens whose zoomFactor is <= requested wins; ties broken by the smallest lens).
     */
    private fun maybeSnapLensForZoom(zoomFactor: Double): Boolean {
        if (lensFacing != CameraSelector.LENS_FACING_BACK) return false
        val lenses = cachedLenses.filter { it.cameraInfo != null }
        if (lenses.size < 2) return false

        val sorted = lenses.sortedBy { it.zoomFactor }
        val target = sorted.lastOrNull { zoomFactor >= it.zoomFactor - 0.01 } ?: sorted.first()

        // Skip rebind if we're already on the target lens. When `selectedLensCameraInfo` is null (first frame after the
        // back camera bound), fall back to identifying the lens by matching cameraInfo against the currently-bound
        // camera so a first pinch on the wide lens doesn't trigger an unnecessary rebind.
        val currentInfo = selectedLensCameraInfo ?: camera?.cameraInfo
        if (currentInfo == target.cameraInfo) return false

        try {
            // Preserve the user-requested effective zoom across the rebind: the new lens starts at physical 1.0x =
            // effective `target.zoomFactor`, which is the same FOV the user was pinching toward.
            pendingEffectiveZoomToEmit = target.zoomFactor
            switchToLens(target)
            emitEvent(mapOf("type" to "lens", "label" to target.label))
            return true
        } catch (e: Exception) {
            Log.w(TAG, "Lens snap to ${target.label} failed", e)
            pendingEffectiveZoomToEmit = null
            return false
        }
    }

    // Turns the torch on/off and returns the actual resulting state (false when there is no flash unit), so callers
    // can keep their cached state in sync with the hardware.
    fun setTorchMode(enabled: Boolean): Boolean {
        camera?.let { cam ->
            if (cam.cameraInfo.hasFlashUnit()) {
                cam.cameraControl.enableTorch(enabled)
                return enabled
            }
        }
        return false
    }

    // endregion

    // region Model / Task

    // Recently-loaded predictors kept in memory so switching back to a model is instant instead of re-building the
    // TFLite interpreter every time. Bounded by predictorCacheLimit to cap memory. Accessed on the main thread
    // (setModel is called from the platform channel; cache writes happen inside post {}).
    private val predictorCache = HashMap<String, Predictor>()
    private val predictorCacheOrder = ArrayList<String>()  // LRU: oldest first, newest last
    private val predictorCacheLimit = 3

    private fun cachePredictor(key: String, predictor: Predictor) {
        val previous = predictorCache.put(key, predictor)
        if (previous != null && previous !== predictor) {
            closePredictor(previous)
        }
        predictorCacheOrder.remove(key)
        predictorCacheOrder.add(key)
        while (predictorCacheOrder.size > predictorCacheLimit) {
            // The current predictor is always newest (just touched), so it is never the eviction target.
            val evictedKey = predictorCacheOrder.removeAt(0)
            val evictedPredictor = predictorCache.remove(evictedKey)
            if (evictedPredictor != null && evictedPredictor !== predictor) {
                closePredictor(evictedPredictor)
            }
        }
    }

    private fun closePredictor(predictor: Predictor) {
        // Cache eviction/replacement runs on the main thread, but onFrame() calls predict() on this predictor on the
        // camera executor thread. Defer the close onto that same single thread so the native interpreter is never freed
        // while a frame is still mid-predict() (use-after-free). When no executor is live (e.g. during stop()) close
        // directly — stop() drains the executor before closing, so no inference can be in flight there.
        val exec = cameraExecutor
        if (exec != null && !exec.isShutdown) {
            exec.execute {
                try {
                    (predictor as? BasePredictor)?.close()
                } catch (e: Exception) {
                    Log.e(TAG, "Error closing cached predictor", e)
                }
            }
        } else {
            try {
                (predictor as? BasePredictor)?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing cached predictor", e)
            }
        }
    }

    fun setModel(modelPath: String, task: YOLOTask, useGpu: Boolean = true, callback: ((Boolean) -> Unit)? = null) {
        val cacheKey = "$modelPath|$task|$useGpu"
        inferenceResult = null
        post {
            overlayView.invalidate()
        }

        // Fast path: reuse an already-loaded predictor (re-applying the current thresholds) for an instant switch.
        predictorCache[cacheKey]?.let { cached ->
            cached.setConfidenceThreshold(confidenceThreshold)
            cached.setIouThreshold(iouThreshold)
            cached.setNumItemsThreshold(numItemsThreshold)
            post {
                this.task = task
                this.predictor = cached
                this.modelName = modelPath.substringAfterLast("/")
                cachePredictor(cacheKey, cached)
                modelLoadCallback?.invoke(true)
                callback?.invoke(true)
                if (allPermissionsGranted() && lifecycleOwner != null && (camera == null || isStopped)) {
                    startCamera()
                }
            }
            return
        }

        Executors.newSingleThreadExecutor().execute {
            try {
                val newPredictor = when (task) {
                    YOLOTask.DETECT -> ObjectDetector(context = context, modelPath = modelPath, labels = emptyList(), useGpu = useGpu)
                    YOLOTask.SEGMENT -> Segmenter(context, modelPath, labels = emptyList(), useGpu = useGpu)
                    YOLOTask.SEMANTIC -> SemanticSegmenter(context, modelPath, labels = emptyList(), useGpu = useGpu)
                    YOLOTask.CLASSIFY -> Classifier(context, modelPath, labels = emptyList(), useGpu = useGpu)
                    YOLOTask.POSE -> PoseEstimator(context, modelPath, labels = emptyList(), useGpu = useGpu)
                    YOLOTask.OBB -> ObbDetector(context, modelPath, labels = emptyList(), useGpu = useGpu)
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
                    cachePredictor(cacheKey, newPredictor)
                    modelLoadCallback?.invoke(true)
                    callback?.invoke(true)
                    // Ensure camera starts after model loads if it's not already running
                    if (allPermissionsGranted() && lifecycleOwner != null && (camera == null || isStopped)) {
                        startCamera()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load model: $modelPath. Keeping the previously loaded model if one is present.", e)
                post {
                    // The new predictor was built into a local and never assigned, so the previously loaded model is
                    // untouched. Only drop inference when there is nothing to fall back to (an initial-load failure);
                    // for an in-place switch failure keep the previous predictor running so the camera doesn't
                    // silently stop detecting while the UI reverts to the still-loaded model.
                    if (this.predictor == null) {
                        this.modelName = "No Model"
                    }
                    modelLoadCallback?.invoke(false)
                    callback?.invoke(false)
                }
            }
        }
    }

    // endregion

    private fun syncTargetRotation() {
        val rotation = previewView.display?.rotation ?: return
        if (rotation == targetRotation) return
        targetRotation = rotation
        previewUseCase?.targetRotation = rotation
        imageAnalysisUseCase?.targetRotation = rotation
        imageCaptureUseCase?.targetRotation = rotation
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        syncTargetRotation()
    }

    /**
     * Called when a LifecycleOwner is available for camera operations
     */
    fun onLifecycleOwnerAvailable(owner: LifecycleOwner) {
        // Detach from any previous owner before re-registering so a re-attach (or owner change) can't leave a stale
        // observer wired to this view, and so disposal can fully release it.
        this.lifecycleOwner?.lifecycle?.removeObserver(this)
        this.lifecycleOwner = owner
        owner.lifecycle.addObserver(this)

        if (allPermissionsGranted() && (camera == null || isStopped)) {
            startCamera()
        }
    }

    /**
     * Detach from the lifecycle owner. Called on platform-view disposal so the Activity's lifecycle no longer holds a
     * strong reference to this (now-dead) view — otherwise a later onStart/onResume would invoke startCamera() on it.
     */
    fun detachLifecycle() {
        lifecycleOwner?.lifecycle?.removeObserver(this)
        lifecycleOwner = null
    }

    /**
     * Pause the camera pipeline without tearing down the predictor.
     *
     * The "pause" method channel call routes here (not [stop]) so that "resume" -> [startCamera] can rebind: [stop]
     * closes and nulls the predictor, and [startCamera] early-returns while `predictor == null`, which would otherwise
     * leave the preview dead after a single pause/resume cycle. This only unbinds the camera use-cases and clears the
     * analyzer; the predictor, callbacks and cache stay intact for an instant resume.
     */
    fun pauseCamera() {
        isStopped = true
        intentionallyPaused = true
        try {
            imageAnalysisUseCase?.clearAnalyzer()
            if (::cameraProviderFuture.isInitialized) {
                try {
                    cameraProviderFuture.get(1, TimeUnit.SECONDS).unbindAll()
                } catch (e: Exception) {
                    Log.e(TAG, "Error unbinding camera on pause", e)
                }
            }
            previewUseCase?.setSurfaceProvider(null)
            camera = null
        } catch (e: Exception) {
            Log.e(TAG, "Error during camera pause", e)
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
        // Defer binding the camera until a model is loaded. Otherwise the preview starts on view-attach and the heavy
        // first GPU model compile runs while the preview is live, disrupting it. With this guard the camera binds
        // exactly once, from setModel's callback after the predictor is ready. setModel re-invokes startCamera once it
        // sets predictor.
        if (predictor == null) {
            return
        }
        isStopped = false
        // An explicit start/resume clears the intentional-pause flag so lifecycle events resume the camera again.
        intentionallyPaused = false

        try {
            cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()

                    // Stale-listener guard: if pauseCamera()/stop() ran after this listener was posted but before it
                    // fired (e.g. rapid pause during model load), bail so we don't rebind a camera that was just stopped.
                    if (isStopped) {
                        return@addListener
                    }

                    // Tear down the previous analyzer + executor before rebinding. Rebind paths (setLens/auto-snap,
                    // switchCamera, setLensFacing, onStart/onResume) reach startCamera() without going through stop(),
                    // and each call builds a fresh ImageAnalysis + executor below. Without this, the old executor's
                    // non-daemon analyzer thread and the old ImageAnalysis analyzer would be orphaned on every rebind.
                    imageAnalysisUseCase?.clearAnalyzer()
                    // Drain the old executor (not just shutdown) so any frame already inside onFrame()/predict() on the
                    // previous analyzer thread finishes before the new analyzer binds — the generation guard below only
                    // stops not-yet-started frames, and the predictor is not thread-safe. We're on the camera-provider
                    // listener thread here, not the analyzer thread, so awaiting does not self-deadlock.
                    cameraExecutor?.let { exec ->
                        exec.shutdown()
                        try {
                            if (!exec.awaitTermination(500, TimeUnit.MILLISECONDS)) {
                                exec.shutdownNow()
                            }
                        } catch (e: InterruptedException) {
                            exec.shutdownNow()
                            Thread.currentThread().interrupt()
                        }
                    }
                    cameraExecutor = null

                    targetRotation = previewView.display?.rotation ?: Surface.ROTATION_0

                    previewUseCase = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setTargetRotation(targetRotation)
                        .build()

                    val analysisBuilder = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetRotation(targetRotation)
                        // Ask CameraX for RGBA frames so toBitmap() is a direct buffer copy. The default YUV_420_888
                        // forced a per-frame JPEG encode@100 + decode round-trip (~100ms/frame, ~5 FPS).
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    val analysisWidth = streamConfig?.analysisWidth
                    val analysisHeight = streamConfig?.analysisHeight
                    if (analysisWidth != null && analysisHeight != null && analysisWidth > 0 && analysisHeight > 0) {
                        // Opt-in higher analysis resolution: by default CameraX delivers ~640x480 frames, which caps
                        // the detail reaching models with larger inputs. CameraX picks the nearest supported size.
                        analysisBuilder.setResolutionSelector(
                            ResolutionSelector.Builder()
                                .setResolutionStrategy(
                                    ResolutionStrategy(
                                        android.util.Size(analysisWidth, analysisHeight),
                                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                    )
                                )
                                .build()
                        )
                    } else {
                        if (analysisWidth != null || analysisHeight != null) {
                            Log.w(TAG, "Ignoring invalid analysisResolution ${analysisWidth}x${analysisHeight}")
                        }
                        analysisBuilder.setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    }
                    imageAnalysisUseCase = analysisBuilder.build()

                    cameraExecutor = Executors.newSingleThreadExecutor()
                    val myGeneration = cameraGeneration.incrementAndGet()
                    imageAnalysisUseCase!!.setAnalyzer(cameraExecutor!!) { imageProxy ->
                        // Drop frames from a superseded binding: the old executor may still deliver one in-flight frame
                        // after a rebind, and overlapping predict() calls on the shared predictor are not thread-safe.
                        if (myGeneration != cameraGeneration.get()) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
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

                        // Preferred path: bind Preview + ImageAnalysis + ImageCapture so capturePhoto() can grab a
                        // full-resolution still. Some low-tier devices cannot bind three use-cases simultaneously; in
                        // that case fall back to Preview + ImageAnalysis only and rely on captureFrame() for snapshots.
                        imageCaptureUseCase = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            .setTargetRotation(targetRotation)
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

                            // Sync the lens tracking state with whatever lens we actually bound to so the first pinch
                            // on the wide lens doesn't think it needs to rebind. Default the selectedLensZoomFactor to
                            // 1.0 (the wide reference) when we can't identify the bound camera in `cachedLenses` (e.g.
                            // front-camera path).
                            val bound = cachedLenses.firstOrNull { it.cameraInfo == cameraInfo }
                            if (bound != null) {
                                selectedLensCameraInfo = bound.cameraInfo
                                selectedLensZoomFactor = bound.zoomFactor
                                selectedLensLabel = bound.label
                            } else if (selectedLensZoomFactor == null) {
                                selectedLensZoomFactor = 1.0
                            }

                            pendingZoomRatioToApply?.let { zoom ->
                                pendingZoomRatioToApply = null
                                val physical = zoom.coerceIn(
                                    minZoomRatio,
                                    cameraInfo.zoomState.value?.maxZoomRatio ?: maxZoomRatio
                                )
                                cam.cameraControl.setZoomRatio(physical)
                                currentZoomRatio = physical
                                onZoomChanged?.invoke(physical)
                            }

                            // setLens() / auto-snap stashes the effective zoom that should appear in Dart after the
                            // rebind; emit it now so the ZoomIndicator/LensPicker stay consistent across the change.
                            pendingEffectiveZoomToEmit?.let { effective ->
                                pendingEffectiveZoomToEmit = null
                                emitEvent(mapOf("type" to "zoom", "value" to effective))
                                onZoomChanged?.invoke(currentZoomRatio)
                            }
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
        clearLensSelection()
        // Restart camera if already started
        if (::cameraProviderFuture.isInitialized) {
            startCamera()
        }
    }

    fun switchCamera() {
        preferWideBackCamera = false
        // Clear any sticky lens selection when the user explicitly flips cameras.
        clearLensSelection()
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        startCamera()
    }

    private fun clearLensSelection() {
        cachedLenses = emptyList()
        selectedLensCameraInfo = null
        selectedLensZoomFactor = null
        selectedLensLabel = null
        pendingEffectiveZoomToEmit = null
        pendingZoomRatioToApply = null
    }

    // endregion

    // region multi-lens / focus / capture (Dart-driven setters)

    /**
     * Enumerate physical lenses for the active camera side. Back cameras include public CameraX cameras plus Camera2
     * physical IDs from logical multi-camera devices; front cameras return their single active lens.
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
        data class Raw(val info: CameraInfo?, val focalLength: Float, val sensorWidth: Float)

        val publicInfos = cameraProvider.availableCameraInfos.mapNotNull { info ->
            try {
                val c2 = Camera2CameraInfo.from(info)
                val facing = c2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.minOrNull() ?: return@mapNotNull null
                val sensor: SizeF? = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                Triple(facing, c2.cameraId, Raw(info, focal, sensor?.width ?: 0f))
            } catch (e: Exception) {
                Log.w(TAG, "computeLensInfos: skipping camera with unreadable metadata", e)
                null
            }
        }

        if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            val front = publicInfos.firstOrNull { it.first == CameraCharacteristics.LENS_FACING_FRONT }
            return listOf(LensInfo(zoomFactor = 1.0, label = "Front camera", cameraInfo = front?.third?.info))
        }

        // Read physical IDs from Camera2 in addition to CameraX's public CameraInfo list. Samsung and other flagship
        // devices often expose telephoto lenses only as hidden physical cameras under a logical back camera; CameraX's
        // availableCameraInfos may therefore report ultra-wide + wide but omit telephoto.
        val publicInfoById = publicInfos.associate { it.second to it.third.info }
        val rawsById = linkedMapOf<String, Raw>()
        publicInfos
            .filter { it.first == CameraCharacteristics.LENS_FACING_BACK }
            .forEach { rawsById[it.second] = it.third }

        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        if (cameraManager != null) {
            for (id in cameraManager.cameraIdList) {
                try {
                    val chars = cameraManager.getCameraCharacteristics(id)
                    val facing = chars.get(CameraCharacteristics.LENS_FACING)
                    if (facing != CameraCharacteristics.LENS_FACING_BACK) continue
                    val physicalIds: Set<String> = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        chars.physicalCameraIds
                    } else {
                        emptySet<String>()
                    }
                    val ids = physicalIds.ifEmpty { setOf(id) }
                    for (physicalId in ids) {
                        val physicalChars = if (physicalId == id) {
                            chars
                        } else {
                            cameraManager.getCameraCharacteristics(physicalId)
                        }
                        val physicalFacing = physicalChars.get(CameraCharacteristics.LENS_FACING)
                        if (physicalFacing != CameraCharacteristics.LENS_FACING_BACK) continue
                        val focal = physicalChars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                            ?.minOrNull() ?: continue
                        val sensor = physicalChars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                        rawsById[physicalId] = Raw(
                            info = publicInfoById[physicalId],
                            focalLength = focal,
                            sensorWidth = sensor?.width ?: 0f
                        )
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "computeLensInfos: skipping Camera2 id $id", e)
                }
            }
        }

        val raws = rawsById.values.toList()
        if (raws.isEmpty()) return emptyList()
        if (raws.size == 1) {
            return listOf(LensInfo(zoomFactor = 1.0, label = "Wide camera", cameraInfo = raws[0].info))
        }

        // Convert each lens's focal length to a 35mm-equivalent by scaling against the full-frame sensor width (36mm).
        // When sensor width is unavailable we fall back to a synthetic equivalent based on raw focal length × a
        // typical smartphone crop factor (~7.0). Phone lens equivalents land roughly:
        //   ultra-wide: 13-20mm   wide: 22-32mm   telephoto: 50mm+
        fun equiv(raw: Raw): Float {
            val sensorWidth = raw.sensorWidth
            return if (sensorWidth > 0f) raw.focalLength * 36f / sensorWidth
            else raw.focalLength * 7f
        }

        val withEquiv = raws.map { it to equiv(it) }
        // Identify the main (wide) lens: closest to the 26mm ideal among lenses that aren't obviously ultra-wide. If
        // every lens is ultra-wide-ish, pick the longest focal as main.
        val mainRaw = withEquiv
            .filter { (_, e) -> e >= 21f }
            .minByOrNull { (_, e) -> abs(e - 26f) }
            ?.first
            ?: withEquiv.maxByOrNull { it.second }!!.first
        val mainEquiv = equiv(mainRaw)

        val deduped = withEquiv
            .sortedBy { it.second }
            .fold(mutableListOf<Pair<Raw, Float>>()) { acc, item ->
                val previous = acc.lastOrNull()
                val sameFocal = previous != null && abs(previous.second - item.second) < 1f
                if (sameFocal) {
                    // Prefer the public CameraX camera when a logical and physical ID describe the same lens.
                    if (previous!!.first.info == null && item.first.info != null) {
                        acc[acc.lastIndex] = item
                    }
                } else {
                    acc.add(item)
                }
                acc
            }

        val logicalWideInfo = deduped
            .firstOrNull { (raw, equivMm) -> raw.info != null && abs(equivMm - mainEquiv) < 1f }
            ?.first
            ?.info
        val logicalZoomState = logicalWideInfo?.zoomState?.value
        val logicalMinZoom = logicalZoomState?.minZoomRatio ?: 1f
        val logicalMaxZoom = logicalZoomState?.maxZoomRatio ?: 1f

        return deduped.mapNotNull { (raw, equivMm) ->
            val lensInfo = when {
                abs(equivMm - mainEquiv) < 1f -> LensInfo(zoomFactor = 1.0, label = "Wide camera", cameraInfo = raw.info)
                equivMm < mainEquiv - 4f -> {
                    // Ultra-wide. iOS exposes these as 0.5x relative to the main lens.
                    val zoom = (equivMm.toDouble() / mainEquiv.toDouble()).coerceAtLeast(0.1)
                    val rounded = if (abs(zoom - 0.5) < 0.15) 0.5 else zoom
                    LensInfo(zoomFactor = rounded, label = "Ultra wide camera", cameraInfo = raw.info)
                }
                else -> {
                    val zoom = equivMm.toDouble() / mainEquiv.toDouble()
                    LensInfo(zoomFactor = zoom, label = "Telephoto camera", cameraInfo = raw.info)
                }
            }
            if (raw.info == null) {
                val zoom = lensInfo.zoomFactor.toFloat()
                if (logicalWideInfo == null || zoom < logicalMinZoom - 0.01f || zoom > logicalMaxZoom + 0.01f) {
                    return@mapNotNull null
                }
            }
            lensInfo
        }
    }

    /**
     * Switch the active back-camera lens to the one whose computed zoom factor is closest
     * to [zoomFactor]. Emits a `{type:"lens",label}` event on the existing event sink.
     */
    fun setLens(zoomFactor: Double) {
        if (lensFacing != CameraSelector.LENS_FACING_BACK) return
        val lenses = if (cachedLenses.isEmpty()) enumerateLenses() else cachedLenses
        if (lenses.isEmpty()) return
        val target = lenses.minByOrNull { abs(it.zoomFactor - zoomFactor) } ?: return
        if (target.cameraInfo == null) {
            selectLogicalBackLens(target)
            return
        }
        // After the rebind the new lens starts at physical 1.0x; that maps to effective `target.zoomFactor`
        // (e.g. 0.5x on ultra-wide, 2.0x on tele) which is exactly what the user asked for.
        pendingEffectiveZoomToEmit = target.zoomFactor
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

    private fun selectLogicalBackLens(target: LensInfo) {
        lensFacing = CameraSelector.LENS_FACING_BACK
        preferWideBackCamera = false
        val logicalWide = cachedLenses
            .filter { it.cameraInfo != null }
            .minByOrNull { abs(it.zoomFactor - 1.0) }
            ?: return
        val logicalWideCameraInfo = logicalWide.cameraInfo ?: return

        val targetPhysicalZoom = target.zoomFactor.toFloat()
        val logicalZoomState = logicalWideCameraInfo.zoomState.value
        val logicalMinZoom = logicalZoomState?.minZoomRatio ?: 1f
        val logicalMaxZoom = logicalZoomState?.maxZoomRatio ?: maxZoomRatio
        if (targetPhysicalZoom < logicalMinZoom - 0.01f || targetPhysicalZoom > logicalMaxZoom + 0.01f) {
            Log.w(TAG, "Hidden lens ${target.label} is not reachable through logical camera zoom")
            return
        }

        selectedLensCameraInfo = logicalWideCameraInfo
        selectedLensZoomFactor = 1.0
        selectedLensLabel = target.label

        if (camera?.cameraInfo != logicalWideCameraInfo) {
            pendingZoomRatioToApply = targetPhysicalZoom
            pendingEffectiveZoomToEmit = target.zoomFactor
            switchToLens(logicalWide)
            selectedLensLabel = target.label
            emitEvent(mapOf("type" to "lens", "label" to target.label))
            return
        }

        val physical = targetPhysicalZoom.coerceIn(
            minZoomRatio,
            camera?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: maxZoomRatio
        )
        camera?.cameraControl?.setZoomRatio(physical)
        currentZoomRatio = physical
        onZoomChanged?.invoke(physical)
        emitEvent(mapOf("type" to "zoom", "value" to physical.toDouble()))
        emitEvent(mapOf("type" to "lens", "label" to target.label))
    }

    /**
     * Tap-to-focus. [x] and [y] are normalized view-relative coordinates in 0..1. Builds a FocusMeteringAction via the
     * PreviewView's MeteringPointFactory and triggers AF/AE. Emits `{type:"focus",x,y}` when the future completes
     * successfully so the Dart `FocusReticle` can animate.
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
     * Capture a still photo. Preferred path uses the bound ImageCapture use-case so we get a full-resolution JPEG; if
     * [withOverlays] is true the current overlay bitmap is composited on top of the still before re-encoding. If
     * ImageCapture binding isn't available (e.g. three-use-case bind failed), falls back to [captureFrame] which
     * snapshots the preview + overlay composite.
     */
    fun capturePhoto(withOverlays: Boolean = true, callback: (ByteArray?) -> Unit) {
        val ic = imageCaptureUseCase
        if (ic == null) {
            // Three-use-case bind failed at startup; honor withOverlays via captureFrame's matching flag so callers
            // asking for a raw photo don't silently get an annotated one.
            callback(captureFrame(withOverlays))
            return
        }
        try {
            ic.takePicture(
                ContextCompat.getMainExecutor(context),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            // Carry the capture rotation + mirroring forward; ImageCapture hands us a JPEG that is not
                            // yet rotated for portrait sensors, and on the front camera we also need to flip
                            // horizontally before re-encoding. Without this every portrait share ends up sideways.
                            val rotationDegrees = image.imageInfo.rotationDegrees
                            val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
                            val jpegBytes = imageProxyToJpegBytes(image)
                            if (jpegBytes == null) {
                                callback(captureFrame(withOverlays))
                                return
                            }
                            if (!withOverlays) {
                                callback(normalizeJpegOrientation(jpegBytes, rotationDegrees, isFront) ?: jpegBytes)
                                return
                            }
                            // Composite the current overlay bitmap on top of the still.
                            val composed = compositeOverlayOnJpeg(jpegBytes, rotationDegrees, isFront)
                            callback(composed ?: jpegBytes)
                        } catch (e: Exception) {
                            Log.e(TAG, "capturePhoto: error processing capture", e)
                            callback(captureFrame(withOverlays))
                        } finally {
                            image.close()
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.w(TAG, "capturePhoto: ImageCapture failed, falling back to captureFrame", exception)
                        callback(captureFrame(withOverlays))
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "capturePhoto: takePicture threw, falling back", e)
            callback(captureFrame(withOverlays))
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
                try {
                    val out = java.io.ByteArrayOutputStream()
                    bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
                    out.toByteArray()
                } finally {
                    bmp.recycle()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "imageProxyToJpegBytes failed", e)
            null
        }
    }

    private fun compositeOverlayOnJpeg(jpegBytes: ByteArray, rotationDegrees: Int, isFront: Boolean): ByteArray? {
        return try {
            val decoded = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
            // Apply the capture orientation (and mirror for the front camera) BEFORE compositing — the overlay is
            // drawn in display coordinates, so the still bitmap has to be in the same upright orientation or boxes land
            // at the wrong positions and the shared JPEG ends up sideways.
            val still = applyOrientation(decoded, rotationDegrees, isFront)
            if (still !== decoded) decoded.recycle()
            // Render overlay onto a bitmap sized to match the upright still.
            val composite = Bitmap.createBitmap(still.width, still.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(composite)
            canvas.drawBitmap(still, 0f, 0f, null)
            // Capture the overlay at its current view size and scale it to the still.
            val overlayBitmap = Bitmap.createBitmap(
                overlayView.width.coerceAtLeast(1),
                overlayView.height.coerceAtLeast(1),
                Bitmap.Config.ARGB_8888,
            )
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

    /** Decode + rotate/mirror a JPEG to the display-correct orientation, then re-encode. */
    private fun normalizeJpegOrientation(jpegBytes: ByteArray, rotationDegrees: Int, isFront: Boolean): ByteArray? {
        if (rotationDegrees == 0 && !isFront) return jpegBytes
        return try {
            val decoded = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
            val oriented = applyOrientation(decoded, rotationDegrees, isFront)
            if (oriented === decoded) {
                jpegBytes
            } else {
                val out = java.io.ByteArrayOutputStream()
                oriented.compress(Bitmap.CompressFormat.JPEG, 90, out)
                decoded.recycle()
                oriented.recycle()
                out.toByteArray()
            }
        } catch (e: Exception) {
            Log.e(TAG, "normalizeJpegOrientation failed", e)
            null
        }
    }

    /** Rotate `bitmap` clockwise by `rotationDegrees`, mirroring horizontally when `isFront` is true. */
    private fun applyOrientation(bitmap: Bitmap, rotationDegrees: Int, isFront: Boolean): Bitmap {
        if (rotationDegrees == 0 && !isFront) return bitmap
        val matrix = Matrix().apply {
            if (rotationDegrees != 0) postRotate(rotationDegrees.toFloat())
            if (isFront) postScale(-1f, 1f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    // endregion
    
    // Lifecycle methods from DefaultLifecycleObserver
    override fun onStart(owner: LifecycleOwner) {
        if (allPermissionsGranted()) {
            // Restart the camera on start if it was stopped by a lifecycle event (e.g. navigating back), but NOT if the
            // Dart layer intentionally paused it — that pause must hold until an explicit resume.
            if (!intentionallyPaused && (isStopped || camera == null)) {
                startCamera()
            }
        }
    }

    override fun onResume(owner: LifecycleOwner) {
        if (allPermissionsGranted()) {
            // Double-check camera is running on resume, unless the Dart layer intentionally paused it.
            if (!intentionallyPaused && (isStopped || camera == null)) {
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
                syncTargetRotation()
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
                    basePredictor.includeRawMaskData = streamConfig?.includeMasks == true
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
                        // Dimensions of the upright frame that (normalized) detection coordinates refer to (#506)
                        enhancedStreamData["imageWidth"] = resultWithOriginalImage.origShape.width
                        enhancedStreamData["imageHeight"] = resultWithOriginalImage.origShape.height
                        
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
            
            // Scale factor from camera image to view
            val scaleX = vw / iw
            val scaleY = vh / ih
            val scale = max(scaleX, scaleY)
            

            // Check if using front camera
            val isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT

            val imageRect = RectF(
                (vw - iw * scale) / 2f,
                (vh - ih * scale) / 2f,
                (vw + iw * scale) / 2f,
                (vh + ih * scale) / 2f
            )

            fun mapPoint(x: Float, y: Float): PointF {
                val px = imageRect.left + x * scale
                val py = imageRect.top + y * scale
                return PointF(if (isFrontCamera) vw - px else px, py)
            }

            fun mapRect(rect: RectF): RectF {
                val topLeft = mapPoint(rect.left, rect.top)
                val bottomRight = mapPoint(rect.right, rect.bottom)
                return RectF(
                    minOf(topLeft.x, bottomRight.x),
                    minOf(topLeft.y, bottomRight.y),
                    maxOf(topLeft.x, bottomRight.x),
                    maxOf(topLeft.y, bottomRight.y)
                )
            }

            when (task) {
                // ----------------------------------------
                // DETECT
                // ----------------------------------------
                YOLOTask.DETECT -> {
                    for (box in result.boxes) {
                        val newColor = colorFor(box.index, box.conf)

                        val rect = mapRect(box.xywh)
                        // Draw at the box's true position and let the canvas clip whatever falls outside the view. Do
                        // NOT pin an edge to the bound while keeping the width — that shifts a partially off-screen box
                        // inward (a left edge clamped to 0 pushes the right edge too far right, and vice versa).

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            rect.left, rect.top, rect.right, rect.bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        drawLabel(canvas, labelText(box.cls, box.conf), newColor, rect.left, rect.top, rect.right, vw, vh)
                    }
                }
                // ----------------------------------------
                // SEGMENT
                // ----------------------------------------
                YOLOTask.SEGMENT -> {
                    // Bounding boxes & labels
                    for (box in result.boxes) {
                        val newColor = colorFor(box.index, box.conf)

                        val rect = mapRect(box.xywh)

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            rect.left, rect.top, rect.right, rect.bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )

                        drawLabel(canvas, labelText(box.cls, box.conf), newColor, rect.left, rect.top, rect.right, vw, vh)
                    }

                    // Segmentation mask
                    result.masks?.combinedMask?.let { maskBitmap ->
                        val src = Rect(0, 0, maskBitmap.width, maskBitmap.height)
                        val maskPaint = Paint().apply {
                            alpha = 128
                            isFilterBitmap = false
                        }
                        
                        if (isFrontCamera) {
                            // For front camera, flip the mask horizontally
                            canvas.save()
                            // Translate to center, flip horizontally, translate back
                            canvas.translate(vw / 2f, 0f)
                            canvas.scale(-1f, 1f)
                            canvas.translate(-vw / 2f, 0f)
                            canvas.drawBitmap(maskBitmap, src, imageRect, maskPaint)
                            canvas.restore()
                        } else {
                            canvas.drawBitmap(maskBitmap, src, imageRect, maskPaint)
                        }
                    }
                }
                // ----------------------------------------
                // SEMANTIC
                // ----------------------------------------
                YOLOTask.SEMANTIC -> {
                    result.semanticMask?.maskImage?.let { maskBitmap ->
                        val src = Rect(0, 0, maskBitmap.width, maskBitmap.height)
                        val maskPaint = Paint().apply {
                            alpha = 128
                            isFilterBitmap = false
                        }

                        if (isFrontCamera) {
                            canvas.save()
                            canvas.translate(vw / 2f, 0f)
                            canvas.scale(-1f, 1f)
                            canvas.translate(-vw / 2f, 0f)
                            canvas.drawBitmap(maskBitmap, src, imageRect, maskPaint)
                            canvas.restore()
                        } else {
                            canvas.drawBitmap(maskBitmap, src, imageRect, maskPaint)
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

                        val rect = mapRect(box.xywh)

                        paint.color = newColor
                        paint.style = Paint.Style.STROKE
                        paint.strokeWidth = BOX_LINE_WIDTH
                        canvas.drawRoundRect(
                            rect.left, rect.top, rect.right, rect.bottom,
                            BOX_CORNER_RADIUS, BOX_CORNER_RADIUS,
                            paint
                        )
                        
                        drawLabel(canvas, labelText(box.cls, box.conf), newColor, rect.left, rect.top, rect.right, vw, vh)
                    }

                    // Keypoints & skeleton
                    for (person in result.keypointsList) {
                        val points = arrayOfNulls<PointF>(person.xyn.size)
                        for (i in person.xyn.indices) {
                            val kp = person.xyn[i]
                            val conf = person.conf[i]
                            if (conf > 0.25f) {
                                val point = mapPoint(kp.first * iw, kp.second * ih)

                                val colorIdx = if (i < kptColorIndices.size) kptColorIndices[i] else 0
                                val rgbArray = posePalette[colorIdx % posePalette.size]
                                paint.color = Color.argb(
                                    255,
                                    rgbArray[0].toInt().coerceIn(0,255),
                                    rgbArray[1].toInt().coerceIn(0,255),
                                    rgbArray[2].toInt().coerceIn(0,255)
                                )
                                paint.style = Paint.Style.FILL
                                canvas.drawCircle(point.x, point.y, 8f, paint)

                                points[i] = point
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
                        val polygon = obbRes.box.toPolygon(iw, ih).map { pt -> mapPoint(pt.x * iw, pt.y * ih) }
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
                    detection["className"] = result.names.getOrNull(0)?.takeIf { it.isNotBlank() } ?: "class 0"
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

                // Normalized (0-1) corners and rotation angle, always present so custom overlays
                // can transform OBB detections without enabling includeOBB (#506)
                val pointsNormalized = polygon.map { point ->
                    mapOf(
                        "x" to point.x.toDouble(),
                        "y" to point.y.toDouble()
                    )
                }
                detection["polygonNormalized"] = pointsNormalized
                detection["angle"] = obbRes.box.angle.toDouble()

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
                    val obbDataMap = mapOf(
                        "centerX" to obbRes.box.cx.toDouble(),
                        "centerY" to obbRes.box.cy.toDouble(),
                        "width" to obbRes.box.w.toDouble(),
                        "height" to obbRes.box.h.toDouble(),
                        "angle" to obbRes.box.angle.toDouble(),
                        "angleDegrees" to (obbRes.box.angle * 180.0 / Math.PI),
                        "area" to obbRes.box.area.toDouble(),
                        "points" to pointsNormalized,
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
                    "classMap" to semanticMask.classMap.toList(),
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
            map["preMs"] = result.preMs
            map["inferenceMs"] = result.inferenceMs
            map["postMs"] = result.postMs
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
     * Capture current camera frame. When [withOverlays] is true the overlay bitmap (bounding boxes / mask / pose) is
     * composited on top of the preview snapshot before encoding. Used as the fallback path when [capturePhoto]'s
     * preferred ImageCapture binding is unavailable. Returns the captured image as a ByteArray (JPEG format).
     */
    fun captureFrame(withOverlays: Boolean = true): ByteArray? {
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
            
            // Conditionally draw the overlay on top — callers asking for a raw photo (e.g.
            // capturePhoto(withOverlays=false) hitting the fallback path) get the unannotated preview snapshot.
            if (withOverlays) {
                overlayView.draw(canvas)
            }
            
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
        // A full teardown is not an intentional pause; a later lifecycle restart should rebind normally.
        intentionallyPaused = false

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
            
            // Close the active predictor AND release every other cached predictor (prior setModel() instances), so a
            // disposed view doesn't leak their native LiteRT interpreters / tensor buffers. Closing also makes a later
            // same-key setModel() fast path unable to serve a now-closed instance (use-after-close).
            val closing = predictor
            try {
                (closing as? BasePredictor)?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing predictor", e)
            }
            for (cached in predictorCache.values) {
                if (cached !== closing) {
                    try {
                        (cached as? BasePredictor)?.close()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error closing cached predictor", e)
                    }
                }
            }
            predictorCache.clear()
            predictorCacheOrder.clear()
            predictor = null
            inferenceCallback = null
            streamCallback = null
            inferenceResult = null
        } catch (e: Exception) {
            Log.e(TAG, "Error during YOLOView stop", e)
        }
    }

}
