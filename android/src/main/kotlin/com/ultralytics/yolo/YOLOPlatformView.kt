// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.util.Log
import android.view.View
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import java.util.concurrent.atomic.AtomicBoolean

/**
 * YOLOPlatformView - Native view bridge from Flutter
 */
class YOLOPlatformView(
    private val context: Context,
    private val viewId: Int,
    creationParams: Map<String?, Any?>?,
    private val streamHandler: CustomStreamHandler,
    private val methodChannel: MethodChannel?,
    private val factory: YOLOPlatformViewFactory
) : PlatformView, MethodChannel.MethodCallHandler {

    private val yoloView: YOLOView = YOLOView(context)
    
    // Getter for external access to yoloView
    val yoloViewInstance: YOLOView
        get() = yoloView
    private val TAG = "YOLOPlatformView"
    
    // Track if we're actively streaming
    private val isStreaming = AtomicBoolean(false)
    
    // Store last event to resend after reconnection
    @Volatile
    private var lastStreamData: Map<String, Any>? = null
    
    // Initialization flag
    private var initialized = false
    
    // Unique ID to send to Flutter
    private val viewUniqueId: String
    
    // Retry handler for reconnection
    private val retryHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var retryRunnable: Runnable? = null
    
    init {
        val dartViewIdParam = creationParams?.get("viewId")
        viewUniqueId = dartViewIdParam as? String ?: viewId.toString().also {
            Log.w(TAG, "YOLOPlatformView[$viewId init]: Using platform int viewId '$it' as fallback")
        }
        Log.d(TAG, "YOLOPlatformView[$viewId init]: Initialized with viewUniqueId: $viewUniqueId")

        // Parse model path and task from creation params
        var modelPath = creationParams?.get("modelPath") as? String ?: "yolo11n"
        val taskString = creationParams?.get("task") as? String ?: "detect"
        val confidenceParam = creationParams?.get("confidenceThreshold") as? Double ?: 0.5
        val iouParam = creationParams?.get("iouThreshold") as? Double ?: 0.45

        // Set up the method channel handler
        methodChannel?.setMethodCallHandler(this)

        // Set initial thresholds
        Log.d(TAG, "Setting initial thresholds: conf=$confidenceParam, iou=$iouParam")
        yoloView.setConfidenceThreshold(confidenceParam)
        yoloView.setIouThreshold(iouParam)
        
        // Configure YOLOView streaming functionality
        setupYOLOViewStreaming(creationParams)

        // Initialize camera
        Log.d(TAG, "Attempting early camera initialization")
        yoloView.initCamera()

        // Notify lifecycle if available
        if (context is LifecycleOwner) {
            Log.d(TAG, "Initial context is a LifecycleOwner, notifying YOLOView")
            yoloView.onLifecycleOwnerAvailable(context)
        }
        
        try {
            // Resolve model path
            modelPath = resolveModelPath(context, modelPath)
            val task = YOLOTask.valueOf(taskString.uppercase())
            
            Log.d(TAG, "Initializing with model: $modelPath, task: $task")
            
            // Set up model loading callback
            yoloView.setOnModelLoadCallback { success ->
                if (success) {
                    Log.d(TAG, "Model loaded successfully")
                    initialized = true
                    // Start streaming if not already started
                    startStreaming()
                } else {
                    Log.w(TAG, "Failed to load model")
                    initialized = true
                }
            }
            
            // Set up inference callback
            yoloView.setOnInferenceCallback { result ->
                // Callback for compatibility
            }
            
            // Load model
            val useGpu = creationParams?.get("useGpu") as? Boolean ?: true
            yoloView.setModel(modelPath, task, useGpu)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YOLOPlatformView", e)
        }
    }
    
    /**
     * Configure YOLOView streaming functionality with setState resilience
     */
    private fun setupYOLOViewStreaming(creationParams: Map<String?, Any?>?) {
        val streamingConfigParam = creationParams?.get("streamingConfig") as? Map<*, *>
        
        val streamConfig = streamingConfigParam?.let { config ->
            YOLOStreamConfig(
                includeDetections = config["includeDetections"] as? Boolean ?: true,
                includeClassifications = config["includeClassifications"] as? Boolean ?: true,
                includeProcessingTimeMs = config["includeProcessingTimeMs"] as? Boolean ?: true,
                includeFps = config["includeFps"] as? Boolean ?: true,
                includeMasks = config["includeMasks"] as? Boolean ?: false,
                includePoses = config["includePoses"] as? Boolean ?: false,
                includeOBB = config["includeOBB"] as? Boolean ?: false,
                includeOriginalImage = config["includeOriginalImage"] as? Boolean ?: false,
                maxFPS = (config["maxFPS"] as? Number)?.toInt(),
                throttleIntervalMs = (config["throttleInterval"] as? Number)?.toInt(),
                inferenceFrequency = (config["inferenceFrequency"] as? Number)?.toInt(),
                skipFrames = (config["skipFrames"] as? Number)?.toInt()
            )
        }
        
        yoloView.setStreamConfig(streamConfig)
        
        // Set up streaming callback with resilience
        yoloView.setStreamCallback { streamData ->
            sendStreamDataWithRetry(streamData)
        }
        
        Log.d(TAG, "YOLOView streaming configured with setState resilience")
    }
    
    /**
     * Send stream data with automatic retry on failure
     */
    private fun sendStreamDataWithRetry(streamData: Map<String, Any>) {
        try {
            // Store last data for potential resend
            lastStreamData = streamData
            
            // Cancel any pending retry
            retryRunnable?.let { retryHandler.removeCallbacks(it) }
            
            // Try to send data
            val sent = sendStreamData(streamData)
            
            if (!sent && isStreaming.get()) {
                // Schedule retry if sending failed
                scheduleRetry()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in sendStreamDataWithRetry", e)
            if (isStreaming.get()) {
                scheduleRetry()
            }
        }
    }
    
    /**
     * Attempt to send stream data
     */
    private fun sendStreamData(streamData: Map<String, Any>): Boolean {
        return try {
            val sink = streamHandler.sink
            
            if (sink != null) {
                // Send on main thread
                if (android.os.Looper.myLooper() == android.os.Looper.getMainLooper()) {
                    sink.success(streamData)
                } else {
                    var success = false
                    val latch = java.util.concurrent.CountDownLatch(1)
                    
                    retryHandler.post {
                        try {
                            sink.success(streamData)
                            success = true
                        } catch (e: Exception) {
                            Log.e(TAG, "Error sending on main thread", e)
                        } finally {
                            latch.countDown()
                        }
                    }
                    
                    latch.await(100, java.util.concurrent.TimeUnit.MILLISECONDS)
                    success
                }
                true
            } else {
                Log.w(TAG, "Event sink is null, will retry")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending stream data", e)
            false
        }
    }
    
    /**
     * Schedule a retry to resend data
     */
    private fun scheduleRetry() {
        retryRunnable?.let { retryHandler.removeCallbacks(it) }
        
        retryRunnable = Runnable {
            if (isStreaming.get()) {
                Log.d(TAG, "Retrying to send stream data")
                
                // Check if sink is available
                if (streamHandler.sink != null) {
                    // Resend last data if available
                    lastStreamData?.let { data ->
                        sendStreamData(data)
                    }
                } else {
                    // Request Flutter to recreate the event channel
                    Log.d(TAG, "Requesting Flutter to reconnect event channel")
                    methodChannel?.invokeMethod("reconnectEventChannel", mapOf(
                        "viewId" to viewUniqueId,
                        "reason" to "sink_disconnected"
                    ))
                    
                    // Schedule another retry
                    scheduleRetry()
                }
            }
        }
        
        // Retry after 500ms
        retryHandler.postDelayed(retryRunnable!!, 500)
    }
    
    /**
     * Start streaming
     */
    private fun startStreaming() {
        if (isStreaming.compareAndSet(false, true)) {
            Log.d(TAG, "Started streaming for view $viewId")
            
            // Send initial test message to verify connection
            sendStreamData(mapOf(
                "test" to "Streaming started",
                "viewId" to viewUniqueId,
                "timestamp" to System.currentTimeMillis()
            ))
        }
    }
    
    /**
     * Stop streaming
     */
    private fun stopStreaming() {
        if (isStreaming.compareAndSet(true, false)) {
            Log.d(TAG, "Stopped streaming for view $viewId")
            retryRunnable?.let { retryHandler.removeCallbacks(it) }
            retryRunnable = null
        }
    }
    
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            when (call.method) {
                "setConfidenceThreshold" -> {
                    val threshold = call.argument<Double>("threshold")
                    if (threshold != null) {
                        yoloView.setConfidenceThreshold(threshold)
                        result.success(null)
                    } else {
                        result.error("invalid_args", "threshold is required", null)
                    }
                }
                "setIoUThreshold", "setIouThreshold" -> {
                    val threshold = call.argument<Double>("threshold")
                    if (threshold != null) {
                        yoloView.setIouThreshold(threshold)
                        result.success(null)
                    } else {
                        result.error("invalid_args", "threshold is required", null)
                    }
                }
                "setNumItemsThreshold" -> {
                    val numItems = call.argument<Int>("numItems")
                    if (numItems != null) {
                        yoloView.setNumItemsThreshold(numItems)
                        result.success(null)
                    } else {
                        result.error("invalid_args", "numItems is required", null)
                    }
                }
                "setThresholds" -> {
                    val confidence = call.argument<Double>("confidenceThreshold")
                    val iou = call.argument<Double>("iouThreshold")
                    val numItems = call.argument<Int>("numItemsThreshold")
                    
                    confidence?.let { yoloView.setConfidenceThreshold(it) }
                    iou?.let { yoloView.setIouThreshold(it) }
                    numItems?.let { yoloView.setNumItemsThreshold(it) }
                    
                    result.success(null)
                }
                "setModel" -> {
                    var modelPath = call.argument<String>("modelPath")
                    val taskString = call.argument<String>("task")
                    val useGpu = call.argument<Boolean>("useGpu") ?: true
                    
                    if (modelPath == null || taskString == null) {
                        result.error("invalid_args", "modelPath and task are required", null)
                        return
                    }
                    
                    modelPath = resolveModelPath(context, modelPath)
                    val task = YOLOTask.valueOf(taskString.uppercase())
                    
                    yoloView.setModel(modelPath, task, useGpu) { success ->
                        if (success) {
                            Log.d(TAG, "Model switched successfully")
                            result.success(null)
                        } else {
                            Log.e(TAG, "Failed to switch model")
                            result.error("MODEL_NOT_FOUND", "Failed to load model", null)
                        }
                    }
                }
                "switchCamera" -> {
                    yoloView.switchCamera()
                    result.success(null)
                }
                "setZoomLevel" -> {
                    val zoomLevel = call.argument<Double>("zoomLevel")
                    if (zoomLevel != null) {
                        yoloView.setZoomLevel(zoomLevel.toFloat())
                        result.success(null)
                    } else {
                        result.error("invalid_args", "zoomLevel is required", null)
                    }
                }
                "setStreamingConfig" -> {
                    // Parse streaming config from arguments
                    val configMap = call.arguments as? Map<*, *>
                    if (configMap != null) {
                        val streamConfig = YOLOStreamConfig(
                            includeDetections = configMap["includeDetections"] as? Boolean ?: true,
                            includeClassifications = configMap["includeClassifications"] as? Boolean ?: true,
                            includeProcessingTimeMs = configMap["includeProcessingTimeMs"] as? Boolean ?: true,
                            includeFps = configMap["includeFps"] as? Boolean ?: true,
                            includeMasks = configMap["includeMasks"] as? Boolean ?: false,
                            includePoses = configMap["includePoses"] as? Boolean ?: false,
                            includeOBB = configMap["includeOBB"] as? Boolean ?: false,
                            includeOriginalImage = configMap["includeOriginalImage"] as? Boolean ?: false,
                            maxFPS = (configMap["maxFPS"] as? Number)?.toInt(),
                            throttleIntervalMs = (configMap["throttleInterval"] as? Number)?.toInt(),
                            inferenceFrequency = (configMap["inferenceFrequency"] as? Number)?.toInt(),
                            skipFrames = (configMap["skipFrames"] as? Number)?.toInt()
                        )
                        yoloView.setStreamConfig(streamConfig)
                        Log.d(TAG, "Streaming config updated")
                        result.success(null)
                    } else {
                        result.error("invalid_args", "Invalid streaming config", null)
                    }
                }
                "stop" -> {
                    yoloView.stop()
                    Log.d(TAG, "Camera and inference stopped")
                    result.success(null)
                }
                "captureFrame" -> {
                    val imageData = yoloView.captureFrame()
                    if (imageData != null) {
                        Log.d(TAG, "Frame captured: ${imageData.size} bytes")
                        result.success(imageData)
                    } else {
                        result.error("capture_failed", "Failed to capture frame", null)
                    }
                }
                "reconnectStream" -> {
                    // Handle reconnection request from Flutter
                    Log.d(TAG, "Received reconnect request from Flutter")
                    startStreaming()
                    result.success(null)
                }
                "setShowUIControls" -> {
                    val show = call.argument<Boolean>("show") ?: false
                    yoloView.setShowUIControls(show)
                    result.success(null)
                }
                else -> {
                    result.notImplemented()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling method call: ${call.method}", e)
            result.error("method_call_error", e.message, null)
        }
    }
    
    override fun getView(): View {
        return yoloView
    }
    
    override fun dispose() {
        Log.d(TAG, "Disposing YOLOPlatformView for viewId: $viewId")
        
        stopStreaming()
        
        try {
            yoloView.stop()
            // Clear callbacks by setting them to empty implementations
            yoloView.setStreamCallback { }
            yoloView.setOnInferenceCallback { }
            yoloView.setOnModelLoadCallback { }
        } catch (e: Exception) {
            Log.e(TAG, "Error during disposal", e)
        }
        
        methodChannel?.setMethodCallHandler(null)
        factory.onPlatformViewDisposed(viewId)
        
        Log.d(TAG, "YOLOPlatformView disposed successfully")
    }
    
    private fun resolveModelPath(context: Context, modelPath: String): String {
        return when {
            modelPath.startsWith("/") -> modelPath
            modelPath.startsWith("internal://") -> {
                val filename = modelPath.removePrefix("internal://")
                context.filesDir.resolve(filename).absolutePath
            }
            else -> modelPath
        }
    }
}