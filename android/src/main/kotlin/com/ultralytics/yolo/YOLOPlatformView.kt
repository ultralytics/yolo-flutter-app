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

/**
 * YOLOPlatformView - Native view bridge from Flutter
 */
class YOLOPlatformView(
    private val context: Context,
    private val viewId: Int,
    creationParams: Map<String?, Any?>?,
    private val streamHandler: EventChannel.StreamHandler,
    private val methodChannel: MethodChannel?,
    private val factory: YOLOPlatformViewFactory // Added factory reference
) : PlatformView, MethodChannel.MethodCallHandler {

    private val yoloView: YOLOView = YOLOView(context)
    private val TAG = "YOLOPlatformView"
    
    // Initialization flag
    private var initialized = false
    
    // Unique ID to send to Flutter
    private val viewUniqueId: String
    
    init {
        val dartViewIdParam = creationParams?.get("viewId")
        viewUniqueId = dartViewIdParam as? String ?: viewId.toString().also {
            Log.w(TAG, "YOLOPlatformView[$viewId init]: Using platform int viewId '$it' as fallback for viewUniqueId because Dart 'viewId' was null or not a String.")
        }
        Log.d(TAG, "YOLOPlatformView[$viewId init]: Initialized with creationParams: $creationParams. Resolved viewUniqueId for channels: $viewUniqueId")

        // Parse model path and task from creation params
        var modelPath = creationParams?.get("modelPath") as? String ?: "yolo11n"
        val taskString = creationParams?.get("task") as? String ?: "detect"
        // These will use defaults if not in creationParams, which is expected
        // as Dart side sets them via method channel after view creation.
        val confidenceParam = creationParams?.get("confidenceThreshold") as? Double ?: 0.5
        val iouParam = creationParams?.get("iouThreshold") as? Double ?: 0.45

        // Set up the method channel handler
        methodChannel?.setMethodCallHandler(this)

        // Set initial thresholds on YOLOView instance from creationParams or defaults.
        // YOLOView.setModel will use these when creating the predictor.
        Log.d(TAG, "Setting initial thresholds on YOLOView: conf=$confidenceParam, iou=$iouParam")
        yoloView.setConfidenceThreshold(confidenceParam)
        yoloView.setIouThreshold(iouParam)
        // numItemsThreshold defaults within YOLOView.kt
        
        // Configure YOLOView streaming functionality
        setupYOLOViewStreaming(creationParams)

        // Attempt to initialize camera as soon as the view is created.
        // YOLOView.initCamera() handles permissions and starts the camera preview.
        Log.d(TAG, "Attempting early camera initialization for YOLOView.")
        yoloView.initCamera() // This will attempt to start camera or request permissions

        // If context is already a LifecycleOwner, inform YOLOView immediately
        if (context is LifecycleOwner) {
            Log.d(TAG, "Initial context is a LifecycleOwner (${context.javaClass.simpleName}), notifying YOLOView.")
            yoloView.onLifecycleOwnerAvailable(context)
        } else {
            Log.w(TAG, "Initial context (${context.javaClass.simpleName}) is NOT a LifecycleOwner. YOLOView will wait for one to be provided via notifyLifecycleOwnerAvailable.")
        }
        
        try {
            // Resolve model path (handling absolute paths, internal:// scheme, or asset paths)
            modelPath = resolveModelPath(context, modelPath)
            
            // Convert task string to enum
            val task = YOLOTask.valueOf(taskString.uppercase())
            
            Log.d(TAG, "Initializing YOLOPlatformView with model: $modelPath, task: $task, viewId: $viewId")
            
            // Set up callback for model loading result
            yoloView.setOnModelLoadCallback { success ->
                if (success) {
                    Log.d(TAG, "Model loaded successfully: $modelPath.")
                    // Camera initialization was already attempted.
                    // Mark that the full initialization sequence (including model load) is complete.
                    initialized = true
                } else {
                    Log.w(TAG, "Failed to load model: $modelPath. Camera will run without inference.")
                    // Still mark as initialized since camera can work without model
                    initialized = true
                }
            }
            
            // YOLOView streaming is now configured separately
            // Keep simple inference callback for compatibility
            yoloView.setOnInferenceCallback { result ->
            }
            
            // Load model with the specified path and task
            yoloView.setModel(modelPath, task)
            
            // Setup zoom callback
            yoloView.onZoomChanged = { zoomLevel ->
                methodChannel?.invokeMethod("onZoomChanged", zoomLevel.toDouble())
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YOLOPlatformView", e)
        }
    }
    
    // Handle method calls from Flutter
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            Log.d(TAG, "Received method call: ${call.method} with arguments: ${call.arguments}")
            
            when (call.method) {
                "setThreshold" -> {
                    val threshold = call.argument<Double>("threshold") ?: 0.5
                    Log.d(TAG, "Setting confidence threshold to $threshold")
                    yoloView.setConfidenceThreshold(threshold)
                    result.success(null)
                }
                "setConfidenceThreshold" -> {
                    val threshold = call.argument<Double>("threshold") ?: 0.5
                    Log.d(TAG, "Setting confidence threshold to $threshold")
                    yoloView.setConfidenceThreshold(threshold)
                    result.success(null)
                }
                // Support both "setIoUThreshold" (from Dart) and "setIouThreshold" (internal method)
                "setIoUThreshold", "setIouThreshold" -> {
                    val threshold = call.argument<Double>("threshold") ?: 0.45
                    Log.d(TAG, "Setting IoU threshold to $threshold")
                    yoloView.setIouThreshold(threshold)
                    result.success(null)
                }
                "setNumItemsThreshold" -> {
                    val numItems = call.argument<Int>("numItems") ?: 30
                    Log.d(TAG, "Setting numItems threshold to $numItems")
                    yoloView.setNumItemsThreshold(numItems)
                    result.success(null)
                }
                "setThresholds" -> {
                    val confidenceThreshold = call.argument<Double>("confidenceThreshold")
                    val iouThreshold = call.argument<Double>("iouThreshold")
                    val numItemsThreshold = call.argument<Int>("numItemsThreshold")
                    
                    if (confidenceThreshold != null) {
                        Log.d(TAG, "Setting confidence threshold to $confidenceThreshold")
                        yoloView.setConfidenceThreshold(confidenceThreshold)
                    }
                    if (iouThreshold != null) {
                        Log.d(TAG, "Setting IoU threshold to $iouThreshold")
                        yoloView.setIouThreshold(iouThreshold)
                    }
                    if (numItemsThreshold != null) {
                        Log.d(TAG, "Setting numItems threshold to $numItemsThreshold")
                        yoloView.setNumItemsThreshold(numItemsThreshold)
                    }
                    
                    result.success(null)
                }
                "switchCamera" -> {
                    Log.d(TAG, "Switching camera")
                    yoloView.switchCamera()
                    result.success(null)
                }
                "setShowUIControls" -> {
                    // Android doesn't have UI controls like iOS, so we just acknowledge the call
                    Log.d(TAG, "setShowUIControls called, but not applicable for Android")
                    result.success(null)
                }
                "setZoomLevel" -> {
                    val zoomLevel = call.argument<Double>("zoomLevel")
                    if (zoomLevel != null) {
                        Log.d(TAG, "Setting zoom level to $zoomLevel")
                        yoloView.setZoomLevel(zoomLevel.toFloat())
                        result.success(null)
                    } else {
                        result.error("invalid_args", "Zoom level is required", null)
                    }
                }
                "setStreamingConfig" -> {
                    Log.d(TAG, "Received setStreamingConfig call")
                    val streamConfig = YOLOStreamConfig(
                        includeDetections = call.argument<Boolean>("includeDetections") ?: true,
                        includeClassifications = call.argument<Boolean>("includeClassifications") ?: true,
                        includeProcessingTimeMs = call.argument<Boolean>("includeProcessingTimeMs") ?: true,
                        includeFps = call.argument<Boolean>("includeFps") ?: true,
                        includeMasks = call.argument<Boolean>("includeMasks") ?: false,
                        includePoses = call.argument<Boolean>("includePoses") ?: false,
                        includeOBB = call.argument<Boolean>("includeOBB") ?: false,
                        includeOriginalImage = call.argument<Boolean>("includeOriginalImage") ?: false,
                        maxFPS = call.argument<Int>("maxFPS"),
                        throttleIntervalMs = call.argument<Int>("throttleInterval"),
                        inferenceFrequency = call.argument<Int>("inferenceFrequency"),
                        skipFrames = call.argument<Int>("skipFrames")
                    )
                    yoloView.setStreamConfig(streamConfig)
                    Log.d(TAG, "YOLOView streaming config updated: $streamConfig")
                    result.success(null)
                }
                "stop" -> {
                    Log.d(TAG, "Received manual stop call from Flutter")
                    try {
                        yoloView.stop()
                        Log.d(TAG, "YOLOView stopped successfully via method call")
                        result.success(null)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error stopping YOLOView via method call", e)
                        result.error("stop_error", "Error stopping YOLOView: ${e.message}", null)
                    }
                }
                "setModel" -> {
                    val modelPath = call.argument<String>("modelPath")
                    val taskString = call.argument<String>("task")
                    
                    if (modelPath == null || taskString == null) {
                        result.error("invalid_args", "modelPath and task are required", null)
                        return
                    }
                    
                    val task = YOLOTask.valueOf(taskString.uppercase())
                    Log.d(TAG, "Received setModel call with modelPath: $modelPath, task: $task")
                    
                    yoloView.setModel(modelPath, task) { success ->
                        if (success) {
                            Log.d(TAG, "Model switched successfully")
                            result.success(null)
                        } else {
                            Log.e(TAG, "Failed to switch model")
                            result.error("MODEL_NOT_FOUND", "Failed to load model: $modelPath", null)
                        }
                    }
                }
                "captureFrame" -> {
                    Log.d(TAG, "Received captureFrame call")
                    val imageData = yoloView.captureFrame()
                    if (imageData != null) {
                        Log.d(TAG, "Frame captured successfully: ${imageData.size} bytes")
                        result.success(imageData)
                    } else {
                        Log.e(TAG, "Failed to capture frame")
                        result.error("capture_failed", "Failed to capture frame from camera", null)
                    }
                }
                "listen" -> {
                    Log.d(TAG, "EventChannel listen method called")
                    // Called when EventChannel starts the stream
                    result.success(null)
                }
                "cancel" -> {
                    Log.d(TAG, "EventChannel cancel method called")
                    // Called when EventChannel cancels the stream
                    result.success(null)
                }
                else -> {
                    Log.w(TAG, "Method not implemented: ${call.method}")
                    result.notImplemented()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling method call: ${call.method}", e)
            result.error("method_call_error", "Error handling method call: ${e.message}", null)
        }
    }
    
    /**
     * Configure YOLOView streaming functionality based on creation parameters
     */
    private fun setupYOLOViewStreaming(creationParams: Map<String?, Any?>?) {
        // Parse streaming configuration from creationParams
        val streamingConfigParam = creationParams?.get("streamingConfig") as? Map<String, Any>
        
        val streamConfig = if (streamingConfigParam != null) {
            Log.d(TAG, "Creating YOLOStreamConfig from creation params: $streamingConfigParam")
            
            // Convert creation params to YOLOStreamConfig
            YOLOStreamConfig(
                includeDetections = streamingConfigParam["includeDetections"] as? Boolean ?: true,
                includeClassifications = streamingConfigParam["includeClassifications"] as? Boolean ?: true,
                includeProcessingTimeMs = streamingConfigParam["includeProcessingTimeMs"] as? Boolean ?: true,
                includeFps = streamingConfigParam["includeFps"] as? Boolean ?: true,
                includeMasks = streamingConfigParam["includeMasks"] as? Boolean ?: true,
                includePoses = streamingConfigParam["includePoses"] as? Boolean ?: true,
                includeOBB = streamingConfigParam["includeOBB"] as? Boolean ?: true,
                includeOriginalImage = streamingConfigParam["includeOriginalImage"] as? Boolean ?: false,
                maxFPS = when (val maxFPS = streamingConfigParam["maxFPS"]) {
                    is Int -> maxFPS
                    is Double -> maxFPS.toInt()
                    is String -> maxFPS.toIntOrNull()
                    else -> null
                },
                throttleIntervalMs = when (val throttleMs = streamingConfigParam["throttleIntervalMs"]) {
                    is Int -> throttleMs
                    is Double -> throttleMs.toInt()
                    is String -> throttleMs.toIntOrNull()
                    else -> null
                }
            )
        } else {
            // Use default minimal configuration for optimal performance
            Log.d(TAG, "Using default streaming config")
            YOLOStreamConfig.DEFAULT
        }
        
        // Configure YOLOView with the stream config
        yoloView.setStreamConfig(streamConfig)
        Log.d(TAG, "YOLOView streaming configured: $streamConfig")
        
        // Set up streaming callback to forward data to Flutter via event channel
        yoloView.setStreamCallback { streamData ->
            // Forward streaming data from YOLOView to Flutter
            sendStreamDataToFlutter(streamData)
        }
    }
    
    /**
     * Send stream data to Flutter via event channel
     */
    private fun sendStreamDataToFlutter(streamData: Map<String, Any>) {
        try {
            
            // Create a runnable to ensure we're on the main thread
            val sendResults = Runnable {
                try {
                    if (streamHandler is CustomStreamHandler) {
                        val customHandler = streamHandler as CustomStreamHandler
                        
                        // Use the safe send method
                        val sent = customHandler.safelySend(streamData)
                        if (sent) {
                        } else {
                            Log.w(TAG, "Failed to send stream data via CustomStreamHandler")
                            // Notify Flutter to recreate the channel
                            methodChannel?.invokeMethod("recreateEventChannel", null)
                        }
                    } else {
                        // Use reflection to access the sink property regardless of exact type
                        Log.d(TAG, "Attempting to access sink via reflection")
                        val fields = streamHandler.javaClass.declaredFields
                        Log.d(TAG, "Available fields: ${fields.joinToString { it.name }}")
                        
                        val sinkField = streamHandler.javaClass.getDeclaredField("sink")
                        sinkField.isAccessible = true
                        val sink = sinkField.get(streamHandler) as? EventChannel.EventSink
                        
                        if (sink != null) {
                            sink.success(streamData)
                        } else {
                            Log.w(TAG, "Event sink is NOT available via reflection, skipping data")
                            // Try alternative approach - recreate the event channel
                            Log.d(TAG, "Requesting Flutter to recreate event channel")
                            methodChannel?.invokeMethod("recreateEventChannel", null)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error sending stream data on main thread", e)
                    e.printStackTrace()
                }
            }
            
            // Make sure we're on the main thread when sending events
            val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
            mainHandler.post(sendResults)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing stream data", e)
            e.printStackTrace()
        }
    }

    override fun getView(): View {
        Log.d(TAG, "Getting view: ${yoloView.javaClass.simpleName}")
        
        // Check if context is a LifecycleOwner
        if (context is androidx.lifecycle.LifecycleOwner) {
            val lifecycleOwner = context as androidx.lifecycle.LifecycleOwner
            Log.d(TAG, "Context is a LifecycleOwner with state: ${lifecycleOwner.lifecycle.currentState}")
        } else {
            Log.e(TAG, "Context is NOT a LifecycleOwner! This may cause camera issues.")
        }
        
        // Try setting custom layout parameters
        if (yoloView.layoutParams == null) {
            yoloView.layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT
            )
            Log.d(TAG, "Set layout params for YOLOView")
        }
        
        return yoloView
    }

    override fun dispose() {
        Log.d(TAG, "Disposing YOLOPlatformView for viewId: $viewId")

        try {
            // Stop camera and inference before disposing
            Log.d(TAG, "Calling yoloView.stop() to stop camera and inference")
            yoloView.stop()

            // Clean up method channel
            Log.d(TAG, "Clearing method channel handler")
            methodChannel?.setMethodCallHandler(null)

            // Notify the factory that this view is disposed
            Log.d(TAG, "Notifying factory of disposal")
            factory.onPlatformViewDisposed(viewId)

            Log.d(TAG, "YOLOPlatformView disposal completed successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Error during YOLOPlatformView disposal", e)
        }
    }

    /**
     * Called by YOLOPlatformViewFactory when the Activity (which is a LifecycleOwner)
     * becomes available or changes.
     */
    fun notifyLifecycleOwnerAvailable(owner: LifecycleOwner) {
        Log.d(TAG, "LifecycleOwner (${owner.javaClass.simpleName}) is now available for viewId: $viewId. Notifying YOLOView.")
        yoloView.onLifecycleOwnerAvailable(owner)
    }
        
        // Called by YOLOPlugin to delegate permission results
        fun passRequestPermissionsResult(
            requestCode: Int,
            permissions: Array<String>, 
            grantResults: IntArray
        ) {
            Log.d(TAG, "passRequestPermissionsResult called in YOLOPlatformView for viewId $viewId, delegating to yoloView")
            yoloView.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
        
        /**
         * Sets a new model on the YoloView
         * @param modelPath Path to the new model
         * @param task The YOLO task type
         * @param callback Callback to report success/failure
         */
        fun setModel(modelPath: String, task: YOLOTask, callback: ((Boolean) -> Unit)? = null) {
            Log.d(TAG, "setModel called for viewId $viewId with model: $modelPath, task: $task")
            yoloView.setModel(modelPath, task, callback)
        }
    
    /**
     * Resolves a model path that might be relative to app's internal storage
     * @param context Application context
     * @param modelPath The model path from Flutter
     * @return Resolved absolute path or original asset path
     */
    private fun resolveModelPath(context: Context, modelPath: String): String {
        // If it's already an absolute path, return it
        if (YOLOUtils.isAbsolutePath(modelPath)) {
            return modelPath
        }
        
        // Check if it's a relative path to internal storage
        if (modelPath.startsWith("internal://")) {
            val relativePath = modelPath.substring("internal://".length)
            return "${context.filesDir.absolutePath}/$relativePath"
        }
        
        // Otherwise, consider it an asset path
        return modelPath
    }
}
