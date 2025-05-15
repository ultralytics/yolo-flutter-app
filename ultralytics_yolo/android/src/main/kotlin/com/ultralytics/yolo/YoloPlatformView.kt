// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import android.view.View
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import java.io.ByteArrayOutputStream
import java.util.ArrayList
import java.util.HashMap

/**
 * YoloPlatformView - Native view bridge from Flutter
 */
class YoloPlatformView(
    private val context: Context,
    private val viewId: Int,
    creationParams: Map<String?, Any?>?,
    private val streamHandler: EventChannel.StreamHandler,
    private val methodChannel: MethodChannel?
) : PlatformView, MethodChannel.MethodCallHandler {

    private val yoloView: YoloView = YoloView(context)
    private val TAG = "YoloPlatformView"
    
    // Initialization flag
    private var initialized = false
    
    // Frame counter for tracking inference results
    private var frameNumberCounter: Long = 0
    
    // Unique ID to send to Flutter
    private val viewUniqueId: String = creationParams?.get("viewId") as? String ?: viewId.toString()
    
    init {
        // Parse model path and task from creation params
        var modelPath = creationParams?.get("modelPath") as? String ?: "yolo11n"
        val taskString = creationParams?.get("task") as? String ?: "detect"
        val threshold = creationParams?.get("confidenceThreshold") as? Double ?: 0.5
        val iouThreshold = creationParams?.get("iouThreshold") as? Double ?: 0.45
        
        // Set up the method channel handler
        methodChannel?.setMethodCallHandler(this)
        
        try {
            // Resolve model path (handling absolute paths, internal:// scheme, or asset paths)
            modelPath = resolveModelPath(context, modelPath)
            
            // Convert task string to enum
            val task = YOLOTask.valueOf(taskString.uppercase())
            
            Log.d(TAG, "Initializing YoloPlatformView with model: $modelPath, task: $task, " +
                   "threshold: $threshold, iouThreshold: $iouThreshold, viewId: $viewId")
            
            // Set up callback for model loading result
            yoloView.setOnModelLoadCallback { success ->
                if (success) {
                    Log.d(TAG, "Model loaded successfully: $modelPath")
                    
                    // Set thresholds
                    yoloView.setConfidenceThreshold(threshold)
                    yoloView.setIouThreshold(iouThreshold)
                    
                    // Initialize camera after model is loaded
                    if (!initialized) {
                        initialized = true
                        Log.d(TAG, "Initializing camera")
                        yoloView.initCamera()
                    }
                } else {
                    Log.e(TAG, "Failed to load model: $modelPath")
                }
            }
            
            // Set up callback for inference results
            yoloView.setOnInferenceCallback { result ->
                Log.d(TAG, "*** Inference result received with ${result.boxes.size} detections ***")
                
                // Get the event sink property from our stream handler
                try {
                    Log.d(TAG, "StreamHandler class: ${streamHandler.javaClass.name}")
                    
                    // First convert results to map - do this outside the sink checks to debug data
                    val resultsMap = convertResultToMap(result)
                    Log.d(TAG, "Results converted to map, ready to send: ${resultsMap.keys.joinToString()}")
                    
                    // Create a runnable to ensure we're on the main thread
                    val sendResults = Runnable {
                        try {
                            if (streamHandler is CustomStreamHandler) {
                                val customHandler = streamHandler as CustomStreamHandler
                                Log.d(TAG, "Using CustomStreamHandler - is sink valid: ${customHandler.isSinkValid()}")
                                
                                // Add timestamp and frame information to the results
                                val enhancedResultsMap = HashMap<String, Any>(resultsMap)
                                enhancedResultsMap["timestamp"] = System.currentTimeMillis()
                                enhancedResultsMap["frameNumber"] = frameNumberCounter++
                                
                                // Use the safe send method
                                val sent = customHandler.safelySend(enhancedResultsMap)
                                if (sent) {
                                    Log.d(TAG, "Successfully sent results via CustomStreamHandler's safelySend")
                                } else {
                                    Log.w(TAG, "Failed to send results via CustomStreamHandler's safelySend")
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
                                    Log.d(TAG, "Sending results to Flutter via event sink (reflection)")
                                    
                                    // Add timestamp and frame info
                                    val enhancedResultsMap = HashMap<String, Any>(resultsMap)
                                    enhancedResultsMap["timestamp"] = System.currentTimeMillis()
                                    enhancedResultsMap["frameNumber"] = frameNumberCounter++
                                    
                                    sink.success(enhancedResultsMap)
                                    Log.d(TAG, "Successfully sent results via reflection")
                                } else {
                                    Log.w(TAG, "Event sink is NOT available via reflection, skipping result")
                                    
                                    // Try alternative approach - recreate the event channel
                                    Log.d(TAG, "Requesting Flutter to recreate event channel")
                                    methodChannel?.invokeMethod("recreateEventChannel", null)
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error sending results on main thread", e)
                            e.printStackTrace()
                        }
                    }
                    
                    // Make sure we're on the main thread when sending events
                    val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
                    mainHandler.post(sendResults)
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing inference result", e)
                    e.printStackTrace()
                }
            }
            
            // Load model with the specified path and task
            yoloView.setModel(modelPath, task, context)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YoloPlatformView", e)
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
                "setThresholds" -> {
                    val confidenceThreshold = call.argument<Double>("confidenceThreshold")
                    val iouThreshold = call.argument<Double>("iouThreshold")
                    
                    if (confidenceThreshold != null) {
                        Log.d(TAG, "Setting confidence threshold to $confidenceThreshold")
                        yoloView.setConfidenceThreshold(confidenceThreshold)
                    }
                    if (iouThreshold != null) {
                        Log.d(TAG, "Setting IoU threshold to $iouThreshold")
                        yoloView.setIouThreshold(iouThreshold)
                    }
                    
                    result.success(null)
                }
                "switchCamera" -> {
                    Log.d(TAG, "Switching camera")
                    yoloView.switchCamera()
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
    
    // Convert YOLOResult to a Map for sending to Flutter
    private fun convertResultToMap(result: YOLOResult): Map<String, Any> {
        val map = HashMap<String, Any>()
        val detections = ArrayList<Map<String, Any>>()
        
        // Convert detection boxes
        for (box in result.boxes) {
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
            
            detections.add(detection)
        }
        
        map["detections"] = detections
        
        // Convert speed metrics
        map["processingTimeMs"] = result.speed
        
        // Optionally convert annotated image if available
        result.annotatedImage?.let { bitmap ->
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
            map["annotatedImage"] = outputStream.toByteArray()
        }

        // Log the structure of the sent data for debugging
        Log.d(TAG, "Sending detection data to Flutter: ${detections.size} boxes")
        if (detections.isNotEmpty()) {
            val firstDetection = detections.first()
            Log.d(TAG, "First detection keys: ${firstDetection.keys.joinToString()}")
            Log.d(TAG, "First detection class: ${firstDetection["className"]}, confidence: ${firstDetection["confidence"]}")
        }
        
        return map
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
            Log.d(TAG, "Set layout params for YoloView")
        }
        
        return yoloView
    }

    override fun dispose() {
        Log.d(TAG, "Disposing YoloPlatformView")
        // Clean up resources
        methodChannel?.setMethodCallHandler(null)
    }
    
    /**
     * Resolves a model path that might be relative to app's internal storage
     * @param context Application context
     * @param modelPath The model path from Flutter
     * @return Resolved absolute path or original asset path
     */
    private fun resolveModelPath(context: Context, modelPath: String): String {
        // If it's already an absolute path, return it
        if (YoloUtils.isAbsolutePath(modelPath)) {
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