package com.ultralytics.yolo

import android.app.Activity
import android.content.Context
import android.util.Log
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory

// Custom stream handler class to expose the event sink
class CustomStreamHandler(private val viewId: Int) : EventChannel.StreamHandler {
    // Make sink volatile to ensure visibility across threads
    @Volatile
    var sink: EventChannel.EventSink? = null
    private val TAG = "CustomStreamHandler"
    
    // Add a timestamp to track when the sink was last set
    private var sinkSetTime: Long = 0
    
    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        Log.d(TAG, "Event channel for view $viewId started listening")
        
        // Ensure we're on the main thread for sink operations
        if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
            Log.w(TAG, "onListen not called on main thread! Current thread: ${Thread.currentThread().name}")
            val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
            mainHandler.post {
                handleOnListen(arguments, events)
            }
        } else {
            handleOnListen(arguments, events)
        }
    }
    
    private fun handleOnListen(arguments: Any?, events: EventChannel.EventSink?) {
        sink = events
        sinkSetTime = System.currentTimeMillis()
        Log.d(TAG, "Sink set on main thread at ${sinkSetTime}, sink: $sink")
        
        if (events == null) {
            Log.w(TAG, "onListen called with null EventSink!")
        } else {
            // Test the sink by sending a simple message
            try {
                Log.d(TAG, "Testing event sink with a test message")
                events.success(mapOf(
                    "test" to "Event channel active", 
                    "viewId" to viewId,
                    "timestamp" to System.currentTimeMillis()
                ))
                Log.d(TAG, "Test message sent successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error sending test message to event sink", e)
            }
            
            // Schedule a delayed test message to verify sink stays active
            val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
            mainHandler.postDelayed({
                try {
                    if (sink != null) {
                        Log.d(TAG, "Sending delayed test message to verify sink")
                        sink?.success(mapOf(
                            "test" to "Event sink verification", 
                            "viewId" to viewId,
                            "timestamp" to System.currentTimeMillis(),
                            "sinkAge" to (System.currentTimeMillis() - sinkSetTime)
                        ))
                        Log.d(TAG, "Delayed test message sent successfully")
                    } else {
                        Log.w(TAG, "Sink no longer available for delayed test")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error sending delayed test message", e)
                }
            }, 1000) // 1 second delay
        }
    }
    
    override fun onCancel(arguments: Any?) {
        Log.d(TAG, "Event channel for view $viewId cancelled after ${System.currentTimeMillis() - sinkSetTime}ms, clearing sink")
        
        // Ensure we're on the main thread for sink operations
        if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
            val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
            mainHandler.post {
                sink = null
                Log.d(TAG, "Sink cleared on main thread")
            }
        } else {
            sink = null
            Log.d(TAG, "Sink cleared directly")
        }
    }
    
    // Method to check if sink is valid
    fun isSinkValid(): Boolean {
        return sink != null
    }
    
    // Method to safely send a message
    fun safelySend(data: Map<String, Any>): Boolean {
        if (sink == null) return false
        
        try {
            // Always send on main thread
            if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
                val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
                mainHandler.post {
                    try {
                        sink?.success(data)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error sending data on main thread", e)
                    }
                }
            } else {
                sink?.success(data)
            }
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error in safelySend", e)
            return false
        }
    }
}

/**
 * Factory for creating YoloPlatformView instances
 */
class YoloPlatformViewFactory(
    private val messenger: BinaryMessenger
) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {
    private var activity: Activity? = null
    private val TAG = "YoloPlatformViewFactory"
    private val eventChannelHandlers = mutableMapOf<Int, EventChannel.StreamHandler>()
    
    // Store activity reference to pass to the YoloPlatformView
    fun setActivity(activity: Activity?) {
        this.activity = activity
        Log.d(TAG, "Activity set: ${activity?.javaClass?.simpleName}")
    }
    
    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        val creationParams = args as? Map<String?, Any?>
        
        // Use activity if available, otherwise use the provided context
        val effectiveContext = activity ?: context
        Log.d(TAG, "Creating YoloPlatformView with context: ${effectiveContext.javaClass.simpleName}")
        
        // Get the unique ID for this view
        val viewUniqueId = creationParams?.get("viewId") as? String ?: viewId.toString()
        
        // Create event channel for detection results
        val resultChannelName = "com.ultralytics.yolo/detectionResults_$viewUniqueId"
        val controlChannelName = "com.ultralytics.yolo/controlChannel_$viewUniqueId"
        
        Log.d(TAG, "Creating channels: $resultChannelName, $controlChannelName")
        
        // Event channel for streaming detection results
        val eventChannel = EventChannel(messenger, resultChannelName)
        // Method channel for controlling the view
        val methodChannel = MethodChannel(messenger, controlChannelName)
        
        // Create stream handler for detection results
        val eventHandler = CustomStreamHandler(viewId)
        Log.d(TAG, "Created CustomStreamHandler for view $viewId")
        
        // Set event handler and store it
        eventChannel.setStreamHandler(eventHandler)
        eventChannelHandlers[viewId] = eventHandler
        
        // Create the platform view with stream handler, not just the sink
        return YoloPlatformView(
            effectiveContext,
            viewId,
            creationParams,
            eventHandler, // Pass the entire StreamHandler now
            methodChannel
        )
    }
    
    fun dispose() {
        // Clean up event channels when the plugin is disposed
        eventChannelHandlers.clear()
    }
}