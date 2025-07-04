// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.app.Activity
import android.content.Context
import android.util.Log
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory
import java.util.ArrayList

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
 * Factory for creating YOLOPlatformView instances
 */
class YOLOPlatformViewFactory(
    private val messenger: BinaryMessenger
) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {
    private var activity: Activity? = null
    private val TAG = "YOLOPlatformViewFactory"
    // Map to store active views, accessible by YoloPlugin
    internal val activeViews = mutableMapOf<Int, YOLOPlatformView>()
    // Map to store event channel handlers, keyed by viewId
    private val eventChannelHandlers = mutableMapOf<Int, EventChannel.StreamHandler>()
    
    // Store activity reference to pass to the YOLOPlatformView
    fun setActivity(activity: Activity?) {
        this.activity = activity
        Log.d(TAG, "Activity set: ${activity?.javaClass?.simpleName}")

        // If activity is available, notify all active views
        if (activity != null) {
            // Iterate over a copy of the values to avoid concurrent modification if a view disposes itself during this call
            val viewsToNotify = ArrayList(activeViews.values)
            for (view in viewsToNotify) {
                try {
                    // YoloPlatformView's notifyLifecycleOwnerAvailable expects a LifecycleOwner.
                    // Activity is a LifecycleOwner if it's a ComponentActivity or FragmentActivity.
                    if (activity is LifecycleOwner) {
                        view.notifyLifecycleOwnerAvailable(activity as LifecycleOwner)
                        // It's good practice to log the viewId for easier debugging if multiple views exist.
                        // Assuming YoloPlatformView has a public 'viewId' property or a getter.
                        // If not, you might need to adjust how you log this.
                        // For now, let's assume view.viewId is accessible or use a placeholder.
                        Log.d(TAG, "Notified YoloPlatformView (viewId: ${view.hashCode()}) that activity is available.")
                    } else {
                        Log.w(TAG, "Activity is not a LifecycleOwner, cannot notify view ${view.hashCode()}")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error notifying YoloPlatformView (viewId: ${view.hashCode()}) about available activity", e)
                }
            }
        }
    }
    
    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        val creationParams = args as? Map<String?, Any?>
        
        // Use activity if available, otherwise use the provided context
        val effectiveContext = activity ?: context
        Log.d(TAG, "Creating YOLOPlatformView with context: ${effectiveContext.javaClass.simpleName}, platform int viewId: $viewId")
        Log.d(TAG, "Raw creationParams from Dart: $args")

        // Get the unique ID for this view
        val dartViewIdParam = creationParams?.get("viewId")
        Log.d(TAG, "Extracted 'viewId' from creationParams: $dartViewIdParam (type: ${dartViewIdParam?.javaClass?.name})")
        val viewUniqueId = dartViewIdParam as? String ?: viewId.toString().also {
            Log.w(TAG, "Using platform int viewId '$it' as fallback for viewUniqueId because Dart 'viewId' was null or not a String.")
        }
        Log.d(TAG, "Resolved viewUniqueId for channel naming: $viewUniqueId")
        
        // Create event channel for detection results
        val resultChannelName = "com.ultralytics.yolo/detectionResults_$viewUniqueId"
        val controlChannelName = "com.ultralytics.yolo/controlChannel_$viewUniqueId"
        
        Log.d(TAG, "Final channel names - Result: $resultChannelName, Control: $controlChannelName")
        
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
        val platformView = YOLOPlatformView(
            effectiveContext,
            viewId,
            creationParams,
            eventHandler, // Pass the entire StreamHandler now
            methodChannel,
            this // Pass the factory itself for disposal callback
        )
        activeViews[viewId] = platformView
        Log.d(TAG, "Created and stored YOLOPlatformView for viewId: $viewId")
        return platformView
    }
    
    // Called by YOLOPlatformView when it's disposed
    internal fun onPlatformViewDisposed(viewId: Int) {
        activeViews.remove(viewId)
        eventChannelHandlers.remove(viewId) // Assuming CustomStreamHandler doesn't need explicit cancel on its EventChannel
        Log.d(TAG, "YOLOPlatformView for viewId $viewId disposed and removed from factory.")
    }
    
    fun dispose() { // Called when the FlutterEngine is detached
        // Clean up event channels when the plugin is disposed
        eventChannelHandlers.clear()
        activeViews.clear()
        Log.d(TAG, "YOLOPlatformViewFactory disposed, all views and handlers cleared.")
    }
}