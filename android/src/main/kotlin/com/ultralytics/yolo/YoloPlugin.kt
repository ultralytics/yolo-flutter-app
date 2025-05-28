// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Log
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry // Added for RequestPermissionsResultListener
import java.io.ByteArrayOutputStream

class YoloPlugin : FlutterPlugin, ActivityAware, MethodChannel.MethodCallHandler, PluginRegistry.RequestPermissionsResultListener {

  private lateinit var methodChannel: MethodChannel
  private var yolo: YOLO? = null
  private lateinit var applicationContext: android.content.Context
  private var activity: Activity? = null
  private var activityBinding: ActivityPluginBinding? = null // Added to store the binding
  private val TAG = "YoloPlugin"
  private lateinit var viewFactory: YoloPlatformViewFactory

  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    // Store application context for later use
    applicationContext = flutterPluginBinding.applicationContext

    // Create and store the view factory for later activity updates
    viewFactory = YoloPlatformViewFactory(flutterPluginBinding.binaryMessenger)
    
    // Register platform view
    flutterPluginBinding.platformViewRegistry.registerViewFactory(
      "com.ultralytics.yolo/YoloPlatformView",
      viewFactory
    )

    // Register method channel for single-image
    methodChannel = MethodChannel(
      flutterPluginBinding.binaryMessenger,
      "yolo_single_image_channel"
    )
    methodChannel.setMethodCallHandler(this)
    
    Log.d(TAG, "YoloPlugin attached to engine")
  }
  
  override fun onAttachedToActivity(binding: ActivityPluginBinding) {
    activity = binding.activity
    activityBinding = binding // Store the binding
    viewFactory.setActivity(activity)
    activityBinding?.addRequestPermissionsResultListener(this)
    Log.d(TAG, "YoloPlugin attached to activity: ${activity?.javaClass?.simpleName}, stored binding, and added RequestPermissionsResultListener")
  }

  override fun onDetachedFromActivityForConfigChanges() {
    Log.d(TAG, "YoloPlugin detached from activity for config changes. Listener will be removed in onDetachedFromActivity.")
    // activity and viewFactory.setActivity(null) will be handled by onDetachedFromActivity
    // activityBinding will also be cleared in onDetachedFromActivity
  }

  override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
    activity = binding.activity
    activityBinding = binding // Store the new binding
    viewFactory.setActivity(activity)
    activityBinding?.addRequestPermissionsResultListener(this) // Add listener with new binding
    Log.d(TAG, "YoloPlugin reattached to activity: ${activity?.javaClass?.simpleName}, stored new binding, and re-added RequestPermissionsResultListener")
  }

  override fun onDetachedFromActivity() {
    Log.d(TAG, "YoloPlugin detached from activity")
    activityBinding?.removeRequestPermissionsResultListener(this)
    activityBinding = null
    activity = null
    viewFactory.setActivity(null)
    Log.d(TAG, "Cleared activity, activityBinding, and removed RequestPermissionsResultListener")
  }

  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    methodChannel.setMethodCallHandler(null)
    Log.d(TAG, "YoloPlugin detached from engine")
    // Clean up view factory resources
    viewFactory.dispose()
    // YOLO class doesn't need explicit release
  }
  
  /**
   * Gets the absolute path to the app's internal storage directory
   */
  private fun getInternalStoragePath(): String {
    return applicationContext.filesDir.absolutePath
  }

  /**
   * Resolves a model path that might be relative to app's internal storage
   * @param modelPath The model path from Flutter
   * @return Resolved absolute path or original asset path
   */
  private fun resolveModelPath(modelPath: String): String {
    // If it's already an absolute path, return it
    if (YoloUtils.isAbsolutePath(modelPath)) {
      return modelPath
    }
    
    // Check if it's a relative path to internal storage
    if (modelPath.startsWith("internal://")) {
      val relativePath = modelPath.substring("internal://".length)
      return "${applicationContext.filesDir.absolutePath}/$relativePath"
    }
    
    // Otherwise, consider it an asset path
    return modelPath
  }

  override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
    when (call.method) {
      "loadModel" -> {
        try {
          val args = call.arguments as? Map<*, *>
          var modelPath = args?.get("modelPath") as? String ?: "yolo11n"
          val taskString = args?.get("task") as? String ?: "detect"
          
          // Resolve the model path (handling absolute paths, internal:// scheme, or asset paths)
          modelPath = resolveModelPath(modelPath)
          
          // Convert task string to enum
          val task = YOLOTask.valueOf(taskString.uppercase())
          
          // Load labels (in real implementation, you would load from metadata)
          val labels = loadLabels(modelPath)
          
          // Initialize YOLO with new implementation
          yolo = YOLO(
            context = applicationContext,
            modelPath = modelPath,
            task = task,
            labels = labels,
            useGpu = true
          )
          
          Log.d(TAG, "Model loaded successfully: $modelPath for task: $task")
          result.success(true)
        } catch (e: Exception) {
          Log.e(TAG, "Failed to load model", e)
          result.error("model_error", "Failed to load model: ${e.message}", null)
        }
      }

      "predictSingleImage" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val imageData = args?.get("image") as? ByteArray

          if (imageData == null) {
            result.error("bad_args", "No image data", null)
            return
          }
          
          if (yolo == null) {
            result.error("not_initialized", "Model not loaded", null)
            return
          }
          
          // Convert byte array to bitmap
          val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
          if (bitmap == null) {
            result.error("image_error", "Failed to decode image", null)
            return
          }
          
          // Run inference with new YOLO implementation
          val yoloResult = yolo!!.predict(bitmap, rotateForCamera = false)
          
          // Create response
          val response = HashMap<String, Any>()
          
          // Convert boxes to map for Flutter
          response["boxes"] = yoloResult.boxes.map { box ->
            mapOf(
              "x1" to box.xywh.left,
              "y1" to box.xywh.top,
              "x2" to box.xywh.right,
              "y2" to box.xywh.bottom,
              "class" to box.cls,
              "confidence" to box.conf
            )
          }
          
          // Add task-specific data to response
          when (yolo!!.task) {
            YOLOTask.SEGMENT -> {
              // Include segmentation mask if available
              yoloResult.masks?.combinedMask?.let { mask ->
                val stream = ByteArrayOutputStream()
                mask.compress(Bitmap.CompressFormat.PNG, 90, stream)
                response["mask"] = stream.toByteArray()
              }
            }
            YOLOTask.CLASSIFY -> {
              // Include classification results if available
              yoloResult.probs?.let { probs ->
                response["classification"] = mapOf(
                  "topClass" to probs.top1,
                  "topConfidence" to probs.top1Conf,
                  "top5Classes" to probs.top5,
                  "top5Confidences" to probs.top5Confs
                )
              }
            }
            YOLOTask.POSE -> {
              // Include pose keypoints if available
              if (yoloResult.keypointsList.isNotEmpty()) {
                response["keypoints"] = yoloResult.keypointsList.map { keypoints ->
                  mapOf(
                    "coordinates" to keypoints.xyn.mapIndexed { i, (x, y) ->
                      mapOf("x" to x, "y" to y, "confidence" to keypoints.conf[i])
                    }
                  )
                }
              }
            }
            YOLOTask.OBB -> {
              // Include oriented bounding boxes if available
              if (yoloResult.obb.isNotEmpty()) {
                response["obb"] = yoloResult.obb.map { obb ->
                  val poly = obb.box.toPolygon()
                  mapOf(
                    "points" to poly.map { mapOf("x" to it.x, "y" to it.y) },
                    "class" to obb.cls,
                    "confidence" to obb.confidence
                  )
                }
              }
            }
            else -> {} // DETECT is handled by boxes
          }
          
          // Include annotated image in response
          yoloResult.annotatedImage?.let { annotated ->
            val stream = ByteArrayOutputStream()
            annotated.compress(Bitmap.CompressFormat.JPEG, 90, stream)
            response["annotatedImage"] = stream.toByteArray()
          }
          
          // Include inference speed
          response["speed"] = yoloResult.speed
          
          result.success(response)
        } catch (e: Exception) {
          Log.e(TAG, "Error during prediction", e)
          result.error("prediction_error", "Error during prediction: ${e.message}", null)
        }
      }

      "checkModelExists" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val originalPath = args?.get("modelPath") as? String ?: ""
          val modelPath = resolveModelPath(originalPath)
          
          val checkResult = YoloUtils.checkModelExistence(applicationContext, modelPath)
          result.success(checkResult)
        } catch (e: Exception) {
          result.error("check_error", "Failed to check model: ${e.message}", null)
        }
      }
      // END OF "checkModelExists" case
      
      "getStoragePaths" -> {
        try {
          val paths = mapOf(
            "internal" to applicationContext.filesDir.absolutePath,
            "cache" to applicationContext.cacheDir.absolutePath,
            "external" to applicationContext.getExternalFilesDir(null)?.absolutePath,
            "externalCache" to applicationContext.externalCacheDir?.absolutePath
          )
          result.success(paths)
        } catch (e: Exception) {
          result.error("path_error", "Failed to get storage paths: ${e.message}", null)
        }
      }
      
      "setModel" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val viewId = args?.get("viewId") as? Int
          val modelPath = args?.get("modelPath") as? String
          val taskString = args?.get("task") as? String
          
          if (viewId == null || modelPath == null || taskString == null) {
            result.error("bad_args", "Missing required arguments for setModel", null)
            return
          }
          
          // Get the YoloView instance from the factory
          val yoloView = viewFactory.activeViews[viewId]
          if (yoloView != null) {
            // Resolve the model path
            val resolvedPath = resolveModelPath(modelPath)
            
            // Convert task string to enum
            val task = YOLOTask.valueOf(taskString.uppercase())
            
            // Call setModel on the YoloView
            yoloView.setModel(resolvedPath, task) { success ->
              if (success) {
                result.success(null)
              } else {
                result.error("MODEL_NOT_FOUND", "Failed to load model: $modelPath", null)
              }
            }
          } else {
            result.error("VIEW_NOT_FOUND", "YoloView with id $viewId not found", null)
          }
        } catch (e: Exception) {
          Log.e(TAG, "Error setting model", e)
          result.error("set_model_error", "Error setting model: ${e.message}", null)
        }
      }
      
      else -> result.notImplemented()
    }
  }

  // Implementation for PluginRegistry.RequestPermissionsResultListener
  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ): Boolean {
    Log.d(TAG, "onRequestPermissionsResult called in YoloPlugin. requestCode: $requestCode, activeViews: ${viewFactory.activeViews.size}")
    var handled = false
    // Iterate over a copy of the values to avoid concurrent modification issues.
    val viewsToNotify = ArrayList(viewFactory.activeViews.values)
    for (platformView in viewsToNotify) {
        try {
            // YoloPlatformView has the passRequestPermissionsResult method
            platformView.passRequestPermissionsResult(requestCode, permissions, grantResults)
            // YoloPlatformView's passRequestPermissionsResult will log its own viewId
            Log.d(TAG, "Successfully attempted to delegate permission result to an active YoloPlatformView.")
            handled = true
            // Assuming only one view actively requests permissions at a time.
            // If multiple views could request, 'handled' logic might need adjustment
            // or ensure only the correct view processes it.
        } catch (e: Exception) {
            Log.e(TAG, "Error delegating onRequestPermissionsResult to a YoloPlatformView instance", e)
        }
    }
    if (!handled && viewsToNotify.isNotEmpty()) {
        // This log means we iterated views but none seemed to handle it, or an exception occurred.
        Log.w(TAG, "onRequestPermissionsResult was iterated but not confirmed handled by any YoloPlatformView, or an error occurred during delegation.")
    } else if (viewsToNotify.isEmpty()) {
        Log.d(TAG, "onRequestPermissionsResult: No active YoloPlatformViews to notify.")
    }
    return handled // Return true if any view instance successfully processed it.
  }
  
  // Helper function to load labels
  private fun loadLabels(modelPath: String): List<String> {
    // This is a placeholder - in a real implementation, you would load labels from metadata
    return listOf(
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack"
    )
  }
}