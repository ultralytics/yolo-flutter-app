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

class YOLOPlugin : FlutterPlugin, ActivityAware, MethodChannel.MethodCallHandler, PluginRegistry.RequestPermissionsResultListener {

  private lateinit var methodChannel: MethodChannel
  private val instanceChannels = mutableMapOf<String, MethodChannel>()
  private lateinit var applicationContext: android.content.Context
  private var activity: Activity? = null
  private var activityBinding: ActivityPluginBinding? = null // Added to store the binding
  private val TAG = "YOLOPlugin"
  private lateinit var viewFactory: YOLOPlatformViewFactory
  private lateinit var binaryMessenger: io.flutter.plugin.common.BinaryMessenger

  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    // Store application context and binary messenger for later use
    applicationContext = flutterPluginBinding.applicationContext
    binaryMessenger = flutterPluginBinding.binaryMessenger

    // Create and store the view factory for later activity updates
    viewFactory = YOLOPlatformViewFactory(flutterPluginBinding.binaryMessenger)
    
    // Register platform view
    flutterPluginBinding.platformViewRegistry.registerViewFactory(
      "com.ultralytics.yolo/YOLOPlatformView",
      viewFactory
    )

    // Register default method channel for backward compatibility
    methodChannel = MethodChannel(
      flutterPluginBinding.binaryMessenger,
      "yolo_single_image_channel"
    )
    methodChannel.setMethodCallHandler(this)
    
    Log.d(TAG, "YOLOPlugin attached to engine")
  }
  
  override fun onAttachedToActivity(binding: ActivityPluginBinding) {
    activity = binding.activity
    activityBinding = binding // Store the binding
    viewFactory.setActivity(activity)
    activityBinding?.addRequestPermissionsResultListener(this)
    Log.d(TAG, "YOLOPlugin attached to activity: ${activity?.javaClass?.simpleName}, stored binding, and added RequestPermissionsResultListener")
  }

  override fun onDetachedFromActivityForConfigChanges() {
    Log.d(TAG, "YOLOPlugin detached from activity for config changes. Listener will be removed in onDetachedFromActivity.")
    // activity and viewFactory.setActivity(null) will be handled by onDetachedFromActivity
    // activityBinding will also be cleared in onDetachedFromActivity
  }

  override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
    activity = binding.activity
    activityBinding = binding // Store the new binding
    viewFactory.setActivity(activity)
    activityBinding?.addRequestPermissionsResultListener(this) // Add listener with new binding
    Log.d(TAG, "YOLOPlugin reattached to activity: ${activity?.javaClass?.simpleName}, stored new binding, and re-added RequestPermissionsResultListener")
  }

  override fun onDetachedFromActivity() {
    Log.d(TAG, "YOLOPlugin detached from activity")
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
    if (YOLOUtils.isAbsolutePath(modelPath)) {
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
      "createInstance" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val instanceId = args?.get("instanceId") as? String
          
          if (instanceId == null) {
            result.error("bad_args", "Missing instanceId", null)
            return
          }
          
          // Create instance placeholder
          YOLOInstanceManager.shared.createInstance(instanceId)
          
          // Register a new channel for this instance
          val channelName = "yolo_single_image_channel_$instanceId"
          val instanceChannel = MethodChannel(binaryMessenger, channelName)
          instanceChannel.setMethodCallHandler(this)
          instanceChannels[instanceId] = instanceChannel
          
          result.success(null)
        } catch (e: Exception) {
          Log.e(TAG, "Error creating instance", e)
          result.error("create_error", "Failed to create instance: ${e.message}", null)
        }
      }
      
      "loadModel" -> {
        try {
          val args = call.arguments as? Map<*, *>
          var modelPath = args?.get("modelPath") as? String ?: "yolo11n"
          val taskString = args?.get("task") as? String ?: "detect"
          val instanceId = args?.get("instanceId") as? String ?: "default"
          val useGpu = args?.get("useGpu") as? Boolean ?: true
          val classifierOptionsMap = args?.get("classifierOptions") as? Map<String, Any>
          
          // Resolve the model path (handling absolute paths, internal:// scheme, or asset paths)
          modelPath = resolveModelPath(modelPath)
          
          // Convert task string to enum
          val task = YOLOTask.valueOf(taskString.uppercase())
          
          // Use classifier options map directly (follows existing pattern)
          val classifierOptions = classifierOptionsMap
          
          // Log classifier options for debugging
          if (classifierOptions != null) {
            Log.d(TAG, "Parsed classifier options: $classifierOptions")
          }
          
          // Load labels (in real implementation, you would load from metadata)
          val labels = loadLabels(modelPath)
          
          // Initialize YOLO with instance manager
          YOLOInstanceManager.shared.loadModel(
            instanceId = instanceId,
            context = applicationContext,
            modelPath = modelPath,
            task = task,
            useGpu = useGpu,
            classifierOptions = classifierOptions
          ) { loadResult ->
            if (loadResult.isSuccess) {
              Log.d(TAG, "Model loaded successfully: $modelPath for task: $task, instance: $instanceId, useGpu: $useGpu ${if (classifierOptions != null) "with classifier options" else ""}")
              result.success(true)
            } else {
              Log.e(TAG, "Failed to load model for instance $instanceId", loadResult.exceptionOrNull())
              result.error("MODEL_NOT_FOUND", loadResult.exceptionOrNull()?.message ?: "Failed to load model", null)
            }
          }
        } catch (e: Exception) {
          Log.e(TAG, "Failed to load model", e)
          result.error("model_error", "Failed to load model: ${e.message}", null)
        }
      }

      "predictSingleImage" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val imageData = args?.get("image") as? ByteArray
          val confidenceThreshold = args?.get("confidenceThreshold") as? Double
          val iouThreshold = args?.get("iouThreshold") as? Double
          val instanceId = args?.get("instanceId") as? String ?: "default"

          if (imageData == null) {
            result.error("bad_args", "No image data", null)
            return
          }
          
          // Convert byte array to bitmap
          val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
          if (bitmap == null) {
            result.error("image_error", "Failed to decode image", null)
            return
          }
          
          // Run inference using instance manager
          val yoloResult = YOLOInstanceManager.shared.predict(
            instanceId = instanceId,
            bitmap = bitmap,
            confidenceThreshold = confidenceThreshold?.toFloat(),
            iouThreshold = iouThreshold?.toFloat()
          )
          
          if (yoloResult == null) {
            result.error("MODEL_NOT_LOADED", "Model has not been loaded. Call loadModel() first.", null)
            return
          }
          
          // Create response
          val response = HashMap<String, Any>()
          
          // Get image dimensions for normalization
          val imageWidth = bitmap.width.toFloat()
          val imageHeight = bitmap.height.toFloat()
          
          // Convert boxes to map for Flutter
          response["boxes"] = yoloResult.boxes.map { box ->
            mapOf(
              "x1" to box.xywh.left,
              "y1" to box.xywh.top,
              "x2" to box.xywh.right,
              "y2" to box.xywh.bottom,
              "x1_norm" to box.xywh.left / imageWidth,
              "y1_norm" to box.xywh.top / imageHeight,
              "x2_norm" to box.xywh.right / imageWidth,
              "y2_norm" to box.xywh.bottom / imageHeight,
              "class" to box.cls,
              "className" to box.cls, // Add className for compatibility with YOLOResult
              "confidence" to box.conf
            )
          }
          
          // Include image size in response
          response["imageSize"] = mapOf(
            "width" to imageWidth.toInt(),
            "height" to imageHeight.toInt()
          )
          
          // Get instance to check task type
          val yolo = YOLOInstanceManager.shared.getInstance(instanceId)
          
          // Add task-specific data to response
          when (yolo?.task) {
            YOLOTask.SEGMENT -> {
              // Include raw segmentation masks if available
              yoloResult.masks?.let { masks ->
                // Send raw mask data for each detected instance
                val rawMasks = mutableListOf<List<List<Double>>>()
                for (instanceMask in masks.masks) {
                  val mask2D = mutableListOf<List<Double>>()
                  for (row in instanceMask) {
                    mask2D.add(row.map { it.toDouble() })
                  }
                  rawMasks.add(mask2D)
                }
                response["masks"] = rawMasks
                
                // Also send PNG for backward compatibility (optional)
                masks.combinedMask?.let { combinedMask ->
                  val stream = ByteArrayOutputStream()
                  combinedMask.compress(Bitmap.CompressFormat.PNG, 90, stream)
                  response["maskPng"] = stream.toByteArray()
                }
              }
            }
            YOLOTask.CLASSIFY -> {
              Log.d(TAG, "Processing CLASSIFY task result")
              // Include classification results if available
              yoloResult.probs?.let { probs ->
                Log.d(TAG, "Found probs: top1=${probs.top1}, top1Conf=${probs.top1Conf}, top1Index=${probs.top1Index}")
                
                // Use the original labels from the model (no hardcoded mapping)
                val topClass = probs.top1
                val top5Classes = probs.top5
                
                response["classification"] = mapOf(
                  "topClass" to topClass,
                  "topConfidence" to probs.top1Conf.toDouble(),
                  "top5Classes" to top5Classes,
                  "top5Confidences" to probs.top5Confs.map { it.toDouble() },
                  "top1Index" to probs.top1Index
                )
                
                // Also add classification data to the boxes array for compatibility
                response["boxes"] = listOf(
                  mapOf(
                    "class" to topClass,
                    "className" to topClass,
                    "confidence" to probs.top1Conf.toDouble(),
                    "classIndex" to probs.top1Index,
                    "x1" to 0.0,
                    "y1" to 0.0,
                    "x2" to imageWidth.toDouble(),
                    "y2" to imageHeight.toDouble(),
                    "x1_norm" to 0.0,
                    "y1_norm" to 0.0,
                    "x2_norm" to 1.0,
                    "y2_norm" to 1.0
                  )
                )
                Log.d(TAG, "Added classification data to response")
              } ?: run {
                Log.w(TAG, "YOLOResult.probs is null for CLASSIFY task")
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
          
          val checkResult = YOLOUtils.checkModelExistence(applicationContext, modelPath)
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
          val useGpu = args?.get("useGpu") as? Boolean ?: true
          
          if (viewId == null || modelPath == null || taskString == null) {
            result.error("bad_args", "Missing required arguments for setModel", null)
            return
          }
          
          // Get the YOLOPlatformView instance from the factory
          val platformView = viewFactory.activeViews[viewId]
          if (platformView != null) {
            // Resolve the model path
            val resolvedPath = resolveModelPath(modelPath)
            
            // Convert task string to enum
            val task = YOLOTask.valueOf(taskString.uppercase())
            
            // Call setModel on the YOLOView inside the platform view
            platformView.yoloViewInstance.setModel(resolvedPath, task, useGpu) { success ->
              if (success) {
                result.success(null)
              } else {
                result.error("MODEL_NOT_FOUND", "Failed to load model: $modelPath", null)
              }
            }
          } else {
            result.error("VIEW_NOT_FOUND", "YOLOPlatformView with id $viewId not found", null)
          }
        } catch (e: Exception) {
          Log.e(TAG, "Error setting model", e)
          result.error("set_model_error", "Error setting model: ${e.message}", null)
        }
      }
      
      "disposeInstance" -> {
        try {
          val args = call.arguments as? Map<*, *>
          val instanceId = args?.get("instanceId") as? String
          
          if (instanceId == null) {
            result.error("bad_args", "Missing instanceId", null)
            return
          }
          
          // Remove instance from manager
          YOLOInstanceManager.shared.removeInstance(instanceId)
          
          // Remove the channel for this instance
          instanceChannels[instanceId]?.setMethodCallHandler(null)
          instanceChannels.remove(instanceId)
          
          result.success(null)
        } catch (e: Exception) {
          Log.e(TAG, "Error disposing instance", e)
          result.error("dispose_error", "Failed to dispose instance: ${e.message}", null)
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
            // Log that we're processing permission results
            Log.d(TAG, "Processing permission result for YOLOPlatformView")
            handled = true
            // Assuming only one view actively requests permissions at a time.
            // If multiple views could request, 'handled' logic might need adjustment
            // or ensure only the correct view processes it.
        } catch (e: Exception) {
            Log.e(TAG, "Error processing permission result for YOLOPlatformView instance", e)
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