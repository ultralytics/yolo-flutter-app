package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import org.autojs.plugin.sdk.Plugin

class YOLOPlugin(context: Context, selfContext: Context, runtime: Any, topLevelScope: Any) : Plugin(context, selfContext, runtime, topLevelScope) {

    override fun getAssetsScriptDir(): String {
        return "plugin-yolo"
    }

    private var yolo: YOLO? = null

    fun loadModel(modelPath: String, useGpu: Boolean = true) {
        // Resolve model path via YOLOUtils (supports assets/fs)
        // Since YOLOUtils.loadModelFile handles checking, we pass path directly to YOLO
        // However, YOLO constructor expects path. YOLO internally calls YOLOUtils.
        
        // We use "DETECT" task hardcoded as requested
        yolo = YOLO(
            context = getSelfContext(), // Use plugin context for assets
            modelPath = modelPath,
            task = YOLOTask.DETECT,
            useGpu = useGpu
        )
        // Since we removed 'lazy', we might want to trigger init or check if it works.
        // YOLO uses lazy 'predictor', so it won't fail until predict is called.
        // But we can call predictorInstance() to force init and catch errors.
        try {
            yolo?.predictorInstance()
        } catch (e: Exception) {
            yolo = null
            throw RuntimeException("Failed to load model: ${e.message}", e)
        }
    }

    fun detect(bitmap: Bitmap): YOLOResult? {
        return yolo?.predict(bitmap)
    }

    // Helper to decode Base64 to Bitmap (useful if calling from JS with base64)
    fun detectBase64(base64: String): YOLOResult? {
        val decodedBytes = Base64.decode(base64, Base64.DEFAULT)
        val bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size)
        return detect(bitmap)
    }
    
    fun setConfidenceThreshold(conf: Float) {
        yolo?.setConfidenceThreshold(conf)
    }
    
    fun setIouThreshold(iou: Float) {
        yolo?.setIouThreshold(iou)
    }
}