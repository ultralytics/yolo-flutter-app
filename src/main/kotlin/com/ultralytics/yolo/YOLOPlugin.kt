package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import org.autojs.plugin.sdk.Plugin

/**
 * Auto.js YOLO 插件入口类
 * 负责桥接 JavaScript 调用与 Native 逻辑
 */
class YOLOPlugin(
    context: Context, 
    selfContext: Context, 
    runtime: Any, 
    topLevelScope: Any
) : Plugin(context, selfContext, runtime, topLevelScope) {

    private var yolo: YOLO? = null

    /**
     * 返回插件 Assets 下的脚本目录名称
     */
    override fun getAssetsScriptDir(): String {
        return "plugin-yolo"
    }

    /**
     * 加载模型 (仅支持绝对路径)
     * @param modelPath 模型的本地绝对路径
     * @param useGpu 是否使用 GPU
     * @throws RuntimeException 如果路径无效
     */
    fun loadModel(modelPath: String, useGpu: Boolean = true) {
        // 不再传递 getSelfContext()，因为现在只支持绝对路径加载
        yolo = YOLO(modelPath, YOLOTask.DETECT, useGpu)
    }

    /**
     * 执行检测
     * @param bitmap Auto.js 传入的 Bitmap 对象
     * @return 检测结果
     */
    fun detect(bitmap: Bitmap): YOLOResult? {
        return yolo?.predict(bitmap)
    }

    /**
     * Base64 图像检测
     * @param base64 string
     */
    fun detectBase64(base64: String): YOLOResult? {
        val decodedBytes = Base64.decode(base64, Base64.DEFAULT)
        val bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size)
        return try {
            detect(bitmap)
        } finally {
            // 确保 Bitmap 被回收，避免内存泄漏
            bitmap?.recycle()
        }
    }

    fun setConfidenceThreshold(conf: Float) {
        yolo?.setConfidenceThreshold(conf)
    }

    fun setIouThreshold(iou: Float) {
        yolo?.setIouThreshold(iou)
    }
}