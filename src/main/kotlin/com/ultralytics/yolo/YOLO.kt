package com.ultralytics.yolo

import android.graphics.Bitmap
import android.util.Log

/**
 * YOLO 统一接口封装类
 * 目前仅支持 物体检测 (Object Detection)
 * 
 * @param modelPath 模型文件绝对路径
 * @param task 任务类型 (仅支持 DETECT)
 * @param useGpu 是否使用 GPU (默认 true)
 */
class YOLO(
    private val modelPath: String,
    val task: YOLOTask = YOLOTask.DETECT,
    private val useGpu: Boolean = true
) {
    private val TAG = "YOLO"
    
    // 实际的检测器实例 (Lazy 加载)
    private val detector: ObjectDetector by lazy {
        Log.d(TAG, "初始化 YOLO 检测器, 路径: $modelPath, GPU: $useGpu")
        ObjectDetector(modelPath, useGpu)
    }

    /**
     * 预测/检测
     * @param bitmap 输入图片
     * @return 结果
     */
    fun predict(bitmap: Bitmap): YOLOResult {
        return detector.detect(bitmap)
    }

    /**
     * 设置置信度阈值
     * @param conf 阈值 (0.0 - 1.0)
     */
    fun setConfidenceThreshold(conf: Float) {
        detector.setConfidence(conf)
    }

    /**
     * 设置 IoU 阈值
     * @param iou 阈值 (0.0 - 1.0)
     */
    fun setIouThreshold(iou: Float) {
        detector.setIou(iou)
    }
}