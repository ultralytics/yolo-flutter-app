package com.ultralytics.yolo

/**
 * YOLO 相关常量定义
 */
object YOLOConstants {
    
    // 默认阈值配置
    const val DEFAULT_CONFIDENCE_THRESHOLD = 0.25f
    const val DEFAULT_IOU_THRESHOLD = 0.45f
    
    // NMS 相关常量
    const val NMS_CANDIDATE_MULTIPLIER = 5
    const val DEFAULT_NUM_ITEMS_THRESHOLD = 30
    
    // 模型输出维度常量
    const val COORDINATE_COUNT = 4 // x, y, w, h
    
    // 图像处理常量
    const val BYTES_PER_FLOAT = 4
    const val MAX_NORMALIZATION_VALUE = 255f
    
    // 颜色透明度
    const val ULTRALYTICS_ALPHA = 153
    
    // 日志标签
    const val TAG_UTILS = "YOLOUtils"
    const val TAG_DETECTOR = "ObjectDetector"
    const val TAG_YOLO = "YOLO"
    
    // 错误消息
    const val ERROR_MODEL_NOT_FOUND = "模型文件未找到"
    const val ERROR_MODEL_LOAD_FAILED = "读取模型文件失败"
    const val ERROR_GPU_INIT_FAILED = "启用 GPU 失败，回退到 CPU"
    
    // 成功消息
    const val SUCCESS_GPU_ENABLED = "已启用 GPU 代理"
    const val SUCCESS_MODEL_LOADED = "模型加载成功"
    const val SUCCESS_LABELS_LOADED = "加载标签成功"
}
