package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * 尺寸数据类
 * @property width 宽度
 * @property height 高度
 */
data class Size(
    val width: Int,
    val height: Int
)

/**
 * 检测框数据类
 * @property index 类别索引
 * @property cls 类别名称
 * @property conf 置信度 (0.0 - 1.0)
 * @property xywh 绝对坐标 (x, y, w, h)
 * @property xywhn 归一化坐标 (x, y, w, h)
 */
data class Box(
    val index: Int,
    val cls: String,
    val conf: Float,
    val xywh: RectF,
    val xywhn: RectF
)

/**
 * YOLO 推理结果类
 * @property origShape 原始图片尺寸
 * @property boxes 检测到的目标框列表
 * @property speed 推理耗时 (ms)
 * @property fps 估算帧率
 * @property names 类别名称列表
 * @property originalImage 原始图片 (可选)
 * @property annotatedImage 标注图片 (可选)
 */
data class YOLOResult(
    val origShape: Size,
    val boxes: List<Box> = emptyList(),
    val speed: Double = 0.0,
    val fps: Double = 0.0,
    val names: List<String> = emptyList(),
    val originalImage: Bitmap? = null,
    val annotatedImage: Bitmap? = null
)
