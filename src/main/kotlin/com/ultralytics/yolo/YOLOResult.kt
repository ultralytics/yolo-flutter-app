package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.RectF

data class Size(
    val width: Int,
    val height: Int
)

data class Box(
    val index: Int,
    val cls: String,
    val conf: Float,
    val xywh: RectF,
    val xywhn: RectF
)

data class YOLOResult(
    val origShape: Size,
    val boxes: List<Box> = emptyList(),
    val speed: Double = 0.0,
    val fps: Double = 0.0,
    val names: List<String> = emptyList(),
    val originalImage: Bitmap? = null,
    val annotatedImage: Bitmap? = null
)
