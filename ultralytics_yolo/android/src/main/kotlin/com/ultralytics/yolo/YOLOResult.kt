package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF

data class YOLOResult(
    val origShape: Size,
    val boxes: List<Box> = emptyList(),
    val masks: Masks? = null,
    val probs: Probs? = null,
    val keypointsList: List<Keypoints> = emptyList(),
    val obb: List<OBBResult> = emptyList(),
    val annotatedImage: Bitmap? = null,
    val speed: Double,
    val fps: Double? = null,
    val originalImage: Bitmap? = null,
    val names: List<String>
)

data class Box(
    val index: Int,
    val cls: String,
    val conf: Float,
    val xywh: RectF,    // 実画像座標
    val xywhn: RectF    // 正規化座標（0～1）
)

data class Masks(
    val masks: List<List<List<Float>>>, // 個々の確率マップ（行列リスト）
    val combinedMask: Bitmap?           // 合成済みマスク画像
)

data class Probs(
    var top1: String,
    var top5: List<String>,
    var top1Conf: Float,
    var top5Confs: List<Float>,
    var top1Index: Int
)

data class Keypoints(
    val xyn: List<Pair<Float, Float>>,
    val xy: List<Pair<Float, Float>>,
    val conf: List<Float>
)

data class OBBResult(
    val box: OBB,
    val confidence: Float,
    val cls: String,
    val index: Int
)

data class Size(val width: Int, val height: Int)
