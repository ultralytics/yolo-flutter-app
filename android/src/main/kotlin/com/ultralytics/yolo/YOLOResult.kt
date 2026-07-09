// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF

data class YOLOResult(
    val origShape: Size,
    val boxes: List<Box> = emptyList(),
    val masks: Masks? = null,
    val semanticMask: SemanticMask? = null,
    val depthMap: DepthMap? = null,
    val probs: Probs? = null,
    val keypointsList: List<Keypoints> = emptyList(),
    val obb: List<OBBResult> = emptyList(),
    val annotatedImage: Bitmap? = null,
    val speed: Double,
    val fps: Double? = null,
    val preMs: Double = 0.0,
    val inferenceMs: Double = 0.0,
    val postMs: Double = 0.0,
    val originalImage: Bitmap? = null,
    val names: List<String>
)

data class Box(
    var index: Int,
    var cls: String,
    var conf: Float,
    val xywh: RectF,    // Real image coordinates
    val xywhn: RectF    // Normalized coordinates (0~1)
)

data class Masks(
    val masks: List<List<List<Float>>>, // Individual probability maps (matrix list)
    val combinedMask: Bitmap?           // Combined mask image
)

data class SemanticMask(
    val classMap: IntArray,
    val width: Int,
    val height: Int,
    val maskImage: Bitmap?
)

data class DepthMap(
    val values: FloatArray?,
    val width: Int,
    val height: Int,
    val minDepth: Float,
    val maxDepth: Float,
    val image: Bitmap?,
)

data class Probs(
    var top1Label: String,
    var top5Labels: List<String>,
    var top1Conf: Float,
    var top5Confs: List<Float>,
    var top1Index: Int,
    var top5Indices: List<Int>
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
