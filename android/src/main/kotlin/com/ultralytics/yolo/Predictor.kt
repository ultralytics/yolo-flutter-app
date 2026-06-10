// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Runtime-agnostic inference model behind the predictors: NHWC interleaved-RGB float32 in, flat float32 outputs out.
 * Implemented by [LiteRtModel] (TFLite on LiteRT, GPU→CPU ladder) and [OrtQnnModel] (QNN context-binary ONNX on the
 * Snapdragon NPU).
 */
interface InferenceModel {
    /** Accelerator in use: "NPU", "GPU" or "CPU". */
    val accelerator: String

    /** Input tensor dimensions in NHWC convention, e.g. [1, 640, 640, 3]. */
    val inputDims: IntArray

    /** Float element count of each output buffer, in order. */
    val outputElementCounts: IntArray

    /** Output tensor dimensions, in order (e.g. [[1, 84, 8400]] for detect). */
    val outputDims: List<IntArray>

    /** Run inference on NHWC interleaved-RGB floats, returning each output as a flat float array. */
    fun run(input: FloatArray): List<FloatArray>

    fun close()

    companion object {
        /**
         * Create the model wrapper for [modelPath]: a QNN context-binary ONNX (`*.onnx`) runs on the Snapdragon NPU
         * via ONNX Runtime; everything else is TFLite on LiteRT. QNN models have no CPU fallback (the context binary
         * is precompiled Hexagon code), so failures here should be handled by falling back to a TFLite model.
         */
        fun create(context: Context, modelPath: String, useGpu: Boolean, tag: String): InferenceModel =
            if (modelPath.lowercase().endsWith(".onnx")) {
                try {
                    OrtQnnModel(context, modelPath, tag)
                } catch (e: NoClassDefFoundError) {
                    throw IllegalStateException(
                        "QNN (.onnx) models require the optional 'com.microsoft.onnxruntime:onnxruntime-android-qnn' " +
                            "dependency in your app's build.gradle",
                        e,
                    )
                }
            } else {
                LiteRtModel(modelPath, useGpu, tag)
            }
    }
}

interface Predictor {
    /**
     * Predict method for processing a bitmap
     * @param bitmap Input bitmap to process
     * @param origWidth Original width of the source image
     * @param origHeight Original height of the source image
     * @param rotateForCamera Whether this is a camera feed that requires rotation (true) or a single image (false)
     * @param isLandscape Whether the device is in landscape orientation (true) or portrait (false)
     * @return YOLOResult containing detection results
     */
    fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean = false, isLandscape: Boolean = false): YOLOResult
    
    abstract fun setIouThreshold(iou: Double)
    abstract fun setConfidenceThreshold(conf: Double)
    abstract fun setNumItemsThreshold(progress: Int)
    abstract fun getConfidenceThreshold(): Double
    abstract fun getIouThreshold(): Double

    var labels: List<String>
    var isUpdating: Boolean
    var inputSize: Size
}

abstract class BasePredictor : Predictor {
    override var isUpdating: Boolean = false
    override lateinit var labels: List<String>
    protected lateinit var rtModel: InferenceModel
    override lateinit var inputSize: Size
    protected lateinit var modelInputSize: Pair<Int, Int>
    protected fun isInterpreterInitialized() = this::rtModel.isInitialized

    protected var t0: Long = 0L
    protected var t2: Double = 0.0
    protected var t3: Long = System.nanoTime()
    protected var t4: Double = 0.0

    var CONFIDENCE_THRESHOLD:Float = 0.25f
    var IOU_THRESHOLD:Float = 0.7f
    var isFrontCamera: Boolean = false
    var cameraRotationDegrees: Int? = null
    var includeRawMaskData: Boolean = true

    fun close() {
        if (isInterpreterInitialized()) {
            rtModel.close()
        }
    }

    protected data class FrameTiming(
        val speedMs: Double,
        val fps: Double
    )

    protected fun finishTiming(): FrameTiming {
        val now = System.nanoTime()
        val dtMs = (now - t0) / 1_000_000.0
        t2 = 0.05 * dtMs + 0.95 * t2
        t4 = 0.05 * ((now - t3) / 1e9) + 0.95 * t4
        t3 = now
        return FrameTiming(
            speedMs = dtMs,
            fps = if (t4 > 0.0) 1.0 / t4 else 0.0
        )
    }

    protected fun labelName(index: Int): String {
        if (index < 0) return "class $index"
        return labels.getOrNull(index)?.takeIf { it.isNotBlank() } ?: "class $index"
    }

    override fun setIouThreshold(iou: Double) {
        IOU_THRESHOLD = iou.toFloat()
    }

    override fun setConfidenceThreshold(conf: Double) {
        CONFIDENCE_THRESHOLD = conf.toFloat()
    }

    override fun setNumItemsThreshold(progress: Int) {

    }
    
    override fun getConfidenceThreshold(): Double {
        return CONFIDENCE_THRESHOLD.toDouble()
    }
    
    override fun getIouThreshold(): Double {
        return IOU_THRESHOLD.toDouble()
    }

    protected fun inputRectFromOutputRect(
        outputRect: RectF,
        origWidth: Int,
        origHeight: Int,
        outputIsNormalized: Boolean = outputCoordinatesAreNormalized(outputRect)
    ): RectF? {
        val modelRect = modelRectFromOutputRect(outputRect, outputIsNormalized)
        return inputRectFromModelRect(modelRect, origWidth, origHeight)
    }

    // YOLO exports here use either normalized 0..1 or model-pixel coordinates.
    protected fun outputCoordinatesAreNormalized(rect: RectF): Boolean {
        val maxCoordinate = max(
            max(abs(rect.left), abs(rect.top)),
            max(abs(rect.right), abs(rect.bottom))
        )
        return maxCoordinate <= 2f
    }

    protected fun normalizedRectFromInputRect(rect: RectF, origWidth: Int, origHeight: Int): RectF {
        return RectF(
            rect.left / origWidth,
            rect.top / origHeight,
            rect.right / origWidth,
            rect.bottom / origHeight
        )
    }

    protected fun inputPointFromOutputPoint(
        x: Float,
        y: Float,
        origWidth: Int,
        origHeight: Int,
        outputIsNormalized: Boolean = outputCoordinatesAreNormalized(x, y)
    ): Pair<Float, Float> {
        val modelX = modelCoordinate(x, modelInputSize.first, outputIsNormalized)
        val modelY = modelCoordinate(y, modelInputSize.second, outputIsNormalized)
        return inputPointFromModelPoint(modelX, modelY, origWidth, origHeight, clamp = true)
    }

    protected fun inputOBBFromModelOBB(obb: OBB, origWidth: Int, origHeight: Int): OBB {
        val modelWidth = modelInputSize.first
        val modelHeight = modelInputSize.second
        val outputIsNormalized = outputCoordinatesAreNormalized(obb)
        val modelOBB = OBB(
            cx = modelCoordinate(obb.cx, modelWidth, outputIsNormalized),
            cy = modelCoordinate(obb.cy, modelHeight, outputIsNormalized),
            w = modelCoordinate(obb.w, modelWidth, outputIsNormalized),
            h = modelCoordinate(obb.h, modelHeight, outputIsNormalized),
            angle = obb.angle
        )
        val points = modelOBB.toPolygon().map { point ->
            inputPointFromModelPoint(point.x, point.y, origWidth, origHeight, clamp = false)
        }
        val p0 = points[0]
        val p1 = points[1]
        val p2 = points[2]
        val centerX = points.sumOf { it.first.toDouble() }.toFloat() / points.size
        val centerY = points.sumOf { it.second.toDouble() }.toFloat() / points.size
        val width = hypot(p1.first - p0.first, p1.second - p0.second)
        val height = hypot(p2.first - p1.first, p2.second - p1.second)

        return OBB(
            cx = centerX / origWidth,
            cy = centerY / origHeight,
            w = width / origWidth,
            h = height / origHeight,
            angle = atan2(p1.second - p0.second, p1.first - p0.first)
        )
    }

    private fun letterboxTransform(origWidth: Int, origHeight: Int): ImageUtils.LetterboxTransform? =
        ImageUtils.letterboxTransform(origWidth, origHeight, modelInputSize.first, modelInputSize.second)

    protected fun modelMaskCropRect(maskWidth: Int, maskHeight: Int, origWidth: Int, origHeight: Int): Rect? {
        val transform = letterboxTransform(origWidth, origHeight) ?: return null
        val modelWidth = modelInputSize.first.toFloat()
        val modelHeight = modelInputSize.second.toFloat()
        val left = (transform.padX / modelWidth * maskWidth).roundToInt()
        val top = (transform.padY / modelHeight * maskHeight).roundToInt()
        val right = maskWidth - (transform.padRight / modelWidth * maskWidth).roundToInt()
        val bottom = maskHeight - (transform.padBottom / modelHeight * maskHeight).roundToInt()
        val crop = Rect(
            left.coerceIn(0, maskWidth),
            top.coerceIn(0, maskHeight),
            right.coerceIn(0, maskWidth),
            bottom.coerceIn(0, maskHeight)
        )
        if (crop.left >= crop.right || crop.top >= crop.bottom) return null
        if (crop.left == 0 && crop.top == 0 && crop.right == maskWidth && crop.bottom == maskHeight) return null
        return crop
    }

    private fun inputRectFromModelRect(modelRect: RectF, origWidth: Int, origHeight: Int): RectF? {
        // Do NOT clamp to the image bounds: a partially off-frame object has a box that legitimately extends past the
        // edge, and clamping each side to [0, size] distorts it (e.g. a left edge pinned to 0 shifts the box right /
        // stretches its width). Keep the true coordinates and let the overlay clip them, matching the iOS app.
        val leftTop = inputPointFromModelPoint(modelRect.left, modelRect.top, origWidth, origHeight, clamp = false)
        val rightBottom = inputPointFromModelPoint(
            modelRect.right,
            modelRect.bottom,
            origWidth,
            origHeight,
            clamp = false
        )
        val left = leftTop.first
        val top = leftTop.second
        val right = rightBottom.first
        val bottom = rightBottom.second
        val rect = RectF(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        return rect.takeIf { it.width() > 0f && it.height() > 0f }
    }

    private fun inputPointFromModelPoint(
        modelX: Float,
        modelY: Float,
        origWidth: Int,
        origHeight: Int,
        clamp: Boolean
    ): Pair<Float, Float> {
        val transform = letterboxTransform(origWidth, origHeight) ?: return modelX to modelY
        val x = (modelX - transform.padX) / transform.gain
        val y = (modelY - transform.padY) / transform.gain
        if (!clamp) return x to y
        return x.coerceIn(0f, origWidth.toFloat()) to y.coerceIn(0f, origHeight.toFloat())
    }

    protected fun modelRectFromOutputRect(rect: RectF, outputIsNormalized: Boolean): RectF {
        return if (outputIsNormalized) {
            RectF(
                rect.left * modelInputSize.first,
                rect.top * modelInputSize.second,
                rect.right * modelInputSize.first,
                rect.bottom * modelInputSize.second
            )
        } else {
            rect
        }
    }

    private fun outputCoordinatesAreNormalized(x: Float, y: Float): Boolean {
        return max(abs(x), abs(y)) <= 2f
    }

    private fun outputCoordinatesAreNormalized(obb: OBB): Boolean {
        val maxCoordinate = max(
            max(abs(obb.cx), abs(obb.cy)),
            max(abs(obb.w), abs(obb.h))
        )
        return maxCoordinate <= 2f
    }

    private fun modelCoordinate(value: Float, axisSize: Int, outputIsNormalized: Boolean): Float {
        return if (outputIsNormalized) value * axisSize else value
    }
}
