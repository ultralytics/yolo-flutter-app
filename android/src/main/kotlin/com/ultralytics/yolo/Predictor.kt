// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

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
    protected lateinit var rtModel: LiteRtModel
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

    fun close() {
        if (isInterpreterInitialized()) {
            rtModel.close()
        }
    }

    protected fun updateTiming() {
        val now = System.nanoTime()
        val dt = (now - t0) / 1e9
        t2 = 0.05 * dt + 0.95 * t2
        t4 = 0.05 * ((now - t3) / 1e9) + 0.95 * t4
        t3 = now
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
        val transform = letterboxTransform(origWidth, origHeight) ?: return x to y
        return (
            ((modelX - transform.padX) / transform.gain).coerceIn(0f, origWidth.toFloat())
        ) to (
            ((modelY - transform.padY) / transform.gain).coerceIn(0f, origHeight.toFloat())
        )
    }

    protected fun inputOBBFromModelOBB(obb: OBB, origWidth: Int, origHeight: Int): OBB {
        val transform = letterboxTransform(origWidth, origHeight) ?: return obb
        val modelWidth = modelInputSize.first
        val modelHeight = modelInputSize.second
        val outputIsNormalized = outputCoordinatesAreNormalized(obb)
        val cx = modelCoordinate(obb.cx, modelWidth, outputIsNormalized)
        val cy = modelCoordinate(obb.cy, modelHeight, outputIsNormalized)
        val w = modelCoordinate(obb.w, modelWidth, outputIsNormalized)
        val h = modelCoordinate(obb.h, modelHeight, outputIsNormalized)

        return OBB(
            cx = ((cx - transform.padX) / transform.gain) / origWidth,
            cy = ((cy - transform.padY) / transform.gain) / origHeight,
            w = (w / transform.gain) / origWidth,
            h = (h / transform.gain) / origHeight,
            angle = obb.angle
        )
    }

    private data class LetterboxTransform(
        val gain: Float,
        val padX: Float,
        val padY: Float,
        val padRight: Float,
        val padBottom: Float
    )

    private fun letterboxTransform(origWidth: Int, origHeight: Int): LetterboxTransform? {
        val modelWidth = modelInputSize.first.toFloat()
        val modelHeight = modelInputSize.second.toFloat()
        if (modelWidth <= 0f || modelHeight <= 0f || origWidth <= 0 || origHeight <= 0) return null

        val gain = min(modelWidth / origWidth, modelHeight / origHeight)
        if (gain <= 0f) return null
        val resizedWidth = (origWidth * gain).roundToInt()
        val resizedHeight = (origHeight * gain).roundToInt()
        val padWidth = modelWidth - resizedWidth
        val padHeight = modelHeight - resizedHeight
        // Match Ultralytics LetterBox leading-pad rounding: round(d - 0.1).
        val padX = (padWidth / 2f - 0.1f).roundToInt().toFloat()
        val padY = (padHeight / 2f - 0.1f).roundToInt().toFloat()
        val padRight = (padWidth / 2f + 0.1f).roundToInt().toFloat()
        val padBottom = (padHeight / 2f + 0.1f).roundToInt().toFloat()
        return LetterboxTransform(gain, padX, padY, padRight, padBottom)
    }

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
        val transform = letterboxTransform(origWidth, origHeight) ?: return modelRect
        // Do NOT clamp to the image bounds: a partially off-frame object has a box that legitimately extends past the
        // edge, and clamping each side to [0, size] distorts it (e.g. a left edge pinned to 0 shifts the box right /
        // stretches its width). Keep the true coordinates and let the overlay clip them, matching the iOS app.
        val left = (modelRect.left - transform.padX) / transform.gain
        val top = (modelRect.top - transform.padY) / transform.gain
        val right = (modelRect.right - transform.padX) / transform.gain
        val bottom = (modelRect.bottom - transform.padY) / transform.gain
        val rect = RectF(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        return rect.takeIf { it.width() > 0f && it.height() > 0f }
    }

    private fun modelRectFromOutputRect(rect: RectF, outputIsNormalized: Boolean): RectF {
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
