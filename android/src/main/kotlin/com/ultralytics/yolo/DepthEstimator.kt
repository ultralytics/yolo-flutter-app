// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.ln
import kotlin.math.roundToInt

/** Monocular metric-depth predictor shared by LiteRT GPU/CPU and ONNX Runtime QNN backends. */
class DepthEstimator(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    useGpu: Boolean = true,
) : BasePredictor() {
    private lateinit var floatInput: FloatArray
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray
    private val outputShape: IntArray
    private val depthHeight: Int
    private val depthWidth: Int
    private val logarithmicColors = IntArray(4096)
    private var colorPixels = IntArray(0)

    init {
        YOLOFileUtils.loadModelLabels(context, modelPath)?.let { labels = it }
        rtModel = InferenceModel.create(context, modelPath, useGpu, "DepthEstimator")

        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        outputShape = rtModel.outputDims.getOrNull(0) ?: IntArray(0)
        val channelFirst = outputShape.size >= 2 && outputShape.dropLast(2).all { it == 1 }
        val channelLast = outputShape.size == 4 && outputShape[0] == 1 && outputShape[3] == 1
        require(channelFirst || channelLast) {
            "Depth output tensor shape not supported: ${outputShape.joinToString()}"
        }
        depthHeight = if (channelLast) outputShape[1] else outputShape[outputShape.size - 2]
        depthWidth = if (channelLast) outputShape[2] else outputShape.last()
        require(depthWidth > 0 && depthHeight > 0 && depthWidth * depthHeight == rtModel.outputElementCounts[0]) {
            "Invalid depth output tensor shape: ${outputShape.joinToString()}"
        }

        floatInput = FloatArray(inWidth * inHeight * 3)
        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
    }

    override fun predict(
        bitmap: Bitmap,
        origWidth: Int,
        origHeight: Int,
        rotateForCamera: Boolean,
        isLandscape: Boolean,
    ): YOLOResult {
        t0 = System.nanoTime()
        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = inputBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees,
        )
        ImageUtils.copyRgbBitmapToFloatArray(
            inputBitmap,
            floatInput,
            intValues,
            channelsFirst = rtModel.inputUsesNchw,
        )

        val preEnd = System.nanoTime()
        val output = rtModel.run(floatInput)[0]
        val inferEnd = System.nanoTime()
        val depthMap = postProcessDepth(output, origWidth, origHeight)
        val timing = finishTiming(preEnd, inferEnd)
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            depthMap = depthMap,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
            names = labels,
        )
    }

    internal fun postProcessDepth(output: FloatArray, origWidth: Int, origHeight: Int): DepthMap {
        require(output.size == depthWidth * depthHeight) {
            "Depth output size ${output.size} does not match ${depthWidth * depthHeight}"
        }
        val crop = modelMaskCropRect(depthWidth, depthHeight, origWidth, origHeight)
        val left = crop?.left ?: 0
        val top = crop?.top ?: 0
        val right = crop?.right ?: depthWidth
        val bottom = crop?.bottom ?: depthHeight
        val width = right - left
        val height = bottom - top
        require(width > 0 && height > 0) { "Invalid depth crop ${width}x$height" }

        val values = if (includeRawMaskData) FloatArray(width * height) else null
        var minDepth = Float.POSITIVE_INFINITY
        var maxDepth = Float.NEGATIVE_INFINITY
        for (y in 0 until height) {
            var source = (y + top) * depthWidth + left
            var target = y * width
            for (x in 0 until width) {
                val value = output[source++]
                values?.set(target, value)
                target++
                if (value.isFinite() && value > 0f) {
                    if (value < minDepth) minDepth = value
                    if (value > maxDepth) maxDepth = value
                }
            }
        }
        require(minDepth.isFinite() && maxDepth.isFinite()) { "Depth output contains no valid values" }

        return DepthMap(
            values = values,
            width = width,
            height = height,
            minDepth = minDepth,
            maxDepth = maxDepth,
            image = colorizeDepth(output, left, top, width, height, minDepth, maxDepth),
        )
    }

    private fun colorizeDepth(
        output: FloatArray,
        left: Int,
        top: Int,
        width: Int,
        height: Int,
        minDepth: Float,
        maxDepth: Float,
    ): Bitmap {
        val size = width * height
        if (colorPixels.size != size) colorPixels = IntArray(size)
        val linearRange = maxDepth - minDepth
        if (linearRange <= 0f) {
            java.util.Arrays.fill(colorPixels, colors[0])
        } else {
            val logMax = ln(maxDepth)
            val logRange = (logMax - ln(minDepth)).coerceAtLeast(1e-6f)
            for (bin in logarithmicColors.indices) {
                val depth = minDepth + linearRange * bin / logarithmicColors.lastIndex
                val colorIndex = ((logMax - ln(depth)) / logRange * colors.lastIndex)
                    .roundToInt()
                    .coerceIn(colors.indices)
                logarithmicColors[bin] = colors[colorIndex]
            }
            val binScale = logarithmicColors.lastIndex / linearRange
            for (y in 0 until height) {
                var source = (y + top) * depthWidth + left
                var target = y * width
                for (x in 0 until width) {
                    val value = output[source++]
                    colorPixels[target++] = if (value.isFinite() && value > 0f) {
                        val bin = ((value - minDepth) * binScale)
                            .roundToInt()
                            .coerceIn(logarithmicColors.indices)
                        logarithmicColors[bin]
                    } else {
                        Color.TRANSPARENT
                    }
                }
            }
        }
        return Bitmap.createBitmap(colorPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    companion object {
        private val colors = IntArray(256).also { table ->
            val stops = arrayOf(
                intArrayOf(48, 18, 59),
                intArrayOf(50, 100, 200),
                intArrayOf(40, 190, 140),
                intArrayOf(245, 210, 60),
                intArrayOf(180, 20, 40),
            )
            for (index in table.indices) {
                val position = index.toFloat() / table.lastIndex * (stops.size - 1)
                val lower = position.toInt().coerceAtMost(stops.size - 2)
                val fraction = position - lower
                val first = stops[lower]
                val second = stops[lower + 1]
                table[index] = Color.rgb(
                    (first[0] + (second[0] - first[0]) * fraction).roundToInt(),
                    (first[1] + (second[1] - first[1]) * fraction).roundToInt(),
                    (first[2] + (second[2] - first[2]) * fraction).roundToInt(),
                )
            }
        }
    }
}
