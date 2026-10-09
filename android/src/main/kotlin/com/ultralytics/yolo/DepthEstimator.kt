// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
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
    private var colorPixels = IntArray(0)

    init {
        YOLOFileUtils.loadModelLabels(context, modelPath)?.let { labels = it }
        val model = InferenceModel.create(context, modelPath, useGpu, "DepthEstimator")
        try {
            val inDims = model.inputDims
            val inHeight = if (inDims.size >= 4) inDims[1] else 640
            val inWidth = if (inDims.size >= 4) inDims[2] else 640
            inputSize = Size(inWidth, inHeight)
            modelInputSize = Pair(inWidth, inHeight)

            outputShape = model.outputDims.getOrNull(0) ?: IntArray(0)
            val channelFirst = outputShape.size >= 2 && outputShape.dropLast(2).all { it == 1 }
            val channelLast = outputShape.size == 4 && outputShape[0] == 1 && outputShape[3] == 1
            require(channelFirst || channelLast) {
                "Depth output tensor shape not supported: ${outputShape.joinToString()}"
            }
            depthHeight = if (channelLast) outputShape[1] else outputShape[outputShape.size - 2]
            depthWidth = if (channelLast) outputShape[2] else outputShape.last()
            require(depthWidth > 0 && depthHeight > 0 && depthWidth * depthHeight == model.outputElementCounts[0]) {
                "Invalid depth output tensor shape: ${outputShape.joinToString()}"
            }

            floatInput = FloatArray(inWidth * inHeight * 3)
            inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
            intValues = IntArray(inWidth * inHeight)
        } catch (t: Throwable) {
            model.close()
            throw t
        }
        rtModel = model
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
            stretch = true, // match Ultralytics depth validation and calibration
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
        val depthMap = postProcessDepth(output)
        val timing = finishTiming(preEnd, inferEnd)
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            depthMap = depthMap,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
            accelerator = rtModel.accelerator,
            names = labels,
        )
    }

    internal fun postProcessDepth(output: FloatArray): DepthMap {
        require(output.size == depthWidth * depthHeight) {
            "Depth output size ${output.size} does not match ${depthWidth * depthHeight}"
        }
        val width = depthWidth
        val height = depthHeight
        val values = if (includeRawMaskData) output.copyOf() else null
        val size = width * height
        if (colorPixels.size != size) colorPixels = IntArray(size)
        val range = colorizeDepth(
            output, colorPixels, colors
        )
            ?: error("Depth output contains no valid values")

        return DepthMap(
            values = values,
            width = width,
            height = height,
            minDepth = range[0],
            maxDepth = range[1],
            image = Bitmap.createBitmap(colorPixels, width, height, Bitmap.Config.ARGB_8888),
        )
    }

    private external fun colorizeDepth(
        output: FloatArray,
        colorPixels: IntArray,
        colors: IntArray,
    ): FloatArray?

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

        init {
            System.loadLibrary("ultralytics")
        }
    }
}
