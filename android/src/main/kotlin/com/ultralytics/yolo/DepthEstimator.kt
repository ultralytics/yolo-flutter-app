// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap

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
        loadModelLabels(context, modelPath)?.let { labels = it }
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
        val range = colorizeDepth(output, colorPixels, depthColors) ?: error("Depth output contains no valid values")

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
        init {
            System.loadLibrary("ultralytics")
        }
    }
}
