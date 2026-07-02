// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Classifier(
    context: Context,
    modelPath: String,
    override var labels: List<String> = emptyList(),
    private val useGpu: Boolean = true,
    private val classifierOptions: Map<String, Any>? = null
) : BasePredictor() {

    var numClass: Int = 0
    private var modelInputChannels: Int = 3  // Default to 3-channel, will be detected
    private var isGrayscaleModel: Boolean = false

    private lateinit var floatInput: FloatArray
    private lateinit var grayBuffer: ByteBuffer // only allocated for 1-channel grayscale models
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray

    init {
        val loadedLabels = YOLOFileUtils.loadModelLabels(context, modelPath)
        if (loadedLabels != null) {
            this.labels = loadedLabels
            Log.i(TAG, "Labels successfully loaded from appended ZIP.")
        } else {
            val optionsLabels = classifierOptions?.get("labels") as? List<*>
            if (optionsLabels != null) {
                this.labels = optionsLabels.map { it.toString() }
            } else if (this.labels.isEmpty()) {
                Log.w(TAG, "No embedded labels found and none provided; predictions may lack class names.")
            }
        }

        rtModel = InferenceModel.create(context, modelPath, useGpu, "Classifier")

        // Input dims [1, H, W, C].
        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 224
        val inWidth = if (inDims.size >= 4) inDims[2] else 224
        val inChannels = if (inDims.size >= 4) inDims[3] else 3
        modelInputChannels = inChannels
        isGrayscaleModel = inChannels == 1

        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // Output dims [1, numClass]; fall back to the element count.
        val outDims = rtModel.outputDims.getOrNull(0) ?: IntArray(0)
        numClass = if (outDims.size >= 2) outDims[1] else rtModel.outputElementCounts.getOrElse(0) { labels.size }

        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
        floatInput = FloatArray(inWidth * inHeight * modelInputChannels)
        if (isGrayscaleModel) {
            grayBuffer = ByteBuffer.allocateDirect(inWidth * inHeight * 4).apply { order(ByteOrder.nativeOrder()) }
        }
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()

        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = inputBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees,
            centerCrop = true
        )

        if (isGrayscaleModel) {
            // Use custom grayscale processing for 1-channel models
            // Extract options from classifierOptions map
            val enableColorInversion = classifierOptions?.get("enableColorInversion") as? Boolean ?: false
            val enableMaxNormalization = classifierOptions?.get("enableMaxNormalization") as? Boolean ?: false
            val inputMean = (classifierOptions?.get("inputMean") as? Number)?.toFloat() ?: 0f
            val inputStd = (classifierOptions?.get("inputStd") as? Number)?.toFloat() ?: 255f
            
            ImageUtils.processGrayscaleImage(
                bitmap = inputBitmap,
                targetWidth = inputSize.width,
                targetHeight = inputSize.height,
                outputBuffer = grayBuffer,
                pixels = intValues,
                enableColorInversion = enableColorInversion,
                enableMaxNormalization = enableMaxNormalization,
                inputMean = inputMean,
                inputStd = inputStd
            )
            grayBuffer.rewind()
            grayBuffer.asFloatBuffer().get(floatInput, 0, floatInput.size)
        } else {
            ImageUtils.copyRgbBitmapToFloatArray(
                inputBitmap,
                floatInput,
                intValues,
                INPUT_MEAN,
                INPUT_STD,
                channelsFirst = rtModel.inputUsesNchw
            )
        }

        val preEnd = System.nanoTime()
        val scores = rtModel.run(floatInput)[0] // flat FloatArray(numClass)
        val inferEnd = System.nanoTime()

        val topCount = minOf(5, scores.size)
        val topIndices = IntArray(topCount) { -1 }
        val topScores = FloatArray(topCount) { Float.NEGATIVE_INFINITY }
        for (index in scores.indices) {
            val score = scores[index]
            var insert = 0
            while (insert < topCount && score <= topScores[insert]) insert++
            if (insert == topCount) continue
            var move = topCount - 1
            while (move > insert) {
                topScores[move] = topScores[move - 1]
                topIndices[move] = topIndices[move - 1]
                move--
            }
            topScores[insert] = score
            topIndices[insert] = index
        }

        val top1Index = topIndices.firstOrNull()?.takeIf { it >= 0 } ?: 0
        val top1Score = topScores.firstOrNull()?.takeIf { top1Index >= 0 } ?: 0f
        val top1Label = labelName(top1Index)

        val top5Indices = topIndices.filter { it >= 0 }
        val top5Labels = top5Indices.map { labelName(it) }
        val top5Scores = topScores.take(top5Indices.size)

        val probs = Probs(
            top1Label = top1Label,
            top5Labels = top5Labels,
            top1Conf = top1Score,
            top5Confs = top5Scores,
            top1Index = top1Index,
            top5Indices = top5Indices
        )

        val timing = finishTiming(preEnd, inferEnd)
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            probs = probs,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
            names = labels
        )
    }

    companion object {
        private const val TAG = "Classifier"

        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
    }
}
