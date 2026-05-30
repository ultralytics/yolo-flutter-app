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

        rtModel = LiteRtModel(modelPath, useGpu, "Classifier")

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
            ImageUtils.copyRgbBitmapToFloatArray(inputBitmap, floatInput, intValues, INPUT_MEAN, INPUT_STD)
        }

        val scores = rtModel.run(floatInput)[0] // flat FloatArray(numClass)
        val indexedScores = scores.mapIndexed { index, score -> index to score }
        val sorted = indexedScores.sortedByDescending { it.second }

        // Top1
        val top1 = sorted.firstOrNull()
        // Top5
        val top5 = sorted.take(5)

        val top1Label = if (top1 != null) labelName(top1.first) else "class 0"
        val top1Score = top1?.second ?: 0f
        val top1Index: Int = if (top1 != null) top1.first else 0

        val top5Indices = top5.map { it.first }
        val top5Labels = top5.map { (idx, _) -> labelName(idx) }
        val top5Scores = top5.map { it.second }

        val probs = Probs(
            top1Label = top1Label,
            top5Labels = top5Labels,
            top1Conf = top1Score,
            top5Confs = top5Scores,
            top1Index = top1Index,
            top5Indices = top5Indices
        )

        updateTiming()
        val fpsVal = if (t4 > 0) 1.0 / t4 else 0.0
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            probs = probs,
            speed = elapsedMsSinceStart(),
            fps = fpsVal,
            names = labels
        )
    }

    companion object {
        private const val TAG = "Classifier"

        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
    }
}
