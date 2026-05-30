// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import kotlin.math.roundToInt

class SemanticSegmenter(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true
) : BasePredictor() {
    private lateinit var floatInput: FloatArray
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray
    private lateinit var outputFloat: Array<Array<Array<FloatArray>>>
    private lateinit var outputShape: IntArray
    private var colorCache = IntArray(0)

    init {
        YOLOFileUtils.loadModelLabels(context, modelPath)?.let { labels = it }

        rtModel = LiteRtModel(modelPath, useGpu, "SemanticSegmenter")

        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // fp16 semantic models output FLOAT [1, A, B, C]. The int8/uint8 output paths are dropped with LiteRT 2.x.
        outputShape = rtModel.outputDims.getOrNull(0) ?: IntArray(0)
        require(outputShape.size == 4 && outputShape[0] == 1) {
            "Semantic output tensor shape not supported: ${outputShape.joinToString()}"
        }
        outputFloat = Array(outputShape[0]) {
            Array(outputShape[1]) { Array(outputShape[2]) { FloatArray(outputShape[3]) } }
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
        isLandscape: Boolean
    ): YOLOResult {
        t0 = System.nanoTime()

        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = inputBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees
        )
        ImageUtils.copyRgbBitmapToFloatArray(inputBitmap, floatInput, intValues)

        // Reshape the flat float output into outputFloat[0].
        val flat = rtModel.run(floatInput)[0]
        val o = outputFloat[0]
        var idx = 0
        for (a in o.indices) {
            val plane = o[a]
            for (b in plane.indices) {
                val cell = plane[b]
                for (c in cell.indices) {
                    cell[c] = flat[idx++]
                }
            }
        }
        val semanticMask = postProcessSemantic(origWidth, origHeight)
        val annotatedImage = drawSemanticOverlay(bitmap, semanticMask)
        updateTiming()
        val fpsDouble = if (t4 > 0f) (1f / t4).toDouble() else 0.0
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            boxes = emptyList(),
            semanticMask = semanticMask,
            annotatedImage = annotatedImage,
            speed = elapsedMsSinceStart(),
            fps = fpsDouble,
            names = labels
        )
    }

    private fun postProcessSemantic(origWidth: Int, origHeight: Int): SemanticMask? {
        val isNCHW = outputShape[1] <= outputShape[3] || outputShape[1] == labels.size
        val classCount = if (isNCHW) outputShape[1] else outputShape[3]
        val maskHeight = if (isNCHW) outputShape[2] else outputShape[1]
        val maskWidth = if (isNCHW) outputShape[3] else outputShape[2]
        if (classCount <= 0 || maskWidth <= 0 || maskHeight <= 0) return null

        val crop = modelMaskCropRect(maskWidth, maskHeight, origWidth, origHeight)
        val left = crop?.left ?: 0
        val top = crop?.top ?: 0
        val right = crop?.right ?: maskWidth
        val bottom = crop?.bottom ?: maskHeight
        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return null

        val classMap = IntArray(width * height)
        val pixels = IntArray(width * height)
        val colors = semanticColors(classCount)
        for (y in 0 until height) {
            val sourceY = y + top
            for (x in 0 until width) {
                val sourceX = x + left
                val classIndex = bestClass(classCount, sourceX, sourceY, isNCHW)
                val outputIndex = y * width + x
                classMap[outputIndex] = classIndex
                pixels[outputIndex] = colors[classIndex]
            }
        }

        val maskImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        maskImage.setPixels(pixels, 0, width, 0, 0, width, height)
        return SemanticMask(classMap.toList(), width, height, maskImage)
    }

    private fun bestClass(
        classCount: Int,
        x: Int,
        y: Int,
        isNCHW: Boolean
    ): Int {
        if (classCount == 1) return 0

        var bestIndex = 0
        var bestScore = -Float.MAX_VALUE
        for (classIndex in 0 until classCount) {
            val score = if (isNCHW) outputFloat[0][classIndex][y][x] else outputFloat[0][y][x][classIndex]
            if (score > bestScore) {
                bestScore = score
                bestIndex = classIndex
            }
        }
        return bestIndex
    }

    private fun semanticColors(classCount: Int): IntArray {
        if (colorCache.size == classCount) return colorCache

        colorCache = IntArray(classCount) { classIndex ->
            val color = ultralyticsColors[classIndex % ultralyticsColors.size]
            Color.argb(255, Color.red(color), Color.green(color), Color.blue(color))
        }
        return colorCache
    }

    private fun drawSemanticOverlay(bitmap: Bitmap, semanticMask: SemanticMask?): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val mask = semanticMask?.maskImage ?: return output
        val scaledMask = if (mask.width == output.width && mask.height == output.height) {
            mask
        } else {
            Bitmap.createScaledBitmap(mask, output.width, output.height, true)
        }
        Canvas(output).drawBitmap(
            scaledMask,
            0f,
            0f,
            android.graphics.Paint().apply {
                alpha = 128
                isFilterBitmap = true
            }
        )
        if (scaledMask !== mask) scaledMask.recycle()
        return output
    }

    companion object {
        private const val TAG = "SemanticSegmenter"
    }
}
