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
    // The model output kept FLAT (the [1,A,B,C] tensor row-major). Indexing it directly avoids copying the whole
    // output into a jagged Array<Array<Array<FloatArray>>> every frame and re-reading it during argmax.
    private var flatOutput: FloatArray = FloatArray(0)
    private lateinit var outputShape: IntArray
    private var colorCache = IntArray(0)
    // Reusable running-best-score buffer for the class-major NCHW argmax (see postProcessSemantic).
    private var scoreScratch = FloatArray(0)

    init {
        YOLOFileUtils.loadModelLabels(context, modelPath)?.let { labels = it }

        rtModel = InferenceModel.create(context, modelPath, useGpu, "SemanticSegmenter")

        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // fp16 semantic models output FLOAT [1, A, B, C]. The int8/uint8 output paths are dropped with LiteRT 2.x.
        outputShape = rtModel.outputDims.getOrNull(0) ?: IntArray(0)
        // 4D [1, C, H, W]/[1, H, W, C] logits, or a 3D [1, H, W] class map from in-graph-ArgMax exports
        require((outputShape.size == 4 || outputShape.size == 3) && outputShape[0] == 1) {
            "Semantic output tensor shape not supported: ${outputShape.joinToString()}"
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

        // Keep the model output flat; postProcessSemantic indexes it directly (no per-frame reshape copy).
        val preEnd = System.nanoTime()
        flatOutput = rtModel.run(floatInput)[0]
        val inferEnd = System.nanoTime()
        val semanticMask = postProcessSemantic(origWidth, origHeight)
        val annotatedImage = drawSemanticOverlay(bitmap, semanticMask)
        val timing = finishTiming(preEnd, inferEnd)
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            boxes = emptyList(),
            semanticMask = semanticMask,
            annotatedImage = annotatedImage,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
            names = labels
        )
    }

    private fun postProcessSemantic(origWidth: Int, origHeight: Int): SemanticMask? {
        val isClassMap = outputShape.size == 3 // [1, H, W] class indices, argmax already done on the NPU
        val isNCHW = !isClassMap && (outputShape[1] <= outputShape[3] || outputShape[1] == labels.size)
        val classCount = if (isClassMap) labels.size.coerceAtLeast(2) else if (isNCHW) outputShape[1] else outputShape[3]
        val maskHeight = if (isClassMap) outputShape[1] else if (isNCHW) outputShape[2] else outputShape[1]
        val maskWidth = if (isClassMap) outputShape[2] else if (isNCHW) outputShape[3] else outputShape[2]
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
        val out = flatOutput
        val plane = maskWidth * maskHeight
        if (isNCHW && classCount > 1) {
            // Class-major argmax for the NCHW [1, C, H, W] logits: walk each class plane's crop rows sequentially
            // (cache-friendly) keeping a running best per output pixel, instead of jumping a whole plane per class
            // for every pixel. ~640x640x19 reads become 19 streaming passes rather than 12M strided gathers.
            if (scoreScratch.size != width * height) scoreScratch = FloatArray(width * height)
            val best = scoreScratch
            for (y in 0 until height) {
                var src = (y + top) * maskWidth + left
                var dst = y * width
                for (x in 0 until width) {
                    best[dst] = out[src]
                    classMap[dst] = 0
                    src++
                    dst++
                }
            }
            for (c in 1 until classCount) {
                val planeBase = c * plane
                for (y in 0 until height) {
                    var src = planeBase + (y + top) * maskWidth + left
                    var dst = y * width
                    for (x in 0 until width) {
                        val score = out[src]
                        if (score > best[dst]) {
                            best[dst] = score
                            classMap[dst] = c
                        }
                        src++
                        dst++
                    }
                }
            }
        } else {
            // isClassMap: out is already a [1, H, W] class map (NPU in-graph ArgMax). classCount == 1: single class.
            // NHWC: classes are contiguous per pixel, so argmax walks one cache line per pixel.
            for (y in 0 until height) {
                val sourceY = y + top
                for (x in 0 until width) {
                    val sourceX = x + left
                    classMap[y * width + x] = if (isClassMap) {
                        out[sourceY * maskWidth + sourceX].toInt().coerceIn(0, classCount - 1)
                    } else if (classCount == 1) {
                        0
                    } else {
                        val base = (sourceY * maskWidth + sourceX) * classCount
                        var bestIndex = 0
                        var bestScore = out[base]
                        for (c in 1 until classCount) {
                            val score = out[base + c]
                            if (score > bestScore) {
                                bestScore = score
                                bestIndex = c
                            }
                        }
                        bestIndex
                    }
                }
            }
        }
        for (i in classMap.indices) {
            pixels[i] = colors[classMap[i]]
        }

        val maskImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        maskImage.setPixels(pixels, 0, width, 0, 0, width, height)
        return SemanticMask(classMap.toList(), width, height, maskImage)
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
