// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class SemanticSegmenter(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {
    private val interpreterOptions = (customOptions ?: Interpreter.Options()).apply {
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
            } catch (e: Exception) {
                Log.e(TAG, "GPU delegate error: ${e.message}")
            }
        }
    }

    private lateinit var inputBuffer: ByteBuffer
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray
    private lateinit var outputFloat: Array<Array<Array<FloatArray>>>
    private lateinit var outputByte: Array<Array<Array<ByteArray>>>
    private lateinit var outputShape: IntArray
    private lateinit var inputDataType: DataType
    private lateinit var outputDataType: DataType
    private var inputScale = 0f
    private var inputZeroPoint = 0
    private var outputScale = 0f
    private var outputZeroPoint = 0

    init {
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)
        YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)?.let {
            labels = it
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)
        interpreter.allocateTensors()

        val inputShape = interpreter.getInputTensor(0).shape()
        val inputTensor = interpreter.getInputTensor(0)
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        inputDataType = inputTensor.dataType()
        inputTensor.quantizationParams().let {
            inputScale = it.scale
            inputZeroPoint = it.zeroPoint
        }

        val outputTensor = interpreter.getOutputTensor(0)
        outputShape = outputTensor.shape()
        outputDataType = outputTensor.dataType()
        outputTensor.quantizationParams().let {
            outputScale = it.scale
            outputZeroPoint = it.zeroPoint
        }
        require(outputShape.size == 4 && outputShape[0] == 1) {
            "Semantic output tensor shape not supported: ${outputShape.joinToString()}"
        }
        when (outputDataType) {
            DataType.FLOAT32 -> outputFloat = Array(outputShape[0]) {
                Array(outputShape[1]) {
                    Array(outputShape[2]) {
                        FloatArray(outputShape[3])
                    }
                }
            }
            DataType.UINT8, DataType.INT8 -> outputByte = Array(outputShape[0]) {
                Array(outputShape[1]) {
                    Array(outputShape[2]) {
                        ByteArray(outputShape[3])
                    }
                }
            }
            else -> throw IllegalArgumentException("Semantic output type not supported: $outputDataType")
        }

        val inputElementBytes = when (inputDataType) {
            DataType.FLOAT32 -> 4
            DataType.UINT8, DataType.INT8 -> 1
            else -> throw IllegalArgumentException("Semantic input type not supported: $inputDataType")
        }
        inputBuffer = ByteBuffer.allocateDirect(inHeight * inWidth * 3 * inputElementBytes).apply {
            order(ByteOrder.nativeOrder())
        }
        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
    }

    private fun outputBuffer(): Any = when (outputDataType) {
        DataType.FLOAT32 -> outputFloat
        DataType.UINT8, DataType.INT8 -> outputByte
        else -> throw IllegalArgumentException("Semantic output type not supported: $outputDataType")
    }

    private fun copyBitmapToInputBuffer(bitmap: Bitmap) {
        inputBuffer.clear()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixel in intValues) {
            writeInputValue(((pixel shr 16) and 0xFF) / 255f)
            writeInputValue(((pixel shr 8) and 0xFF) / 255f)
            writeInputValue((pixel and 0xFF) / 255f)
        }
        inputBuffer.rewind()
    }

    private fun writeInputValue(value: Float) {
        when (inputDataType) {
            DataType.FLOAT32 -> inputBuffer.putFloat(value)
            DataType.UINT8 -> {
                val scale = inputScale.takeIf { it > 0f } ?: (1f / 255f)
                val quantized = (value / scale + inputZeroPoint).roundToInt().coerceIn(0, 255)
                inputBuffer.put(quantized.toByte())
            }
            DataType.INT8 -> {
                val scale = inputScale.takeIf { it > 0f } ?: (1f / 127f)
                val quantized = (value / scale + inputZeroPoint).roundToInt().coerceIn(-128, 127)
                inputBuffer.put(quantized.toByte())
            }
            else -> throw IllegalArgumentException("Semantic input type not supported: $inputDataType")
        }
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
        copyBitmapToInputBuffer(inputBitmap)

        interpreter.run(inputBuffer, outputBuffer())
        updateTiming()

        val semanticMask = postProcessSemantic(origWidth, origHeight)
        val fpsDouble = if (t4 > 0f) (1f / t4).toDouble() else 0.0
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            boxes = emptyList(),
            semanticMask = semanticMask,
            annotatedImage = drawSemanticOverlay(bitmap, semanticMask),
            speed = t2,
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
        for (y in 0 until height) {
            val sourceY = y + top
            for (x in 0 until width) {
                val sourceX = x + left
                val classIndex = bestClass(classCount, sourceX, sourceY, isNCHW)
                val outputIndex = y * width + x
                classMap[outputIndex] = classIndex
                val color = ultralyticsColors[classIndex % ultralyticsColors.size]
                pixels[outputIndex] = Color.argb(
                    255,
                    Color.red(color),
                    Color.green(color),
                    Color.blue(color)
                )
            }
        }

        val maskImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        maskImage.setPixels(pixels, 0, width, 0, 0, width, height)
        return SemanticMask(classMap.toList(), width, height, maskImage)
    }

    private fun bestClass(classCount: Int, x: Int, y: Int, isNCHW: Boolean): Int {
        if (classCount == 1) return 0
        var bestIndex = 0
        var bestScore = -Float.MAX_VALUE
        for (classIndex in 0 until classCount) {
            val score = if (isNCHW) {
                outputValue(classIndex, y, x)
            } else {
                outputValue(y, x, classIndex)
            }
            if (score > bestScore) {
                bestScore = score
                bestIndex = classIndex
            }
        }
        return bestIndex
    }

    private fun outputValue(first: Int, second: Int, third: Int): Float {
        return when (outputDataType) {
            DataType.FLOAT32 -> outputFloat[0][first][second][third]
            DataType.UINT8 -> {
                val scale = outputScale.takeIf { it > 0f } ?: 1f
                ((outputByte[0][first][second][third].toInt() and 0xFF) - outputZeroPoint) * scale
            }
            DataType.INT8 -> {
                val scale = outputScale.takeIf { it > 0f } ?: 1f
                (outputByte[0][first][second][third].toInt() - outputZeroPoint) * scale
            }
            else -> throw IllegalArgumentException("Semantic output type not supported: $outputDataType")
        }
    }

    private fun drawSemanticOverlay(bitmap: Bitmap, semanticMask: SemanticMask?): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val mask = semanticMask?.maskImage ?: return output
        val scaledMask = Bitmap.createScaledBitmap(mask, output.width, output.height, true)
        Canvas(output).drawBitmap(
            scaledMask,
            0f,
            0f,
            android.graphics.Paint().apply { alpha = 128 }
        )
        if (scaledMask !== mask) scaledMask.recycle()
        return output
    }

    companion object {
        private const val TAG = "SemanticSegmenter"
    }
}
