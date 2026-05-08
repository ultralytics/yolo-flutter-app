// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class Classifier(
    context: Context,
    modelPath: String,
    override var labels: List<String> = emptyList(),
    private val useGpu: Boolean = true,
    private val customOptions: Interpreter.Options? = null,
    private val classifierOptions: Map<String, Any>? = null
) : BasePredictor() {

    private val interpreterOptions: Interpreter.Options = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(4)
        }
        
        // Add GPU delegate if requested
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
            } catch (e: Exception) {
                Log.e(TAG, "GPU delegate error: ${e.message}")
            }
        }
    }

    var numClass: Int = 0
    private var modelInputChannels: Int = 3  // Default to 3-channel, will be detected
    private var isGrayscaleModel: Boolean = false

    private lateinit var inputBuffer: ByteBuffer
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray

    init {
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        val loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (loadedLabels != null) {
            this.labels = loadedLabels // Use labels from appended ZIP
            Log.i(TAG, "Labels successfully loaded from appended ZIP.")
        } else {
            Log.w(TAG, "Could not load labels from appended ZIP, trying FlatBuffers metadata...")
            // Try FlatBuffers as a fallback
            if (loadLabelsFromFlatbuffers(modelBuffer)) {
                labelsWereLoaded = true
                Log.i(TAG, "Labels successfully loaded from FlatBuffers metadata.")
            }
        }

        if (!labelsWereLoaded) {
            Log.w(TAG, "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list.")
            
            // Check if labels are provided via classifierOptions
            val optionsLabels = classifierOptions?.get("labels") as? List<*>
            if (optionsLabels != null) {
                this.labels = optionsLabels.map { it.toString() }
                Log.i(TAG, "Using labels from classifierOptions (${this.labels.size} classes): ${this.labels}")
            } else if (this.labels.isEmpty()) {
                Log.w(TAG, "Warning: No labels loaded and no labels provided via constructor or classifierOptions. Detections might lack class names.")
            }
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)

        val inputShape = interpreter.getInputTensor(0).shape()
        val inBatch = inputShape[0]
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        val inChannels = inputShape[3]
        
        // Detect model input channels and configure accordingly
        modelInputChannels = inChannels
        
        // Check if 1-channel support is enabled via classifier options
        val enable1ChannelSupport = classifierOptions?.get("enable1ChannelSupport") as? Boolean ?: false
        isGrayscaleModel = (inChannels == 1) || (enable1ChannelSupport && inChannels == 1)
        
        // Validate input shape based on detected or expected channels
        val expectedChannels = classifierOptions?.get("expectedChannels") as? Int ?: inChannels
        require(inBatch == 1) {
            "Unexpected batch size. Expect batch=1, but got batch=$inBatch"
        }
        require(inChannels == expectedChannels || (expectedChannels == 1 && inChannels == 1) || (expectedChannels == 3 && inChannels == 3)) {
            "Unexpected input channels. Expected $expectedChannels channels, but got $inChannels channels. Input shape: ${inputShape.joinToString()}"
        }

        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        val outputShape = interpreter.getOutputTensor(0).shape()
        // e.g. outputShape = [1, 1000] for ImageNet, [1, 12] for EMNIST
        numClass = outputShape[1]

        // Validate expected classes if specified
        (classifierOptions?.get("expectedClasses") as? Int)?.let { expectedClasses ->
            if (numClass != expectedClasses) {
                Log.w(TAG, "Warning: Expected $expectedClasses output classes, but model has $numClass classes")
            }
        }

        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
        inputBuffer = ByteBuffer.allocateDirect(inWidth * inHeight * modelInputChannels * 4).apply {
            order(ByteOrder.nativeOrder())
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
            val modelInputSize = inputSize
            
            // Extract options from classifierOptions map
            val enableColorInversion = classifierOptions?.get("enableColorInversion") as? Boolean ?: false
            val enableMaxNormalization = classifierOptions?.get("enableMaxNormalization") as? Boolean ?: false
            val inputMean = (classifierOptions?.get("inputMean") as? Number)?.toFloat() ?: 0f
            val inputStd = (classifierOptions?.get("inputStd") as? Number)?.toFloat() ?: 255f
            
            inputBuffer = ImageUtils.processGrayscaleImage(
                bitmap = inputBitmap,
                targetWidth = modelInputSize.width,
                targetHeight = modelInputSize.height,
                enableColorInversion = enableColorInversion,
                enableMaxNormalization = enableMaxNormalization,
                inputMean = inputMean,
                inputStd = inputStd
            )
        } else {
            ImageUtils.copyRgbBitmapToFloatBuffer(inputBitmap, inputBuffer, intValues, INPUT_MEAN, INPUT_STD)
        }

        val outputArray = Array(1) { FloatArray(numClass) }
        interpreter.run(inputBuffer, outputArray)

        updateTiming()

        val scores = outputArray[0]   // FloatArray(numClass)
        val indexedScores = scores.mapIndexed { index, score -> index to score }
        val sorted = indexedScores.sortedByDescending { it.second }

        // Top1
        val top1 = sorted.firstOrNull()
        // Top5
        val top5 = sorted.take(5)

        val top1Label = if (top1 != null) labels.getOrElse(top1.first) { "Unknown" } else "Unknown"
        val top1Score = top1?.second ?: 0f
        val top1Index: Int = if (top1 != null) top1.first else 0

        val top5Indices = top5.map { it.first }
        val top5Labels = top5.map { (idx, _) -> labels.getOrElse(idx) { "Unknown" } }
        val top5Scores = top5.map { it.second }

        val probs = Probs(
            top1Label = top1Label,
            top5Labels = top5Labels,
            top1Conf = top1Score,
            top5Confs = top5Scores,
            top1Index = top1Index,
            top5Indices = top5Indices
        )

        val fpsVal = if (t4 > 0) 1.0 / t4 else 0.0

        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            probs = probs,
            speed = t2,
            fps = fpsVal,
            names = labels
        )
    }

    companion object {
        private const val TAG = "Classifier"

        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
    }
    
    /**
     * Load labels from FlatBuffers metadata
     */
    private fun loadLabelsFromFlatbuffers(buf: MappedByteBuffer): Boolean = try {
        val extractor = MetadataExtractor(buf)
        val files = extractor.associatedFileNames
        if (!files.isNullOrEmpty()) {
            for (fileName in files) {
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()
                            return true
                        }
                    }
                }
            }
        }
        false
    } catch (e: Exception) {
        Log.e(TAG, "Failed to extract metadata: ${e.message}")
        false
    }
}
