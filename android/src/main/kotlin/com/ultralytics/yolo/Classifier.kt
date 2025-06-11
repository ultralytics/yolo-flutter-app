// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.metadata.schema.ModelMetadata
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class Classifier(
    context: Context,
    modelPath: String,
    override var labels: List<String> = emptyList(),
    private val useGpu: Boolean = true
) : BasePredictor() {

    private val interpreterOptions: Interpreter.Options = Interpreter.Options().apply {
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d(TAG, "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e(TAG, "GPU delegate error: ${e.message}")
            }
        }
        setNumThreads(4)
    }

    var numClass: Int = 0

    private lateinit var imageProcessorCameraPortrait: ImageProcessor
    private lateinit var imageProcessorCameraLandscape: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor

    init {
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        var loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (labelsWereLoaded) {
            this.labels = loadedLabels!! // Use labels from appended ZIP
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
            if (this.labels.isEmpty()) {
                Log.w(TAG, "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names.")
            }
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)

        val inputShape = interpreter.getInputTensor(0).shape()
        val inBatch = inputShape[0]
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        val inChannels = inputShape[3]
        require(inBatch == 1 && inChannels == 3) {
            "Unexpected input tensor shape. Expect [1,H,W,3], but got ${inputShape.joinToString()}"
        }

        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        Log.d(TAG, "Model input size = $inWidth x $inHeight")

        val outputShape = interpreter.getOutputTensor(0).shape()
        // e.g. outputShape = [1, 1000]
        numClass = outputShape[1]
        Log.d(TAG, "Model output shape = [1, $numClass]")

        // For camera feed in portrait mode (with rotation)
        imageProcessorCameraPortrait = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 270-degree rotation
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For camera feed in landscape mode (no rotation)
        imageProcessorCameraLandscape = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For single images (no rotation)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
            .add(CastOp(DataType.FLOAT32))
            .build()

        Log.d(TAG, "Classifier initialized.")
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // Choose appropriate processor based on input source and orientation
        val processedImage = if (rotateForCamera) {
            // Apply appropriate rotation based on device orientation
            if (isLandscape) {
                imageProcessorCameraLandscape.process(tensorImage)
            } else {
                imageProcessorCameraPortrait.process(tensorImage)
            }
        } else {
            // No rotation for single image
            imageProcessorSingleImage.process(tensorImage)
        }
        val inputBuffer = processedImage.buffer

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

        val top5Labels = top5.map { (idx, _) -> labels.getOrElse(idx) { "Unknown" } }
        val top5Scores = top5.map { it.second }

        val probs = Probs(
            top1 = top1Label,
            top5 = top5Labels,
            top1Conf = top1Score,
            top5Confs = top5Scores,
            top1Index = top1Index
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
                Log.d(TAG, "Found associated file: $fileName")
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)
                    Log.d(TAG, "Associated file contents:\n$fileString")

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()
                            Log.d(TAG, "Loaded labels from metadata: $labels")
                            return true
                        }
                    }
                }
            }
        } else {
            Log.d(TAG, "No associated files found in the metadata.")
        }
        false
    } catch (e: Exception) {
        Log.e(TAG, "Failed to extract metadata: ${e.message}")
        false
    }
}
