// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import kotlin.math.max
import kotlin.math.min

class PoseEstimator(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var confidenceThreshold: Float = 0.25f,   // Can be changed as needed
    private var iouThreshold: Float = 0.7f,
    private var numItemsThreshold: Int = 30,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {

    private enum class OutputLayout {
        FEATURES_FIRST,
        ANCHORS_FIRST
    }

    companion object {
        // xywh(4) + conf(1) + keypoints(17*3=51) = 56
        private const val OUTPUT_FEATURES = 56
        private const val KEYPOINTS_COUNT = 17
    }

    private val interpreterOptions = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
        
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
            } catch (e: Exception) {
                Log.e("PoseEstimator", "GPU delegate error: ${e.message}")
            }
        }
    }

    // Reuse ByteBuffer for input to reduce allocations
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray
    
    // Reuse output arrays to reduce allocations
    private lateinit var outputArray: Array<Array<FloatArray>>
    
    // Output dimensions
    private var batchSize = 0
    private var numAnchors = 0
    private lateinit var outputLayout: OutputLayout

    init {
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        val loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (loadedLabels != null) {
            this.labels = loadedLabels // Use labels from appended ZIP
            Log.i("PoseEstimator", "Labels successfully loaded from appended ZIP.")
        } else {
            Log.w("PoseEstimator", "Could not load labels from appended ZIP, trying FlatBuffers metadata...")
            // Try FlatBuffers as a fallback
            if (loadLabelsFromFlatbuffers(modelBuffer)) {
                labelsWereLoaded = true
                Log.i("PoseEstimator", "Labels successfully loaded from FlatBuffers metadata.")
            }
        }

        if (!labelsWereLoaded) {
            Log.w("PoseEstimator", "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list.")
            if (this.labels.isEmpty()) {
                Log.w("PoseEstimator", "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names.")
            }
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)
        // Call allocateTensors() once during initialization
        interpreter.allocateTensors()

        val inputShape = interpreter.getInputTensor(0).shape()
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        inputSize = com.ultralytics.yolo.Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        
        val outputShape = interpreter.getOutputTensor(0).shape()
        batchSize = outputShape[0]
        outputLayout = when {
            outputShape[1] == OUTPUT_FEATURES -> OutputLayout.FEATURES_FIRST
            outputShape[2] == OUTPUT_FEATURES -> OutputLayout.ANCHORS_FIRST
            else -> throw IllegalArgumentException(
                "Unexpected output feature size. Expected $OUTPUT_FEATURES in one output axis, Actual=${outputShape.contentToString()}"
            )
        }

        val outFeatures: Int
        when (outputLayout) {
            OutputLayout.FEATURES_FIRST -> {
                outFeatures = outputShape[1]
                numAnchors = outputShape[2]
            }
            OutputLayout.ANCHORS_FIRST -> {
                outFeatures = outputShape[2]
                numAnchors = outputShape[1]
            }
        }

        outputArray = Array(batchSize) {
            when (outputLayout) {
                OutputLayout.FEATURES_FIRST -> Array(outFeatures) { FloatArray(numAnchors) }
                OutputLayout.ANCHORS_FIRST -> Array(numAnchors) { FloatArray(outFeatures) }
            }
        }

        val inputBytes = 1 * inHeight * inWidth * 3 * 4 // FLOAT32 is 4 bytes
        inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
            order(java.nio.ByteOrder.nativeOrder())
        }
        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()
        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = inputBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees
        )
        ImageUtils.copyRgbBitmapToFloatBuffer(inputBitmap, inputBuffer, intValues)

        interpreter.run(inputBuffer, outputArray)
        // Update processing time measurement
        updateTiming()

        val rawDetections = postProcessPose(
            features = outputArray[0],  // shape: [56][numAnchors]
            numAnchors = numAnchors,
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold,
            origWidth = origWidth,
            origHeight = origHeight
        )

        // Apply numItemsThreshold limit
        val limitedDetections = rawDetections.take(numItemsThreshold)
        
        val boxes = limitedDetections.map { it.box }
        val keypointsList = limitedDetections.map { it.keypoints }

        val fpsDouble: Double = if (t4 > 0) (1.0 / t4) else 0.0
        // Pack into YOLOResult and return
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            keypointsList = keypointsList,
            speed = t2,   // Measurement values in milliseconds etc. depend on BasePredictor implementation
            fps = fpsDouble,
            names = labels
        )
    }

    private fun postProcessPose(
        features: Array<FloatArray>,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<PoseDetection> {

        val detections = mutableListOf<PoseDetection>()

        for (j in 0 until numAnchors) {
            val rawX = featureValue(features, 0, j)
            val rawY = featureValue(features, 1, j)
            val rawW = featureValue(features, 2, j)
            val rawH = featureValue(features, 3, j)
            val conf = featureValue(features, 4, j)

            if (conf < confidenceThreshold) continue

            val outputRect = RectF(
                rawX - rawW / 2f,
                rawY - rawH / 2f,
                rawX + rawW / 2f,
                rawY + rawH / 2f
            )
            val outputIsNormalized = outputCoordinatesAreNormalized(outputRect)
            val rectF = inputRectFromOutputRect(
                outputRect,
                origWidth,
                origHeight,
                outputIsNormalized
            ) ?: continue
            val normBox = normalizedRectFromInputRect(rectF, origWidth, origHeight)

            val kpArray = mutableListOf<Pair<Float, Float>>()
            val kpConfArray = mutableListOf<Float>()
            for (k in 0 until KEYPOINTS_COUNT) {
                val rawKx = featureValue(features, 5 + k * 3, j)
                val rawKy = featureValue(features, 5 + k * 3 + 1, j)
                val kpC   = featureValue(features, 5 + k * 3 + 2, j)

                val (finalKx, finalKy) = inputPointFromOutputPoint(
                    rawKx,
                    rawKy,
                    origWidth,
                    origHeight,
                    outputIsNormalized
                )

                kpArray.add(finalKx to finalKy)
                kpConfArray.add(kpC)
            }

            val xynList = kpArray.map { (fx, fy) ->
                (fx / origWidth) to (fy / origHeight)
            }
            val boxObj = Box(0, "person", conf, rectF, normBox)
            
            val keypoints = Keypoints(
                xyn = xynList,
                xy = kpArray,
                conf = kpConfArray
            )
            
            detections.add(
                PoseDetection(
                    box = boxObj,
                    keypoints = keypoints
                )
            )
        }

        val finalDetections = nmsPoseDetections(detections, iouThreshold)
        return finalDetections
    }

    private fun featureValue(
        features: Array<FloatArray>,
        featureIndex: Int,
        anchorIndex: Int
    ): Float {
        return when (outputLayout) {
            OutputLayout.FEATURES_FIRST -> features[featureIndex][anchorIndex]
            OutputLayout.ANCHORS_FIRST -> features[anchorIndex][featureIndex]
        }
    }


    private fun nmsPoseDetections(
        detections: List<PoseDetection>,
        iouThreshold: Float
    ): List<PoseDetection> {
        val confidenceThreshold = 0.25f  // Configurable threshold
        val filteredDetections = detections.filter { it.box.conf >= confidenceThreshold }
        
        if (filteredDetections.size <= 1) {
            return filteredDetections
        }
        
        val sorted = filteredDetections.sortedByDescending { it.box.conf }
        val picked = mutableListOf<PoseDetection>()
        val used = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (used[i]) continue

            val d1 = sorted[i]
            picked.add(d1)

            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                val d2 = sorted[j]
                if (iou(d1.box.xywh, d2.box.xywh) > iouThreshold) {
                    used[j] = true
                }
            }
        }
        return picked
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        return if (unionArea <= 0f) 0f else (interArea / unionArea)
    }


    override fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf.toFloat()
        super.setConfidenceThreshold(conf)
    }

    override fun setIouThreshold(iou: Double) {
        iouThreshold = iou.toFloat()
        super.setIouThreshold(iou)
    }
    
    override fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        super.setNumItemsThreshold(n)
    }

    override fun getConfidenceThreshold(): Double {
        return confidenceThreshold.toDouble()
    }

    override fun getIouThreshold(): Double {
        return iouThreshold.toDouble()
    }

    private data class PoseDetection(
        val box: Box,
        val keypoints: Keypoints
    )
    
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
        Log.e("PoseEstimator", "Failed to extract metadata: ${e.message}")
        false
    }
}
