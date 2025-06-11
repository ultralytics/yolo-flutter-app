// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import android.util.Size
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
import kotlin.math.max
import kotlin.math.min
import androidx.collection.ArrayMap

class PoseEstimator(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var confidenceThreshold: Float = 0.25f,   // Can be changed as needed
    private var iouThreshold: Float = 0.45f,          // Can be changed as needed
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {

    companion object {
        // xywh(4) + conf(1) + keypoints(17*3=51) = 56
        private const val OUTPUT_FEATURES = 56
        private const val KEYPOINTS_COUNT = 17
        private const val KEYPOINTS_FEATURES = KEYPOINTS_COUNT * 3 // x, y, conf per keypoint
        private const val MAX_POOL_SIZE = 100 
        
        private const val INPUT_SIZE = 640
    }
    
    private val boxPool = ObjectPool<Box>(MAX_POOL_SIZE) { Box(0, "", 0f, RectF(), RectF()) }
    private val keypointsPool = ObjectPool<Keypoints>(MAX_POOL_SIZE) {
        Keypoints(
            List(KEYPOINTS_COUNT) { 0f to 0f },
            List(KEYPOINTS_COUNT) { 0f to 0f },
            List(KEYPOINTS_COUNT) { 0f }
        )
    }
    
    private class ObjectPool<T>(
        private val maxSize: Int,
        private val factory: () -> T
    ) {
        private val pool = ArrayList<T>(maxSize)
        
        @Synchronized
        fun acquire(): T {
            return if (pool.isEmpty()) {
                factory()
            } else {
                pool.removeAt(pool.size - 1)
            }
        }
        
        @Synchronized
        fun release(obj: T) {
            if (pool.size < maxSize) {
                pool.add(obj)
            }
        }
        
        @Synchronized
        fun clear() {
            pool.clear()
        }
    }

    private val interpreterOptions = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
        
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d("PoseEstimator", "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e("PoseEstimator", "GPU delegate error: ${e.message}")
            }
        }
    }

    private lateinit var imageProcessorCameraPortrait: ImageProcessor
    private lateinit var imageProcessorCameraLandscape: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor
    
    // Reuse ByteBuffer for input to reduce allocations
    private lateinit var inputBuffer: ByteBuffer
    
    // Reuse output arrays to reduce allocations
    private lateinit var outputArray: Array<Array<FloatArray>>
    
    // Output dimensions
    private var batchSize = 0
    private var numAnchors = 0

    init {
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        var loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (labelsWereLoaded) {
            this.labels = loadedLabels!! // Use labels from appended ZIP
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
        Log.d("PoseEstimator", "TFLite model loaded and tensors allocated")

        val inputShape = interpreter.getInputTensor(0).shape()
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        inputSize = com.ultralytics.yolo.Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        
        val outputShape = interpreter.getOutputTensor(0).shape()  // e.g.: [1, 56, 2100]
        batchSize = outputShape[0]           // 1
        val outFeatures = outputShape[1]     // 56
        numAnchors = outputShape[2]          // 2100 etc.
        require(outFeatures == OUTPUT_FEATURES) {
            "Unexpected output feature size. Expected=$OUTPUT_FEATURES, Actual=$outFeatures"
        }
        
        outputArray = Array(batchSize) {
            Array(outFeatures) { FloatArray(numAnchors) }
        }
        
        val inputBytes = 1 * inHeight * inWidth * 3 * 4 // FLOAT32 is 4 bytes
        inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
            order(java.nio.ByteOrder.nativeOrder())
        }
        Log.d("PoseEstimator", "Direct ByteBuffer allocated with native ordering: $inputBytes bytes")
        
        // For camera feed in portrait mode (with rotation)
        imageProcessorCameraPortrait = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 270-degree rotation
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For camera feed in landscape mode (no rotation)
        imageProcessorCameraLandscape = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For single images (no rotation needed)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // Choose the appropriate processor based on input source and orientation
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
        
        inputBuffer.clear()
        inputBuffer.put(processedImage.buffer)
        inputBuffer.rewind()

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

        val boxes = rawDetections.map { it.box }
        val keypointsList = rawDetections.map { it.keypoints }

//        val annotatedImage = drawPoseOnBitmap(bitmap, keypointsList, boxes)

        val fpsDouble: Double = if (t4 > 0) (1.0 / t4) else 0.0
        // Pack into YOLOResult and return
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            keypointsList = keypointsList,
//            annotatedImage = annotatedImage,
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

        // Assume modelInputSize = (640, 640) for example
        val (modelW, modelH) = modelInputSize
        val scaleX = origWidth.toFloat() / modelW
        val scaleY = origHeight.toFloat() / modelH

        for (j in 0 until numAnchors) {
            val rawX = features[0][j]       // 0..1
            val rawY = features[1][j]       // 0..1
            val rawW = features[2][j]       // 0..1
            val rawH = features[3][j]       // 0..1
            val conf = features[4][j]       // 0..1

            if (conf < confidenceThreshold) continue

            val xScaled = rawX * modelW
            val yScaled = rawY * modelH
            val wScaled = rawW * modelW
            val hScaled = rawH * modelH

            val left   = xScaled - wScaled / 2f
            val top    = yScaled - hScaled / 2f
            val right  = xScaled + wScaled / 2f
            val bottom = yScaled + hScaled / 2f

            // modelInputSize スケールでの RectF
            val normBox = RectF(left / modelW, top / modelH, right / modelW, bottom / modelH)

            val rectF = RectF(
                left   * scaleX,
                top    * scaleY,
                right  * scaleX,
                bottom * scaleY
            )

            val kpArray = mutableListOf<Pair<Float, Float>>()
            val kpConfArray = mutableListOf<Float>()
            for (k in 0 until KEYPOINTS_COUNT) {
                val rawKx = features[5 + k * 3][j]
                val rawKy = features[5 + k * 3 + 1][j]
                val kpC   = features[5 + k * 3 + 2][j]

                // Check if values are already in pixel coordinates (>1) or normalized (0-1)
                val isNormalized = rawKx <= 1.0f && rawKy <= 1.0f
                
                val finalKx: Float
                val finalKy: Float
                
                if (isNormalized) {
                    val kxScaled = rawKx * modelW
                    val kyScaled = rawKy * modelH
                    finalKx = kxScaled * scaleX
                    finalKy = kyScaled * scaleY
                } else {
                    finalKx = rawKx * scaleX
                    finalKy = rawKy * scaleY
                }

                kpArray.add(finalKx to finalKy)
                kpConfArray.add(kpC)
            }

            val boxObj = boxPool.acquire()
            val keypointsObj = keypointsPool.acquire()
            
            val xynList = kpArray.map { (fx, fy) ->
                (fx / origWidth) to (fy / origHeight)
            }
            
            boxObj.index = 0
            boxObj.cls = "person"
            boxObj.conf = conf
            boxObj.xywh.set(rectF)
            boxObj.xywhn.set(normBox)
            
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
            
            keypointsPool.release(keypointsObj)
        }

        val finalDetections = nmsPoseDetections(detections, iouThreshold)
        return finalDetections
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


    private fun drawPoseOnBitmap(
        bitmap: Bitmap,
        keypointsList: List<Keypoints>,
        boxes: List<Box>
    ): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.GREEN
            strokeWidth = 5f
        }
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 3f
        }

        for ((index, person) in keypointsList.withIndex()) {
            val boxRect = boxes[index].xywh
            canvas.drawRect(boxRect, boxPaint)

            for ((i, kp) in person.xy.withIndex()) {
                if (person.conf[i] > 0.25f) {
                    canvas.drawCircle(kp.first, kp.second, 8f, paint)
                }
            }
        }
        return output
    }

    override fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf.toFloat()
        super.setConfidenceThreshold(conf)
    }

    override fun setIouThreshold(iou: Double) {
        iouThreshold = iou.toFloat()
        super.setIouThreshold(iou)
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
                Log.d("PoseEstimator", "Found associated file: $fileName")
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)
                    Log.d("PoseEstimator", "Associated file contents:\n$fileString")

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()
                            Log.d("PoseEstimator", "Loaded labels from metadata: $labels")
                            return true
                        }
                    }
                }
            }
        } else {
            Log.d("PoseEstimator", "No associated files found in the metadata.")
        }
        false
    } catch (e: Exception) {
        Log.e("PoseEstimator", "Failed to extract metadata: ${e.message}")
        false
    }
}
